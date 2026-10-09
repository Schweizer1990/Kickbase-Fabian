"""Forward-only MV forecast audit using dated API observations, never clock cutoffs."""
import json
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


def _id(value):
    text = str(value)
    return text[:-2] if text.endswith('.0') else text


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def summarize_forecasts(rows):
    def metrics(items):
        if not items:
            return {"count": 0, "direction_accuracy_percent": None, "mae": None, "rmse": None}
        errors = [r['predicted_change'] - r['actual_change'] for r in items]
        sign = lambda x: (x > 0) - (x < 0)
        correct = sum(sign(r['predicted_change']) == sign(r['actual_change']) for r in items)
        return {"count": len(items), "direction_accuracy_percent": round(100 * correct / len(items), 2),
                "mae": round(sum(abs(e) for e in errors) / len(items), 2),
                "rmse": round(math.sqrt(sum(e * e for e in errors) / len(items)), 2)}
    resolved = [r for r in rows if r.get('status') == 'evaluated']
    groups = {name: metrics([r for r in resolved if name in r['cohorts']])
              for name in ('traders', 'flattening', 'losses')}
    flattening = [r for r in resolved if 'flattening' in r['cohorts']
                  and r.get('previous_daily_change') is not None]
    groups['flattening']['flattening_accuracy_percent'] = (
        round(100 * sum(r['actual_change'] < r['previous_daily_change'] for r in flattening)
              / len(flattening), 2) if flattening else None)
    return {"overall": metrics(resolved), "cohorts": groups,
            "pending_count": sum(r.get('status') == 'pending' for r in rows),
            "invalid_count": sum(r.get('status') == 'invalid' for r in rows)}


def update_forecast_audit(player_df, forecasts_df, path='reports/forecast_audit.json', observed_at=None):
    """Freeze first forecast per player/base-date and score only the next calendar day.

    Same-day reruns cannot overwrite predictions or create extra scoring samples.
    Missing target dates stay pending; a later date is never used as a 1d result.
    """
    observed_at = observed_at or datetime.now(ZoneInfo('Europe/Zurich')).isoformat()
    path = Path(path)
    # Fail visibly if persisted audit data is corrupt rather than silently losing it.
    state = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    rows = state.get('forecasts', [])
    observations = {}
    valid = player_df.dropna(subset=['player_id', 'date', 'mv']).copy()
    valid['date'] = pd.to_datetime(valid['date']).dt.date
    for row in valid.to_dict('records'):
        mv = _number(row['mv'])
        if mv is not None and mv > 0:
            observations.setdefault(_id(row['player_id']), {})[row['date'].isoformat()] = mv
    latest = {pid: {'date': max(values), 'mv': values[max(values)]}
              for pid, values in observations.items()}
    previous = state.get('latest_observations', {})
    common = set(latest) & set(previous)
    advanced = [pid for pid in common if latest[pid]['date'] > previous[pid]['date']]
    revised = [pid for pid in common if latest[pid]['date'] == previous[pid]['date']
               and latest[pid]['mv'] != previous[pid]['mv']]
    # One transferred/newly refreshed player does not prove a league-wide daily update.
    threshold = max(1, math.ceil(len(common) * 0.5))
    confirmed = bool(common) and len(advanced) >= threshold
    update = {'status': 'baseline' if not previous else 'confirmed' if confirmed else 'not_confirmed',
              'confirmed': confirmed, 'advanced_player_count': len(advanced),
              'compared_player_count': len(common), 'required_player_count': threshold,
              'same_date_revision_count': len(revised), 'observed_at': observed_at,
              'last_confirmed_at': observed_at if confirmed else state.get('market_update', {}).get('last_confirmed_at'),
              'latest_mv_date': max((r['date'] for r in latest.values()), default=None),
              'evidence': 'dated_market_value_history; at least half of comparable players advance'}
    known = {(r['player_id'], r['base_date']) for r in rows}
    for row in forecasts_df.to_dict('records'):
        if pd.isna(row.get('player_id')) or pd.isna(row.get('date')):
            continue
        pid, base = _id(row['player_id']), pd.Timestamp(row['date']).date().isoformat()
        mv, prediction = _number(row.get('mv')), _number(row.get('predicted_mv_target'))
        if (pid, base) in known or mv is None or prediction is None or mv <= 0:
            continue
        # A forecast computed from an old value after its result is known is not a live sample.
        if pid not in latest or latest[pid]['date'] != base or observations[pid].get(base) != mv:
            continue
        daily = _number(row.get('mv_change_1d'))
        cohorts = []
        if prediction > 0:
            cohorts.append('traders')
        if daily is not None and daily > 0 and prediction < daily:
            cohorts.append('flattening')
        if prediction < 0:
            cohorts.append('losses')
        rows.append({'player_id': pid, 'base_date': base,
                     'target_date': (date.fromisoformat(base) + timedelta(days=1)).isoformat(),
                     'base_mv': mv, 'predicted_change': prediction, 'cohorts': cohorts,
                     'previous_daily_change': daily,
                     'created_at': observed_at, 'status': 'pending'})
        known.add((pid, base))
    for row in rows:
        if row['status'] != 'pending':
            continue
        values = observations.get(row['player_id'], {})
        actual = values.get(row['target_date'])
        if actual is None:
            continue
        if values.get(row['base_date']) != row['base_mv']:
            row.update(status='invalid', reason='base_value_revised_or_unavailable')
            continue
        row.update(status='evaluated', actual_mv=actual, actual_change=actual - row['base_mv'],
                   evaluated_at=observed_at)
    # Retain a transparent rolling window of 90 base dates, not 90 individual players.
    retained_dates = set(sorted({r['base_date'] for r in rows})[-90:])
    rows = [r for r in rows if r['base_date'] in retained_dates]
    summary = summarize_forecasts(rows)
    state = {'schema_version': 1, 'market_update': update, 'latest_observations': latest,
             'summary': summary, 'forecasts': rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)
    return {'market_update': update, 'live_forecast_evaluation': summary}
