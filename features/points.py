import math

import pandas as pd

from kickbase_api.config import BASE_URL, get_json_with_token


def _num(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _minutes(value):
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace("'", "").strip()
    try:
        return float(text)
    except ValueError:
        return 0.0


def _season_matches(season):
    matches = []
    for match in season.get("ph", []) if isinstance(season, dict) else []:
        points = _num(match.get("p"))
        minutes = _minutes(match.get("mp"))
        if points is None and minutes <= 0:
            continue
        matches.append({
            "points": points or 0.0,
            "minutes": minutes,
            "status": match.get("st"),
        })
    return matches


def _weighted_average(values):
    if not values:
        return 0.0
    weights = list(range(1, len(values) + 1))
    return sum(v * w for v, w in zip(values, weights)) / sum(weights)


def _p90(matches):
    total_minutes = sum(m["minutes"] for m in matches)
    if total_minutes <= 0:
        return 0.0
    return sum(m["points"] for m in matches) / total_minutes * 90.0


def _raw_performance(payload):
    seasons = payload.get("it", []) if isinstance(payload, dict) else []
    current = _season_matches(seasons[-1]) if seasons else []
    previous = _season_matches(seasons[-2]) if len(seasons) >= 2 else []
    return current, previous


def _league_baseline(raw_profiles):
    """Robust points-per-90 prior for shrinking tiny samples.

    Use only players with a meaningful recent-minute sample. A fallback keeps the
    model stable at the very start of a season when almost nobody qualifies.
    """
    values = []
    for current, previous in raw_profiles.values():
        recent = current[-5:] if sum(m["minutes"] for m in current[-5:]) >= 270 else previous[-5:]
        minutes = sum(m["minutes"] for m in recent)
        if minutes >= 270:
            value = _p90(recent)
            if math.isfinite(value):
                values.append(max(-50.0, min(250.0, value)))

    if not values:
        return 85.0
    values.sort()
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2


def _profile_from_matches(current, previous, baseline_p90):
    current_recent = current[-5:]
    previous_recent = previous[-5:]

    current_minutes_total = sum(m["minutes"] for m in current_recent)
    previous_minutes_total = sum(m["minutes"] for m in previous_recent)
    current_p90 = _p90(current_recent)
    previous_p90 = _p90(previous_recent)

    # Blend current and previous season by actual minutes, then shrink small
    # samples aggressively to a league prior. This prevents a single cameo or
    # one freak scoring game from becoming a 150+ xP projection.
    current_weight = min(current_minutes_total / 450.0, 1.0)
    if current_minutes_total > 0 and previous_minutes_total > 0:
        history_p90 = current_p90 * current_weight + previous_p90 * (1.0 - current_weight)
    elif current_minutes_total > 0:
        history_p90 = current_p90
    elif previous_minutes_total > 0:
        history_p90 = previous_p90
    else:
        history_p90 = baseline_p90

    effective_minutes = current_minutes_total + min(previous_minutes_total, 450.0) * 0.35
    reliability = min(effective_minutes / 540.0, 1.0)
    regressed_p90 = baseline_p90 * (1.0 - reliability) + history_p90 * reliability
    # Raw fantasy p90 can be noisy even with a few games. Keep the heuristic in
    # a plausible band until a trained xP model has enough season data.
    regressed_p90 = max(-30.0, min(220.0, regressed_p90))

    current_minutes = [m["minutes"] for m in current_recent]
    previous_minutes = [m["minutes"] for m in previous_recent]
    if current_minutes:
        expected_minutes_raw = _weighted_average(current_minutes[-3:])
    elif previous_minutes:
        expected_minutes_raw = _weighted_average(previous_minutes[-3:]) * 0.70
    else:
        expected_minutes_raw = 0.0

    # Recent starting/appearance proxies. We deliberately do not interpret the
    # undocumented Kickbase status code. Minutes are observable and robust.
    role_sample = current_recent if current_recent else previous_recent
    starts = sum(1 for m in role_sample if m["minutes"] >= 60)
    appearances = sum(1 for m in role_sample if m["minutes"] > 0)
    sample_games = len(role_sample)
    starter_rate = starts / sample_games if sample_games else 0.0
    appearance_rate = appearances / sample_games if sample_games else 0.0

    # Starter probability is a conservative proxy, not an official lineup
    # probability. It combines start rate and general appearance frequency.
    starter_probability = min(1.0, starter_rate * 0.80 + appearance_rate * 0.20)

    if not current_recent and previous_recent:
        starter_probability *= 0.75

    expected_minutes = expected_minutes_raw
    if sample_games:
        role_cap = 20.0 + 70.0 * starter_probability
        expected_minutes = min(expected_minutes, role_cap)
    expected_minutes = max(0.0, min(90.0, expected_minutes))

    expected_points = regressed_p90 * expected_minutes / 90.0

    current_apps = len(current)
    current_total_points = sum(m["points"] for m in current)
    current_total_minutes = sum(m["minutes"] for m in current)

    if current_minutes_total >= 270 and len(current_recent) >= 3:
        confidence = "high"
    elif effective_minutes >= 180:
        confidence = "medium"
    else:
        confidence = "low"

    # Low-confidence xP is useful as a directional signal, but must never
    # dominate the win ranking. Apply an explicit uncertainty haircut.
    confidence_factor = {"high": 1.0, "medium": 0.85, "low": 0.60}[confidence]
    expected_points_adjusted = expected_points * confidence_factor

    if starter_probability >= 0.65 and expected_minutes >= 55:
        role = "starter"
    elif appearance_rate >= 0.60 and expected_minutes >= 25:
        role = "rotation"
    else:
        role = "bench_risk"

    return {
        "current_matchdays": current_apps,
        "current_points": round(current_total_points, 2),
        "current_minutes": round(current_total_minutes, 1),
        "recent_points_per_90": round(regressed_p90, 2),
        "expected_minutes_next": round(expected_minutes, 1),
        "starter_probability_proxy": round(starter_probability, 3),
        "role_signal": role,
        "expected_points_raw": round(expected_points, 2),
        "expected_points_next": round(expected_points_adjusted, 2),
        "xp_confidence": confidence,
    }


def build_points_profiles(token, league_id, market_df, squad_df):
    """Build stabilized expected-points profiles for squad and market players."""
    source = {}
    for label, df in (("market", market_df), ("squad", squad_df)):
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            player_id = row.get("player_id")
            if player_id is None:
                continue
            key = str(player_id)
            item = source.setdefault(key, {
                "player_id": key,
                "player_name": row.get("last_name"),
                "team": row.get("team_name"),
                "market_value": _num(row.get("mv")),
                "on_market": False,
                "in_squad": False,
            })
            item["on_market"] = item["on_market"] or label == "market"
            item["in_squad"] = item["in_squad"] or label == "squad"

    raw_profiles = {}
    failures = set()
    for player_id in source:
        try:
            url = f"{BASE_URL}/leagues/{league_id}/players/{player_id}/performance"
            payload = get_json_with_token(url, token)
            raw_profiles[player_id] = _raw_performance(payload)
        except Exception as exc:
            print(f"Warning: Could not fetch league performance for player {player_id}: {exc}")
            raw_profiles[player_id] = ([], [])
            failures.add(player_id)

    baseline_p90 = _league_baseline(raw_profiles)
    rows = []
    for player_id, base in source.items():
        current, previous = raw_profiles[player_id]
        profile = _profile_from_matches(current, previous, baseline_p90)
        if player_id in failures:
            profile["xp_confidence"] = "unavailable"
            profile["expected_points_next"] = 0.0
            profile["expected_points_raw"] = 0.0
            profile["role_signal"] = "unknown"
            profile["starter_probability_proxy"] = 0.0

        mv = base.get("market_value")
        xp = profile["expected_points_next"]
        ppm = xp / (mv / 1_000_000) if mv and mv > 0 else None
        rows.append({
            **base,
            **profile,
            "league_p90_prior": round(baseline_p90, 2),
            "expected_points_per_million": round(ppm, 3) if ppm is not None else None,
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["expected_points_next", "expected_points_per_million"],
        ascending=[False, False],
        ignore_index=True,
    )


def build_win_ranking(market_strategy_df, points_df):
    """Separate championship value from pure trading value.

    Low-confidence and bench-risk point projections are deliberately discounted
    so cheap one-game outliers cannot dominate the ranking.
    """
    if market_strategy_df is None or market_strategy_df.empty:
        return pd.DataFrame()

    points_lookup = {}
    if points_df is not None and not points_df.empty:
        points_lookup = {
            str(row["player_id"]): row
            for _, row in points_df[points_df["on_market"] == True].iterrows()  # noqa: E712
        }

    rows = []
    for _, row in market_strategy_df.iterrows():
        player_id = str(row.get("player_id"))
        p = points_lookup.get(player_id, {})
        xp = _num(p.get("expected_points_next"), 0.0)
        xp_per_m = _num(p.get("expected_points_per_million"), 0.0)
        capital = _num(row.get("capital_score"), 0.0)
        confidence = p.get("xp_confidence", "unavailable")
        role = p.get("role_signal", "unknown")
        starter_probability = _num(p.get("starter_probability_proxy"), 0.0)

        confidence_weight = {"high": 1.0, "medium": 0.85, "low": 0.55, "unavailable": 0.0}.get(confidence, 0.0)
        role_weight = {"starter": 1.0, "rotation": 0.72, "bench_risk": 0.40, "unknown": 0.25}.get(role, 0.25)
        point_weight = confidence_weight * role_weight

        points_component = (xp * 0.60 + xp_per_m * 0.15) * point_weight
        capital_component = capital * 0.25
        win_score = points_component + capital_component

        if role == "starter" and confidence in ("high", "medium"):
            championship_label = "startelf_target"
        elif role == "rotation" and xp > 40:
            championship_label = "points_upside"
        elif capital >= 5:
            championship_label = "trading_target"
        else:
            championship_label = "watch_or_avoid"

        rows.append({
            "player_id": player_id,
            "player_name": row.get("player_name"),
            "team": row.get("team"),
            "market_value": row.get("market_value"),
            "expected_points_next": round(xp, 2),
            "expected_points_per_million": round(xp_per_m, 3),
            "expected_minutes_next": p.get("expected_minutes_next"),
            "starter_probability_proxy": round(starter_probability, 3),
            "role_signal": role,
            "xp_confidence": confidence,
            "capital_score": row.get("capital_score"),
            "suggested_bid": row.get("suggested_bid"),
            "hard_max_bid": row.get("hard_max_bid"),
            "budget_fit": row.get("budget_fit"),
            "strategy_label": row.get("strategy_label"),
            "championship_label": championship_label,
            "win_score": round(win_score, 3),
        })

    return pd.DataFrame(rows).sort_values(
        ["budget_fit", "win_score"], ascending=[False, False], ignore_index=True
    )
