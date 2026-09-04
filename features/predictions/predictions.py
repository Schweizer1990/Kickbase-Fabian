from kickbase_api.league import get_league_players_on_market
from kickbase_api.manager import get_managers, get_manager_squad
from kickbase_api.user import get_players_in_squad
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np


SQUAD_COLUMNS = [
    "manager_id", "manager_name", "player_id", "first_name", "last_name", "position",
    "team_name", "mv", "mv_change_yesterday", "mv_change_7d", "mv_trend_7d",
    "mv_avg_daily_change_7d", "predicted_mv_target", "s_11_prob", "prediction_available"
]


def live_data_predictions(today_df, model, features):
    """Make live data predictions for today_df using the trained model."""

    today_df_features = today_df[features]
    today_df_results = today_df.copy()
    today_df_results["predicted_mv_target"] = np.round(model.predict(today_df_features), 2)
    today_df_results = today_df_results.sort_values("predicted_mv_target", ascending=False)

    now = datetime.now(ZoneInfo("Europe/Berlin"))
    cutoff_time = now.replace(hour=22, minute=15, second=0, microsecond=0)
    date = (now - timedelta(days=1)) if now <= cutoff_time else now
    date = date.date()

    today_df_results = today_df_results.dropna(subset=["mv"])
    today_df_results = today_df_results[[
        "player_id", "first_name", "last_name", "position", "team_name", "date",
        "mv_change_1d", "mv_trend_1d", "mv_change_7d", "mv_trend_7d",
        "mv_avg_daily_change_7d", "mv", "predicted_mv_target"
    ]]

    return today_df_results


def _normalize_player_id(value):
    """Normalize compact Kickbase IDs so numeric and string payloads merge reliably."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))

    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text or None


def _extract_squad_items(payload):
    """Return the player list from own or manager squad payloads across API variants."""
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []

    for key in ("it", "players", "squad", "pl"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return []


def _enrich_squad_payload(payload, today_df_results, manager_id=None, manager_name=None):
    """Normalize one Kickbase squad payload and enrich it with current ML/MV data."""
    squad_df = pd.DataFrame(_extract_squad_items(payload))
    if squad_df.empty:
        return pd.DataFrame(columns=SQUAD_COLUMNS)

    if "i" not in squad_df.columns:
        if "id" in squad_df.columns:
            squad_df["i"] = squad_df["id"]
        else:
            squad_df["i"] = np.nan

    # Manager-squad payloads can expose player IDs as floats while the historical
    # model stores them as strings. Merge through a normalized text key instead of
    # relying on pandas' dtype coercion.
    squad_df["_merge_player_id"] = squad_df["i"].map(_normalize_player_id)
    prediction_df = today_df_results.copy()
    prediction_df["_merge_player_id"] = prediction_df["player_id"].map(_normalize_player_id)

    squad_df = pd.merge(
        squad_df,
        prediction_df,
        on="_merge_player_id",
        how="left",
        suffixes=("_kickbase", "")
    )

    if "prob" not in squad_df.columns:
        squad_df["prob"] = np.nan
    squad_df = squad_df.rename(columns={"prob": "s_11_prob", "mv_change_1d": "mv_change_yesterday"})

    # Prefer the model data's current MV, but retain Kickbase's payload value as fallback.
    if "mv" not in squad_df.columns:
        squad_df["mv"] = np.nan
    if "mv_kickbase" in squad_df.columns:
        squad_df["mv"] = squad_df["mv"].fillna(squad_df["mv_kickbase"])

    if "player_id" not in squad_df.columns:
        squad_df["player_id"] = np.nan
    squad_df["player_id"] = squad_df["player_id"].fillna(squad_df["_merge_player_id"])
    squad_df["player_id"] = squad_df["player_id"].map(_normalize_player_id)

    # Fall back to compact Kickbase name fields when historical/model data is missing.
    if "first_name" not in squad_df.columns:
        squad_df["first_name"] = np.nan
    if "last_name" not in squad_df.columns:
        squad_df["last_name"] = np.nan

    for candidate in ("fn", "firstName", "first_name_kickbase"):
        if candidate in squad_df.columns:
            squad_df["first_name"] = squad_df["first_name"].fillna(squad_df[candidate])
    for candidate in ("ln", "n", "lastName", "last_name_kickbase"):
        if candidate in squad_df.columns:
            squad_df["last_name"] = squad_df["last_name"].fillna(squad_df[candidate])

    if "position" not in squad_df.columns:
        squad_df["position"] = np.nan
    for candidate in ("pos", "position_kickbase"):
        if candidate in squad_df.columns:
            squad_df["position"] = squad_df["position"].fillna(squad_df[candidate])

    if "team_name" not in squad_df.columns:
        squad_df["team_name"] = np.nan
    for candidate in ("tn", "teamName", "team_name_kickbase"):
        if candidate in squad_df.columns:
            squad_df["team_name"] = squad_df["team_name"].fillna(squad_df[candidate])

    for column in [
        "mv_change_yesterday", "mv_change_7d", "mv_trend_7d",
        "mv_avg_daily_change_7d", "predicted_mv_target", "s_11_prob"
    ]:
        if column not in squad_df.columns:
            squad_df[column] = np.nan

    squad_df["prediction_available"] = squad_df["predicted_mv_target"].notna()
    squad_df["manager_id"] = str(manager_id) if manager_id is not None else None
    squad_df["manager_name"] = manager_name

    return squad_df[SQUAD_COLUMNS]


def join_current_squad(token, league_id, today_df_results):
    """Return every player currently in the user's squad, enriched with predictions where available."""

    squad_players = get_players_in_squad(token, league_id)
    enriched = _enrich_squad_payload(squad_players, today_df_results)
    return enriched.drop(columns=["manager_id", "manager_name"], errors="ignore")


def join_all_manager_squads(token, league_id, today_df_results):
    """Return current squads for every league manager with the same MV fields as the user's squad.

    Opponent squads are read from Kickbase's manager-squad endpoint. Individual manager
    failures are isolated so one unavailable profile does not break the full daily report.
    """
    frames = []
    managers = get_managers(token, league_id)

    for manager_name, manager_id in managers:
        try:
            payload = get_manager_squad(token, league_id, manager_id)
            enriched = _enrich_squad_payload(
                payload,
                today_df_results,
                manager_id=manager_id,
                manager_name=manager_name,
            )
            if not enriched.empty:
                frames.append(enriched)
        except Exception as exc:
            print(f"Warning: Could not fetch squad for manager {manager_name}: {exc}")

    if not frames:
        return pd.DataFrame(columns=SQUAD_COLUMNS)

    result = pd.concat(frames, ignore_index=True)
    result.sort_values(["manager_name", "mv"], ascending=[True, False], inplace=True, na_position="last")
    return result


def join_current_market(token, league_id, today_df_results):
    """Return the complete current market, enriched with ML predictions where available."""

    players_on_market = get_league_players_on_market(token, league_id)
    market_df = pd.DataFrame(players_on_market)

    if market_df.empty:
        return pd.DataFrame(columns=[
            "player_id", "last_name", "team_name", "mv", "mv_change_yesterday",
            "mv_change_7d", "mv_trend_7d", "mv_avg_daily_change_7d",
            "predicted_mv_target", "model_recommended", "s_11_prob", "hours_to_exp",
            "expiring_today", "prediction_available"
        ])

    bid_df = pd.merge(
        market_df,
        today_df_results,
        left_on="id",
        right_on="player_id",
        how="left"
    )

    bid_df["hours_to_exp"] = np.round((bid_df["exp"] / 3600), 2)

    now = datetime.now(ZoneInfo("Europe/Berlin"))
    next_22 = now.replace(hour=22, minute=0, second=0, microsecond=0)
    if now >= next_22:
        next_22 += timedelta(days=1)
    diff = np.round((next_22 - now).total_seconds() / 3600, 2)

    bid_df["expiring_today"] = bid_df["hours_to_exp"] < diff
    bid_df["prediction_available"] = bid_df["predicted_mv_target"].notna()
    bid_df["model_recommended"] = bid_df["predicted_mv_target"].fillna(0) > 5000

    if "prob" not in bid_df.columns:
        bid_df["prob"] = np.nan
    bid_df = bid_df.rename(columns={"prob": "s_11_prob", "mv_change_1d": "mv_change_yesterday"})
    bid_df["player_id"] = bid_df["player_id"].fillna(bid_df["id"])

    for column in [
        "last_name", "team_name", "mv", "mv_change_yesterday", "mv_change_7d",
        "mv_trend_7d", "mv_avg_daily_change_7d", "predicted_mv_target"
    ]:
        if column not in bid_df.columns:
            bid_df[column] = np.nan

    bid_df = bid_df.sort_values(
        ["model_recommended", "predicted_mv_target"],
        ascending=[False, False],
        na_position="last"
    )

    return bid_df[[
        "player_id", "last_name", "team_name", "mv", "mv_change_yesterday",
        "mv_change_7d", "mv_trend_7d", "mv_avg_daily_change_7d",
        "predicted_mv_target", "model_recommended", "s_11_prob", "hours_to_exp",
        "expiring_today", "prediction_available"
    ]]
