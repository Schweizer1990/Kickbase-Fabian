from kickbase_api.league import get_league_players_on_market
from kickbase_api.user import get_players_in_squad
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np


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


def join_current_squad(token, league_id, today_df_results):
    """Return every player currently in the user's squad, enriched with predictions where available."""

    squad_players = get_players_in_squad(token, league_id)
    squad_df = pd.DataFrame(squad_players.get("it", []))

    if squad_df.empty:
        return pd.DataFrame(columns=[
            "player_id", "last_name", "team_name", "mv", "mv_change_yesterday",
            "mv_change_7d", "mv_trend_7d", "mv_avg_daily_change_7d",
            "predicted_mv_target", "s_11_prob", "prediction_available"
        ])

    # Keep all current squad players even when the historical ML database has no row yet.
    squad_df = pd.merge(
        squad_df,
        today_df_results,
        left_on="i",
        right_on="player_id",
        how="left",
        suffixes=("_kickbase", "")
    )

    if "prob" not in squad_df.columns:
        squad_df["prob"] = np.nan
    squad_df = squad_df.rename(columns={"prob": "s_11_prob", "mv_change_1d": "mv_change_yesterday"})

    # Kickbase squad payload may contain its own current market value. Prefer model-data mv when available.
    if "mv" not in squad_df.columns:
        squad_df["mv"] = np.nan

    # Fall back to names from the live Kickbase payload if present.
    if "last_name" not in squad_df.columns:
        squad_df["last_name"] = np.nan
    for candidate in ["ln", "n"]:
        if candidate in squad_df.columns:
            squad_df["last_name"] = squad_df["last_name"].fillna(squad_df[candidate])

    squad_df["player_id"] = squad_df["player_id"].fillna(squad_df["i"])
    squad_df["prediction_available"] = squad_df["predicted_mv_target"].notna()

    for column in [
        "team_name", "mv_change_yesterday", "mv_change_7d", "mv_trend_7d",
        "mv_avg_daily_change_7d", "predicted_mv_target", "s_11_prob"
    ]:
        if column not in squad_df.columns:
            squad_df[column] = np.nan

    return squad_df[[
        "player_id", "last_name", "team_name", "mv", "mv_change_yesterday",
        "mv_change_7d", "mv_trend_7d", "mv_avg_daily_change_7d",
        "predicted_mv_target", "s_11_prob", "prediction_available"
    ]]


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
