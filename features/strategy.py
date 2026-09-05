import math

import numpy as np
import pandas as pd


def _num(value, default=None):
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _price_segment(market_value):
    value = _num(market_value, 0)
    if value < 2_000_000:
        return "cheap_under_2m"
    if value < 10_000_000:
        return "mid_2m_to_10m"
    return "premium_over_10m"


def build_opponent_bid_profiles(transfer_df):
    """Learn observed winning-bid behaviour from completed purchases.

    Only completed/winning purchases are observable. The profile deliberately
    avoids pretending that losing bids are known.
    """
    columns = [
        "manager", "segment", "purchases", "priced_purchases", "median_overpay_pct",
        "average_overpay_pct", "p75_overpay_pct", "max_overpay_pct", "total_spend",
    ]
    if transfer_df is None or transfer_df.empty:
        return pd.DataFrame(columns=columns)

    purchases = transfer_df[transfer_df["buyer"].notna()].copy()
    if purchases.empty:
        return pd.DataFrame(columns=columns)

    purchases["segment"] = purchases["market_value_at_transfer"].apply(_price_segment)
    rows = []
    for (manager, segment), group in purchases.groupby(["buyer", "segment"]):
        priced = group[group["overpay_pct"].notna()].copy()
        rows.append({
            "manager": manager,
            "segment": segment,
            "purchases": int(len(group)),
            "priced_purchases": int(len(priced)),
            "median_overpay_pct": round(float(priced["overpay_pct"].median()), 2) if not priced.empty else None,
            "average_overpay_pct": round(float(priced["overpay_pct"].mean()), 2) if not priced.empty else None,
            "p75_overpay_pct": round(float(priced["overpay_pct"].quantile(0.75)), 2) if not priced.empty else None,
            "max_overpay_pct": round(float(priced["overpay_pct"].max()), 2) if not priced.empty else None,
            "total_spend": float(group["price"].fillna(0).sum()),
        })

    return pd.DataFrame(rows, columns=columns).sort_values(
        ["segment", "purchases", "total_spend"], ascending=[True, False, False], ignore_index=True
    )


def _league_segment_stats(transfer_df):
    if transfer_df is None or transfer_df.empty:
        return {}
    purchases = transfer_df[
        transfer_df["buyer"].notna() & transfer_df["overpay_pct"].notna()
    ].copy()
    if purchases.empty:
        return {}
    purchases["segment"] = purchases["market_value_at_transfer"].apply(_price_segment)
    result = {}
    for segment, group in purchases.groupby("segment"):
        values = group["overpay_pct"].astype(float)
        result[segment] = {
            "sample_size": int(len(values)),
            "median_overpay_pct": float(values.median()),
            "p75_overpay_pct": float(values.quantile(0.75)),
        }
    return result


def build_market_strategy(market_df, transfer_df, own_budget=None):
    """Rank current market players for capital growth and bid discipline.

    Combines the next-day model, current daily momentum and the sustained
    seven-day Kickbase market-value trend. Completed league transfers are used
    only to estimate how much competition tends to overpay.
    """
    columns = [
        "player_id", "player_name", "team", "market_value", "predicted_mv_change",
        "predicted_return_pct", "daily_return_pct", "mv_change_7d", "return_7d_pct",
        "avg_daily_return_7d_pct", "segment", "league_bid_sample",
        "league_median_overpay_pct", "suggested_bid", "hard_max_bid", "capital_score",
        "budget_fit", "strategy_label",
    ]
    if market_df is None or market_df.empty:
        return pd.DataFrame(columns=columns)

    segment_stats = _league_segment_stats(transfer_df)
    rows = []
    for _, player in market_df.iterrows():
        mv = _num(player.get("mv"))
        if not mv or mv <= 0:
            continue
        pred = _num(player.get("predicted_mv_target"), 0.0)
        daily = _num(player.get("mv_change_yesterday"), 0.0)
        change_7d = _num(player.get("mv_change_7d"), 0.0)
        trend_7d = _num(player.get("mv_trend_7d"), 0.0)
        segment = _price_segment(mv)
        stats = segment_stats.get(segment, {})
        median_overpay = _num(stats.get("median_overpay_pct"), 0.0)
        p75_overpay = _num(stats.get("p75_overpay_pct"), median_overpay)

        median_overpay = max(-5.0, min(median_overpay, 15.0))
        p75_overpay = max(median_overpay, min(p75_overpay, 20.0))

        suggested = mv * (1 + median_overpay / 100)
        competition_max = mv * (1 + p75_overpay / 100)
        roi_max = mv + max(pred, 0.0) * 3
        hard_max = min(competition_max, roi_max) if pred > 0 else mv
        suggested = min(suggested, hard_max)

        pred_pct = pred / mv * 100
        daily_pct = daily / mv * 100
        return_7d_pct = trend_7d * 100
        avg_daily_7d_pct = return_7d_pct / 7

        capital_score = pred_pct * 0.50 + daily_pct * 0.30 + avg_daily_7d_pct * 0.20

        budget_fit = own_budget is None or suggested <= own_budget
        sustained_positive = change_7d > 0
        if pred_pct >= 8 and daily_pct > 0 and sustained_positive:
            label = "strong_trade"
        elif pred_pct >= 3 and daily_pct >= 0 and sustained_positive:
            label = "trade"
        elif pred_pct > 0 and daily_pct >= 0:
            label = "watch_trend_unconfirmed"
        elif pred_pct > 0:
            label = "watch"
        else:
            label = "avoid_for_trading"

        rows.append({
            "player_id": str(player.get("player_id")) if player.get("player_id") is not None else None,
            "player_name": player.get("last_name"),
            "team": player.get("team_name"),
            "market_value": int(round(mv)),
            "predicted_mv_change": int(round(pred)),
            "predicted_return_pct": round(pred_pct, 2),
            "daily_return_pct": round(daily_pct, 2),
            "mv_change_7d": int(round(change_7d)),
            "return_7d_pct": round(return_7d_pct, 2),
            "avg_daily_return_7d_pct": round(avg_daily_7d_pct, 2),
            "segment": segment,
            "league_bid_sample": int(stats.get("sample_size", 0)),
            "league_median_overpay_pct": round(median_overpay, 2),
            "suggested_bid": int(round(suggested)),
            "hard_max_bid": int(round(max(suggested, hard_max))),
            "capital_score": round(capital_score, 3),
            "budget_fit": bool(budget_fit),
            "strategy_label": label,
        })

    result = pd.DataFrame(rows, columns=columns)
    if result.empty:
        return result
    return result.sort_values(
        ["budget_fit", "capital_score", "predicted_mv_change"],
        ascending=[False, False, False],
        ignore_index=True,
    )


def build_squad_signals(squad_df):
    """Create hold/sell signals using next-day, daily and seven-day MV direction."""
    columns = [
        "player_id", "player_name", "team", "market_value", "predicted_mv_change",
        "daily_mv_change", "mv_change_7d", "return_7d_pct", "s11_indicator", "signal", "reason",
    ]
    if squad_df is None or squad_df.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for _, player in squad_df.iterrows():
        mv = _num(player.get("mv"), 0.0)
        pred = _num(player.get("predicted_mv_target"), 0.0)
        daily = _num(player.get("mv_change_yesterday"), 0.0)
        change_7d = _num(player.get("mv_change_7d"), 0.0)
        trend_7d = _num(player.get("mv_trend_7d"), 0.0)
        s11 = player.get("s_11_prob")

        if pred < 0 and daily < 0 and change_7d < 0:
            signal = "sell_pressure_strong"
            reason = "model_daily_and_7d_market_value_negative"
        elif pred < 0 and daily < 0:
            signal = "sell_pressure"
            reason = "model_and_daily_market_value_both_negative"
        elif pred > 0 and daily >= 0 and change_7d > 0:
            signal = "hold_positive_strong"
            reason = "model_daily_and_7d_market_value_positive"
        elif pred > 0 and daily >= 0:
            signal = "hold_positive"
            reason = "model_and_daily_market_value_positive"
        elif pred < 0:
            signal = "watch_sell"
            reason = "model_negative"
        else:
            signal = "hold_watch"
            reason = "mixed_market_value_signal"

        rows.append({
            "player_id": str(player.get("player_id")) if player.get("player_id") is not None else None,
            "player_name": player.get("last_name"),
            "team": player.get("team_name"),
            "market_value": int(round(mv)) if mv else None,
            "predicted_mv_change": int(round(pred)),
            "daily_mv_change": int(round(daily)),
            "mv_change_7d": int(round(change_7d)),
            "return_7d_pct": round(trend_7d * 100, 2),
            "s11_indicator": s11,
            "signal": signal,
            "reason": reason,
        })

    return pd.DataFrame(rows, columns=columns)


def build_market_bid_guardrails(
    market_df,
    transfer_df,
    manager_budgets_df=None,
    manager_squads_df=None,
    own_manager_name="Fabian",
):
    """Estimate protection bids from observable completed league bidding behaviour.

    This is deliberately a shadow model. Kickbase does not reveal rivals' live
    bids on somebody else's listing, so the output uses completed winning
    transfers, estimated rival spending power and the two-players-per-club rule.
    It never claims that a specific manager has actually placed a live bid.
    """
    columns = [
        "player_id", "player_name", "team", "market_value", "segment",
        "league_sample", "league_median_overpay_pct", "league_p75_overpay_pct",
        "league_p90_overpay_pct", "shadow_bid_median", "shadow_bid_p75",
        "shadow_bid_p90", "rival_candidates", "top_rival", "top_rival_sample",
        "top_rival_p75_overpay_pct", "top_rival_shadow_bid",
        "roi_guardrail_bid", "trade_protection_bid", "competition_risk",
        "model_confidence",
    ]
    if market_df is None or market_df.empty:
        return pd.DataFrame(columns=columns)

    purchases = pd.DataFrame()
    if transfer_df is not None and not transfer_df.empty:
        purchases = transfer_df[
            transfer_df["buyer"].notna() & transfer_df["overpay_pct"].notna()
        ].copy()
        if not purchases.empty:
            purchases["segment"] = purchases["market_value_at_transfer"].apply(_price_segment)

    budget_map = {}
    if manager_budgets_df is not None and not manager_budgets_df.empty:
        for _, row in manager_budgets_df.iterrows():
            name = row.get("User")
            if not name:
                continue
            available = _num(row.get("Available Budget"))
            cash = _num(row.get("Budget"))
            budget_map[str(name)] = available if available is not None else cash

    club_counts = {}
    if manager_squads_df is not None and not manager_squads_df.empty:
        for (manager_name, team_name), group in manager_squads_df.groupby(
            ["manager_name", "team_name"], dropna=False
        ):
            if manager_name is None or team_name is None:
                continue
            club_counts[(str(manager_name), str(team_name))] = int(len(group))

    manager_profiles = {}
    if not purchases.empty:
        for (manager, segment), group in purchases.groupby(["buyer", "segment"]):
            values = pd.to_numeric(group["overpay_pct"], errors="coerce").dropna()
            if values.empty:
                continue
            manager_profiles[(str(manager), segment)] = {
                "sample": int(len(values)),
                "median": float(values.median()),
                "p75": float(values.quantile(0.75)),
            }

    rows = []
    for _, player in market_df.iterrows():
        mv = _num(player.get("mv"))
        if not mv or mv <= 0:
            continue

        segment = _price_segment(mv)
        pred = _num(player.get("predicted_mv_target"), 0.0)
        team = player.get("team_name")

        league_values = pd.Series(dtype=float)
        if not purchases.empty:
            league_values = pd.to_numeric(
                purchases.loc[purchases["segment"] == segment, "overpay_pct"],
                errors="coerce",
            ).dropna()

        if league_values.empty:
            league_median = 0.0
            league_p75 = 3.0
            league_p90 = 6.0
            league_sample = 0
        else:
            league_median = float(league_values.median())
            league_p75 = float(league_values.quantile(0.75))
            league_p90 = float(league_values.quantile(0.90))
            league_sample = int(len(league_values))

        league_median = max(-5.0, min(league_median, 15.0))
        league_p75 = max(league_median, min(league_p75, 20.0))
        league_p90 = max(league_p75, min(league_p90, 25.0))

        candidates = []
        managers = set(budget_map) | {key[0] for key in manager_profiles}
        for manager in managers:
            if manager == own_manager_name:
                continue
            if team is not None and club_counts.get((manager, str(team)), 0) >= 2:
                continue

            available = budget_map.get(manager)
            if available is not None and available < mv:
                continue

            profile = manager_profiles.get((manager, segment))
            if profile:
                p75 = max(-5.0, min(profile["p75"], 25.0))
                sample = profile["sample"]
            else:
                p75 = league_p75
                sample = 0

            projected = mv * (1 + p75 / 100)
            if available is not None:
                projected = min(projected, available)

            candidates.append({
                "manager": manager,
                "sample": sample,
                "p75": p75,
                "projected": projected,
            })

        candidates.sort(key=lambda item: (item["projected"], item["sample"]), reverse=True)
        top = candidates[0] if candidates else None

        shadow_median = mv * (1 + league_median / 100)
        shadow_p75 = mv * (1 + league_p75 / 100)
        shadow_p90 = mv * (1 + league_p90 / 100)
        if top is not None:
            shadow_p75 = max(shadow_p75, top["projected"])

        roi_guardrail = mv + max(pred, 0.0) * 3
        trade_protection = min(shadow_p75, roi_guardrail) if pred > 0 else mv

        if len(candidates) >= 5:
            risk = "high"
        elif len(candidates) >= 2:
            risk = "medium"
        else:
            risk = "low"

        if league_sample >= 50 and top is not None and top["sample"] >= 5:
            confidence = "medium_high"
        elif league_sample >= 20:
            confidence = "medium"
        else:
            confidence = "low"

        rows.append({
            "player_id": str(player.get("player_id")) if player.get("player_id") is not None else None,
            "player_name": player.get("last_name"),
            "team": team,
            "market_value": int(round(mv)),
            "segment": segment,
            "league_sample": league_sample,
            "league_median_overpay_pct": round(league_median, 2),
            "league_p75_overpay_pct": round(league_p75, 2),
            "league_p90_overpay_pct": round(league_p90, 2),
            "shadow_bid_median": int(round(shadow_median)),
            "shadow_bid_p75": int(round(shadow_p75)),
            "shadow_bid_p90": int(round(shadow_p90)),
            "rival_candidates": int(len(candidates)),
            "top_rival": top["manager"] if top else None,
            "top_rival_sample": int(top["sample"]) if top else None,
            "top_rival_p75_overpay_pct": round(top["p75"], 2) if top else None,
            "top_rival_shadow_bid": int(round(top["projected"])) if top else None,
            "roi_guardrail_bid": int(round(roi_guardrail)),
            "trade_protection_bid": int(round(max(mv, trade_protection))),
            "competition_risk": risk,
            "model_confidence": confidence,
        })

    result = pd.DataFrame(rows, columns=columns)
    if result.empty:
        return result
    risk_rank = {"high": 0, "medium": 1, "low": 2}
    result["_risk_rank"] = result["competition_risk"].map(risk_rank).fillna(3)
    result = result.sort_values(
        ["_risk_rank", "shadow_bid_p75", "market_value"],
        ascending=[True, False, False],
        ignore_index=True,
    )
    return result.drop(columns=["_risk_rank"])
