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

        # A hot single day is less valuable than momentum confirmed over a week.
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
