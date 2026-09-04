from features.predictions.predictions import live_data_predictions, join_current_market, join_current_squad, join_all_manager_squads
from features.predictions.preprocessing import preprocess_player_data, split_data
from features.predictions.modeling import train_model, evaluate_model
from kickbase_api.league import get_league_id, get_my_open_offers
from kickbase_api.user import login
from features.notifier import send_mail
from features.reporting import save_latest_report
from features.transfers import build_transfer_history, summarize_manager_bidding
from features.strategy import build_market_strategy, build_squad_signals, build_opponent_bid_profiles
from features.points import build_points_profiles, build_win_ranking
from features.ligainsider import build_ligainsider_signals
from features.predictions.data_handler import (
    create_player_data_table,
    check_if_data_reload_needed,
    save_player_data_to_db,
    load_player_data_from_db,
)
from features.budgets import calc_manager_budgets
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

last_mv_values = 365
last_pfm_values = 50

features = [
    "p", "mv", "days_to_next",
    "mv_change_1d", "mv_trend_1d",
    "mv_change_3d", "mv_vol_3d",
    "mv_trend_7d", "market_divergence"
]
target = "mv_target_clipped"

pd.options.display.float_format = lambda x: '{:,.0f}'.format(x).replace(',', '.')

competition_ids = [1]
league_name = os.getenv("KICKBASE_LEAGUE_NAME")
league_start_date = os.getenv("KICKBASE_LEAGUE_START_DATE")
start_budget_raw = os.getenv("KICKBASE_START_BUDGET")
email_enabled = os.getenv("ENABLE_EMAIL", "false").lower() == "true"
email = os.getenv("EMAIL_USER") if email_enabled else None

USERNAME = os.getenv("KICK_USER")
PASSWORD = os.getenv("KICK_PASS")

required = {
    "KICK_USER": USERNAME,
    "KICK_PASS": PASSWORD,
    "KICKBASE_LEAGUE_NAME": league_name,
    "KICKBASE_LEAGUE_START_DATE": league_start_date,
    "KICKBASE_START_BUDGET": start_budget_raw,
}
missing = [name for name, value in required.items() if not value]
if missing:
    raise RuntimeError("Missing required GitHub Secret/Variable(s): " + ", ".join(missing))

try:
    start_budget = int(start_budget_raw)
except ValueError as exc:
    raise RuntimeError("KICKBASE_START_BUDGET must be an integer, e.g. 50000000") from exc

if email_enabled and (not os.getenv("EMAIL_USER") or not os.getenv("EMAIL_PASS")):
    raise RuntimeError("ENABLE_EMAIL is true but EMAIL_USER/EMAIL_PASS are missing")

token = login(USERNAME, PASSWORD)
print("Logged in to Kickbase successfully.")

league_id = get_league_id(token, league_name)
print("Configured Kickbase league found successfully.")

manager_budgets_df = calc_manager_budgets(token, league_id, league_start_date, start_budget)
print(f"Manager budget analysis completed for {len(manager_budgets_df)} managers.")

transfer_history_df = build_transfer_history(token, league_id, league_start_date, competition_id=competition_ids[0])
bidding_behavior_df = summarize_manager_bidding(transfer_history_df)
opponent_bid_profiles_df = build_opponent_bid_profiles(transfer_history_df)
print(f"Transfer analysis completed for {len(transfer_history_df)} completed transfers and {len(bidding_behavior_df)} managers with purchase activity.")

create_player_data_table()
reload_data = check_if_data_reload_needed()
save_player_data_to_db(token, competition_ids, last_mv_values, last_pfm_values, reload_data)
player_df = load_player_data_from_db()
print("Player data loaded from database.")

proc_player_df, today_df = preprocess_player_data(player_df)
X_train, X_test, y_train, y_test = split_data(proc_player_df, features, target)
print("Player data preprocessed.")

model = train_model(X_train, y_train)
signs_percent, rmse, mae, r2 = evaluate_model(model, X_test, y_test)
print(f"Model evaluation: direction accuracy={signs_percent:.2f}%, RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.2f}")

live_predictions_df = live_data_predictions(today_df, model, features)
market_recommendations_df = join_current_market(token, league_id, live_predictions_df)
squad_recommendations_df = join_current_squad(token, league_id, live_predictions_df)
manager_squads_df = join_all_manager_squads(token, league_id, live_predictions_df)
open_offers = get_my_open_offers(token, league_id)

own_budget = None
exact_budget_rows = manager_budgets_df[manager_budgets_df["Budget Confidence"] == "exact"] if "Budget Confidence" in manager_budgets_df.columns else pd.DataFrame()
if not exact_budget_rows.empty:
    own_budget = float(exact_budget_rows.iloc[0]["Budget"])

market_strategy_df = build_market_strategy(market_recommendations_df, transfer_history_df, own_budget=own_budget)
squad_signals_df = build_squad_signals(squad_recommendations_df)
ligainsider_signals_df = build_ligainsider_signals(market_recommendations_df, squad_recommendations_df)
points_profiles_df = build_points_profiles(token, league_id, market_recommendations_df, squad_recommendations_df, ligainsider_signals_df)
win_ranking_df = build_win_ranking(market_strategy_df, points_profiles_df)

print(f"Market analysis completed for {len(market_recommendations_df)} market players.")
print(f"Squad analysis completed for {len(squad_recommendations_df)} players.")
manager_count = manager_squads_df["manager_name"].nunique() if not manager_squads_df.empty else 0
print(f"League-wide squad analysis completed for {len(manager_squads_df)} players across {manager_count} managers.")
print(f"Visible own open offers found: {len(open_offers)}.")
print(f"LigaInsider layer completed for {len(ligainsider_signals_df)} players.")
print(f"Strategy layer completed for {len(market_strategy_df)} market players, {len(squad_signals_df)} squad players and {len(opponent_bid_profiles_df)} opponent bid profiles.")
print(f"Expected-points layer completed for {len(points_profiles_df)} players and {len(win_ranking_df)} ranked market targets.")

report_path = save_latest_report(
    league_name,
    {
        "direction_accuracy_percent": round(float(signs_percent), 2),
        "rmse": round(float(rmse), 2),
        "mae": round(float(mae), 2),
        "r2": round(float(r2), 4),
    },
    manager_budgets_df,
    market_recommendations_df,
    squad_recommendations_df,
    transfer_history_df,
    bidding_behavior_df,
    open_offers,
    market_strategy_df,
    squad_signals_df,
    opponent_bid_profiles_df,
    points_profiles_df,
    win_ranking_df,
    ligainsider_signals_df,
    manager_squads_df,
)
print(f"Machine-readable report written to {report_path}.")

if email_enabled:
    send_mail(manager_budgets_df, market_recommendations_df, squad_recommendations_df, email)
else:
    print("Email delivery disabled (ENABLE_EMAIL=false).")
