from kickbase_api.user import get_budget, get_username
from kickbase_api.league import (
    get_league_activities,
    get_league_ranking
)
from kickbase_api.manager import (
    get_managers,
    get_manager_performance,
    get_manager_info,
)
from kickbase_api.others import get_achievement_reward
import pandas as pd


def calc_manager_budgets(token, league_id, league_start_date, start_budget):
    """Estimate manager budgets from completed transfers and observable bonuses.

    Own budget is synced from Kickbase and is exact. Opponent budgets remain
    estimates because login/achievement bonuses are not fully attributable from
    the public league activity feed.
    """

    try:
        activities, login_bonus, achievement_bonus = get_league_activities(token, league_id, league_start_date)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch activities: {e}")

    activities_df = pd.DataFrame(activities)

    # Bonuses visible in the league feed. Attribution to individual opponents is
    # incomplete, so these remain estimates and are marked as such in the report.
    total_login_bonus = sum(entry.get("data", {}).get("bn", 0) for entry in login_bonus)

    total_achievement_bonus = 0
    for item in achievement_bonus:
        try:
            a_id = item.get("data", {}).get("t")
            if a_id is None:
                continue
            amount, reward = get_achievement_reward(token, league_id, a_id)
            total_achievement_bonus += amount * reward
        except Exception as e:
            print(f"Warning: Failed to process achievement bonus: {e}")

    try:
        managers = get_managers(token, league_id)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch managers: {e}")

    performances = []
    for manager_name, manager_id in managers:
        try:
            info = get_manager_info(token, league_id, manager_id)
            team_value = info.get("tv", 0)

            perf = get_manager_performance(token, league_id, manager_id, manager_name)
            perf["Team Value"] = team_value
            performances.append(perf)
        except Exception as e:
            print(f"Warning: Could not fully process manager {manager_name}: {e}")
            performances.append({"name": manager_name, "tp": 0, "Team Value": 0})

    perf_df = pd.DataFrame(performances)
    if not perf_df.empty:
        perf_df["point_bonus"] = perf_df["tp"].fillna(0) * 1000
    else:
        perf_df = pd.DataFrame(columns=["name", "point_bonus", "Team Value"])

    # Important: initialize EVERY current league manager, not only managers who
    # already appeared in a transfer activity. This keeps late joiners visible.
    budgets = {manager_name: float(start_budget) for manager_name, _ in managers}
    transfer_counts = {manager_name: 0 for manager_name, _ in managers}

    if not activities_df.empty:
        for _, row in activities_df.iterrows():
            byr, slr, trp = row.get("byr"), row.get("slr"), row.get("trp", 0)
            trp = 0 if pd.isna(trp) else float(trp)

            if pd.notna(byr):
                budgets.setdefault(byr, float(start_budget))
                transfer_counts.setdefault(byr, 0)
            if pd.notna(slr):
                budgets.setdefault(slr, float(start_budget))
                transfer_counts.setdefault(slr, 0)

            if pd.isna(byr) and pd.notna(slr):
                budgets[slr] += trp
                transfer_counts[slr] += 1
            elif pd.isna(slr) and pd.notna(byr):
                budgets[byr] -= trp
                transfer_counts[byr] += 1
            elif pd.notna(byr) and pd.notna(slr):
                budgets[byr] -= trp
                budgets[slr] += trp
                transfer_counts[byr] += 1
                transfer_counts[slr] += 1

    budget_df = pd.DataFrame(
        [
            {"User": user, "Budget": budget, "Observed Transfers": transfer_counts.get(user, 0)}
            for user, budget in budgets.items()
        ]
    )

    budget_df = budget_df.merge(
        perf_df[["name", "point_bonus", "Team Value"]],
        left_on="User",
        right_on="name",
        how="left"
    ).drop(columns=["name"], errors="ignore")

    budget_df["Budget"] = budget_df["Budget"] + budget_df["point_bonus"].fillna(0)
    budget_df.drop(columns=["point_bonus"], inplace=True, errors="ignore")

    # Legacy estimation: the feed does not reliably attribute every login bonus.
    budget_df["Budget"] += total_login_bonus

    budget_df["Budget"] = budget_df["Budget"].astype(float)

    for user in budget_df["User"]:
        estimated_achievement_bonus = calc_achievement_bonus_by_points(
            token, league_id, user, total_achievement_bonus
        )
        budget_df.loc[budget_df["User"] == user, "Budget"] += estimated_achievement_bonus

    budget_df["Budget Confidence"] = "estimated"

    # Own budget from Kickbase is authoritative.
    try:
        own_budget = get_budget(token, league_id)
        own_username = get_username(token)
        mask = budget_df["User"] == own_username
        if mask.any():
            budget_df.loc[mask, "Budget"] = own_budget
            budget_df.loc[mask, "Budget Confidence"] = "exact"
    except Exception as e:
        print(f"Warning: Could not sync own budget: {e}")

    budget_df["Max Negative"] = (budget_df["Team Value"].fillna(0) + budget_df["Budget"]) * -0.33
    budget_df["Available Budget"] = (budget_df["Max Negative"].fillna(0) - budget_df["Budget"]) * -1

    budget_df.sort_values("Available Budget", ascending=False, inplace=True, ignore_index=True)

    return budget_df


def calc_achievement_bonus_by_points(token, league_id, username, anchor_achievement_bonus):
    """Estimate achievement bonus for a user based on total points vs. own account."""

    ranking = get_league_ranking(token, league_id)
    ranking_df = pd.DataFrame(ranking, columns=["Name", "Total Points"])

    if len(ranking_df) == 0:
        return 0

    anchor_user = get_username(token)
    anchor_row = ranking_df[ranking_df["Name"] == anchor_user]
    if anchor_row.empty:
        return 0
    anchor_points = anchor_row["Total Points"].values[0]

    if username == anchor_user:
        return anchor_achievement_bonus

    user_row = ranking_df[ranking_df["Name"] == username]
    if user_row.empty:
        return 0
    user_points = user_row["Total Points"].values[0]

    scale = 1.0 if anchor_points == 0 else user_points / anchor_points
    return anchor_achievement_bonus * scale


def calc_achievement_bonus_by_rank(token, league_id, username, anchor_achievement_bonus):
    """Estimate achievement bonus for a user based on ranking. Currently unused."""

    ranking = get_league_ranking(token, league_id)
    ranking_df = pd.DataFrame(ranking, columns=["Name", "Total Points"])

    if len(ranking_df) == 0:
        return 0

    anchor_user = get_username(token)
    anchor_row = ranking_df[ranking_df["Name"] == anchor_user]
    if anchor_row.empty:
        return 0
    anchor_rank = anchor_row.index[0] + 1

    if username == anchor_user:
        return anchor_achievement_bonus

    user_row = ranking_df[ranking_df["Name"] == username]
    if user_row.empty:
        return 0
    user_rank = user_row.index[0] + 1

    rank_diff = anchor_rank - user_rank
    scale = 1.0 + (rank_diff * 0.1)
    return anchor_achievement_bonus * scale
