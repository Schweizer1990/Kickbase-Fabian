from kickbase_api.config import BASE_URL, get_json_with_token

# All functions related to manager data

def get_managers(token, league_id):
    """Get a list of all managers in the league with their IDs and names."""

    url = f"{BASE_URL}/leagues/{league_id}/ranking"
    data = get_json_with_token(url, token)

    user_info = [(user["n"], user["i"]) for user in data["us"]]

    return user_info


def get_manager_info(token, league_id, manager_id):
    """Get detailed information about a specific manager in the league."""

    url = f"{BASE_URL}/leagues/{league_id}/managers/{manager_id}/dashboard"
    data = get_json_with_token(url, token)

    return data


def get_manager_squad(token, league_id, manager_id):
    """Get the current squad of any manager in the league."""

    url = f"{BASE_URL}/leagues/{league_id}/managers/{manager_id}/squad"
    return get_json_with_token(url, token)


def get_manager_performance(token, league_id, manager_id, manager_name):
    """Get performance data for a specific manager in the current season.

    Kickbase changes season IDs between seasons, so do not hard-code one.
    The performance endpoint returns the seasons in current-to-older order;
    therefore the first entry is the current season for that manager.
    """

    url = f"{BASE_URL}/leagues/{league_id}/managers/{manager_id}/performance"
    data = get_json_with_token(url, token)

    seasons = data.get("it", [])
    if not seasons:
        print(f"Warning: No season performance found for {manager_name}")
        tp_value = 0
    else:
        current_season = seasons[0]
        tp_value = current_season.get("tp", 0)

    return {
        "name": manager_name,
        "tp": tp_value
    }
