from kickbase_api.config import BASE_URL, get_json_with_token

# All functions related to league data

def get_league_id(token, league_name):
    """Get the league ID based on an exact league-name match."""

    league_infos = get_leagues_infos(token)

    if not league_infos:
        raise RuntimeError("The Kickbase account is not part of any league.")

    selected_league = [league for league in league_infos if league["name"] == league_name]

    if not selected_league:
        available_names = ", ".join(sorted(league["name"] for league in league_infos))
        raise RuntimeError(
            f"No exact Kickbase league match for '{league_name}'. "
            f"Available leagues: {available_names}"
        )

    return selected_league[0]["id"]

def get_leagues_infos(token):
    """Get information about all leagues the user is part of."""

    url = f"{BASE_URL}/leagues/selection"
    data = get_json_with_token(url, token)

    result = []

    for item in data.get("it", []):
        result.append({
            "id": item.get("i"),
            "name": item.get("n")
        })

    return result

def get_league_activities(token, league_id, league_start_date):
    """Get league activities such as trades, logins, and achievements since the league start date."""

    # TODO magic number with 5000, have to find a better solution
    url = f"{BASE_URL}/leagues/{league_id}/activitiesFeed?max=5000"
    data = get_json_with_token(url, token)

    filtered_activities = []
    for entry in data["af"]:
        entry_date = entry.get("dt", "")
        if entry_date >= league_start_date:
            filtered_activities.append(entry)

    login = [entry for entry in filtered_activities if entry.get("t") == 22]
    achievements = [entry for entry in filtered_activities if entry.get("t") == 26]
    trade = [entry for entry in filtered_activities if entry.get("t") == 15]
    trading = [
        {k: entry["data"].get(k) for k in ["byr", "slr", "pi", "pn", "tid", "trp"]}
        for entry in trade
        if entry.get("t") == 15
    ]

    return trading, login, achievements

def get_league_players_on_market(token, league_id):
    """Get all players currently available on the market in the league."""

    url = f"{BASE_URL}/leagues/{league_id}/market"
    data = get_json_with_token(url, token)

    result = []

    for player in data.get('it', []):
        result.append({
            'id': player.get('i'),
            'prob': player.get('prob'),
            "exp": player.get("exs"),
        })

    return result

def get_league_ranking(token, league_id):
    """Get the overall league ranking."""

    url = f"{BASE_URL}/leagues/{league_id}/ranking"
    data = get_json_with_token(url, token)

    players = [(user["n"], user["sp"]) for user in data["us"]]
    ranked = sorted(players, key=lambda x: x[1], reverse=True)

    return ranked
