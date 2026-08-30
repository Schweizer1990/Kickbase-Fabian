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

    # The API does not currently expose pagination here in this client. 5000 is
    # sufficient for a young league, while the report exposes the activity count
    # so we can detect when this needs to be revisited.
    url = f"{BASE_URL}/leagues/{league_id}/activitiesFeed?max=5000"
    data = get_json_with_token(url, token)

    filtered_activities = []
    for entry in data.get("af", []):
        entry_date = entry.get("dt", "")
        if entry_date >= league_start_date:
            filtered_activities.append(entry)

    login = [entry for entry in filtered_activities if entry.get("t") == 22]
    achievements = [entry for entry in filtered_activities if entry.get("t") == 26]
    trade = [entry for entry in filtered_activities if entry.get("t") == 15]
    trading = []
    for entry in trade:
        transfer = {k: entry.get("data", {}).get(k) for k in ["byr", "slr", "pi", "pn", "tid", "trp"]}
        transfer["dt"] = entry.get("dt")
        trading.append(transfer)

    return trading, login, achievements


def get_league_market_raw(token, league_id):
    """Return the raw Kickbase market entries.

    Market entries can contain `ofs[]`. For players listed by somebody else,
    Kickbase only exposes the authenticated user's own outgoing offer there;
    competing managers' bids remain hidden.
    """
    url = f"{BASE_URL}/leagues/{league_id}/market"
    data = get_json_with_token(url, token)
    return data.get("it", [])


def get_my_open_offers(token, league_id):
    """Extract the authenticated user's visible outgoing market offers.

    We intentionally do not guess which numeric field is the bid price. The raw
    offer object is retained, and common observed price keys are checked. This
    makes the report useful immediately while remaining resilient to API changes.
    """
    result = []
    for player in get_league_market_raw(token, league_id):
        offers = player.get("ofs") or []
        for offer in offers:
            # On somebody else's listing, any visible offer is our own bid.
            # If ownership/listing flags are present, keep them for auditability.
            price = None
            price_key = None
            for key in ("prc", "p", "price", "amt", "a", "v"):
                value = offer.get(key)
                if isinstance(value, (int, float)) and value >= 0:
                    price = value
                    price_key = key
                    break

            result.append({
                "player_id": str(player.get("i")) if player.get("i") is not None else None,
                "player_name": player.get("ln") or player.get("n") or player.get("pn"),
                "market_value": player.get("mv"),
                "expires": player.get("exs"),
                "offer_count": player.get("ofc"),
                "my_bid": price,
                "bid_price_key": price_key,
                "offer_user_id": offer.get("u") or offer.get("uoid"),
                "offer_user_name": offer.get("unm"),
                "raw_offer": offer,
            })
    return result


def get_league_players_on_market(token, league_id):
    """Get all players currently available on the market in the league."""

    result = []
    for player in get_league_market_raw(token, league_id):
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

    players = [(user["n"], user["sp"]) for user in data.get("us", [])]
    ranked = sorted(players, key=lambda x: x[1], reverse=True)

    return ranked
