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
    """Return the raw Kickbase market entries."""
    url = f"{BASE_URL}/leagues/{league_id}/market"
    data = get_json_with_token(url, token)
    return data.get("it", [])


def _offer_record(player, offer, source, context=None):
    """Normalize one visible offer from Kickbase."""
    context = context or {}
    price = offer.get("uop")
    price_key = "uop" if isinstance(price, (int, float)) else None

    # Compatibility fallback in case Kickbase changes the compact field name.
    if price_key is None:
        for key in ("prc", "p", "price", "amt", "a", "v"):
            value = offer.get(key)
            if isinstance(value, (int, float)) and value >= 0:
                price = value
                price_key = key
                break

    return {
        "player_id": str(player.get("i")) if player.get("i") is not None else None,
        "player_name": player.get("ln") or player.get("n") or player.get("pn"),
        "market_value": context.get("mv", player.get("mv")),
        "listed_price": context.get("prc", player.get("prc")),
        "expires": player.get("exs"),
        "offer_count": player.get("ofc"),
        "my_bid": price,
        "bid_price_key": price_key,
        "offer_user_id": offer.get("u") or offer.get("uoid"),
        "offer_user_name": offer.get("unm"),
        "source": source,
        "raw_offer": offer,
    }


def get_my_open_offers(token, league_id):
    """Return the authenticated user's visible outgoing open bids.

    Kickbase exposes the bid amount as `uop`. We first read inline `ofs[]` from
    the market endpoint. If that yields no outgoing offers, we fall back to the
    documented player-transfers endpoint for market entries that report offers.
    On somebody else's listing that endpoint exposes at most our own bid; players
    on our own sell list (`iposl`) are skipped so incoming bids are not mistaken
    for our outgoing bids.
    """
    market = get_league_market_raw(token, league_id)
    result = []

    for player in market:
        for offer in player.get("ofs") or []:
            result.append(_offer_record(player, offer, "market_inline"))

    if result:
        return result

    # Fallback only for entries that actually report offers, keeping API traffic
    # low while covering cases where `/market` omits the inline `ofs[]` payload.
    for player in market:
        if not player.get("ofc"):
            continue

        player_id = player.get("i")
        if player_id is None:
            continue

        try:
            url = f"{BASE_URL}/leagues/{league_id}/players/{player_id}/transfers"
            data = get_json_with_token(url, token)
        except Exception as exc:
            print(f"Warning: Could not inspect open offers for player {player_id}: {exc}")
            continue

        # `iposl` means the player is on our own sell list. In that case `ofs[]`
        # contains incoming bids from other managers, not our own outgoing bid.
        if data.get("iposl"):
            continue

        for offer in data.get("ofs") or []:
            result.append(_offer_record(player, offer, "player_transfers", data))

    return result


def get_league_players_on_market(token, league_id):
    """Get all players currently available on the market, including bid competition counts.

    Kickbase exposes the total number of offers as `ofc`. On another manager's
    listing, inline `ofs[]` contains our visible outgoing offer (if any), while the
    competing managers' bid amounts remain hidden. This lets us derive the number
    of competing offers without pretending to know their prices.
    """

    result = []
    for player in get_league_market_raw(token, league_id):
        raw_offer_count = player.get("ofc")
        try:
            offer_count = int(raw_offer_count or 0)
        except (TypeError, ValueError):
            offer_count = 0

        is_own_listing = bool(player.get("iposl"))
        visible_offers = player.get("ofs") or []
        my_bid_present = bool(visible_offers) and not is_own_listing

        # If `ofc` is unexpectedly absent, the visible offer list is still a safe
        # lower bound. Normally `ofc` is the total count and therefore preferred.
        if raw_offer_count is None:
            offer_count = len(visible_offers)

        competitor_offer_count = None
        if not is_own_listing:
            competitor_offer_count = max(offer_count - (1 if my_bid_present else 0), 0)

        result.append({
            "id": player.get("i"),
            "prob": player.get("prob"),
            "exp": player.get("exs"),
            "offer_count": offer_count,
            "my_bid_present": my_bid_present,
            "competitor_offer_count": competitor_offer_count,
            "is_own_listing": is_own_listing,
        })

    return result


def get_league_ranking(token, league_id):
    """Get the overall league ranking."""

    url = f"{BASE_URL}/leagues/{league_id}/ranking"
    data = get_json_with_token(url, token)

    players = [(user["n"], user["sp"]) for user in data.get("us", [])]
    ranked = sorted(players, key=lambda x: x[1], reverse=True)

    return ranked
