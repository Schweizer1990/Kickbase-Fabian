import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from kickbase_api.league import get_league_market_raw


TZ = ZoneInfo("Europe/Zurich")


def _int_or_none(value):
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _num_or_none(value):
    return value if isinstance(value, (int, float)) else None


def _offer_price(offer):
    for key in ("uop", "prc", "p", "price", "amt", "a", "v"):
        value = _num_or_none((offer or {}).get(key))
        if value is not None:
            return value
    return None


def capture_market_snapshot(token, league_id):
    """Fetch the market once and normalize fields needed for last-minute checks.

    The timestamp is created immediately after the API response arrives, so a
    caller can judge exactly how fresh the market data is. Kickbase only exposes
    the authenticated user's own offer on listings owned by somebody else. `ofc`
    is kept verbatim as an experimental upstream signal and is never converted
    into a claimed competitor count.
    """
    raw_market = get_league_market_raw(token, league_id)
    fetched_at_dt = datetime.now(TZ)
    fetched_at = fetched_at_dt.isoformat()

    entries = []
    for player in raw_market:
        seller = player.get("u") if isinstance(player.get("u"), dict) else {}
        visible_offers = player.get("ofs") or []
        is_own_listing = bool(player.get("iposl"))

        my_bid = None
        my_offer_id = None
        if not is_own_listing:
            my_bid = _num_or_none(player.get("uop"))
            if my_bid is None:
                for offer in visible_offers:
                    candidate = _offer_price(offer)
                    if candidate is not None:
                        my_bid = candidate
                        my_offer_id = offer.get("i") or offer.get("id") or offer.get("oi")
                        break
            if my_offer_id is None and visible_offers:
                offer = visible_offers[0]
                my_offer_id = offer.get("i") or offer.get("id") or offer.get("oi")

        incoming_prices = [
            price for price in (_offer_price(offer) for offer in visible_offers)
            if price is not None
        ] if is_own_listing else []

        expires_seconds = _int_or_none(player.get("exs"))
        expires_at = (
            (fetched_at_dt + timedelta(seconds=max(expires_seconds, 0))).isoformat()
            if expires_seconds is not None
            else None
        )

        raw_ofc = _int_or_none(player.get("ofc"))
        my_bid_present = bool(not is_own_listing and (my_bid is not None or visible_offers))

        entries.append({
            "player_id": str(player.get("i")) if player.get("i") is not None else None,
            "first_name": player.get("fn"),
            "player_name": player.get("n") or player.get("ln") or player.get("pn"),
            "team_id": str(player.get("tid")) if player.get("tid") is not None else None,
            "market_value": _num_or_none(player.get("mv")),
            "listed_price": _num_or_none(player.get("prc")),
            "expires_seconds": expires_seconds,
            "expires_at": expires_at,
            "listed_since": player.get("dt"),
            "seller_id": str(seller.get("i")) if seller.get("i") is not None else None,
            "seller_name": seller.get("n"),
            "is_own_listing": is_own_listing,
            "my_bid_present": my_bid_present,
            "my_bid": my_bid,
            "my_offer_id": my_offer_id,
            "visible_offer_count": len(visible_offers),
            "incoming_visible_offer_count": len(visible_offers) if is_own_listing else None,
            "incoming_highest_visible_bid": max(incoming_prices) if incoming_prices else None,
            "ofc_raw": raw_ofc,
            "competition_visibility": (
                "incoming_offers_visible_on_own_listing"
                if is_own_listing
                else "competitor_bids_hidden"
            ),
            "competitor_offer_count": None,
            "offer_count_interpretation": "experimental_opaque_signal",
            "uoid_raw": player.get("uoid"),
        })

    return {
        "fetched_at": fetched_at,
        "league_id": str(league_id),
        "entry_count": len(entries),
        "entries": entries,
    }


def visible_open_offers_from_snapshot(snapshot):
    """Return the authenticated user's visible outgoing bids from one snapshot."""
    offers = []
    for row in (snapshot or {}).get("entries", []):
        if row.get("is_own_listing") or not row.get("my_bid_present"):
            continue
        offers.append({
            "player_id": row.get("player_id"),
            "player_name": row.get("player_name"),
            "market_value": row.get("market_value"),
            "listed_price": row.get("listed_price"),
            "expires": row.get("expires_seconds"),
            "expires_at": row.get("expires_at"),
            "ofc_raw": row.get("ofc_raw"),
            "my_bid": row.get("my_bid"),
            "competition_visibility": row.get("competition_visibility"),
            "source": "live_market_snapshot",
        })
    return offers


def append_market_snapshot(snapshot, path="reports/market_snapshots.json", max_snapshots=300):
    """Persist compact market snapshots so `ofc` can be validated empirically."""
    target = Path(path)
    snapshots = []

    if target.exists():
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
            snapshots = payload.get("snapshots", []) if isinstance(payload, dict) else []
        except Exception as exc:
            print(f"Warning: Could not read market snapshot history: {exc}")

    compact_entries = []
    for row in (snapshot or {}).get("entries", []):
        compact_entries.append({
            "player_id": row.get("player_id"),
            "player_name": row.get("player_name"),
            "market_value": row.get("market_value"),
            "listed_price": row.get("listed_price"),
            "expires_seconds": row.get("expires_seconds"),
            "seller_id": row.get("seller_id"),
            "seller_name": row.get("seller_name"),
            "is_own_listing": row.get("is_own_listing"),
            "my_bid_present": row.get("my_bid_present"),
            "my_bid": row.get("my_bid"),
            "visible_offer_count": row.get("visible_offer_count"),
            "incoming_visible_offer_count": row.get("incoming_visible_offer_count"),
            "incoming_highest_visible_bid": row.get("incoming_highest_visible_bid"),
            "ofc_raw": row.get("ofc_raw"),
        })

    snapshots.append({
        "fetched_at": (snapshot or {}).get("fetched_at"),
        "league_id": (snapshot or {}).get("league_id"),
        "entries": compact_entries,
    })
    snapshots = snapshots[-max_snapshots:]

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({"snapshots": snapshots}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return target


def write_quick_market_report(snapshot, path="reports/market_quick.json"):
    """Write a minimal report optimised for fast auction checks."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return target
