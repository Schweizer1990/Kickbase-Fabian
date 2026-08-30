import math
import re
import unicodedata
from html import unescape
from html.parser import HTMLParser

import pandas as pd
import requests

BASE = "https://www.ligainsider.de"
INJURIES_URL = f"{BASE}/bundesliga/verletzte-und-gesperrte-spieler/"

TEAM_PATHS = {
    "Bayern": "/fc-bayern-muenchen/1/",
    "Leverkusen": "/bayer-04-leverkusen/9/",
    "Frankfurt": "/eintracht-frankfurt/3/",
    "Dortmund": "/borussia-dortmund/14/",
    "Freiburg": "/sc-freiburg/18/",
    "Mainz": "/1-fsv-mainz-05/17/",
    "Leipzig": "/rb-leipzig/12/",
    "Bremen": "/sv-werder-bremen/2/",
    "Stuttgart": "/vfb-stuttgart/11/",
    "M'gladbach": "/borussia-moenchengladbach/5/",
    "Gladbach": "/borussia-moenchengladbach/5/",
    "Wolfsburg": "/vfl-wolfsburg/4/",
    "Augsburg": "/fc-augsburg/7/",
    "Union Berlin": "/1-fc-union-berlin/23/",
    "St. Pauli": "/fc-st-pauli/26/",
    "Hoffenheim": "/tsg-hoffenheim/10/",
    "Heidenheim": "/1-fc-heidenheim-1846/29/",
    "Köln": "/1-fc-koeln/15/",
    "Hamburg": "/hamburger-sv/8/",
    "HSV": "/hamburger-sv/8/",
    "Schalke": "/fc-schalke-04/13/",
    "Paderborn": "/sc-paderborn-07/20/",
}

# LigaInsider exposes the status as the alt text of the icon immediately before
# the affected player. Parsing that structure is much safer than looking for
# keywords in a large text window around a surname.
STATUS_ALTS = {
    "verletzung": ("injured", 0.0),
    "aufbautraining": ("rehab", 0.45),
    "nicht im kader": ("out", 0.0),
    "gelb rote karte": ("suspended", 0.0),
    "rote karte": ("suspended", 0.0),
    "gelbe karte": ("suspended", 0.0),
    "gesperrt": ("suspended", 0.0),
    "schwerer angeschlagen": ("likely_out", 0.15),
    "angeschlagen": ("doubtful", 0.60),
    "leicht angeschlagen": ("slight_knock", 0.82),
}


def _clean_text_value(value):
    """Return a real string or None; pandas NaN must not leak into matching/sorting."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _norm(value):
    value = _clean_text_value(value)
    if not value:
        return ""
    text = unicodedata.normalize("NFKD", value)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _plain(html):
    html = re.sub(r"<script\b[^>]*>.*?</script>", " ", html, flags=re.I | re.S)
    html = re.sub(r"<style\b[^>]*>.*?</style>", " ", html, flags=re.I | re.S)
    html = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", unescape(html)).strip()


def _get(url):
    headers = {"User-Agent": "Kickbase-Fabian/1.0 (+personal fantasy-football analysis)"}
    response = requests.get(url, headers=headers, timeout=12)
    response.raise_for_status()
    return response.text


def _collect_source(market_df, squad_df):
    players = {}
    for df in (market_df, squad_df):
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            pid = row.get("player_id")
            if pid is None or (isinstance(pid, float) and math.isnan(pid)):
                continue
            players[str(pid)] = {
                "player_id": str(pid),
                "player_name": _clean_text_value(row.get("last_name")),
                "team": _clean_text_value(row.get("team_name")),
            }
    return players


class _InjuryStatusParser(HTMLParser):
    """Extract (player name, status icon) pairs from LigaInsider's table."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.pending = None
        self.in_anchor = False
        self.anchor_parts = []
        self.records = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag.lower() == "img":
            alt = _norm(attrs.get("alt"))
            if alt in STATUS_ALTS:
                label, factor = STATUS_ALTS[alt]
                self.pending = {
                    "status": label,
                    "availability_factor": factor,
                    "status_raw": attrs.get("alt"),
                }
        elif tag.lower() == "a" and self.pending:
            self.in_anchor = True
            self.anchor_parts = []

    def handle_data(self, data):
        if self.in_anchor and self.pending:
            text = data.strip()
            if text:
                self.anchor_parts.append(text)

    def handle_endtag(self, tag):
        if tag.lower() != "a" or not self.in_anchor:
            return
        self.in_anchor = False
        name = " ".join(self.anchor_parts).strip()
        self.anchor_parts = []
        if not name or not self.pending:
            return
        # News links can follow a player inside the same row. The first textual
        # anchor after a recognized status icon is the player name, so consume
        # the status immediately after recording it.
        self.records.append({
            "full_name": name,
            "name_norm": _norm(name),
            **self.pending,
        })
        self.pending = None


def _parse_injury_statuses(html):
    parser = _InjuryStatusParser()
    parser.feed(html)
    return parser.records


def _match_status(player_name, records):
    """Match Kickbase's usually-short surname against structured LigaInsider rows.

    Exact normalized names win. For Kickbase surname-only values we accept a
    unique token/suffix match. Ambiguous matches deliberately return unknown.
    """
    name = _norm(player_name)
    if not name:
        return None

    exact = [r for r in records if r["name_norm"] == name]
    if len(exact) == 1:
        return exact[0]

    candidates = []
    for record in records:
        full = record["name_norm"]
        tokens = full.split()
        if not tokens:
            continue
        if name == tokens[-1] or name in tokens or full.endswith(" " + name):
            candidates.append(record)

    return candidates[0] if len(candidates) == 1 else None


def build_ligainsider_signals(market_df, squad_df):
    """Read public LigaInsider status and Topelf pages conservatively.

    The integration is intentionally fail-open: if LigaInsider is unavailable or
    a name cannot be matched, Kickbase projections are left unchanged. Topelf
    presence means projected XI *or an explicitly shown alternative*, not a
    guaranteed starter.
    """
    players = _collect_source(market_df, squad_df)
    if not players:
        return pd.DataFrame()

    injury_records = []
    try:
        injury_html = _get(INJURIES_URL)
        injury_records = _parse_injury_statuses(injury_html)
        print(f"LigaInsider status rows parsed: {len(injury_records)}.")
    except Exception as exc:
        print(f"Warning: LigaInsider injury page unavailable: {exc}")

    team_text = {}
    teams = sorted({p["team"] for p in players.values() if isinstance(p.get("team"), str) and p["team"]})
    for team in teams:
        path = TEAM_PATHS.get(team)
        if not path:
            continue
        try:
            raw = _plain(_get(BASE + path))
            normalized = _norm(raw)
            # Limit matching to the forecast block when the markers are present.
            start = normalized.find("topelf 2026 27")
            end = normalized.find("letzte aktualisierung", start + 1) if start >= 0 else -1
            if start >= 0 and end > start:
                normalized = normalized[start:end]
            team_text[team] = normalized
        except Exception as exc:
            print(f"Warning: LigaInsider lineup page unavailable for {team}: {exc}")

    rows = []
    for player in players.values():
        name = _norm(player.get("player_name"))
        team = player.get("team")
        lineup_text = team_text.get(team, "")
        in_topelf_pool = bool(name and re.search(rf"\b{re.escape(name)}\b", lineup_text))

        matched = _match_status(player.get("player_name"), injury_records)
        if matched:
            status = matched["status"]
            availability_factor = matched["availability_factor"]
            matched_name = matched["full_name"]
            status_raw = matched["status_raw"]
        else:
            # "available" means the player was successfully checked against the
            # current structured absence list and was not found there. This is
            # more useful than calling every healthy player "unknown".
            status = "available" if injury_records else "unknown"
            availability_factor = 1.0
            matched_name = None
            status_raw = None

        rows.append({
            **player,
            "ligainsider_status": status,
            "availability_factor": availability_factor,
            "ligainsider_topelf_pool": in_topelf_pool,
            "ligainsider_matched_name": matched_name,
            "ligainsider_status_raw": status_raw,
            "ligainsider_source": "public_web",
        })

    return pd.DataFrame(rows)
