import re
import unicodedata
from html import unescape

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

STATUS_KEYWORDS = [
    ("gesperrt", "suspended", 0.0),
    ("nicht im kader", "out", 0.0),
    ("verletzung", "injured", 0.0),
    ("schwerer angeschlagen", "likely_out", 0.15),
    ("aufbautraining", "rehab", 0.45),
    ("angeschlagen", "doubtful", 0.60),
    ("leicht angeschlagen", "slight_knock", 0.82),
]


def _norm(value):
    text = unicodedata.normalize("NFKD", str(value or ""))
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
            if pid is None:
                continue
            players[str(pid)] = {
                "player_id": str(pid),
                "player_name": row.get("last_name"),
                "team": row.get("team_name"),
            }
    return players


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

    try:
        injury_text = _norm(_plain(_get(INJURIES_URL)))
    except Exception as exc:
        print(f"Warning: LigaInsider injury page unavailable: {exc}")
        injury_text = ""

    team_text = {}
    for team in sorted({p["team"] for p in players.values() if p.get("team")}):
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
        in_topelf_pool = bool(name and name in lineup_text)

        status = "unknown"
        availability_factor = 1.0
        matched_status = False
        if name and injury_text:
            positions = [m.start() for m in re.finditer(rf"\b{re.escape(name)}\b", injury_text)]
            for pos in positions:
                window = injury_text[max(0, pos - 180): pos + 220]
                for keyword, label, factor in STATUS_KEYWORDS:
                    if _norm(keyword) in window:
                        status = label
                        availability_factor = factor
                        matched_status = True
                        break
                if matched_status:
                    break

        rows.append({
            **player,
            "ligainsider_status": status,
            "availability_factor": availability_factor,
            "ligainsider_topelf_pool": in_topelf_pool,
            "ligainsider_source": "public_web",
        })

    return pd.DataFrame(rows)
