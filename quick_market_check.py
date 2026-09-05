import os

from dotenv import load_dotenv

from features.market_monitor import (
    append_market_snapshot,
    capture_market_snapshot,
    write_quick_market_report,
)
from kickbase_api.league import get_league_id
from kickbase_api.user import login


load_dotenv()

username = os.getenv("KICK_USER")
password = os.getenv("KICK_PASS")
league_name = os.getenv("KICKBASE_LEAGUE_NAME")

missing = [
    name for name, value in {
        "KICK_USER": username,
        "KICK_PASS": password,
        "KICKBASE_LEAGUE_NAME": league_name,
    }.items()
    if not value
]
if missing:
    raise RuntimeError("Missing required GitHub Secret/Variable(s): " + ", ".join(missing))

token = login(username, password)
league_id = get_league_id(token, league_name)

snapshot = capture_market_snapshot(token, league_id)
quick_path = write_quick_market_report(snapshot)
history_path = append_market_snapshot(snapshot)

print(
    f"Quick market snapshot captured at {snapshot['fetched_at']} "
    f"with {snapshot['entry_count']} market players."
)
print(f"Quick report written to {quick_path}.")
print(f"Market snapshot history updated at {history_path}.")
