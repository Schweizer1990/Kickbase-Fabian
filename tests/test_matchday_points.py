import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from features.matchday_points import (
    build_own_matchday_report, completed_matchday, current_season,
    manager_matchday_report, player_matchday_history,
)
from features.points import build_points_profiles
from features.reporting import save_latest_report


class MatchdayPointsTests(unittest.TestCase):
    def test_season_order_does_not_select_previous_year(self):
        old = {"sn": "2025/2026", "tp": 9000, "it": []}
        new = {"sn": "2026/2027", "tp": 600, "it": []}
        for seasons in ([old, new], [new, old]):
            self.assertEqual(current_season({"it": seasons}), new)
        self.assertIsNone(current_season({"it": [{"sid": "1"}, {"sid": "2"}]}))

    def test_manager_zero_is_real_and_future_zero_not_latest(self):
        payload = {"it": [{"sn": "2026/2027", "tp": 650, "it": [
            {"day": 7, "mdp": 0, "cur": True},
            {"day": 5, "mdp": 650}, {"day": 6, "mdp": 0},
            {"day": 8, "mdp": 0},
        ]}]}
        report = manager_matchday_report(payload, "me", "Fabian", 6)
        self.assertEqual(report["latest_played_matchday"], report["matchdays"][1])
        self.assertEqual(report["latest_played_matchday"]["points"], 0)
        self.assertEqual(report["current_matchday"]["matchday"], 7)
        self.assertEqual(report["total_points"], 650)
        self.assertFalse(report["historical_lineup_available"])

    def test_player_history_preserves_missing_negative_and_dates(self):
        payload = {"it": [{"ti": "2026/2027", "ph": [
            {"day": 7, "p": None, "mp": "0'", "md": "2026-10-18T12:00:00Z", "mdst": 0},
            {"day": 6, "p": -12, "mp": "45'", "md": "2026-10-11T12:00:00Z", "mdst": 2},
        ]}]}
        history = player_matchday_history(payload)
        self.assertEqual(history[0]["points"], -12)
        self.assertEqual(history[0]["minutes"], 45)
        self.assertIsNone(history[1]["points"])
        df = pd.DataFrame([{"matchday_history": history}])
        self.assertEqual(completed_matchday(df, datetime(2026, 10, 12, tzinfo=timezone.utc)), 6)
        history[1]["matchday_status"] = 2
        self.assertEqual(completed_matchday(df, datetime(2026, 10, 12, tzinfo=timezone.utc)), 6)

    def test_fetch_failure_is_not_reported_as_zero_points(self):
        with patch("features.matchday_points.get_username", side_effect=RuntimeError("private")):
            report = build_own_matchday_report("secret", "league", pd.DataFrame())
        self.assertEqual(report["status"], "fetch_failed")
        self.assertIsNone(report["total_points"])

    def test_manager_identity_and_authoritative_total(self):
        payload = {"it": [{"sn": "2026/2027", "tp": 4321, "it": [{"day": 6, "mdp": 789, "cur": True}]}]}
        with patch("features.matchday_points.get_username", return_value="Fabian"), \
             patch("features.matchday_points.get_managers", return_value=[("Other", "2"), ("Fabian", "1")]), \
             patch("features.matchday_points.get_json_with_token", return_value=payload) as fetch:
            report = build_own_matchday_report("token", "league", pd.DataFrame())
        self.assertTrue(fetch.call_args.args[0].endswith("/managers/1/performance"))
        self.assertEqual(report["current_matchday"]["points"], 789)
        self.assertEqual(report["total_points"], 4321)

    def test_profiles_capture_raw_results_without_extra_request(self):
        squad = pd.DataFrame([{"player_id": "1", "last_name": "Player", "mv": 1000000}])
        payload = {"it": [{"ti": "2026/2027", "ph": [{"day": 6, "p": 120, "mp": "90'", "md": "2026-10-11T12:00:00Z", "mdst": 2}]}]}
        with patch("features.points.get_json_with_token", return_value=payload) as fetch:
            profiles = build_points_profiles("token", "league", pd.DataFrame(), squad)
        self.assertEqual(fetch.call_count, 1)
        self.assertEqual(profiles.iloc[0]["matchday_history"][0]["points"], 120)

    def test_report_and_history_persist_manager_score_separately(self):
        empty = pd.DataFrame()
        manager = {"status": "available", "total_points": 4321,
                   "latest_recorded_matchday": {"matchday": 6, "points": 789}}
        profiles = pd.DataFrame([{"player_id": "1", "in_squad": True,
                                  "matchday_history": [{"matchday": 6, "points": 120}]}])
        with tempfile.TemporaryDirectory() as directory, \
             patch("features.reporting.Path", side_effect=lambda name: Path(directory) / name), \
             patch("features.reporting.build_decision_plan", return_value={"auction_bids": []}):
            output = save_latest_report("league", {}, empty, empty, empty,
                                        points_profiles_df=profiles, own_matchday_points=manager)
            latest = json.loads(output.read_text())
            history = json.loads((Path(directory) / "reports/history.json").read_text())["snapshots"][-1]
        self.assertEqual(latest["own_matchday_points"], manager)
        self.assertEqual(history["own_matchday_points"], manager)
        self.assertEqual(history["player_matchday_results"][0]["matchday_history"][0]["points"], 120)
        self.assertNotEqual(manager["latest_recorded_matchday"]["points"], 120)


if __name__ == "__main__":
    unittest.main()
