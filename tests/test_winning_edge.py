"""Regression tests for risk-aware team decisions, no API credentials needed."""
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo
from features.winning_edge import build_winning_edge, deadline

TZ = ZoneInfo("Europe/Zurich")


def player(pid, name, position, delta, prev):
    return {"player_id": str(pid), "last_name": name, "position": position,
            "mv": 1_000_000, "predicted_mv_target": delta,
            "mv_change_yesterday": prev}


class WinningEdgeTests(unittest.TestCase):
    def test_deadline_next_friday(self):
        self.assertEqual(deadline(datetime(2026, 11, 5, 19, tzinfo=TZ)).date().isoformat(),
                         "2026-11-06")
        self.assertEqual(deadline(datetime(2026, 11, 6, 21, tzinfo=TZ)).date().isoformat(),
                         "2026-11-13")

    def test_lineup_retains_trader_and_flags_unverified_health(self):
        squad = [player(1, "Goalie", 1, 30, 20)]
        squad += [player(i, "Def"+str(i), 2, 100, 90) for i in range(2, 7)]
        squad += [player(i, "Mid"+str(i), 3, 200, 150) for i in range(7, 12)]
        squad += [player(i, "Att"+str(i), 4, 300, 250) for i in range(12, 16)]
        points = [{"player_id": p["player_id"], "expected_points_next": 80}
                  for p in squad]
        points[-1]["expected_points_next"] = 4
        report = {"generated_at": "2026-11-05T18:00:00+01:00",
                  "squad": squad, "points_profiles": points,
                  "ligainsider_signals": [
                      {"player_id": "12", "ligainsider_status": "available",
                       "availability_factor": 1.0}],
                  "manager_budgets": [{"Budget Confidence": "exact", "Budget": -500}],
                  "my_open_offers": [{"my_bid": 1000}]}
        result = build_winning_edge(report, overrides={})
        self.assertEqual(len(result["lineup"]["players"]), 11)
        self.assertEqual(result["cash_to_recover_for_deadline"], 500)
        self.assertEqual(result["open_bid_commitment"], 1000)
        self.assertEqual(result["free_slots"], 0)
        self.assertEqual(result["player_decisions"][-1]["decision"], "hold_for_trading")
        self.assertIn("Att12", result["lineup_unverified_players"])
        risk = build_winning_edge(report, overrides={"12": {"status": "injured"}})
        starter = next(x for x in risk["player_decisions"] if x["player_id"] == "12")
        self.assertEqual(starter["expected_points_risk_adjusted"], 0)
        self.assertEqual(starter["availability_evidence"], "manual_override")
        self.assertNotIn("Att12", [p["name"] for p in risk["lineup"]["players"]])

    def test_sell_watch_requires_weak_points(self):
        report = {"generated_at": "2026-11-05T18:00:00+01:00",
                  "squad": [player(1, "Strong", 4, -20000, 1000),
                            player(2, "Weak", 4, -20000, -1000)],
                  "points_profiles": [
                      {"player_id": "1", "expected_points_next": 90},
                      {"player_id": "2", "expected_points_next": 5}]}
        result = build_winning_edge(report, overrides={})
        self.assertEqual(result["player_decisions"][0]["decision"],
                         "protect_points_watch_trend")
        self.assertEqual(result["player_decisions"][1]["decision"], "sell_candidate")


if __name__ == "__main__":
    unittest.main()
