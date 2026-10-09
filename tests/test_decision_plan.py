import copy
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from features.decision_plan import DEFAULTS, build_decision_plan, simulate_auctions, update_count


def fixture():
    squad = [{"player_id": str(i), "team_name": "club" + str(i), "mv": 1000,
              "last_name": str(i), "predicted_mv_target": 10} for i in range(14)]
    markets = [{"player_id": pid, "team_name": team, "mv": mv,
                "predicted_mv_target": delta, "mv_change_yesterday": delta}
               for pid, team, mv, delta in [("s", "Stuttgart", 870020, 281664),
                                           ("a", "Bremen", 1706759, 316271),
                                           ("t", "Schalke", 5692298, 271286)]]
    ends = ["2026-10-10T15:05:00+02:00", "2026-10-10T14:08:00+02:00", "2026-10-11T01:55:00+02:00"]
    return {"market_fetched_at": "2026-10-09T21:30:00+02:00", "squad": squad, "market": markets,
            "market_live": [{"player_id": p["player_id"], "player_name": p["player_id"],
                             "market_value": p["mv"], "expires_at": end,
                             "my_bid": 7111111 if p["player_id"] == "t" else None}
                            for p, end in zip(markets, ends)],
            "manager_budgets": [{"User": "Fabian", "Budget Confidence": "exact", "Budget": 27202496}],
            "points_profiles": [{"player_id": p["player_id"], "expected_points_next": 20} for p in markets] +
                               [{"player_id": str(i), "expected_points_next": 0 if i < 2 else 100} for i in range(14)],
            "transfer_history": [{"player_id": str(i), "buyer": "Fabian", "price": 500} for i in range(14)],
            "strategy": {"bid_guardrails": [{"player_id": p["player_id"], "league_p75_overpay_pct": 14.34} for p in markets]}}


class DecisionsTest(unittest.TestCase):
    def test_count_updates_and_boundary(self):
        now = datetime(2026, 10, 9, 21, 30, tzinfo=ZoneInfo("Europe/Zurich"))
        self.assertEqual(update_count(now, now.replace(hour=22, minute=0), DEFAULTS), 0)
        self.assertEqual(update_count(now, now.replace(day=10, hour=15), DEFAULTS), 1)
        self.assertEqual(update_count(now, now.replace(day=11, hour=1), DEFAULTS), 2)
        self.assertEqual(update_count(now.replace(hour=22), now.replace(day=10, hour=15), DEFAULTS), 0)

    def test_pricing_includes_updates_and_premium(self):
        plan = build_decision_plan(fixture(), DEFAULTS)
        spalt = plan["auction_bids"][0]
        self.assertEqual(spalt["projected_mv_at_expiry"], 1151684)
        self.assertGreater(spalt["suggested_bid"], spalt["projected_mv_at_expiry"])
        self.assertLessEqual(spalt["suggested_bid"], spalt["hard_max_bid"])
        self.assertEqual(plan["auction_bids"][2]["updates_before_expiry"], 2)

    def test_two_sales_preserve_later_tanaka(self):
        plan = build_decision_plan(fixture(), DEFAULTS)
        self.assertEqual(len(plan["recommended_replacements"]), 2)
        self.assertTrue(plan["recommended_plan_scenario"]["safe"])
        self.assertEqual(plan["recommended_plan_scenario"]["final_squad_size"], 15)
        bids = [{"player_id": p["player_id"], "team": p["team"], "expires_at": p["expires_at"]} for p in plan["auction_bids"]]
        scenario = simulate_auctions(fixture()["squad"], bids, ["0"], DEFAULTS)
        self.assertEqual(scenario["cancelled_bid_ids"], ["t"])

    def test_manager_stale_unknown_and_club_excluded(self):
        for field, value, reason in [("seller_name", "Pedro", "manager_listing_excluded"),
                                     ("market_value", 1200000, "prediction_market_value_mismatch_refresh_required"),
                                     ("expires_at", None, "unknown_or_expired_auction")]:
            report = fixture()
            report["market_live"][0][field] = value
            row = build_decision_plan(report, DEFAULTS)["auction_bids"][0]
            self.assertFalse(row["eligible"])
            self.assertEqual(row["reason"], reason)
            self.assertIsNone(row["suggested_bid"])
        report = fixture()
        for player in report["squad"][:2]:
            player["team_name"] = "Stuttgart"
        self.assertEqual(build_decision_plan(report, DEFAULTS)["auction_bids"][0]["reason"], "club_limit_requires_sale")

    def test_core_losses_cash_and_missing_data_protected(self):
        report = fixture()
        config = dict(DEFAULTS, protected_player_ids=["0", "1"])
        self.assertEqual(build_decision_plan(report, config)["recommended_replacements"], [])
        report["transfer_history"] = []
        self.assertEqual(build_decision_plan(report, DEFAULTS)["recommended_replacements"], [])
        report = fixture()
        report["manager_budgets"][0]["Budget"] = 0
        self.assertEqual(build_decision_plan(report, DEFAULTS)["recommended_replacements"], [])

    def test_unknown_existing_expiry_blocks_new_plan(self):
        report = fixture()
        report["market_live"][2]["expires_at"] = None
        self.assertEqual(build_decision_plan(report, DEFAULTS)["recommended_replacements"], [])
        self.assertFalse(build_decision_plan(report, DEFAULTS)["existing_bids_scenario"]["safe"])

    def test_snapshot_required_and_input_not_mutated(self):
        report = fixture()
        original = copy.deepcopy(report)
        build_decision_plan(report, DEFAULTS)
        self.assertEqual(report, original)
        report["market_fetched_at"] = None
        self.assertEqual(build_decision_plan(report, DEFAULTS)["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
