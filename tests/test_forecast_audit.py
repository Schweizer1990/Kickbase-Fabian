import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from features.predictions.audit import update_forecast_audit, summarize_forecasts
from features.predictions.modeling import evaluate_model
from features.predictions.preprocessing import preprocess_player_data


def history(day='2026-10-09', mv=1000, players=4):
    return pd.DataFrame([{'player_id': str(i), 'date': day, 'mv': mv} for i in range(players)])


def forecasts(day='2026-10-09', mv=1000, predicted=100):
    return pd.DataFrame([{'player_id': '0', 'date': day, 'mv': mv,
                          'predicted_mv_target': predicted, 'mv_change_1d': 200}])


class ForecastAuditTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'audit.json'

    def tearDown(self):
        self.temp.cleanup()

    def run_audit(self, data, predictions=None):
        return update_forecast_audit(data, predictions if predictions is not None else pd.DataFrame(),
                                     self.path, observed_at='2026-10-09T22:00:00+02:00')

    def state(self):
        return json.loads(self.path.read_text())

    def test_rerun_freezes_prediction_without_fake_update(self):
        first = self.run_audit(history(), forecasts())
        self.assertEqual(first['market_update']['status'], 'baseline')
        second = self.run_audit(history(), forecasts(predicted=-300))
        self.assertFalse(second['market_update']['confirmed'])
        self.assertEqual(len(self.state()['forecasts']), 1)
        self.assertEqual(self.state()['forecasts'][0]['predicted_change'], 100)
        self.assertEqual(second['live_forecast_evaluation']['overall']['count'], 0)

    def test_dated_update_and_score_once(self):
        self.run_audit(history(), forecasts())
        data = pd.concat([history(), history('2026-10-10', 1200)])
        result = self.run_audit(data)
        self.assertTrue(result['market_update']['confirmed'])
        metrics = result['live_forecast_evaluation']['overall']
        self.assertEqual(metrics['count'], 1)
        self.assertEqual(metrics['rmse'], 100)
        self.assertEqual(metrics['direction_accuracy_percent'], 100)
        self.assertEqual(result['live_forecast_evaluation']['cohorts']['flattening']['flattening_accuracy_percent'], 0)
        self.assertEqual(self.run_audit(data)['live_forecast_evaluation']['overall']['count'], 1)

    def test_unchanged_prices_with_new_date_still_confirm_update(self):
        self.run_audit(history())
        result = self.run_audit(pd.concat([history(), history('2026-10-10')]))
        self.assertTrue(result['market_update']['confirmed'])

    def test_one_player_update_is_not_league_update(self):
        self.run_audit(history())
        result = self.run_audit(pd.concat([history(), history('2026-10-10', players=1)]))
        self.assertFalse(result['market_update']['confirmed'])
        self.assertEqual(result['market_update']['advanced_player_count'], 1)

    def test_same_date_revision_does_not_prove_update(self):
        self.run_audit(history())
        result = self.run_audit(history(mv=1300))
        self.assertFalse(result['market_update']['confirmed'])
        self.assertEqual(result['market_update']['same_date_revision_count'], 4)

    def test_skipped_day_is_not_one_day_result(self):
        self.run_audit(history(), forecasts())
        result = self.run_audit(pd.concat([history(), history('2026-10-11', 1300)]))
        self.assertEqual(result['live_forecast_evaluation']['pending_count'], 1)
        self.assertEqual(result['live_forecast_evaluation']['overall']['count'], 0)

    def test_retrospective_or_mismatched_base_is_rejected(self):
        self.run_audit(pd.concat([history(), history('2026-10-10', 1300)]), forecasts())
        self.assertEqual(self.state()['forecasts'], [])
        self.run_audit(history(), forecasts(mv=1234))
        self.assertEqual(self.state()['forecasts'], [])

    def test_revised_base_invalidates_pending_sample(self):
        self.run_audit(history(), forecasts())
        result = self.run_audit(pd.concat([history(mv=999), history('2026-10-10', 1300)]))
        self.assertEqual(result['live_forecast_evaluation']['invalid_count'], 1)
        self.assertEqual(result['live_forecast_evaluation']['overall']['count'], 0)

    def test_cohorts_and_root_mean_squared_error(self):
        rows = [{'status': 'evaluated', 'predicted_change': p, 'actual_change': a, 'cohorts': c}
                for p, a, c in [(3, 0, ['traders', 'flattening']), (-4, 0, ['losses'])]]
        summary = summarize_forecasts(rows)
        self.assertAlmostEqual(summary['overall']['rmse'], 3.54)
        self.assertEqual(summary['cohorts']['flattening']['count'], 1)
        self.assertEqual(summary['cohorts']['losses']['direction_accuracy_percent'], 0)
        self.assertIsNone(summarize_forecasts([])['overall']['rmse'])
        class Model:
            def predict(self, X):
                return np.array([3., -4.])
        _, rmse, mae, _ = evaluate_model(Model(), None, np.array([0., 0.]))
        self.assertAlmostEqual(rmse, np.sqrt(12.5))
        self.assertEqual(mae, 3.5)

    def test_preprocessing_selects_observed_base_once_per_player(self):
        rows = []
        for pid in ('0', '1'):
            for offset in range(10):
                rows.append({'player_id': pid, 'team_id': 1, 't1': 1, 't2': 2,
                             'date': pd.Timestamp('2026-10-01') + pd.Timedelta(days=offset),
                             'md': '2026-10-01' if offset < 9 else '2026-10-20',
                             'mv': 1000 + offset * 10 if offset < 9 else np.nan,
                             'p': 0, 'ppm': 0, 'won': 0})
        training, live = preprocess_player_data(pd.DataFrame(rows))
        self.assertEqual(len(live), 2)
        self.assertTrue((live['date'] == pd.Timestamp('2026-10-09')).all())
        self.assertTrue((training['date'] < pd.Timestamp('2026-10-09')).all())


if __name__ == '__main__':
    unittest.main()
