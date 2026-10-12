import unittest
from unittest.mock import Mock, patch

import requests

from kickbase_api.config import MAX_GET_ATTEMPTS, REQUEST_TIMEOUT, get_json_with_token


def response(status, payload=None, headers=None):
    result = Mock(status_code=status, headers=headers or {})
    result.json.return_value = payload
    if status >= 400:
        result.raise_for_status.side_effect = requests.exceptions.HTTPError(str(status))
    return result


class ApiRetryTests(unittest.TestCase):
    @patch("kickbase_api.config.time.sleep")
    @patch("kickbase_api.config.requests.get")
    def test_exact_502_failure_recovers(self, get, sleep):
        failed = response(502)
        get.side_effect = [failed, response(200, {"it": []})]
        self.assertEqual(get_json_with_token("https://example.test/marketvalue", "secret"), {"it": []})
        self.assertEqual(get.call_count, 2)
        self.assertEqual(get.call_args.kwargs["timeout"], REQUEST_TIMEOUT)
        self.assertEqual(get.call_args.kwargs["headers"], {"Authorization": "Bearer secret"})
        sleep.assert_called_once_with(1)
        failed.close.assert_called_once()

    @patch("kickbase_api.config.time.sleep")
    @patch("kickbase_api.config.requests.get")
    def test_timeout_and_connection_failure_recover(self, get, sleep):
        get.side_effect = [requests.exceptions.Timeout(), requests.exceptions.ConnectionError(), response(200, {"ok": True})]
        self.assertEqual(get_json_with_token("https://example.test", "secret"), {"ok": True})
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2])

    @patch("kickbase_api.config.time.sleep")
    @patch("kickbase_api.config.requests.get")
    def test_permanent_errors_not_retried(self, get, sleep):
        for status in (400, 401, 403, 404):
            get.reset_mock()
            get.side_effect = None
            get.return_value = response(status)
            with self.assertRaises(requests.exceptions.HTTPError):
                get_json_with_token("https://example.test", "secret")
            self.assertEqual(get.call_count, 1)
        sleep.assert_not_called()

    @patch("kickbase_api.config.time.sleep")
    @patch("kickbase_api.config.requests.get")
    def test_persistent_502_fails_instead_of_faking_data(self, get, sleep):
        get.side_effect = [response(502) for _ in range(MAX_GET_ATTEMPTS)]
        with self.assertRaises(requests.exceptions.HTTPError):
            get_json_with_token("https://example.test", "secret")
        self.assertEqual(get.call_count, MAX_GET_ATTEMPTS)
        self.assertEqual(sleep.call_count, MAX_GET_ATTEMPTS - 1)

    @patch("kickbase_api.config.time.sleep")
    @patch("kickbase_api.config.requests.get")
    def test_rate_limit_respects_bounded_retry_after(self, get, sleep):
        get.side_effect = [response(429, headers={"Retry-After": "999"}), response(200, {})]
        self.assertEqual(get_json_with_token("https://example.test", "secret"), {})
        sleep.assert_called_once_with(30)

    @patch("kickbase_api.config.time.sleep")
    @patch("kickbase_api.config.requests.get")
    def test_persistent_timeout_has_bounded_attempts(self, get, sleep):
        get.side_effect = requests.exceptions.Timeout()
        with self.assertRaises(requests.exceptions.Timeout):
            get_json_with_token("https://example.test", "secret")
        self.assertEqual(get.call_count, MAX_GET_ATTEMPTS)
        self.assertEqual(sleep.call_count, MAX_GET_ATTEMPTS - 1)


if __name__ == "__main__":
    unittest.main()
