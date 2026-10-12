import time

import requests

BASE_URL = "https://api.kickbase.com/v4"
REQUEST_TIMEOUT = (5, 30)
MAX_GET_ATTEMPTS = 4
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


def _retry_delay(response, attempt):
    delay = 2 ** attempt
    if response is not None:
        try:
            delay = max(delay, float(response.headers.get("Retry-After", 0)))
        except (ValueError, TypeError):
            pass
    return min(delay, 30)


def get_json_with_token(url, token):
    """Retry transient failures on read-only GETs; never silently use stale data."""
    headers = {"Authorization": f"Bearer {token}"}
    for attempt in range(MAX_GET_ATTEMPTS):
        response = None
        try:
            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            if attempt == MAX_GET_ATTEMPTS - 1:
                raise
        else:
            if response.status_code not in RETRYABLE_STATUSES or attempt == MAX_GET_ATTEMPTS - 1:
                response.raise_for_status()
                return response.json()
            response.close()
        print(f"Temporary Kickbase GET failure; retry {attempt + 1}/{MAX_GET_ATTEMPTS - 1}.")
        time.sleep(_retry_delay(response, attempt))
