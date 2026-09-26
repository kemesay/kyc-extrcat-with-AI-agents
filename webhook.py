"""
Webhook delivery for completed extraction jobs.

Lets a caller pass callback_url to POST /extract/id instead of polling or
holding a long synchronous connection open — the standard pattern production
KYC/identity-verification vendors (Onfido, Jumio, Veriff) use for calls that
can take minutes: extractions on this hardware have taken up to ~640s on a
real document. Retries a few times with backoff since the receiving endpoint
may be transiently unavailable; never retries forever and never blocks or
delays the extraction pipeline itself — delivery happens after the job is
already done.
"""

import time

import httpx

MAX_ATTEMPTS = 3
BACKOFF_SECONDS = (2, 8, 30)  # delay before retry attempts 2 and 3
REQUEST_TIMEOUT_SECONDS = 15.0


def is_valid_callback_url(url: str) -> bool:
    return isinstance(url, str) and url.startswith(("http://", "https://"))


def deliver_callback(callback_url: str, payload: dict) -> dict:
    """POSTs payload to callback_url, retrying a few times on failure.

    Returns {"delivered": bool, "attempts": int, "last_error": str | None}.
    """
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = httpx.post(callback_url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            return {"delivered": True, "attempts": attempt, "last_error": None}
        except Exception as e:
            last_error = str(e)
            print(f"  ⚠️  callback delivery attempt {attempt}/{MAX_ATTEMPTS} to {callback_url} failed: {e}")
            if attempt < MAX_ATTEMPTS:
                time.sleep(BACKOFF_SECONDS[attempt - 1])
    return {"delivered": False, "attempts": MAX_ATTEMPTS, "last_error": last_error}
