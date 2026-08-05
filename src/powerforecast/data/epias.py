"""Authentication against the EPİAŞ Şeffaflık Platform.

EPİAŞ does not issue a static API key. Credentials are exchanged for a
ticket-granting ticket (TGT), a string beginning with ``TGT-`` that is then sent
in the ``TGT`` header of every subsequent request.

Two independent expiry rules apply, and both must be respected:

* an **absolute lifetime** of two hours from issue, and
* an **idle timeout** of 45 minutes, which resets on each use.

A ticket kept warm by regular polling therefore still dies at the two-hour mark,
while an idle one dies much sooner. Getting this wrong produces the worst kind of
bug: a pipeline that works all through development and fails in the third hour of
its first unattended run.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

TGT_URL = "https://giris.epias.com.tr/cas/v1/tickets"
BASE_URL = "https://seffaflik.epias.com.tr/electricity-service"

# Turkish market time. Türkiye has used a fixed UTC+03:00 since 2016 with no
# daylight saving, which is why request timestamps can carry a literal offset.
ISTANBUL_OFFSET = "+03:00"

TGT_LIFETIME = timedelta(hours=2)
TGT_IDLE_TIMEOUT = timedelta(minutes=45)

# Refresh slightly early rather than discovering expiry through a failed request.
# The window has to cover clock skew plus the flight time of a request that was
# valid when it left and expired before it arrived.
EXPIRY_MARGIN = timedelta(minutes=2)

DEFAULT_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


class EpiasAuthError(RuntimeError):
    """Raised when a ticket cannot be obtained."""


def _utcnow() -> datetime:
    return datetime.now(UTC)


class EpiasAuth:
    """Obtains and caches an EPİAŞ ticket, refreshing it when it expires.

    The clock is injected rather than read from the module. Expiry is the whole
    point of this class, and a two-hour lifetime cannot be tested against a real
    clock — so the thing most likely to break is also the thing hardest to
    verify unless time is a parameter.
    """

    def __init__(
        self,
        username: str,
        password: str,
        *,
        client: httpx.Client | None = None,
        clock: Callable[[], datetime] = _utcnow,
        tgt_url: str = TGT_URL,
    ) -> None:
        self._username = username
        self._password = password
        self._client = client or httpx.Client(timeout=DEFAULT_TIMEOUT)
        self._clock = clock
        self._tgt_url = tgt_url

        self._ticket: str | None = None
        self._issued_at: datetime | None = None
        self._last_used_at: datetime | None = None

    def ticket(self) -> str:
        """Return a valid ticket, fetching a new one only when necessary."""
        now = self._clock()

        if self._ticket is None or self._is_expired(now):
            self._ticket = self._fetch_ticket()
            self._issued_at = now

        self._last_used_at = now
        return self._ticket

    def invalidate(self) -> None:
        """Discard the cached ticket.

        Needed because the server is the authority on validity: it may drop a
        ticket during maintenance or a session reset while our clock still
        considers it fresh. Without this, a client would keep resending a ticket
        the server has already forgotten.
        """
        self._ticket = None
        self._issued_at = None
        self._last_used_at = None

    def _is_expired(self, now: datetime) -> bool:
        if self._issued_at is None or self._last_used_at is None:
            return True

        aged_out = now - self._issued_at >= TGT_LIFETIME - EXPIRY_MARGIN
        idled_out = now - self._last_used_at >= TGT_IDLE_TIMEOUT - EXPIRY_MARGIN
        return aged_out or idled_out

    def _fetch_ticket(self) -> str:
        try:
            response = self._client.post(
                self._tgt_url,
                # httpx form-encodes a dict and percent-escapes the values, so a
                # password containing & or = cannot corrupt the request body.
                data={"username": self._username, "password": self._password},
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "text/plain",
                },
            )
        except httpx.HTTPError as exc:
            raise EpiasAuthError(f"could not reach the EPİAŞ login service: {exc}") from exc

        if response.status_code in (401, 403):
            raise EpiasAuthError(
                "EPİAŞ rejected the credentials (check EPIAS_USERNAME and EPIAS_PASSWORD in .env)"
            )
        if response.status_code >= 400:
            raise EpiasAuthError(
                f"EPİAŞ login failed with HTTP {response.status_code}: {response.text[:200]}"
            )

        body = response.text.strip()
        if not body.startswith("TGT-"):
            # Almost always an HTML login or maintenance page. Failing here names
            # the real problem; storing it would surface as a puzzling 401 later.
            raise EpiasAuthError(
                "unexpected response from the EPİAŞ login service: expected a ticket "
                f"starting with 'TGT-', got {body[:80]!r}"
            )
        return body

    def __repr__(self) -> str:
        # Never interpolate the password: reprs end up in logs and tracebacks.
        state = "authenticated" if self._ticket else "unauthenticated"
        return f"EpiasAuth(username={self._username!r}, state={state!r})"


# --------------------------------------------------------------------------- #
# Series catalogue
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SeriesSpec:
    """One time series available from the platform.

    Endpoints are not uniform — the value field is named differently on each, and
    responses wrap their summary under `statistics` on one endpoint and
    `statistic` on another. Describing each series in data rather than writing a
    function per endpoint keeps that irregularity in one readable place.
    """

    name: str
    path: str
    value_field: str
    column: str
    unit: str


CONSUMPTION = SeriesSpec(
    name="consumption",
    path="/v1/consumption/data/realtime-consumption",
    value_field="consumption",
    column="consumption_mwh",
    unit="MWh",
)

DAY_AHEAD_PRICE = SeriesSpec(
    name="day_ahead_price",
    path="/v1/markets/dam/data/mcp",
    value_field="price",
    column="price_try_mwh",
    unit="TRY/MWh",
)

# The system operator's own published forecast. Worth collecting as a benchmark:
# beating a seasonal-naive baseline says little, while comparing against the
# forecast the market actually runs on is a real test.
LOAD_PLAN = SeriesSpec(
    name="load_plan",
    path="/v1/consumption/data/load-estimation-plan",
    value_field="lep",
    column="load_plan_mwh",
    unit="MWh",
)

ALL_SERIES = (CONSUMPTION, DAY_AHEAD_PRICE, LOAD_PLAN)


# --------------------------------------------------------------------------- #
# Request layer
# --------------------------------------------------------------------------- #

# Statuses worth trying again. The common thread is that the same request, sent
# later, can legitimately succeed: the server was busy, throttling, or briefly
# unavailable. Everything else — a bad date range, a wrong path, rejected
# credentials — will fail identically on every attempt, so retrying it only
# delays the error and consumes rate-limit budget.
TRANSIENT_STATUS = frozenset({429, 500, 502, 503, 504})

MAX_ATTEMPTS = 4
BACKOFF_BASE_SECONDS = 1.0
MIN_REQUEST_INTERVAL = 0.2

# Observed in practice: the platform's quota is cumulative across endpoints, not
# per-endpoint. A long backfill can run for dozens of requests and then hit 429
# once the window fills. Backing off a single request is not enough — the whole
# loop has to slow down, and stay slow, because the next request will meet the
# same exhausted quota.
RATE_LIMIT_GROWTH = 2.0
MAX_REQUEST_INTERVAL = 8.0


class EpiasRequestError(RuntimeError):
    """Raised when a request fails permanently or exhausts its retries."""


def _full_jitter(delay: float) -> float:
    """Spread retries randomly across the interval instead of synchronising them.

    Without jitter, every client that failed at the same moment retries at the
    same moment, and the recovering server is hit by the same spike that knocked
    it over — the thundering herd. Sampling uniformly from [0, delay] breaks the
    lockstep.
    """
    return random.uniform(0, delay)


class EpiasClient:
    """Sends authenticated requests to EPİAŞ, with retries and rate limiting.

    Both `sleep` and the monotonic clock are injected for the same reason the
    auth clock is: a test covering a 30-second backoff must not take 30 seconds,
    and the delays actually requested are the thing worth asserting on.
    """

    def __init__(
        self,
        auth: EpiasAuth,
        *,
        base_url: str = BASE_URL,
        client: httpx.Client | None = None,
        min_interval: float = MIN_REQUEST_INTERVAL,
        max_attempts: int = MAX_ATTEMPTS,
        backoff_base: float = BACKOFF_BASE_SECONDS,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[float], float] = _full_jitter,
    ) -> None:
        self._auth = auth
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=DEFAULT_TIMEOUT)
        self._min_interval = min_interval
        self._max_attempts = max_attempts
        self._backoff_base = backoff_base
        # Monotonic, not wall clock: rate limiting measures elapsed time, and a
        # wall clock can jump backwards on an NTP correction.
        self._monotonic = monotonic
        self._sleep = sleep
        self._jitter = jitter

        self._last_request_at: float | None = None

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST a JSON payload and return the decoded response.

        Raises:
            EpiasRequestError: on a permanent failure, or after retries run out.
            EpiasAuthError: if a ticket cannot be obtained at all. Deliberately
                not wrapped — a login problem is not a request problem, and
                hiding the distinction sends you debugging the wrong layer.
        """
        url = f"{self._base_url}{path}"
        refreshed = False
        last_error = "unknown error"

        for attempt in range(1, self._max_attempts + 1):
            self._wait_for_rate_limit()

            try:
                response = self._client.post(
                    url,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "TGT": self._auth.ticket(),
                    },
                )
            except httpx.HTTPError as exc:
                # A dropped connection or timeout says nothing about whether the
                # request was valid, so it is worth another attempt.
                last_error = f"network error: {exc}"
                if attempt == self._max_attempts:
                    break
                self._sleep(self._jitter(self._backoff_for(attempt)))
                continue

            if response.status_code == 401 and not refreshed:
                # Our ticket may have been dropped server-side while our own
                # clock still considered it fresh. Worth exactly one refresh,
                # and with no backoff: nothing is overloaded here.
                self._auth.invalidate()
                refreshed = True
                continue

            if response.status_code in TRANSIENT_STATUS:
                last_error = f"HTTP {response.status_code}"
                if response.status_code == 429:
                    self._slow_down()
                if attempt == self._max_attempts:
                    break
                self._sleep(self._retry_delay(response, attempt))
                continue

            if response.status_code >= 400:
                raise EpiasRequestError(
                    f"EPİAŞ rejected {path} with HTTP {response.status_code}: {response.text[:300]}"
                )

            return self._decode(response, path)

        raise EpiasRequestError(
            f"EPİAŞ request to {path} failed after {self._max_attempts} attempts ({last_error})"
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> EpiasClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @property
    def interval(self) -> float:
        """Current spacing between requests. Grows after throttling."""
        return self._min_interval

    def _slow_down(self) -> None:
        """Widen the gap between all future requests, not just this retry.

        The quota is shared, so once it is exhausted the next request meets the
        same wall. Retrying one request harder does not help; the loop has to
        proceed more slowly from here on. The interval is not lowered again —
        recovering it would just walk back into the limit.
        """
        self._min_interval = min(self._min_interval * RATE_LIMIT_GROWTH, MAX_REQUEST_INTERVAL)

    def _wait_for_rate_limit(self) -> None:
        """Keep a minimum gap between requests.

        Staying under the limit by construction is cheaper than being throttled
        and backing off: a 429 costs a full round trip plus the server's chosen
        penalty, while spacing requests costs only the gap.
        """
        if self._last_request_at is not None:
            remaining = self._min_interval - (self._monotonic() - self._last_request_at)
            if remaining > 0:
                self._sleep(remaining)
        self._last_request_at = self._monotonic()

    def _backoff_for(self, attempt: int) -> float:
        return self._backoff_base * (2 ** (attempt - 1))

    def _retry_delay(self, response: httpx.Response, attempt: int) -> float:
        """Prefer the server's own instruction over our guess."""
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                # Retry-After may also be an HTTP date; the seconds form is what
                # EPİAŞ sends, and a bad parse falls through to the backoff.
                return float(retry_after)
            except ValueError:
                pass
        return self._jitter(self._backoff_for(attempt))

    def _decode(self, response: httpx.Response, path: str) -> dict[str, Any]:
        try:
            body = response.json()
        except ValueError as exc:
            # A maintenance page returned with HTTP 200 is not hypothetical.
            raise EpiasRequestError(
                f"expected JSON from {path} but got {response.text[:200]!r}"
            ) from exc

        if not isinstance(body, dict):
            raise EpiasRequestError(
                f"expected a JSON object from {path}, got {type(body).__name__}"
            )
        return body
