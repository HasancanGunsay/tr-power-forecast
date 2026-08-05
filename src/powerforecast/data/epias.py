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

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import httpx

TGT_URL = "https://giris.epias.com.tr/cas/v1/tickets"

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
