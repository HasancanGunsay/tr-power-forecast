"""Tests for EPİAŞ ticket handling.

None of these tests touch the network. The TGT lifetime is two hours and the
idle timeout is 45 minutes, so testing expiry against a real clock would mean a
two-hour test run. Instead the clock is injected, which lets us step time
forward instantly and assert on exactly the boundary we care about.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest
import respx

from powerforecast.data.epias import (
    TGT_URL,
    EpiasAuth,
    EpiasAuthError,
)


class FakeClock:
    """A clock the test drives by hand."""

    def __init__(self, start: datetime | None = None) -> None:
        self.now = start or datetime(2026, 8, 5, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, delta: timedelta) -> None:
        self.now += delta


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def auth(clock: FakeClock) -> EpiasAuth:
    return EpiasAuth(username="user", password="pass", clock=clock)


@respx.mock
def test_fetches_a_ticket_on_first_use(auth: EpiasAuth) -> None:
    route = respx.post(TGT_URL).mock(return_value=httpx.Response(201, text="TGT-1-abcdef-cas"))

    assert auth.ticket() == "TGT-1-abcdef-cas"
    assert route.call_count == 1


@respx.mock
def test_credentials_are_form_encoded(auth: EpiasAuth) -> None:
    route = respx.post(TGT_URL).mock(return_value=httpx.Response(201, text="TGT-x"))

    auth.ticket()

    request = route.calls[0].request
    assert request.headers["content-type"] == "application/x-www-form-urlencoded"
    # Special characters must survive encoding, or a valid password fails to log in.
    assert b"username=user" in request.content
    assert b"password=pass" in request.content


@respx.mock
def test_reuses_the_ticket_while_it_is_still_valid(auth: EpiasAuth, clock: FakeClock) -> None:
    route = respx.post(TGT_URL).mock(return_value=httpx.Response(201, text="TGT-x"))

    auth.ticket()
    clock.advance(timedelta(minutes=30))
    auth.ticket()

    # A second network round trip here would mean we are re-authenticating on
    # every call, which is both slow and a good way to get rate limited.
    assert route.call_count == 1


@respx.mock
def test_refreshes_after_the_idle_timeout(auth: EpiasAuth, clock: FakeClock) -> None:
    route = respx.post(TGT_URL).mock(
        side_effect=[
            httpx.Response(201, text="TGT-first"),
            httpx.Response(201, text="TGT-second"),
        ]
    )

    assert auth.ticket() == "TGT-first"
    clock.advance(timedelta(minutes=46))  # unused for longer than 45 minutes
    assert auth.ticket() == "TGT-second"
    assert route.call_count == 2


@respx.mock
def test_idle_timer_resets_on_each_use(auth: EpiasAuth, clock: FakeClock) -> None:
    route = respx.post(TGT_URL).mock(return_value=httpx.Response(201, text="TGT-x"))

    auth.ticket()
    # Three 30-minute gaps: never idle for 45 minutes, so the ticket survives
    # even though 90 minutes have passed in total.
    for _ in range(3):
        clock.advance(timedelta(minutes=30))
        auth.ticket()

    assert route.call_count == 1


@respx.mock
def test_refreshes_after_the_absolute_lifetime(auth: EpiasAuth, clock: FakeClock) -> None:
    route = respx.post(TGT_URL).mock(
        side_effect=[
            httpx.Response(201, text="TGT-first"),
            httpx.Response(201, text="TGT-second"),
        ]
    )

    auth.ticket()
    # Kept warm by regular use, but the two-hour absolute cap still applies.
    for _ in range(5):
        clock.advance(timedelta(minutes=25))
        auth.ticket()

    assert route.call_count == 2


@respx.mock
def test_rejects_a_response_that_is_not_a_ticket(auth: EpiasAuth) -> None:
    # A login page or an HTML error would otherwise be stored as the "ticket"
    # and produce a confusing 401 on the next call instead of here.
    respx.post(TGT_URL).mock(return_value=httpx.Response(200, text="<html>login</html>"))

    with pytest.raises(EpiasAuthError, match="unexpected"):
        auth.ticket()


@respx.mock
def test_bad_credentials_raise_immediately(auth: EpiasAuth) -> None:
    respx.post(TGT_URL).mock(return_value=httpx.Response(401, text="Unauthorized"))

    with pytest.raises(EpiasAuthError, match="credentials"):
        auth.ticket()


@respx.mock
def test_invalidate_forces_a_new_ticket(auth: EpiasAuth) -> None:
    route = respx.post(TGT_URL).mock(
        side_effect=[
            httpx.Response(201, text="TGT-first"),
            httpx.Response(201, text="TGT-second"),
        ]
    )

    assert auth.ticket() == "TGT-first"
    # The server may drop a ticket before our clock says it expired; the client
    # needs a way to give up on one it still believes in.
    auth.invalidate()
    assert auth.ticket() == "TGT-second"
    assert route.call_count == 2


@respx.mock
def test_password_is_not_exposed_in_repr(auth: EpiasAuth) -> None:
    assert "pass" not in repr(auth)
