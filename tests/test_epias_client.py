"""Tests for the EPİAŞ request layer.

The interesting behaviour here is not "does a request work" — it is what happens
when it does not. Retrying the wrong error wastes time and can lock an account;
failing on the right one throws away data that a second attempt would have got.

Both the clock and `sleep` are injected, so a test that exercises a 30-second
backoff finishes instantly and asserts on the exact delays that were requested.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from powerforecast.data.epias import (
    EpiasAuth,
    EpiasClient,
    EpiasRequestError,
)

BASE_URL = "https://example.test/electricity-service"
PATH = "/v1/consumption/data/realtime-consumption"
URL = BASE_URL + PATH
PAYLOAD = {"startDate": "2026-08-01T00:00:00+03:00", "endDate": "2026-08-02T00:00:00+03:00"}


class FakeTime:
    """A monotonic clock whose only way of advancing is `sleep`."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


class StubAuth:
    """Stands in for EpiasAuth so client tests do not exercise login too."""

    def __init__(self) -> None:
        self.tickets = ["TGT-first", "TGT-second"]
        self.invalidate_calls = 0

    def ticket(self) -> str:
        return self.tickets[min(self.invalidate_calls, len(self.tickets) - 1)]

    def invalidate(self) -> None:
        self.invalidate_calls += 1


@pytest.fixture
def fake_time() -> FakeTime:
    return FakeTime()


@pytest.fixture
def auth() -> StubAuth:
    return StubAuth()


@pytest.fixture
def client(auth: StubAuth, fake_time: FakeTime) -> EpiasClient:
    return EpiasClient(
        auth,  # type: ignore[arg-type]
        base_url=BASE_URL,
        min_interval=0.0,
        max_attempts=4,
        backoff_base=1.0,
        monotonic=fake_time.monotonic,
        sleep=fake_time.sleep,
        jitter=lambda delay: delay,  # deterministic in tests
    )


@respx.mock
def test_sends_the_ticket_and_returns_parsed_json(client: EpiasClient) -> None:
    route = respx.post(URL).mock(return_value=httpx.Response(200, json={"items": [1, 2]}))

    assert client.post(PATH, PAYLOAD) == {"items": [1, 2]}

    request = route.calls[0].request
    assert request.headers["TGT"] == "TGT-first"
    assert request.headers["content-type"] == "application/json"


@respx.mock
def test_retries_on_429_then_succeeds(client: EpiasClient, fake_time: FakeTime) -> None:
    route = respx.post(URL).mock(
        side_effect=[
            httpx.Response(429),
            httpx.Response(200, json={"items": []}),
        ]
    )

    assert client.post(PATH, PAYLOAD) == {"items": []}
    assert route.call_count == 2
    assert fake_time.slept == [1.0]


@respx.mock
def test_honours_retry_after(client: EpiasClient, fake_time: FakeTime) -> None:
    respx.post(URL).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "7"}),
            httpx.Response(200, json={}),
        ]
    )

    client.post(PATH, PAYLOAD)

    # The server knows when it will accept traffic again; guessing is worse.
    assert fake_time.slept == [7.0]


@respx.mock
def test_backoff_grows_exponentially(client: EpiasClient, fake_time: FakeTime) -> None:
    respx.post(URL).mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(503),
            httpx.Response(200, json={}),
        ]
    )

    client.post(PATH, PAYLOAD)

    # Hammering a struggling server at a fixed interval keeps it struggling.
    assert fake_time.slept == [1.0, 2.0]


@respx.mock
def test_gives_up_after_max_attempts(client: EpiasClient) -> None:
    route = respx.post(URL).mock(return_value=httpx.Response(503))

    with pytest.raises(EpiasRequestError, match="4 attempts"):
        client.post(PATH, PAYLOAD)

    assert route.call_count == 4


@respx.mock
def test_client_errors_are_not_retried(client: EpiasClient) -> None:
    # A malformed request will be malformed on the second attempt too. Retrying
    # only delays the error and burns rate-limit budget.
    route = respx.post(URL).mock(return_value=httpx.Response(400, text="bad range"))

    with pytest.raises(EpiasRequestError, match="400"):
        client.post(PATH, PAYLOAD)

    assert route.call_count == 1


@respx.mock
def test_401_refreshes_the_ticket_once(
    client: EpiasClient, auth: StubAuth, fake_time: FakeTime
) -> None:
    route = respx.post(URL).mock(
        side_effect=[
            httpx.Response(401),
            httpx.Response(200, json={"items": []}),
        ]
    )

    client.post(PATH, PAYLOAD)

    # The server may drop a ticket we still consider valid, so one refresh is
    # worth attempting — but without a backoff, since nothing is overloaded.
    assert auth.invalidate_calls == 1
    assert route.calls[1].request.headers["TGT"] == "TGT-second"
    assert fake_time.slept == []


@respx.mock
def test_repeated_401_is_fatal(client: EpiasClient, auth: StubAuth) -> None:
    route = respx.post(URL).mock(return_value=httpx.Response(401))

    with pytest.raises(EpiasRequestError, match="401"):
        client.post(PATH, PAYLOAD)

    # Retrying rejected credentials in a loop is how accounts get locked.
    assert auth.invalidate_calls == 1
    assert route.call_count == 2


@respx.mock
def test_network_errors_are_retried(client: EpiasClient) -> None:
    route = respx.post(URL).mock(
        side_effect=[
            httpx.ConnectTimeout("timed out"),
            httpx.Response(200, json={"ok": True}),
        ]
    )

    assert client.post(PATH, PAYLOAD) == {"ok": True}
    assert route.call_count == 2


@respx.mock
def test_rate_limit_spaces_out_requests(auth: StubAuth, fake_time: FakeTime) -> None:
    client = EpiasClient(
        auth,  # type: ignore[arg-type]
        base_url=BASE_URL,
        min_interval=0.5,
        monotonic=fake_time.monotonic,
        sleep=fake_time.sleep,
    )
    respx.post(URL).mock(return_value=httpx.Response(200, json={}))

    client.post(PATH, PAYLOAD)
    client.post(PATH, PAYLOAD)

    # First call is free; the second waits out the remaining interval.
    assert fake_time.slept == [0.5]


@respx.mock
def test_non_json_body_is_reported_clearly(client: EpiasClient) -> None:
    respx.post(URL).mock(return_value=httpx.Response(200, text="<html>maintenance</html>"))

    with pytest.raises(EpiasRequestError, match="JSON"):
        client.post(PATH, PAYLOAD)


def test_client_can_be_used_as_a_context_manager(auth: StubAuth) -> None:
    with EpiasClient(auth, base_url=BASE_URL) as client:  # type: ignore[arg-type]
        assert isinstance(client, EpiasClient)


@respx.mock
def test_auth_failures_propagate_unchanged() -> None:
    # A login problem is not a request problem; wrapping it would hide the cause.
    from powerforecast.data.epias import EpiasAuthError

    respx.post("https://login.test/tickets").mock(return_value=httpx.Response(401))
    auth = EpiasAuth(username="u", password="p", tgt_url="https://login.test/tickets")
    client = EpiasClient(auth, base_url=BASE_URL, min_interval=0.0)

    with pytest.raises(EpiasAuthError):
        client.post(PATH, PAYLOAD)
