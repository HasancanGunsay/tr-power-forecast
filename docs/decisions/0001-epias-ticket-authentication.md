# 0001 — Ticket handling for EPİAŞ authentication

**Status:** Accepted
**Date:** 2026-08-05

## Context

The EPİAŞ Şeffaflık Platform does not issue a static API key. Credentials are
exchanged for a ticket-granting ticket (TGT) at
`POST https://giris.epias.com.tr/cas/v1/tickets`, and that ticket is then sent in the
`TGT` header of every subsequent request.

Two expiry rules apply simultaneously:

* an **absolute lifetime** of two hours from issue, and
* an **idle timeout** of 45 minutes that resets on each use.

A client that tracks only the idle timeout survives development — where calls are
frequent — and dies in the third hour of the first unattended run. A client that
re-authenticates on every call works correctly but wastes a round trip per request
and invites rate limiting.

The ingestion pipeline is intended to run unattended on a schedule, so ticket
handling has to be correct rather than merely adequate for interactive use.

## Decision

A dedicated `EpiasAuth` object owns the ticket and its lifecycle.

* It caches the ticket and tracks both `issued_at` and `last_used_at`, refreshing
  when **either** rule is violated.
* It refreshes two minutes early. Renewing exactly at the boundary races against
  clock skew and against a request that was valid when it left and expired before it
  arrived.
* **The clock is injected** as a constructor argument rather than read from the
  `datetime` module.
* A response body that does not start with `TGT-` raises immediately instead of being
  cached as a ticket.
* `invalidate()` lets a caller discard a ticket the server has already dropped.
* The password never appears in `__repr__`, and settings hold it as a `SecretStr`.

## Alternatives considered

**Re-authenticate on every request.** Simplest possible implementation and immune to
expiry bugs. Rejected: it doubles the request count against a rate-limited API, and
the login endpoint is the one most likely to throttle an abusive client.

**Track only the idle timeout.** Covers the common case and is simpler. Rejected: it
fails precisely in the unattended long-running scenario this pipeline is built for,
and it fails silently until the third hour.

**Catch the 401 and retry.** Treat expiry as an error to recover from rather than a
state to track. Rejected as the *primary* mechanism — it turns every expiry into a
wasted request plus a retry, and it conflates "ticket expired" with "credentials are
wrong", which must not be retried. It remains useful as a secondary safety net, which
is what `invalidate()` exists for.

**Read the clock directly.** Conventional and shorter. Rejected: expiry is the entire
purpose of this class, and a two-hour lifetime cannot be exercised against a real
clock inside a test suite that must finish in seconds. The behaviour most likely to
break would have been the behaviour least likely to be tested.

## Consequences

* Expiry logic is fully covered by fast unit tests, including the case where regular
  use defeats the idle timeout but the absolute lifetime still applies.
* Callers must construct `EpiasAuth` rather than calling a module-level function, and
  tests must pass a clock. This is a small, deliberate cost.
* The two-minute margin and both expiry constants are hardcoded from the platform
  documentation. If EPİAŞ changes them, the tests will still pass while production
  fails — so these values should be re-checked whenever authentication starts
  behaving oddly.
* Rate limiting and retry policy are explicitly **not** handled here. They belong to
  the request layer and will be recorded separately.
