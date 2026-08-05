# 0002 — Retry policy and rate limiting

**Status:** Accepted
**Date:** 2026-08-05

## Context

Ingestion runs unattended against a shared public API. Two failure modes matter and
they pull in opposite directions:

* Give up too easily and a transient hiccup — a throttled request, a brief 503 —
  leaves a hole in the time series. Gaps in an hourly series are expensive: they
  corrupt lag features and silently bias a backtest.
* Retry too eagerly and the client becomes the problem. Retrying a malformed request
  cannot succeed, consumes rate-limit budget, and retrying rejected credentials in a
  loop is a reliable way to get an account locked.

The platform also throttles, so a client that simply issues requests as fast as it can
will be rate limited regardless of how carefully it handles the response.

## Decision

**Classify failures by whether the identical request could succeed later.**

* **Transient** — `429`, `500`, `502`, `503`, `504`, and network-level errors
  (timeouts, dropped connections). Retried up to four attempts total with exponential
  backoff and **full jitter**.
* **Permanent** — every other `4xx`. Raised immediately, no retry.
* **`401`** — treated as neither. The first one triggers exactly one ticket refresh
  with **no backoff**, because a dropped ticket means nothing is overloaded. A second
  `401` is fatal.

**Prefer the server's instruction to our guess.** When a `429` carries `Retry-After`,
that value is used instead of the computed backoff.

**Stay under the limit by construction.** A minimum interval is enforced between
requests, measured with a **monotonic** clock — a wall clock can jump backwards on an
NTP correction and produce a negative elapsed time.

**Inject `sleep` and the clock.** Tests assert on the delays that were requested,
without waiting for them.

## Alternatives considered

**Retry everything, including 4xx.** Simpler classification: on failure, try again.
Rejected — it converts a fast, clear error into a slow, identical one, and the account
lockout risk on repeated `401` is real.

**Retry nothing; let the scheduler re-run the job.** Also simple, and appealing for
idempotent batch work. Rejected: a single throttled request would fail a whole
multi-hour backfill, and the retry would restart from the beginning.

**Fixed-delay retries.** Easier to reason about than exponential backoff. Rejected:
a fixed interval keeps a struggling server struggling, and synchronised clients
produce a thundering herd on recovery. Exponential backoff plus jitter addresses both.

**No jitter.** One fewer moving part, and deterministic without a test seam. Rejected:
clients that fail together retry together, so the recovering server is hit by the same
spike that knocked it over.

**A retry library (tenacity, backoff).** Battle-tested and less code to own. Rejected
for this layer: the policy is small, and the interesting part — the three-way split
between transient, permanent and `401` — is domain logic that would end up expressed
as library configuration and be harder to read, not easier.

## Consequences

* Worst case for a permanently failing endpoint is four attempts spread over roughly
  seven seconds before the error surfaces.
* `MIN_REQUEST_INTERVAL` is a guess (0.2 s) rather than a documented figure. If EPİAŞ
  publishes a real limit, it should replace this constant.
* Retry state is per-call, so a long backfill loop does not accumulate backoff across
  requests. If sustained throttling becomes common, an adaptive limiter that slows the
  whole loop after repeated `429`s would be the next step.
* The `401` path assumes a refreshed ticket fixes the problem. If EPİAŞ ever returns
  `401` for authorisation rather than authentication — a valid ticket lacking access
  to an endpoint — this would waste one extra request before failing.
