# 0003 — Raw storage layout and merge semantics

**Status:** Accepted
**Date:** 2026-08-05

## Context

Raw series are stored as one parquet file per month under
`data/raw/<series>/YYYY-MM.parquet`. Monthly partitioning keeps a re-fetch of one
month from rewriting six years of history.

Two facts interact badly here:

* Requests are made in **Istanbul local time**, because that is what the platform
  accepts, and a request is naturally scoped to a calendar month.
* Storage partitions on **UTC**, because European bidding zones observe daylight
  saving and a local-time index would contain a duplicated hour and a missing one
  twice a year.

Istanbul is UTC+03:00, so a request for local January returns rows that belong to
both the UTC December and UTC January files, and the February request returns
rows belonging to UTC January as well.

The first implementation replaced a month file outright on write. That is correct
when an entire series is written in one call, and it was — until the backfill was
made resumable and started writing one month at a time. Each write then truncated
its neighbour's file to the three overlapping hours. The failure was silent: every
month reported the right number of rows as it was fetched, and only the
post-write validation revealed that 48,711 of 49,008 hours were gone.

## Decision

**Merge on write.** An existing month file is read, concatenated with the
incoming rows, de-duplicated on the timestamp index keeping the **last**
occurrence, sorted, and written back.

Keeping the last occurrence matters beyond conflict resolution: EPİAŞ revises
published figures, so a re-fetch must overwrite a stored value rather than
preserve a stale one.

## Alternatives considered

**Align partitions with Istanbul local months.** Removes the overlap entirely,
since requests and files would use the same calendar. Rejected: it re-introduces
local time into storage, which is the thing UTC storage exists to avoid, and it
would not survive the European series where local months contain 23- and 25-hour
days.

**Write the whole series in one call, as before.** The original behaviour was
correct. Rejected: it makes a backfill all-or-nothing, and a single 429 midway
through six years of history discards everything fetched so far — which is
exactly what happened before the run was made resumable.

**Append without de-duplication.** Simplest possible merge. Rejected: re-running
a backfill would then duplicate rows, and duplicated hours corrupt every lag
feature built downstream. Idempotence is worth more than the saved comparison.

**One file per series.** No boundary problem at all. Rejected: refreshing recent
data would rewrite the entire history on every run, and the file grows without
bound.

## Consequences

* Writing a month now costs a read as well. Irrelevant at this scale — the
  largest monthly file holds 744 rows.
* A stored value can only be corrected by re-fetching it. There is deliberately
  no way to delete rows through `write_raw`; removing data is a manual act.
* The bug that motivated this record was found only because validation runs
  against **what is on disk** rather than against what was just fetched in
  memory. That distinction is load-bearing and should not be optimised away.
