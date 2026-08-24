# Architecture Decision Records

Short records of decisions that were not obvious, kept next to the code they
explain. Each one states the problem, the choice, the alternatives that were
considered, and what the choice costs.

The point is not documentation for its own sake. Code shows *what* was built; it
rarely shows *what else was on the table and why it lost*. Six months later that
is the expensive thing to reconstruct — and reviewers, including future
maintainers, ask about it first.

Records are immutable once merged. A decision that turns out to be wrong is not
edited; a new record supersedes it, and the old one is marked as superseded. The
history of a mistake is part of what makes the log useful.

## Index

| # | Decision | Status |
|---|---|---|
| [0001](0001-epias-ticket-authentication.md) | Ticket handling for EPİAŞ authentication | Accepted |
| [0002](0002-retry-and-rate-limiting.md) | Retry policy and rate limiting | Accepted |
| [0003](0003-raw-storage-layout.md) | Raw storage layout and merge semantics | Accepted |
| [0004](0004-forecast-origin.md) | Forecast origin and the information set | Accepted |
| [0005](0005-weather-and-fair-comparison.md) | Archived weather forecasts, and correcting our own bias | Accepted |
| [0006](0006-model-store.md) | A model store, not a pickle | Accepted |
| [0007](0007-forecast-store-and-drift.md) | Recording forecasts, and how drift is judged | Accepted |
| [0008](0008-deployable-bias-correction.md) | A bias correction that can be deployed, and what it is worth | Superseded by [0009](0009-no-monotonic-trend-feature.md) |
| [0009](0009-no-monotonic-trend-feature.md) | Removing the linear trend, and what it had been hiding | Accepted |

## Template

```markdown
# NNNN — Title

**Status:** Proposed / Accepted / Superseded by [NNNN](...)
**Date:** YYYY-MM-DD

## Context
What forced a decision? Constraints, and what happens if nothing is done.

## Decision
What was chosen, stated plainly.

## Alternatives considered
What else was on the table, and the specific reason each one lost.

## Consequences
What this costs, what it rules out, and what would make us revisit it.
```
