# 0006 — A model store, not a pickle

**Status:** Accepted
**Date:** 2026-08-07

## Context

Everything up to this point has been a backtest. A backtest never loads a model: it
fits one per fold, scores it, and throws it away. Serving inverts that — the model is
fitted once and then used, possibly for months, by a process that has no memory of how
it was made.

That inversion creates failure modes the backtest cannot have, and both of them are
silent:

* **Column drift.** `build_design_matrix` is under active development; the holiday
  features were added days ago and more will follow. A tree model handed the wrong set
  of columns does not complain. It predicts. The service returns numbers, monitoring
  shows no errors, and the accuracy degrades in a way nobody attributes to the cause.
* **Environment drift.** A pickle fitted under one scikit-learn or LightGBM and loaded
  under another is not guaranteed to behave identically. Nothing in the format records
  which versions were present at fit time, so the mismatch cannot even be detected.

There is also a slower problem. Six weeks from now the question about a file called
`model.pkl` is never "is this a LightGBM?" — it is "was this trained before or after the
holiday calendar went in, and on which window?" A file that cannot answer that is a file
nobody trusts enough to keep or dares to delete.

## Decision

A save writes a **version directory**, never a bare file:

```
models/<name>/<version>/model.pkl    the fitted estimator
models/<name>/<version>/card.json    what it is
```

Four properties, each chosen against a specific failure:

1. **The card is JSON, not pickle.** It records the feature columns *in order*, the
   `FeatureSpec`, the training window and row count, the estimator class and its
   hyperparameters, the seed, and the versions of pandas, NumPy, scikit-learn and
   LightGBM. Readable with `cat`, from a container, from a shell script — anything that
   needs this project's code to be legible is barely better than nothing.
2. **Versions are immutable.** `<version>` is a UTC timestamp, and saving over an
   existing one raises. History that can be rewritten is not history.
3. **The checksum is taken from disk after writing.** Hashing the in-memory bytes would
   certify a file that was never written, which is precisely the case the checksum
   exists to catch. A load verifies it before unpickling.
4. **`predict` cannot skip the feature check, and there is no flag to bypass it.**
   Column mismatch — especially reordering, where the shapes match and nothing else
   objects — is the failure this module exists to prevent. An override would be used
   exactly once, in a hurry, in production.

The environment check is the one thing that is opt-in. `load_model(...,
require_environment=True)` refuses a version mismatch; the default reports it and
proceeds. Exploring an old model in a notebook and answering a request are different
risks, and a check that fires on the harmless case is a check people learn to disable.

## Alternatives considered

**joblib instead of pickle.** The scikit-learn convention, and faster on large NumPy
arrays. Rejected for now because it is a transitive dependency we do not declare, and it
buys nothing at this model size. If models grow, this is a one-line change.

**ONNX or another portable format.** Solves the pickle trust problem properly and
removes the Python dependency at serving time. Rejected as premature: it costs a
conversion step and a class of silent numerical differences, to buy portability nothing
in this project currently needs. Revisit if the model ever has to be served outside
Python.

**MLflow or a tracking server.** Does all of this and much more. Rejected because
"much more" is the problem — a service to run, a schema to learn, and a dependency that
would dwarf the thing it manages. Two hundred lines that are fully understood are worth
more here than a platform that is not.

**Hash the design matrix into the version.** Would make the version identify the data
exactly. Rejected: it makes versions unsortable and unreadable, and the training window
plus row count already answer the question a human is actually asking.

## Consequences

* Every model has to be saved through `save_model` with the design matrix it was fitted
  on. Passing anything else makes the card a lie, which is worse than no card, because a
  card is trusted.
* The store is append-only, so it grows. Pruning is a deliberate act, not a side effect
  of a training run — which is the correct default when the file is small and the
  question it answers is expensive to reconstruct.
* **Pickle limitation, stated plainly:** unpickling executes code, so this format is
  safe only for files this project wrote. The checksum protects against corruption, not
  against a hostile file. If a model ever crosses a trust boundary, the format must
  change.
* The serving layer now has a fixed contract to build against: load a version, check the
  card, refuse anything that does not match it.
