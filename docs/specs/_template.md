---
id: M0-0
title: Short name
status: draft
approved:
---

# M0-0: Short name

## Goal

Two or three sentences: what exists after this checkpoint that didn't before, and why it matters
for laterank.

## Out of scope

- Things someone might expect here that this checkpoint will **not** do, and where they go
  instead.

## Interfaces

The agreement that other code relies on: exact function signatures with shapes and dtypes, file
layouts, JSON fields, error types. Code blocks are fine.

```python
def example(x: np.ndarray, k: int) -> np.ndarray:
    """x: (n, d) float32 -> (n, k) int64 indices."""
```

## Decisions

| decision | why | differs from design doc? |
|---|---|---|
| | | no |

## Acceptance tests

Each line is one check that can be verified. Tick it (`- [x]`) only when the test exists and passes.

- [ ] AT1: ...

## Tasks

| # | task | files it owns | depends on | parallel? |
|---|---|---|---|---|
| 1 | | | | |

## Changes

Dated lines for anything changed after approval: `- 2026-10-06: <what> - <why>`.
