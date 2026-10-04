---
id: M0-1
title: Skeleton
status: done
approved: 2026-10-03
---

# M0-1: Skeleton

Written after the fact on 2026-10-03, when the spec workflow was introduced. The owner's M0
scaffold prompt served as the approved plan.

## Goal

An installable, empty `laterank` package with the five-layer layout, packaging, lint and CI, so
that every later checkpoint only adds code.

## Out of scope

- Any retrieval code (M0-2 onward).
- Encoder dependencies inside the extras (M1).

## Interfaces

- `laterank.__version__`, derived from git tags by hatch-vcs (`0.0.0` before the first tag).
- Subpackages `laterank.kernels` and `laterank.storage`.
- Extras with fixed names: `bge-m3`, `colbert`, `rerank`, `pandas`, `all`.

## Decisions

| decision | why | differs from design doc? |
|---|---|---|
| uv + hatchling + hatch-vcs | matches smve-lab tooling; version from tags | no |
| extras declared empty | names are public API and should never change | no |

## Acceptance tests

- [x] AT1: `uv sync` installs the package and `import laterank` works.
- [x] AT2: `tests/test_layers.py` passes.
- [x] AT3: the CI workflow runs ruff and pytest on 3 OSes × Python 3.12 and 3.13.

## Tasks

| # | task | files it owns | depends on | parallel? |
|---|---|---|---|---|
| 1 | packaging, layout, CI | `pyproject.toml`, `src/`, `.github/` | - | no |

## Changes

- 2026-10-03: committed as `b2db6e5`.
