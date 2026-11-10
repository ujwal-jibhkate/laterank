---
id: M5-1
title: 0.1.0b2 release
status: done
approved: 2026-11-10
---

# M5-1: 0.1.0b2 release

## Goal

PyPI's `laterank` page stops linking to the missing `START_HERE.md`, and the declared dependency
floors are versions that actually import and pass the tests. Both need a new release, because an
uploaded version can never be changed.

## Out of scope

- The README link itself: already fixed on `main` (it points to `CONTRIBUTING.md`). This release
  only publishes it.
- Skipped: Trusted Publishing workflow, add at 0.1.0 (M4-2's decision stands).
- Skipped: testing the oldest `PyStemmer`/`tqdm`, add when a user reports one. They have no floor,
  and their oldest releases predate Python 3.12.
- Not changed: M4-1's remaining low reviewer findings.

## Interfaces

No change to `src/` or `tests/`.

`pyproject.toml`:

```toml
dependencies = [
    "numpy>=2.2",
    "scipy>=1.16",   # was >=1.15: 1.15's compiled _spropack fails to load on macOS 27
    "PyStemmer",
    "tqdm",
]
```

`uv.lock` regenerated with `uv lock`.

`.github/workflows/ci.yml` gains one job that runs the tests on the floor versions:

```yaml
  test-floor:
    name: test floor (${{ matrix.os }})
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: astral-sh/setup-uv@v6
        with:
          python-version: "3.12"
      - run: uv sync --locked
      # The oldest numpy and scipy that pyproject.toml allows.
      # Exact pins: "1.16.*" could resolve to a later 1.16.x, not the floor itself.
      - run: uv run --with "numpy==2.2.0" --with "scipy==1.16.0" python -c "import numpy, scipy; assert (numpy.__version__, scipy.__version__) == ('2.2.0', '1.16.0'), (numpy.__version__, scipy.__version__)"
      - run: uv run --with "numpy==2.2.0" --with "scipy==1.16.0" pytest -ra
```

`CONTRIBUTING.md`: one line saying a floor change must change this job too.

Release steps: the same as M4-2, with `v0.1.0b2` and `dist/laterank-0.1.0b2*`. The owner tags,
uploads in their own terminal with the project-scoped token, and pushes the tag after the upload.

## Decisions

| decision | why | differs from design doc? |
|---|---|---|
| Raise the floor to `scipy>=1.16` | measured on this Mac (macOS 27.0.1, arm64): 1.15 fails at `import scipy.sparse` (loader rejects `_spropack`'s `__thread_bss` section), 1.16.0 imports. A floor only matters to someone who already has the old version, and for them 1.15 crashes | no |
| Test the floors in CI on 3 OS, py3.12 only | the floor was never tested, which is how 1.15 got in. CI's macOS runners are older than 27, so this job would not have caught this exact bug, but it catches API breaks | no |
| `uv run --with` instead of `--resolution lowest-direct` | lowest-direct also drops `PyStemmer`/`tqdm` to their oldest releases, which don't support 3.12 | no |
| No Linux check of 1.15 | Docker was not running; the raise is justified by macOS 27 alone | no |

## Acceptance tests

- [x] AT1: `pyproject.toml` says `scipy>=1.16` and `uv lock --check` passes.
- [x] AT2: `uv run pytest -q`, `uv run ruff check .`, `uv run ruff format --check .` pass, and CI is
  green on all jobs, including the 3 `test-floor` jobs, for the commit to be tagged.
- [x] AT3: from the tag, `dist/` holds exactly the `0.1.0b2` sdist and wheel, `twine check` passes,
  and the wheel's metadata says `scipy>=1.16`.
- [x] AT4: after the upload, in a new venv outside the repo, `pip install "laterank[colbert]"`
  installs `0.1.0b2`, the quick-start ranks doc `a` first, and PyPI's JSON description has no
  `START_HERE` text.
- [x] AT5: the `reviewer` agent returns `merge` (before the tag).

## Tasks

| # | task | files it owns | depends on | parallel? |
|---|---|---|---|---|
| 1 | floor, lock, CI job, CONTRIBUTING line (AT1) | `pyproject.toml`, `uv.lock`, `.github/workflows/ci.yml`, `CONTRIBUTING.md` | none | no |
| 2 | reviewer (AT5), commit on the owner's OK; owner pushes, CI (AT2) | none | 1 | no |
| 3 | owner tags; build and check (AT3) | none | 2 | no |
| 4 | owner uploads, pushes the tag; verify from PyPI (AT4) | none | 3 | no |

## Changes

- 2026-11-10: the CI comment's reason for exact pins corrected - reviewer finding 1: `uv.lock`
  holds scipy 1.18.1, not a 1.16.x.
- 2026-11-10: `test-floor` first asserts that numpy 2.2.0 and scipy 1.16.0 are the versions loaded
  (owner OK) - reviewer finding 2: if `uv run --with` ever fell back to the locked versions, the job
  would pass while testing the newest ones.
