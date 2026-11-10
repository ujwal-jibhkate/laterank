# Contributing to laterank

Thanks for helping. Bug reports, fixes, docs and measurements on new datasets are all welcome.

## Before you start

- **Small fix** (a bug, a typo, a clearer error message): open a pull request directly.
- **Anything bigger** (a new first stage, encoder or option): open an issue first. laterank is
  built one checkpoint at a time, and each checkpoint has a one-page spec in `docs/specs/`
  (rules in `docs/specs/README.md`). The maintainer writes or updates the spec with you, so you
  don't build something that won't be merged.

## Set up

You need Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/ujwal-jibhkate/laterank.git && cd laterank
uv sync                        # core + dev dependencies
uv run pytest -q               # all tests
uv run ruff check . && uv run ruff format --check .   # lint
```

`uv sync --extra colbert` or `--extra bge-m3` adds an encoder (and torch) when your change needs one.

Tests under `tests/golden/` re-check reference numbers from the research repo and skip themselves
unless `LATERANK_GOLDEN_DIR` points at its embeddings (about 14 GB, not public). CI skips them too;
the maintainer runs them before merging anything that could change results.

## Rules the code follows

- Five layers, imports point down only: `api → orchestration → components → storage → kernels`.
- `kernels/` and `storage/` use numpy and scipy only, never torch. Torch and model libraries are
  imported only inside an encoder's `__init__`.
- `.npy` files are always loaded with `allow_pickle=False`.
- Manifest writes are atomic (temp file, fsync, `os.replace`).
- Public functions validate their inputs and raise a clear `ValueError` or `TypeError`.
- New behaviour comes with a test. Kernels keep their naive-loop reference tests.
- The least code that works: prefer the standard library, numpy and scipy over a new dependency.
- Raising the `numpy` or `scipy` floor in `pyproject.toml` means changing the pins in CI's
  `test-floor` job too, so CI keeps testing the oldest versions we allow.

## Pull requests

- Keep one change per pull request, with a message that says *why*.
- CI runs lint and the tests on Linux, macOS and Windows with Python 3.12 and 3.13; it must pass.
- By contributing you agree your work is licensed under Apache-2.0, like the rest of the project.
