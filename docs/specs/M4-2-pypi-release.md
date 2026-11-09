---
id: M4-2
title: PyPI beta release
status: approved
approved: 2026-11-09
---

# M4-2: PyPI beta release

## Goal

`pip install "laterank[colbert]"` installs `0.1.0b1` from PyPI, and the README's quick-start works
on it. An uploaded version can never be replaced, so this checkpoint first fixes what a release
would otherwise publish wrongly, and checks the exact files before they leave this machine.

## Out of scope

- Skipped: a GitHub Actions publish workflow with Trusted Publishing, add when releases become
  routine (a good time: 0.1.0).
- Skipped: a TestPyPI rehearsal, add when a release does something this one can't check locally. A
  bad beta costs one yanked version; `0.1.0b2` is available.
- Skipped: tests in the sdist, add when a packager asks (they also carry the CC BY-SA qrels).
- Not changed: the M4-1 reviewer's low findings (second quick-start run, 11-document label-free
  minimum, the two local imports in `api/cli.py`).

## Interfaces

No change to `src/`. In `tests/orchestration/test_index.py`, the five `h.score` comparisons use
`pytest.approx` (see Changes).

`pyproject.toml`:

```toml
classifiers = ["Development Status :: 4 - Beta", ...]  # replaces 2 - Pre-Alpha; the rest stay

[project.urls]
Repository = "https://github.com/ujwal-jibhkate/laterank"

[project.optional-dependencies]  # rerank and pandas removed, the "Names are fixed" comment too
bge-m3 = ["FlagEmbedding>=1.4.2"]
colbert = ["sentence-transformers>=6.0.0"]
all = ["laterank[bge-m3,colbert]"]

[tool.hatch.build.targets.sdist]  # without it: .claude/, CLAUDE.md, docs/dev, the qrels file
include = ["/src", "/README.md", "/LICENSE"]
```

`uv.lock` is regenerated with `uv lock` (CI runs `uv sync --locked`).

`README.md`: the install line becomes `pip install "laterank[colbert]"`. The three relative links
(`docs/dev/tune_validation.md`, `START_HERE.md`, `LICENSE`) become
`https://github.com/ujwal-jibhkate/laterank/blob/main/...`, because relative links break on PyPI.

`docs/dev/progress.md`: M4-3's "CI not yet run" becomes CI green (`4492766`).

Release steps, from a clean tree on the commit CI passed. A dirty tree gives a version PyPI refuses.

```bash
git tag v0.1.0b1                      # owner; hatch-vcs reads the version from this tag
uv build                              # dist/laterank-0.1.0b1.tar.gz and -py3-none-any.whl
uvx twine check dist/*
uv publish dist/laterank-0.1.0b1*     # owner, in their own terminal; only the intended files
git push origin v0.1.0b1              # owner, after the upload succeeds
```

The token never enters this session: no `! uv publish`, and it is never pasted here. At uv's
prompts, type `__token__` as the username and paste the token as the password. The first upload
needs an account-wide PyPI token. Replace it with a project-scoped one afterward.

## Decisions

| decision | why | differs from design doc? |
|---|---|---|
| Upload by hand with `uv publish` | one beta needs no new file; `uv` is already the project's tool | no |
| No TestPyPI | `twine check` and a clean-venv wheel install cover what it would catch; a second account for little gain | no |
| The sdist lists what it includes | an allowlist can't leak a file added later; tried on a copy: 6 top-level entries, 63 KB, wheel unchanged at 44 files, `twine check` passes | no |
| Tag locally, push the tag after the upload | a failed check leaves only a local tag to delete | no |
| Keep the README's "Pre-release" line | still true for a beta | no |

## Acceptance tests

- [ ] AT1: `pyproject.toml` matches Interfaces and `uv lock --check` passes.
- [ ] AT2: `uv run pytest -q`, `uv run ruff check .`, `uv run ruff format --check .` pass, and CI is
  green on all 6 jobs and lint for the commit to be tagged.
- [ ] AT3: the README has no `git+` install line and no relative links.
- [ ] AT4: from the tag, `dist/` holds exactly `laterank-0.1.0b1.tar.gz` and
  `laterank-0.1.0b1-py3-none-any.whl`; `twine check` passes; the sdist has only `src/`, `README.md`,
  `LICENSE`, `pyproject.toml`, `PKG-INFO`, `.gitignore`; the metadata says Beta and has no
  `pandas` or `rerank` extra.
- [ ] AT5: in a new venv outside the repo, the built wheel installs with no extras; `laterank.__version__`
  is `0.1.0b1` and `laterank --help` exits 0.
- [ ] AT6: after the upload, in a new venv outside the repo, the README's install line (no `--pre`)
  installs `0.1.0b1`, the quick-start copied unchanged ranks doc `a` first, and PyPI's JSON for
  the release has the README text and the Repository link.
- [ ] AT7: the `reviewer` agent returns `merge` (it runs before the tag; see Tasks).

## Tasks

| # | task | files it owns | depends on | parallel? |
|---|---|---|---|---|
| 1 | packaging, README and log edits, `uv lock`, the score tolerance (AT1, AT3) | `pyproject.toml`, `uv.lock`, `README.md`, `docs/dev/progress.md`, `tests/orchestration/test_index.py` | none | no |
| 2 | reviewer (AT7), then commit on the owner's OK; owner pushes, CI (AT2) | none | 1 | no |
| 3 | owner tags; build and check the files (AT4, AT5) | none (`dist/` is ignored) | 2 | no |
| 4 | owner uploads and pushes the tag; verify from PyPI (AT6) | none | 3 | no |

## Changes

- 2026-11-09: `uv publish` names the release files, and the token prompts are spelled out -
  reviewer findings 1 and 2: `dist/*` also takes older builds, and uv asks for a username first.
- 2026-11-09: the five exact `h.score` comparisons in `test_index.py` get a tolerance (owner OK) -
  CI failed on `995b92e` (macOS, py3.12): float32 products from Accelerate round differently at
  different memory addresses (16 outputs from 16 offsets), and these tests compare a memory-mapped
  index with an in-memory copy. Seen once in 27 CI runs. AT7's `merge` predates this test-only edit.
