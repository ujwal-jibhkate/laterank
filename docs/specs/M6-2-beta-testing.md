---
id: M6-2
title: Beta testing: report template, 0.2.1, public post
status: done
approved: 2026-11-13
---

# M6-2: Beta testing: report template, 0.2.1, public post

## Goal

Outside users can try laterank and tell us what broke or confused them (design doc §11, "beta
testing: outside users try it", Nov 16-27). After this checkpoint, `0.2.1` is on PyPI with M6-1's
faster `add` refusals, the README links to a beta-report issue template, and a short public post
is ready for the owner to publish.

## Out of scope

- Fixing what testers report: each report becomes its own spec (M6-3, ...) once the owner picks it.
- Any change to `src/` or `tests/`. This checkpoint changes only docs, `.github/` and the release.
- Skipped: a GitHub issue *form* (`.yml`, required fields, dropdowns), add when free-text reports
  come back missing the version or OS. A Markdown template is 20 lines and has no schema to get wrong.
- Skipped: telemetry, a feedback survey, a Discussions board. Add one when issues aren't enough.
- Skipped: a docs site (M4's plan named one; the README is the only doc). Add it when testers say
  the README isn't enough.

## Interfaces

`.github/ISSUE_TEMPLATE/beta-report.md`, a Markdown issue template:

```markdown
---
name: Beta report
about: Something broke, was slow, or confused you while trying laterank
labels: beta
---

**What you tried** (the command or code, and roughly how many documents)

**What happened** (paste the full error or output)

**What you expected**

**Setup**: output of
`python -c "import laterank, platform, sys; print(laterank.__version__, platform.platform(), sys.version)"`,
and the model (`--hf-id` or encoder) you used.
```

`README.md`: one line under the Beta banner:
`Trying it? Tell us what broke or confused you: [open a beta report](https://github.com/ujwal-jibhkate/laterank/issues/new?template=beta-report.md).`

`docs/dev/beta_post.md`: in a helping-hand tone, an X thread (5 posts, each at most 280
characters, links counted as 23) and a LinkedIn post. Each says what laterank is, why
(single-vector vs late interaction), how (cheap first stage, then exact MaxSim) and how to use it
(`pip install "laterank[colbert]"`, `build --docs`, `search`), the measured SciFact nDCG@10
(0.7459, `bm25 > maxsim@100`, from the M5-3 log, not from memory), a repo link and the dates
(Nov 16-27); LinkedIn also has the beta-report link.

Release `0.2.1`: the same steps as M5-3, with tag `v0.2.1` and `dist/laterank-0.2.1*`. The owner
tags, uploads with the project-scoped token, and pushes the tag after the upload.

## Decisions

| decision | why | differs from design doc? |
|---|---|---|
| Release `0.2.1` before testers start (owner, 2026-11-13) | testers should get M6-1's refusals; a patch number because nothing in the API breaks | no |
| Public post, owner chooses where (owner, 2026-11-13) | reaches people the owner doesn't know; Claude drafts, never posts | no |
| One Markdown template, label `beta` | smallest thing that asks for version, OS and model; the label lets the owner filter reports | no |
| The post lives in `docs/dev/` (committed) | the log can point at the exact text that went out | no |

## Acceptance tests

- [x] AT1: after the push, `https://github.com/ujwal-jibhkate/laterank/issues/new/choose` lists
  "Beta report", and the README link opens the template with its four headings.
- [x] AT2: the `beta` label exists on the repo (`gh label list`), so new reports carry it.
- [x] AT3: both posts in `docs/dev/beta_post.md` have every item listed in Interfaces, each X post
  is at most 280 characters as X counts them, and the nDCG matches the M5-3 log entry.
- [x] AT4: `uv run pytest -q`, `uv run ruff check .`, `uv run ruff format --check .` pass; CI is
  green on every job for the commit to be tagged.
- [x] AT5: from the tag, `dist/` holds exactly the `0.2.1` sdist and wheel and `twine check`
  passes. After the upload, in a new venv outside the repo, `pip install "laterank[colbert]"`
  installs `0.2.1`, the quick-start ranks doc `a` first, and `laterank add IDX --docs empty.jsonl`
  fails with "has no documents" in under 2 s.
- [x] AT6: the owner has published both posts (X, LinkedIn); their links are in the progress log.
- [x] AT7: the `reviewer` agent returns `merge` (before the tag).

## Tasks

| # | task | files it owns | depends on | parallel? |
|---|---|---|---|---|
| 1 | issue template, README line, `beta` label | `.github/ISSUE_TEMPLATE/beta-report.md`, `README.md` | - | yes (with 2) |
| 2 | draft the post | `docs/dev/beta_post.md` | - | yes (with 1) |
| 3 | reviewer (AT7), commit on the owner's OK; owner pushes, CI (AT1, AT4) | none | 1, 2 | no |
| 4 | owner tags `v0.2.1`, builds, uploads, pushes the tag; verify from PyPI (AT5) | none | 3 | no |
| 5 | owner publishes the post; link into the log (AT6) | `docs/dev/progress.md` | 4 | no |

## Changes

- 2026-11-13 (reviewer round 1, "fix first", 4 findings, all fixed): the post's build time names
  the machine (Apple-silicon Mac, after downloads), the build command drops the default
  `--bm25 --pipeline` (matches the README), and the score names the model and its commit
  `bb13a29e` - public text must not claim more than was measured. Round 2: "merge".
- 2026-11-13 (owner): `beta_post.md` becomes an X post and a LinkedIn post, short, asking for
  help rather than "looking for testers"; the 3-command try moves out (the README has it) and
  AT3/AT6 now cover both posts - the owner posts on X and LinkedIn.
- 2026-11-13 (owner): the X post becomes a 5-post thread, and both posts explain what, why and
  how to use it (a 3-line try is back in) - one short post didn't say enough.
