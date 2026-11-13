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
