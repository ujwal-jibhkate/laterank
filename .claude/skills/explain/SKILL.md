---
name: explain
description: Teach the owner a laterank concept or piece of code in plain words, with a small runnable example. Read-only; changes no files.
argument-hint: "<topic, file, or function>"
disable-model-invocation: true
---

# Explain: $ARGUMENTS

The owner is learning. Teach, don't just summarize. Don't edit any file.

1. **One sentence**: what it is, in everyday words.
2. **Why laterank needs it**: what would break or get slow without it. Tie it to an smve-lab result
   when there is one (use the `golden-values` skill for numbers, never memory).
3. **Walk through the code**: the 3 to 6 lines that matter, cited as `path:line`. Skip
   boilerplate.
4. **A tiny example you actually run**: about 10 lines of numpy with toy sizes (3 tokens, 4
   dimensions), run with `uv run python -c` or a scratch file outside the repo. Show the real
   output and explain each number.
5. **One common mistake** someone could make with it.
6. **Two short questions** the owner can answer to check they understood. Put the answers in a
   collapsed `<details>` block.

Keep it under about 60 lines. Use a short mermaid diagram only if the idea is about flow between
parts.
