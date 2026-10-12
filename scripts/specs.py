#!/usr/bin/env python3
"""Enforce laterank's spec-driven workflow. Standard library only.

Every checkpoint has a spec in docs/specs/<id>-<slug>.md (see docs/specs/README.md). A spec moves
through three statuses: draft, then approved (the owner agreed to the plan), then done (every
acceptance test passes). This script enforces the rules in three places:

  check              validate every spec. CI runs this.
  active             print the id of the approved spec, or exit 1 if there is none.
  claude-hook        Claude Code PreToolUse hook. It reads the tool call as JSON on stdin and:
                       - denies edits to src/ or tests/ while no spec is approved;
                       - forces a permission prompt when an edit would set a spec's status
                         to approved or done, because only the owner approves.
  commit-msg FILE    git commit-msg hook. It rejects Claude attribution, and requires a
                     "Spec: <id>" trailer on commits that touch src/ or tests/.

Shell commands get a text check only: one that mentions docs/specs and "status: approved" or
"status: done" forces the same permission prompt. Other shell edits bypass the Claude Code hook,
and the commit-msg hook is the backstop for src/ and tests/.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

STATUSES = ("draft", "approved", "done")
SECTIONS = (
    "Goal",
    "Out of scope",
    "Interfaces",
    "Decisions",
    "Acceptance tests",
    "Tasks",
    "Changes",
)
GUARDED = ("src", "tests")  # top-level folders that may change only under an approved spec
SPEC_DIR = Path("docs/specs")

FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)
STATUS_LINE = re.compile(r"^status:\s*(\S+)", re.M)
SHELL_STATUS = re.compile(r"status:\s*(approved|done)")
TRAILER = re.compile(r"^Spec:[ \t]*(\S+)(.*)$", re.M)
ATTRIBUTION = re.compile(r"co-authored-by:.*(claude|anthropic)|generated with.*claude", re.I)
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


@dataclass
class Spec:
    path: Path
    meta: dict[str, str]
    text: str

    @property
    def id(self) -> str:
        return self.meta.get("id", "")

    @property
    def status(self) -> str:
        return self.meta.get("status", "")


def parse(path: Path, text: str | None = None) -> Spec:
    text = path.read_text(encoding="utf-8") if text is None else text
    meta: dict[str, str] = {}
    m = FRONT_MATTER.match(text.replace("\r\n", "\n"))
    if m:
        for line in m.group(1).splitlines():
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    return Spec(path, meta, text)


def load_specs(root: Path) -> list[Spec]:
    """Every spec under root/docs/specs, skipping README.md and files starting with '_'."""
    folder = root / SPEC_DIR
    if not folder.is_dir():
        return []
    return [
        parse(p)
        for p in sorted(folder.glob("*.md"))
        if p.name != "README.md" and not p.name.startswith("_")
    ]


def section(text: str, title: str) -> str:
    m = re.search(rf"^## {re.escape(title)}[ \t]*\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    return m.group(1) if m else ""


def spec_problems(spec: Spec) -> list[str]:
    errors = []
    if not spec.id:
        errors.append("front matter has no 'id'")
    elif not spec.path.name.startswith(spec.id + "-"):
        errors.append(f"file name must start with '{spec.id}-'")
    if spec.status not in STATUSES:
        errors.append(f"status must be one of {', '.join(STATUSES)}, not '{spec.status}'")
    if spec.status in ("approved", "done") and not DATE.fullmatch(spec.meta.get("approved", "")):
        errors.append("an approved or done spec needs 'approved: YYYY-MM-DD'")
    headings = set(re.findall(r"^## (.+?)[ \t]*$", spec.text, re.M))
    errors += [f"missing section '## {s}'" for s in SECTIONS if s not in headings]
    tests = section(spec.text, "Acceptance tests")
    if not re.search(r"^- \[[ xX]\]", tests, re.M):
        errors.append("'## Acceptance tests' needs at least one '- [ ]' checkbox")
    if spec.status == "done" and re.search(r"^- \[ \]", tests, re.M):
        errors.append("status is done, but some acceptance tests are not ticked")
    return [f"{spec.path.name}: {e}" for e in errors]


def check(root: Path) -> list[str]:
    specs = load_specs(root)
    errors = [e for s in specs for e in spec_problems(s)]
    ids = [s.id for s in specs if s.id]
    errors += [f"duplicate spec id '{i}'" for i in sorted({i for i in ids if ids.count(i) > 1})]
    approved = [s.id for s in specs if s.status == "approved"]
    if len(approved) > 1:
        errors.append(f"only one spec may be approved at a time, found: {', '.join(approved)}")
    return errors


def find_root(start: Path) -> Path | None:
    """The nearest folder containing .git (a directory, or a file in a git worktree)."""
    for folder in (start, *start.parents):
        if (folder / ".git").exists():
            return folder
    return None


def decide(decision: str, reason: str) -> None:
    out = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": decision,
            "permissionDecisionReason": reason,
        }
    }
    print(json.dumps(out))


def claude_hook(event: dict) -> None:
    tool_input = event.get("tool_input") or {}
    command = tool_input.get("command")
    if command is not None:  # Bash: a script or sed could set a spec's status
        # ponytail: text match, so it can miss indirect edits; a PostToolUse diff of spec
        # statuses would catch every path if this ever proves too weak.
        if "docs/specs" in command and (m := SHELL_STATUS.search(command)):
            decide(
                "ask",
                f"This shell command may set a spec's status to '{m.group(1)}'. Only the owner "
                "approves specs and marks them done. Allow it only if you asked for this.",
            )
        return
    file_path = tool_input.get("file_path")
    if not file_path:
        return
    path = Path(file_path)
    if not path.is_absolute():
        path = Path(event.get("cwd") or ".") / path
    path = path.resolve()
    root = find_root(path.parent)
    if root is None:
        return
    try:
        rel = path.relative_to(root.resolve())
    except ValueError:
        return

    if rel.parts and rel.parts[0] in GUARDED:
        if not any(s.status == "approved" for s in load_specs(root)):
            decide(
                "deny",
                "laterank is spec-driven: src/ and tests/ can change only while a spec is "
                "approved, and none is. Draft the checkpoint's spec with /spec and ask the "
                "owner to approve it. See docs/specs/README.md.",
            )
        return

    if rel.parent == SPEC_DIR and rel.suffix == ".md" and not rel.name.startswith("_"):
        old = parse(path).status if path.exists() else None
        if "content" in tool_input:  # Write
            new = parse(path, tool_input["content"]).status
        else:  # Edit
            m = STATUS_LINE.search(tool_input.get("new_string") or "")
            new = m.group(1) if m else old
        if new in ("approved", "done") and new != old:
            decide(
                "ask",
                f"This sets {rel.name} to '{new}'. Only the owner approves specs and marks "
                "them done. Allow it only if you asked for this.",
            )


def git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def commit_msg(msg_file: str) -> str | None:
    """Return an error message if the commit must be rejected, otherwise None."""
    raw = Path(msg_file).read_text(encoding="utf-8")
    message = "\n".join(line for line in raw.splitlines() if not line.startswith("#"))
    if ATTRIBUTION.search(message):
        return "Claude attribution is not allowed in this repo (see CLAUDE.md)."

    git_dir = Path(git("rev-parse", "--git-dir"))
    if (git_dir / "MERGE_HEAD").exists():
        return None  # merges carry the Spec trailers of the commits they bring in
    staged = git("diff", "--cached", "--name-only").splitlines()
    touched = [f for f in staged if f.split("/", 1)[0] in GUARDED]
    if not touched:
        return None

    m = TRAILER.search(message)
    if not m:
        return (
            f"this commit changes {len(touched)} file(s) in src/ or tests/, so it needs a "
            "trailer naming its spec as the last line, e.g. 'Spec: M0-3'. For a small fix "
            "outside any checkpoint, write 'Spec: none - <reason>'."
        )
    spec_id, rest = m.group(1), m.group(2).strip(" -")
    if spec_id == "none":
        return None if rest else "'Spec: none' needs a reason, e.g. 'Spec: none - fix typo'."
    root = Path(git("rev-parse", "--show-toplevel"))
    spec = next((s for s in load_specs(root) if s.id == spec_id), None)
    if spec is None:
        return f"no spec with id '{spec_id}' in docs/specs/."
    if spec.status == "draft":
        return f"spec {spec_id} is still a draft. Ask the owner to approve it first."
    return None


def main(argv: list[str]) -> int:
    command = argv[1] if len(argv) > 1 else ""
    if command == "check":
        root = find_root(Path.cwd()) or Path.cwd()
        errors = check(root)
        for e in errors:
            print(f"spec error: {e}", file=sys.stderr)
        if not errors:
            print(f"{len(load_specs(root))} spec(s) OK")
        return 1 if errors else 0
    if command == "active":
        root = find_root(Path.cwd()) or Path.cwd()
        active = [s for s in load_specs(root) if s.status == "approved"]
        if not active:
            print("no approved spec", file=sys.stderr)
            return 1
        print(f"{active[0].id}  {active[0].path.relative_to(root).as_posix()}")
        return 0
    if command == "claude-hook":
        claude_hook(json.load(sys.stdin))
        return 0
    if command == "commit-msg" and len(argv) == 3:
        error = commit_msg(argv[2])
        if error:
            print(f"commit-msg: {error}", file=sys.stderr)
            return 1
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
