"""Tests for scripts/specs.py, the rules behind the spec-driven workflow.

A bug here would silently switch off the safeguards, so each rule gets a test.
"""

from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("specs", REPO / "scripts" / "specs.py")
specs = importlib.util.module_from_spec(_spec)
sys.modules["specs"] = specs  # dataclasses needs the module registered
_spec.loader.exec_module(specs)

TEMPLATE = (REPO / "docs" / "specs" / "_template.md").read_text(encoding="utf-8")


def make_spec(root: Path, spec_id: str, status: str, ticked: bool = False) -> Path:
    text = TEMPLATE.replace("id: M0-0", f"id: {spec_id}")
    text = text.replace("status: draft", f"status: {status}")
    if status != "draft":
        text = text.replace("approved:\n", "approved: 2026-10-03\n")
    if ticked:
        text = text.replace("- [ ] AT1", "- [x] AT1")
    path = root / "docs" / "specs" / f"{spec_id}-thing.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / ".git").mkdir()
    return tmp_path


def run_hook(event: dict, monkeypatch, capsys) -> dict | None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(event)))
    assert specs.main(["specs.py", "claude-hook"]) == 0
    out = capsys.readouterr().out.strip()
    return json.loads(out)["hookSpecificOutput"] if out else None


# --- check -----------------------------------------------------------------------------------


def test_real_repo_specs_are_valid():
    assert specs.check(REPO) == []


def test_template_based_spec_is_valid(repo):
    make_spec(repo, "M9-1", "draft")
    assert specs.check(repo) == []


def test_missing_section_is_reported(repo):
    path = make_spec(repo, "M9-1", "draft")
    path.write_text(path.read_text().replace("## Out of scope", "## Something else"))
    assert any("Out of scope" in e for e in specs.check(repo))


def test_done_with_unticked_acceptance_test_is_reported(repo):
    make_spec(repo, "M9-1", "done")
    assert any("not ticked" in e for e in specs.check(repo))
    make_spec(repo, "M9-1", "done", ticked=True)
    assert specs.check(repo) == []


def test_two_approved_specs_are_reported(repo):
    make_spec(repo, "M9-1", "approved")
    make_spec(repo, "M9-2", "approved")
    assert any("one spec may be approved" in e for e in specs.check(repo))


def test_approved_needs_a_date(repo):
    path = make_spec(repo, "M9-1", "approved")
    path.write_text(path.read_text().replace("approved: 2026-10-03", "approved:"))
    assert any("YYYY-MM-DD" in e for e in specs.check(repo))


# --- Claude Code hook -----------------------------------------------------------------------


def test_hook_denies_src_edit_without_approved_spec(repo, monkeypatch, capsys):
    make_spec(repo, "M9-1", "draft")
    event = {"tool_name": "Edit", "tool_input": {"file_path": str(repo / "src" / "x.py")}}
    assert run_hook(event, monkeypatch, capsys)["permissionDecision"] == "deny"


def test_hook_allows_src_edit_with_approved_spec(repo, monkeypatch, capsys):
    make_spec(repo, "M9-1", "approved")
    event = {"tool_name": "Write", "tool_input": {"file_path": str(repo / "tests" / "t.py")}}
    assert run_hook(event, monkeypatch, capsys) is None


def test_hook_ignores_files_outside_guarded_folders(repo, monkeypatch, capsys):
    event = {"tool_name": "Write", "tool_input": {"file_path": str(repo / "README.md")}}
    assert run_hook(event, monkeypatch, capsys) is None


def test_hook_asks_before_approving_a_spec(repo, monkeypatch, capsys):
    path = make_spec(repo, "M9-1", "draft")
    edit = {"file_path": str(path), "old_string": "status: draft", "new_string": "status: approved"}
    decision = run_hook({"tool_name": "Edit", "tool_input": edit}, monkeypatch, capsys)
    assert decision["permissionDecision"] == "ask"


def test_hook_asks_before_writing_a_done_spec(repo, monkeypatch, capsys):
    path = make_spec(repo, "M9-1", "approved")
    content = path.read_text().replace("status: approved", "status: done")
    event = {"tool_name": "Write", "tool_input": {"file_path": str(path), "content": content}}
    assert run_hook(event, monkeypatch, capsys)["permissionDecision"] == "ask"


def test_hook_allows_ordinary_spec_edits(repo, monkeypatch, capsys):
    path = make_spec(repo, "M9-1", "draft")
    edit = {"file_path": str(path), "old_string": "Short name", "new_string": "Storage"}
    assert run_hook({"tool_name": "Edit", "tool_input": edit}, monkeypatch, capsys) is None


def test_hook_asks_before_a_shell_command_sets_a_status(repo, monkeypatch, capsys):
    sed = {"command": "sed -i 's/status: draft/status: done/' docs/specs/M9-1-x.md"}
    assert (
        run_hook({"tool_name": "Bash", "tool_input": sed}, monkeypatch, capsys)[
            "permissionDecision"
        ]
        == "ask"
    )
    for harmless in ("python3 scripts/specs.py check", "grep -n status: docs/specs/M9-1-x.md"):
        event = {"tool_name": "Bash", "tool_input": {"command": harmless}}
        assert run_hook(event, monkeypatch, capsys) is None


# --- git commit-msg hook --------------------------------------------------------------------


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


@pytest.fixture
def git_repo(tmp_path: Path, monkeypatch) -> Path:
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "t@example.com")
    git(tmp_path, "config", "user.name", "t")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def commit_error(root: Path, message: str, staged: str) -> str | None:
    target = root / staged
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("x\n")
    git(root, "add", staged)
    msg = root / "MSG"
    msg.write_text(message)
    return specs.commit_msg(str(msg))


def test_commit_rejects_claude_attribution(git_repo):
    msg = "Add x\n\nCo-Authored-By: Claude <noreply@anthropic.com>\n"
    assert "attribution" in commit_error(git_repo, msg, "README.md")


def test_commit_outside_guarded_folders_needs_no_trailer(git_repo):
    assert commit_error(git_repo, "Update readme\n", "README.md") is None


def test_commit_to_src_needs_a_spec_trailer(git_repo):
    assert "trailer" in commit_error(git_repo, "Add x\n", "src/x.py")


def test_commit_to_src_rejects_draft_and_unknown_specs(git_repo):
    make_spec(git_repo, "M9-1", "draft")
    assert "draft" in commit_error(git_repo, "Add x\n\nSpec: M9-1\n", "src/x.py")
    assert "no spec" in commit_error(git_repo, "Add x\n\nSpec: M9-7\n", "src/x.py")


def test_commit_to_src_accepts_approved_spec_and_reasoned_none(git_repo):
    make_spec(git_repo, "M9-1", "approved")
    assert commit_error(git_repo, "Add x\n\nSpec: M9-1\n", "src/x.py") is None
    assert commit_error(git_repo, "Fix x\n\nSpec: none - typo\n", "src/y.py") is None
    assert "reason" in commit_error(git_repo, "Fix x\n\nSpec: none\n", "src/z.py")
