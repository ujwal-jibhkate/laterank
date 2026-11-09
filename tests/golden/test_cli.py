"""The real `laterank` command on SciFact with a real ColBERT model: build, evaluate, search.

Needs the `colbert` extra (`uv sync --extra colbert`), mxbai at the pinned sha (~134 MB) and the
SciFact download (2.8 MB). This is also the first real ColBERT build followed by `Index.open`.
`-s` prints the numbers and how long one `search` takes, model load included.

0.7453 / 0.9249 come from the golden-values skill. They were measured on smve-lab's stored texts,
which differ slightly from BEIR's (M3-1), so the band is 0.01.
"""

import subprocess
import sys
import time
from pathlib import Path

import pytest

from laterank.api.cli import main

pytestmark = pytest.mark.slow

HF_ID = "mixedbread-ai/mxbai-edge-colbert-v0-32m"
SHA = "bb13a29ec9b1e7edd4ba8f7a0776c48b55cbad66"


def test_build_evaluate_search(golden_dir, tmp_path, capsys):
    pytest.importorskip("sentence_transformers", reason="needs the colbert extra")
    index = str(tmp_path / "index")
    build = ["build", index, "--dataset", "scifact", "--encoder", "colbert"]
    assert main([*build, "--hf-id", HF_ID, "--revision", SHA]) == 0
    assert "default pipeline 'bm25 > maxsim@100'" in capsys.readouterr().out

    assert main(["evaluate", index, "--dataset", "scifact"]) == 0
    lines = capsys.readouterr().out.splitlines()
    ndcg, recall = (float(x) for x in lines[1].split()[-3:-1])

    laterank = Path(sys.executable).parent / "laterank"  # the installed console script
    t0 = time.perf_counter()
    done = subprocess.run(
        [str(laterank), "search", index, "statin side effects"], capture_output=True, text=True
    )
    seconds = time.perf_counter() - t0
    with capsys.disabled():
        print(f"\n{lines[1]}\n{lines[2]}\none `laterank search`: {seconds:.1f} s")

    assert done.returncode == 0, done.stderr
    assert len(done.stdout.splitlines()) == 10
    assert ndcg == pytest.approx(0.7453, abs=0.01)
    assert recall == pytest.approx(0.9249, abs=0.01)
