"""The `laterank` command, run through main([...]) with a fake encoder: no model, no network."""

import contextlib
import io
import json
import sys
from importlib.metadata import entry_points

import pytest

from laterank.api import cli
from laterank.api.cli import main
from laterank.components import bgem3, colbert
from laterank.components import encoder as encoder_mod
from laterank.orchestration.index import Index
from laterank.orchestration.tune import Budget, Point, TuneReport

WORDS = "red apple green pear yellow lemon purple plum orange peach sweet fig".split()
CORPUS = [
    {"_id": f"d{i}", "title": f"Title {i}", "text": " ".join(WORDS[i:] + WORDS[:i])}
    for i in range(12)
]
HEADER = "query-id\tcorpus-id\tscore\n"


@pytest.fixture
def docs(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in CORPUS), encoding="utf-8")
    return path


@pytest.fixture
def beir(tmp_path):
    folder = tmp_path / "tiny"
    (folder / "qrels").mkdir(parents=True)
    (folder / "corpus.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in CORPUS), encoding="utf-8"
    )
    (folder / "queries.jsonl").write_text('{"_id": "q1", "text": "red apple"}\n', encoding="utf-8")
    (folder / "qrels/test.tsv").write_text(f"{HEADER}q1\td0\t1\n", encoding="utf-8")
    return folder


@pytest.fixture
def encoder(monkeypatch, fake_encoder_cls):
    """`build` gets a fake with dense and lexical heads; `load_encoder` rebuilds the same one."""
    enc = fake_encoder_cls()
    monkeypatch.setattr(cli, "_make_encoder", lambda name, hf_id, revision: enc)
    monkeypatch.setitem(encoder_mod.ENCODERS, "fake", lambda info: enc)
    return enc


@pytest.fixture
def no_model(monkeypatch):
    def fail(*args):
        raise AssertionError("the model must not load")

    monkeypatch.setattr(cli, "_make_encoder", fail)


def _build(path, docs, *extra, encoder="bge-m3"):
    assert main(["build", str(path), "--docs", str(docs), "--encoder", encoder, *extra]) == 0


# AT1


def test_build_from_docs_uses_the_bge_m3_default(tmp_path, docs, encoder, capsys):
    _build(tmp_path / "idx", docs)
    index = Index.open(tmp_path / "idx")
    assert str(index.default_pipeline) == "hybrid > maxsim@100"
    assert capsys.readouterr().out == (
        f"built {tmp_path / 'idx'}: 12 documents, stages ['dense', 'hybrid', 'lexical'], "
        "default pipeline 'hybrid > maxsim@100'\n"
    )


def test_build_from_a_dataset_folder(tmp_path, beir, encoder):
    args = ["build", str(tmp_path / "idx"), "--dataset", str(beir), "--encoder", "bge-m3"]
    assert main(args) == 0
    assert len(Index.open(tmp_path / "idx")._docs) == 12


@pytest.mark.parametrize("named", [False, True], ids=["folder", "name"])
def test_build_needs_only_the_corpus(tmp_path, beir, encoder, monkeypatch, named):
    (beir / "queries.jsonl").unlink()
    (beir / "qrels/test.tsv").unlink()
    monkeypatch.setattr(cli, "download_beir", lambda name: beir)
    dataset = "scifact" if named else str(beir)
    args = ["build", str(tmp_path / "idx"), "--dataset", dataset, "--encoder", "bge-m3"]
    assert main(args) == 0
    assert len(Index.open(tmp_path / "idx")._docs) == 12


def test_build_colbert_default_needs_only_tokens(tmp_path, docs, monkeypatch, fake_encoder_cls):
    enc = fake_encoder_cls(dense=False, lexical=False)
    monkeypatch.setattr(cli, "_make_encoder", lambda name, hf_id, revision: enc)
    _build(tmp_path / "idx", docs, "--hf-id", "x/y", encoder="colbert")
    index = Index.open(tmp_path / "idx")
    assert str(index.default_pipeline) == "bm25 > maxsim@100"
    assert list(index.manifest.stages) == ["bm25"]


def test_stage_flag_builds_the_stage(tmp_path, docs, encoder):
    _build(tmp_path / "idx", docs, "--muvera")
    assert "muvera" in Index.open(tmp_path / "idx").manifest.stages


def test_pipeline_first_stage_builds_without_its_flag(tmp_path, docs, encoder):
    _build(tmp_path / "idx", docs, "--pipeline", "bm25 > maxsim@5")
    index = Index.open(tmp_path / "idx")
    assert "bm25" in index.manifest.stages
    assert str(index.default_pipeline) == "bm25 > maxsim@5"


@pytest.mark.parametrize(
    ("extra", "stage"),
    [
        (["--bm25"], "bm25"),
        (["--smve"], "smve"),
        (["--muvera"], "muvera"),
        (["--pipeline", "smve"], "smve"),
        (["--pipeline", "muvera > maxsim@5"], "muvera"),
    ],
)
def test_stage_choices_reach_the_build(tmp_path, docs, encoder, monkeypatch, extra, stage):
    seen = {}

    def spy(path, documents, enc, **kw):
        seen.update(kw)
        raise ValueError("stop here")

    monkeypatch.setattr(Index, "build", spy)
    args = ["build", str(tmp_path / "idx"), "--docs", str(docs), "--encoder", "bge-m3", *extra]
    assert main(args) == 1
    assert [s for s in ("bm25", "smve", "muvera") if seen[s]] == [stage]


def test_build_passes_the_model_and_its_pin(tmp_path, docs, encoder, monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "_make_encoder", lambda *a: seen.append(a) or encoder)
    _build(tmp_path / "idx", docs, "--hf-id", "x/y", "--revision", "abc")
    assert seen == [("bge-m3", "x/y", "abc")]


def test_make_encoder_builds_the_named_model(monkeypatch):
    monkeypatch.setattr(bgem3, "BGEM3Encoder", lambda *a: ("bge-m3", *a))
    monkeypatch.setattr(colbert, "ColBERTEncoder", lambda *a: ("colbert", *a))
    assert cli._make_encoder("bge-m3", None, None) == ("bge-m3", "BAAI/bge-m3", None)
    assert cli._make_encoder("colbert", "x/y", "sha") == ("colbert", "x/y", "sha")


# AT2


@pytest.mark.usefixtures("no_model")
@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--encoder", "colbert"], "--hf-id"),
        (["--encoder", "bge-m3", "--pipeline", "dense >"], "must be a reranker"),
    ],
)
def test_bad_build_arguments_fail_before_the_model_loads(tmp_path, docs, capsys, extra, message):
    assert main(["build", str(tmp_path / "idx"), "--docs", str(docs), *extra]) == 1
    assert message in capsys.readouterr().err


@pytest.mark.usefixtures("no_model")
def test_build_refuses_a_folder_that_holds_files(tmp_path, docs, capsys):
    (tmp_path / "idx").mkdir()
    (tmp_path / "idx/keep.txt").write_text("x", encoding="utf-8")
    args = ["build", str(tmp_path / "idx"), "--docs", str(docs), "--encoder", "bge-m3"]
    assert main(args) == 1
    assert "isn't empty" in capsys.readouterr().err


# AT3


def test_dataset_folder_is_read_without_a_download(beir, monkeypatch):
    monkeypatch.setattr(cli, "download_beir", lambda *a: pytest.fail("must not download"))
    assert cli._dataset(str(beir), "test").qrels == {"q1": {"d0": 1}}


def test_folder_without_a_corpus_doesnt_hide_the_beir_name(tmp_path, beir, monkeypatch, capsys):
    (tmp_path / "scifact").mkdir()  # an index folder named like the dataset
    monkeypatch.chdir(tmp_path)
    asked = []
    monkeypatch.setattr(cli, "download_beir", lambda name: asked.append(name) or beir)
    assert cli._dataset("scifact", "test").qrels == {"q1": {"d0": 1}}
    assert asked == ["scifact"]
    captured = capsys.readouterr()  # the notice stays out of the results
    assert captured.out == "" and "BEIR dataset 'scifact'" in captured.err


@pytest.mark.parametrize("named", [False, True], ids=["folder", "name"])
@pytest.mark.parametrize("command", ["evaluate", "tune"])
def test_split_reaches_the_dataset(tmp_path, docs, beir, encoder, monkeypatch, command, named):
    (beir / "qrels/test.tsv").rename(beir / "qrels/dev.tsv")  # only `--split dev` can load it
    monkeypatch.setattr(cli, "download_beir", lambda name: beir)
    _build(tmp_path / "idx", docs)
    idx = str(tmp_path / "idx")
    args = {
        "evaluate": ["evaluate", idx],
        "tune": ["tune", idx, "--budget-ms", "1e9"],
    }[command]
    assert main([*args, "--dataset", "scifact" if named else str(beir), "--split", "dev"]) == 0


@pytest.mark.usefixtures("no_model")
def test_unknown_dataset_name_lists_the_known_ones(tmp_path, capsys):
    args = ["build", str(tmp_path / "idx"), "--dataset", "nope", "--encoder", "bge-m3"]
    assert main(args) == 1
    assert "known:" in capsys.readouterr().err


# AT4


def test_search_prints_one_line_per_hit(tmp_path, docs, encoder, capsys):
    _build(tmp_path / "idx", docs)
    capsys.readouterr()
    assert main(["search", str(tmp_path / "idx"), "red apple", "-k", "3"]) == 0
    index = Index.open(tmp_path / "idx")
    hits = index.search(encoder.encode_query("red apple"), 3)
    assert capsys.readouterr().out.splitlines() == [
        f"{h.rank:>3}  {h.score:.4f}  {h.id}  {h.text}" for h in hits
    ]


def test_search_collapses_whitespace_and_cuts_long_text(tmp_path, encoder, capsys):
    path = tmp_path / "long.jsonl"
    rows = [{"_id": "a", "text": "word\n  " * 40}, {"_id": "b", "text": "short"}]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    _build(tmp_path / "idx", path, "--pipeline", "exhaustive")
    capsys.readouterr()
    assert main(["search", str(tmp_path / "idx"), "word", "-k", "2"]) == 0
    line = next(x for x in capsys.readouterr().out.splitlines() if "  a  " in x)
    assert line.split("  a  ")[1] == ("word " * 40)[:80]


def test_search_pipeline_overrides_the_default(tmp_path, docs, encoder, capsys):
    _build(tmp_path / "idx", docs)
    capsys.readouterr()
    args = ["search", str(tmp_path / "idx"), "red apple", "-k", "12", "--pipeline", "lexical"]
    assert main(args) == 0
    index = Index.open(tmp_path / "idx")
    hits = index.search(encoder.encode_query("red apple"), 12, pipeline="lexical")
    assert [x.split()[2] for x in capsys.readouterr().out.splitlines()] == [h.id for h in hits]


# AT5


def test_evaluate_defaults_to_the_index_pipeline(tmp_path, docs, beir, encoder, capsys):
    _build(tmp_path / "idx", docs)
    capsys.readouterr()
    assert main(["evaluate", str(tmp_path / "idx"), "--dataset", str(beir)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3 and lines[1].startswith("hybrid > maxsim@100")
    assert lines[2].startswith("query encoding: p50 ") and lines[2].endswith(" ms over 1 queries")


def test_evaluate_prints_one_row_per_pipeline(tmp_path, docs, beir, encoder, capsys):
    _build(tmp_path / "idx", docs)
    capsys.readouterr()
    args = ["evaluate", str(tmp_path / "idx"), "--dataset", str(beir)]
    assert main([*args, "--pipeline", "lexical", "--pipeline", "dense > maxsim@5"]) == 0
    rows = capsys.readouterr().out.splitlines()[1:3]
    assert rows[0].startswith("lexical") and rows[1].startswith("dense > maxsim@5")


def test_evaluate_uses_the_datasets_identical_ids_rule(tmp_path, docs, encoder, capsys):
    folder = tmp_path / "arguana"  # the query is the document d0, judged relevant to d1
    (folder / "qrels").mkdir(parents=True)
    (folder / "corpus.jsonl").write_bytes(docs.read_bytes())
    text = f"{CORPUS[0]['title']}. {CORPUS[0]['text']}"
    (folder / "queries.jsonl").write_text(
        json.dumps({"_id": "d0", "text": text}) + "\n", encoding="utf-8"
    )
    (folder / "qrels/test.tsv").write_text(f"{HEADER}d0\td1\t1\n", encoding="utf-8")
    _build(tmp_path / "idx", docs, "--pipeline", "exhaustive")
    capsys.readouterr()
    assert main(["evaluate", str(tmp_path / "idx"), "--dataset", str(folder)]) == 0
    row = capsys.readouterr().out.splitlines()[1].split()

    index = Index.open(tmp_path / "idx")
    queries = {"d0": encoder.encode_query(text)}
    qrels = {"d0": {"d1": 1}}
    drop = index.evaluate(queries, qrels, ["exhaustive"], ignore_identical_ids=True).scores[0]
    keep = index.evaluate(queries, qrels, ["exhaustive"]).scores[0]
    assert float(row[1]) == pytest.approx(drop.ndcg_at_10, abs=5e-5)
    assert drop.ndcg_at_10 != keep.ndcg_at_10  # the rule matters on this data


@pytest.mark.parametrize("command", ["evaluate", "search"])
def test_a_missing_stage_fails_before_the_model_loads(
    tmp_path, docs, beir, encoder, monkeypatch, capsys, command
):
    _build(tmp_path / "idx", docs)
    monkeypatch.setitem(encoder_mod.ENCODERS, "fake", lambda info: pytest.fail("loaded"))
    rest = ["--dataset", str(beir)] if command == "evaluate" else ["red apple"]
    assert main([command, str(tmp_path / "idx"), *rest, "--pipeline", "muvera"]) == 1
    assert "starts with 'muvera'" in capsys.readouterr().err


def test_a_dataset_with_no_judged_queries_is_refused(tmp_path, docs, beir, encoder, capsys):
    (beir / "qrels/test.tsv").write_text(HEADER, encoding="utf-8")
    _build(tmp_path / "idx", docs)
    assert main(["evaluate", str(tmp_path / "idx"), "--dataset", str(beir)]) == 1
    assert "qrels is empty" in capsys.readouterr().err


# AT6


def _stub_tune(monkeypatch, best):
    """Index.tune returns a fixed report; the returned dict collects its keyword arguments."""
    points = (Point("lexical", 0.5, 1.0),)
    report = TuneReport(best, points, 1, Budget(20), "nDCG@10")
    seen = {}
    monkeypatch.setattr(Index, "tune", lambda self, budget, **kw: seen.update(kw) or report)
    return seen


def test_tune_uses_the_datasets_identical_ids_rule(tmp_path, docs, beir, encoder, monkeypatch):
    folder = beir.rename(tmp_path / "arguana")
    _build(tmp_path / "idx", docs)
    seen = _stub_tune(monkeypatch, "lexical")
    args = ["tune", str(tmp_path / "idx"), "--budget-ms", "20", "--dataset", str(folder)]
    assert main(args) == 0
    assert seen["ignore_identical_ids"] is True


def test_tune_with_a_dataset_uses_labels(tmp_path, docs, beir, encoder, capsys):
    _build(tmp_path / "idx", docs)
    capsys.readouterr()
    args = ["tune", str(tmp_path / "idx"), "--budget-ms", "1e9", "--dataset", str(beir)]
    assert main(args) == 0
    captured = capsys.readouterr()
    assert "nDCG@10" in captured.out.splitlines()[0]
    assert "without labels" not in captured.err


def test_tune_without_a_dataset_warns_about_agreement(tmp_path, docs, encoder, capsys):
    _build(tmp_path / "idx", docs)
    capsys.readouterr()
    assert main(["tune", str(tmp_path / "idx"), "--budget-ms", "1e9"]) == 0
    captured = capsys.readouterr()
    assert "agreement" in captured.out.splitlines()[0]
    assert "agreement with exact MaxSim, not relevance" in captured.err


def test_tune_apply_changes_the_default_and_search_uses_it(
    tmp_path, docs, encoder, monkeypatch, capsys
):
    _build(tmp_path / "idx", docs)
    _stub_tune(monkeypatch, "lexical")
    capsys.readouterr()
    assert main(["tune", str(tmp_path / "idx"), "--budget-ms", "20", "--apply"]) == 0
    assert capsys.readouterr().out.splitlines()[-2:] == [
        "best: lexical",
        "default pipeline: lexical",
    ]
    index = Index.open(tmp_path / "idx")
    assert str(index.default_pipeline) == "lexical"
    assert main(["search", str(tmp_path / "idx"), "red apple", "-k", "12"]) == 0
    hits = index.search(encoder.encode_query("red apple"), 12, pipeline="lexical")
    assert [x.split()[2] for x in capsys.readouterr().out.splitlines()] == [h.id for h in hits]


def test_tune_without_apply_leaves_the_manifest_alone(tmp_path, docs, encoder, monkeypatch):
    _build(tmp_path / "idx", docs)
    before = (tmp_path / "idx/manifest.json").read_bytes()
    _stub_tune(monkeypatch, "lexical")
    assert main(["tune", str(tmp_path / "idx"), "--budget-ms", "20"]) == 0
    assert (tmp_path / "idx/manifest.json").read_bytes() == before


@pytest.mark.parametrize(("apply", "code"), [([], 0), (["--apply"], 1)])
def test_tune_when_nothing_fits_the_budget(tmp_path, docs, beir, encoder, capsys, apply, code):
    _build(tmp_path / "idx", docs)
    before = (tmp_path / "idx/manifest.json").read_bytes()
    capsys.readouterr()
    args = ["tune", str(tmp_path / "idx"), "--budget-ms", "1e-9", "--dataset", str(beir), *apply]
    assert main(args) == code
    assert capsys.readouterr().out.splitlines()[-1] == "best: none (nothing fits 1e-09 ms)"
    assert (tmp_path / "idx/manifest.json").read_bytes() == before


# AT7


def test_a_zero_budget_is_one_line_and_no_traceback(tmp_path, capsys):
    assert main(["tune", str(tmp_path / "idx"), "--budget-ms", "0"]) == 1
    err = capsys.readouterr().err
    assert "latency_ms must be" in err and err.count("\n") == 1


def test_missing_index_is_one_line_and_no_traceback(tmp_path, capsys):
    assert main(["search", str(tmp_path / "nope"), "q"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("laterank: error: ") and err.count("\n") == 1


def test_missing_extra_keeps_its_install_hint(tmp_path, docs, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "FlagEmbedding", None)  # None makes `import` fail
    args = ["build", str(tmp_path / "idx"), "--docs", str(docs), "--encoder", "bge-m3"]
    assert main(args) == 1
    err = capsys.readouterr().err
    assert 'pip install "laterank[bge-m3]"' in err and err.count("\n") == 1


@pytest.mark.usefixtures("no_model")
def test_a_missing_docs_file_exits_1(tmp_path, capsys):
    args = ["build", str(tmp_path / "idx"), "--docs", str(tmp_path / "no.jsonl"), "--encoder"]
    assert main([*args, "bge-m3"]) == 1
    assert "no.jsonl" in capsys.readouterr().err


def test_a_multi_line_message_is_printed_on_one_line(tmp_path, docs, monkeypatch, capsys):
    def boom(*args):
        raise OSError("first line\n  second line")

    monkeypatch.setattr(cli, "_make_encoder", boom)
    args = ["build", str(tmp_path / "idx"), "--docs", str(docs), "--encoder", "bge-m3"]
    assert main(args) == 1
    assert capsys.readouterr().err == "laterank: error: first line second line\n"


@pytest.mark.parametrize(
    "args",
    [
        ["tune", "{idx}", "--budget-ms", "20", "--dataset", ""],
        ["build", "{new}", "--docs", "", "--encoder", "bge-m3"],
        ["build", "{new}", "--docs", "{docs}", "--encoder", "bge-m3", "--pipeline", ""],
        ["search", "{idx}", "q", "--pipeline", ""],
    ],
    ids=["tune-dataset", "build-docs", "build-pipeline", "search-pipeline"],
)
def test_an_empty_value_is_refused_not_ignored(tmp_path, docs, encoder, capsys, args):
    _build(tmp_path / "idx", docs)
    capsys.readouterr()
    names = {"idx": tmp_path / "idx", "new": tmp_path / "new", "docs": docs}
    assert main([a.format(**names) for a in args]) == 1
    assert capsys.readouterr().err.splitlines()[-1].startswith("laterank: error: ")


def test_a_character_the_output_cant_show_doesnt_abort(tmp_path, encoder, monkeypatch):
    path = tmp_path / "greek.jsonl"
    path.write_text(json.dumps({"_id": "a", "text": "\u03b2 blocker"}) + "\n", encoding="utf-8")
    _build(tmp_path / "idx", path, "--pipeline", "exhaustive")
    out = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", out)
    assert main(["search", str(tmp_path / "idx"), "beta", "-k", "1"]) == 0
    out.flush()
    assert b"? blocker" in out.buffer.getvalue()


def test_an_empty_dataset_isnt_the_current_folder(beir, monkeypatch):
    monkeypatch.chdir(beir)  # it holds corpus.jsonl
    with pytest.raises(ValueError, match="no BEIR dataset named ''"):
        cli._dataset("", "test")


def test_main_runs_with_a_stdout_that_cant_be_reconfigured(tmp_path):
    with contextlib.redirect_stdout(io.StringIO()):
        assert main(["search", str(tmp_path / "nope"), "q"]) == 1


def test_a_missing_argument_is_a_usage_error():
    with pytest.raises(SystemExit) as e:
        main(["search"])
    assert e.value.code == 2


def test_a_bug_keeps_its_traceback(tmp_path, docs, monkeypatch):
    def boom(*args):
        raise RuntimeError("a bug")

    monkeypatch.setattr(cli, "_make_encoder", boom)
    with pytest.raises(RuntimeError, match="a bug"):
        main(["build", str(tmp_path / "idx"), "--docs", str(docs), "--encoder", "bge-m3"])


# AT8 (M5-3 AT4: six commands)


def test_help_lists_the_six_commands(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
    words = capsys.readouterr().out.split()
    assert all(c in words for c in ("build", "add", "delete", "search", "evaluate", "tune"))


def test_the_console_script_loads_main():
    assert entry_points(group="console_scripts")["laterank"].load() is main


# M5-3: add and delete


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _files(folder):
    return {p.relative_to(folder): p.read_bytes() for p in folder.rglob("*") if p.is_file()}


# M5-3 AT1


def test_add_puts_the_new_documents_last(tmp_path, docs, encoder, capsys):
    _build(tmp_path / "idx", docs)
    new = _jsonl(
        tmp_path / "new.jsonl",
        [{"_id": "n0", "text": "blue berry"}, {"_id": "n1", "text": "black cherry"}],
    )
    capsys.readouterr()
    assert main(["add", str(tmp_path / "idx"), "--docs", str(new)]) == 0
    assert capsys.readouterr().out == f"added 2 documents to {tmp_path / 'idx'}: 14 in total\n"
    assert Index.open(tmp_path / "idx")._docs.ids()[-2:] == ["n0", "n1"]
    assert main(["search", str(tmp_path / "idx"), "blue berry", "-k", "1"]) == 0
    assert capsys.readouterr().out.split()[2] == "n0"


# M5-3 AT2


def test_delete_removes_the_ids_without_the_model(tmp_path, docs, encoder, monkeypatch, capsys):
    _build(tmp_path / "idx", docs)
    monkeypatch.setitem(encoder_mod.ENCODERS, "fake", lambda info: pytest.fail("loaded"))
    capsys.readouterr()
    assert main(["delete", str(tmp_path / "idx"), "d3", "d5", "d5"]) == 0  # a repeat counts once
    assert capsys.readouterr().out == f"deleted 2 documents from {tmp_path / 'idx'}: 10 in total\n"
    ids = Index.open(tmp_path / "idx")._docs.ids()
    assert len(ids) == 10 and not {"d3", "d5"} & set(ids)


# M5-3 AT3


@pytest.mark.parametrize(
    ("args", "message", "loads"),
    [
        (["add", "{idx}", "--docs", "{clash}"], "already in the index", False),  # M6-1 AT1
        (["add", "{idx}", "--docs", "{empty}"], "empty.jsonl has no documents", False),
        (["add", "{idx}", "--docs", "{repeat}"], "repeat", False),
        (["add", "{idx}", "--docs", "{blank}"], "non-empty string 'id'", False),
        (["add", "{idx}", "--docs", "{missing}"], "no.jsonl", False),
        (["delete", "{idx}", "d0", "nope"], "not in the index", False),
        (["delete", "{idx}", *[f"d{i}" for i in range(12)]], "every document", False),
    ],
    ids=[
        "add-clash",
        "add-empty",
        "add-repeat",
        "add-blank-id",
        "add-missing-file",
        "delete-unknown",
        "delete-all",
    ],
)
def test_a_refusal_is_one_line_and_leaves_the_index_alone(
    tmp_path, docs, encoder, monkeypatch, capsys, args, message, loads
):
    _build(tmp_path / "idx", docs)
    before = _files(tmp_path / "idx")
    if not loads:
        monkeypatch.setitem(encoder_mod.ENCODERS, "fake", lambda info: pytest.fail("loaded"))
    names = {
        "idx": tmp_path / "idx",
        "clash": _jsonl(tmp_path / "clash.jsonl", [CORPUS[0]]),
        "empty": _jsonl(tmp_path / "empty.jsonl", []),
        "repeat": _jsonl(tmp_path / "repeat.jsonl", [{"_id": "n", "text": "a"}] * 2),
        "blank": _jsonl(tmp_path / "blank.jsonl", [{"_id": "", "text": "a"}]),
        "missing": tmp_path / "no.jsonl",
    }
    capsys.readouterr()
    assert main([a.format(**names) for a in args]) == 1
    err = capsys.readouterr().err
    assert err.startswith("laterank: error: ") and message in err and err.count("\n") == 1
    assert _files(tmp_path / "idx") == before
