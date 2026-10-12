"""Export the raw texts smve-lab's BM25 tokenized, next to its embeddings, for the golden tests.

laterank's golden BM25 tests need the exact document and query strings smve-lab used. smve-lab
reads them through Hugging Face `datasets`, which laterank doesn't depend on, so this runs once
inside smve-lab's environment:

    cd ~/Projects/smve-lab && uv run python ~/Projects/laterank/scripts/export_golden_texts.py

It writes artifacts/embeddings/<name>_bgem3/{docs,queries}_texts.json: a JSON list of strings,
aligned with the {docs,queries}_ids.json already there (checked before writing).
"""

import json
import sys

from smve_lab.datasets import emb_dir, load_texts


def main(names: list[str]) -> None:
    for name in names:
        folder = emb_dir(name)
        doc_ids, doc_texts, query_ids, query_texts = load_texts(name)
        for side, ids, texts in (("docs", doc_ids, doc_texts), ("queries", query_ids, query_texts)):
            stored = json.loads((folder / f"{side}_ids.json").read_text(encoding="utf-8"))
            if stored != ids:
                sys.exit(f"{name}: {side} ids differ from {side}_ids.json; not writing")
            out = folder / f"{side}_texts.json"
            out.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
            print(f"wrote {out} ({len(texts)} texts)")


if __name__ == "__main__":
    main(sys.argv[1:] or ["scifact"])
