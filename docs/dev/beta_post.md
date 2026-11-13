# laterank beta: post text

Drafts for the owner to publish (M6-2). Claude never posts them. Record where each went and its
link in `docs/dev/progress.md`. Source of the number: M5-3 log (SciFact, mxbai at `bb13a29e`,
`bm25 > maxsim@100`, build 64 s on an Apple-silicon Mac).

## X thread (5 posts, each at most 280 characters; X counts each link as 23)

**1/5**
I've released laterank, a small open-source Python library for searching your own documents on your own machine with ColBERT-style models.

It's in beta, and I'd be grateful for a few people to try it. A short thread on what it does 🧵

**2/5**
Why: most embedding search squeezes each document into one vector, and the details get lost.

ColBERT-style models keep one vector per token and match them one by one with the query (MaxSim). That's more accurate, but slow to run over every document.

**3/5**
How laterank handles it: a cheap first stage (BM25, dense or hybrid) picks ~100 candidates, then exact MaxSim re-ranks only those.

The index is a plain folder of NumPy arrays and JSON. No server, no database.

**4/5**
Try it:

pip install "laterank[colbert]"
laterank build ./idx --docs my_docs.jsonl --encoder colbert --hf-id mixedbread-ai/mxbai-edge-colbert-v0-32m
laterank search ./idx "your question"

On BEIR SciFact this scores nDCG@10 0.746.

**5/5**
If you have 15 minutes between Nov 16 and 27, please try it on your own documents and tell me what broke, was slow, or was confusing. Every report helps.

Code and quick start: https://github.com/ujwal-jibhkate/laterank

Thank you!

## LinkedIn

I've released laterank, a small open-source Python library for searching your own documents on your own machine, and I would be grateful for your help testing it.

What it is: a local search library for late-interaction models (ColBERT-style or BGE-M3).

Why: most embedding search turns each document into a single vector, which loses detail. Late-interaction models keep one vector per token (roughly per word) and compare them one by one with the query (MaxSim). That is more accurate, but too slow to run over every document.

How it works: a cheap first stage (BM25, dense or hybrid) picks about a hundred candidates, and exact MaxSim re-ranks only those. The index is a plain folder of NumPy arrays and JSON, with no server or database. You can measure any pipeline on your own labelled queries and add or delete documents later.

How to use it:
pip install "laterank[colbert]"

From the command line:
laterank build ./idx --docs my_docs.jsonl --encoder colbert --hf-id mixedbread-ai/mxbai-edge-colbert-v0-32m
laterank search ./idx "your question"

From Python:
from laterank import ColBERTEncoder, Index
docs = [{"id": "a", "text": "Statins lower cholesterol but can cause muscle pain."}, {"id": "b", "text": "Tea is made from the leaves of a plant."}]
encoder = ColBERTEncoder("mixedbread-ai/mxbai-edge-colbert-v0-32m")
index = Index.build("./my_index", docs, encoder, bm25=True, pipeline="bm25 > maxsim@100")
for hit in index.search(encoder.encode_query("do statins hurt muscles?"), k=1):
    print(hit.id, hit.text)

On BEIR SciFact this scores nDCG@10 0.7459 with mxbai-edge-colbert-v0-32m, and the index builds in about a minute on an Apple-silicon Mac.

It is in beta. If you have 15 minutes between Nov 16 and 27, please try it on your own documents and tell me what broke, was slow, or was confusing:
Report a problem: https://github.com/ujwal-jibhkate/laterank/issues/new?template=beta-report.md
Code and quick start: https://github.com/ujwal-jibhkate/laterank

Thank you! 