# laterank beta: post text

Draft for the owner to publish (M6-2). Claude never posts it. Record where it went and its link
in `docs/dev/progress.md`.

---

**laterank 0.2: local late-interaction search, looking for beta testers (Nov 16-27)**

laterank is a small Python library for searching your own documents on your own machine with
late-interaction models (ColBERT-style or BGE-M3). A cheap first stage (BM25, dense, hybrid, SMVE
or MUVERA) picks about a hundred candidates, then exact MaxSim re-orders them.

```bash
pip install "laterank[colbert]"
laterank build ./idx --dataset scifact --encoder colbert \
    --hf-id mixedbread-ai/mxbai-edge-colbert-v0-32m
laterank search ./idx "statin side effects" -k 5
laterank evaluate ./idx --dataset scifact
```

On BEIR SciFact, its default pipeline (`bm25 > maxsim@100`) scored nDCG@10 0.7459 with
mxbai-edge-colbert-v0-32m (model commit bb13a29e), and the build took about a minute on an
Apple-silicon Mac once the model and dataset were downloaded.

It's a beta: please try it on your own documents between Nov 16 and 27 and tell me what broke,
was slow, or confused you: https://github.com/ujwal-jibhkate/laterank/issues/new?template=beta-report.md

Code and docs: https://github.com/ujwal-jibhkate/laterank
