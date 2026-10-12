# Golden-test data

`scifact-test-qrels.tsv` is a byte-for-byte copy of `test.tsv` from the Hugging Face dataset
[`BeIR/scifact-qrels`](https://huggingface.co/datasets/BeIR/scifact-qrels), revision
`2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029`, SHA-256
`0864bb985e0ca2367ba217977e72004d549054b2b06666ed9d4825ac7c21284c`. It is the file smve-lab
evaluated against: 300 queries, 339 relevance judgments.

License: CC BY-SA 4.0, as stated on the BEIR dataset card. The data comes from SciFact
(Wadden et al., 2020) as packaged by BEIR (Thakur et al., 2021). It is test data only: `tests/`
is not part of the laterank package.
