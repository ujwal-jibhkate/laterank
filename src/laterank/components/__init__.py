"""Components: first stages and rerankers (middle layer).

Each one wraps kernels around arrays read from storage. They know nothing about pipelines or
index folders as a whole; orchestration wires them together.
"""
