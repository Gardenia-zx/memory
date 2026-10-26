"""Read-only retrieval diagnostics for the LoCoMo smoke baseline.

The package intentionally lives outside ``src`` and never calls a model. It
rebuilds the exact benchmark snapshot in memory, replays the lexical retriever
and reports why evidence was missed. Production code and tests are untouched.
"""
