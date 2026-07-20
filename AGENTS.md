# File access rules

Never read, search, or modify the following paths unless the user explicitly asks.

Ignored paths:

- data/
- datasets/
- logs/
- dist/
- output/
- __pycache__/
- */__pycache__/
- **/__pycache__/
- *.csv
- *.parquet
- *.pdf

When searching the repository:

- Only search under tracesynth/ and /configs and /scripts directories.
- Only inspect tests/ if debugging tests.
- Do not use global grep across ignored directories.