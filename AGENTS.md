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

- Only search under tracesynth/ and scripts/tool_use_data_gen.py unless the user explicitly asks.
- Only inspect tests/ if debugging tests.
- Do not use global grep across ignored directories.