# File access rules

Never read, search, or modify the following paths unless the user explicitly asks.

Ignored paths:

- data/
- datasets/
- logs/
- dist/
- output/
- *.csv
- *.parquet
- *.pdf

When searching the repository:

- Only search under src/
- Only inspect tests/ if debugging tests.
- Do not use global grep across ignored directories.