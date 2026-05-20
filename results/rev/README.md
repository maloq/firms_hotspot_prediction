# Revision Experiment Outputs

Generated revision experiment outputs live here when they are produced locally.
Git tracks the real PNG plots and CSV tables from this tree, but ignores bulky
intermediate artifacts and duplicate `artifacts/linked_files` symlink paths.
Those linked artifact paths are deeply nested and can exceed Windows path
limits during checkout or sync.

Keep durable summaries in tracked documentation outside this directory, and
store bulky generated artifacts outside Git when they need to be shared.
