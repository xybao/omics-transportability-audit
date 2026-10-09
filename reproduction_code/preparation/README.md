# Preparing the TB task files

First obtain the registered TB expression exports and tb_meta.parquet under their source terms. From the package root, run:

```console
python reproduction_code/preparation/prepare_tb_tasks.py --input-dir PATH_TO_TB_EXPORTS --output-dir NEW_OUTPUT_DIRECTORY
```

Use the recorded evaluation dependencies and a new output directory; do not overwrite accepted inputs. On 8 October 2026, applying the historical transformation to retained exports produced six files with hashes identical to the registered inputs. The supplied command uses separate input and output directories and excludes unrelated exploratory calculations. This check verifies the transformation, not a fresh download or export from curatedTBData. Serialization differences require a separate comparison of contents rather than removal of the registered hash checks.
