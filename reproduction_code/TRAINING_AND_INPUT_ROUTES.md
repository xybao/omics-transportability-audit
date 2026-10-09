# Input acquisition and independent model fitting

This guide describes the code available for independent fitting and the inputs it expects. Models were not retrained for the CBC editorial revision, and a complete reconstruction of the historical predictions has not been independently verified.

## Available procedures

| Operation | Input and entry point |
| --- | --- |
| Recreate the published figures and numerical tables | Supplied aggregate_outputs; export_manuscript.py as documented in README.md |
| Recalculate scientific estimates from saved predictions | 48 registered files; science.py with the fixed input contract |
| Prepare TB task matrices | Provider-derived expression exports and combined metadata; preparation/prepare_tb_tasks.py |
| Fit models independently | Task matrices, metadata and participant-component mappings; historical functions in pilot/bench3.py and pilot/bench.py |

SOURCE_DOWNLOAD_LINKS.md and SOURCE_ACCESS_CATALOG.md in the package root specify provider resources and cohort accessions. INPUT_RECONSTRUCTION_STATUS.csv records dependencies per file. Provider files are not identical to all derived analytical inputs. Historical paths in that table are provenance references, not bundled executable commands. Readers obtain data under provider terms; future availability is outside the authors' control.

## Model fitting

pilot/bench3.py contains fit_method, source_split and run_unit. Its active representation is global. Historical rank and batch branches do not support the manuscript's main inference. The recorded design uses ten seeds (0 to 9), held-out cohort evaluation, and source fractions 0.60/0.15/0.10/0.15 for TRAIN/CAL_P/CAL_C/ID. Participant components are kept together. Setting FULL_SOURCE=1 changes this design and must not be used as a substitute.

The fit_method function specifies LASSO with eight C values from 10**-2.5 to 10**0.5 and source-group cross-validation; RF with 500 trees; GBDT with 300 estimators, learning rate 0.03 and eight leaves; and MLP with layers (128, 32), alpha 0.1 and 150 iterations. Consult that function for the complete parameters and training-only feature filtering. Other helper functions in bench.py include historical alternatives and must not be substituted without documenting a new analysis. TabPFN code is not one of the four models retained in this manuscript.

The historical command-line signature is `python pilot/bench3.py TASK 10 lasso,rf,gbdt,mlp`. This signature documents the historical entry point; it is not a complete, ready-to-run reconstruction recipe. It expects the loader's task files, participant mapping paths and directory layout, and writes runs/preds3_TASK.parquet beneath its configured root. Running it alone does not recover missing preprocessing or establish equivalence to the registered prediction files. Do not overwrite accepted inputs or disable their hash checks.

## Environment and comparison

requirements-evaluation-verified.txt and environment/R_sessionInfo.txt describe the saved-prediction evaluation environment; requirements-presentation.txt describes figure export. They do not certify the full historical fitting environment. Refitted predictions may differ because of library versions or numerical behaviour, and serialized files may differ even when their contents agree. Record dependencies, sample and feature order, labels, partitions and prediction values when conducting independent fitting. A changed prediction file defines a new input version unless equivalence is established explicitly.

The package offers scientific code and the documented provider routes without claiming a newly verified end-to-end fitting workflow. Conditional requests for retained files are optional and subject to the source terms; they are not a guarantee of access to every input.
