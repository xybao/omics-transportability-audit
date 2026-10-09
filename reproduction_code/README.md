# Running the analysis

The evaluation used Python 3.13.5, R 4.5.2 and metafor 5.2-1. Install requirements-evaluation-verified.txt in Python 3.13 and consult environment/R_sessionInfo.txt for the recorded R session. install_R_dependencies.R installs current packages; it does not pin the historical R environment.

science.py requires the 48 files specified in ACCEPTED_INPUTS.json and the two auxiliary GEO metadata extracts listed in ../SOURCE_RECORDS.md. They are not included, and the source links do not directly supply every derived input. DATA_ACCESS.md and DATA_REQUEST_PROCEDURE.md in the package root explain acquisition and conditional requests.

Run from this directory:

```text
python science.py --input-dir <input_directory> --output-dir <new_output_directory> --rscript <Rscript_executable> --r-library <R_library>
```

The output directory must not already exist. Omit --r-library if metafor is installed in the normal R library. Path and hash checks prevent incompatible inputs from entering the calculation. Scientific runtime assertions are retained; these are part of the analysis rather than editorial acceptance tests.

runner.py executes the four tasks, while aggregate.py and meta_analysis.R produce meta-analysis results. supplementary_analysis.py, cohort_table.py, scenario_annotations.py, clinical_summary.py and exploratory_analysis.py produce supplementary estimates. The pilot directory contains imported split, model and calibration functions. TRAINING_AND_INPUT_ROUTES.md describes their use and distinguishes evaluation of saved predictions from historical model fitting.

## Operations without the 48 inputs

The aggregate calibration summaries can be recalculated with:

```text
Rscript recompute_E2.R ../tables/TableS10_E2_cohort_inputs.csv meta_recomputed.csv
```

Synthetic examples are available through:

```text
python demo.py --output-dir <new_demo_directory>
python estimand_examples.py
```

The second command writes results/estimand_examples.json without editing the manuscript. A saved copy is in ../environment/estimand_examples.json. These examples explain the methods but do not reproduce empirical results.

## Figures and tables

After the scientific calculation, export its results with:

```text
python export_manuscript.py --analysis-dir <completed_analysis_directory> --output-dir <new_presentation_directory>
```

For replay of the supplied results, use ../aggregate_outputs as --analysis-dir and add --omit-source-record-tables. This mode exports 24 numerical tables; S20 and S21 remain omitted from the public distribution. Full export without the option requires their source result CSVs. Install requirements-presentation.txt in a separate environment; it includes matplotlib for figure export. The estimation dependencies are recorded separately. Figure 3 shows thresholds, Figure 4 prediction sets, Figure S2 prior correction and Figure S3 local intercept updating. Table numbering and scientific result identifiers are preserved.
