# Input reconstruction and access

Reproducing the figures, recalculating scientific estimates, refitting models and rebuilding data from provider downloads are different operations. The package alone supports replay from aggregate results. Scientific evaluation requires the 48 registered inputs, and TB task preparation requires the permitted expression exports and metadata. Prior local verification used retained inputs and does not establish a complete route from provider downloads to historical model predictions.

INPUT_RECONSTRUCTION_STATUS.csv lists every file, its upstream route, dependencies, located historical generator and unverified steps. Historical project paths document provenance; their presence does not mean that the generator is bundled or that the route has been tested. All 48 retained inputs matched their fixed hashes on 8 October 2026. Four participant-component maps and six TB task files were also reconstructed byte-for-byte from retained upstream files. Those checks do not pre-authorize sharing.

| Input family | Files | Located route | Scope of exact historical reconstruction |
| --- | --- | --- | --- |
| Saved predictions | 4 | Historical model-fitting and calibration code | Identify the precise historical runs, component-aware partitions and environment; regenerate and compare predictions, not just downstream plots |
| Participant component maps | 4 | Historical union of participant identifiers, cross-platform links and duplicate links | Four maps reproduced byte-for-byte using retained metadata and historical link tables. The duplicate-correlation table is empty; no correlation-based links contributed. External acquisition of the exact platform-pair table remains conditional |
| CRC MetaPhlAn4 matrix and metadata | 2 | Historical species aggregation and development-sample selection | Historical preparation is located, but a separate development-only staging recipe has not been verified; sealed-validation paths are not public acquisition instructions |
| Earlier CRC matrix and metadata | 2 | Source selection in curatedMetagenomicData located | Complete export and schema recipe has not been recovered |
| TB expression exports and combined metadata | 30 | curatedTBData 2.8.0 export script | Verify provider/package version, selected assay and metadata serialization |
| TB task matrices and metadata | 6 | Sample filtering, scale transformation and gene-intersection script | Six files reproduced byte-for-byte from retained expression exports and metadata. A preparation command is supplied in reproduction_code/preparation/README.md; fresh provider export remains unverified |


## Available operations

To reproduce figures and numerical tables, follow the presentation command in README.md. It uses aggregate_outputs and needs no participant-level inputs. To recalculate calibration meta-analysis alone, use reproduction_code/recompute_E2.R with TableS10_E2_cohort_inputs.csv.

For the saved-prediction evaluation, obtain permitted files in the manifest directory structure and run science.py as documented. Conditional requests can be sent to Xiaoyuan Bao (xybao@pku.edu.cn) under DATA_REQUEST_PROCEDURE.md. The fixed hash checks must remain enabled. Even if byte differences appear to arise only from serialization, content, sample and feature order, labels and prediction values require comparison. Replacing the accepted manifest creates a new input version rather than demonstrating byte-identical recovery of the old one.

Readers who wish to reconstruct inputs from provider data should follow the known source routes and examine the dependencies above. TRAINING_AND_INPUT_ROUTES.md in reproduction_code explains the available model functions. No full-data download or model retraining was performed for this editorial revision. The untested steps describe the scope of independent historical reconstruction, not additional experiments required for this preparation release.

Third-party inputs are not publicly repackaged. This author decision does not mean that all providers prohibit reuse. Requests depend on the rights and actual availability of each file, and an author email address alone does not guarantee access. Known sources, versions and instructions are supplied without a promise that external services will remain available.
