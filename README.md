# Omics classifier transportability audit

Code and reproduction materials for Calibration and threshold transportability of omics classifiers across cohorts. Version 1.0.4 accompanies the manuscript. Zenodo archive identifier: https://doi.org/10.5281/zenodo.23236395. DOI registration occurs when the record is published.

Authors: Xiaoyuan Bao, Jiao Shan, Yulong Cao, Jingsi Jiang and Haoyu Wang. Correspondence: xybao@pku.edu.cn.

Download all three ZIP files from the same repository record and extract them into the same parent directory. They share the root omics-transportability-audit-1.0.4 and contain different files. Ordinary ZIP extraction is sufficient; these are not split-volume archives.

| Archive | Contents |
| --- | --- |
| 01_source_code_1.0.4.zip | Python and R code, dependency records, input manifests, source routes and documentation |
| 02_aggregate_results_1.0.4.zip | 42 analysis-result CSVs, 24 exported numerical CSVs, static table notes and environment records |
| 03_figures_1.0.4.zip | Four main and three supplementary figures in available PDF, PNG and SVG formats |

## Recreate the figures and tables

Use Python 3.13.5 and run from the extracted package root:

```text
python -m pip install -r reproduction_code/requirements-presentation.txt
python reproduction_code/export_manuscript.py --analysis-dir aggregate_outputs --output-dir presentation_replay --omit-source-record-tables
```

Choose an output directory that does not already exist. This recreates four main figures, three supplementary figures and 24 numerical CSV tables from the supplied analysis results. Table S9 and the table index are supplied as static documentation. Tables S20 and S21 are not redistributed because they reproduce sample-level source labels. Their omission is explicit and does not change the numbering of the remaining tables. The output reports both omissions. All 24 retained numerical tables have the same bytes as the CBC v10 package.

This operation reproduces the presentation of existing results. It does not estimate model performance again from participant-level data or refit models.

## Scientific evaluation

The scientific entry point needs the 48 registered analytical inputs and two auxiliary GEO metadata extracts. These files are not included. SOURCE_RECORDS.md identifies the two additional resources; DATA_ACCESS.md, EXACT_INPUTS.md and the reconstruction guide describe the 48 inputs. The code README provides the full evaluation command. When all source inputs have been obtained and the analysis completed, the normal export command without --omit-source-record-tables also writes Tables S20 and S21.

The package preserves the scientific methods and estimates of v10. The new export option only excludes two tables from public replay, and an early input check gives a clear message if the auxiliary metadata are missing. Historical model refitting and full reconstruction from fresh provider downloads have not been verified. Synthetic examples and the aggregate calibration meta-analysis have separate commands in reproduction_code/README.md.

## Rights and citation

Original source code and its software documentation use the MIT License. This does not relicense third-party materials or apply a new blanket licence to the study's result tables and figures. Read LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md. No original expression matrix, processed participant matrix, saved participant predictions or complete provider metadata extract is included. Provider links are access routes, not a promise of permanent availability.

Use the version-specific DOI shown on the Zenodo record once it is published. The manuscript DOI, if later assigned, identifies a different object. The public code archive contains no manuscript, cover letter, internal review, editorial acceptance script or delivery log.

## Code implementation and language polishing

OpenAI Codex assisted with code implementation according to the study design, analytical methods and algorithms specified by the authors, and with English language polishing. The authors take full responsibility for the code, analyses and manuscript.

## Funding

This research received no specific grant from any funding agency in the public, commercial or not-for-profit sectors.
