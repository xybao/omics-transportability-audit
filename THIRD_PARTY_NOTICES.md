# Third-party sources and reuse terms

These notes record source information checked from 26 to 28 September 2026, with resource pages revisited on 8 October 2026. They do not grant a new licence or certify every downstream use.

| Material | Source statement or verification | Scope |
| --- | --- | --- |
| curatedTBData 2.8.0 | Bioconductor lists MIT + file LICENSE: https://bioconductor.org/packages/release/data/experiment/html/curatedTBData.html | Preserve package notices. A current upstream LICENSE and MIT text were retrieved on 2026-09-28; they do not establish the identity of every historical source dataset. Review source-study conditions for the data being used. |
| curatedMetagenomicData 3.20.0 | Bioconductor lists Artistic-2.0: https://bioconductor.org/packages/release/data/experiment/html/curatedMetagenomicData.html | A package licence is not a blanket grant over all source records. This resource is distinct from the main CRC MetaPhlAn4 source. |
| CRC profiles and metadata | https://zenodo.org/records/15069069 identifies an open dataset and its files | The retrieved licence field was not readable; API verification also failed. Do not infer a specific Creative Commons licence. Obtain data through the source link. |
| GEO molecular data and metadata | https://www.ncbi.nlm.nih.gov/home/about/policies/ | NCBI places no restrictions of its own on molecular-data use or distribution, but notes that third-party rights may exist and are not transferred by NCBI. Retain attribution and accession provenance. |
| Third-party software dependencies | See reproduction_code/requirements-evaluation-verified.txt and the recorded R session | Install dependencies from their providers under their own licences. No dependency binaries are newly bundled by this edit. |


The MIT License covers only the authors' original code and software documentation. Third-party materials retain their own terms. The authors have chosen not to publish an archive of the 48 analytical inputs; this does not imply that every provider forbids redistribution. Users should follow the original provider routes and applicable terms.

The repository preparation in version 1.0.4 excludes the two complete GEO metadata extracts and the S20/S21 source-record tables. Provider licence notices are retained as notices, not repackaged research data. LICENSE_SCOPE.md distinguishes the code licence from rights in analytical results and source-derived content.
