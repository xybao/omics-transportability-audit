# Source-record metadata

Scientific evaluation uses two auxiliary GEO metadata extracts in addition to the 48 files in ACCEPTED_INPUTS.json. They are excluded from this public distribution. Before calling science.py, obtain the source records under the provider terms and place the metadata at the paths in AUXILIARY_SOURCE_MANIFEST.json. A missing-file check stops the command before it creates an analysis output directory.

| Accession | GEO record | Provider matrix directory | Retained metadata SHA-256 |
| --- | --- | --- | --- |
| GSE89403 | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE89403 | https://ftp.ncbi.nlm.nih.gov/geo/series/GSE89nnn/GSE89403/matrix/ | 5e29406660c7a269fac2b7c8e83aa2284444dd5ef46e18eae47790035e45f156 |
| GSE73408 | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE73408 | https://ftp.ncbi.nlm.nih.gov/geo/series/GSE73nnn/GSE73408/matrix/ | de867ee299df90c872db79b96c9f612800c5e87086fdfa5ff4f3f9cf5fe4fa68 |

The historical extracts retain series and sample metadata lines, including !Sample_characteristics_ch1 and public sample accessions. The GEO series-matrix metadata can supply these fields, but current downloads, transformations and line endings have not been verified here against the retained hashes. Do not rename a compressed matrix as a metadata text file. The directory links are more robust than guessing a platform-specific filename; select the file documented by the provider. The GSE89403 historical record uses GPL11154. Provider pages presented a browser check during this preparation, so these links are not described as freshly download-tested.

Where an exact extract is needed, a request may be sent to xybao@pku.edu.cn. Any provision remains conditional on source permissions and availability. Do not suppress scientific assertions to make an incompatible source file pass. This documentation does not add a requirement that the authors retrain models or guarantee the continued availability of the provider.

The source-record tables S20 and S21 and their two result CSVs are also omitted. They contain public sample identifiers and source clinical labels. Their omission is a conservative redistribution choice, not a finding that GEO prohibits all redistribution. Source interpretation summaries remain in Tables S17 to S19. Public table and figure replay does not need these metadata files and uses --omit-source-record-tables. Full evaluation and export, after obtaining the necessary inputs, retain the code to regenerate S20 and S21.
