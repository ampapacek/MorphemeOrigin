# Legal Corpus Analysis

This directory contains the tracked files needed to reproduce the legal-corpus analysis workflow inside the `MorphemeOrigin` repository.

Contents:
- `law_corpus_analysis.ipynb`: notebook for running prediction and plotting yearly trends
- `train_predict_runner.py`: runner used by the notebook
- `download_esbirka_laws.py`: downloader for source law texts from e-Sbírka
- `build_law_corpus_by_year.py`: helper for building a sampled per-year corpus

Notes:
- The notebook first looks for an existing local `MorphemeOrigin` checkout in the current directory or its parents. If none is found, it clones the repository into the working directory, preferring the public HTTPS URL and cleaning up incomplete clone directories before retrying.
- Morphological segmentation and morph-type classification are not included in this directory.
- Before running prediction on your own corpus, prepare `laws_in_years_annotation.tsv` in the indented MorphOrigin annotation format expected by the notebook.
