# acorns
 **A**nnotated **Cor**pus of **N**atural **S**igning

## About
This corpus provides 1000 hours of people using American Sign Language alongside a gold-standard, sentence-aligned interpretation and a silver-standard, word-aligned gloss. It is designed to be used for machine translation applications to and from English. It is compiled by Rany Tith and Lee Kezar.

## Formatting (Anticipated)
A global listing of video IDs and their meta information can be found in `videos_meta.csv`. Note that each video is also marked with `train`, `dev`, or `test` for model development purposes (used in the baseline methods). A blind set exists for model evaluation purposes.

For each video, you will find a folder with 3 files:

| file name | description | use |
|-----------|-------------|-----|
| `<id>_raw.mp4` | the video as it appears on YouTube | training on noisy real-world data |
| `<id>_norm.mp4` | a version of the video with clean background | training on simplified data |
| `<id>_captions.xml` | the gold-standard translation and the silver-standard gloss | training video-to-gloss (silver), gloss-to-English (gold), or video-to-English (end-to-end). |

## Statistics
| video class | num examples (%) | total duration (%) | avg video duration (stdev) | vocabulary size | average English words per second (stdev) | average signs per second |
|--|--|--|--|--|--|--|
| all | | | | | | |
| news | | | | | | |
| educational | | | | | | |
| religious | | | | | | |
| other | | | | | | |

## Methodology
### Normalized Video Construction
TBD

### Silver Standard Construction
TBD

## Baseline Performance
TBD 
### Human accuracy
| video class | n | IIA (kappa) | average accuracy |
|-------------|---|-------------|------------------|
| all | | | |
| news | | | |
| educational | | | |
| religious | | | |
| other | | | |

### Off-the-shelf neural translation models
| base model | training data | BLEU score |
|--------|-------------|------------|
| BERT | noisy, end-to-end | TBD |
| BERT | normalized, end-to-end | TBD |
| BERT | normalized, latent translation | TBD |
| BERT | normalized, face encoding + latent translation | TBD |
| BERT | normalized, face encoding + hand encoding + latent translation | TBD |

