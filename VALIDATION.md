# Validation

Executed October 27, 2026 on macOS 26.6.2, arm64, Python 3.13.7, CPU. Dependencies were installed from `requirements-macos-arm64.txt` with hash checking. `uv pip check` checked 83 packages and reported all compatible. The model and tokenizer used the pinned revision in `common.py`.

All four runnable commands exited successfully: `example.py`, `sweep_real.py`, `run_real.py`, and `viz_data.py`. The sweep and threshold comparison ran with `HF_HUB_OFFLINE=1` after the initial model download. Warnings were not globally suppressed; upstream deprecation notices and model loading progress may appear on stderr.

## Automated checks

`RUN_MODEL_TESTS=1 .venv/bin/python -m unittest discover -s tests -v`: **5 tests passed**. Checks cover fixture length and boundaries, initial-chunk exclusion from scoring, SINGLE's absence of annotated transitions, extra split accounting, expected snippet lengths, full input preservation, and final chunk token lengths.

## Observed results

The snippet printed the following stdout, matching the recorded article output:

```text
0 303 On-call engineers rotate weekly, starting Monday a
1 149 If you swap shifts, update the rotation calendar b
2 374 Travel expenses are reimbursed only with an itemiz
3 387 Company laptops must use full-disk encryption at a
```

The sweep reproduced every recorded chunk count and prefix across all 18 combinations. Corrected transition scores are:

| Document | Buffer | Percentile | Chunks | Topic transitions found | Extra splits |
| --- | ---: | ---: | ---: | --- | ---: |
| MIXED | 0 | 80 | 5 | 2/2 | 2 |
| MIXED | 0 | 90 | 3 | 1/2 | 1 |
| MIXED | 0 | 95 | 2 | 0/2 | 1 |
| MIXED | 1 | 80 | 5 | 1/2 | 3 |
| MIXED | 1 | 90 | 3 | 1/2 | 1 |
| MIXED | 1 | 95 | 2 | 0/2 | 1 |
| MIXED | 2 | 80 | 5 | 0/2 | 4 |
| MIXED | 2 | 90 | 3 | 0/2 | 2 |
| MIXED | 2 | 95 | 2 | 0/2 | 1 |
| SINGLE | 0, 1, or 2 | 80 | 5 | N/A | 4 |
| SINGLE | 0, 1, or 2 | 90 | 3 | N/A | 2 |
| SINGLE | 0, 1, or 2 | 95 | 2 | N/A | 1 |

For both documents, the default percentile, standard deviation, interquartile, and gradient runs produced 2, 1, 1, and 2 chunks respectively. The default MIXED percentile break starts with `Late reports need`; it misses both annotated transitions. The rounded distance arrays matched the original experiment.

The diagram diagnostic reproduced the 80th-percentile threshold **0.876**, fixed-window starts **[0, 276, 591, 896]**, and semantic starts **[0, 304, 454, 829]**. The 60-token window breaks between `Hotel stays are` and `capped at the city rate`. Regex segment counts and maximum unbuffered sentence wordpieces were **19 / 18** for the handbook and **18 / 17** for SINGLE. The sentence-length check does not measure buffered sentence groups.

`handbook.txt` is exactly 1,216 ASCII/UTF-8 bytes with no trailing newline. SHA-256:

```text
802a32bc32b2bda10bb7e9a331b1a4dedcbdabb70eeb02718463dca4521e746e
```

This verifies the sample experiment on one platform. It does not establish retrieval accuracy, production suitability, or cross-platform numerical identity. The original Linux environment was not rerun here.
