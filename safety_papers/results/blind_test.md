# Blind test: compute estimates in H100e-years (TPP)

Authors' figures: the blind test's readings, factors put back to TPP (blind.tpp). Estimates: the components of each answer converted by h100e.py, before calibration (the model's accelerator-hours as given, FLOP at 40% utilization) unless a section says otherwise. Scored: papers where the arm answered and did not see the compute. Typical error: 10^(mean |log10 estimate / authors' figure|). Sizes in H100e-hours; dataset.csv holds H100e-years.

## sonnet-blind: the reported experiments (686 papers whose authors state all or part of their compute; 12 saw it, 0 refused or failed)

|  | papers | median est / true | within x1.5 | within x3 | within x10 | typical error | correlation of logs | slope |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| before calibration | 686 | x0.34 | 17% | 40% | 72% | x5.8 | 0.74 | 0.71 |
| old instructions (H100e by the model) | 672 | x1.00 | 20% | 47% | 79% | x4.2 | 0.76 | 0.71 |

### By year

|  | papers | median est / true | within x1.5 | within x3 | within x10 | typical error | correlation of logs | slope |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2020 | 22 | x0.76 | 23% | 55% | 91% | x3.4 | 0.82 | 0.76 |
| 2021 | 56 | x0.35 | 14% | 46% | 84% | x4.3 | 0.86 | 0.68 |
| 2022 | 53 | x0.36 | 15% | 34% | 74% | x6.4 | 0.74 | 0.60 |
| 2023 | 97 | x0.35 | 14% | 36% | 70% | x6.0 | 0.77 | 0.75 |
| 2024 | 162 | x0.32 | 20% | 41% | 68% | x5.7 | 0.74 | 0.79 |
| 2025 | 296 | x0.30 | 17% | 40% | 72% | x6.4 | 0.71 | 0.71 |

Old instructions (Sonnet), same authors' figures:

|  | papers | median est / true | within x1.5 | within x3 | within x10 | typical error | correlation of logs | slope |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2020 | 22 | x2.64 | 18% | 45% | 64% | x6.2 | 0.67 | 0.60 |
| 2021 | 52 | x1.12 | 27% | 44% | 77% | x3.9 | 0.75 | 0.68 |
| 2022 | 51 | x1.25 | 18% | 37% | 73% | x5.7 | 0.80 | 0.74 |
| 2023 | 98 | x1.26 | 17% | 47% | 78% | x4.6 | 0.75 | 0.67 |
| 2024 | 159 | x0.87 | 22% | 48% | 81% | x3.9 | 0.75 | 0.76 |
| 2025 | 290 | x0.91 | 19% | 48% | 82% | x3.9 | 0.78 | 0.72 |

### By the accelerator the authors name

|  | papers | median est / true | within x1.5 | within x3 | within x10 | typical error | correlation of logs | slope |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A100 class | 264 | x0.26 | 17% | 36% | 68% | x6.6 | 0.74 | 0.72 |
| GeForce RTX 20, GTX | 67 | x0.41 | 19% | 43% | 82% | x4.2 | 0.76 | 0.66 |
| GeForce RTX 30-50 | 56 | x0.40 | 16% | 41% | 79% | x5.2 | 0.69 | 0.79 |
| H100 or newer | 91 | x0.21 | 16% | 35% | 64% | x8.7 | 0.57 | 0.69 |
| V100, T4, TPU v2-3, workstation | 195 | x0.46 | 19% | 48% | 78% | x4.7 | 0.81 | 0.75 |
| none named | 13 | x0.45 | 0% | 23% | 62% | x6.2 | 0.75 | 0.82 |

### By route (how most of the estimate was reached)

|  | papers | median est / true | within x1.5 | within x3 | within x10 | typical error | correlation of logs | slope |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hours named | 219 | x0.52 | 23% | 50% | 81% | x4.0 | 0.80 | 0.68 |
| hours assumed | 252 | x0.37 | 15% | 38% | 75% | x5.3 | 0.79 | 0.70 |
| FLOP | 215 | x0.19 | 14% | 32% | 60% | x9.7 | 0.70 | 0.79 |

### By size of the authors' figure, H100e-hours

|  | papers | median est / true | within x1.5 | within x3 | within x10 | typical error | correlation of logs | slope |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| <1 | 68 | x1.21 | 21% | 46% | 78% | x5.1 | 0.39 | 0.44 |
| 1-30 | 180 | x0.64 | 21% | 49% | 81% | x4.1 | 0.35 | 0.76 |
| 30-1000 | 340 | x0.26 | 17% | 39% | 71% | x6.0 | 0.44 | 0.89 |
| >1000 | 98 | x0.12 | 8% | 24% | 56% | x10.8 | 0.21 | 0.35 |

### Calibration

Hours: on papers whose estimate is mostly accelerator-hours the authors' figure is a median x2.18 the estimate; h100e.py multiplies estimated hours by 2.2 (stated hours stay).

FLOP: papers whose estimate is at least 80% FLOP, by the kind of work holding most of it; the utilization that makes the median estimate equal the authors' figure, and the one h100e.py uses:

| kind | papers | median est / true at 40% | utilization for x1 | h100e.py |
|---|---:|---:|---:|---:|
| train | 105 | x0.25 | 0.100 | 0.1 |
| forward | 34 | x0.08 | 0.033 | 0.033 |
| generate | 50 | x0.06 | 0.026 | 0.026 |
| mixed | 14 | x0.18 | 0.071 | 0.071 |

After calibration:

|  | papers | median est / true | within x1.5 | within x3 | within x10 | typical error | correlation of logs | slope |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| every paper | 686 | x1.01 | 20% | 48% | 81% | x4.1 | 0.77 | 0.73 |
| 2020 | 22 | x1.66 | 27% | 45% | 86% | x3.5 | 0.82 | 0.77 |
| 2021 | 56 | x0.77 | 32% | 64% | 88% | x3.1 | 0.86 | 0.68 |
| 2022 | 53 | x0.96 | 19% | 47% | 74% | x5.2 | 0.73 | 0.58 |
| 2023 | 97 | x0.97 | 26% | 51% | 77% | x4.2 | 0.78 | 0.77 |
| 2024 | 162 | x0.97 | 18% | 41% | 83% | x4.1 | 0.76 | 0.78 |
| 2025 | 296 | x1.09 | 18% | 47% | 81% | x4.2 | 0.76 | 0.72 |

### Dead ends (calibrated estimates)

39 papers state only a whole-project figure (deleted): whole project / estimate of the reported experiments: median x4.62, quartiles x1.24-x16.10. With x2.5:

|  | papers | median est / true | within x1.5 | within x3 | within x10 | typical error | correlation of logs | slope |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| x2.5 x estimate | 39 | x0.54 | 13% | 44% | 79% | x4.3 | 0.38 | 0.39 |

23 targets state both all reported experiments and the whole project: whole / reported median x2.65, quartiles x1.59-x5.00.

### Assumed accelerators

| year | papers | mean share of hours-based compute on an assumed accelerator |
|---|---:|---:|
| 2020 | 23 | 65% |
| 2021 | 58 | 62% |
| 2022 | 57 | 48% |
| 2023 | 103 | 41% |
| 2024 | 168 | 38% |
| 2025 | 316 | 31% |

## opus-blind: the reported experiments (676 papers whose authors state all or part of their compute; 22 saw it, 4 refused or failed)

|  | papers | median est / true | within x1.5 | within x3 | within x10 | typical error | correlation of logs | slope |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| before calibration | 676 | x0.46 | 20% | 45% | 78% | x4.8 | 0.78 | 0.78 |

### By year

|  | papers | median est / true | within x1.5 | within x3 | within x10 | typical error | correlation of logs | slope |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2020 | 22 | x0.81 | 45% | 77% | 91% | x2.5 | 0.85 | 0.81 |
| 2021 | 53 | x0.51 | 15% | 51% | 92% | x3.3 | 0.91 | 0.85 |
| 2022 | 53 | x1.00 | 23% | 51% | 77% | x4.1 | 0.86 | 0.76 |
| 2023 | 98 | x0.64 | 23% | 52% | 85% | x3.7 | 0.84 | 0.78 |
| 2024 | 162 | x0.55 | 22% | 46% | 79% | x4.7 | 0.75 | 0.83 |
| 2025 | 288 | x0.29 | 16% | 37% | 70% | x6.2 | 0.73 | 0.77 |

### By the accelerator the authors name

|  | papers | median est / true | within x1.5 | within x3 | within x10 | typical error | correlation of logs | slope |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A100 class | 261 | x0.32 | 17% | 39% | 74% | x5.4 | 0.78 | 0.83 |
| GeForce RTX 20, GTX | 66 | x0.77 | 27% | 67% | 92% | x2.8 | 0.85 | 0.71 |
| GeForce RTX 30-50 | 57 | x0.74 | 23% | 56% | 81% | x4.1 | 0.75 | 0.97 |
| H100 or newer | 88 | x0.14 | 16% | 28% | 61% | x9.2 | 0.63 | 0.74 |
| V100, T4, TPU v2-3, workstation | 191 | x0.67 | 23% | 49% | 83% | x3.9 | 0.84 | 0.82 |
| none named | 13 | x1.25 | 15% | 62% | 85% | x3.4 | 0.85 | 0.81 |

### By route (how most of the estimate was reached)

|  | papers | median est / true | within x1.5 | within x3 | within x10 | typical error | correlation of logs | slope |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hours named | 214 | x0.74 | 25% | 52% | 85% | x3.6 | 0.83 | 0.74 |
| hours assumed | 227 | x0.66 | 22% | 51% | 85% | x3.5 | 0.85 | 0.80 |
| FLOP | 235 | x0.20 | 14% | 32% | 64% | x8.5 | 0.74 | 0.84 |

### By size of the authors' figure, H100e-hours

|  | papers | median est / true | within x1.5 | within x3 | within x10 | typical error | correlation of logs | slope |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| <1 | 66 | x1.26 | 21% | 42% | 80% | x4.5 | 0.56 | 0.74 |
| 1-30 | 179 | x0.80 | 23% | 54% | 85% | x3.7 | 0.36 | 0.73 |
| 30-1000 | 333 | x0.39 | 20% | 44% | 77% | x5.1 | 0.41 | 0.87 |
| >1000 | 98 | x0.23 | 14% | 34% | 63% | x6.5 | 0.47 | 0.78 |

### Calibration

Hours: on papers whose estimate is mostly accelerator-hours the authors' figure is a median x1.44 the estimate; h100e.py multiplies estimated hours by 1.4 (stated hours stay).

FLOP: papers whose estimate is at least 80% FLOP, by the kind of work holding most of it; the utilization that makes the median estimate equal the authors' figure, and the one h100e.py uses:

| kind | papers | median est / true at 40% | utilization for x1 | h100e.py |
|---|---:|---:|---:|---:|
| train | 94 | x0.26 | 0.102 | 0.1 |
| forward | 56 | x0.16 | 0.064 | 0.064 |
| generate | 55 | x0.11 | 0.045 | 0.041 |
| mixed | 23 | x0.34 | 0.137 | 0.14 |

After calibration:

|  | papers | median est / true | within x1.5 | within x3 | within x10 | typical error | correlation of logs | slope |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| every paper | 676 | x0.99 | 22% | 51% | 84% | x3.8 | 0.81 | 0.80 |
| 2020 | 22 | x1.14 | 41% | 73% | 86% | x2.5 | 0.85 | 0.81 |
| 2021 | 53 | x0.71 | 28% | 51% | 92% | x3.0 | 0.91 | 0.85 |
| 2022 | 53 | x1.40 | 19% | 47% | 79% | x4.3 | 0.86 | 0.76 |
| 2023 | 98 | x1.00 | 22% | 59% | 90% | x3.3 | 0.86 | 0.78 |
| 2024 | 162 | x1.04 | 18% | 48% | 82% | x3.8 | 0.77 | 0.82 |
| 2025 | 288 | x0.94 | 22% | 48% | 82% | x4.1 | 0.76 | 0.79 |

### Dead ends (calibrated estimates)

37 papers state only a whole-project figure (deleted): whole project / estimate of the reported experiments: median x2.63, quartiles x1.12-x7.99. With x2.5:

|  | papers | median est / true | within x1.5 | within x3 | within x10 | typical error | correlation of logs | slope |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| x2.5 x estimate | 37 | x0.95 | 24% | 51% | 78% | x3.9 | 0.46 | 0.55 |

### Assumed accelerators

| year | papers | mean share of hours-based compute on an assumed accelerator |
|---|---:|---:|
| 2020 | 23 | 74% |
| 2021 | 55 | 62% |
| 2022 | 57 | 47% |
| 2023 | 104 | 44% |
| 2024 | 168 | 36% |
| 2025 | 306 | 21% |

Errors of sonnet-blind and opus-blind on the 669 papers both scored: correlation of log errors 0.76.

## sonnet-full: whole text, reading the stated figures

|  | papers | median est / true | within x1.5 | within x3 | within x10 | typical error | correlation of logs | slope |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 313 | x1.00 | 93% | 98% | 99% | x1.1 | 0.99 | 0.99 |
| part | 387 | x1.00 | 94% | 97% | 99% | x1.1 | 0.99 | 0.98 |

Whole-project figures: 30 of 39 papers that state only one got a project component:

|  | papers | median est / true | within x1.5 | within x3 | within x10 | typical error | correlation of logs | slope |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| project component vs authors | 30 | x1.00 | 90% | 97% | 100% | x1.2 | 0.97 | 1.00 |

28 of 29 targets with both figures got a project component; 7 others got one without a known whole-project figure.
Stated share of the reported compute (mean): 96%.

## opus-full: whole text, reading the stated figures

|  | papers | median est / true | within x1.5 | within x3 | within x10 | typical error | correlation of logs | slope |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 314 | x1.00 | 94% | 98% | 100% | x1.1 | 0.99 | 0.99 |
| part | 387 | x1.00 | 95% | 97% | 99% | x1.1 | 0.99 | 0.96 |

Whole-project figures: 28 of 39 papers that state only one got a project component:

|  | papers | median est / true | within x1.5 | within x3 | within x10 | typical error | correlation of logs | slope |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| project component vs authors | 28 | x1.00 | 86% | 100% | 100% | x1.1 | 0.99 | 1.02 |

28 of 29 targets with both figures got a project component; 6 others got one without a known whole-project figure.
Stated share of the reported compute (mean): 96%.
