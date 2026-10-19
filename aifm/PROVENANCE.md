# aifm/ (vendored, do not edit)

Subset of https://github.com/AI-Futures-Project/aifm-public (MIT license, see LICENSE), the code behind
https://www.aifuturesmodel.com/ (AI Futures Model, Aug 2026 version).

- commit: 1c40ecdb246c25980515441a66571931c5604e11 (2026-09-09 13:00:40 -0700)
- vendored on: 2026-09-16; every file is byte-identical to upstream. This file and UPSTREAM_COMMIT are the only
  additions.
- adaptations for the pause model live outside this directory (../pause.py).

| path | what |
|---|---|
| progress_model/ | the Python model (ODE, calibration, milestones). Guide: progress_model/README.md |
| inputs/input_data.csv | default exogenous series (leading company): L_HUMAN, inference/experiment/training H100e, world and company totals, 2017-2200 |
| inputs/plan-a.csv, inputs/2200.csv | alternative series (AI-2040 Plan A consortium; older 2200 file) |
| inputs/metr/, inputs/epoch/, inputs/old_model/ | METR horizon benchmarks and Epoch revenue/ECI observations used to fit the observable trends |
| config/*.yaml | Monte Carlo parameter distributions (sampling_config.yaml is the root; *_q2_{daniel,eli,brendan}.yaml are the Aug 2026 forecaster configs) |
| docs/input-series-provenance.md | how training_H100e was derived (rate, not stock) |
| tests/ (+ golden_data/) | upstream regression tests; tests needing plan_a_covert_proj, scripts/export or matplotlib were dropped |
| scripts/pipelines/batch_rollout.py | the Monte Carlo driver (run with cwd = this directory) |
| constants/parameters.ts | frontend parameter presets (daniel/eli/brendan-08-16-26) and their rationale comments |
| utils/revenueData.ts, utils/eciData.ts | frontend copies of the Epoch revenue/ECI observations; only kept so tests/test_epoch_*.py run |
| app/forecast/data/simulations/*/ | summary statistics of the published 10k-sample forecasts per preset (p10/p50/p90 per milestone) |

Not vendored: the Next.js website, api/, plan_a_covert_proj/ (AI-2040 covert-project simulation), analysis/plot/export scripts, public/.
