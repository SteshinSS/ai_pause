# progress_model

The capabilities model behind the AI Futures calculator. Given forecasts of
human labor and compute, plus beliefs about AI capabilities and research
productivity, it projects capability trajectories and milestone dates.

The central feedback loop is:

```text
Better AI → more effective research → better algorithms → better AI
```

Human coding labor, inference compute, experiment compute, and frontier
training compute enter as time series. The model tracks two accumulated
quantities: effective training compute and software research effort. Coding
automation, research taste, and displayed capability measures are functions
of the first quantity.

Read the next two sections for the model and its calibration assumptions.
The [execution reference](#execution-reference) gives the implementation
sequence.

## Contents

- [How the model works](#how-the-model-works)
- [How beliefs become model parameters](#how-beliefs-become-model-parameters)
- [Running the model](#running-the-model)
- [Reading the results](#reading-the-results)
- [Execution reference](#execution-reference)
- [Approximations and open questions](#approximations-and-open-questions)
- [Module map](#module-map)
- [Development history](#development-history)

## How the model works

### State and equations

The model integrates two state variables over calendar time `t` (decimal
years). An OOM is an order of magnitude, or a factor of ten.

| symbol | meaning | units |
|---|---|---|
| `logE` | log10 of **cumulative effective compute**. This is what the codebase calls **progress**; every schedule below is indexed by it. | OOMs |
| `RS` | **research stock**, the accumulated stock of software R&D effort | effort-years |

The equations are:

```text
SE = r_software · log10(RS / RS_0)
EC = TC(t) + SE

d(logE)/dt = 10^(EC − logE) / ln 10
d(RS)/dt   = RE(t, logE, RS)
```

`TC(t)` is log10 of the instantaneous training compute in FLOP/yr, read from
the input series. `SE` is log10 **software efficiency**: algorithmic progress
multiplies the effectiveness of raw hardware by `10^SE`. `RS_0` is the initial
research stock. `EC` is the
instantaneous effective-compute *rate* in log FLOP/yr, and `logE` is the log of
its integral. So progress advances at a rate set by how fast effective compute
is currently being spent, relative to how much has been spent so far.

### From AI capability to research effort

`RE`, research effort, is where all the AI feedback lives. A CES (constant
elasticity of substitution) function combines inputs while allowing limited
substitution between them. At each `(t, logE)`:

1. **Automation fraction** `A(logE)` from the automation schedule (a logistic
   in progress by default).
2. **AI research taste** (research judgment): a floor in standard-deviation
   (SD) units that rises linearly in progress, mapped through the human taste
   distribution to a taste value `F(logE)`.
   **Aggregate taste** is `E[max(T, F)]` over that distribution, i.e. the mean
   taste of a workforce whose weakest members are lifted to the AI's level.
3. **Coding labor**: a CES combination of human coding labor `L_HUMAN(t)` and
   AI labor (inference compute `C(t)` at automation fraction `A`), raised to
   `parallel_penalty` to convert parallel labor into **serial** coding labor.
   With `coding_labor_mode='optimal_ces'` the AI side is the optimal frontier
   over tasks given the automation-efficiency slope (see the
   [implementation](automation_model.py)); otherwise
   it is the simple two-input CES.
4. **Research effort**: a second CES of discounted experiment compute
   `experiment_H100e(t)^exponent` and serial coding labor, times aggregate
   taste.

Then `software progress rate = r_software · RE / RS / ln 10`, which is
`d(SE)/dt`, and `ETSP rate = software rate + training-compute growth rate` is
`d(EC)/dt`. Both are reported; neither is `d(progress)/dt`, which is the first
equation above.

In the **human-only** counterfactual `A = 0`, aggregate taste is 1, and coding
labor is human labor alone. That run has no feedback from `logE` into `RE`, so
it uses numerical quadrature (integrating a known effort path) and provides
the baseline for calibration.

### Modeling assumptions

These choices define the model's behavior:

- One scalar, cumulative effective compute, indexes automation, research
  taste, horizon length, revenue, and ECI. Their schedules determine how
  those capabilities develop relative to one another.
- Research stock accumulates effort, and software efficiency is a power of
  that stock: `10^SE = (RS / RS_0)^r_software`.
- Research taste raises the workforce's minimum taste and multiplies the
  output of the experiment-compute/coding-labor CES.
- Resource paths are supplied externally. AI feedback changes research
  productivity; the ODE does not itself decide future labor or compute budgets.

These are distinct from the numerical approximations used to calibrate and
integrate the model, described [below](#approximations-and-open-questions).

## How beliefs become model parameters

### Why dates must be converted to progress

People supply beliefs in calendar terms: “the horizon is X minutes today,”
“taste improves Y SD per year,” or “AI doubles coding productivity at this
date.” The model's schedules instead take progress as their input.

For example, suppose taste is believed to improve by 2 SD per year, and
progress at the anchor date is estimated to grow by 0.5 OOM per year. The
schedule then needs a slope of `2 / 0.5 = 4 SD per OOM`. A horizon doubling
time of half a year becomes `0.5 × 0.5 = 0.25 OOM per doubling`.

This requires an **anchoring mapping**: an estimate of progress and its rate
at each date. The difficulty is circular: the mapping depends on historical
AI assistance, but the AI schedules themselves need the mapping to be fitted.

### Estimating the anchoring mapping

The model first computes a human-only trajectory, then reconstructs the
effect of historical AI assistance using two beliefs:

| Parameter | Meaning |
|---|---|
| `serial_uplift_at_reference_date` | Serial coding-labor multiplier `M` at date `t*` |
| `excess_uplift_doubling_time` | Doubling time of the excess `M − 1`, in months |
| `uplift_reference_date` | `t*`; defaults to `present_day` |

The reconstruction extrapolates excess uplift exponentially in time:

```text
M(s) = 1 + (M − 1) · 2^((s − t*) / T_d)
```

Here `s`, `t*`, and `T_d` must use the same time units; the input doubling
time is converted from months to years. The multiplier is capped at
`max_serial_coding_labor_multiplier`.

The model multiplies human serial coding labor by this path and recomputes
research stock and effective compute by numerical quadrature. Automation
and taste otherwise retain their human-only values in this reconstruction.
Because the multiplier depends only on time, this step needs no ODE solve.

The resulting progress and progress-rate arrays form the anchoring mapping.
If usable uplift statistics are unavailable, the human-only mapping is used.

### Reference dates and resource conventions

`present_day` is the default date at which beliefs bind. Horizon, revenue,
ECI, and uplift can each have a separate `*_reference_date`. An observable's
value and companion growth rate bind at its own date.

The taste slope in “SDs per progress-year” and the automation-efficiency
slope use the mapping's **present-day** progress rate for conversion to
per-OOM units. These “anchor-progress-years” describe a fixed amount of
progress, rather than future calendar years.

Uplift beliefs use **present-day resources**: human labor and inference
compute are held fixed while capability changes. Results named
`*_ref_present_day` use this convention too. Realized multipliers and the
ODE use the resources at each date, so resource growth can make their paths
differ from the fixed-resource schedules.

### Choosing the automated-coder anchor

The automated-coder milestone is abbreviated **AC**. Its progress threshold
has the historical code name `progress_at_aa`.

The `ac_anchor_mode` determines how that threshold is set:

| mode | how `progress_at_aa` is set |
|---|---|
| `time-horizon` | invert the horizon trend at `ac_time_horizon_minutes` |
| `revenue` | invert the revenue trend at `revenue_required_for_ac` |
| `eci` | invert the ECI trend at `eci_required_for_ac` |
| `uplift-solved` | derive AC from the automation schedule fitted to uplift beliefs |

In the three observable modes an optional **gap** (`include_gap='gap'`,
`gap_years` in anchor-progress-years, converted at the present-day rate) is
added on top of the inversion.

In the three **observable modes**, the AC threshold and present-day uplift
determine the automation schedule. Its implied uplift doubling time is an
output. The model measures that time, rebuilds the reconstruction, and fits
the anchors and schedule once more.

In **`uplift-solved` mode**, uplift and its excess doubling time directly
constrain the automation schedule; AC is an output. The reconstruction is
left as built. Horizon, revenue, and ECI are still fitted for display, but do
not determine AC.

Thus `excess_uplift_doubling_time` has different roles by mode: it is a
schedule constraint in `uplift-solved`, and an initial reconstruction input
that the correction pass replaces in observable modes. The detailed solves
and fallback conditions are in the [execution reference](#execution-reference).

## Running the model

Run from the repository root:

```python
from progress_model import Parameters, ProgressModel, load_time_series_data

data = load_time_series_data("inputs/input_data.csv")
overrides = {}  # Optional parameter overrides
model = ProgressModel(Parameters(**overrides), data)
times, progress, research_stock = model.compute_progress_trajectory([2019.0, 2060.0])

results = model.results
milestones = results["milestones"]
```

### Inputs and parameter lifecycle

`Parameters` is a dataclass with defaults in `config.DEFAULT_PARAMETERS`.
Its `__post_init__` sanitizes non-finite values, validates categorical
options, builds and caches the `TasteDistribution`, converts the taste-at-AC
anchor from SD units to a taste value, and resolves the uplift belief.
An explicitly stated `serial_uplift_at_reference_date ≤ 1` raises.

A run **mutates the parameters**: `r_software`, `progress_at_aa`,
`automation_anchors`, `automation_model`, converted slopes, and
experiment-capacity CES parameters hold derived values afterward.
Construct a fresh `Parameters` and model for a fresh run.

`TimeSeriesData` holds `time`, `L_HUMAN`, `inference_H100e`,
`experiment_H100e`, `training_H100e`, and optional display series. Logs
are precomputed for log-space interpolation. Training growth is
piecewise constant between grid points and averaged at the points themselves.

`training_H100e` is instantaneous frontier training capacity, not the
cumulative FLOPs of a completed run. See
[input-series provenance](../docs/input-series-provenance.md) for the
conversion from training-run forecasts. `time_series.py` produces
perturbed inputs for Monte Carlo; `pause_transform.py` implements the
frontend's freeze-and-reduce pause.

## Reading the results

### Series, rates, and diagnostics

`model.results` is a flat dictionary containing arrays, scalars, and
structured diagnostics. Arrays have one value per output time.

| Output | Meaning |
|---|---|
| `progress` | `logE`: log10 cumulative effective compute, in OOMs |
| `progress_rates` | `d(logE)/dt`, the growth rate of that cumulative stock |
| `etsp_rates` | `d(EC)/dt`: software-efficiency growth plus training-capacity growth |
| `automation_fraction` | Fraction from the fitted automation schedule |
| `horizon_lengths`, `predicted_revenue`, `capability_ecis` | Observable trends evaluated on the realized trajectory |
| `aa_time`, `ai2027_sc_time` | AC and AI2027-SC crossing times |
| `r_software`, `anchor_progress_rate` | Calibrated software exponent and anchoring rate |
| `milestones` | Per-milestone `time`, `effective_compute_ooms`, `progress_multiplier`, and `research_stock` |
| `uplift_solve` | Solved asymptote, implied uplift at AC, and reference date/rate; `None` outside `uplift-solved` |
| `td_correction` | Mode, believed and measured uplift statistics, mismatch, whether correction applied, and reason if skipped |
| `input_time_series`, `exp_capacity_params`, `timing` | Inputs, derived CES parameters, and wall/CPU timing by step |

The two progress rates are different quantities; do not use `etsp_rates`
as `d(progress)/dt`.

The metrics also include taste (value, SD, quantile, and aggregate), AI and
human coding contributions, serial coding labor, realized and fixed-resource
multipliers, research effort, software efficiency, experiment capacity and
its asymptotes, compute-versus-labor marginal substitution rates, training
compute, and optimal training-run length.

Some calibration diagnostics live on the **model object**, outside
`results`: `_uplift_reconstruction_info`, `_uplift_reconstruction`
(mapping arrays), `_anchor_stats_used`, `_anchor_mapping`,
`_td_correction_info`, and `_uplift_solve_diagnostics`.

`scripts/pipelines/batch_rollout.py` persists `results` verbatim.
`/api/plan-a` also includes `td_correction` in its summary. The public
correction record expresses doubling times in months.

### Milestones

Each milestone is a threshold crossing of a metric:

| milestone | metric | target |
|---|---|---|
| AC | progress | `progress_at_aa` |
| AI2027-SC | ref-present-day coding-labor multiplier | `30^(1/pp) · 30 · c` |
| AIR-5x … AIR-10000x | AI software-progress multiplier | 5, 25, 250, 2000, 10000 |
| SAR, SIAR | AI taste in SD | top-researcher SD, and 3× that; both **require AC** |
| STRAT-AI | AI taste in SD | top SD · (1 + `strat_ai_m2b`) |
| TED-AI, ASI | AI taste in SD | baseline + top SD · `ted_ai_m2b`, and + 2 more M2Bs |
| Maximum skill | AI taste in SD | `AI_RESEARCH_TASTE_MAX_SD` |

Here `pp` is `parallel_penalty` and `c` is
`PARALLEL_LABOR_MULT_BETWEEN_AVERAGE_AND_TOP_FOR_AI2027_SC`.

The TED-AI baseline is the SD at AC if SAR would otherwise precede AC, else
the top-researcher SD. Crossing times use exponential (log-space)
interpolation for multiplier and SD metrics. Effective compute at each
milestone is read by interpolating EC against the metric rather than against
time, which avoids grid-quantisation noise where takeoff makes EC nearly
vertical in time. In SOS mode each milestone also gets the human-only years to
the next one.

### Display rebasing

After integration, horizon, revenue, and ECI are re-anchored to the realized
progress at their own reference dates. Their progress slopes remain fixed.
The AC threshold is not re-derived, because it already shaped the automation
schedule used by the ODE.

The reported `software_efficiency` and `effective_compute` series are also
shifted so software efficiency is zero at
`config.TRAINING_COMPUTE_REFERENCE_YEAR`. This display normalization does
not change progress, growth rates, or milestone times.

## Execution reference

`ProgressModel.compute_progress_trajectory` in [_impl.py](_impl.py) runs
the following sequence. An ordinary run calls `integrate_progress` once;
that integration can try several solvers. SOS/SIE mode can request a second
integration.

### 1. Initialize progress and experiment-capacity parameters

If `initial_progress` is `None` or `0.0`, derive
`logE₀ = TC(t₀) − 1`. This is an approximate initialization; its influence
declines as new effective compute accumulates.

Unless `direct_input_exp_cap_ces_params` is set, solve the research-effort
CES's `rho`, `alpha`, and `experiment_compute_exponent` from:

- `inf_labor_asymptote` and `inf_compute_asymptote`;
- `compute_anchor_exp_cap` and `labor_anchor_exp_cap`, evaluated at
  present-day experiment compute and human labor.

### 2. Calibrate software progress on a short reconstruction

Run `compute_human_only_trajectory` over `[t₀, present_day + 1]`.
The human-only integrator uses a 300× refined time grid: evaluate resources
and research effort, accumulate research stock by cumulative trapezoidal
integration, compute `EC`, then integrate `10^EC` to obtain `E`.

It returns the series, present-day anchor statistics (progress and its rate,
ETSP and software rates, effort, stock, and resources), and the software
progress rate at the reference year.

Apply the uplift reconstruction to this short run. With its multiplier path
fixed, the reconstructed software rate is linear in `r_software`, allowing:

```text
r_software ← r_software · elicited_rate / reconstructed_rate_at_2024.0
```

The target is `software_progress_rate_at_reference_year`, at
`config.SOFTWARE_PROGRESS_SCALE_REFERENCE_YEAR` (2024.0). If reconstruction
statistics are unavailable, use the human-only rate.

### 3. Build the full anchoring mapping

Recompute the human-only trajectory over the full `time_range` with the
calibrated `r_software`, storing it in `model.human_only_results`.
Apply `uplift_reconstruction.reconstruct_from_params` to obtain the full
anchoring mapping described above.

`_uplift_reconstruction_info` records its availability and the shift from
human-only present-day progress and rate.

### 4. Fit observables, set AC, and convert slopes

[ac_anchor.py](ac_anchor.py) fits all three observables against the mapping:

| Observable | Trend and inputs |
|---|---|
| Horizon | Log p80 METR horizon versus progress; `exponential` or `decaying doubling time`, using `present_horizon`, `present_doubling_time`, and `doubling_difficulty_growth_factor` |
| Revenue | `present_revenue · growth^((p − p0)/rate)` |
| ECI | `eci_at_present_day + (eci_growth_rate / rate) · (p − p0)` |

Here `p0` and `rate` come from the mapping at the observable's reference
date. METR benchmark data fills horizon parameters only when they are
`None`. Each `FittedObservable` provides `predict(progress)`,
`invert(target)`, and `rebase_anchor(new_p0)` over the same fit.

`_compute_ac_anchor` inverts the selected observable and applies any gap;
`uplift-solved` defers AC to the next step.

`_convert_anchor_year_slopes` divides the automation-efficiency slope,
and the taste slope when expressed in SDs per progress-year, by the mapping's
present-day progress rate. The original slopes are captured once on the
model, so repeating this step does not divide an already converted value.

### 5. Derive the automation schedule

`_derive_automation_schedule` creates `params.automation_model`, whose
`get_automation_fraction(progress)` is used by the ODE.

In **observable modes**, `solve_lower_anchor_via_automation_model` finds
the present-day automation fraction `A₀` that matches the implied parallel
coding-labor belief `swe_multiplier_at_present_day` (derived from the serial
uplift belief), using present-day labor and inference compute. The logistic,
linear, or exponential schedule passes through:

```text
{p0: A₀, progress_at_aa: automation_fraction_at_coding_automation_anchor}
```

In **`uplift-solved`**, `_solve_uplift_anchored_automation_model` solves
logistic parameters `(k, x0)` so the schedule's serial uplift satisfies:

```text
M(p*)                = serial_uplift_at_reference_date
d ln(M − 1)/dp at p* = ln 2 / D_progress
```

`p*` is the mapping's progress at `uplift_reference_date`;
`D_progress` converts the doubling time using the rate at that same date.
Optional `serial_uplift_at_ac` supplies a third condition to solve the
asymptote `L`. AC is the `A(p) = 1` crossing plus the gap.
Unreachable beliefs raise with the reachable range in the message.
Diagnostics are stored in `results['uplift_solve']`.

### 6. Apply the uplift doubling-time correction

`_measure_schedule_uplift_statistics` evaluates serial uplift at three
progress levels, holding resources at present-day values. It measures
`M` at `t*` and the excess-uplift log slope across
`present_day ± 0.25 years`.

`_apply_td_correction_pass` records the believed and measured statistics.
In `uplift-solved`, it leaves the mapping unchanged. In other modes, it
rebuilds the reconstruction from the measured statistics and repeats steps
4–5 once. If statistics are unusable or reconstruction fails, it retains the
existing mapping and records the reason.

This step does not recalibrate `r_software` or iterate to convergence.
The recorded measurement describes the schedule before this re-derivation.

### 7. Initialize research stock and integrate

`calculate_initial_research_stock` estimates:

```text
RS₀ = RE₀² / (dRE/dt)₀
```

It uses **human-only** effort and its numerical time derivative at the start
of the input data, consistent with an exponentially growing historical effort
path. Small denominators or invalid stocks trigger fallbacks.
`compute_initial_conditions` also evaluates initial with-AI quantities,
using the legacy coding-labor function; those do not set this stock estimate.

`integration.integrate_progress` calls `scipy.integrate.solve_ivp` with
`progress_rate.progress_rate_at_time` as its right-hand side. It tries
Radau, RK23, RK45 at two tolerances, DOP853, and LSODA until one succeeds.
Clamps and fallback handling protect states and rates; non-finite output
values may be replaced by interpolation.

If `sos_mode` is enabled and `sos_start_milestone` (default AC) is reached,
`freeze_time_series_at_time` holds labor and compute at that date's values,
sets training-capacity growth to zero, and reruns the integration. This is
the software-only singularity (SOS/SIE) scenario.

### 8. Measure residuals and produce outputs

Compare realized present-day progress with the mapping's anchor and record
the residual in `_uplift_reconstruction_info`. Then apply the observable
rebasing described above.

`metrics_computation.compute_metrics_loop` evaluates the metrics over the
output grid. The orchestrator applies display normalization and computes
crossing times, horizon doubling time at its anchor, converted slopes in both
unit systems, top-taste diagnostics (`m/β` and the SIE doubling ratio), and
software-multiplier doubling-time series.

Finally, `compute_milestones` assembles the milestone records and
`compute_metr_mse` scores the fitted horizon against METR benchmark points
in progress coordinates. Wall and CPU timings are stored by step.

## Approximations and open questions

### Calibration is approximate

The reconstruction, software-rate rescale, and doubling-time correction are
each applied once. They avoid repeated full ODE integrations, but do not
guarantee that every elicited level and rate binds on the final trajectory.
Observable rebasing corrects levels at reference dates without refitting the
dynamics or slopes.

The present-day progress residual measures one part of this discrepancy.
Its size depends on the parameters; historical examples appear in the
development notes below.

### The doubling-time measurement can mix dates

The correction measures uplift level at `uplift_reference_date` but its
doubling time around `present_day`. When those dates differ, the recorded
mismatch compares rates at different dates, and the reconstruction receives
a level and slope from different dates. The Q2 2026 presets use equal dates.

Possible changes include recording separate measurement dates or centering
the doubling-time measurement at the uplift reference date.

### Reconstructing the schedule's full path

The current correction compresses the automation schedule into `(M, T_d)`
and expands those statistics back into an exponential path in time.
An alternative is to evaluate the schedule's uplift along the first mapping,
using each date's resources, and feed that path to the quadrature.

That would avoid the two-statistic approximation and its separate exponential
path cap. It would be one update toward self-consistency, using the cheap
inner model, rather than a converged solution. It would change observable-mode
golden outputs and requires deciding whether `uplift-solved` uses it too.

### Sampling depends on the anchoring mode

The Q2 Monte Carlo configurations sample the mode as well as
`excess_uplift_doubling_time`. In two of their three sampled modes, the
correction replaces that sampled doubling time with a schedule-derived value.
Whether to sample it in those modes remains an elicitation question.

## Module map

| module | role |
|---|---|
| `_impl.py` | `ProgressModel`: run orchestration, milestones, METR MSE |
| `parameters.py` | `Parameters` dataclass and its `__post_init__` resolution |
| `config.py` | `DEFAULT_PARAMETERS`, reference years, clamps, unit constants |
| `types.py` | `TimeSeriesData`, `InitialConditions` |
| `progress_rate.py` | the ODE right-hand side and the research-effort / software-rate / taste functions it calls |
| `integration.py` | initial conditions, the human-only quadrature, the ODE solve with its method ladder |
| `uplift_reconstruction.py` | `uplift_statistics`, the exponential multiplier path, `reconstruct_uplifted_trajectory` |
| `ac_anchor.py` | `FittedObservable`; horizon, revenue and ECI trend fits; anchor-date resolution; `progress_at_aa` inversion |
| `automation_model.py` | `AutomationModel` (logistic/linear/exponential schedules, optimal-CES frontier cache); the lower-anchor solver; the uplift-solved `(k, x0[, L])` solvers; `serial_uplift_for_schedule` |
| `ces_functions.py` | the CES production functions and the experiment-capacity parameter solver |
| `taste_distribution.py` | bounded-normal human taste distribution; SD ↔ taste ↔ quantile; `E[max(T, F)]` |
| `metrics_computation.py` | the vectorized metrics loop |
| `time_series.py` | perturbed input series for Monte Carlo |
| `pause_transform.py` | the frontend-identical freeze-and-reduce transform |
| `utils.py` | log interpolation, Gauss-Hermite expectation, exception filtering |

## Development history

These measurements motivated the current implementation. They depend on the
defaults and parameter draws used at the time; they are not accuracy or
runtime guarantees.

- Human-only anchoring left present-day progress discrepancies of roughly
  0.08 OOM in reported examples, with a shift of roughly 0.16 OOM under older
  defaults. Reconstruction reduced residuals to a few thousandths of an OOM
  in typical reported runs.
- The `anchor-fixed-point` branch implemented iterative anchoring, with
  roughly nine integrations in the reported comparison. It served as a
  benchmark for the one-pass approximation.
- Reported randomized draws showed schedule-versus-belief uplift-statistic
  mismatches of 0.01–0.5% in `uplift-solved`, compared with doubling-time
  mismatches of 9–127% in time-horizon and 13–67% in revenue mode. Applying
  the correction in `uplift-solved` worsened the measured anchor residual.
- Until September 2026, a 5% mismatch threshold controlled the correction.
  In those draws it behaved like a mode switch, except when differing
  reference dates triggered it within `uplift-solved`. The current rule
  switches explicitly on mode.
- The correction's extra quadrature and anchor derivation took about 25 ms
  in a reported 350 ms run.
