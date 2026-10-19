# How the input time series is built

Provenance for `inputs/input_data.csv` — in particular why `training_H100e` is what it is,
and where `TRAINING_COMPUTE_REFERENCE_OOMS = 26.54` in `constants/parameters.ts` comes from.

This was previously documented only in the docstring of `scripts/utilities/xlsx_to_input_csv.py`,
which was deleted in `72a97ea6` ("Remove unreachable and unreferenced Python modules") along
with its source workbook, `inputs/[AI Futures Model] Input Time Series Forecasts.xlsx`. The
day after that deletion, `constants/parameters.ts` was edited to point readers at the very
docstring that had just been removed — while reverting a derivation of this constant that had
gone dimensionally wrong. That is the reason this file exists: the derivation is subtle, it has
already tripped one attempt to redo it, and it should not live only in git history.

Both artefacts are recoverable if needed:

```bash
git show 72a97ea6^:scripts/utilities/xlsx_to_input_csv.py
git show <sha>^:"inputs/[AI Futures Model] Input Time Series Forecasts.xlsx" > /tmp/forecast.xlsx
```

## The columns are rates, not stocks

`inputs/input_data.csv` has the columns

```
time, L_HUMAN, inference_H100e, experiment_H100e, training_H100e, world_H100e, total_ai_company_H100e
```

`training_H100e` is the **instantaneous H100e currently running the frontier training run** — a
capacity, i.e. a rate. The loader converts it to an instantaneous FLOP/yr rate as
`log10(training_H100e) + log10(FLOP_PER_H100E_YEAR)` (`progress_model/integration.py:386`), and
the ODE *integrates* that rate into cumulative effective compute.

(The older names for these columns were `inference_compute`, `experiment_compute` and
`training_compute`; the H100e port renamed them. Anything still saying `training_compute` is
log10(FLOP/yr) predates that port and is wrong about today's units.)

## The trap when sourcing it from a FLOP forecast

The upstream forecast supplies the largest-training-run series as **cumulative FP16-eq FLOP** —
column AA, "Largest training run complete … FP16-eq FLOP", of the workbook's *Company compute
allocation* sheet. Call it `AA(t)`.

**Do not** populate `training_H100e` by dividing `AA` by `FLOP_PER_H100E_YEAR`. That yields the
run's compute in H100e-*years* — a stock, roughly amortised chips — which is not what the column
means, and feeding a stock where the ODE expects a rate silently rescales all downstream progress.

Instead treat `AA(t)` as the cumulative output of a fictitious continuous run and reconstruct the
**instantaneous** accumulation rate in H100e, using the log-mean reconstruction implemented in
`scripts/utilities/flop_to_rate.py`. Fed back through the loader, this integrates exactly to the
cumulative hardware-FLOP stock `= AA` (the `training_compute` series, not `effective_compute`).

## Where 26.54 comes from

Because the reconstruction integrates exactly to `AA`, the **cumulative hardware-FLOP stock**
reaches `log10(AA(ref))` at the reference year, with no rebasing fudge:

```
TRAINING_COMPUTE_REFERENCE_YEAR = 2025.13          # progress_model/config.py:137
training_compute(ref) = log10(AA(ref))
                      = log10(Grok 3 training FLOP)
                      = 26.54                      # TRAINING_COMPUTE_REFERENCE_OOMS
```

It is the training FLOP of the reference model (Grok 3) at that reference year.

Note the series this pins is `training_compute` (the stock), **not** `effective_compute`. Those
two do not coincide at the reference year, and it is easy to assume they do:

- The post-integration rebase in `progress_model/_impl.py` subtracts `software_efficiency` at
  the reference year, which makes `SE(ref) == 0` and therefore `ETSP == TSP` at that date — the
  instantaneous effective-compute *rate* equals raw hardware FLOP/yr there. That is a statement
  about rates.
- Cumulative `logE` does not inherit it. `logE` integrates `10**(TSP+SE)` while the stock
  integrates `10**TSP`, so `logE - log(stock) != SE` pointwise and a scalar shift cannot align
  them. They end up a fraction of an OOM apart at the reference year, which is why the two lines
  on the "Effective & Training Compute" chart cross some months after the reference date rather
  than on it. The size of that gap depends on the parameters and the input series, so no number
  is quoted here — read it off `results['effective_compute']` and `results['training_compute']`
  at `cfg.TRAINING_COMPUTE_REFERENCE_YEAR` for whichever config you care about.

`constants/parameters.ts` additionally uses this constant as a client-side
effective-compute → progress offset (`progress = effectiveCompute_OOM - 26.54`). The backend
does not produce that offset, so do not assume the two conventions agree.

Python no longer declares it — `progress_model/config.py:138` is commented out — so the only
live copy is `TRAINING_COMPUTE_REFERENCE_OOMS` in `constants/parameters.ts`. Nothing enforces
agreement between that constant and the input series: if the series is ever rebuilt from a new
forecast, this value has to be recomputed by hand as `log10(AA(ref))`, and a mismatch shows up
as a uniform shift in progress rather than as a test failure.

The remaining columns (`time`, `L_HUMAN`, `inference_H100e`, `experiment_H100e`) came from the
workbook's *Export format* sheet, which was already in the input CSV's units and layout.
