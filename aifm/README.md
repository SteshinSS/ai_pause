# AI Futures Model

The model and website behind [aifuturesmodel.com](https://www.aifuturesmodel.com/),
an interactive forecast of AI timelines and takeoff speeds.

The Python model explores a feedback loop: more capable AI can automate more AI
research, accelerating algorithmic progress and making AI more capable. The
Next.js website lets you explore individual trajectories, compare forecasts,
and inspect the assumptions behind them.

- **Understand the model:** read the [model guide](progress_model/README.md), including its calibration assumptions and approximations.
- **Run the website locally:** follow [Quick start](#quick-start).
- **Generate forecasts:** see [Monte Carlo runs and website exports](#monte-carlo-runs-and-website-exports).

## Quick start

Use Node.js 20.9 or newer and Python 3.14 (see [.python-version](.python-version)).
Run these commands from the repository root; the virtual-environment activation
below is for macOS/Linux.

```bash
python3.14 -m venv .venv
source .venv/bin/activate

npm install
python -m pip install -r requirements-dev.txt
npm run dev
```

Open [localhost:3000](http://localhost:3000). This starts both the Next.js frontend
and the Flask API on port 5328. Next.js forwards development API requests to Flask.

Keep the virtual environment active when running the backend or Python scripts.
[requirements-dev.txt](requirements-dev.txt) includes the runtime dependencies
plus testing and offline-analysis tools; [requirements.txt](requirements.txt)
is the smaller serverless-runtime set.

If port 5328 is already occupied, use `FLASK_DEV_PORT=5329 npm run dev`; the API
and development proxy will both use the new port. To choose a frontend port,
start the two services separately with `npm run flask-dev` and
`npm run next-dev -- --port 3002` (set the same `FLASK_DEV_PORT` in both shells
if you changed it).

| Command | Purpose |
| --- | --- |
| `npm run dev` | Start the frontend and API together |
| `npm run next-dev` / `npm run flask-dev` | Start either service separately |
| `npm run build` / `npm run start` | Build / serve the production frontend |
| `npm run lint` | Run ESLint |
| `npm run typecheck` | Regenerate Next.js types and check TypeScript |
| `python -m pytest` | Run Python tests |
| `npm run mermaid` | Regenerate the model architecture diagram |

## How the model works

The model combines forecasts of human labor and compute with assumptions about
coding automation and AI research judgment. It tracks cumulative effective
compute and accumulated software research effort. Research improves software
efficiency, which makes compute more productive; increasing capabilities then
feed back into research.

Progress is measured in orders of magnitude (OOMs) of cumulative effective
compute. Capability milestones are thresholds on that scale:

| Label | Milestone | Meaning |
| --- | --- | --- |
| AC | Automated Coder | Fully automates an AGI project's coding work |
| AI2027-SC | Superhuman Coder | The superhuman-coder milestone used in AI 2027 |
| SAR | Superhuman AI Researcher | Fully automates AI R&D |
| SIAR | Superintelligent AI Researcher | Exceeds the top AGI-project researcher by twice the top-to-median researcher gap |
| TED-AI | Top-Expert-Dominating AI | At least as good as top human experts at virtually all cognitive tasks |
| ASI | Artificial Superintelligence | Exceeds the best humans by twice the best-to-median professional gap |

The AC threshold can be calibrated using task time horizons, revenue, the Epoch
Capabilities Index, or research uplift. These are alternative anchoring
assumptions, not separate models. Forecasts depend on both the chosen
calibration and the sampled parameter assumptions.

See the [model guide](progress_model/README.md) for equations, runnable Python
examples, output definitions, and a module map. Parameter defaults and bounds
live in [progress_model/config.py](progress_model/config.py); input data and its
sources are described in [Input-series provenance](docs/input-series-provenance.md).

## Website pages

| Route | Purpose |
| --- | --- |
| `/` | Default-parameter trajectory charts |
| `/p` | Interactive trajectories with parameters encoded in the URL |
| `/forecast` | Monte Carlo milestone distributions by forecaster and AC-anchor mode |
| `/atc` | All-things-considered forecast editor for adjusting cumulative probability curves |
| `/analysis` | Run figures, correlations, and sensitivity analysis; `/results` redirects here |
| `/plan-a` | Plan A covert-project explorer; add `?embed=1` for an iframe |
| `/playground` | Parameter exploration |
| `/about`, `/gdoc/[id]` | Explanatory prose backed by Google Docs |

## Repository map

| Path | What to look for |
| --- | --- |
| [progress_model/](progress_model/) | Python capabilities model, calibration, and configuration |
| [api/index.py](api/index.py) | Flask API, including `/api/compute` and `/api/plan-a` |
| [app/](app/), [components/](components/) | Next.js routes and React UI |
| [lib/](lib/), [utils/](utils/), [constants/](constants/), [types/](types/) | Shared helpers, chart logic, and domain types |
| [config/](config/) | Monte Carlo sampling configurations |
| [inputs/](inputs/) | Compute/labor time series, METR benchmarks, and Epoch observations |
| [scripts/](scripts/) | Simulation, analysis, plotting, and export pipelines |
| [plan_a_covert_proj/](plan_a_covert_proj/) | Separate covert-project simulation |
| [tests/](tests/) | Model tests and golden reference data |
| `outputs/`, `mc_outputs/` | Generated simulation results, not source files |

## Monte Carlo runs and website exports

Run these commands from the repository root with the Python environment active.
Start with a small batch before launching a full forecast:

```bash
python scripts/pipelines/batch_rollout.py \
  --config config/sampling_config_q2_daniel.yaml \
  --num-samples 100 --num-workers 2 --seed 42 \
  --output-dir outputs/readme-example
```

Each batch creates a timestamped run directory under the output directory.
Choose a configuration explicitly: forecasters and experiments use different
assumptions. Check each config’s `parent:` chain to see the settings it inherits.

For the combined rollout-and-figure pipeline:

```bash
python scripts/pipelines/results_figures.py \
  --num-samples 100 --num-workers 2 \
  --output-dir outputs/results-example
```

This runs the configurations selected inside
[results_figures.py](scripts/pipelines/results_figures.py), not every YAML file
in `config/`. Check those selections before starting a large run. Production
batches can take substantial compute and disk space; 100 samples is a trial,
not a recommended final forecast size.

Exporting is a separate step. Replace the quoted run path below with the
timestamped directory created for Daniel:

```bash
python scripts/pipelines/forecast_export.py \
  --run-dir "outputs/results-example/daniel/REPLACE_WITH_TIMESTAMP" \
  --forecaster Daniel --set-default

python scripts/export/update_website_results.py \
  --results-dir outputs/results-example --default-run daniel
```

The first command writes forecast assets to `app/forecast/data/simulations/`
and selects the default forecast. The second copies analysis figures and data
into `public/results/` and `app/results/data/runs/`. These commands modify
local website assets; they do not deploy the site. Review the generated changes
before committing.

[scripts/hf_results.py](scripts/hf_results.py) handles uploading and downloading
rollout archives.

## Checks

```bash
npm run lint
npm run typecheck
python -m pytest
```

Use `npm run typecheck` rather than bare `tsc` so stale generated Next.js types
do not cause misleading failures. Python tests run from the repository root,
with import paths configured by [pytest.ini](pytest.ini).

Many model tests compare against golden reference data. Do not regenerate it
without explicit approval: differences may indicate changed model behavior.

## Deployment and further reading

The site deploys to Vercel: a Next.js frontend plus the Flask API as a Python
serverless function. [vercel.json](vercel.json) sets the API's maximum duration
to 60 seconds. `npm run start` serves the frontend only; it does not start Flask.

- [Plan A simulation guide](plan_a_covert_proj/README.md)

## License

The code is released under the [MIT License](LICENSE).
