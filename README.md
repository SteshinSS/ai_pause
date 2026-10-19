# 1000 Years of AI Safety

Code behind [1000yearsofaisafety.com](https://1000yearsofaisafety.com): how much AI safety research a US pause on
frontier AI could buy, built on the [AI Futures Model](https://www.aifuturesmodel.com/).

- `pause.py`: the model; `pause_inputs.csv`: its inputs
- `sensitivity.py`, `random_checks.py`, `slow_scaling.py`: side analyses on the site
- `aifm/`: the AI Futures Model, unmodified ([aifm-public](https://github.com/AI-Futures-Project/aifm-public))
- `safety_papers/`: AI safety papers on arXiv, 2020–2025
- `site/`: the website

```bash
python3.14 -m venv .venv && source .venv/bin/activate
pip install -r aifm/requirements.txt
python pause.py
```
