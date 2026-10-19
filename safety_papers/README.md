# AI safety papers on arXiv

Which arXiv ML papers of 2020–2025 are AI safety research, and how much compute each one used. Claude labels every
paper from its title and abstract ([prompt.md](prompt.md)), then estimates the compute of the safety papers from their
full texts ([estimate_prompt.md](estimate_prompt.md)).

[results/safety_papers_compute.csv](results/safety_papers_compute.csv) is the list, with compute in H100e-years.
[results/blind_test.md](results/blind_test.md) checks the estimates on papers that state their compute.

```bash
pip install -r requirements.txt
python harvest.py                  # arXiv metadata of every ML paper -> data/
python classify.py run             # label them (Batches API; asks for an API key)
python report.py
python fulltext.py                 # full texts of the safety papers
python labs.py
python blind.py ids                # then extract, estimate, dataset, score
python estimate_compute.py run --all --run census --model claude-opus-5-5 --effort medium
python citations.py
python export.py                   # -> results/
```
