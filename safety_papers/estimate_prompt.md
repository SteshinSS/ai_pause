You read the full text of one machine-learning research paper and report the accelerator compute its authors used,
as components: accelerator-hours on a named accelerator, or FLOP. We convert the components to H100-equivalents
ourselves. The request names the work: all experiments the paper reports, or a part of them. Passages that state run
times, costs or GPU-hours may have been deleted from the text; then estimate from what the paper describes.

## What to count

Unless the request names a part of the work, count every experiment the paper reports, once, as the authors ran it:
all models x datasets x conditions x seeds, ablations, appendix experiments, baselines the authors ran themselves,
hyperparameter searches the paper describes, data generation with local models, evaluations with local models
(including local LLM judges), training of probes, sparse autoencoders or reward models, RL rollouts. When the request
names a part ("the pretraining of the 162M model"), count only that part.

Do not count: pretraining of models the authors only downloaded; experiments of cited work; calls to hosted models
through an API (OpenAI, Anthropic, Google Gemini or other hosted models): leave them out entirely; human work.

Pilots, failed runs and debugging that the paper does not describe are not reported experiments. They enter only
through a figure the paper states for the whole research project, as a `project` component (below).

## Components

Split the work into components: each large one on its own, small ones grouped. For each component:

- `what`: the work, the number of runs, model and data size, and how you got the amount, in a few words.
- `scope`: `reported` for the reported experiments (or the part the request names). `project` only for a figure the
  paper says includes work it does not report: preliminary, exploratory, failed or development runs, debugging
  ("the full project, including preliminary experiments, used about 2,000 GPU hours"; "development took about 3x the
  compute of the final experiments": then the project amount is 3 x the reported total, written out). A total for
  "all experiments", "the study" or "this paper", or a compute or carbon statement that does not mention such work,
  is `reported`, even when the paper calls it the total. Give `project` components only next to the `reported`
  ones, which you still give in full.
- `source`: `stated` when the amount comes from the authors' figures (run times, GPU-hours, FLOP, cost), else
  `estimated`.
- `unit`: `hours` (accelerator-hours: accelerators x wall-clock hours) or `FLOP`.
- `amount`: one arithmetic expression (numbers, + - * / and parentheses only) whose value is the amount, e.g.
  `36*6*8` or `6*7e9*2e9*3`.
- `accelerator`: for `hours`, the accelerator the work ran on, from the list below; for `FLOP`, `none`.
- `accelerator_named`: `yes` if the paper names that accelerator for this work; `no` if you assumed it or the unit
  is `FLOP`.
- `kind`: `train` (anything with backpropagation: training, fine-tuning, LoRA, steering vectors, gradient-based
  attacks), `forward` (forward passes only: scoring, activation collection, probing), `generate` (autoregressive
  generation, rollouts, local judges) or `mixed`.

## How to get the amount

a. **Stated figures**: use them ("each of the 36 runs took 6 hours on 8 A100s" -> hours, A100, `36*6*8`). GPU-days
   x 24, GPU-weeks x 168. A stated cloud cost: hours = cost / the hourly price of that instance. Stated FLOP: unit
   `FLOP`.

b. **Models of ~1B parameters or more, and autoregressive generation with any model**: FLOP, with N = parameters
   (active parameters for mixture-of-experts) and T = tokens processed, summed over all runs. Count the FLOP the
   work needs; how efficiently it ran is our side:
   - training that updates the weights (pretraining, fine-tuning, RL policy updates): 6 N T
   - backpropagation through frozen weights (LoRA, adapters, prompt tuning, steering vectors): 4 N T
   - forward passes without sampling (scoring, activation collection, patching, probing, prompt tokens): 2 N T
   - autoregressive generation (answers, rollouts, local judges): 2 N T, T = generated plus prompt tokens

   Tokens: stated dataset sizes and lengths; otherwise typical sizes of the named benchmarks, prompts of 200-500
   tokens, chat answers of 300-500 tokens, reasoning-model answers of 2,000-8,000 tokens.

c. **Everything else** (training and forward passes of models under ~1B parameters: CNNs, BERT-size and smaller
   transformers; RL in small environments; CPU work): realistic wall-clock hours on the accelerator the work ran on,
   times the number of runs. Small models do not use a fast accelerator fully: estimate how long a run really takes
   on that accelerator. For example, standard training of ResNet-18 on CIFAR-10 for 100 epochs takes about 0.7 hours
   on a GTX 1080 Ti or RTX 2080 Ti, 0.4 on a V100 and 0.3 on an A100, RTX 3090 or RTX 4090, and PGD-10 adversarial
   training about 11 times as long; fine-tuning BERT-base on MNLI (393k pairs, 3 epochs) about 2 hours on a V100 and
   1 on an A100 or RTX 3090.

d. **The largest components decide the total**: check their number of runs (models x datasets x seeds x conditions)
   and their token counts. A paper with no computation of its own: no components.

## Accelerator

Use the accelerator the paper names for the work. A paper that names several: split the work by accelerator, or use
the one that ran most of it. "GPU-hours" without a type: the accelerator the paper names elsewhere. If the paper
names none, assume the one a group like the authors' most likely used for this work in the paper's year, and set
`accelerator_named` to `no`:

- 2020-2021: GTX 1080 Ti or RTX 2080 Ti for small models; V100 for larger jobs.
- 2022-2023: RTX 3090 for small models; A100 or V100 for larger jobs and language models.
- 2024-2025: RTX 3090, RTX 4090 or RTX A6000 for small models; A100 for language models; H100 for the largest jobs.
- Google authors: TPU v3 in 2020-2022, TPU v4 or v5e in 2023-2025. Free Google Colab: T4.

TPUs count per chip: a v2-8, v3-8, v4-8 or v5p-8 device has 4 chips, a v5e-8 or v6e-8 has 8. A K80 board has two
GPUs. CPU-only work: `CPU core`, hours = cores x wall-clock hours. An accelerator not on the list: the closest one by
generation and class.

Accelerators: {ACCELERATORS}.

## Answer

- `components`: as above; empty for a paper with no computation of its own.
- `organizations`: the authors' organizations as the paper's header or footnotes give them, each once, short form
  ("Google DeepMind", "UC Berkeley", "Apollo Research"); empty if the text gives none.
- `note`: at most 30 words: what decides the total and any reading you had to choose.
- `seen`: `yes` if the text states how long the work ran or how much compute or money it used, else `no`.

Answer with the JSON object only.
