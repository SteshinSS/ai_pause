You read the full text of one machine-learning research paper and report **the compute the authors state for
their own experiments**, as one number in H100e-years, together with every passage that reveals it.

## Why we ask

We are building a dataset of papers whose compute is known from the authors' own words. Later we will delete every
passage that reveals that compute, ask a model to estimate the compute from what remains, and compare its estimate
with the authors' figure. So we need from you:

1. **Every passage that reveals the time, cost, energy or hardware of the authors' runs, copied exactly**, wherever
   it appears (main text, footnotes, table captions and cells, figure captions, appendices, checklists,
   acknowledgements). A passage you miss stays in the paper and leaks the answer. In an earlier test, answers leaked
   through a table column of run times ("6h", "72h"), a per-image runtime in a caption and a stated carbon footprint.
2. **The authors' figure as one number**: the accelerator time their stated figures give, converted to H100e-years,
   and what work that number covers.

Accelerator compute only. Calls to hosted models through an API (OpenAI, Anthropic, Google Gemini or other hosted
models) are not counted at all.

## Quotes

Quote the authors' own runs only, three kinds:

- `amount`: time, cost, energy or FLOP of a whole run, a set of runs or all experiments: "GPU-hours", "GPU-days",
  "TPU-hours", "8 A100s for 3 days", "each run takes 2 hours on one V100", "1.2e20 FLOP", "$3,000 on AWS", "1,500
  kWh", "120 kg CO2eq". An `amount` quote always contains such a figure.
- `speed`: time or cost per smaller unit, or relative speed: per epoch, step, example, query, image or episode ("40
  minutes per epoch", "0.8 s per query"), throughput, run-time columns or rows of tables, comparisons of training or
  inference time between methods ("ours trains in 3 h, the baseline in 10 h", "2x faster").
- `hardware`: the accelerators or machines they ran on (type, count, memory) without a time or amount.

How to copy:

- Copy one contiguous stretch of the text exactly as it appears in the text given to you, character for character,
  including odd spacing, hyphenation and extraction artefacts; you may write a line break as a single space.
- Never join pieces that are apart in the text: no merging of two sentences, no rebuilding of table rows from cells
  that the text shows in another order. If a footnote, page number or heading interrupts a sentence, quote the
  pieces separately. For a table, quote its caption and, separately, each line of the text that holds its times or
  costs, exactly as the text shows them.
- One quote per passage, the shortest stretch that holds the statement (at most about 400 characters). If the same
  statement appears twice (say in the body and in the checklist), quote both.
- Do not quote: inputs to an estimate (numbers of epochs, steps, seeds, runs, samples or tokens; model and dataset
  sizes; batch sizes) unless the same sentence states a time or cost; compute of work the authors did not run
  (pretraining of downloaded models, figures from cited papers, hypothetical systems); time limits inside a task
  ("each agent gets 30 minutes"); human time and pay; checklist answers without a figure or hardware ("[Yes] see
  Appendix C"; quote the passage they point to instead).

## The number

`level` says what the authors' figures give:

- `all`: the accelerator compute of every experiment the paper reports: one total, or figures for every part with the
  number of runs stated where they are per run. Leaving out something plainly tiny is still `all`.
- `part`: the accelerator compute of a well-defined part of the work, completely: for example "the pretraining of
  the 162M model", "the four LoRA fine-tuning runs", "all experiments of Table 2". Per-run, per-epoch or per-example
  figures count only together with the stated number of runs, epochs or examples.
- `project`: only a figure for the whole research project, including preliminary or failed work the paper does not
  report ("the complete suite, including studies not in the paper, used 200-300 GPU hours").
- `other`: figures exist but give no accelerator compute: energy or emissions only, wall-clock time without the
  hardware, per-run times without the number of runs, or figures too vague to read as a number.
- `hardware`: hardware only, no figure.
- `none`: nothing about the authors' own compute, or no computation of their own.

If the work is mostly API calls to hosted models, set `api` to `main` (we leave such papers out); still fill the
other fields for the accelerator part.

For `all`, `part` and `project`, `calc` is one arithmetic expression (numbers, + - * / and parentheses only) whose
value is the H100e-years of the work `covers` describes:

- accelerator-hours x the H100e factor of the accelerator (table below) / 8766 (hours in a year);
- GPU-days x 24, GPU-weeks x 168, GPU-years x 8766;
- "N GPUs for T hours" = N x T accelerator-hours; per-run figures x the stated number of runs;
- FLOP / (989e12 x u x 31557600), where u is the utilization the paper states, else 0.4;
- "GPU-hours" without a type: use the accelerator the paper names for that work; if it names none, the level is
  `other`;
- a range: its midpoint; "several hundred": 300; "500+", "over 500", "under 500", "about 500": 500;
- the stated number of accelerators, not a guess of how busy they were, unless the paper gives a share ("50% of 8
  GPUs").

Example: "each of the 36 fine-tuning runs took 6 hours on 8 A100s" -> `36*6*8*0.32/8766`.

`covers`: for `all`, "all reported experiments"; for `part`, the work the number covers, described so that someone
who reads the paper without the figures can tell exactly which runs are meant, without any time, cost or hardware
figure in it; for `project`, "the whole project".

`project_calc`: when the level is `all` or `part` and the paper also gives a whole-project figure (including
preliminary or failed work), its expression; else empty.

H100e factor = H100-hours per accelerator-hour (dense 16-bit throughput):
H100 / H800 / H200 / GH200 1.0; H20 0.15; B200 2.3; GB200 2.5; A100 / A800 0.32; RTX A6000 0.16; RTX 6000 Ada 0.37;
L40S 0.37; L40 0.18; A40 0.15; A10 / A10G 0.13; A5000 0.11; A4000 / A4500 0.08; RTX 5090 0.21; RTX 4090 0.17;
RTX 4080 0.10; RTX 3090 0.07; RTX 3080 0.06; RTX 2080 Ti / Titan RTX 0.06; V100 0.13; P100 0.02; T4 0.07;
TPU v3 0.12, v4 0.28, v5e 0.20, v5p 0.46, v6e 0.93 (per chip); Ascend 910B 0.32; MI300X 1.3; MI250X 0.39;
MI210 0.18; Gaudi 2 0.43; Gaudi 3 1.85; Apple silicon or CPU 0.01; Titan V 0.11; Quadro RTX 8000 / 6000 0.13;
RTX 2080 / 2070 0.04; GTX 1080 Ti, 1080, Titan X (Pascal), Titan Xp 0.011; GTX 980 / 980 Ti, Titan X (Maxwell)
0.006; GTX 1070 / 1060 0.007; P40 0.012; P4 0.006; M40 0.007; K40 0.005; K80 0.005 (per GPU; a K80 board has two);
TPU v2 0.045 (per chip; a v2-8 or v3-8 device has 4 chips). An accelerator not listed: the closest one by
generation and class.

## Fields, in this order

- `quotes`: every passage of the three kinds, in the order they appear, each `{"kind": ..., "text": ...}`.
- `stated`: the authors' figures in their own units, short, with what each covers, joined by "; ".
- `api`: `none`, `minor` (a judge or data generation next to local experiments) or `main`.
- `level`, `covers`, `calc`, `project_calc`: as above; empty strings where they do not apply.
- `why`: at most 25 words: what the number covers, what it leaves out, and any reading you had to choose.

## How to read

The text was extracted from a PDF: two-column layouts may interleave, tables lose their grid, hyphenated words may
be split. Read the whole paper, including the appendices and any checklist at the end; these passages are most often
in an "Implementation details", "Experimental setup", "Compute resources" or "Training details" section, a table or
figure caption, a footnote, the acknowledgements, or the NeurIPS / ICLR reproducibility checklist. Judge `all`
against the list of experiments the paper actually reports: a paper that states the time of its main training run
but also runs large evaluation sweeps on 70B models is `part`, with `covers` naming the training run.

Answer with the JSON object only.
