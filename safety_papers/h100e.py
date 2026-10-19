"""Compute in H100e-years as AI 2027's compute forecast counts it (ai-2027.com/research/compute-forecast): an
accelerator's Total Processing Performance (TPP, the US export-control metric: 2 x dense multiply-accumulate TOPS x
the bit length of the operands, the largest over number formats) over the H100 SXM's 15,832. An H100e-year is one
H100's TPP used for a year, so accelerator-hours convert at TPP / 15,832 / 8766 whatever the job.

The models report accelerator-hours on a named accelerator, or FLOP and the kind of work; this module converts. FLOP
convert at the H100's dense 16-bit 989.4e12 FLOP/s (its TPP / 16) and a utilization by kind of work. A paper's
compute with dead ends is the whole-project figure the paper states, else DEAD_ENDS x its reported experiments.

TPP from dense (non-sparse) peak throughput in vendor specifications. GeForce cards count at their FP16-accumulate
(or INT8 / FP8) tensor rate, twice the FP32-accumulate rate; cards without tensor cores count FP32 FLOP/s x 32, as
TPP does.
"""
import re

from flag_compute import evaluate

H100_TPP = 15832
H100_FLOPS = 989.4e12
HOURS = 8766
SECONDS = 31_557_600
DEAD_ENDS = 2.5
# Calibrated per model on the blind test (~680 papers, blind.py score; stable when fitted on either half): hours =
# median authors' figure / estimate over papers estimated mostly in hours (the authors' runs take longer than the
# model expects; estimated hours only, stated hours unchanged); utilization = the one that makes the median
# FLOP-based estimate equal the authors' figure, over papers at least 80% FLOP before calibration, by the kind of work
# holding most of it (blind.flop_kinds).
CALIBRATION = {
    "claude-sonnet-5-5": {"hours": 2.2, "utilization": {"train": 0.10, "forward": 0.033, "generate": 0.026,
                                                        "mixed": 0.071}},
    "claude-opus-5-5": {"hours": 1.4, "utilization": {"train": 0.10, "forward": 0.064, "generate": 0.041,
                                                      "mixed": 0.14}},
}
MODEL = "claude-sonnet-5-5"  # the model whose calibration applies when none is named
# FLOP the authors state: no model error to correct, so the 40% AI 2027 and AIFM use (as the blind test's readings
# converted the authors' figures)
STATED_UTILIZATION = 0.4
# Before calibration: the model's hours as given, FLOP at 40% utilization (as AI 2027 and AIFM turn H100e into FLOP).
RAW = {"hours": 1.0, "utilization": {"train": 0.4, "forward": 0.4, "generate": 0.4, "mixed": 0.4}}

TPP = {
    # NVIDIA data centre
    "H100 / H800 / H200 / GH200": 15832, "H100 PCIe / H800 PCIe": 12104, "H100 NVL": 13368, "H20": 2368,
    "B200": 36000, "GB200": 40000, "B300 / GB300": 60000,
    "A100 / A800": 4992, "A30": 2640, "A10": 2000, "A10G": 1120, "A40": 2395,
    "L40S": 5864, "L40": 2896, "L4": 1940,
    "V100": 2000, "P100": 339, "P40": 376, "P4": 176, "T4": 1040,
    "K80 (per GPU; a board has two)": 140, "K40": 137, "M40": 224,
    # NVIDIA workstation
    "RTX PRO 6000 Blackwell": 8000, "RTX 6000 Ada": 5828, "RTX 5000 Ada": 4178, "RTX 4000 Ada": 1227,
    "RTX A6000": 2478, "RTX A5000 / A5500": 1778, "RTX A4000 / A4500": 1227,
    "Quadro RTX 8000 / 6000": 2088, "Quadro RTX 5000": 1427, "Quadro RTX 4000": 912,
    "Titan V / Quadro GV100": 1760, "Quadro P6000 / P5000": 350,
    # NVIDIA GeForce
    "RTX 5090": 6704, "RTX 5080": 3602, "RTX 4090": 5285, "RTX 4080": 3119, "RTX 4070 Ti": 2565, "RTX 4070": 1864,
    "RTX 4060 / 4060 Ti": 1200, "RTX 3090 / 3090 Ti": 2272, "RTX 3080 / 3080 Ti": 1904, "RTX 3070 / 3070 Ti": 1301,
    "RTX 3060 / 3060 Ti": 816, "Titan RTX": 2088, "RTX 2080 Ti": 1722, "RTX 2080 / 2080 Super": 1288,
    "RTX 2070 / 2070 Super": 955, "RTX 2060": 824, "GTX 1660 / 1650": 174,
    "Titan Xp / Titan X (Pascal)": 389, "GTX 1080 Ti": 363, "GTX 1080 / 1070 Ti": 284, "GTX 1070": 207,
    "GTX 1060 / 1050 Ti": 140, "GTX Titan X (Maxwell) / 980 Ti": 214, "GTX 980 / 970": 159,
    # AMD
    "MI355X / MI350X": 40000, "MI300X / MI325X": 20919, "MI250X": 6128, "MI250": 5792, "MI210": 2896,
    "MI100": 2954, "MI50 / MI60": 424,
    # Google TPU, per chip
    "TPU v2 (per chip; a v2-8 has 4)": 720, "TPU v3 (per chip; a v3-8 has 4)": 1968,
    "TPU v4 (per chip; a v4-8 has 4)": 4400, "TPU v5e (per chip)": 3152, "TPU v5p (per chip; a v5p-8 has 4)": 7344,
    "TPU v6e (per chip)": 14688,
    # others
    "Gaudi 2": 6920, "Gaudi 3": 29360, "Ascend 910B": 5120, "Ascend 910": 4096, "Trainium / Inferentia2": 3040,
    "Trainium2": 10672, "Apple silicon (M1-M4)": 300, "CPU core": 5,
}
ACCELERATORS = list(TPP)


def factor(accelerator: str) -> float:
    """H100e per accelerator."""
    return TPP[accelerator] / H100_TPP


BIG = re.compile(r"(?<![\w.])\d+(?:\.\d+)?[eE]\+?\d+")
HOUR = re.compile(r"(?<![\w.])(?:3600|3\.6[eE]\+?3)(?![\w.])")


def seconds_as_hours(expr: str) -> bool:
    """An hours amount the model worked out as FLOP over an accelerator's FLOP/s (a division and a number of 1e11 or
    more) without dividing by 3600 in the end, so its value is in seconds. Seen in 3 of 25,702 census answers
    ("8e20/(275e12*0.3)", "3.9e20/5e13/3600*3600"). Found by putting 7200 for 3600: a value in hours halves, one in
    seconds stays."""
    if "/" not in (expr or "") or not any(float(m.group(0)) >= 1e11 for m in BIG.finditer(expr)):
        return False
    v, w = evaluate(expr, zero=True), evaluate(HOUR.sub("7200", expr), zero=True)
    return bool(v) and w is not None and abs(w / v - 1) < 1e-9


def component(c: dict, model: str = MODEL, raw: bool = False):
    """H100e-years of one component of an answer (estimate_prompt.md) by `model`, calibrated unless `raw`; None if
    its amount does not evaluate."""
    cal = RAW if raw else CALIBRATION[model]
    v = evaluate(c.get("amount", ""), zero=True)
    if v is None:
        return None
    if c.get("unit") == "FLOP":
        u = STATED_UTILIZATION if c.get("source") == "stated" else cal["utilization"][c.get("kind") or "train"]
        return v / (H100_FLOPS * u * SECONDS)
    if c.get("accelerator") not in TPP:
        return None
    if seconds_as_hours(c.get("amount", "")):
        v /= 3600
    return v * factor(c["accelerator"]) / HOURS * (cal["hours"] if c.get("source") == "estimated" else 1)


def totals(answer: dict, model: str = MODEL, raw: bool = False) -> dict:
    """reported: the reported experiments; project: a whole-project figure the paper states (None without one);
    with_dead_ends: project, else DEAD_ENDS x reported; a project figure below the reported total is a misreading
    and is ignored. None where a component does not evaluate."""
    parts = answer.get("components")
    if not isinstance(parts, list):
        return {"reported": None, "project": None, "with_dead_ends": None}
    out = {}
    for scope in ("reported", "project"):
        vals = [component(c, model, raw) for c in parts if c.get("scope") == scope]
        out[scope] = None if None in vals else sum(vals) if vals else (0.0 if scope == "reported" else None)
    rep, proj = out["reported"], out["project"]
    if proj is not None and rep is not None and proj < rep:
        proj = None
    out["with_dead_ends"] = proj if proj is not None else None if rep is None else DEAD_ENDS * rep
    return out
