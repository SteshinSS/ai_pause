"""Paths, label codes and file helpers shared by the scripts."""
import gzip
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
HARVEST = DATA / "harvest"
PAPERS = DATA / "papers.jsonl.gz"
RUNS = DATA / "runs"
PROMPT = HERE / "prompt.md"

YEARS = range(2020, 2026)
# ML papers: at least one of these categories (the rule of the 2025 census: 127,661 papers in 2025)
ML_CATS = {"cs.LG", "cs.CL", "cs.AI", "cs.CV", "stat.ML", "cs.NE", "cs.MA", "cs.CR", "cs.CY", "cs.HC", "cs.SE",
           "cs.RO", "cs.IR", "cs.MM", "cs.SD", "eess.AS", "eess.IV", "cs.GT"}

IN_TOPICS = ["IN", "SD", "AV", "JB", "UL", "AL", "EV"]          # in scope when the kind is E
OUT_TOPICS = ["CP", "HF", "PR", "FA", "WM", "PL", "SR", "GV"]   # out of scope, neighbours of safety
TOPICS = IN_TOPICS + OUT_TOPICS + ["NO"]
TOPIC_NAMES = {"IN": "interpretability", "SD": "science of deep learning", "AV": "adversarial robustness",
               "JB": "jailbreaks, red-teaming, safeguards", "UL": "unlearning", "AL": "alignment experiments",
               "EV": "dangerous-capability evaluations, control"}
KINDS = ["E", "T", "S"]
GAVE_UP = {"RF": "refused by the model", "ER": "no valid answer in 3 tries"}

ID_RE = re.compile(r"^(\d{2})(\d{2})\.\d{4,5}$")


def year_month(pid: str) -> tuple:
    m = ID_RE.match(pid)
    return 2000 + int(m[1]), int(m[2])


def in_scope(label: dict) -> bool:
    return label["topic"] in IN_TOPICS and label["kind"] == "E"


def load_papers(path=PAPERS, fields=None, ids=None) -> dict:
    """{id: paper}, only the papers in `ids` if given; fills year, month and primary category if the file lacks
    them (the 2025 census file)."""
    papers = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            d = json.loads(line)
            if ids is not None and d["id"] not in ids:
                continue
            if "year" not in d:
                d["year"], d["month"] = year_month(d["id"])
            d.setdefault("primary", d["cats"].split()[0])
            papers[d["id"]] = {k: d[k] for k in fields} if fields else d
    return papers


def read_jsonl(path) -> list:
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, rows) -> None:
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_json(path, default):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def write_json(path, obj) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def load_labels(run_dir) -> dict:
    """{id: label} of a run; the last line written for a paper wins."""
    return {r["id"]: r for r in read_jsonl(Path(run_dir) / "labels.jsonl")}
