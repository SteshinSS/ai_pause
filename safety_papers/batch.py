"""One request per paper through the Message Batches API: send, collect, status, run. Used by flag_compute.py and
estimate_compute.py; each defines a Task (instructions, answer schema, request text, default papers).

A run lives in <task.out>/<run>/: run.json (papers, model, effort, instructions hash), ids.txt, batches.json (batch id
-> paper ids), answers/<batch>.jsonl (raw answers), <task.results> (one parsed answer per paper; key field RF =
refused, ER = no valid answer in 3 tries), tries.json, usage.json. `send` only uploads; `collect` saves the answers;
`run` does both, sends failed papers again and waits, asking for the API key once (see classify.make_client).
"""
import datetime as dt
import hashlib
import json
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from classify import PRICES, dollars, make_client
from common import append_jsonl, read_json, read_jsonl, write_json

TRIES = 3
SEED = 20261003
BATCH_BYTES = 100_000_000   # per batch; the API limit is 256 MB
CHARS_PER_TOKEN = 2.44      # pilot of 3 Oct 2026: 41.3M characters of paper text were 16.9M input tokens


@dataclass
class Task:
    out: Path                 # folder of the runs
    prompt: Path              # instructions (system prompt, cached)
    schema: dict              # JSON schema of the answer
    results: str              # file name of the parsed answers in a run folder
    key: str                  # answer field shown by `status`; RF / ER are written into it
    user: Callable            # (paper id, run.json) -> user message
    size: Callable            # paper id -> characters of the user message, None if the paper has no text
    ids: Callable             # () -> default paper ids
    model: str = "claude-sonnet-5-5"
    max_tokens: int = 8000
    answer_tokens: int = 300  # for --dry-run
    think_tokens: int = 0     # for --dry-run, when thinking is on
    fill: dict = None         # {NAME: text} put in place of {NAME} in the instructions

    def __post_init__(self):
        self.system = self.prompt.read_text()
        for k, v in (self.fill or {}).items():
            self.system = self.system.replace("{" + k + "}", v)
        self.sha = hashlib.sha256(self.system.encode()).hexdigest()[:12]

    def parse(self, text: str):
        try:
            a = json.loads(text)
        except json.JSONDecodeError:
            return None
        props = self.schema["properties"]
        if not isinstance(a, dict) or any(k not in a for k in props):
            return None
        if "enum" in props[self.key] and a[self.key] not in props[self.key]["enum"]:
            return None
        return {k: a[k] for k in props}


def params(task: Task, pid: str, cfg: dict) -> dict:
    model, effort = cfg["model"], cfg.get("effort")
    p = {"model": model, "max_tokens": task.max_tokens,
         "system": [{"type": "text", "text": task.system, "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
         "messages": [{"role": "user", "content": task.user(pid, cfg)}],
         "output_config": {"format": {"type": "json_schema", "schema": task.schema}}}
    if effort:  # adaptive thinking at this effort
        p["thinking"] = {"type": "adaptive"}
        p["output_config"]["effort"] = effort
    elif model == "claude-sonnet-5-5":
        p["thinking"] = {"type": "between_tools"}  # no tools in these requests, so no thinking
    elif model.startswith("claude-opus-5"):
        p["output_config"]["effort"] = "low"  # Opus 5.x cannot switch thinking off
    return p


def open_run(task: Task, args) -> tuple:
    """(run.json, paper ids). The first `send` of a run fixes its papers, model and effort."""
    rd = task.out / args.run
    cfg = read_json(rd / "run.json", None)
    if cfg:
        return cfg, (rd / "ids.txt").read_text().split()
    ids = [x.strip() for x in open(args.ids) if x.strip()] if args.ids else task.ids()
    ids = sorted(i for i in set(ids) if task.size(i) is not None)
    if args.limit:
        ids = sorted(random.Random(SEED).sample(ids, min(args.limit, len(ids))))
    cfg = {"ids_file": str(args.ids or ""), "limit": args.limit, "model": args.model or task.model,
           "effort": args.effort, "prompt": task.sha, "n": len(ids),
           "created": dt.datetime.now().isoformat(timespec="seconds")}
    if not args.dry_run:
        rd.mkdir(parents=True, exist_ok=True)
        write_json(rd / "run.json", cfg)
        (rd / "ids.txt").write_text("\n".join(ids) + "\n")
    return cfg, ids


def send(task: Task, args) -> None:
    cfg, ids = open_run(task, args)
    rd, model = task.out / args.run, cfg["model"]
    done = {r["id"] for r in read_jsonl(rd / task.results)}
    batches, tries = read_json(rd / "batches.json", []), read_json(rd / "tries.json", {})
    pending = {i for b in batches if not b["ended"] for i in b["ids"]}
    todo = [i for i in ids if i not in done and i not in pending]
    gave = [i for i in todo if tries.get(i, 0) >= TRIES]
    if gave and not args.dry_run:
        append_jsonl(rd / task.results, [{"id": i, task.key: "ER"} for i in gave])
    todo = [i for i in todo if tries.get(i, 0) < TRIES]
    effort = f", effort {cfg['effort']}" if cfg.get("effort") else ""
    print(f"{args.run}: {len(ids):,} papers, {len(done):,} answered, {len(pending):,} waiting in open batches, "
          f"{len(gave)} given up, {len(todo):,} to send ({model}{effort})")
    if not todo:
        return
    if args.dry_run:
        pin, pout, pread = PRICES[model]
        tok = sum(task.size(i) for i in todo) / CHARS_PER_TOKEN
        sys_tok = len(task.system) / CHARS_PER_TOKEN
        out = len(todo) * (task.answer_tokens + (task.think_tokens if cfg.get("effort") or "opus" in model else 0))
        usd = 0.5 * (tok * pin + len(todo) * sys_tok * pread + out * pout) / 1e6
        print(f"  ~{tok / 1e6:,.1f}M input tokens, ~{out / 1e6:,.1f}M output tokens (a guess) -> ~${usd:,.0f} at the "
              f"batch price; nothing sent")
        return

    import anthropic
    client = make_client()
    group, size = [], 0

    def submit() -> bool:
        try:
            batch = client.messages.batches.create(requests=group)
        except anthropic.RateLimitError as e:
            print(f"  the batch queue is full ({e.message}); `send` again once some batches have ended")
            return False
        batches.append({"batch_id": batch.id, "ids": [r["custom_id"].replace("_", ".") for r in group],
                        "model": model, "prompt": task.sha,
                        "created": dt.datetime.now().isoformat(timespec="seconds"), "ended": False})
        write_json(rd / "batches.json", batches)
        print(f"  batch {batch.id}: {len(group):,} papers", flush=True)
        return True

    for i in todo:
        r = {"custom_id": i.replace(".", "_"), "params": params(task, i, cfg)}
        b = len(json.dumps(r))
        if group and (len(group) >= 100_000 or size + b > BATCH_BYTES):
            if not submit():
                return
            group, size = [], 0
        group.append(r)
        size += b
    submit()


def collect(task: Task, args) -> None:
    import anthropic
    rd = task.out / args.run
    batches = read_json(rd / "batches.json", [])
    if all(b["ended"] for b in batches):
        status(task, args)
        return
    client = make_client()
    tries, usage = read_json(rd / "tries.json", {}), read_json(rd / "usage.json", {})
    while True:
        waiting = 0
        for b in batches:
            if b["ended"]:
                continue
            try:
                info = client.messages.batches.retrieve(b["batch_id"])
            except anthropic.NotFoundError:
                sys.exit(f"batch {b['batch_id']} not found: batches belong to a workspace, and this key is from "
                         "another one; use a key of the workspace that sent them")
            if info.processing_status != "ended":
                print(f"  {b['batch_id']}: {info.request_counts.processing:,} processing, "
                      f"{info.request_counts.succeeded:,} done", flush=True)
                waiting += 1
                continue
            rows, raw, tally = [], [], Counter()
            for r in client.messages.batches.results(b["batch_id"]):
                pid, res = r.custom_id.replace("_", "."), r.result
                if res.type != "succeeded":
                    raw.append({"id": pid, "type": res.type, "error": str(getattr(res, "error", ""))[:300]})
                    tries[pid] = tries.get(pid, 0) + 1
                    tally[res.type] += 1
                    continue
                m = res.message
                a = usage.setdefault(b["model"], {"requests": 0, "input": 0, "cache_write": 0, "cache_read": 0,
                                                  "output": 0})
                a["requests"] += 1
                a["input"] += m.usage.input_tokens or 0
                a["cache_write"] += m.usage.cache_creation_input_tokens or 0
                a["cache_read"] += m.usage.cache_read_input_tokens or 0
                a["output"] += m.usage.output_tokens or 0
                text = "".join(x.text for x in m.content if x.type == "text")
                raw.append({"id": pid, "stop": m.stop_reason, "text": text})
                ans = None if m.stop_reason == "refusal" else task.parse(text)
                if m.stop_reason == "refusal":
                    rows.append({"id": pid, task.key: "RF"})
                    tally["refused"] += 1
                elif ans is None:
                    tries[pid] = tries.get(pid, 0) + 1
                    tally["without a valid answer"] += 1
                else:
                    rows.append({"id": pid, **ans})
                    tally["answered"] += 1
            (rd / "answers").mkdir(exist_ok=True)
            with open(rd / "answers" / f"{b['batch_id']}.jsonl", "w") as f:
                f.writelines(json.dumps(x) + "\n" for x in raw)
            append_jsonl(rd / task.results, rows)
            b["ended"] = True
            write_json(rd / "tries.json", tries)
            write_json(rd / "usage.json", usage)
            write_json(rd / "batches.json", batches)
            print(f"  {b['batch_id']}: " + ", ".join(f"{v:,} {k}" for k, v in tally.items()), flush=True)
        if not waiting or not args.wait:
            break
        time.sleep(120)
    status(task, args)


def status(task: Task, args) -> None:
    rd = task.out / args.run
    cfg = read_json(rd / "run.json", None)
    if not cfg:
        sys.exit(f"no run {args.run}")
    ids = (rd / "ids.txt").read_text().split()
    res = {r["id"]: r for r in read_jsonl(rd / task.results)}
    batches, usage = read_json(rd / "batches.json", []), read_json(rd / "usage.json", {})
    counts = Counter(str(r.get(task.key)) for r in res.values())
    read = sum(a["cache_read"] for a in usage.values())
    inp = read + sum(a["cache_write"] + a["input"] for a in usage.values())
    left = sum(i not in res for i in ids)
    print(f"run {args.run} ({cfg['model']}): {len(res):,} of {len(ids):,} papers answered ("
          + ", ".join(f"{v:,} {k}" for k, v in counts.most_common()) + f"); "
          f"{sum(not b['ended'] for b in batches)} of {len(batches)} batches open; ${dollars(usage):,.2f}"
          + (f", {100 * read / inp:.0f}% of input tokens from the cache" if inp else "")
          + (f"; {left:,} left: `send` again once no batch is open" if left else "")
          + ("" if {b["prompt"] for b in batches} <= {task.sha} else "; WARNING: the instructions changed since"))


def run(task: Task, args) -> None:
    """send, wait, collect, send again what failed (up to 3 tries per paper)."""
    if args.dry_run:
        send(task, args)
        return
    rd = task.out / args.run
    args.wait = True
    for _ in range(TRIES + 1):
        send(task, args)
        collect(task, args)
        done = {r["id"] for r in read_jsonl(rd / task.results)}
        if all(i in done for i in (rd / "ids.txt").read_text().split()):
            break


def add_arguments(ap, default_run: str) -> None:
    ap.add_argument("--run", default=default_run, help=f"run folder (default {default_run})")
    ap.add_argument("--ids", help="file of arXiv ids: only these papers; first send of a run only")
    ap.add_argument("--limit", type=int, help="this many random papers; first send of a run only")
    ap.add_argument("--model", choices=list(PRICES), help="model; first send of a run only")
    ap.add_argument("--effort", choices=["low", "medium", "high"],
                    help="adaptive thinking at this effort (default: Sonnet without thinking, Opus at low); "
                         "first send of a run only")
    ap.add_argument("--dry-run", action="store_true", help="estimate the cost; send nothing")
    ap.add_argument("--wait", action="store_true", help="collect: check every 2 minutes until all batches end")
