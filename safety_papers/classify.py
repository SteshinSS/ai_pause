#!/usr/bin/env python3
"""Label every paper from its title and abstract with Claude: one request per paper, through the Batches API.

A request = the instructions (prompt.md, ~6,800 tokens, cached) + one paper's id, categories, title and abstract.
The answer is JSON fixed by a schema: why (at most 15 words), topic, kind, conf. The Message Batches API takes up to
100,000 requests at a time and answers within about an hour, at half price.

A paper whose answer is missing or broken is sent again by the next `send` (3 tries, then ER). A paper the model
refuses (its safety filter can trip on bio or cyber abstracts) is labelled RF, to be read by hand.

    python classify.py send --dry-run   # count requests, tokens and dollars; send nothing
    python classify.py send             # every paper of data/papers.jsonl.gz without a label
    python classify.py collect --wait   # fetch the answers; `send` again if it says papers are left
    python classify.py run              # send, wait, collect, send again what failed: one command, one key paste
    python classify.py status

Other runs: the first `send` with --run NAME fixes the run's papers and model (--papers FILE, --ids FILE, --limit N,
--model M); later commands only need --run NAME. `send` and `collect` ask for the API key (hidden while you type
or paste it) unless ANTHROPIC_API_KEY is set. Files in data/runs/<run>/: run.json,
ids.txt, batches.json (batch id -> paper ids), answers/<batch>.jsonl (raw), labels.jsonl, tries.json, usage.json.
"""
import argparse
import datetime as dt
import getpass
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

from common import (GAVE_UP, KINDS, PAPERS, PROMPT, RUNS, TOPICS, append_jsonl, in_scope, load_labels, load_papers,
                    read_json, write_json)

MODEL = "claude-sonnet-5-5"   # default model of a new run
OPUS_EFFORT = "low"           # Opus 5.x cannot switch thinking off; keep it short
TRIES = 3
SEED = 20261002
BATCH_BYTES = 200_000_000     # per batch; the API limit is 256 MB and 100,000 requests
# $ per million tokens (input, output, cache read), claude-api skill of 25 Sep 2026. Batches cost half; writing
# the instructions to the 1-hour cache costs twice the input price.
PRICES = {"claude-sonnet-5-5": (2.0, 10.0, 0.20), "claude-opus-5-5": (4.0, 20.0, 0.20),
          "claude-haiku-4-5": (1.0, 5.0, 0.10)}

SYSTEM = PROMPT.read_text()
PROMPT_SHA = hashlib.sha256(SYSTEM.encode()).hexdigest()[:12]
FIELDS = {"why": {"type": "string"}, "topic": {"type": "string", "enum": TOPICS},
          "kind": {"type": "string", "enum": KINDS}, "conf": {"type": "string", "enum": ["h", "l"]}}
SCHEMA = {"type": "object", "properties": FIELDS, "required": list(FIELDS), "additionalProperties": False}


def params(model: str, d: dict) -> dict:
    p = {"model": model, "max_tokens": 16000,
         "system": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
         "messages": [{"role": "user", "content": f'<paper id="{d["id"]}" categories="{d["cats"]}">\n'
                       f'<title>{d["title"]}</title>\n<abstract>{d["abstract"]}</abstract>\n</paper>'}],
         "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}}}
    if model == "claude-sonnet-5-5":
        p["thinking"] = {"type": "between_tools"}  # no tools in these requests, so no thinking
    elif model.startswith("claude-opus-5"):
        p["output_config"]["effort"] = OPUS_EFFORT
    return p


def make_client():
    """Anthropic client. Without ANTHROPIC_API_KEY it asks for the key once per command (later calls in the same
    command reuse it; nothing is saved to disk) and checks it with a free request before anything is sent."""
    import anthropic
    if not os.environ.get("ANTHROPIC_API_KEY"):
        key = getpass.getpass("Paste your Anthropic API key and press Enter (the key stays hidden): ").strip()
        key = key.removeprefix("export ").removeprefix("ANTHROPIC_API_KEY=").strip("'\" ")
        if not key:
            sys.exit("no API key")
        os.environ["ANTHROPIC_API_KEY"] = key  # this process only
        client = anthropic.Anthropic()
        try:
            client.models.retrieve(MODEL)
        except anthropic.AuthenticationError:
            sys.exit(f"the API rejected the key {key[:7]}...{key[-4:]} ({len(key)} characters); run again and paste "
                     "the whole key")
        print(f"  key {key[:7]}...{key[-4:]} accepted")
    return anthropic.Anthropic()


def parse(text: str):
    try:
        a = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(a, dict) and a.get("topic") in TOPICS and a.get("kind") in KINDS and a.get("conf") in ("h", "l"):
        return {k: a.get(k, "") for k in FIELDS}
    return None


def open_run(args) -> tuple:
    """(run.json, paper ids). The first `send` of a run fixes its papers and model."""
    rd = RUNS / args.run
    cfg = read_json(rd / "run.json", None)
    if cfg:
        if args.papers or args.ids or args.limit or args.model:
            print(f"run {args.run} exists: --papers, --ids, --limit and --model are ignored")
        return cfg, (rd / "ids.txt").read_text().split()
    path = Path(args.papers).resolve() if args.papers else PAPERS
    if not path.exists():
        sys.exit(f"no {path}: run harvest.py, or pass --papers")
    ids = sorted(load_papers(path, fields=["id"]))
    if args.ids:
        want = {line.strip() for line in open(args.ids) if line.strip()}
        ids = [i for i in ids if i in want]
        if len(ids) < len(want):
            print(f"  {len(want) - len(ids)} of {len(want)} ids in {args.ids} are not in {path.name}")
    if args.limit:
        ids = sorted(random.Random(SEED).sample(ids, min(args.limit, len(ids))))
    cfg = {"papers": str(path), "ids_file": args.ids, "limit": args.limit, "model": args.model or MODEL,
           "n": len(ids), "created": dt.datetime.now().isoformat(timespec="seconds")}
    if not args.dry_run:
        rd.mkdir(parents=True, exist_ok=True)
        write_json(rd / "run.json", cfg)
        (rd / "ids.txt").write_text("\n".join(ids) + "\n")
    return cfg, ids


def dollars(usage: dict) -> float:
    total = 0.0
    for model, a in usage.items():
        pin, pout, pread = PRICES.get(model, (0.0, 0.0, 0.0))
        total += 0.5 * (a["input"] * pin + 2 * a["cache_write"] * pin + a["cache_read"] * pread
                        + a["output"] * pout) / 1e6
    return total


def estimate(todo: list, papers: dict, model: str) -> None:
    sys_tokens = len(SYSTEM) / 3.6
    paper_tokens = sum(len(papers[i]["title"]) + len(papers[i]["abstract"]) + 90 for i in todo) / 3.8
    out = len(todo) * (40 if model == "claude-sonnet-5-5" else 200)  # Opus: + ~150 thinking tokens (a guess)
    pin, pout, pread = PRICES[model]
    base = paper_tokens * pin + out * pout
    print(f"  {model}: {len(todo):,} requests, {paper_tokens / 1e6:,.1f}M paper tokens, instructions "
          f"~{sys_tokens:,.0f} tokens each, ~{out / 1e6:,.1f}M output tokens -> "
          f"${0.5 * (base + len(todo) * sys_tokens * pread) / 1e6:,.0f} if the instructions are read from the cache, "
          f"${0.5 * (base + len(todo) * sys_tokens * pin) / 1e6:,.0f} if never; ~3.8 characters per token")


def send(args) -> None:
    cfg, ids = open_run(args)
    rd, model = RUNS / args.run, cfg["model"]
    labels = load_labels(rd)
    batches = read_json(rd / "batches.json", [])
    tries = read_json(rd / "tries.json", {})
    pending = {i for b in batches if not b["ended"] for i in b["ids"]}
    todo = [i for i in ids if i not in labels and i not in pending]
    gave = [i for i in todo if tries.get(i, 0) >= TRIES]
    if gave and not args.dry_run:
        append_jsonl(rd / "labels.jsonl", [{"id": i, "topic": "ER", "kind": "", "conf": "", "why": ""} for i in gave])
    todo = [i for i in todo if tries.get(i, 0) < TRIES]
    print(f"{len(ids):,} papers, {len(labels):,} labelled, {len(pending):,} waiting in open batches, "
          f"{len(gave)} given up, {len(todo):,} to send ({model})")
    if not todo:
        return
    papers = load_papers(cfg["papers"], fields=["id", "cats", "title", "abstract"])
    if args.dry_run:
        rd.mkdir(parents=True, exist_ok=True)
        write_json(rd / "dry_run_request.json", {"custom_id": todo[0].replace(".", "_"),
                                                 "params": params(model, papers[todo[0]])})
        estimate(todo, papers, model)
        print(f"  first request -> {rd / 'dry_run_request.json'}; nothing sent")
        return

    import anthropic
    client = make_client()

    def submit(group: list) -> bool:
        try:
            batch = client.messages.batches.create(requests=group)
        except anthropic.RateLimitError as e:
            print(f"  the batch queue is full ({e.message}); `send` again once some batches have ended")
            return False
        batches.append({"batch_id": batch.id, "ids": [r["custom_id"].replace("_", ".") for r in group],
                        "model": model, "prompt": PROMPT_SHA,
                        "created": dt.datetime.now().isoformat(timespec="seconds"), "ended": False})
        write_json(rd / "batches.json", batches)
        print(f"  batch {batch.id}: {len(group):,} papers")
        return True

    group, size = [], 0
    for i in todo:
        r = {"custom_id": i.replace(".", "_"), "params": params(model, papers[i])}
        b = len(json.dumps(r))
        if group and (len(group) >= 100_000 or size + b > BATCH_BYTES):
            if not submit(group):
                return
            group, size = [], 0
        group.append(r)
        size += b
    submit(group)


def read_answers(client, b: dict, rd: Path, tries: dict, usage: dict) -> Counter:
    rows, raw, tally = [], [], Counter()
    for r in client.messages.batches.results(b["batch_id"]):
        pid, res = r.custom_id.replace("_", "."), r.result
        if res.type != "succeeded":
            err = getattr(res, "error", None)
            raw.append({"id": pid, "type": res.type, "error": str(getattr(err, "error", err) or "")[:300]})
            tries[pid] = tries.get(pid, 0) + 1
            tally[res.type] += 1
            continue
        m = res.message
        a = usage.setdefault(b["model"], {"requests": 0, "input": 0, "cache_write": 0, "cache_read": 0, "output": 0})
        a["requests"] += 1
        a["input"] += m.usage.input_tokens or 0
        a["cache_write"] += m.usage.cache_creation_input_tokens or 0
        a["cache_read"] += m.usage.cache_read_input_tokens or 0
        a["output"] += m.usage.output_tokens or 0
        text = "".join(x.text for x in m.content if x.type == "text")
        raw.append({"id": pid, "stop": m.stop_reason, "text": text})
        if m.stop_reason == "refusal":
            rows.append({"id": pid, "topic": "RF", "kind": "", "conf": "", "why": ""})
            tally["refused"] += 1
            continue
        lab = parse(text)
        if lab is None:
            tries[pid] = tries.get(pid, 0) + 1
            tally["without a valid answer"] += 1
        else:
            rows.append({"id": pid, **lab})
            tally["labelled"] += 1
    (rd / "answers").mkdir(exist_ok=True)
    with open(rd / "answers" / f"{b['batch_id']}.jsonl", "w") as f:
        for x in raw:
            f.write(json.dumps(x) + "\n")
    append_jsonl(rd / "labels.jsonl", rows)
    return tally


def collect(args) -> None:
    client = make_client()
    rd = RUNS / args.run
    batches = read_json(rd / "batches.json", [])
    tries, usage = read_json(rd / "tries.json", {}), read_json(rd / "usage.json", {})
    while True:
        waiting = 0
        for b in batches:
            if b["ended"]:
                continue
            info = client.messages.batches.retrieve(b["batch_id"])
            if info.processing_status != "ended":
                c = info.request_counts
                print(f"  {b['batch_id']}: {c.processing:,} processing, {c.succeeded:,} done")
                waiting += 1
                continue
            tally = read_answers(client, b, rd, tries, usage)
            b["ended"] = True
            write_json(rd / "tries.json", tries)
            write_json(rd / "usage.json", usage)
            write_json(rd / "batches.json", batches)
            print(f"  {b['batch_id']}: " + ", ".join(f"{v:,} {k}" for k, v in tally.items()))
        if not waiting or not args.wait:
            break
        time.sleep(120)
    status(args)


def run(args) -> None:
    """send, wait, collect, send again what failed (up to 3 tries per paper)."""
    if args.dry_run:
        send(args)
        return
    args.wait = True
    for _ in range(TRIES + 1):
        send(args)
        args.papers = args.ids = args.limit = args.model = None  # the run exists now
        collect(args)
        rd = RUNS / args.run
        if len(load_labels(rd)) >= len((rd / "ids.txt").read_text().split()):
            break


def status(args) -> None:
    rd = RUNS / args.run
    cfg = read_json(rd / "run.json", None)
    if not cfg:
        sys.exit(f"no run {args.run}")
    ids = (rd / "ids.txt").read_text().split()
    labels = load_labels(rd)
    batches = read_json(rd / "batches.json", [])
    usage = read_json(rd / "usage.json", {})
    gave = Counter(lab["topic"] for lab in labels.values() if lab["topic"] in GAVE_UP)
    left = sum(i not in labels for i in ids)
    read = sum(a["cache_read"] for a in usage.values())
    inp = read + sum(a["cache_write"] + a["input"] for a in usage.values())
    print(f"run {args.run} ({cfg['model']}): {len(ids) - left:,} of {len(ids):,} papers labelled"
          + "".join(f", {v} {k}" for k, v in gave.items())
          + f", {sum(in_scope(lab) for lab in labels.values()):,} in scope; "
          f"{sum(not b['ended'] for b in batches)} of {len(batches)} batches open; ${dollars(usage):,.2f}"
          + (f", {100 * read / inp:.0f}% of input tokens read from the cache" if inp else "")
          + (f"; {left:,} left: `send` again once no batch is open" if left else "")
          + ("" if {b["prompt"] for b in batches} <= {PROMPT_SHA} else "; WARNING: prompt.md changed since"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["send", "collect", "run", "status"])
    ap.add_argument("--run", default="census", help="folder in data/runs (default census)")
    ap.add_argument("--papers", help="papers file (default data/papers.jsonl.gz); first send of a run only")
    ap.add_argument("--ids", help="file of arXiv ids: only these papers; first send of a run only")
    ap.add_argument("--limit", type=int, help="this many random papers; first send of a run only")
    ap.add_argument("--model", choices=list(PRICES), help=f"default {MODEL}; first send of a run only")
    ap.add_argument("--dry-run", action="store_true", help="build the requests and estimate the cost; send nothing")
    ap.add_argument("--wait", action="store_true", help="collect: check every 2 minutes until all batches end")
    args = ap.parse_args()
    {"send": send, "collect": collect, "run": run, "status": status}[args.action](args)


if __name__ == "__main__":
    main()
