#!/usr/bin/env python3
"""Citation counts of every safety paper from Semantic Scholar (S2) -> data/s2.json (cache, resumable).

S2's batch lookup by arXiv id, 400 papers a request, no key and no e-mail; S2 throttles anonymous calls, so a
refused call is retried every 15 s. A paper S2 does not know gets null. Counts are as of the day of the download
(in data/s2.json under "_fetched").

    python citations.py      # ~65 requests, a few minutes
"""
import csv
import datetime as dt
import json
import subprocess
import time

from common import DATA, RUNS

S2 = "https://api.semanticscholar.org/graph/v1/paper/batch"
FIELDS = "citationCount,influentialCitationCount,venue,publicationDate"
CACHE = DATA / "s2.json"


def post(ids: list) -> list:
    for _ in range(40):
        r = subprocess.run(["curl", "-s", "-m", "120", "-w", "\n%{http_code}", "-X", "POST", f"{S2}?fields={FIELDS}",
                            "-H", "Content-Type: application/json",
                            "-d", json.dumps({"ids": [f"ARXIV:{i}" for i in ids]})], capture_output=True, text=True)
        body, _, code = r.stdout.rpartition("\n")
        if code == "200":
            return json.loads(body)
        time.sleep(15)
    raise RuntimeError("Semantic Scholar refused 40 times in a row")


def main() -> None:
    ids = [r["id"] for r in csv.DictReader(open(RUNS / "census" / "safety_papers.csv"))]
    out = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = [i for i in ids if i not in out]
    for k in range(0, len(todo), 400):
        chunk = todo[k:k + 400]
        for i, rec in zip(chunk, post(chunk)):
            out[i] = rec and {f: rec.get(f) for f in FIELDS.split(",")}
        out["_fetched"] = dt.date.today().isoformat()
        CACHE.write_text(json.dumps(out))
        print(f"{k + len(chunk):,} / {len(todo):,}", flush=True)
    print(f"{sum(bool(out.get(i)) for i in ids):,} of {len(ids):,} papers found -> {CACHE}")


if __name__ == "__main__":
    main()
