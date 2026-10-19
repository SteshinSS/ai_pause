#!/usr/bin/env python3
"""Full text of the listed papers -> data/fulltext/<id>.txt, for the readers of the next steps.

Source: arXiv's public copy of its PDFs on Google Cloud Storage (bucket arxiv-dataset, the one behind Kaggle's arXiv
dataset), which needs no key and has no request limit, unlike arxiv.org (robots.txt: 15 s between requests). The
latest version in the bucket is taken, its MD5 checked against the bucket's, the text extracted by pdftotext
(poppler) and the PDF deleted: the 25,703 papers of the 3 Oct 2026 census are ~59 GB of PDF, ~2 GB of text. A paper
missing from the bucket is fetched from arxiv.org afterwards, one request per 15 s. Resumable: papers already in
data/fulltext/index.jsonl with text are skipped.

    python fulltext.py                  # the papers of data/runs/census/safety_papers.csv, ~1 h
    python fulltext.py --run pilot      # another run's list
    python fulltext.py --ids FILE       # any arXiv ids, one per line
    python fulltext.py --limit 50       # the first 50, to test

A text file: one header line (arXiv id with version, source, pages, characters), then the text with whitespace
collapsed, blank lines dropped and lines wrapped at 160 characters, then END OF PAPER. data/fulltext/index.csv: one
row per paper: version, bucket object and generation, MD5, PDF bytes, pages, characters, status (ok; short = under
5,000 characters, likely a scan or a withdrawal notice; missing = no PDF anywhere).
"""
import argparse
import base64
import csv
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from common import DATA, RUNS, append_jsonl, read_jsonl

OUT = DATA / "fulltext"
TMP = OUT / "tmp"
INDEX = OUT / "index.jsonl"
BUCKET = "arxiv-dataset"
LIST = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
GET = f"https://storage.googleapis.com/{BUCKET}/"
ARXIV_DELAY = 15
SHORT = 5_000
MIN_FREE = 5e9  # bytes of free disk below which the run stops
UA = "safety-papers-census/1.0 (bibliometrics)"
FIELDS = ["id", "version", "source", "object", "generation", "md5", "pdf_bytes", "pages", "chars", "status",
          "fetched", "tool"]


def http(url: str, dest=None):
    """Body of a GET (or into dest, with its MD5); None on 404; 3 retries with growing waits on other errors."""
    for wait in (5, 30, 120, None):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=120) as r:
                if dest is None:
                    return r.read()
                md5 = hashlib.md5()
                with open(dest, "wb") as f:
                    while chunk := r.read(1 << 20):
                        md5.update(chunk)
                        f.write(chunk)
                return base64.b64encode(md5.digest()).decode()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            err = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            err = e
        if wait is None:
            raise RuntimeError(f"{url}: {err}")
        time.sleep(wait)


def latest(pid: str):
    """The bucket's latest version of a paper: {name, generation, md5Hash, size} or None."""
    q = urllib.parse.urlencode({"prefix": f"arxiv/arxiv/pdf/{pid[:4]}/{pid}v",
                                "fields": "items(name,generation,md5Hash,size)"})
    body = http(f"{LIST}?{q}")
    items = json.loads(body).get("items", []) if body else []
    return max(items, key=lambda x: int(x["name"].rsplit("v", 1)[1].removesuffix(".pdf")), default=None)


def extract(pdf, pid: str, version: str, source: str) -> tuple:
    """(pages, characters) of the text written to OUT/<id>.txt."""
    r = subprocess.run(["pdftotext", "-enc", "UTF-8", str(pdf), "-"], capture_output=True, text=True,
                       errors="replace", timeout=300)
    pages = r.stdout.count("\f")
    lines = []
    for line in r.stdout.split("\n"):
        line = " ".join(line.split())
        if line:
            lines += textwrap.wrap(line, 160) if len(line) > 160 else [line]
    text = "\n".join(lines)
    tmp = OUT / f".{pid}.txt"
    tmp.write_text(f"arXiv:{pid}{version} | source: {source} | {pages} pages | {len(text):,} characters\n\n"
                   f"{text}\nEND OF PAPER\n")
    tmp.replace(OUT / f"{pid}.txt")
    return pages, len(text)


def row(pid: str, **kw) -> dict:
    r = {k: "" for k in FIELDS}
    r.update(id=pid, fetched=dt.date.today().isoformat(), **kw)
    if r["status"] != "missing":
        r["status"] = "ok" if r["chars"] >= SHORT else "short"
    return r


def from_bucket(pid: str, tool: str) -> dict:
    obj = latest(pid)
    if obj is None:
        return row(pid, status="missing")
    version = "v" + obj["name"].rsplit("v", 1)[1].removesuffix(".pdf")
    pdf = TMP / f"{pid}.pdf"
    try:
        md5 = http(GET + urllib.parse.quote(obj["name"]), pdf)
        if md5 != obj["md5Hash"]:
            raise RuntimeError(f"{pid}: MD5 {md5} is not the bucket's {obj['md5Hash']}")
        pages, chars = extract(pdf, pid, version, "gcs")
    finally:
        pdf.unlink(missing_ok=True)
    return row(pid, version=version, source="gcs", object=obj["name"], generation=obj["generation"], md5=md5,
               pdf_bytes=int(obj["size"]), pages=pages, chars=chars, status="", tool=tool)


def from_arxiv(pid: str, tool: str) -> dict:
    pdf = TMP / f"{pid}.pdf"
    try:
        md5 = http(f"https://arxiv.org/pdf/{pid}", pdf)
        if md5 is None:
            return row(pid, status="missing")
        pages, chars = extract(pdf, pid, "", "arxiv.org")
        size = pdf.stat().st_size
    finally:
        pdf.unlink(missing_ok=True)
    return row(pid, source="arxiv.org", md5=md5, pdf_bytes=size, pages=pages, chars=chars, status="", tool=tool)


def write_csv() -> dict:
    last = {r["id"]: r for r in read_jsonl(INDEX)}
    with open(OUT / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(last[i] for i in sorted(last))
    return last


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="census", help="take the ids of data/runs/<run>/safety_papers.csv")
    ap.add_argument("--ids", help="file of arXiv ids instead")
    ap.add_argument("--limit", type=int, help="only the first N ids")
    ap.add_argument("--workers", type=int, default=8, help="parallel downloads from the bucket (default 8)")
    args = ap.parse_args()

    if args.ids:
        ids = [x.strip() for x in open(args.ids) if x.strip()]
    else:
        ids = [r["id"] for r in csv.DictReader(open(RUNS / args.run / "safety_papers.csv"))]
    ids = ids[:args.limit] if args.limit else ids
    if not shutil.which("pdftotext"):
        sys.exit("pdftotext not found: brew install poppler")
    tool = subprocess.run(["pdftotext", "-v"], capture_output=True, text=True).stderr.split("\n")[0]
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(exist_ok=True)
    done = {r["id"] for r in read_jsonl(INDEX) if r["status"] in ("ok", "short")}
    todo = [i for i in ids if i not in done]
    print(f"{len(ids):,} papers, {len(ids) - len(todo):,} have text, {len(todo):,} to fetch ({tool})", flush=True)

    missing, k, t0 = [], 0, time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        jobs = {pool.submit(from_bucket, i, tool): i for i in todo}
        for job in as_completed(jobs):
            pid = jobs[job]
            try:
                r = job.result()
            except Exception as e:  # noqa: BLE001 (logged; the paper is tried again by the next run)
                print(f"  {pid}: {e}", flush=True)
                continue
            if r["status"] == "missing":
                missing.append(pid)
            else:
                append_jsonl(INDEX, [r])
            k += 1
            if k % 200 == 0:
                free = shutil.disk_usage(OUT).free
                print(f"  {k:,} of {len(todo):,}, {(time.time() - t0) / 60:.0f} min, {len(missing)} not in the "
                      f"bucket, {free / 1e9:.0f} GB free", flush=True)
                if free < MIN_FREE:
                    pool.shutdown(cancel_futures=True)
                    sys.exit("less than 5 GB of free disk: stopped; run again after freeing space")

    if missing:
        print(f"{len(missing)} papers not in the bucket: arxiv.org, one per {ARXIV_DELAY} s", flush=True)
    for pid in missing:
        try:
            r = from_arxiv(pid, tool)
        except Exception as e:  # noqa: BLE001
            print(f"  {pid}: {e}", flush=True)
            r = row(pid, status="missing")
        append_jsonl(INDEX, [r])
        print(f"  {pid}: {r['status']}", flush=True)
        time.sleep(ARXIV_DELAY)

    last = write_csv()
    mine = [last[i] for i in ids if i in last]
    status = {s: sum(r["status"] == s for r in mine) for s in ("ok", "short", "missing")}
    chars = sum(r["chars"] or 0 for r in mine)
    print(f"{len(ids):,} papers: " + ", ".join(f"{v:,} {k}" for k, v in status.items())
          + f", {len(ids) - len(mine):,} failed (run again); {chars / 1e9:.2f} G characters"
          + f" -> {OUT}/<id>.txt, {OUT / 'index.csv'}")


if __name__ == "__main__":
    main()
