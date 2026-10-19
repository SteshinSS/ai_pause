#!/usr/bin/env python3
"""Download the metadata of every ML paper first posted to arXiv in 2020-2025 -> data/papers.jsonl.gz.

arXiv's OAI-PMH interface lists papers by the month they last changed, so every month from 2020-01 to today is
downloaded (sets cs and stat) and a paper is kept if its id says 2020-2025 (yymm.nnnnn) and it has one of the ML
categories of the 2025 census. One file per set and month in data/harvest/; a finished month gets a .done file, so
an interrupted download resumes. At the end the months are merged into data/papers.jsonl.gz, one line per paper (a
paper that changed during the download appears twice; its later record is kept): id, year, month, primary category,
categories, title, abstract, authors [[surname, forenames, affiliation]], comments, journal ref, DOI, v1 date.

--year Y downloads one later year into data/harvest_Y/ and data/papers_Y.jsonl.gz instead (months Y-01 to today;
--through M keeps ids of months 1 to M only).

    python harvest.py                          # ~2 h (1 h 42 min on 2 Oct 2026), one request at a time
    python harvest.py --year 2026 --through 9  # Jan-Sep 2026, ~25 min
"""
import argparse
import datetime as dt
from collections import Counter
import gzip
import json
import subprocess
import time
import xml.etree.ElementTree as ET

from common import DATA, HARVEST, ID_RE, ML_CATS, PAPERS, YEARS, year_month

BASE = "https://oaipmh.arxiv.org/oai"
NS = {"o": "http://www.openarchives.org/OAI/2.0/", "a": "http://arxiv.org/OAI/arXiv/"}
SETS = ["cs", "stat"]
UA = "safety-papers-census/1.0 (bibliometrics; one request at a time)"
PAUSE = 2  # seconds between pages


def months(start: dt.date, until: dt.date) -> list:
    out = []
    while start <= until:
        nxt = (start.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
        out.append((start.isoformat(), min(nxt - dt.timedelta(days=1), until).isoformat()))
        start = nxt
    return out


def fetch(url: str) -> bytes:
    """curl (urllib got HTTP 406 on resumption pages); waits as long as arXiv's Retry-After asks."""
    for attempt in range(12):
        r = subprocess.run(["curl", "-s", "-m", "300", "-A", UA, "-w", "\n%{http_code} %header{retry-after}", url],
                           capture_output=True)
        body, _, tail = r.stdout.rpartition(b"\n")
        code, _, retry = tail.decode(errors="replace").partition(" ")
        if r.returncode == 0 and code == "200":
            return body
        wait = int(retry) if retry.strip().isdigit() else min(600, 30 * 2 ** attempt)
        print(f"  curl rc={r.returncode} HTTP {code}, waiting {wait} s", flush=True)
        time.sleep(wait)
    raise RuntimeError(f"giving up on {url}")


def text(el, path: str) -> str:
    x = el.find(path, NS)
    return " ".join(x.text.split()) if x is not None and x.text else ""


def parse(xml: bytes, keep) -> tuple:
    """(ML papers on one page whose (year, month) passes `keep`, resumption token or None, OAI error code or None)."""
    root = ET.fromstring(xml)
    err = root.find("o:error", NS)
    if err is not None:
        return [], None, err.get("code")
    papers = []
    for rec in root.iterfind(".//o:record", NS):
        head, a = rec.find("o:header", NS), rec.find(".//a:arXiv", NS)
        if head is None or head.get("status") == "deleted" or a is None:
            continue
        pid, cats = text(a, "a:id"), text(a, "a:categories")
        if not ID_RE.match(pid) or not keep(*year_month(pid)) or not ML_CATS & set(cats.split()):
            continue
        authors = [[text(au, "a:keyname"), text(au, "a:forenames"),
                    "; ".join(" ".join(x.text.split()) for x in au.iterfind("a:affiliation", NS) if x.text)]
                   for au in a.iterfind("a:authors/a:author", NS)]
        y, m = year_month(pid)
        papers.append({"id": pid, "year": y, "month": m, "primary": cats.split()[0], "cats": cats,
                       "title": text(a, "a:title"), "abstract": text(a, "a:abstract"), "authors": authors,
                       "comments": text(a, "a:comments"), "journal_ref": text(a, "a:journal-ref"),
                       "doi": text(a, "a:doi"), "created": text(a, "a:created"),
                       "datestamp": text(head, "o:datestamp")})
    tok = root.find(".//o:resumptionToken", NS)
    return papers, (tok.text.strip() if tok is not None and tok.text and tok.text.strip() else None), None


def harvest(set_: str, start: str, end: str, folder, keep) -> None:
    done = folder / f"{set_}_{start[:7]}.done"
    if done.exists() and json.loads(done.read_text())["until"] >= end:
        return
    n = pages = 0
    url = f"{BASE}?verb=ListRecords&set={set_}&metadataPrefix=arXiv&from={start}&until={end}"
    with gzip.open(folder / f"{set_}_{start[:7]}.jsonl.gz", "wt") as f:
        while url:
            papers, token, err = parse(fetch(url), keep)
            if err and err != "noRecordsMatch":
                raise RuntimeError(f"{set_} {start}: OAI error {err}")
            for d in papers:
                f.write(json.dumps(d) + "\n")
            n, pages = n + len(papers), pages + 1
            url = f"{BASE}?verb=ListRecords&resumptionToken={token}" if token else None
            print(f"  {set_} {start[:7]} page {pages}: {n} papers kept", flush=True)
            time.sleep(PAUSE)
    done.write_text(json.dumps({"until": end, "papers": n, "finished": dt.datetime.now().isoformat()}) + "\n")


def merge(folder, out, years) -> None:
    best = {}  # id -> (datestamp, year, line); raw lines keep memory near the file size
    for fn in sorted(folder.glob("*.jsonl.gz")):
        with gzip.open(fn, "rt") as f:
            for line in f:
                d = json.loads(line)
                if d["id"] not in best or d["datestamp"] >= best[d["id"]][0]:
                    best[d["id"]] = (d["datestamp"], d["year"], line)
    with gzip.open(out, "wt") as f:
        for pid in sorted(best):
            f.write(best[pid][2])
    per_year = Counter(v[1] for v in best.values())
    print(f"{len(best):,} papers -> {out}")
    for y in years:
        print(f"  {y}: {per_year[y]:,}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, help="one later year instead of 2020-2025")
    ap.add_argument("--through", type=int, default=12, help="with --year: keep ids of months 1 to this one")
    args = ap.parse_args()
    if args.year:
        folder, out, years = DATA / f"harvest_{args.year}", DATA / f"papers_{args.year}.jsonl.gz", [args.year]
        first = dt.date(args.year, 1, 1)
        keep = lambda y, m: y == args.year and m <= args.through  # noqa: E731
    else:
        folder, out, years, first = HARVEST, PAPERS, YEARS, dt.date(YEARS[0], 1, 1)
        keep = lambda y, m: y in YEARS  # noqa: E731
    folder.mkdir(parents=True, exist_ok=True)
    for s in SETS:
        for start, end in months(first, dt.date.today()):
            harvest(s, start, end, folder, keep)
    merge(folder, out, years)


if __name__ == "__main__":
    main()
