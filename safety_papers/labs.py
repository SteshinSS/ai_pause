#!/usr/bin/env python3
"""Papers with an author at one of the top three US AI labs (OpenAI, Anthropic, Google incl. DeepMind), read from
the full texts (data/fulltext/, fulltext.py) -> data/labs.csv.

arXiv metadata has no affiliations, so they come from the text. A paper is a lab paper when a lab is named in:
- the header: the text before the abstract (authors and affiliations), without title lines, links and the arXiv stamp;
- an e-mail address at openai.com, anthropic.com, google.com or deepmind.com before the references;
- a footnote line before the references that gives an affiliation ("† OpenAI", "1University of X 2Google DeepMind");
- a footnote, the header or the arXiv comments saying the work was done at a lab or during an internship there;
- the authors' organizations that the compute census read from the full text (estimate_compute.py --all --run
  census, field `organizations`), once that run exists: it catches affiliations given by symbols or long lists.
A header match alone does not count when the census lists the authors' organizations and none is a lab: such matches
were a thesis examiner, an acknowledgement or an introduction sentence.
Papers that mention the Anthropic Fellows Program anywhere are flagged apart (anthropic_fellows): the fellows are
funded by Anthropic and use its compute, but are not on its staff. A paper without a usable text is unknown (lab
empty).

    python labs.py                 # every listed paper -> data/labs.csv, counts by year
    python labs.py check [N]       # N random flagged papers per kind of evidence, and N unflagged papers whose first
                                   # pages name a lab, with the lines that matched: read them to judge the rules
"""
import csv
import random
import re
import sys
from collections import Counter, defaultdict

from common import DATA, RUNS

FULLTEXT = DATA / "fulltext"
CENSUS = DATA / "compute_estimates" / "census" / "compute.csv"
OUT = DATA / "labs.csv"
LAB_OF = {"openai": "openai", "anthropic": "anthropic", "google": "google", "deepmind": "google"}
LABS = ("openai", "anthropic", "google")
NAME = re.compile(r"\b(OpenAI|Anthropic|Google|DeepMind)\b")
EMAIL = re.compile(r"@(?:[\w-]+\.)*(openai|anthropic|google|deepmind)\.com\b", re.I)
MARK = r"[0-9∗*†‡§¶♠♣♦♥⋆♮♯]"
FOOTLINE = re.compile(rf"^\s*{MARK}")                                          # a line that starts a footnote
FOOT = re.compile(rf"^\s*{MARK}{{1,3}}\s*,?\s*(OpenAI|Anthropic|Google|DeepMind)\b")  # "† OpenAI", "2Google ..."
INLINE = re.compile(rf"(?:^|[\s,;]){MARK}{{1,2}}\s?(OpenAI|Anthropic|Google|DeepMind)\b")  # "... 2Google DeepMind"
AFFIL = re.compile(r"Universit|Institute|College|School|Department|Correspond|Lab\b|Labs\b|Research|Inc\b|LLC|Ltd|"
                   r"Center|Centre|Equal contribution|equal contribution")
WORK = re.compile(r"\b(?:work|research|project|paper)\s+(?:was\s+)?(?:done|conducted|performed|carried out|completed|"
                  r"started)\b[^.\n]{0,80}?\b(OpenAI|Anthropic|Google|DeepMind)\b", re.I)
INTERN = re.compile(r"\bintern(?:ship|ing)?\b[^.\n]{0,60}?\b(OpenAI|Anthropic|Google|DeepMind)\b", re.I)
FELLOWS = re.compile(r"Anthropic\s+Fellow", re.I)
ABSTRACT = re.compile(r"(?:^|\n)\s*(?:A\s?BSTRACT|Abstract)\b|\bAbstract\s*[—–:.-]")
REFS = re.compile(r"\n\s*(?:\d+\s*)?(?:References|REFERENCES|R\s?EFERENCES|Bibliography)\s*\n")
STAMP = re.compile(r"arXiv:\d{4}\.\d{4,5}v\d+\s*\[[^\]]*\]\s*\d{1,2}\s+\w{3}\s+\d{4}")
URL = re.compile(r"\S*(?:https?://|www\.|github\.com|huggingface\.co)\S*")
WORDS = re.compile(r"[a-z]+")
PROSE_AFTER = re.compile(r"^(?:\s*[’']s\b|,?\s*(?:et al|19\d\d|20\d\d)|\s+(?:API|model|models|shares|released|has|have|is|"
                         r"was|uses|used|provides)\b)")          # "OpenAI's", "(OpenAI, 2024)", "Google shares ..."
PROSE_BEFORE = re.compile(r"(?:such as|like|by|from|using|via|of|with|and|or|\()\s*$", re.I)
PROSE_LINE = re.compile(r"\b(?:the|we|our|have|has|are|is|that|which|these|their|implemented|propose|show)\b")
MOVED = re.compile(r"\b(?:now|currently)\b", re.I)    # "work done at CMU, now at Google": not a lab paper
EARLY = 12000          # characters read for e-mails and footnotes (the first pages)
NO_ABSTRACT = 800      # header length when no abstract heading is found


def affiliation(s: str, m) -> bool:
    """A lab name used as an affiliation, not in prose ("OpenAI's GPT-4o", "models from Google", "(Anthropic, 2025)")."""
    return not PROSE_AFTER.match(s[m.end():m.end() + 12]) and not PROSE_BEFORE.search(s[max(0, m.start() - 12):m.start()])


def split(text: str):
    """(header, early text, abstract heading found) of a full text from fulltext.py."""
    body = "\n".join(text.split("\n")[2:])                    # drop fulltext.py's first line
    early = body[:EARLY]
    m = REFS.search(early)
    early = early[:m.start()] if m else early
    a = ABSTRACT.search(early)
    found = bool(a and a.start() >= 40)
    return (early[:a.start()] if found else early[:NO_ABSTRACT]), early, found


def clean_header(head: str, title: str) -> str:
    """The header without title lines (70% of their words in the title), links and the arXiv stamp."""
    tw = set(WORDS.findall(title.lower()))
    keep = []
    for line in head.split("\n"):
        s = URL.sub(" ", STAMP.sub(" ", line))
        w = WORDS.findall(s.lower())
        if w and sum(x in tw for x in w) / len(w) >= 0.7:
            continue
        keep.append(s)
    return "\n".join(keep)


def snippet(s: str, m) -> str:
    return " ".join(s[max(0, m.start() - 50):m.end() + 50].split())[:140]


def evidence(pid: str, title: str, comments: str):
    """[(lab, kind, snippet)], anthropic_fellows, abstract heading found; None without a usable text."""
    p = FULLTEXT / f"{pid}.txt"
    if not p.exists():
        return None
    text = p.read_text(errors="replace")
    if len(text) < 1500:
        return None
    head, early, found = split(text)
    head = clean_header(head, title)
    # without an abstract heading the header may run into the abstract: then only affiliation-like or short lines
    ev = [(LAB_OF[m[1].lower()], "header", " ".join(line.split())[:140]) for line in head.split("\n")
          for m in NAME.finditer(line) if affiliation(line, m) and not PROSE_LINE.search(line)
          and (found or AFFIL.search(line) or len(line.split()) <= 6)]
    ev += [(LAB_OF[m[1].lower()], "email", snippet(early, m)) for m in EMAIL.finditer(early)]
    lines = early.split("\n")
    for k, line in enumerate(lines):
        if len(line) > 220:
            continue
        m = FOOT.match(line) if FOOTLINE.match(line) else None
        if m and not (len(line.split()) <= 6 or AFFIL.search(line) or re.search(r"\b\w+\s*,", line[m.end():m.end() + 30])) \
                or m and (re.search(r"\[\d", line) or not affiliation(line, m)):
            m = None
        m = m or (INLINE.search(line) if AFFIL.search(line) else None)
        if m:
            ev.append((LAB_OF[m[1].lower()], "footnote", " ".join(line.split())[:140]))
        if not FOOTLINE.match(line):
            continue
        two = line + " " + (lines[k + 1] if k + 1 < len(lines) else "")
        for rx, kind in ((WORK, "work done at"), (INTERN, "internship")):
            ev += [(LAB_OF[m[1].lower()], kind, snippet(two, m)) for m in rx.finditer(two) if not MOVED.search(m[0])]
    for src, where in ((head, "header"), (comments or "", "comments")):
        for rx, kind in ((WORK, "work done at"), (INTERN, "internship")):
            ev += [(LAB_OF[m[1].lower()], f"{kind} ({where})", snippet(src, m)) for m in rx.finditer(src)
                   if not MOVED.search(m[0])]
    return ev, bool(FELLOWS.search(text)), found


def census_organizations() -> dict:
    """{id: [organization, ...]} from the compute census's answers; {} before that run."""
    if not CENSUS.exists():
        return {}
    return {r["id"]: [o.strip() for o in (r.get("organizations") or "").split(";") if o.strip()]
            for r in csv.DictReader(open(CENSUS))}


def run() -> None:
    papers = list(csv.DictReader(open(RUNS / "census" / "safety_papers.csv")))
    orgs = census_organizations()
    cols = ["id", "year", "topic", "lab", *LABS, "anthropic_fellows", "kinds", "evidence", "abstract_found"]
    by_year = defaultdict(Counter)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        for r in papers:
            res = evidence(r["id"], r["title"], r["comments"])
            row = {"id": r["id"], "year": r["year"], "topic": r["topic"]}
            c = by_year[r["year"]]
            c["papers"] += 1
            if res is None:
                row.update({"lab": "", "kinds": "no text"})
                c["unknown"] += 1
            else:
                ev, fellows, found = res
                text_labs = {lab for lab, _, _ in ev}
                named = orgs.get(r["id"], [])
                ev += [(LAB_OF[m[1].lower()], "organizations", o) for o in named if not FELLOWS.search(o)
                       for m in NAME.finditer(o)]
                if named and {k for _, k, _ in ev} == {"header"}:       # organizations listed, none a lab
                    ev = []
                labs = {lab for lab, _, _ in ev}
                c["organizations only"] += bool(labs) and not text_labs
                c["header dropped"] += bool(text_labs) and not labs
                row.update({"lab": int(bool(labs)), **{lab: int(lab in labs) for lab in LABS},
                            "anthropic_fellows": int(fellows), "kinds": ";".join(sorted({k for _, k, _ in ev})),
                            "evidence": ev[0][2] if ev else "", "abstract_found": int(found)})
                c["lab"] += bool(labs)
                c["fellows"] += fellows
                c["fellows, no lab"] += fellows and not labs
                for lab in labs:
                    c[lab] += 1
            w.writerow(row)
    print(f"{OUT}" + ("" if orgs else "  (no compute census yet: organizations not used)"))
    print("year  papers  lab papers   openai  anthropic  google  Anthropic Fellows (of them without a lab)  no text"
          "  by organizations only  header match dropped")
    for y, c in sorted(by_year.items()):
        print(f"{y}  {c['papers']:6d}  {c['lab']:5d} ({c['lab'] / c['papers']:4.1%})  {c['openai']:6d}  {c['anthropic']:9d}  "
              f"{c['google']:6d}  {c['fellows']:5d} ({c['fellows, no lab']})  {c['unknown']:7d}"
              f"  {c['organizations only']:21d}  {c['header dropped']:20d}")


def check(n: int = 15) -> None:
    rows = list(csv.DictReader(open(OUT)))
    title = {r["id"]: r["title"] for r in csv.DictReader(open(RUNS / "census" / "safety_papers.csv"))}
    random.seed(1)
    by_kind = defaultdict(list)
    for r in rows:
        for k in filter(None, r["kinds"].split(";")):
            by_kind[k].append(r)
    for k, rs in sorted(by_kind.items()):
        print(f"\n== flagged by {k}: {len(rs)} papers; {min(n, len(rs))} random ==")
        for r in random.sample(rs, min(n, len(rs))):
            print(f"{r['id']}  {r['evidence']}")
    miss = []
    for r in rows:
        if r["lab"] == "0":
            _, early, _ = split((FULLTEXT / f"{r['id']}.txt").read_text(errors="replace"))
            hits = [" ".join(l.split())[:140] for l in early.split("\n") if NAME.search(l)]
            if hits:
                miss.append((r["id"], hits))
    print(f"\n== not flagged, a lab named before the references: {len(miss)} papers; {min(n, len(miss))} random ==")
    for pid, hits in random.sample(miss, min(n, len(miss))):
        print(f"{pid}  {title[pid][:70]}")
        for h in hits[:3]:
            print(f"      {h}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check(int(sys.argv[2]) if len(sys.argv) > 2 else 15)
    else:
        run()
