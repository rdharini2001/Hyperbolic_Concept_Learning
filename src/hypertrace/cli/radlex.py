"""Fill RadLex IDs in a concept ontology from an official RadLex source (never guessed).

    hypertrace-radlex --source RADLEX.csv                       # BioPortal CSV export of RadLex
    hypertrace-radlex --source RadLex.owl                       # OWL release from radlex.org
    hypertrace-radlex --bioportal-key $BIOPORTAL_API_KEY        # BioPortal REST API (free key)
    hypertrace-radlex ... --ontology <file> --write             # write the IDs into the ontology file

Every node's ``radlex.term`` list is matched against RadLex preferred labels and synonyms,
case-insensitively and EXACTLY (ignoring hyphens and spaces): no fuzzy matching.

* exactly one RID matched on a preferred label (or, failing that, exactly one on synonyms)
  -> ``id`` filled (with ``--write``);
* several RIDs -> left empty, every candidate listed in the report for you to choose;
* none -> left empty.

The report (``--report``, default radlex_report.csv next to the ontology) lists, per node, the
terms tried, the decision and all candidates (RID | label | matched on), so each filled ID can be
checked before it is cited. Nodes whose ``radlex`` is null (size bins: thresholds, not RadLex
concepts) are skipped. RadLex is licensed by RSNA: download it yourself after accepting the terms
(radlex.org, or BioPortal: Ontologies > RADLEX > Download).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

from hypertrace.config import default_config

RID = re.compile(r"(RID\d+)")


def norm(text: str) -> str:
    """Case-, space- and hyphen-insensitive key: "Juxta-pleural" == "juxtapleural" == "juxta pleural"."""
    return re.sub(r"[\s\-_]+", "", text.strip().lower())


@dataclass(frozen=True)
class Hit:
    rid: str
    label: str
    on: str  # "preferred" | "synonym"


class RadLexIndex:
    """Exact-match index over preferred labels and synonyms."""

    def __init__(self) -> None:
        self.pref: dict[str, set[tuple[str, str]]] = defaultdict(set)
        self.syn: dict[str, set[tuple[str, str]]] = defaultdict(set)

    def add(self, rid: str, preferred: str, synonyms: list[str]) -> None:
        if preferred:
            self.pref[norm(preferred)].add((rid, preferred))
        for s in synonyms:
            if s:
                self.syn[norm(s)].add((rid, preferred or s))

    def lookup(self, term: str) -> list[Hit]:
        k = norm(term)
        return ([Hit(r, lbl, "preferred") for r, lbl in sorted(self.pref.get(k, ()))]
                + [Hit(r, lbl, "synonym") for r, lbl in sorted(self.syn.get(k, ()))])

    def __len__(self) -> int:
        return len({r for v in self.pref.values() for r, _ in v})

    @classmethod
    def from_csv(cls, path: Path) -> RadLexIndex:
        """BioPortal CSV export: columns 'Class ID', 'Preferred Label', 'Synonyms' ('|'-separated)."""
        idx = cls()
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            cols = {c.lower().strip(): c for c in reader.fieldnames or []}
            cid = next((cols[c] for c in cols if c in ("class id", "id", "rid")), None)
            pref = next((cols[c] for c in cols if c in ("preferred label", "preferred name", "label", "name")), None)
            syn = next((cols[c] for c in cols if c.startswith("synonym")), None)
            if not cid or not pref:
                raise SystemExit(f"{path}: need a 'Class ID' and a 'Preferred Label' column, found {reader.fieldnames}")
            for row in reader:
                m = RID.search(row[cid] or "")
                if m:
                    idx.add(m.group(1), row[pref] or "", (row.get(syn) or "").split("|") if syn else [])
        return idx

    @classmethod
    def from_owl(cls, path: Path) -> RadLexIndex:
        """RadLex OWL (RDF/XML): classes whose IRI ends in RIDnnn; labels from Preferred_name / rdfs:label,
        synonyms from any *Synonym* element."""
        idx = cls()
        for _, el in ET.iterparse(path, events=("end",)):
            if not el.tag.endswith("}Class"):
                continue
            about = next((v for k, v in el.attrib.items() if k.endswith("about")), "")
            m = RID.search(about)
            if m:
                pref, syns = "", []
                for ch in el:
                    tag = ch.tag.rsplit("}", 1)[-1].lower()
                    text = (ch.text or "").strip()
                    if tag in ("preferred_name", "preferred_name_english", "preflabel") and text:
                        pref = text
                    elif tag == "label" and text and not pref:
                        pref = text
                    elif "synonym" in tag and text:
                        syns.append(text)
                idx.add(m.group(1), pref, syns)
            el.clear()
        return idx


def bioportal_lookup(term: str, key: str) -> list[Hit]:
    q = urllib.parse.urlencode({"q": term, "ontologies": "RADLEX", "require_exact_match": "true",
                                "include": "prefLabel,synonym", "apikey": key})
    with urllib.request.urlopen(f"https://data.bioontology.org/search?{q}", timeout=30) as r:
        data = json.load(r)
    hits = []
    for c in data.get("collection", []):
        m = RID.search(c.get("@id", ""))
        if not m:
            continue
        pref = c.get("prefLabel", "")
        on = "preferred" if norm(pref) == norm(term) else "synonym"
        if on == "synonym" and not any(norm(s) == norm(term) for s in c.get("synonym", []) or []):
            continue  # BioPortal "exact" also matches other fields; keep label / synonym matches only
        hits.append(Hit(m.group(1), pref, on))
    return hits


def decide(hits: list[Hit]) -> tuple[str | None, str]:
    """One RID on preferred labels, else one on synonyms; otherwise no decision."""
    for on in ("preferred", "synonym"):
        rids = sorted({h.rid for h in hits if h.on == on})
        if len(rids) == 1:
            return rids[0], f"unique {on} match"
        if len(rids) > 1:
            return None, f"ambiguous: {len(rids)} RIDs on {on} labels"
    return None, "no exact match"


def write_ids(path: Path, ids: dict[str, str]) -> int:
    """Set ``id:`` inside each node's ``radlex: {id: ..., term: [...]}`` line, keeping comments and layout."""
    lines, node, n = path.read_text().splitlines(keepends=True), None, 0
    for i, line in enumerate(lines):
        m = re.match(r"\s*-\s*name:\s*(\S+)", line)
        if m:
            node = m.group(1)
            continue
        if node in ids and re.match(r"\s*radlex:\s*\{", line):
            new = re.sub(r"id:\s*[^,}]+", f"id: {ids[node]}", line, count=1)
            if new != line:
                lines[i], n = new, n + 1
    path.write_text("".join(lines))
    return n


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ontology", type=Path, default=default_config("ontology/lung_nodule"))
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--source", type=Path, help="RadLex release: BioPortal CSV export (.csv) or OWL (.owl/.xml)")
    src.add_argument("--bioportal-key", help="BioPortal API key (https://bioportal.bioontology.org/account)")
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--write", action="store_true", help="write unique matches into the ontology file")
    p.add_argument("--overwrite", action="store_true", help="also replace IDs that are already filled")
    a = p.parse_args(argv)

    onto = yaml.safe_load(a.ontology.read_text())
    if a.source:
        idx = RadLexIndex.from_csv(a.source) if a.source.suffix.lower() == ".csv" else RadLexIndex.from_owl(a.source)
        print(f"RadLex index: {len(idx)} concepts from {a.source}")
        if not len(idx):
            raise SystemExit("no RadLex concepts read: is this the RadLex CSV or OWL file?")

        def look(t: str) -> list[Hit]:
            return idx.lookup(t)
    else:
        def look(t: str) -> list[Hit]:
            return bioportal_lookup(t, a.bioportal_key)

    rows, fill = [], {}
    for n in onto["nodes"]:
        rl = n.get("radlex")
        if not isinstance(rl, dict):
            continue
        terms = rl.get("term") or []
        terms = [terms] if isinstance(terms, str) else terms
        hits: list[Hit] = []
        for t in terms:
            hits += look(t)
        hits = list(dict.fromkeys(hits))
        rid, why = decide(hits)
        current = rl.get("id")
        if rid and (current in (None, "") or a.overwrite):
            fill[n["name"]] = rid
        if current and rid and current != rid:
            why += f" (ontology has {current})"
        rows.append({"node": n["name"], "terms": " | ".join(terms), "decision": rid or "", "why": why,
                     "current": current or "",
                     "candidates": " ; ".join(f"{h.rid} | {h.label} | {h.on}" for h in hits)})
    report = a.report or a.ontology.with_name("radlex_report.csv")
    with report.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["node"])
        w.writeheader()
        w.writerows(rows)
    n_unique = sum(1 for r in rows if r["decision"])
    n_amb = sum(1 for r in rows if r["why"].startswith("ambiguous"))
    print(f"{len(rows)} nodes looked up: {n_unique} unique, {n_amb} ambiguous, "
          f"{len(rows) - n_unique - n_amb} without an exact match. Report: {report}")
    if a.write:
        print(f"wrote {write_ids(a.ontology, fill)} IDs into {a.ontology}. Review them against the report.")
    else:
        print("dry run (add --write to fill the unique matches into the ontology file)")


if __name__ == "__main__":
    main()
