#!/usr/bin/env python
"""Buscador BM25 sobre el corpus de literatura del pipeline SLIP.

El corpus vive en ``docs/SLIP/`` con dos vistas del mismo conjunto de 126
papers:

- ``papers_text.json``: ``clean_text`` (abstract o texto completo según el
  paper) + ``clean_sections`` (texto completo concatenado por sección) +
  ``pdf_file``.
- ``papers_zoned.json``: párrafos con ``section`` (abstract, introduction,
  methodology, results, conclusion, figures_tables_captions, references) y
  ``zone`` (method, evidence, claim, background, gap, rest, future_work).

Este script indexa por defecto la vista ``zoned`` a nivel de párrafo (mayor
precisión y permite filtrar por sección/rol). ``--view text`` indexa la vista
``papers_text.json`` a nivel de paper para comparar.

Uso:
    python scripts/search_slip_docs.py "missing data imputation" --top 10
    python scripts/search_slip_docs.py "calibration confidence" --section results
    python scripts/search_slip_docs.py "motion artifact" --view text --top 5
    python scripts/search_slip_docs.py "degradation" --papers

No requiere dependencias externas (BM25 implementado con la stdlib).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
ZONED_PATH = REPO_ROOT / "docs" / "SLIP" / "papers_zoned.json"
TEXT_PATH = REPO_ROOT / "docs" / "SLIP" / "papers_text.json"
# Metadatos opcionales del proyecto SLIP (títulos, año, venue, DOI). Solo se
# usan para identificar los papers en la salida; la búsqueda no depende de él.
METADATA_PATH = (
    REPO_ROOT.parent
    / "SLIP"
    / "results"
    / "output"
    / "SensoresACL"
    / "papers_metadata.json"
)

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can",
    "could", "did", "do", "does", "doing", "don", "down", "during", "each",
    "few", "for", "from", "further", "had", "has", "have", "having", "he",
    "her", "here", "hers", "herself", "him", "himself", "his", "how", "i",
    "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more",
    "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on",
    "once", "only", "or", "other", "our", "ours", "ourselves", "out",
    "over", "own", "same", "she", "should", "so", "some", "such", "than",
    "that", "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "would", "you",
    "your", "yours", "yourself", "yourselves",
    # Ruido académico habitual que no aporta a la consulta.
    "et", "al", "fig", "figs", "figure", "figures", "table", "tables",
    "study", "studies", "paper", "article", "work", "works", "using",
    "used", "use", "also", "however", "therefore", "thus", "may", "might",
    "one", "two", "three", "first", "second", "new", "results", "result",
    "method", "methods", "proposed", "propose", "show", "shows", "shown",
    "based", "approach", "approaches", "well", "much", "many", "like",
    "including", "within", "without", "across", "among", "via", "per",
}

SECTION_WEIGHTS = {
    "abstract": 1.20,
    "methodology": 1.10,
    "results": 1.05,
    "conclusion": 0.95,
    "introduction": 0.90,
    "figures_tables_captions": 0.55,
    "references": 0.15,
}
ZONE_WEIGHTS = {
    "method": 1.15,
    "evidence": 1.10,
    "claim": 1.10,
    "gap": 1.20,
    "future_work": 1.20,
    "background": 0.90,
    "rest": 0.80,
}


def tokenize(text: str, keep_phrases: bool = False) -> list[str]:
    """Tokeniza a minúsculas eliminando stopwords."""
    tokens = TOKEN_RE.findall(text.lower())
    if keep_phrases:
        return tokens
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def load_metadata() -> dict[str, dict[str, Any]]:
    if not METADATA_PATH.is_file():
        return {}
    try:
        rows = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {row["paper_id"]: row for row in rows if "paper_id" in row}


def load_documents(view: str) -> list[dict[str, Any]]:
    """Devuelve documentos indexables: párrafos (zoned) o papers (text)."""
    docs: list[dict[str, Any]] = []
    if view == "zoned":
        rows = json.loads(ZONED_PATH.read_text(encoding="utf-8"))
        for paper in rows:
            for para in paper.get("paragraphs", []):
                text = (para.get("text") or "").strip()
                if not text:
                    continue
                docs.append(
                    {
                        "paper_id": paper["paper_id"],
                        "section": para.get("section") or "unknown",
                        "zone": para.get("zone") or "rest",
                        "paragraph_idx": para.get("paragraph_idx"),
                        "text": text,
                    }
                )
    elif view == "text":
        rows = json.loads(TEXT_PATH.read_text(encoding="utf-8"))
        for paper in rows:
            parts = [paper.get("clean_text") or ""]
            for section, body in (paper.get("clean_sections") or {}).items():
                parts.append(f"\n[{section}]\n{body}")
            docs.append(
                {
                    "paper_id": paper["paper_id"],
                    "section": "all",
                    "zone": "all",
                    "paragraph_idx": None,
                    "text": "\n".join(parts),
                }
            )
    else:
        raise SystemExit(f"Vista no soportada: {view}")
    return docs


class BM25:
    """BM25 Okapi con pesos por sección y zona."""

    def __init__(self, docs: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(d["text"]) for d in docs]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avg_len = sum(self.doc_len) / len(self.doc_len) if docs else 0.0
        self.tf: list[Counter] = [Counter(t) for t in self.doc_tokens]
        df: Counter = Counter()
        for tf in self.tf:
            df.update(set(tf))
        n = len(docs)
        self.idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def search(
        self,
        query: str,
        top: int = 10,
        sections: set[str] | None = None,
        zones: set[str] | None = None,
        paper_ids: set[str] | None = None,
    ) -> list[tuple[float, dict[str, Any]]]:
        terms = tokenize(query)
        phrases = [p.lower() for p in re.findall(r'"([^"]+)"', query)]
        scored: list[tuple[float, dict[str, Any]]] = []
        for i, doc in enumerate(self.docs):
            if sections and doc["section"] not in sections:
                continue
            if zones and doc["zone"] not in zones:
                continue
            if paper_ids and doc["paper_id"] not in paper_ids:
                continue
            score = 0.0
            tf = self.tf[i]
            length = self.doc_len[i] or 1
            for term in terms:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                idf = self.idf.get(term, 0.0)
                denom = freq + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                score += idf * freq * (self.k1 + 1) / denom
            if score <= 0:
                continue
            weight = SECTION_WEIGHTS.get(doc["section"], 1.0)
            weight *= ZONE_WEIGHTS.get(doc["zone"], 1.0)
            score *= weight
            for phrase in phrases:
                occurrences = doc["text"].lower().count(phrase)
                if occurrences:
                    score *= 1.0 + 0.5 * min(occurrences, 3)
            scored.append((score, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:top]


def aggregate_by_paper(
    hits: Iterable[tuple[float, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Agrupa aciertos por paper: mejor score + nº de párrafos relevantes."""
    grouped: dict[str, dict[str, Any]] = {}
    for score, doc in hits:
        entry = grouped.setdefault(
            doc["paper_id"],
            {"paper_id": doc["paper_id"], "best_score": 0.0, "hits": []},
        )
        entry["best_score"] = max(entry["best_score"], score)
        entry["hits"].append((score, doc))
    return sorted(grouped.values(), key=lambda e: e["best_score"], reverse=True)


def _snippet(text: str, query: str, width: int = 320) -> str:
    """Devuelve un fragmento centrado en el primer término de la consulta."""
    terms = [t for t in tokenize(query) if len(t) > 3]
    lowered = text.lower()
    pos = -1
    for term in terms:
        pos = lowered.find(term)
        if pos >= 0:
            break
    if pos < 0:
        return text[:width].replace("\n", " ")
    start = max(0, pos - width // 3)
    end = min(len(text), start + width)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return prefix + re.sub(r"\s+", " ", text[start:end]).strip() + suffix


def format_hits(
    hits: list[tuple[float, dict[str, Any]]],
    metadata: dict[str, dict[str, Any]],
    query: str,
    snippet_chars: int,
    group: bool,
) -> str:
    out: list[str] = []
    if group:
        for entry in aggregate_by_paper(hits):
            meta = metadata.get(entry["paper_id"], {})
            title = meta.get("title") or "(sin título en metadatos)"
            year = meta.get("year") or "?"
            venue = (meta.get("venue") or "")[:40]
            out.append(
                f"\n[{entry['paper_id']}] {title}\n"
                f"  year={year} venue={venue} score={entry['best_score']:.3f} "
                f"paragraphs={len(entry['hits'])}"
            )
            for score, doc in sorted(entry["hits"], key=lambda x: -x[0])[:3]:
                out.append(
                    f"  · {doc['section']}/{doc['zone']} p={doc['paragraph_idx']} "
                    f"({score:.3f}): {_snippet(doc['text'], query, snippet_chars)}"
                )
        return "\n".join(out)

    for score, doc in hits:
        meta = metadata.get(doc["paper_id"], {})
        title = (meta.get("title") or "")[:90]
        out.append(
            f"\n[{doc['paper_id']}] {title}\n"
            f"  {doc['section']}/{doc['zone']} p={doc['paragraph_idx']} "
            f"score={score:.3f}\n"
            f"  {_snippet(doc['text'], query, snippet_chars)}"
        )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Consulta en lenguaje natural (admite \"frases\").")
    parser.add_argument("--view", choices=["zoned", "text"], default="zoned")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--section", default="", help="Filtra por sección (coma-separada).")
    parser.add_argument("--zone", default="", help="Filtra por zona (coma-separada).")
    parser.add_argument("--paper", default="", help="Restringe a paper_id(s).")
    parser.add_argument("--snippet-chars", type=int, default=320)
    parser.add_argument("--papers", action="store_true", help="Agrupa por paper.")
    parser.add_argument("--json", action="store_true", help="Salida JSON.")
    args = parser.parse_args(argv)

    docs = load_documents(args.view)
    index = BM25(docs)
    sections = {s.strip() for s in args.section.split(",") if s.strip()} or None
    zones = {z.strip() for z in args.zone.split(",") if z.strip()} or None
    paper_ids = {p.strip() for p in args.paper.split(",") if p.strip()} or None

    hits = index.search(
        args.query, top=args.top, sections=sections, zones=zones, paper_ids=paper_ids
    )
    metadata = load_metadata()

    if args.json:
        payload = [
            {
                "score": round(score, 4),
                "paper_id": doc["paper_id"],
                "section": doc["section"],
                "zone": doc["zone"],
                "paragraph_idx": doc["paragraph_idx"],
                "text": doc["text"],
            }
            for score, doc in hits
        ]
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    print(f"Vista: {args.view} | documentos indexados: {len(docs)} | consulta: {args.query!r}")
    print(format_hits(hits, metadata, args.query, args.snippet_chars, args.papers))


if __name__ == "__main__":
    sys.exit(main())
