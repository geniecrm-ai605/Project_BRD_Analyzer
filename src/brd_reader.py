"""BRD (.docx / plain text) -> short 'condensed' text in the same format the model was trained on.

Training rows look like:
  Objective: ... Scope: a, b, c. Integrations: x, y. Data: ... Non-functional needs: ... Rollout: ...
Real BRDs are 3000+ words with 15+ sections, so we first squeeze them into that same shape.
"""
import re
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

_SENT = re.compile(r"(?<=[.!?])\s+")
_NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+)*)[\.\)]?\s+\S")


def _words(s, n):
    w = s.split()
    return " ".join(w[:n]).rstrip(" ,;:.")


def _sentences(text):
    out = []
    for line in text.split("\n"):
        out += [x.strip() for x in _SENT.split(line) if x.strip()]
    return out


# ---------- docx -> sections ----------
def read_docx_sections(file):
    """file = path or file-like. Returns list of {level, title, lines}."""
    doc = Document(file)
    sections, cur = [], {"level": 0, "title": "(preamble)", "lines": []}
    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(child, doc)
            txt = p.text.strip()
            if not txt:
                continue
            style = (p.style.name if p.style is not None else "") or ""
            m = re.match(r"Heading (\d)", style)
            level = int(m.group(1)) if m else 0
            if not m:
                h = _NUMBERED_HEADING.match(txt)
                if h and len(txt) <= 90 and not txt.endswith("."):
                    level = h.group(1).count(".") + 1
            if level:
                sections.append(cur)
                cur = {"level": level, "title": txt, "lines": []}
            else:
                cur["lines"].append(txt)
        elif tag == "tbl":
            for row in Table(child, doc).rows:
                cells, seen = [], set()
                for c in row.cells:
                    t = c.text.strip().replace("\n", " ")
                    if t and id(c._tc) not in seen:
                        seen.add(id(c._tc))
                        cells.append(t)
                if cells:
                    cur["lines"].append(" | ".join(cells))
    sections.append(cur)
    return [s for s in sections if s["lines"] or s["title"] != "(preamble)"]


def _section_text(sections, *keywords):
    """Lines of first section whose title has a keyword. If that heading has no body of its own,
    use only a sub-section whose own title matches (never swallow unrelated sub-sections/tables)."""
    for i, sec in enumerate(sections):
        if any(k in sec["title"].lower() for k in keywords):
            if sec["lines"]:
                return list(sec["lines"])
            for t in sections[i + 1:]:
                if t["level"] <= sec["level"]:
                    break
                if any(k in t["title"].lower() for k in keywords):
                    return list(t["lines"])
            return []
    return []


# ---------- condenser ----------
def _clean_row(line):
    parts = [p.strip() for p in line.split("|")]
    if parts and re.match(r"^[A-Z]{2,5}-?\d+$", parts[0]):
        parts = parts[1:]
    return ". ".join(parts)


def condense_sections(sections):
    full = "\n".join(l for sec in sections for l in [sec["title"]] + sec["lines"])
    _skip = ("out of scope", "open question", "assumption", "risk", "approval", "glossary", "reference")
    in_text = "\n".join(l for sec in sections if not any(k in sec["title"].lower() for k in _skip)
                         for l in sec["lines"] if "|" not in l)
    sents = _sentences(in_text)
    _neg = re.compile(r"not (be )?(required|in scope|included|planned|part of)|excluded|out of scope|"
                      r"unless separately|no (data )?(migration|import)", re.I)

    # Objective
    exec_lines = _section_text(sections, "executive summary", "overview", "purpose")
    exec_sents = _sentences(" ".join(exec_lines))
    obj = _words(exec_sents[0], 30) if exec_sents else _words(full, 30)

    # Scope (bullets of 'In Scope')
    scope_lines = _section_text(sections, "in scope")
    scope = "; ".join(_words(l, 8) for l in scope_lines[:10]) if scope_lines else ""

    # Integrations
    integ_lines = _section_text(sections, "integration")
    integ_txt = " ".join(integ_lines)
    if integ_lines and re.search(
            r"\b(no|none|not)\b[^.]{0,40}integrat|integrat[^.]{0,40}(not in scope|none)", integ_txt, re.I):
        integ = "none"
    elif len(integ_lines) > 1:
        integ = ", ".join(_words(l, 6).replace(",", " ") for l in integ_lines[:10])
    elif integ_lines:
        integ = ", ".join(_words(x, 6).replace(",", " ") for x in _sentences(integ_txt)[:6])
    else:  # no dedicated section: look for sentences that describe an integration
        ig = [x for x in sents if re.search(r"integrat|interface with|connect(s|ed)? to|\bsync|webhook|\bapi\b", x, re.I)
              and not re.search(r"\bno\b[^.]{0,30}integrat|not (be )?(required|in scope|included)", x, re.I)]
        items, seen = [], set()
        for x in ig:
            it = _words(x, 6).replace(",", " ")
            key = " ".join(it.lower().split()[:3])
            if key not in seen:
                seen.add(key); items.append(it)
        integ = ", ".join(items[:4]) if items else "none"

    # Data / migration
    mig = [x for x in sents if not _neg.search(x) and re.search(
        r"data migration|migrat\w* (all |existing |open |historical |initial )?(data|records|accounts|customers|orders|cases)|"
        r"historical (data|records)|legacy data|data import|bulk load|csv import|import(ed)? (of )?(initial|opening|historical|existing)|"
        r"supplied as csv|opening[- ](stock|balance)", x, re.I)]
    data = " ".join(_words(x, 25) for x in mig[:2]) if mig else "no migration"

    # Non-functional
    nfr_rows = [_clean_row(l) for l in _section_text(sections, "non-functional")]
    pri = [r for r in nfr_rows if re.search(r"\d|availab|perform|volume|scal|secur|concurren", r, re.I)] or nfr_rows
    nfr = " ".join(_words(r, 15) for r in pri[:3])

    # Rollout
    roll = [x for x in sents if re.search(r"\bpilot|roll-?out|go-?live|phased|wave\b", x, re.I)]
    rollout = " ".join(_words(x, 20) for x in roll[:2])

    return (f"Objective: {obj}. Scope: {scope}. Integrations: {integ}. Data: {data}. "
            f"Non-functional needs: {nfr}. Rollout: {rollout}.")


def condense_docx(file):
    return condense_sections(read_docx_sections(file))


def condense_plain_text(text):
    """Fallback for pasted text without headings (weaker than a structured .docx BRD)."""
    sents = _sentences(text)

    def pick(pattern, n, words):
        return [x for x in sents if re.search(pattern, x, re.I)][:n], words

    obj = _words(sents[0], 30) if sents else ""
    scope = "; ".join(_words(x, 8) for x in sents[:10])
    ig, _ = pick(r"integrat|interface with|connect(s|ed)? to|\bsync|\bapi\b", 4, 14)
    integ = "none" if (not ig or any(re.search(r"\bno\b[^.]{0,30}integrat", x, re.I) for x in ig)) \
        else ", ".join(_words(x, 14) for x in ig)
    mg, _ = pick(r"migrat|historical|legacy|data import|bulk load", 2, 25)
    data = " ".join(_words(x, 25) for x in mg) if mg else "no migration"
    nf, _ = pick(r"\d|availab|perform|secur|scal|concurren", 3, 15)
    rl, _ = pick(r"\bpilot|roll-?out|go-?live|phased|wave\b", 2, 20)
    return (f"Objective: {obj}. Scope: {scope}. Integrations: {integ}. Data: {data}. "
            f"Non-functional needs: {' '.join(_words(x, 15) for x in nf)}. "
            f"Rollout: {' '.join(_words(x, 20) for x in rl)}.")


def ensure_condensed(text):
    if re.search(r"Objective:", text) and re.search(r"Scope:", text):
        return text.strip()
    return condense_plain_text(text)
