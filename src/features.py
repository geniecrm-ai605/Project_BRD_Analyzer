"""Rule-based features from the condensed BRD text (our own code, no ML)."""
import math
import re
import numpy as np

LABELS = ["Objective", "Scope", "Integrations", "Data", "Non-functional needs", "Rollout"]
_LABEL_RE = re.compile(r"\b(Objective|Scope|Integrations|Data|Non-functional needs|Rollout):\s*")

FEATURE_NAMES = [
    "n_integrations", "integration_kw", "migration", "compliance", "log_volume",
    "log_sites", "scope_items", "complexity_kw", "simple_kw", "availability", "pilot", "words",
]

_MULT = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6}
_VOLUME = re.compile(
    r"(\d[\d,]*\.?\d*)\s*(k|m|million|thousand)?\+?\s+(?:[a-z\-/]+\s+){0,2}?"
    r"(orders|users|records|transactions|requests|customers|cases|claims|patients|tickets|invoices|devices|"
    r"shipments|messages|calls|accounts|documents|employees|sessions|leads|contacts|students|members)\b", re.I)
_SITES = re.compile(
    r"(\d[\d,]*)\s+(?:[a-z\-]+\s+)?(stores|locations|sites|offices|branches|clinics|plants|regions|countries|"
    r"warehouses|teams|departments|hospitals|schools|depots|factories|hubs|markets)\b", re.I)

_INTEG_KW = ["api", "erp", "crm", "gateway", "payment", "sso", "identity", "sms", "email", "warehouse", "sap",
             "oracle", "ldap", "bank", "courier", "webhook", "sftp", "edi", "iot", "scada", "gps", "ehr", "hl7",
             "fhir", "legacy", "third-party", "external", "provider", "system"]
_COMPLEX_KW = ["real-time", "near-real-time", "high availability", "failover", "multi-region", "multi-currency",
               "multilingual", "single sign-on", "sso", "mobile", "offline", "dashboard", "workflow", "approval",
               "portal", "machine learning", "forecast", "analytics", "payment", "bulk", "batch", "audit trail",
               "role-based", "multi-tenant", "reconcil", "scheduling", "optimi", "notification", "concurrent"]
_SIMPLE_KW = ["simple", "basic", "small", "short", "single", "one team", "one office", "existing configuration",
              "existing crm", "no migration", "none", "minimal", "optional", "pilot with", "narrow", "out-of-the-box"]
_COMPLIANCE = re.compile(r"hipaa|pci|gdpr|privacy|regulat|audit|retention|consent|health data|sox|encrypt|pii|"
                         r"personal data|data protection|compliance|clinical|financial", re.I)


def split_segments(text):
    parts = _LABEL_RE.split(text)
    seg = {k: "" for k in LABELS}
    for i in range(1, len(parts) - 1, 2):
        seg[parts[i]] = parts[i + 1].strip()
    return seg


def _count_integrations(s):
    s = s.lower().strip()
    if not s or s.startswith("none") or re.search(r"\bno (external )?integration", s):
        return 0
    items = [i for i in re.split(r",|;|\band\b|\bplus\b", s) if re.search(r"[a-z]{2,}", i)]
    return min(len(items), 12)


def _migration(s):
    s = s.lower()
    if not s.strip():
        return 0
    if re.search(r"\bno (historical )?(data )?(migration|import)|not (be )?migrat|without migrat|clean (start|log|slate)|"
                 r"start(s)? (at|from) (launch|scratch)|begin(s)? at launch|no import", s):
        return 0
    score = 0
    if re.search(r"migrat|import|convert|load|transfer|cutover", s):
        score += 1
    if re.search(r"histor|years|archive|legacy|multiple|all existing|reconcil|dedup|cleans", s):
        score += 1
    if re.search(r"million|\d{3},\d{3}|\d+\s?m\b|decade", s):
        score += 1
    return score


def _max_volume(text):
    best = 0.0
    for num, suf, _ in _VOLUME.findall(text):
        try:
            best = max(best, float(num.replace(",", "")) * _MULT.get((suf or "").lower(), 1))
        except ValueError:
            pass
    return best


def _max_sites(text):
    best = 0.0
    for num, _ in _SITES.findall(text):
        try:
            best = max(best, float(num.replace(",", "")))
        except ValueError:
            pass
    return best


def extract_features(text):
    t = text.lower()
    seg = split_segments(text)
    comp_txt = " ".join([seg["Non-functional needs"], seg["Data"], seg["Scope"], seg["Objective"]])
    no_comp = re.search(r"no (new )?(compliance|regulatory)", comp_txt, re.I)
    scope = seg["Scope"]
        # items are separated by ';' (condenser / generator) - fall back to ',' only for old comma-style rows
    sep = ";" if ";" in scope else ","
    scope_items = 0 if not scope else min(len([x for x in scope.split(sep) if x.strip()]), 12)
    feats = {
        "n_integrations": _count_integrations(seg["Integrations"]),
        "integration_kw": len({k for k in _INTEG_KW if k in seg["Integrations"].lower()}),
        "migration": _migration(seg["Data"]),
        "compliance": 0 if no_comp else len(set(m.lower() for m in _COMPLIANCE.findall(comp_txt))),
        "log_volume": math.log10(1 + _max_volume(text)),
        "log_sites": math.log10(1 + _max_sites(text)),
        "scope_items": scope_items,
        "complexity_kw": len({k for k in _COMPLEX_KW if k in t}),
        "simple_kw": len({k for k in _SIMPLE_KW if k in t}),
        "availability": 1 if re.search(r"99\.\d|high availability|24/7", t) else 0,
        "pilot": 1 if re.search(r"pilot|phase|wave", seg["Rollout"].lower()) else 0,
        "words": len(text.split()),
    }
    return np.array([feats[n] for n in FEATURE_NAMES], dtype=float)
