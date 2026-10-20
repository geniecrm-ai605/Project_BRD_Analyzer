"""Risks, mitigations and duration reasoning for each estimate.

Data-driven (learned from the dataset's risk / reasoning fields) + rule gates. No LLM, CPU only.
"""
import re
from collections import Counter, defaultdict

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES

RISK_RE = re.compile(r"R\d+ \[(High|Medium|Low) \| ([^\]]+)\]\s*(.*)", re.S)
MIT_RE = re.compile(r"M\d+ \(([^)]+)\):\s*(.*)", re.S)
DELAY_RE = re.compile(r"Indicative impact if unmanaged:\s*(.+?)\.\s*$", re.S)
PHASE_RE = re.compile(r"discovery and design (\d+), build and configuration (\d+), integration and data work (\d+), "
                      r"testing and UAT (\d+), cutover and hypercare (\d+)")
PHASES = ["discovery and design", "build and configuration", "integration and data work",
          "testing and UAT", "cutover and hypercare"]
SEV_RANK = {"High": 0, "Medium": 1, "Low": 2}

# Generic wording (the dataset's own risk sentences are project-specific, so we do not copy them).
RISK_TEXT = {
    "Data migration": "Trial loads fail reconciliation because of duplicates, orphaned records and missing mandatory fields, so UAT cannot start on time.",
    "Phased rollout & coexistence": "Old and new processes run side by side longer than planned, so users double-enter data and reports disagree.",
    "Integration dependencies": "An interface owner changes a contract or is late with test access, so end-to-end testing is blocked or fails late in UAT.",
    "Availability & performance": "Load and availability targets are only tested late, so tuning and re-testing push the release date.",
    "User adoption & training": "Users keep using spreadsheets or email because training and floor support are too thin at go-live.",
    "Payments, billing & reconciliation": "Amounts or statuses do not reconcile between systems, so finance will not sign off until differences are explained.",
    "Key-person & skills dependency": "One or two people hold critical platform or domain knowledge; their absence stalls design and build.",
    "Data quality & reporting trust": "Business users do not trust the new reports because metric definitions and source data are not agreed.",
    "AI answer quality & knowledge": "Model or knowledge answers are wrong or outdated for edge cases, so review effort and tuning cycles exceed the plan.",
    "Real-time processing": "Event volume or latency in real-time flows is higher than tested, causing delays or duplicate processing.",
    "Big-bang cutover": "A single cutover leaves no safe rollback window if critical defects appear after go-live.",
    "Scope growth": "New requests arrive during build and are absorbed without re-planning, pushing delivery dates.",
    "Offline use & data sync": "Offline edits conflict on sync, creating duplicate or overwritten records in the field.",
    "Identity & access mapping": "Roles and permissions do not map cleanly from the source, leaving users blocked or over-privileged.",
    "Privacy, consent & retention": "Consent, retention or deletion rules are interpreted late, forcing design changes and legal review before launch.",
    "External users & partners": "Partner or customer onboarding is slower than planned because their teams and support capacity are not ready.",
    "Device & telemetry reliability": "Device or telemetry feeds are unreliable or inconsistent, so test data does not reflect production behaviour.",
    "Security & compliance": "Security review findings arrive late and need rework before go-live.",
    "UAT readiness & environments": "Test environments, test data or business testers are not ready, so UAT starts late and runs short.",
    "Stakeholder decisions & approvals": "Open business decisions or approvals take longer than planned, leaving the team waiting.",
    "Platform limits & release changes": "Platform limits or vendor release changes affect the design and need rework.",
    "Custom-build complexity": "Custom code takes longer to build, test and maintain than configuration would, so estimates prove optimistic.",
    "Localisation & accessibility": "Language, regional or accessibility needs surface late and require rework of screens and content.",
    "Support handover & hypercare": "Support teams are not ready to take over, so the project team stays in hypercare longer than planned.",
}
MITIGATION = {
    "Data migration": "Profile and cleanse source data before build, freeze the field mapping early, and run timed trial loads with record-count reconciliation before UAT.",
    "Phased rollout & coexistence": "Define which system is the source of truth per wave, set a fixed coexistence end date, and publish a daily reconciliation report.",
    "Integration dependencies": "Agree interface contracts and test accounts in the first sprints, build contract tests and mocks, and name one accountable owner per interface.",
    "Availability & performance": "Set load and availability targets up front, test early with production-like data, and reserve time for tuning.",
    "User adoption & training": "Train by role before go-live, provide floor support for the first weeks, and track usage against the old process.",
    "Payments, billing & reconciliation": "Agree reconciliation rules with finance early and build a daily reconciliation report before UAT.",
    "Key-person & skills dependency": "Pair on critical areas, document decisions as you go, and name a deputy for each key role.",
    "Data quality & reporting trust": "Agree metric definitions with the business before building reports and validate them against a known baseline.",
    "AI answer quality & knowledge": "Keep a human in the loop, test on a labelled sample, set confidence thresholds and review drift regularly.",
    "Real-time processing": "Load-test at peak event rates, make processing idempotent and monitor queue depth and latency.",
    "Big-bang cutover": "Rehearse cutover at least twice, define go/no-go criteria and keep a tested rollback path.",
    "Scope growth": "Baseline the scope, route changes through one owner and re-estimate before accepting them.",
    "Offline use & data sync": "Define conflict rules up front and test sync with poor connectivity before rollout.",
    "Identity & access mapping": "Map source roles to target roles early and have business owners sign off the matrix before UAT.",
    "Privacy, consent & retention": "Confirm consent, retention and deletion rules with legal in discovery and test them as acceptance criteria.",
    "External users & partners": "Onboard a small group of partners first and agree support capacity and integration limits before expanding.",
    "Device & telemetry reliability": "Test with real device data early and design for gaps, duplicates and late events.",
    "Security & compliance": "Run the security review early, fix findings in the sprint they appear and keep audit evidence as you build.",
    "UAT readiness & environments": "Book test environments and business testers in week one and prepare test data before UAT starts.",
    "Stakeholder decisions & approvals": "Keep a decision log with owners and due dates and escalate anything older than a week.",
    "Platform limits & release changes": "Check platform limits and the vendor release calendar during design and keep a solution backup before each release.",
    "Custom-build complexity": "Prefer configuration over code, keep code small and covered by tests, and review designs before building.",
    "Localisation & accessibility": "Collect language and accessibility needs in discovery and test them with real users before go-live.",
    "Support handover & hypercare": "Agree the handover checklist and support runbook early and keep the team on call for a short, fixed hypercare window.",
}
_has = lambda pat: (lambda f, t: bool(re.search(pat, t)))
GATES = {
    "Data migration": lambda f, t: f["migration"] > 0,
    "Integration dependencies": lambda f, t: f["n_integrations"] > 0,
    "Payments, billing & reconciliation": _has(r"payment|billing|invoice|subscription|refund|reconcil"),
    "AI answer quality & knowledge": _has(r"\bai\b|chatbot|virtual agent|machine learning|triage|knowledge|model endpoint|copilot"),
    "Offline use & data sync": _has(r"offline|mobile|field"),
    "Device & telemetry reliability": _has(r"iot|telemetry|telematics|device|meter|scada"),
    "Real-time processing": _has(r"real-time|near-real|streaming|event-driven|webhook"),
    "Privacy, consent & retention": _has(r"consent|privacy|gdpr|retention|personal data|deletion"),
    "External users & partners": _has(r"partner|portal|external|customer-facing|self-service"),
    "Big-bang cutover": _has(r"single cutover|cutover weekend|all-at-once|big-bang"),
}
# (feature columns, baseline) - ablated one group at a time to see what drives the duration
DRIVER_GROUPS = {
    "integrations": (["n_integrations", "integration_kw"], 0),
    "migration": (["migration"], 0),
    "compliance": (["compliance"], 0),
    "volume": (["log_volume"], 0),
    "scope": (["scope_items"], 3),
    "availability": (["availability"], 0),
    "rollout": (["pilot"], 0),
}


def bucket(weeks):
    return "small" if weeks <= 8 else "medium" if weeks <= 20 else "large" if weeks <= 35 else "xl"


def _plural(n, word):
    return f"{n} {word}" + ("" if n == 1 else "s")


class Explainer:
    def fit(self, df, X, n_feat, n_emb, n_pca=16):
        """df needs columns: risks, mitigations, reasoning (lists / str); X = model matrix for the same rows."""
        self.n_feat = n_feat
        sev, delay, mit = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
        cats_per_row, weeks = [], df["weeks"].to_numpy()
        for risks, mits, w in zip(df["risks"], df["mitigations"], weeks):
            cs = set()
            for r in risks:
                m = RISK_RE.match(r)
                if not m:
                    continue
                s, c, body = m.groups()
                cs.add(c)
                sev[(c, bucket(w))][s] += 1
                d = DELAY_RE.search(body)
                if d:
                    delay[(c, bucket(w))][d.group(1).strip()] += 1
            for x in mits:
                m = MIT_RE.match(x)
                if m and not re.search(r"\d", m.group(2)):
                    mit[m.group(1)][m.group(2).strip()] += 1
            cats_per_row.append(cs)
        counts = Counter(c for cs in cats_per_row for c in cs)
        self.cats = [c for c, n in counts.items() if n >= 15]
        self.sev = {k: v.most_common(1)[0][0] for k, v in sev.items()}
        self.sev_any = {c: Counter({s: n for (cc, _), v in sev.items() if cc == c for s, n in v.items()}).most_common(1)[0][0]
                        for c in self.cats}
        self.delay = {k: v.most_common(1)[0][0] for k, v in delay.items()}
        self.delay_any = {c: Counter({s: n for (cc, _), v in delay.items() if cc == c for s, n in v.items()}).most_common(1)[0][0]
                          for c in self.cats if any(cc == c for (cc, _) in delay)}
        # mitigation text: dataset's most common digit-free wording if it is common enough, else our default
        self.mitigation = {c: (mit[c].most_common(1)[0][0] if mit[c] and mit[c].most_common(1)[0][1] >= 5
                               else MITIGATION.get(c, "Agree an owner and a date for this risk in the first sprint."))
                           for c in self.cats}

        # multi-label classifier: which risk categories apply to this BRD?
        idx = [i for i, cs in enumerate(cats_per_row) if cs]
        Y = np.array([[c in cats_per_row[i] for c in self.cats] for i in idx], dtype=int)
        Xr = X[idx]
        self.clf = self._make(n_feat, n_emb, n_pca).fit(Xr, Y)
        aucs = []
        for tr, va in KFold(3, shuffle=True, random_state=0).split(Xr):
            m = self._make(n_feat, n_emb, n_pca).fit(Xr[tr], Y[tr])
            P = m.predict_proba(Xr[va])
            for j in range(len(self.cats)):
                if 0 < Y[va, j].sum() < len(va):
                    aucs.append(roc_auc_score(Y[va, j], P[:, j]))
        self.risk_auc = float(np.mean(aucs)) if aucs else float("nan")

        # phase split fractions by size bucket
        acc = defaultdict(list)
        for txt, w in zip(df["reasoning"], weeks):
            m = PHASE_RE.search(txt or "")
            if m:
                v = np.array([int(x) for x in m.groups()], float)
                if v.sum() > 0:
                    acc[bucket(w)].append(v / v.sum())
        self.phase_frac = {b: np.mean(v, axis=0) for b, v in acc.items()}
        allv = [x for v in acc.values() for x in v]
        self.phase_default = np.mean(allv, axis=0) if allv else np.array([.12, .33, .22, .22, .11])
        return self

    @staticmethod
    def _make(n_feat, n_emb, n_pca):
        pre = ColumnTransformer([
            ("rules", StandardScaler(), list(range(n_feat))),
            ("emb", Pipeline([("pca", PCA(n_components=min(n_pca, n_emb), random_state=0)), ("sc", StandardScaler())]),
             list(range(n_feat, n_feat + n_emb))),
        ])
        return Pipeline([("pre", pre), ("clf", OneVsRestClassifier(LogisticRegression(C=0.5, max_iter=400)))])

    # ------------------------------------------------------------------ inference
    def explain(self, X, weeks_model, weeks, staff, text):
        f = dict(zip(FEATURE_NAMES, X[0, :self.n_feat]))
        t = text.lower()
        b = bucket(weeks)
        return {"risks": self._risks(X, f, t, b),
                **self._reasoning(X, weeks_model, weeks, staff, f, b)}

    def _risks(self, X, f, t, b):
        p = self.clf.predict_proba(X)[0]
        scored = []
        for c, pi in zip(self.cats, p):
            gate = GATES.get(c)
            if gate is not None and not gate(f, t):
                continue
            scored.append((pi, c))
        scored.sort(reverse=True)
        k = min(6, max(3, sum(pi >= 0.5 for pi, _ in scored)))
        chosen = scored[:k]
        # strong, explicit triggers in the BRD must always show up
        strong = {"AI answer quality & knowledge": bool(re.search(r"\bai\b|chatbot|virtual agent|machine learning|triage|copilot", t)),
                  "Integration dependencies": f["n_integrations"] >= 2,
                  "Data migration": f["migration"] >= 2}
        have = {c for _, c in chosen}
        for pi, c in scored:
            if strong.get(c) and c not in have:
                chosen = chosen[:5] + [(pi, c)]
                have.add(c)
        out = []
        for pi, c in chosen:
            out.append({
                "category": c,
                "severity": self.sev.get((c, b), self.sev_any.get(c, "Medium")),
                "likelihood_score": round(float(pi), 2),
                "risk": RISK_TEXT.get(c, c),
                "impact_if_unmanaged": self.delay.get((c, b), self.delay_any.get(c, "")),
                "mitigation": self.mitigation[c],
            })
        out.sort(key=lambda r: (SEV_RANK[r["severity"]], -r["likelihood_score"]))
        return out

    def _reasoning(self, X, weeks_model, weeks, staff, f, b):
        base = float(np.expm1(weeks_model.predict(X)[0]))
        drivers = []
        for name, (cols, baseline) in DRIVER_GROUPS.items():
            Xa = X.copy()
            for c in cols:
                j = FEATURE_NAMES.index(c)
                Xa[0, j] = min(Xa[0, j], baseline) if name == "scope" else baseline
            delta = base - float(np.expm1(weeks_model.predict(Xa)[0]))
            if delta >= max(0.5, 0.06 * weeks):
                drivers.append((delta, name))
        drivers.sort(reverse=True)

        def line(name, d):
            n = lambda k: int(f[k])
            return {
                "integrations": f"{_plural(n('n_integrations'), 'external integration')} - each needs a specified contract, build against another team's schedule and end-to-end testing",
                "migration": "data migration needs profiling, mapping, cleansing and repeated trial loads before UAT",
                "compliance": f"{_plural(n('compliance'), 'compliance/privacy control')} must be designed, built and evidenced in testing",
                "volume": "high user or record volume adds performance testing and tuning cycles",
                "scope": f"{_plural(n('scope_items'), 'scope item')} to design, configure or build, and test",
                "availability": "the high-availability target adds resilience design and extra test cycles",
                "rollout": "a pilot or phased rollout adds repeated readiness and support cycles",
            }[name] + f" (about +{max(1, round(d))} wk)"

        fr = self.phase_frac.get(b, self.phase_default)
        raw = fr * weeks
        ph = np.floor(raw).astype(int)
        for i in np.argsort(-(raw - ph))[: int(round(weeks)) - int(ph.sum())]:
            ph[i] += 1
        phases = {n: int(v) for n, v in zip(PHASES, ph)}

        parts = [_plural(staff["developers"], "developer")]
        if staff["qa"]:
            parts.append(f"{staff['qa']} QA")
        if staff["business_analyst"]:
            parts.append(_plural(staff["business_analyst"], "business analyst"))
        if staff["infrastructure"]:
            parts.append(f"{staff['infrastructure']} infrastructure")
        extra = ["1 project manager"]
        if staff["solution_architect"]:
            extra.append(_plural(staff["solution_architect"], "solution architect"))
        if staff["delivery_lead"]:
            extra.append(_plural(staff["delivery_lead"], "delivery lead"))
        team = ", ".join(parts) + ", plus " + (", ".join(extra[:-1]) + " and " + extra[-1] if len(extra) > 1 else extra[0])
        why = ("; ".join(f"({i + 1}) {line(n, d)}" for i, (d, n) in enumerate(drivers))
               if drivers else "the scope is small and well bounded (few items, no external integrations, no migration), so most "
                                "of the time is design, build, testing, UAT and handover")
        text = (f"The model estimates {int(round(weeks))} weeks (about {weeks / 4.345:.1f} months) with a peak team of "
                f"{sum(staff.values())}: {team}. Indicative phase split (phases overlap in practice; the split sums to "
                f"{int(round(weeks))}): " + ", ".join(f"{k} {v}" for k, v in phases.items()) + f". Time drivers: {why}.")
        return {"duration_reasoning": text, "time_drivers": [line(n, d) for d, n in drivers], "phase_split_weeks": phases}