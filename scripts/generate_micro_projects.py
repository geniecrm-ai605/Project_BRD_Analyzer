"""Small CRM/IT enhancement projects (1-10 weeks) - the size range the dataset was missing.

    python -m scripts.generate_micro_projects --n 450 --out data/raw/micro_projects.jsonl

Same schema as the main dataset (incl. risks / mitigations / reasoning). Labels come from a formula.
"""
import argparse, json, math, os, random, re
from collections import Counter
from statistics import median
from src.explain import RISK_TEXT, MITIGATION

BUSINESS = ["Retail", "Telecommunications", "Utilities", "Retail banking", "Insurance", "Healthcare provider",
            "Public sector agency", "Manufacturing", "Logistics and freight", "B2B software company",
            "Higher education", "Hospitality", "Professional services", "Pharmaceutical distribution",
            "Automotive dealer group", "Energy retailer", "Wholesale distribution", "Media company"]
PLATFORMS = ["Dynamics 365 Sales", "Dynamics 365 Customer Service", "Power Platform with Dataverse",
             "Salesforce Sales Cloud", "Salesforce Service Cloud", "Zoho CRM", "HubSpot CRM", "ServiceNow ITSM"]
# (item, effort weeks, kind)
ITEMS = [
    ("custom columns and form layout change", 0.4, "config"),
    ("client-side JavaScript validation on the main form", 0.9, "code"),
    ("server-side plug-in that blocks invalid saves", 1.4, "code"),
    ("business rule and field-level security change", 0.5, "config"),
    ("approval flow with Power Automate", 1.0, "flow"),
    ("email template and notification update", 0.4, "config"),
    ("view and dashboard update", 0.6, "report"),
    ("duplicate detection rule", 0.5, "config"),
    ("lead assignment rule", 0.6, "config"),
    ("case routing rule update", 0.7, "config"),
    ("security role clean-up", 0.6, "config"),
    ("custom command bar button", 0.5, "code"),
    ("scheduled report refresh and export", 0.7, "report"),
    ("single webhook to an external system", 1.6, "integration"),
    ("one-off CSV data import", 0.8, "data"),
    ("landing page with lead capture form", 1.2, "code"),
    ("survey form and response tracking", 0.9, "flow"),
    ("SLA timer adjustment", 0.6, "config"),
    ("knowledge article template", 0.4, "config"),
    ("workflow automation for task creation", 0.8, "flow"),
    ("custom field mapping for an existing report", 0.5, "report"),
    ("quote template and discount field", 0.7, "config"),
]
EXT = ["ERP finance system", "email gateway", "SMS gateway", "marketing automation platform", "identity provider",
       "payment gateway", "document management system", "data warehouse"]
ROLLOUT = ["a single release to production after UAT", "a pilot with one team before full release",
           "release to all users on one weekend"]
SPLIT = [.20, .40, .05, .25, .10]


def build(rng):
    biz, plat = rng.choice(BUSINESS), rng.choice(PLATFORMS)
    items = rng.sample(ITEMS, rng.randint(1, 4))
    kinds = Counter(k for _, _, k in items)
    users = int(10 ** rng.uniform(1.2, 2.8))
    n_int = 1 if kinds["integration"] else 0
    imp = kinds["data"] > 0
    w = 1.5 + sum(e for _, e, _ in items) + 0.4 * math.log10(users / 20 + 1) + (0.5 if rng.random() < .3 else 0)
    weeks = int(min(10, max(1, round(w * rng.lognormvariate(0, 0.08)))))
    code = kinds["code"] + kinds["integration"]
    devs = 1 if (weeks <= 4 or code == 0 and weeks <= 6) else 2
    ba = 1 if weeks >= 3 else (1 if rng.random() < .4 else 0)
    qa = 1 if weeks >= 3 else 0
    sa = 1 if (n_int and weeks >= 5) else 0
    infra = 1 if (n_int and weeks >= 6) else 0
    staffing = {"project_manager": 1, "business_analyst": ba, "qa": qa, "developers": devs,
                "solution_architect": sa, "delivery_lead": 0, "infrastructure": infra}
    names = [i for i, _, _ in items]
    scope = "; ".join(names + rng.sample(["solution packaging and deployment", "user acceptance testing",
                                          "short admin guide", "audit history on changed fields"], rng.randint(1, 2)))
    integ = rng.choice(EXT) if n_int else "none"
    n_rec = rng.randint(2, 30) * 1000
    data = f"one-off import of about {n_rec:,} records" if imp else "no migration"
    brd = (f"Objective: {biz} needs a small enhancement to {plat}: {names[0]}. Scope: {scope}. Integrations: {integ}. "
           f"Data: {data}. Non-functional needs: performance for {users} named users, normal form interaction. "
           f"Rollout: {rng.choice(ROLLOUT)}.")
    name = f"{biz} - {names[0]} ({plat})"
    # phases (sum to weeks)
    raw = [s * weeks for s in SPLIT]; ph = [int(x) for x in raw]
    for i in sorted(range(5), key=lambda i: -(raw[i] - ph[i]))[: weeks - sum(ph)]:
        ph[i] += 1
    # risks: 3-4 categories typical for small changes
    pool = [("UAT readiness & environments", .8), ("Stakeholder decisions & approvals", .6), ("User adoption & training", .6),
            ("Scope growth", .5), ("Key-person & skills dependency", .4), ("Platform limits & release changes", .5 if code else .15),
            ("Custom-build complexity", .7 if code else .1), ("Availability & performance", .25),
            ("Data quality & reporting trust", .6 if kinds["report"] else .1),
            ("Identity & access mapping", .5 if kinds["config"] else .1),
            ("Integration dependencies", 1.0 if n_int else 0), ("Data migration", 1.0 if imp else 0)]
    chosen = [c for c, p in pool if rng.random() < p]
    while len(chosen) < 3:
        c = rng.choice([c for c, _ in pool[:6]])
        if c not in chosen: chosen.append(c)
    chosen = chosen[:5]
    risks, mits = [], []
    for i, c in enumerate(chosen, 1):
        sev = "Medium" if c in ("UAT readiness & environments", "Integration dependencies", "Data migration",
                                "Custom-build complexity") else "Low"
        risks.append(f"R{i} [{sev} | {c}] {RISK_TEXT[c]} Indicative impact if unmanaged: up to 1 week of delay.")
        mits.append(f"M{i} ({c}): {MITIGATION[c]}")
    reasoning = (f"The model estimates {weeks} weeks with a peak team of {sum(staffing.values())}. Indicative phase split "
                 f"(phases overlap in practice; the split sums to {weeks}): discovery and design {ph[0]}, build and "
                 f"configuration {ph[1]}, integration and data work {ph[2]}, testing and UAT {ph[3]}, cutover and hypercare {ph[4]}. "
                 f"Time drivers: small, well-bounded change set ({len(items)} items).")
    return {"scenario_id": None, "input": {"project_name": name, "brd_text": brd},
            "target": {"estimated_duration_weeks": weeks, "estimated_duration_months_approx": round(weeks / 4.345, 1),
                       "staffing_recommendation": staffing, "peak_team_headcount": sum(staffing.values())},
            "real_life_risks": risks, "risk_mitigation": mits, "duration_reasoning": reasoning}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=450)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/raw/micro_projects.jsonl")
    ap.add_argument("--exclude", default="data/raw/crm_genie_staffing_merged_dataset_with_risks.jsonl")
    a = ap.parse_args()
    rng, seen_t, seen_n = random.Random(a.seed), set(), set()
    if os.path.exists(a.exclude):
        for l in open(a.exclude, encoding="utf-8"):
            if l.strip():
                r = json.loads(l); seen_t.add(r["input"]["brd_text"].lower()); seen_n.add(r["input"]["project_name"].lower())
    rows, tries = [], 0
    while len(rows) < a.n and tries < a.n * 100:
        tries += 1
        r = build(rng); t, n = r["input"]["brd_text"].lower(), r["input"]["project_name"].lower()
        if t in seen_t or n in seen_n:
            continue
        seen_t.add(t); seen_n.add(n); r["scenario_id"] = f"MI{len(rows) + 1:04d}"; rows.append(r)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    w = [r["target"]["estimated_duration_weeks"] for r in rows]; t = [r["target"]["peak_team_headcount"] for r in rows]
    print(f"written {len(rows)} rows -> {a.out} | weeks min {min(w)} median {median(w)} max {max(w)} | team min {min(t)} median {median(t)} max {max(t)}")

if __name__ == "__main__":
    main()