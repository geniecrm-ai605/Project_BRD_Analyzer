
import json
from collections import Counter
from pathlib import Path

path = Path("data/raw/crm_genie_staffing_merged_dataset_with_risks.jsonl")

records = []
bad_lines = []
required_roles = [
    "business_analyst", "qa", "developers",
    "solution_architect", "delivery_lead", "infrastructure"
]

with path.open(encoding="utf-8") as f:
    for line_no, line in enumerate(f, 1):
        if not line.strip():
            continue
        try:
            records.append((line_no, json.loads(line)))
        except json.JSONDecodeError as e:
            bad_lines.append((line_no, str(e)))

ids = [r.get("scenario_id") for _, r in records]
missing_ids = sum(not x for x in ids)
duplicate_ids = [k for k, n in Counter(ids).items() if k and n > 1]

missing_fields = []
invalid_targets = []

for line_no, r in records:
    inp = r.get("input", {})
    target = r.get("target", {})
    staff = target.get("staffing_recommendation", {})

    if not inp.get("brd_text") or "estimated_duration_weeks" not in target:
        missing_fields.append(line_no)
        continue

    try:
        weeks = float(target["estimated_duration_weeks"])
        roles = [float(staff[k]) for k in required_roles]
        if weeks <= 0 or any(x < 0 for x in roles):
            invalid_targets.append(line_no)
    except (TypeError, ValueError, KeyError):
        invalid_targets.append(line_no)

print("Valid JSON records:", len(records))
print("Malformed JSON lines:", len(bad_lines))
print("Missing scenario IDs:", missing_ids)
print("Duplicate scenario IDs:", len(duplicate_ids))
print("Missing required fields:", len(missing_fields))
print("Invalid duration/staffing targets:", len(invalid_targets))

if bad_lines:
    print("Malformed examples:", bad_lines[:5])
if duplicate_ids:
    print("Duplicate ID examples:", duplicate_ids[:10])
if missing_fields:
    print("Missing-field line examples:", missing_fields[:10])
if invalid_targets:
    print("Invalid-target line examples:", invalid_targets[:10])
