"""Train + cross-validate + save.   python -m scripts.train"""
import argparse
import json
import pandas as pd
from src.pipeline import StaffingModel, load_feedback, load_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--data", nargs="+", default=["data/raw/crm_genie_staffing_merged_dataset_with_risks.jsonl",
                                              "data/raw/micro_projects.jsonl"])
ap.add_argument("--out", default="models/staffing_model.joblib")
ap.add_argument("--use-feedback", action="store_true", help="add real outcomes collected via /feedback")
a = ap.parse_args()

df = pd.concat([load_jsonl(p) for p in a.data], ignore_index=True)
if a.use_feedback:
    fb = load_feedback("data/feedback/requests.jsonl", "data/feedback/actuals.jsonl")
    print(f"feedback rows added: {len(fb)}")
    df = pd.concat([df, fb], ignore_index=True)
print(f"training rows: {len(df)}  (weeks: min {df.weeks.min()}, median {df.weeks.median():.0f}, max {df.weeks.max()})")

m = StaffingModel().fit(df)
m.save(a.out)
with open(a.out.replace(".joblib", "_metrics.json"), "w") as f:
    json.dump({"n_train": m.n_train, "duration_band": m.band, "metrics": m.metrics}, f, indent=2)
print("saved ->", a.out)