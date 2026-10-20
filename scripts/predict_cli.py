"""python -m scripts.predict_cli path/to/BRD.docx   (or a .txt file)"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # model is cached after training: no internet calls / retries / warnings

import json
import sys
from src.brd_reader import condense_docx, ensure_condensed
from src.pipeline import StaffingModel

path = sys.argv[1]
model = StaffingModel.load("models/staffing_model.joblib")
text = condense_docx(path) if path.lower().endswith(".docx") else ensure_condensed(open(path, encoding="utf-8").read())
print("--- what the model saw ---\n" + text + "\n")
print(json.dumps(model.predict(text), indent=2))
