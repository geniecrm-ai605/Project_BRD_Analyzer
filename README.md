# CRM GENIE – Staffing Recommendations model

BRD (.docx) → condensed text → rule features + bge-small embedding → our own regressors → duration range + role counts (+ 3 similar past projects).
No external API, CPU only.

## Setup
```bash
python -m venv .venv && .venv\Scripts\activate        # (Linux/Mac: source .venv/bin/activate)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
Put the dataset at `data/raw/CRM_GENIE_Staffing_Recommendations_Training_Data_Expanded.jsonl`.

## Train (first run downloads BAAI/bge-small-en-v1.5 once, ~130MB)
```bash
python -m scripts.train
```
Prints the CV table (MAE per target vs a dumb "always predict the median" baseline) and saves `models/staffing_model.joblib`.

## Try it on a BRD
```bash
python -m scripts.predict_cli path/to/BRD.docx
```
The output starts with "what the model saw" (the condensed BRD). If that text looks wrong, fix `src/brd_reader.py`, not the model.

## Run the API
```bash
set STAFFING_API_KEY=some-long-secret          # Linux/Mac: export ...
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 2
```
| Endpoint | Use |
|---|---|
| `POST /estimate/base64` | Power Automate: `{"file_name":"x.docx","file_content_base64":"<base64>"}` + header `X-API-Key` |
| `POST /estimate/file` | multipart upload |
| `POST /estimate/text` | pasted text (weaker than .docx) |
| `POST /feedback` | send the project's real duration + team once known |
| `GET /health` | status |

Power Automate: HTTP action, POST, body `file_content_base64` = `base64(outputs('Get_file_content')?['body'])`.

## Improve it over time
Every request is logged in `data/feedback/requests.jsonl`; when a project finishes, call `/feedback` with the real numbers.
Then retrain with real rows included:
```bash
python -m scripts.train --use-feedback
```

## Files
- `src/brd_reader.py` – docx → sections → condensed text (Objective/Scope/Integrations/Data/NFR/Rollout)
- `src/features.py` – rule features (integrations, migration, compliance, volume, sites, complexity…)
- `src/embedder.py` – bge-small (frozen) ; `EMBED_BACKEND=tfidf` is an offline test fallback only
- `src/pipeline.py` – training, CV, prediction, save/load
- `scripts/train.py`, `scripts/predict_cli.py`, `app.py`
