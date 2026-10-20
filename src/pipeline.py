"""Staffing model: rule features + frozen embeddings -> our own regressors trained on our data."""
import json
import re
from sklearn.base import clone
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .embedder import make_embedder
from .features import FEATURE_NAMES, extract_features
from .explain import Explainer



ROLES = ["business_analyst", "qa", "developers", "solution_architect", "delivery_lead", "infrastructure"]
WEEKS_PER_MONTH = 4.345

LONG_WEIGHTING = False   # True = long projects get more weight in training (pushes estimates up)


# ------------------------------------------------------------------ data
_LABELED = re.compile(r"^Objective:.*Scope:.*Integrations:", re.S)


def load_jsonl(path, only_labeled=True):
    """Read a dataset file. Rows whose text is not in the 'Objective/Scope/Integrations' format are skipped:
    their durations cannot be predicted from their text (audit R2 ~ 0), so they only add noise."""
    rows, skipped = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            text = r["input"]["brd_text"]
            if only_labeled and not _LABELED.search(text):
                skipped += 1
                continue
            s = r["target"]["staffing_recommendation"]
            rows.append({"name": r["input"].get("project_name", r.get("scenario_id", "")),
                         "text": text,
                         "weeks": r["target"]["estimated_duration_weeks"],
                         **{k: s[k] for k in ROLES},
                         "risks": r.get("real_life_risks", []),
                         "mitigations": r.get("risk_mitigation", []),
                         "reasoning": r.get("duration_reasoning", "")})
    if skipped:
        print(f"{path}: skipped {skipped} free-text rows (not in labeled format)")
    return pd.DataFrame(rows)


def load_feedback(requests_path, actuals_path):
    """Join logged requests with the actual outcomes people reported -> extra training rows."""
    try:
        reqs = {}
        for l in open(requests_path, encoding="utf-8"):
            r = json.loads(l)
            reqs[r["request_id"]] = r
        rows = []
        for l in open(actuals_path, encoding="utf-8"):
            a = json.loads(l)
            r = reqs.get(a["request_id"])
            if r and all(k in a["actual"] for k in ["duration_weeks"] + ROLES):
                rows.append({"name": a.get("project_name", a["request_id"]), "text": r["condensed_brd"],
                             "weeks": a["actual"]["duration_weeks"], **{k: a["actual"][k] for k in ROLES}})
        return pd.DataFrame(rows)
    except FileNotFoundError:
        return pd.DataFrame()


# ------------------------------------------------------------------ model
def _regressors():
    ridge = Ridge(alpha=5.0)
    gbr = GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0)
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=0, n_jobs=1)
    return {"ridge": ridge, "gbr": gbr, "rf": rf,
            "blend": VotingRegressor([("ridge", Ridge(alpha=5.0)),
                                      ("gbr", GradientBoostingRegressor(n_estimators=150, max_depth=2,
                                                                         learning_rate=0.05, subsample=0.8,
                                                                         random_state=0)),
                                      ("rf", RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                                                   random_state=0, n_jobs=1))])}


def _wrap(reg, n_feat, n_emb, n_pca):
    pre = ColumnTransformer([
        ("rules", StandardScaler(), list(range(n_feat))),
        ("emb", Pipeline([("pca", PCA(n_components=min(n_pca, n_emb), random_state=0)),
                          ("sc", StandardScaler())]), list(range(n_feat, n_feat + n_emb))),
    ])
    return Pipeline([("pre", pre), ("reg", reg)])


class StaffingModel:
    def __init__(self, n_pca=16):
        self.n_pca = n_pca

    # ---- helpers
    def _matrix(self, texts):
        F = np.vstack([extract_features(t) for t in texts])
        E = self.embedder.transform(texts)
        return np.hstack([F, E]), E

    def _post(self, role, pred):
        lo, hi = self.bounds[role]
        return int(np.rint(np.clip(pred, lo, hi)))

    # ---- training
    
    def fit(self, df, cv_repeats=2, verbose=True):
        texts = df["text"].tolist()
        self.embedder = make_embedder().fit(texts)

        X, E = self._matrix(texts)
        n_feat, n_emb = len(FEATURE_NAMES), E.shape[1]

        self.bounds = {
            r: (int(df[r].min()), int(df[r].max()))
            for r in ROLES
        }
        self.metrics, self.models, report = {}, {}, []

        actual_weeks = df["weeks"].to_numpy(dtype=float)
        targets = {
            "weeks": np.log1p(actual_weeks),
            **{
                r: df[r].to_numpy(dtype=float)
                for r in ROLES
            },
        }

        # Moderate weighting: give longer projects more influence.
        median_weeks = max(float(np.median(actual_weeks)), 1.0)
        
        if LONG_WEIGHTING:
            duration_weights = np.clip(np.sqrt(actual_weeks / median_weeks), 0.8, 2.0)
        else:
            duration_weights = np.ones_like(actual_weeks)

        regressors = _regressors()

        for name, y in targets.items():
            best = None

            for cname, reg in regressors.items():
                if name == "weeks":
                    # Manual CV so sample weights apply only to
                    # the training portion of each fold.
                    oof = np.zeros(len(y))

                    for rep in range(cv_repeats):
                        kf = KFold(
                            5, shuffle=True, random_state=rep
                        )

                        for train_idx, val_idx in kf.split(X):
                            pipe = _wrap(
                                clone(reg),
                                n_feat,
                                n_emb,
                                self.n_pca,
                            )
                            pipe.fit(
                                X[train_idx],
                                y[train_idx],
                                reg__sample_weight=(
                                    duration_weights[train_idx]
                                ),
                            )

                            fold_pred = pipe.predict(X[val_idx])
                            oof[val_idx] += (
                                np.expm1(fold_pred) / cv_repeats
                            )

                    pred = np.maximum(oof, 1.0)
                    true = actual_weeks
                    abs_err = np.abs(pred - true)

                    overall_mae = float(abs_err.mean())
                    long_mask = true >= 37

                    if long_mask.any():
                        long_mae = float(abs_err[long_mask].mean())
                    else:
                        long_mae = overall_mae

                    # Balance overall accuracy and long-project accuracy.
                    selection_score = (
                        0.75 * overall_mae + 0.25 * long_mae
                    ) if LONG_WEIGHTING else overall_mae
                    mae = overall_mae

                else:
                    oof = np.zeros(len(y))

                    for rep in range(cv_repeats):
                        kf = KFold(
                            5, shuffle=True, random_state=rep
                        )
                        oof += cross_val_predict(
                            _wrap(
                                clone(reg),
                                n_feat,
                                n_emb,
                                self.n_pca,
                            ),
                            X,
                            y,
                            cv=kf,
                        ) / cv_repeats

                    pred = np.array([
                        self._post(name, p) for p in oof
                    ])
                    true = y
                    mae = float(np.mean(np.abs(pred - true)))
                    selection_score = mae

                if best is None or selection_score < best["score"]:
                    best = {
                        "mae": mae,
                        "score": selection_score,
                        "model": cname,
                        "pred": pred,
                        "true": true,
                    }

            base = float(np.mean(
                np.abs(best["true"] - np.median(best["true"]))
            ))

            self.metrics[name] = {
                "cv_mae": round(best["mae"], 3),
                "baseline_mae": round(base, 3),
                "model": best["model"],
            }

            if name == "weeks":
                # Estimate an error band using out-of-fold predictions.
                ratio = (
                    np.abs(best["pred"] - best["true"])
                    / np.maximum(best["true"], 1.0)
                )
                self.band = float(np.quantile(ratio, 0.8))

            final = _wrap(
                clone(regressors[best["model"]]),
                n_feat,
                n_emb,
                self.n_pca,
            )

            if name == "weeks":
                final.fit(
                    X,
                    y,
                    reg__sample_weight=duration_weights,
                )
            else:
                final.fit(X, y)

            self.models[name] = final
            report.append((
                name,
                best["model"],
                best["mae"],
                base,
            ))

        if verbose:
            print(
                f"{'target':20s} {'model':6s} "
                f"{'CV MAE':>8s} {'baseline':>9s}"
            )
            for n, m, a, b in report:
                print(f"{n:20s} {m:6s} {a:8.2f} {b:9.2f}")

            print(
                f"Duration error band (80th percentile): "
                f"+/- {self.band:.0%}"
            )


                # risks / mitigations / duration reasoning (learned from the dataset's risk + reasoning fields)
        self.explainer = None
        dfx = df.copy()
        for col, empty in (("risks", []), ("mitigations", []), ("reasoning", "")):
            if col not in dfx.columns:
                dfx[col] = [list(empty) if isinstance(empty, list) else empty for _ in range(len(dfx))]
            dfx[col] = dfx[col].apply(lambda v, e=empty: v if isinstance(v, (list, str)) else e)
        if any(len(x) for x in dfx["risks"]):
            self.explainer = Explainer().fit(dfx, X, n_feat, n_emb, self.n_pca)
            if verbose:
                print(f"risk model: {len(self.explainer.cats)} categories, mean CV AUC {self.explainer.risk_auc:.2f}")    

        self.ref = {
            "E": E,
            "names": df["name"].tolist(),
            "weeks": df["weeks"].tolist(),
            "peak": (df[ROLES].sum(axis=1) + 1).tolist(),
        }
        self.n_train = len(df)
        return self


    # ---- inference
    def predict(self, condensed_text, top_k=3):
        X, E = self._matrix([condensed_text])
        weeks = float(np.expm1(self.models["weeks"].predict(X)[0]))
        weeks = max(weeks, 1.0)
        staff = {"project_manager": 1}
        for r in ROLES:
            staff[r] = self._post(r, self.models[r].predict(X)[0])
        sims = (self.ref["E"] @ E[0]).ravel()
        top = np.argsort(-sims)[:top_k]
        top_weeks = [self.ref["weeks"][i] for i in top]
        flags = []
        if max(top_weeks) / max(1, min(top_weeks)) >= 3:
            flags.append("similar past projects disagree widely on duration")
        if weeks >= 20:
            flags.append("large project: few training examples, review manually")
        if float(sims[top[0]]) < 0.75:
            flags.append("no close match in training data")
        confidence = "low" if len(flags) >= 2 else "medium" if flags else "high"
        out = {
            "confidence": confidence,
            "confidence_notes": flags,
            "estimated_duration_weeks": int(round(weeks)),
            "duration_range_weeks": [max(1, int(np.floor(weeks * (1 - self.band)))),
                                     int(np.ceil(weeks * (1 + self.band)))],
            "estimated_duration_months_approx": round(weeks / WEEKS_PER_MONTH, 1),
            "staffing_recommendation": staff,
            "peak_team_headcount": int(sum(staff.values())),
            "similar_projects": [{"project_name": self.ref["names"][i], "similarity": round(float(sims[i]), 3),
                                  "duration_weeks": self.ref["weeks"][i], "peak_headcount": int(self.ref["peak"][i])}
                                 for i in top],
        }
        if getattr(self, "explainer", None) is not None:
            out.update(self.explainer.explain(X, self.models["weeks"], weeks, staff, condensed_text))
        return out

    # ---- persistence
    def save(self, path):
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path):
        return joblib.load(path)
