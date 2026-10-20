"""
Re-save the model stripping internal numpy random state from all fitted estimators.
Random state is only used DURING training (for subsampling, tree splits, etc.).
After training it is embedded inside estimator objects but never used again at inference time.
Removing it makes the joblib pickle cross-compatible between NumPy 1.x and 2.x.

Usage:
    python scripts/resave_model_compat.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import joblib
import numpy as np

MODEL_IN  = os.getenv("MODEL_IN",  "models/staffing_model.joblib")
MODEL_OUT = os.getenv("MODEL_OUT", "models/staffing_model.joblib")  # overwrites in-place by default


def strip_random_state(obj, visited=None):
    """Recursively walk the object graph and replace numpy RandomState /
    BitGenerator instances with a plain integer seed=0 wherever safe."""
    if visited is None:
        visited = set()
    obj_id = id(obj)
    if obj_id in visited:
        return obj
    visited.add(obj_id)

    # numpy RandomState and Generator objects — replace with 0
    if isinstance(obj, (np.random.RandomState, np.random.Generator)):
        return 0  # sklearn accepts int seeds; 0 means deterministic

    # For objects with __dict__ (most sklearn estimators)
    if hasattr(obj, "__dict__"):
        for attr, val in list(obj.__dict__.items()):
            new_val = strip_random_state(val, visited)
            if new_val is not val:
                try:
                    setattr(obj, attr, new_val)
                except (AttributeError, TypeError):
                    pass

    # For lists / tuples
    if isinstance(obj, list):
        for i, item in enumerate(obj):
            new_item = strip_random_state(item, visited)
            if new_item is not item:
                obj[i] = new_item

    # For dicts
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            new_v = strip_random_state(v, visited)
            if new_v is not v:
                obj[k] = new_v

    return obj


def main():
    print(f"NumPy version: {np.__version__}")
    print(f"Loading model from: {MODEL_IN}")
    model = joblib.load(MODEL_IN)
    print("Model loaded successfully.")

    print("Stripping numpy random state from all estimators...")
    strip_random_state(model)
    print("Done stripping random state.")

    print(f"Saving cleaned model to: {MODEL_OUT}")
    joblib.dump(model, MODEL_OUT, compress=3)
    print("Model re-saved successfully!")
    print()
    print("Verify by loading it back...")
    model2 = joblib.load(MODEL_OUT)
    print(f"Re-load OK. n_train={model2.n_train}")


if __name__ == "__main__":
    main()
