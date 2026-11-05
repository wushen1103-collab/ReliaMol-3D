#!/usr/bin/env python3
"""Minimal inference for the ReliaMol-3D 41-feature deployment model."""
import argparse
import joblib
import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--model", default="reliamol3d_logistic_41feature_deployment.joblib")
p.add_argument("--input", required=True, help="CSV containing all 41 named features")
p.add_argument("--output", required=True)
a = p.parse_args()
bundle = joblib.load(a.model)
frame = pd.read_csv(a.input)
x = frame[bundle["feature_order"]].replace([np.inf, -np.inf], np.nan)
x = bundle["imputer"].transform(x)
x = bundle["scaler"].transform(x)
frame["structural_usability_score"] = bundle["model"].predict_proba(x)[:, 1]
frame.to_csv(a.output, index=False)
