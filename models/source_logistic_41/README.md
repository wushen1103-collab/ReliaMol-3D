# ReliaMol-3D Logistic deployment package

This package is the requested fitted 41-feature Logistic model for deployment and reproducibility.
It was fitted once on all 10000 source candidates from 100 pockets.
It is distinct from the pocket-held-out models used for benchmark estimates.

Run:

    python predict.py --model reliamol3d_logistic_41feature_deployment.joblib --input features.csv --output scores.csv

The input CSV must contain the 41 columns in `feature_order.csv`. The package stores the training-set median imputer,
StandardScaler, fitted coefficient vector and intercept, software versions, and the fixed feature order.
