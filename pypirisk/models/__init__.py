"""Train, evaluate, register and batch-score stages, plus the estimators they use.

Candidates are logistic regression and LightGBM; the one with the best validation
PR-AUC is calibrated on the validation snapshot and compared with the "months since
last release" heuristic on the held-out test snapshot.
"""
