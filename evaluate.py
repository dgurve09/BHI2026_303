"""Cross-validation, metrics, subject-level bootstrap. One fixed classifier throughout."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 0        # fold assignment and bootstrap
N_FOLDS = 5
N_BOOT = 2000


def make_model():
    """Median impute -> standardise -> L2 logistic regression, class-balanced.

    Kept as a pipeline so the imputer and scaler never see the held-out fold.
    """
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", class_weight="balanced"),
    )


def out_of_fold_proba(df, cols, y, groups, multiclass=False):
    """Pooled out-of-fold predictions, grouped by subject (record for UCDDB)."""
    x = df[cols].replace([np.inf, -np.inf], np.nan).values
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    proba = cross_val_predict(make_model(), x, y, groups=groups, cv=cv,
                              method="predict_proba", n_jobs=1)
    return proba if multiclass else proba[:, 1]


def binary_metrics(y, proba):
    """AUC from the probabilities, the rest at the default 0.5 threshold."""
    pred = (proba >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y, proba)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
    }


def bootstrap_gain(groups, y, proba_a, proba_b, n_boot=N_BOOT, seed=SEED):
    """Delta AUC, resampling whole subjects. No refitting: the OOF predictions are fixed."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    units = np.unique(groups)
    rows = {u: np.flatnonzero(groups == u) for u in units}
    gains = []
    for _ in range(n_boot):
        idx = np.concatenate([rows[u] for u in rng.choice(units, len(units), replace=True)])
        if len(np.unique(y[idx])) < 2:
            continue
        gains.append(roc_auc_score(y[idx], proba_a[idx]) - roc_auc_score(y[idx], proba_b[idx]))
    gains = np.asarray(gains, dtype=float)
    return {
        "delta_auc": float(gains.mean()),
        "ci_low": float(np.percentile(gains, 2.5)),
        "ci_high": float(np.percentile(gains, 97.5)),
        "p": float(min(1.0, 2 * min((gains <= 0).mean(), (gains >= 0).mean()))),
    }


def bootstrap_auc(groups, y, proba, n_boot=N_BOOT, seed=SEED):
    """Percentile interval for a single AUC, resampling whole subjects."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    units = np.unique(groups)
    rows = {u: np.flatnonzero(groups == u) for u in units}
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rows[u] for u in rng.choice(units, len(units), replace=True)])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], proba[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def bootstrap_metric_gain(groups, y, pred_a, pred_b, metric, n_boot=N_BOOT, seed=SEED):
    """Same bootstrap for a metric computed from hard labels, e.g. an F1 score."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    units = np.unique(groups)
    rows = {u: np.flatnonzero(groups == u) for u in units}
    gains = []
    for _ in range(n_boot):
        idx = np.concatenate([rows[u] for u in rng.choice(units, len(units), replace=True)])
        if len(np.unique(y[idx])) < 2:
            continue
        gains.append(metric(y[idx], pred_a[idx]) - metric(y[idx], pred_b[idx]))
    gains = np.asarray(gains, dtype=float)
    return {
        "delta": float(gains.mean()),
        "ci_low": float(np.percentile(gains, 2.5)),
        "ci_high": float(np.percentile(gains, 97.5)),
        "p": float(min(1.0, 2 * min((gains <= 0).mean(), (gains >= 0).mean()))),
    }


def n1_to_rem_rate(y, pred):
    """Share of true N1 epochs that the five-class model scores as REM."""
    labels = [0, 1, 2, 3, 5]
    cm = confusion_matrix(y, pred, labels=labels)
    n1 = cm[labels.index(1)]
    return float(n1[labels.index(5)] / n1.sum()) if n1.sum() else float("nan")


def directional_auc(y, score):
    """Direction-adjusted AUC for the single-feature figure. Descriptive only."""
    score = np.asarray(score, dtype=float)
    ok = np.isfinite(score)
    raw = float(roc_auc_score(np.asarray(y)[ok], score[ok]))
    return max(raw, 1.0 - raw)


def table(rows):
    return pd.DataFrame(rows)
