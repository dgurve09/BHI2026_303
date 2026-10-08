"""Reviewer RtY2, major comment 3. A true cross-dataset test.

The model is trained on Sleep-EDF cassette and applied to UCDDB with no
retraining. Two scalings are compared, the one used in the manuscript and a
per-dataset z-score, against a model retrained within UCDDB. The direction of
each feature group is also compared between the two cohorts.
"""
from _common import *
from sklearn.metrics import roc_auc_score

ca, uc = load("cassette"), load("ucddb")
shared = [c for c in F.hc_cols(ca) + F.dyn_cols(ca) if c in uc.columns]
hc = [c for c in shared if c.startswith("common_")]
dyn = [c for c in shared if not c.startswith("common_")]
y_c = (ca.stage_name == "REM").astype(int).values
y_u, g_u = label_and_group(uc, "ucddb")

rows = []
for tag, cols in [("HC", hc), ("DYN", dyn), ("HC+DYN", hc + dyn)]:
    within = roc_auc_score(y_u, E.out_of_fold_proba(uc, cols, y_u, g_u))
    m = E.make_model().fit(ca[cols].replace([np.inf, -np.inf], np.nan), y_c)
    p_src = m.predict_proba(uc[cols].replace([np.inf, -np.inf], np.nan))[:, 1]
    zc = ca[cols].replace([np.inf, -np.inf], np.nan)
    zu = uc[cols].replace([np.inf, -np.inf], np.nan)
    mz = E.make_model().fit((zc - zc.mean()) / (zc.std() + F.EPS), y_c)
    p_z = mz.predict_proba((zu - zu.mean()) / (zu.std() + F.EPS))[:, 1]
    rows.append({"feature_set": tag, "source_fitted_scaling": roc_auc_score(y_u, p_src),
                 "per_dataset_zscore": roc_auc_score(y_u, p_z), "within_ucddb": within,
                 "distinct_predicted_values": int(len(np.unique(np.round(p_src, 9))))})
    print("  %-7s source-scaled %.3f (%d distinct values)  z-scored %.3f  within-UCDDB %.3f"
          % (tag, rows[-1]["source_fitted_scaling"], rows[-1]["distinct_predicted_values"],
             rows[-1]["per_dataset_zscore"], within), flush=True)
save("r10_transfer.csv", pd.DataFrame(rows))

groups = {
    "spectral-path shape": [c for c in dyn if "trajectory" in c or "band_state_entropy" in c],
    "band-ratio trajectories": [c for c in dyn if any(r in c for r in
                                ["theta_alpha", "alpha_sigma", "sigma_beta", "slow_fast"])],
    "band-power trajectories": [c for c in dyn if "_power_" in c],
    "window-level descriptors": [c for c in dyn if any(w in c for w in ["slope", "entropy", "centroid"])
                                 and "band_state" not in c],
    "per-band high-state switching": [c for c in dyn if c.endswith("_high_state_switching") and "_power" not in c],
}
grows = []
for tag, cols in groups.items():
    if not cols:
        continue
    rev, aucs = 0, []
    for c in cols:
        a_c = roc_auc_score(y_c, ca[c].fillna(ca[c].median()))
        a_u = roc_auc_score(y_u, uc[c].fillna(uc[c].median()))
        rev += int((a_c - 0.5) * (a_u - 0.5) < 0)
        aucs.append(max(a_u, 1 - a_u))
    grows.append({"group": tag, "n": len(cols), "direction_reversed_pct": 100 * rev / len(cols),
                  "median_ucddb_auc": float(np.median(aucs))})
    print("  %-30s n=%2d reversed %3.0f%%  median UCDDB AUC %.3f"
          % (tag, len(cols), grows[-1]["direction_reversed_pct"], grows[-1]["median_ucddb_auc"]), flush=True)
save("r10_group_direction.csv", pd.DataFrame(grows))
