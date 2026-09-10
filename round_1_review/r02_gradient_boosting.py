"""Reviewer bCwc major concern 3 and Reviewer RtY2 major comment 4.

Does the ordering of the feature sets survive a nonlinear classifier? The same
folds are used, with gradient boosting in place of logistic regression.
"""
from _common import *
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

rows = []
for name in ["cassette", "telemetry", "ucddb"]:
    df = load(name)
    y, g = label_and_group(df, name)
    sets = {"HC": F.hc_cols(df), "HC+DYN": F.hc_cols(df) + F.dyn_cols(df)}
    proba = {}
    for tag, cols in sets.items():
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=E.SEED)
        x = df[cols].replace([np.inf, -np.inf], np.nan).values.astype(np.float32)
        f = OUT / "_oof" / ("r02_%s_%s.npy" % (name, tag.replace("+", "_")))
        f.parent.mkdir(parents=True, exist_ok=True)
        if f.exists():
            proba[tag] = np.load(f)
        else:
            proba[tag] = cross_val_predict(HistGradientBoostingClassifier(random_state=E.SEED),
                                           x, y, groups=g, cv=cv, method="predict_proba")[:, 1]
            np.save(f, proba[tag])
        rows.append({"dataset": name, "classifier": "boosting", "feature_set": tag,
                     "auc": roc_auc_score(y, proba[tag])})
        print("  %-10s boosting %-7s AUC %.3f" % (name, tag, rows[-1]["auc"]), flush=True)
    d = E.bootstrap_gain(g, y, proba["HC+DYN"], proba["HC"])
    rows.append({"dataset": name, "classifier": "boosting", "feature_set": "HC+DYN vs HC", **d})
    print("  %-10s delta %+0.3f [%+0.3f, %+0.3f] p=%.3f"
          % (name, d["delta_auc"], d["ci_low"], d["ci_high"], d["p"]), flush=True)
save("r02_gradient_boosting.csv", pd.DataFrame(rows))
