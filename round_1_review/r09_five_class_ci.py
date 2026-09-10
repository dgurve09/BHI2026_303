"""Reviewer qwPP (limitations). The five-class table had no uncertainty.

Subject-level bootstrap intervals for macro F1 and for the F1 of N1, using the
same folds and the same 2000 resamples as the binary analysis.
"""
from _common import *
from sklearn.metrics import f1_score

rows = []
for name in ["cassette", "telemetry"]:
    df = D.extract_sleep_edf(name, task="five_class")
    if df is None:
        raise SystemExit("five-class extraction not finished, run run_all.py extract again")
    y, g = df.stage.values, df.subject.values
    preds = {}
    for tag, cols in [("HC", F.hc_cols(df)), ("HC+DYN", F.hc_cols(df) + F.dyn_cols(df))]:
        p = E.out_of_fold_proba(df, cols, y, g, multiclass=True)
        classes = np.array(sorted(np.unique(y)))
        preds[tag] = classes[p.argmax(axis=1)]
        rows.append({"subset": name, "feature_set": tag,
                     "macro_f1": f1_score(y, preds[tag], average="macro"),
                     "f1_n1": f1_score(y, preds[tag], labels=[1], average="macro"),
                     "n1_to_rem": E.n1_to_rem_rate(y, preds[tag])})
        print("  %-10s %-7s macro F1 %.3f  F1 N1 %.3f  N1->REM %.1f%%"
              % (name, tag, rows[-1]["macro_f1"], rows[-1]["f1_n1"], 100 * rows[-1]["n1_to_rem"]), flush=True)
    for label, metric in [("macro_f1", lambda a, b: f1_score(a, b, average="macro")),
                          ("f1_n1", lambda a, b: f1_score(a, b, labels=[1], average="macro"))]:
        d = E.bootstrap_metric_gain(g, y, preds["HC+DYN"], preds["HC"], metric)
        rows.append({"subset": name, "feature_set": "gain " + label, **d})
        print("  %-10s gain %-8s %+0.3f [%+0.3f, %+0.3f] p=%.3f"
              % (name, label, d["delta"], d["ci_low"], d["ci_high"], d["p"]), flush=True)
save("r09_five_class_ci.csv", pd.DataFrame(rows))
