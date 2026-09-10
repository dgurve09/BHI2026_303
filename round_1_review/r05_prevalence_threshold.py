"""Reviewer bCwc, question 4. How does the comparison look away from a 0.5 cut?

The threshold is set so that the predicted N1 rate matches the true prevalence,
which is the operating point a scorer would care about.
"""
from _common import *
from sklearn.metrics import balanced_accuracy_score, f1_score

rows = []
for name in ["cassette", "telemetry"]:
    df = load(name)
    y, g = label_and_group(df, name)
    n1 = 1 - y                                   # N1 is the class of interest here
    prevalence = n1.mean()
    for tag, cols in [("HC", F.hc_cols(df)), ("HC+DYN", F.hc_cols(df) + F.dyn_cols(df))]:
        p_n1 = 1 - E.out_of_fold_proba(df, cols, y, g)
        cut = np.quantile(p_n1, 1 - prevalence)  # predict N1 at the true base rate
        pred = (p_n1 >= cut).astype(int)
        tp = ((pred == 1) & (n1 == 1)).sum(); fn = ((pred == 0) & (n1 == 1)).sum()
        tn = ((pred == 0) & (n1 == 0)).sum(); fp = ((pred == 1) & (n1 == 0)).sum()
        rows.append({"subset": name, "prevalence": prevalence, "feature_set": tag,
                     "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
                     "balanced_accuracy": balanced_accuracy_score(n1, pred),
                     "f1_n1": f1_score(n1, pred)})
show(rows)
save("r05_prevalence_threshold.csv", pd.DataFrame(rows))
