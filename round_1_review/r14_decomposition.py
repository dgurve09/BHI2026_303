"""Reviewer afFY. Which part of the representation is doing the work?

The proposed features are grouped by what they measure, and each group is
evaluated on the same folds. Level is the window-averaged value of a
conventional descriptor. Occupancy is the proportion of the epoch spent in a
spectral state. Neither of those can be recovered from a single 30-s spectrum in
the case of occupancy, while level largely can.
"""
from _common import *

name = "cassette"
df = load(name)
y, g = label_and_group(df, name)
dyn = F.dyn_cols(df)
level = [c for c in dyn if c.endswith("_mean") and "run_length" not in c] + ["dyn_band_state_entropy"]
level = [c for c in level if c in df.columns]
occupancy = [c for c in dyn if c.endswith("_dominance_fraction") or c.endswith("_mean_run_length")]
groups = {"level": level, "occupancy": occupancy,
          "level and occupancy": level + occupancy, "all": dyn}
proba, rows = {}, []
for tag, cols in groups.items():
    proba[tag] = E.out_of_fold_proba(df, cols, y, g)
    rows.append({"group": tag, "n_features": len(cols), **E.binary_metrics(y, proba[tag])})
    print("  %-22s n=%2d AUC %.3f" % (tag, len(cols), rows[-1]["auc"]), flush=True)
d = E.bootstrap_gain(g, y, proba["all"], proba["level and occupancy"])
rows.append({"group": "all minus level and occupancy", **d})
print("  all minus level and occupancy  %+0.3f [%+0.3f, %+0.3f] p=%.3f"
      % (d["delta_auc"], d["ci_low"], d["ci_high"], d["p"]), flush=True)
save("r14_decomposition.csv", pd.DataFrame(rows))
