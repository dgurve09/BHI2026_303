"""Reviewer 3uru, weakness 1. Compare against a plain sub-epoch method.

The baseline is the mean and standard deviation of each of the five relative
band powers across the same sub-windows, which is 10 features. It is the summary
that sub-epoch methods normally use.

This is our own construction, not a reimplementation of any published method.
The feature list of Malaekah and Cvetkovic (EMBC 2013), the closest earlier
study, was not obtainable.
"""
from _common import *

name = "cassette"
df = load(name)
y, g = label_and_group(df, name)
bands = ["delta", "theta", "alpha", "sigma", "beta"]
generic = ([c for c in df.columns if c.startswith("dyn_") and c.endswith("_power_std")]
           + [c for c in df.columns if c.startswith("band_rel_")])
models = {"band powers of the 30-s epoch": F.bp_cols(df),
          "generic sub-epoch statistics, mean and SD per band": generic,
          "proposed set": F.dyn_cols(df)}
proba, rows = {}, []
for tag, cols in models.items():
    proba[tag] = E.out_of_fold_proba(df, cols, y, g)
    rows.append({"feature_set": tag, "n_features": len(cols), **E.binary_metrics(y, proba[tag])})
    print("  %-50s n=%2d AUC %.3f" % (tag, len(cols), rows[-1]["auc"]), flush=True)
for a, b, label in [("proposed set", "generic sub-epoch statistics, mean and SD per band",
                     "proposed minus generic sub-epoch"),
                    ("generic sub-epoch statistics, mean and SD per band",
                     "band powers of the 30-s epoch", "generic sub-epoch minus epoch spectrum")]:
    d = E.bootstrap_gain(g, y, proba[a], proba[b])
    rows.append({"feature_set": label, **d})
    print("  %-50s %+0.3f [%+0.3f, %+0.3f] p=%.3f"
          % (label, d["delta_auc"], d["ci_low"], d["ci_high"], d["p"]), flush=True)
save("r11_subepoch_baseline.csv", pd.DataFrame(rows))
