"""Reviewer bCwc minor concern 1 and Reviewer qwPP (limitations).

Is the 2-s window with a 1-s stride the right choice? The proposed features are
recomputed with a 4-s window and a 2-s stride, giving 14 sub-windows per epoch
instead of 29, using the same feature definitions and the same folds.

The 2-s setting was chosen before any of these comparisons were run, so this is
a check on that choice rather than the reason for it.

  python r04_window_ablation.py          full cassette set
  python r04_window_ablation.py 12       a 12-recording smoke test
"""
import re
import sys
from _common import *

MAXREC = int(sys.argv[1]) if len(sys.argv) > 1 else None
name = "cassette"
base = load(name)
alt = extract_variant(name, "win4s", {"win_secs": 4.0, "step_secs": 2.0}, max_records=MAXREC)
if alt is None:
    raise SystemExit("extraction not finished, run this script again")

keys = ["record", "epoch"]
dyn = F.dyn_cols(base)
df = base[keys + ["subject", "stage_name"] + dyn].merge(alt[keys + dyn], on=keys, suffixes=("", "_4s"))
y, g = label_and_group(df, name)
models = {"2-s window, 1-s stride": dyn,
          "4-s window, 2-s stride": [c + "_4s" for c in dyn],
          "both settings together": dyn + [c + "_4s" for c in dyn]}
proba, rows = {}, []
for tag, cols in models.items():
    proba[tag] = oof(df, cols, y, g, "r04_" + re.sub(r"[^a-z0-9]+", "_", tag.lower()))
    rows.append({"setting": tag, "n_features": len(cols), **E.binary_metrics(y, proba[tag])})
    print("  %-26s n=%3d AUC %.3f" % (tag, len(cols), rows[-1]["auc"]), flush=True)
d = E.bootstrap_gain(g, y, proba["2-s window, 1-s stride"], proba["4-s window, 2-s stride"])
rows.append({"setting": "2-s minus 4-s", **d})
print("  2-s minus 4-s  %+0.3f [%+0.3f, %+0.3f] p=%.3f"
      % (d["delta_auc"], d["ci_low"], d["ci_high"], d["p"]), flush=True)
save("r04_window_ablation.csv", pd.DataFrame(rows))
