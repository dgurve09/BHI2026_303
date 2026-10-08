"""Reviewer 3uru, weakness 3. Does the improvement come from boundary epochs?

The comparison is repeated using only epochs whose preceding and following
epochs carry the same stage label, so mixed and transitional epochs are removed.
"""
import sys
from _common import *

# One subset per invocation keeps each run short.
#   python r12_stage_boundary.py            both subsets
#   python r12_stage_boundary.py telemetry  one subset
SUBSETS = sys.argv[1:] or ["cassette", "telemetry"]
rows = []
for name in SUBSETS:
    df = load(name).sort_values(["record", "epoch"]).reset_index(drop=True)
    interior = np.zeros(len(df), dtype=bool)
    for _, idx in df.groupby("record").groups.items():
        i = np.asarray(idx)
        s = df.stage.values[i]
        e = df.epoch.values[i]
        prev_same = np.r_[False, (s[1:] == s[:-1]) & (e[1:] == e[:-1] + 1)]
        next_same = np.r_[(s[:-1] == s[1:]) & (e[:-1] == e[1:] - 1), False]
        interior[i] = prev_same & next_same
    for tag, sub in [("all epochs", df), ("interior only", df[interior].reset_index(drop=True))]:
        y, g = label_and_group(sub, name)
        ph = E.out_of_fold_proba(sub, F.hc_cols(sub), y, g)
        pb = E.out_of_fold_proba(sub, F.hc_cols(sub) + F.dyn_cols(sub), y, g)
        d = E.bootstrap_gain(g, y, pb, ph)
        rows.append({"subset": name, "epochs_used": tag, "n_epochs": len(sub),
                     "hc": E.binary_metrics(y, ph)["auc"], "hc_dyn": E.binary_metrics(y, pb)["auc"], **d})
        print("  %-10s %-14s n=%6d HC %.3f HC+DYN %.3f  delta %+0.3f [%+0.3f, %+0.3f] p=%.3f"
              % (name, tag, len(sub), rows[-1]["hc"], rows[-1]["hc_dyn"],
                 d["delta_auc"], d["ci_low"], d["ci_high"], d["p"]), flush=True)
    print("  %s: %.1f%% of epochs removed" % (name, 100 * (1 - interior.mean())), flush=True)
save("r12_stage_boundary_%s.csv" % "_".join(SUBSETS), pd.DataFrame(rows))
