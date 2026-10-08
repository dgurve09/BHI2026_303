"""Reviewer qwPP (significance) and Reviewer 3uru (weakness 4).

Is the improvement carried by ocular activity picked up in the frontal channel?
Both feature sets are recomputed on the parietal-occipital derivation, which is
far from the eyes, using the same functions, folds and epochs. If the effect
were ocular it should shrink or disappear on Pz-Oz.

  python r08_pz_oz_montage.py          full cassette set
  python r08_pz_oz_montage.py 12       a 12-recording smoke test
"""
import re
import sys
from _common import *

MAXREC = int(sys.argv[1]) if len(sys.argv) > 1 else None
rows = []
for channel in ["EEG Fpz-Cz", "EEG Pz-Oz"]:
    chan_tag = re.sub(r"[^a-z0-9]+", "_", channel.lower()).strip("_")
    df = D.extract_sleep_edf("cassette", task="n1_rem", channel=channel, max_records=MAXREC)
    if df is None:
        raise SystemExit("extraction not finished for %s, run this script again" % channel)
    y, g = label_and_group(df, "cassette")
    proba = {}
    for tag, cols in [("HC", F.hc_cols(df)), ("DYN", F.dyn_cols(df)),
                      ("HC+DYN", F.hc_cols(df) + F.dyn_cols(df))]:
        proba[tag] = oof(df, cols, y, g, "r08_%s_%s" % (chan_tag, tag.replace("+", "_")))
        rows.append({"montage": channel, "feature_set": tag, "auc": E.binary_metrics(y, proba[tag])["auc"]})
        print("  %-12s %-7s AUC %.3f" % (channel, tag, rows[-1]["auc"]), flush=True)
    d = E.bootstrap_gain(g, y, proba["HC+DYN"], proba["HC"])
    rows.append({"montage": channel, "feature_set": "HC+DYN vs HC", **d})
    print("  %-12s delta %+0.3f [%+0.3f, %+0.3f] p=%.3f"
          % (channel, d["delta_auc"], d["ci_low"], d["ci_high"], d["p"]), flush=True)
save("r08_pz_oz_montage.csv", pd.DataFrame(rows))
