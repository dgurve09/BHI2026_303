"""Reviewer qwPP (originality) and Reviewer 3uru (weakness 2).

Does the ORDER of the sub-windows carry information, or only their contents?
The 29 sub-windows of each epoch are permuted before the descriptors are
computed. That keeps the spectral content of the epoch and removes only the
order. The features are also split into those that change under a permutation
and those that cannot.

  python r07_window_order_permutation.py          full cassette set
  python r07_window_order_permutation.py 12       a 12-recording smoke test
"""
import re
import sys
from _common import *

MAXREC = int(sys.argv[1]) if len(sys.argv) > 1 else None
name = "cassette"
base = load(name)
perm = extract_variant(name, "permuted", {"permute_window_order": "per_epoch"}, max_records=MAXREC)
if perm is None:
    raise SystemExit("extraction not finished, run this script again")

keys = ["record", "epoch"]
dyn = F.dyn_cols(base)
df = base[keys + ["subject", "stage_name"] + dyn].merge(
    perm[keys + dyn], on=keys, suffixes=("", "_perm"))
y, g = label_and_group(df, name)
free = order_free(dyn)
dependent = [c for c in dyn if c not in free]
changed = [c for c in dyn if not np.allclose(df[c], df[c + "_perm"], equal_nan=True)]
print("%d of %d features change when the windows are permuted" % (len(changed), len(dyn)), flush=True)
# the split is a claim about the definitions, so check it against the data
assert not [c for c in free if c in changed], "a descriptor called order-free changed"
assert all(c in changed for c in dependent), "a descriptor called order-dependent did not change"
print("split verified: %d order-free, %d order-dependent" % (len(free), len(dependent)), flush=True)

models = {
    "correct window order": dyn,
    "windows permuted within epoch": [c + "_perm" for c in dyn],
    "order-free descriptors only": free,
    "order-dependent only": dependent,
    "order-dependent, permuted": [c + "_perm" for c in dependent],
}
proba, rows = {}, []
for tag, cols in models.items():
    proba[tag] = oof(df, cols, y, g, "r07_" + re.sub(r"[^a-z]+", "_", tag.lower()))
    rows.append({"feature_set": tag, "n_features": len(cols), **E.binary_metrics(y, proba[tag])})
    print("  %-32s n=%2d AUC %.3f" % (tag, len(cols), rows[-1]["auc"]), flush=True)
for a, b, label in [("correct window order", "windows permuted within epoch", "order effect"),
                    ("correct window order", "order-free descriptors only", "cost of dropping order-dependent")]:
    d = E.bootstrap_gain(g, y, proba[a], proba[b])
    rows.append({"feature_set": label, **d})
    print("  %-32s %+0.3f [%+0.3f, %+0.3f] p=%.3f"
          % (label, d["delta_auc"], d["ci_low"], d["ci_high"], d["p"]), flush=True)
save("r07_window_order_permutation.csv", pd.DataFrame(rows))
