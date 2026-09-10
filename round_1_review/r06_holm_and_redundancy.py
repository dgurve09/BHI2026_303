"""Reviewer bCwc, minor concerns 5 and 6.

Two independent checks. Holm correction across the four primary comparisons,
and how redundant the 73 proposed features are with each other.

  python r06_holm_and_redundancy.py            both
  python r06_holm_and_redundancy.py holm       just the correction
  python r06_holm_and_redundancy.py redundancy just the redundancy
"""
import re
import sys

from _common import *


def holm():
    tests = []
    for name in ["cassette", "telemetry"]:
        df = load(name)
        y, g = label_and_group(df, name)
        hc = oof(df, F.hc_cols(df), y, g, "r06_%s_hc" % name)
        for tag, cols in [("DYN vs HC", F.dyn_cols(df)),
                          ("HC+DYN vs HC", F.hc_cols(df) + F.dyn_cols(df))]:
            p = oof(df, cols, y, g, "r06_%s_%s" % (name, re.sub(r"[^a-z]+", "_", tag.lower())))
            tag_id = "r06b_%s_%s" % (name, re.sub(r"[^a-z]+", "_", tag.lower()))
            tests.append({"subset": name, "comparison": tag,
                          **cached(tag_id, lambda: E.bootstrap_gain(g, y, p, hc))})
            print("  %-10s %-13s %+0.3f [%+0.3f, %+0.3f] p=%.3f"
                  % (name, tag, tests[-1]["delta_auc"], tests[-1]["ci_low"],
                     tests[-1]["ci_high"], tests[-1]["p"]), flush=True)
    t = pd.DataFrame(tests).sort_values("p").reset_index(drop=True)
    m = len(t)
    t["holm_p"] = np.maximum.accumulate([(m - i) * p for i, p in enumerate(t.p)]).clip(max=1.0)
    print(t[["subset", "comparison", "delta_auc", "p", "holm_p"]].to_string(index=False))
    save("r06_holm.csv", t)


def redundancy():
    rows = []
    for name in ["cassette", "telemetry"]:
        df = load(name)
        dyn = F.dyn_cols(df)
        x = df[dyn].replace([np.inf, -np.inf], np.nan)
        x = x.fillna(x.median())
        z = ((x - x.mean()) / (x.std() + F.EPS)).values
        c = np.abs(np.corrcoef(z.T))
        upper = c[np.triu_indices_from(c, k=1)]
        ev = np.linalg.svd(z, compute_uv=False) ** 2
        frac = np.cumsum(ev) / ev.sum()
        dup = sum(1 for i in range(len(dyn)) for j in range(i + 1, len(dyn))
                  if c[i, j] > 1 - 1e-12)
        rows.append({"subset": name, "n_features": len(dyn),
                     "mean_abs_correlation": float(upper.mean()),
                     "pairs_above_0.9": int((upper > 0.9).sum()),
                     "exact_duplicate_pairs": dup,
                     "components_for_95pct_variance": int(np.searchsorted(frac, 0.95) + 1)})
        print("  %-10s mean |r| %.2f | pairs >0.9: %d | duplicates: %d | components for 95%%: %d"
              % (name, rows[-1]["mean_abs_correlation"], rows[-1]["pairs_above_0.9"],
                 dup, rows[-1]["components_for_95pct_variance"]), flush=True)
    save("r06_redundancy.csv", pd.DataFrame(rows))


part = sys.argv[1] if len(sys.argv) > 1 else "both"
if part in ("both", "holm"):
    holm()
if part in ("both", "redundancy"):
    redundancy()
