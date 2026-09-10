"""Reproduce the tables in the paper.

  extract [budget_s]   read the EDFs, write the feature tables (slow, resumable)
  binary               Tables I, IV, V
  controls             Table VI
  ucddb                UCDDB section
  fiveclass            Table VII
  all                  all of the above

Everything caches under results/ and is skipped on a rerun.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import datasets as D
import evaluate as E
import features as F

RESULTS = Path(__file__).resolve().parent / "results"
SUBSETS = ["cassette", "telemetry"]


CACHE = RESULTS / "_cache"


def cached(tag, fn):
    """Memoise an expensive result to results/_cache/. Delete the folder to redo."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / (tag + ".npy")
    if path.exists():
        return np.load(path, allow_pickle=True).item()
    value = fn()
    np.save(path, np.array(value, dtype=object), allow_pickle=True)
    return value


def _save(name, obj):
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(path, index=False)
    else:
        path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    print("wrote %s" % path.name)


def _sets(df):
    """Feature sets as labelled in the paper."""
    return {
        "BP": F.bp_cols(df),
        "Alpha/theta": ["common_alpha_theta_ratio"],
        "HC": F.hc_cols(df),
        "DYN": F.dyn_cols(df),
        "HC+DYN": F.hc_cols(df) + F.dyn_cols(df),
    }


# --------------------------------------------------------------------- 1 extract

def step_extract(budget_secs=None):
    """Extract features. Cached per recording, so it can be interrupted and resumed."""
    import time
    started = time.time()

    def left():
        if budget_secs is None:
            return None
        return max(1.0, budget_secs - (time.time() - started))

    jobs = ([(s, "n1_rem") for s in SUBSETS] + [(None, "ucddb")]
            + [(s, "five_class") for s in SUBSETS])   # primary analysis first
    for subset, task in jobs:
        if budget_secs is not None and time.time() - started > budget_secs:
            print("budget reached, stopping. Run this step again to continue.", flush=True)
            return
        if task == "ucddb":
            D.extract_ucddb(task="n1_rem", budget_secs=left())
        else:
            D.extract_sleep_edf(subset, task=task, budget_secs=left())


# ------------------------------------------------------- 2 binary N1/REM, Tables I, IV, V

def step_binary():
    counts, models, gains = [], [], []
    for subset in SUBSETS:
        df = D.extract_sleep_edf(subset, task="n1_rem")
        y = (df.stage_name == "REM").astype(int).values
        g = df.subject.values
        counts.append({"subset": subset, "records": df.record.nunique(),
                       "subjects": df.subject.nunique(), "epochs": len(df),
                       "N1": int((df.stage_name == "N1").sum()),
                       "REM": int((df.stage_name == "REM").sum()),
                       "channel": df.channel.iloc[0]})
        proba = {}
        for name, cols in _sets(df).items():
            tag = "oof_%s_%s" % (subset, name.replace("/", "_").replace("+", "_"))
            proba[name] = cached(tag, lambda c=cols: {"p": E.out_of_fold_proba(df, c, y, g)})["p"]
            lo, hi = cached(tag + "_ci", lambda n=name: dict(zip(
                ["lo", "hi"], E.bootstrap_auc(g, y, proba[n])))).values()
            models.append({"subset": subset, "feature_set": name, "n_features": len(cols),
                           "ci_low": lo, "ci_high": hi, **E.binary_metrics(y, proba[name])})
            print("  %-12s %-11s n=%3d AUC %.3f [%.3f, %.3f]"
                  % (subset, name, len(cols), models[-1]["auc"], lo, hi), flush=True)
        for name in ["DYN", "HC+DYN"]:
            gain = cached("gain_%s_%s" % (subset, name.replace("+", "_")),
                          lambda n=name: E.bootstrap_gain(g, y, proba[n], proba["HC"]))
            gains.append({"subset": subset, "comparison": "%s vs HC" % name, **gain})
            print("  %-12s %-16s %+0.3f [%+0.3f, %+0.3f] p=%.3f"
                  % (subset, gains[-1]["comparison"], gains[-1]["delta_auc"],
                     gains[-1]["ci_low"], gains[-1]["ci_high"], gains[-1]["p"]), flush=True)
    _save("table1_datasets.csv", pd.DataFrame(counts))
    _save("table4_main_results.csv", pd.DataFrame(models))
    _save("table3_bootstrap_gains.csv", pd.DataFrame(gains))


# ------------------------------------------------------------- 3 controls, Table VI

def step_controls():
    """Size-matched controls: each DYN column permuted across epochs, and Gaussian noise."""
    rows = []
    for subset in SUBSETS:
        df = D.extract_sleep_edf(subset, task="n1_rem")
        y = (df.stage_name == "REM").astype(int).values
        g = df.subject.values
        hc, dyn = F.hc_cols(df), F.dyn_cols(df)
        rng = np.random.default_rng(E.SEED)
        dyn_x = df[dyn].replace([np.inf, -np.inf], np.nan).values
        shuffled = np.column_stack([rng.permutation(dyn_x[:, j]) for j in range(dyn_x.shape[1])])
        noise = rng.standard_normal(size=dyn_x.shape)
        hc_x = df[hc].replace([np.inf, -np.inf], np.nan).values
        variants = {
            "HC": df[hc].copy(),
            "HC + shuffled": pd.DataFrame(np.column_stack([hc_x, shuffled])),
            "HC + noise": pd.DataFrame(np.column_stack([hc_x, noise])),
            "HC+DYN": df[hc + dyn].copy(),
        }
        base = None
        for name, block in variants.items():
            block.columns = [str(c) for c in block.columns]
            tag = "ctrl_%s_%s" % (subset, name.replace(" ", "").replace("+", "_"))
            proba = cached(tag, lambda b=block: {"p": E.out_of_fold_proba(b, list(b.columns), y, g)})["p"]
            m = E.binary_metrics(y, proba)
            if name == "HC":
                base = proba
            gain = cached(tag + "_gain", lambda p=proba: E.bootstrap_gain(g, y, p, base))["delta_auc"] \
                if base is not None else 0.0
            rows.append({"subset": subset, "feature_set": name,
                         "n_features": block.shape[1], "auc": m["auc"],
                         "gain_vs_hc": 0.0 if name == "HC" else gain})
            print("  %-12s %-14s n=%2d AUC %.3f" % (subset, name, block.shape[1], m["auc"]), flush=True)
    _save("table5_controls.csv", pd.DataFrame(rows))


# --------------------------------------------------------------------- 4 UCDDB

def step_ucddb():
    df = D.extract_ucddb(task="n1_rem")
    y = (df.stage_name == "REM").astype(int).values
    g = df.record.values           # one record per subject, so records are the groups
    rows, gains, proba = [], [], {}
    for name, cols in _sets(df).items():
        proba[name] = cached("oof_ucddb_%s" % name.replace("/", "_").replace("+", "_"),
                             lambda c=cols: {"p": E.out_of_fold_proba(df, c, y, g)})["p"]
        rows.append({"feature_set": name, "n_features": len(cols), **E.binary_metrics(y, proba[name])})
        print("  UCDDB %-11s n=%3d AUC %.3f" % (name, len(cols), rows[-1]["auc"]), flush=True)
    for name in ["DYN", "HC+DYN"]:
        gain = cached("gain_ucddb_%s" % name.replace("+", "_"),
                      lambda n=name: E.bootstrap_gain(g, y, proba[n], proba["HC"]))
        gains.append({"comparison": "%s vs HC" % name, **gain})
        print("  UCDDB %-16s %+0.3f [%+0.3f, %+0.3f] p=%.3f"
              % (gains[-1]["comparison"], gains[-1]["delta_auc"], gains[-1]["ci_low"],
                 gains[-1]["ci_high"], gains[-1]["p"]), flush=True)
    _save("ucddb_models.csv", pd.DataFrame(rows))
    _save("ucddb_gains.csv", pd.DataFrame(gains))
    _save("ucddb_counts.json", {"records": int(df.record.nunique()), "epochs": int(len(df)),
                                "N1": int((df.stage_name == "N1").sum()),
                                "REM": int((df.stage_name == "REM").sum())})


# ------------------------------------------------------------ 5 five-class, Table VII

def step_fiveclass():
    """Secondary W/N1/N2/N3/REM run. Sample entropy capped at 160 points here, not 384."""
    rows = []
    for subset in SUBSETS:
        df = D.extract_sleep_edf(subset, task="five_class")
        y = df.stage.values
        g = df.subject.values
        preds = {}
        for name in ["HC", "HC+DYN"]:
            cols = F.hc_cols(df) if name == "HC" else F.hc_cols(df) + F.dyn_cols(df)
            proba = cached("oof5_%s_%s" % (subset, name.replace("+", "_")),
                           lambda c=cols: {"p": E.out_of_fold_proba(df, c, y, g, multiclass=True)})["p"]
            classes = np.array(sorted(np.unique(y)))
            preds[name] = classes[proba.argmax(axis=1)]
            rows.append({"subset": subset, "feature_set": name, "n_features": len(cols),
                         "macro_f1": float(f1_score(y, preds[name], average="macro")),
                         "f1_n1": float(f1_score(y, preds[name], labels=[1], average="macro")),
                         "n1_to_rem": E.n1_to_rem_rate(y, preds[name])})
            print("  %-12s %-7s macro F1 %.3f  F1 N1 %.3f  N1->REM %.1f%%"
                  % (subset, name, rows[-1]["macro_f1"], rows[-1]["f1_n1"],
                     100 * rows[-1]["n1_to_rem"]), flush=True)
        for label, metric in [("macro_f1", lambda a, b: f1_score(a, b, average="macro")),
                              ("f1_n1", lambda a, b: f1_score(a, b, labels=[1], average="macro"))]:
            d = cached("gain5_%s_%s" % (subset, label),
                       lambda m=metric: E.bootstrap_metric_gain(g, y, preds["HC+DYN"], preds["HC"], m))
            print("  %-12s %-8s %+0.3f [%+0.3f, %+0.3f] p=%.3f"
                  % (subset, label, d["delta"], d["ci_low"], d["ci_high"], d["p"]), flush=True)
            rows.append({"subset": subset, "feature_set": "gain %s" % label, "n_features": None,
                         "macro_f1": None, "f1_n1": None, "n1_to_rem": None, **d})
    _save("table6_five_class.csv", pd.DataFrame(rows))


STEPS = {"extract": step_extract, "binary": step_binary, "controls": step_controls,
         "ucddb": step_ucddb, "fiveclass": step_fiveclass}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else None
    for name in (STEPS if which == "all" else [which]):
        print("\n=== %s ===" % name, flush=True)
        STEPS[name](budget) if name == "extract" else STEPS[name]()
