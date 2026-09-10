"""Shared imports and helpers for the review analyses. Uses the parent package's
feature and evaluation code and the tables written by run_all.py extract."""
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

import numpy as np          # noqa: E402
import pandas as pd         # noqa: E402

import datasets as D        # noqa: E402
import evaluate as E        # noqa: E402
import features as F        # noqa: E402

OUT = Path(__file__).resolve().parent / "results"


def load(name):
    """N1/REM feature table for cassette, telemetry or ucddb."""
    if name == "ucddb":
        return D.extract_ucddb(task="n1_rem")
    return D.extract_sleep_edf(name, task="n1_rem")


def label_and_group(df, name):
    """REM = positive class. Group by subject, or record for UCDDB."""
    y = (df.stage_name == "REM").astype(int).values
    g = (df.record if name == "ucddb" else df.subject).values
    return y, g


def oof(df, cols, y, g, tag, multiclass=False):
    """Out-of-fold predictions, cached under results/_oof/."""
    d = OUT / "_oof"
    d.mkdir(parents=True, exist_ok=True)
    f = d / (tag + ".npy")
    if f.exists():
        return np.load(f)
    p = E.out_of_fold_proba(df, cols, y, g, multiclass=multiclass)
    np.save(f, p)
    return p


def cached(tag, fn):
    """Memoise to results/_cache/ (the bootstraps are the slow part)."""
    d = OUT / "_cache"
    d.mkdir(parents=True, exist_ok=True)
    f = d / (tag + ".npy")
    if f.exists():
        return np.load(f, allow_pickle=True).item()
    v = fn()
    np.save(f, np.array(v, dtype=object), allow_pickle=True)
    return v


def save(name, df):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False)
    print("wrote %s" % name)


def show(rows, cols=None):
    df = pd.DataFrame(rows)
    print(df[cols].to_string(index=False) if cols else df.to_string(index=False), flush=True)
    return df


def extract_variant(subset, tag, dyn_kwargs, channel="EEG Fpz-Cz", max_records=None):
    """Recompute DYN under a variant setting, per record. HC is reused from the main table."""
    import mne
    if max_records is not None:            # keep a smoke-test run from being
        tag = "%s_first%d" % (tag, max_records)   # mistaken for the full set
    cache_dir = OUT / "_variants" / ("%s_%s" % (tag, subset))
    cache_dir.mkdir(parents=True, exist_ok=True)
    combined = OUT / "_variants" / ("%s_%s.csv" % (tag, subset))
    if combined.exists():
        return pd.read_csv(combined)
    pairs = D._list_pairs(subset)[:max_records]
    for i, (psg, hyp) in enumerate(pairs, 1):
        record = Path(psg).name.split("-")[0]
        out = cache_dir / (record + ".csv")
        if out.exists():
            continue
        print("[%3d/%3d] %s" % (i, len(pairs), record), flush=True)
        raw = mne.io.read_raw_edf(psg, preload=False, verbose="ERROR")
        fs = float(raw.info["sfreq"])
        ch = D._pick_channel(raw, channel)
        eeg = raw.get_data(picks=[ch])[0]
        spe = int(round(D.EPOCH_SECS * fs))
        labels = D._sleep_edf_labels(hyp, len(eeg) // spe)
        rows = []
        for e in np.flatnonzero(np.isin(labels, [1, 5])):
            if (e + 1) * spe > len(eeg):
                continue
            x = F.clean_epoch(eeg[e * spe:(e + 1) * spe], fs)
            row = {"record": record, "subject": D._subject_of(record), "epoch": int(e),
                   "stage_name": D.STAGE_NAMES[int(labels[e])]}
            kw = dict(dyn_kwargs)
            if kw.get("permute_window_order") == "per_epoch":
                kw["permute_window_order"] = np.random.default_rng(E.SEED + int(e))
            row.update(F.dynamic_features(x, fs, **kw))
            rows.append(row)
        D._write_atomic(pd.DataFrame(rows), out)
    files = sorted(cache_dir.glob("*.csv"))
    if len(files) < len(pairs):
        print("%d of %d recordings done, run again to continue" % (len(files), len(pairs)), flush=True)
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    D._write_atomic(df, combined)
    return df


def order_free(cols):
    """Descriptors invariant to reordering the sub-windows: means, SDs, dominance
    fractions, band-state entropy. Steps, crossings, switching, runs and path
    shape all depend on order."""
    keep = []
    for c in cols:
        if c.endswith("_diff_std") or "trajectory" in c:   # built from steps
            continue
        if c.endswith(("_mean", "_std", "_dominance_fraction")) or c == "dyn_band_state_entropy":
            keep.append(c)
    return keep
