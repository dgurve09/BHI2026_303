"""Sleep-EDF Expanded and UCDDB readers. One pass per recording gives BP, HC and DYN.

See README.md for the download links.
"""
import os
import re
import time
import zipfile
from pathlib import Path

import mne
import numpy as np
import pandas as pd

import features as F

EPOCH_SECS = 30
ROOT = Path(__file__).resolve().parent
# Point N1REM_DATA at the folder holding sleep_edf/ and ucddb/ to keep the raw
# recordings outside the repository. Otherwise they are expected in ./data.
DATA_DIR = Path(os.environ.get("N1REM_DATA", ROOT / "data"))
SLEEP_EDF_DIR = DATA_DIR / "sleep_edf"
UCDDB_DIR = DATA_DIR / "ucddb"
CACHE_DIR = ROOT / "results" / "features"
RECORD_DIR = CACHE_DIR / "_records"
WAKE_BUFFER_EPOCHS = 60  # 30 min before and after sleep, for the five-class run

SLEEP_EDF_STAGES = {
    "sleep stage w": 0, "sleep stage 1": 1, "sleep stage 2": 2,
    "sleep stage 3": 3, "sleep stage 4": 3, "sleep stage r": 5,
}
UCDDB_STAGES = {0: 0, 1: 5, 2: 1, 3: 2, 4: 3, 5: 3, 6: -1, 7: -1}  # R&K -> AASM, S3+S4 -> N3
STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 5: "REM"}


# --------------------------------------------------------------------- Sleep-EDF

def _subject_of(record):
    """Sleep-EDF has two nights per subject. SC4001E0 and SC4002E0 -> SC400."""
    return record[:5]


def _list_pairs(subset):
    """Pair each -PSG.edf with its -Hypnogram.edf.

    PhysioNet ships Sleep-EDF Expanded as two folders, sleep-cassette and
    sleep-telemetry. Either the folders themselves or a parent holding both is
    accepted, and a sleep_edf.zip beside them is used if the folders are absent.
    """
    folder = SLEEP_EDF_DIR / subset_folder(subset)
    if not folder.is_dir():
        for parent in SLEEP_EDF_DIR.rglob(subset_folder(subset)):
            if parent.is_dir():
                folder = parent
                break
    if not folder.is_dir():
        zip_path = SLEEP_EDF_DIR / "sleep_edf.zip"
        if zip_path.exists():
            return _list_pairs_zip(subset, zip_path)
        raise FileNotFoundError("No %s folder under %s. See README.md, step 1."
                                % (subset_folder(subset), SLEEP_EDF_DIR))
    psgs = sorted(folder.glob("*-PSG.edf"))
    hyps = {h.name.split("-")[0][:6]: h for h in folder.glob("*-Hypnogram.edf")}
    return [(p, hyps[p.name.split("-")[0][:6]])
            for p in psgs if p.name.split("-")[0][:6] in hyps]


def subset_folder(subset):
    return {"cassette": "sleep-cassette", "telemetry": "sleep-telemetry"}[subset]


def _list_pairs_zip(subset, zip_path):
    """Fallback if the data came as one archive."""
    prefix = subset_folder(subset) + "/"
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    psgs = sorted(n for n in names if n.endswith("-PSG.edf") and prefix in n)
    hyps = {Path(h).name.split("-")[0][:6]: h for h in names if h.endswith("-Hypnogram.edf") and prefix in h}
    out = []
    with zipfile.ZipFile(zip_path) as zf:
        for n in psgs:
            key = Path(n).name.split("-")[0][:6]
            if key in hyps:
                out.append((_unzip(zf, n), _unzip(zf, hyps[key])))
    return out


def _unzip(zf, member):
    out = ROOT / "results" / "_edf" / member
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(member) as src, out.open("wb") as dst:
            dst.write(src.read())
    return out


def _pick_channel(raw, requested):
    normalized = {c.lower().replace(" ", ""): c for c in raw.ch_names}
    key = requested.lower().replace(" ", "")
    if key not in normalized:
        raise ValueError("Channel %r not in %s" % (requested, raw.ch_names))
    return normalized[key]


def _sleep_edf_labels(hyp_path, n_epochs):
    ann = mne.read_annotations(hyp_path)
    labels = np.full(n_epochs, -1, dtype=int)
    for onset, duration, desc in zip(ann.onset, ann.duration, ann.description):
        stage = SLEEP_EDF_STAGES.get(str(desc).strip().lower())
        if stage is None:
            continue
        a = max(0, int(np.floor(float(onset) / EPOCH_SECS)))
        b = min(n_epochs, int(np.ceil((float(onset) + float(duration)) / EPOCH_SECS)))
        labels[a:b] = stage
    return labels


def _trim_wake(labels):
    """Keep sleep plus 30 min of wake on each side. Five-class runs only."""
    asleep = np.flatnonzero(np.isin(labels, [1, 2, 3, 5]))
    keep = np.zeros(len(labels), dtype=bool)
    if len(asleep) == 0:
        return keep
    a = max(0, asleep[0] - WAKE_BUFFER_EPOCHS)
    b = min(len(labels), asleep[-1] + WAKE_BUFFER_EPOCHS + 1)
    keep[a:b] = True
    keep &= labels >= 0
    return keep


def _write_atomic(df, path):
    """Atomic write, so a killed run leaves no truncated file."""
    tmp = path.with_suffix(".csv.part")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _epoch_rows(eeg, fs, labels, wanted, record, subject, ch, sampen_max_points):
    spe = int(round(EPOCH_SECS * fs))
    rows = []
    for e in wanted:
        start, stop = int(e * spe), int(e * spe) + spe
        if stop > len(eeg):
            continue
        x = F.clean_epoch(eeg[start:stop], fs)
        bp = F.band_relative_powers(x, fs)
        row = {"record": record, "subject": subject, "epoch": int(e),
               "stage": int(labels[e]), "stage_name": STAGE_NAMES[int(labels[e])],
               "channel": ch, "fs": fs}
        row.update(bp)
        row.update(F.handcrafted_features(x, fs, bp, sampen_max_points))
        row.update(F.dynamic_features(x, fs))
        rows.append(row)
    return rows


def extract_sleep_edf(subset, task="n1_rem", channel="EEG Fpz-Cz",
                      max_records=None, force=False, sampen_max_points=None,
                      budget_secs=None):
    """Feature table for one Sleep-EDF subset.

    subset       "cassette" or "telemetry"
    task         "n1_rem" keeps N1 and REM only, "five_class" keeps W/N1/N2/N3/REM
    budget_secs  stop after this many seconds and return None. Each recording is
                 cached on its own, so calling again picks up where it stopped.
    """
    if sampen_max_points is None:
        sampen_max_points = 160 if task == "five_class" else 384
    chan_tag = re.sub(r"[^A-Za-z0-9]+", "_", channel).strip("_").lower()
    tag = "sleep_edf_%s_%s_%s" % (task, subset, chan_tag)
    out_path = CACHE_DIR / (tag + ".csv")
    if out_path.exists() and not force:
        return pd.read_csv(out_path)

    per_record = RECORD_DIR / tag
    per_record.mkdir(parents=True, exist_ok=True)
    pairs = _list_pairs(subset)[:max_records]
    keep_stages = [1, 5] if task == "n1_rem" else [0, 1, 2, 3, 5]
    started = time.time()
    for i, (psg, hyp) in enumerate(pairs, 1):
        record = Path(psg).name.split("-")[0]
        cache = per_record / (record + ".csv")
        if cache.exists() and not force:
            continue
        if budget_secs is not None and time.time() - started > budget_secs:
            print("budget reached, %d of %d recordings done" % (i - 1, len(pairs)), flush=True)
            return None
        print("[%3d/%3d] %s" % (i, len(pairs), record), flush=True)
        raw = mne.io.read_raw_edf(psg, preload=False, verbose="ERROR")
        fs = float(raw.info["sfreq"])
        ch = _pick_channel(raw, channel)
        eeg = raw.get_data(picks=[ch])[0]
        n_epochs = len(eeg) // int(round(EPOCH_SECS * fs))
        labels = _sleep_edf_labels(hyp, n_epochs)
        wanted = (np.flatnonzero(_trim_wake(labels)) if task == "five_class"
                  else np.flatnonzero(np.isin(labels, keep_stages)))
        rows = _epoch_rows(eeg, fs, labels, wanted, record, _subject_of(record), ch, sampen_max_points)
        _write_atomic(pd.DataFrame(rows), cache)

    done = sorted(per_record.glob("*.csv"))
    if len(done) < len(pairs):
        print("%d of %d recordings done" % (len(done), len(pairs)), flush=True)
        return None
    df = pd.concat([pd.read_csv(f) for f in done], ignore_index=True)
    df.to_csv(out_path, index=False)
    print("wrote %s  (%d epochs)" % (out_path.name, len(df)), flush=True)
    return df


# ------------------------------------------------------------------------- UCDDB

def _ucddb_labels(stage_path):
    """One integer per line. The last number on the line is the R&K stage."""
    raw = []
    for line in stage_path.read_text(errors="ignore").splitlines():
        nums = re.findall(r"[-+]?\d+", line)
        if nums:
            raw.append(int(nums[-1]))
    return np.asarray([UCDDB_STAGES.get(v, -1) for v in raw], dtype=int)


def _as_edf(path):
    """MNE needs a .edf suffix. UCDDB ships EDF data in .rec files."""
    if path.suffix.lower() == ".edf":
        return path
    link_dir = ROOT / "results" / "_edf_links"
    link_dir.mkdir(parents=True, exist_ok=True)
    link = link_dir / (path.stem + ".edf")
    if link.exists() and link.stat().st_size == path.stat().st_size:
        return link
    if link.exists():
        link.unlink()
    try:
        import os
        os.link(path, link)
    except OSError:
        import shutil
        shutil.copy2(path, link)
    return link


def _ucddb_channel(raw):
    """C3-A2 where present, otherwise C4-A1. All 25 records have C3-A2."""
    lower = {c.lower().replace(" ", ""): c for c in raw.ch_names}
    for name in ["C3-A2", "C4-A1", "C3A2", "C4A1", "EEG C3-A2", "EEG C4-A1"]:
        key = name.lower().replace(" ", "")
        if key in lower:
            return lower[key]
    eeg_like = [c for c in raw.ch_names if "eeg" in c.lower() or re.search(r"c[34].*a[12]", c.lower())]
    if not eeg_like:
        raise ValueError("No EEG-like channel in %s" % raw.ch_names)
    return eeg_like[0]


def extract_ucddb(task="n1_rem", force=False, budget_secs=None):
    """Feature table for UCDDB. One record per subject, so records are the groups."""
    out_path = CACHE_DIR / ("ucddb_%s.csv" % task)
    if out_path.exists() and not force:
        return pd.read_csv(out_path)
    if not UCDDB_DIR.exists():
        raise FileNotFoundError("Missing %s. See README.md, step 1." % UCDDB_DIR)

    per_record = RECORD_DIR / ("ucddb_%s" % task)
    per_record.mkdir(parents=True, exist_ok=True)
    keep_stages = [1, 5] if task == "n1_rem" else [0, 1, 2, 3, 5]
    recs = [r for r in sorted(UCDDB_DIR.glob("ucddb*.rec"))
            if (UCDDB_DIR / (r.stem + "_stage.txt")).exists()]
    started = time.time()
    for i, rec in enumerate(recs, 1):
        cache = per_record / (rec.stem + ".csv")
        if cache.exists() and not force:
            continue
        if budget_secs is not None and time.time() - started > budget_secs:
            print("budget reached, %d of %d records done" % (i - 1, len(recs)), flush=True)
            return None
        print("[%2d/%2d] %s" % (i, len(recs), rec.stem), flush=True)
        labels = _ucddb_labels(UCDDB_DIR / (rec.stem + "_stage.txt"))
        raw = mne.io.read_raw_edf(_as_edf(rec), preload=False, verbose="ERROR")
        fs = float(raw.info["sfreq"])
        ch = _ucddb_channel(raw)
        eeg = raw.get_data(picks=[ch])[0]
        n_epochs = min(len(labels), len(eeg) // int(EPOCH_SECS * fs))
        wanted = [e for e in range(n_epochs) if int(labels[e]) in keep_stages]
        rows = _epoch_rows(eeg, fs, labels, wanted, rec.stem, rec.stem, ch, 384)
        _write_atomic(pd.DataFrame(rows), cache)

    done = sorted(per_record.glob("*.csv"))
    if len(done) < len(recs):
        print("%d of %d records done" % (len(done), len(recs)), flush=True)
        return None
    df = pd.concat([pd.read_csv(f) for f in done], ignore_index=True)
    df.to_csv(out_path, index=False)
    print("wrote %s  (%d epochs)" % (out_path.name, len(df)), flush=True)
    return df
