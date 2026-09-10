"""Feature definitions.

BP  = 5 relative band powers of the 30-s epoch
HC  = 16 standard single-spectrum descriptors
DYN = 73 within-epoch spectral trajectory descriptors
"""
import numpy as np
from scipy import signal, stats
from sklearn.neighbors import KDTree

EPS = 1e-12
STEP_TOL = 1e-9   # a trajectory step shorter than this has no usable direction
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),
    "beta": (16.0, 30.0),
}
BAND_ORDER = list(BANDS)


# ---------------------------------------------------------------- preprocessing

def clean_epoch(x, fs):
    """Linear detrend, median centre, 4th-order zero-phase 0.3-45 Hz bandpass."""
    x = np.asarray(x, dtype=float)
    x = signal.detrend(x, type="linear")
    x = x - np.nanmedian(x)
    hi = min(45.0, fs / 2.0 - 1.0)
    sos = signal.butter(4, [0.3, hi], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


def welch_psd(x, fs, nperseg_secs=4.0):
    """Hann-windowed Welch PSD, floored at EPS."""
    nperseg = min(len(x), max(64, int(nperseg_secs * fs)))
    freqs, psd = signal.welch(x, fs=fs, window="hann", nperseg=nperseg,
                              noverlap=nperseg // 2, detrend=False, scaling="density")
    return freqs, psd + EPS


def band_relative_powers(x, fs, nperseg_secs=4.0):
    """Relative band powers, normalised over 0.5-30 Hz."""
    freqs, psd = welch_psd(x, fs, nperseg_secs=nperseg_secs)
    total_mask = (freqs >= 0.5) & (freqs < 30.0)
    total = float(np.trapezoid(psd[total_mask], freqs[total_mask])) + EPS
    out = {}
    for name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        out["band_rel_" + name] = float(np.trapezoid(psd[mask], freqs[mask])) / total
    return out


def spectral_entropy_from_values(values):
    """Shannon entropy of a non-negative vector, normalised by log(len)."""
    values = np.maximum(np.asarray(values, dtype=float), 0.0) + EPS
    p = values / values.sum()
    return float(-np.sum(p * np.log(p)) / np.log(len(p)))


def sample_entropy(x, m=2, r_scale=0.2, max_points=384):
    """Sample entropy, m=2, r=0.2*SD. Decimated to max_points first."""
    x = np.asarray(x, dtype=float)
    if len(x) > max_points:
        x = x[np.linspace(0, len(x) - 1, max_points).astype(int)]
    x = x - np.nanmean(x)
    sd = float(np.nanstd(x))
    if not np.isfinite(sd) or sd < EPS:
        return 0.0
    r = r_scale * sd

    def count_matches(order):
        emb = np.lib.stride_tricks.sliding_window_view(x, order)
        tree = KDTree(emb, metric="chebyshev")
        return float(np.sum(tree.query_radius(emb, r=r, count_only=True) - 1))

    b, a = count_matches(m), count_matches(m + 1)
    if a <= 0.0 or b <= 0.0:
        return np.nan
    return float(-np.log(a / b))


# ------------------------------------------------------------------ HC, 16 features

def handcrafted_features(x, fs, bands, sampen_max_points=384):
    """The 16 handcrafted baseline descriptors."""
    freqs, psd = welch_psd(x, fs, nperseg_secs=4.0)
    mask = (freqs >= 0.5) & (freqs < 30.0)
    f, p = freqs[mask], psd[mask]
    total = float(np.trapezoid(p, f)) + EPS
    probs = p / (np.sum(p) + EPS)
    dx, ddx = np.diff(x), np.diff(np.diff(x))
    var0 = float(np.nanvar(x)) + EPS
    var1 = float(np.nanvar(dx)) + EPS
    var2 = float(np.nanvar(ddx)) + EPS
    mobility = float(np.sqrt(var1 / var0))
    mobility_dx = float(np.sqrt(var2 / var1))
    slope_mask = (freqs >= 1.0) & (freqs <= 30.0)
    slope = stats.linregress(np.log(freqs[slope_mask]), np.log(psd[slope_mask])).slope
    b = bands
    out = {
        "common_delta_theta_ratio": float(b["band_rel_delta"] / (b["band_rel_theta"] + EPS)),
        "common_delta_beta_ratio": float(b["band_rel_delta"] / (b["band_rel_beta"] + EPS)),
        "common_theta_beta_ratio": float(b["band_rel_theta"] / (b["band_rel_beta"] + EPS)),
        "common_alpha_theta_ratio": float(b["band_rel_alpha"] / (b["band_rel_theta"] + EPS)),
        "common_sigma_beta_ratio": float(b["band_rel_sigma"] / (b["band_rel_beta"] + EPS)),
        "common_slow_fast_ratio": float((b["band_rel_delta"] + b["band_rel_theta"])
                                        / (b["band_rel_alpha"] + b["band_rel_sigma"] + b["band_rel_beta"] + EPS)),
        "common_spectral_entropy": float(-np.sum(probs * np.log(probs + EPS)) / np.log(len(probs))),
        "common_hjorth_activity": var0,
        "common_hjorth_mobility": mobility,
        "common_hjorth_complexity": float(mobility_dx / (mobility + EPS)),
        "common_zero_crossing_rate": float(np.mean(np.diff(np.signbit(x)) != 0)),
        "common_spectral_centroid": float(np.trapezoid(f * p, f) / total),
        "common_spectral_slope": float(slope),
        "common_sample_entropy": sample_entropy(x, max_points=sampen_max_points),
    }
    out["common_peak_frequency"] = float(f[int(np.argmax(p))])
    out["common_spectral_edge_95"] = float(f[np.searchsorted(np.cumsum(p) / (np.sum(p) + EPS), 0.95)])
    return out


# ----------------------------------------------------------------- DYN, 73 features

def _transition_rate(binary):
    """Fraction of adjacent window pairs where the indicator flips."""
    binary = np.asarray(binary, dtype=bool)
    if len(binary) < 2:
        return 0.0
    return float(np.mean(binary[1:] != binary[:-1]))


def _run_lengths(binary):
    """Run lengths in windows, unnormalised."""
    binary = np.asarray(binary, dtype=bool)
    if len(binary) == 0:
        return np.asarray([], dtype=float)
    edges = np.flatnonzero(binary[1:] != binary[:-1]) + 1
    starts = np.r_[0, edges]
    stops = np.r_[edges, len(binary)]
    return stops[starts < len(binary)] - starts[starts < len(binary)]


def subwindow_spectra(x, fs, win_secs=2.0, step_secs=1.0):
    """K x 5 band-power trajectory plus slope, entropy and centroid per window."""
    win, step = int(round(win_secs * fs)), int(round(step_secs * fs))
    band_rows, slopes, entropies, centroids = [], [], [], []
    for start in range(0, len(x) - win + 1, step):
        seg = x[start:start + win]
        bands = band_relative_powers(seg, fs, nperseg_secs=win_secs)
        band_rows.append([bands["band_rel_" + b] for b in BAND_ORDER])
        freqs, psd = welch_psd(seg, fs, nperseg_secs=win_secs)
        mask = (freqs >= 0.5) & (freqs < 30.0)
        f, p = freqs[mask], psd[mask]
        probs = p / (np.sum(p) + EPS)
        entropies.append(float(-np.sum(probs * np.log(probs + EPS)) / np.log(len(probs))))
        centroids.append(float(np.sum(f * p) / (np.sum(p) + EPS)))
        sm = (freqs >= 1.0) & (freqs <= 30.0)
        slopes.append(float(stats.linregress(np.log(freqs[sm]), np.log(psd[sm])).slope))
    return np.asarray(band_rows), np.asarray(slopes), np.asarray(entropies), np.asarray(centroids)


def dynamic_features(x, fs, permute_window_order=None, win_secs=2.0, step_secs=1.0):
    """The 73 trajectory descriptors.

    permute_window_order: numpy Generator, shuffles the windows first (order control).
    win_secs / step_secs: 2 s / 1 s in the paper; exposed for the window ablation.
    """
    bands, slopes, entropies, centroids = subwindow_spectra(x, fs, win_secs, step_secs)
    if permute_window_order is not None:
        order = permute_window_order.permutation(len(bands))
        bands, slopes, entropies, centroids = bands[order], slopes[order], entropies[order], centroids[order]
    out = {}
    idx = {b: i for i, b in enumerate(BAND_ORDER)}

    # 4 band-ratio trajectories x 8 summaries = 32
    ratios = {
        "theta_alpha": np.log((bands[:, idx["theta"]] + EPS) / (bands[:, idx["alpha"]] + EPS)),
        "alpha_sigma": np.log((bands[:, idx["alpha"]] + EPS) / (bands[:, idx["sigma"]] + EPS)),
        "sigma_beta": np.log((bands[:, idx["sigma"]] + EPS) / (bands[:, idx["beta"]] + EPS)),
        "slow_fast": np.log((bands[:, idx["delta"]] + bands[:, idx["theta"]] + EPS)
                            / (bands[:, idx["alpha"]] + bands[:, idx["sigma"]] + bands[:, idx["beta"]] + EPS)),
    }
    for name, series in ratios.items():
        diff = np.diff(series)
        above_zero = series > 0.0            # exact zero = non-dominant
        high = series >= np.nanpercentile(series, 75.0)  # per-epoch threshold
        lengths = _run_lengths(above_zero)
        out["dyn_%s_mean" % name] = float(np.nanmean(series))
        out["dyn_%s_std" % name] = float(np.nanstd(series))
        out["dyn_%s_diff_std" % name] = float(np.nanstd(diff))
        out["dyn_%s_abs_step" % name] = float(np.nanmean(np.abs(diff)))
        out["dyn_%s_crossing_rate" % name] = _transition_rate(above_zero)
        out["dyn_%s_high_state_switching" % name] = _transition_rate(high)
        out["dyn_%s_dominance_fraction" % name] = float(np.nanmean(above_zero))
        out["dyn_%s_mean_run_length" % name] = float(np.nanmean(lengths)) if len(lengths) else 0.0

    # spectral-path shape = 5
    diffs = np.diff(bands, axis=0)
    step_lengths = np.linalg.norm(diffs, axis=1)
    out["dyn_band_trajectory_length"] = float(np.sum(step_lengths))
    out["dyn_band_trajectory_step_mean"] = float(np.mean(step_lengths))
    out["dyn_band_trajectory_step_std"] = float(np.std(step_lengths))
    out["dyn_band_state_entropy"] = spectral_entropy_from_values(bands.mean(axis=0))
    if len(diffs) >= 2:
        # A zero-length step has no direction and would otherwise score 1.000,
        # i.e. the same as an orthogonal turn. Drop those pairs.
        na = np.linalg.norm(diffs[:-1], axis=1)
        nb = np.linalg.norm(diffs[1:], axis=1)
        usable = (na > STEP_TOL) & (nb > STEP_TOL)
        if usable.any():
            cos_turn = np.sum(diffs[:-1][usable] * diffs[1:][usable], axis=1) / (na[usable] * nb[usable])
            out["dyn_band_trajectory_turning"] = float(np.mean(1.0 - np.clip(cos_turn, -1.0, 1.0)))
        else:
            out["dyn_band_trajectory_turning"] = 0.0
    else:
        out["dyn_band_trajectory_turning"] = 0.0

    # 5 band-power trajectories x 4 summaries = 20
    for band in BAND_ORDER:
        series = bands[:, idx[band]]
        diff = np.diff(series)
        high = series >= np.nanpercentile(series, 75.0)
        out["dyn_%s_power_std" % band] = float(np.nanstd(series))
        out["dyn_%s_power_diff_std" % band] = float(np.nanstd(diff))
        out["dyn_%s_power_abs_step" % band] = float(np.nanmean(np.abs(diff)))
        out["dyn_%s_high_state_switching" % band] = _transition_rate(high)

    # 3 window-level descriptors x 5 summaries = 15
    for name, series in [("slope", slopes), ("entropy", entropies), ("centroid", centroids)]:
        diff = np.diff(series)
        high = series >= np.nanpercentile(series, 75.0)
        out["dyn_%s_mean" % name] = float(np.nanmean(series))
        out["dyn_%s_std" % name] = float(np.nanstd(series))
        out["dyn_%s_diff_std" % name] = float(np.nanstd(diff))
        out["dyn_%s_abs_step" % name] = float(np.nanmean(np.abs(diff)))
        out["dyn_%s_high_state_switching" % name] = _transition_rate(high)

    # composite instability = 1
    # (4 of the 5 composites originally listed duplicated existing columns)
    out["smti_spectral_microtrajectory_instability"] = (
        out["dyn_band_trajectory_turning"] * out["dyn_band_trajectory_step_mean"])
    return out


# ------------------------------------------------------------------- column groups

BP_PREFIX = "band_rel_"
HC_PREFIX = "common_"
DYN_PREFIXES = ("dyn_", "smti_")


def bp_cols(df):
    return [c for c in df.columns if c.startswith(BP_PREFIX)]


def hc_cols(df):
    return [c for c in df.columns if c.startswith(HC_PREFIX)]


def dyn_cols(df):
    return [c for c in df.columns if c.startswith(DYN_PREFIXES)]
