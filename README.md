# Within-Epoch Spectral Trajectory Features for the N1/REM Boundary

Code for *Within-Epoch Spectral Trajectory Features for the N1/REM Boundary in
Single-Channel EEG* (IEEE-EMBS BHI 2026). Goes from the two public databases to
every table in the paper.

```
features.py      the three feature sets: BP (5), HC (16), DYN (73)
datasets.py      read Sleep-EDF Expanded and UCDDB, one feature table per dataset
evaluate.py      cross-validation, metrics and the subject-level bootstrap
run_all.py       runs the analysis and writes every table to results/
figures.py       Fig. 2 and Fig. 3
extra_analyses/  controls and secondary analyses
results/         output, created on first run
```

---

## 1. Data

Neither database is redistributed here; both are public.

**Sleep-EDF Expanded** (DOI 10.13026/C2X676), about 8 GB:

```bash
wget -r -N -c -np -nH --cut-dirs=3 https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/
wget -r -N -c -np -nH --cut-dirs=3 https://physionet.org/files/sleep-edfx/1.0.0/sleep-telemetry/
```

**UCDDB** (DOI 10.13026/C26C7D), about 800 MB. The `.rec` files are EDF with a
different suffix and the loader handles that.

```bash
wget -r -N -c -np -nH --cut-dirs=3 https://physionet.org/files/ucddb/1.0.0/
```

Expected layout:

```
data/
  sleep_edf/sleep-cassette/        *-PSG.edf and *-Hypnogram.edf
  sleep_edf/sleep-telemetry/
  ucddb/                           ucddb002.rec, ucddb002_stage.txt, ...
```

To keep the recordings elsewhere, set `export N1REM_DATA=/path/to/folder`; it
must contain `sleep_edf/` and `ucddb/`.

---

## 2. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.10, numpy 2.2, scipy 1.15, scikit-learn 1.7, pandas 2.3,
mne 1.12.

---

## 3. Run

```bash
python run_all.py extract     # read the EDFs and build the feature tables
python run_all.py binary      # Tables I, IV and V
python run_all.py controls    # Table VI
python run_all.py ucddb       # the UCDDB section
python run_all.py fiveclass   # Table VII
python figures.py             # Fig. 2 and Fig. 3
```

`python run_all.py all` does the five analysis steps in order.

Extraction is the slow step, a few hours for the full cassette set on one core.
It caches per recording in `results/features/`, so it can be interrupted and
resumed, and each reporting step then takes about a minute.

---

## 4. Settings

Epochs are detrended, median-centred and bandpass filtered 0.3-45 Hz (4th-order
zero-phase Butterworth). Bands: delta 0.5-4, theta 4-8, alpha 8-12, sigma 12-16,
beta 16-30 Hz. For DYN a 2-s window slides with a 1-s stride, giving K = 29
windows per epoch.

One logistic regression throughout (median imputation, standardisation, L2 at
C = 1, lbfgs, class-balanced), fit inside the pipeline so it sees training folds
only. Five-fold `StratifiedGroupKFold` grouped by subject for Sleep-EDF and by
record for UCDDB. AUC comes from pooled out-of-fold predictions; the bootstrap
resamples whole subjects over those fixed predictions, 2000 times, without
refitting. `evaluate.SEED` sets the folds and the bootstrap, and changing it
moves the AUCs by up to about 0.005.

---

## 5. Feature definitions

**BP, 5.** Relative band power of the 30-s epoch for the five bands, normalised
over 0.5-30 Hz so they sum to one. Prefix `band_rel_`.

**HC, 16.** The single-spectrum baseline over the whole epoch, prefix
`common_`: five band ratios (`delta_theta`, `delta_beta`, `theta_beta`,
`alpha_theta`, `sigma_beta`), `slow_fast` = (delta+theta)/(alpha+sigma+beta),
`spectral_entropy`, `hjorth_activity`, `hjorth_mobility`, `hjorth_complexity`,
`zero_crossing_rate`, `spectral_centroid`, `spectral_slope`, `sample_entropy`
(m = 2, r = 0.2 SD, decimated to 384 points), `peak_frequency` and
`spectral_edge_95`.

**DYN, 73.** All computed from the K = 29 sub-windows. Prefix `dyn_`, plus one
`smti_`.

For a trajectory z(k) over the K windows, write Dz(k) = z(k) - z(k-1), I[.] for
the indicator, and H(k) = I[z(k) >= the 75th percentile of z within this same
epoch]. That threshold is per-epoch, not global.

| Suffix | Definition |
| --- | --- |
| `_mean` | mean of z over the K windows |
| `_std` | standard deviation of z |
| `_diff_std` | standard deviation of Dz |
| `_abs_step` | mean absolute value of Dz |
| `_crossing_rate` | fraction of adjacent window pairs where I[z(k) > 0] changes |
| `_high_state_switching` | fraction of adjacent window pairs where H(k) changes |
| `_dominance_fraction` | fraction of windows with z(k) > 0 |
| `_mean_run_length` | mean length in windows of the maximal runs on which I[z(k) > 0] stays constant, counting both states, unnormalised so it lies in 1 to 29 |

*Band-ratio trajectories, 4 x 8 = 32.* With p_b(k) the relative power of band b
in window k and eps = 1e-12 added to numerator and denominator before every
logarithm: `theta_alpha`, `alpha_sigma`, `sigma_beta` are log ratios of the two
bands, and `slow_fast` = log((p_delta+p_theta+eps)/(p_alpha+p_sigma+p_beta+eps)).
Each gets all eight summaries, giving names such as
`dyn_theta_alpha_dominance_fraction`. An exact zero counts as non-dominant.

*Spectral-path shape, 5.* With p(k) the five-dimensional relative band power
vector of window k, u(k) = p(k+1) - p(k) and v(k) = p(k) - p(k-1):
`dyn_band_trajectory_length` (sum of the L2 norms of u(k), the path length L),
`dyn_band_trajectory_step_mean` and `dyn_band_trajectory_step_std` (mean and SD
of those step norms), `dyn_band_trajectory_turning` (mean over interior k of
1 - cos between u(k) and v(k), the turning T, denominator stabilised by eps and
the cosine clipped to [-1, 1]), and `dyn_band_state_entropy` (Shannon entropy of
the epoch-mean band power vector, normalised by log 5).

*Band-power trajectories, 5 x 4 = 20.* Each of the five relative band powers
across windows gets `_power_std`, `_power_diff_std`, `_power_abs_step` and
`_high_state_switching`, as in `dyn_theta_power_std`.

*Window-level descriptors, 3 x 5 = 15.* Each window also gives a spectral slope
(1-30 Hz), spectral entropy and spectral centroid (both 0.5-30 Hz). Each gets
`_mean`, `_std`, `_diff_std`, `_abs_step` and `_high_state_switching`.

*Composite, 1.* `smti_spectral_microtrajectory_instability` is
`dyn_band_trajectory_turning` times `dyn_band_trajectory_step_mean`.

---

## 6. Which script produces which table

| Paper item | Command | Output |
| --- | --- | --- |
| Table I, dataset summary | `run_all.py binary` | `results/table1_datasets.csv` |
| Tables II and III, feature inventory and summaries | section 5 above | |
| Table IV, bootstrap AUC gains | `run_all.py binary` | `results/table3_bootstrap_gains.csv` |
| Table V, main N1/REM evaluation | `run_all.py binary` | `results/table4_main_results.csv` |
| Table VI, size-matched controls | `run_all.py controls` | `results/table5_controls.csv` |
| Table VII, five-class evaluation | `run_all.py fiveclass` | `results/table6_five_class.csv` |
| UCDDB section | `run_all.py ucddb` | `results/ucddb_models.csv`, `results/ucddb_gains.csv` |
| Fig. 1, pipeline schematic | drawn separately for the paper; `figures.py` writes an equivalent version | `results/fig_pipeline.pdf` |
| Fig. 2, example epochs | `figures.py` | `results/fig_real_epoch_feature_examples.pdf` |
| Fig. 3, single-feature separability | `figures.py` | `results/fig2_feature_separability.png` |

The five-class run caps sample entropy at 160 points rather than 384 so the full
label set stays tractable; everything else matches the binary run. The
direction-adjusted AUCs in Fig. 3 are descriptive only, since the direction is
chosen on the same data.

---

## 7. Citation

M. K. Gurve, N. V. Kulangareth, D. Chanderwal, S. Rastogi and D. Gurve,
"Within-Epoch Spectral Trajectory Features for the N1/REM Boundary in
Single-Channel EEG," in *Proc. IEEE-EMBS Int. Conf. on Biomedical and Health
Informatics (BHI)*, Hong Kong, Dec. 2026.

Sleep-EDF Expanded and UCDDB carry their own citation requirements, given on
their PhysioNet pages.

---

## 8. Reproduction

From the raw EDFs at `evaluate.SEED = 0`, which is what the paper reports:

| | BP | Alpha/theta | HC | DYN | HC+DYN | HC+DYN vs HC |
| --- | --- | --- | --- | --- | --- | --- |
| Cassette | 0.692 | 0.678 | 0.722 | 0.761 | 0.763 | +0.041 [+0.026, +0.057], p < 0.001 |
| Telemetry | 0.758 | 0.779 | 0.793 | 0.819 | 0.831 | +0.037 [+0.016, +0.056], p = 0.001 |
| UCDDB | 0.546 | 0.611 | 0.789 | 0.750 | 0.800 | +0.011 [-0.005, +0.027], p = 0.165 |

Epoch selection matches the paper: 47,357 cassette epochs / 153 recordings / 78
subjects, 12,002 telemetry / 44 / 22, 6,419 UCDDB / 25 records. Balanced
accuracy, macro F1, the controls and the five-class table reproduce the
published values, and the worked-example figure gives the same descriptors
(N1: L = 8.51, T = 1.49, 10 switches; REM: L = 2.80, T = 1.03, 4 switches).

The UCDDB comparison is not stable across fold assignments. It runs from -0.013
to +0.011 depending on the seed and no interval excludes zero, so the proposed
features neither help nor harm the handcrafted baseline there.
