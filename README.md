# Within-Epoch Spectral Trajectory Features for the N1/REM Boundary

Code for *Within-Epoch Spectral Trajectory Features for the N1/REM Boundary in
Single-Channel EEG*.

Goes from the two public databases to every table in the paper.

```
features.py    the three feature sets: BP (5), HC (16), DYN (73)
datasets.py    read Sleep-EDF Expanded and UCDDB, produce one feature table per dataset
evaluate.py    cross-validation, metrics and the subject-level bootstrap
run_all.py     runs the analysis and writes every table to results/
figures.py     Fig. 2 and Fig. 3
results/       output, created on first run
round_1_review/ the extra analyses run for the first round of review
```

---

## 1. Get the data

Neither database is redistributed here; both are public.

### Sleep-EDF Expanded (PhysioNet, DOI 10.13026/C2X676)

About 8 GB. The two subsets are used separately.

```bash
wget -r -N -c -np -nH --cut-dirs=3 \
  https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/
wget -r -N -c -np -nH --cut-dirs=3 \
  https://physionet.org/files/sleep-edfx/1.0.0/sleep-telemetry/
```

This gives `sleep-cassette/` and `sleep-telemetry/`, each with `*-PSG.edf` and
matching `*-Hypnogram.edf` files.

### UCDDB, the St Vincent's University Hospital / UCD database (PhysioNet, DOI 10.13026/C26C7D)

About 800 MB.

```bash
wget -r -N -c -np -nH --cut-dirs=3 \
  https://physionet.org/files/ucddb/1.0.0/
```

You need the `ucddb*.rec` recordings and matching `ucddb*_stage.txt` files. The
`.rec` files are EDF with a different suffix; the loader handles this.

### Where to put it

```
data/
  sleep_edf/
    sleep-cassette/
    sleep-telemetry/
  ucddb/
    ucddb002.rec
    ucddb002_stage.txt
    ...
```

To keep the recordings outside the repository, put them anywhere and set

```bash
export N1REM_DATA=/path/to/that/folder     # must contain sleep_edf/ and ucddb/
```

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
python run_all.py binary      # Tables I, IV and V, and the AUC intervals in the text
python run_all.py controls    # Table VI
python run_all.py ucddb       # the UCDDB section
python run_all.py fiveclass   # Table VII
python figures.py             # Fig. 2 and Fig. 3
```

`python run_all.py all` does the five analysis steps in order.

Extraction is the slow step: 94 features per epoch over every recording, a few
hours for the full cassette set on one core. It caches per recording in
`results/features/`, so it can be interrupted and resumed, and the reporting
steps take about a minute each afterwards.

---

## 4. What the analysis does

**Epochs.** Every 30-s epoch is linearly detrended, median-centred and filtered
with a fourth-order zero-phase Butterworth bandpass from 0.3 to 45 Hz. Bands are
delta 0.5-4, theta 4-8, alpha 8-12, sigma 12-16 and beta 16-30 Hz.

**Sub-windows.** For the proposed features a 2-s window slides across the epoch
with a 1-s stride, giving K = 29 windows. Each window gives a Hann-windowed
Welch PSD, a five-dimensional relative band power vector normalised over
0.5-30 Hz, and three window-level descriptors. The 2-s window and 1-s stride
were chosen before any comparison was run, because 2 s is long enough to
estimate the 4-30 Hz bands and short enough to keep within-epoch change.

**Classifier.** One logistic regression throughout: median imputation,
standardisation, L2 at C = 1, lbfgs, class-balanced loss. Imputation and scaling
sit inside the pipeline so they are fit on training folds only. Nothing is
tuned; the classifier is there to compare feature sets.

**Folds.** Five-fold `StratifiedGroupKFold` grouped by subject for Sleep-EDF, so
both nights of a subject stay in the same fold, and by record for UCDDB, where
there is one record per subject.

**Uncertainty.** AUC is computed once from pooled out-of-fold predictions, not
averaged over folds and not per subject. The bootstrap resamples whole subjects
(records for UCDDB) over those fixed predictions, 2000 resamples, no refitting,
so the interval reflects subject sampling rather than training-set sampling.
Two-sided p from the sign of the resampled differences.

**Seed.** `evaluate.SEED` sets the fold assignment and the bootstrap. Changing
it moves the AUCs by up to about 0.005.

---

## 5. Feature definitions

### BP, 5 features

Relative band power of the whole 30-s epoch for delta, theta, alpha, sigma and
beta, each normalised by total power from 0.5 to 30 Hz so the five sum to one.
Column prefix `band_rel_`.

### HC, 16 features

The standard single-spectrum baseline, computed on the whole 30-s epoch. Column
prefix `common_`.

| Column | Definition |
| --- | --- |
| `common_delta_theta_ratio` | delta / theta relative power |
| `common_delta_beta_ratio` | delta / beta |
| `common_theta_beta_ratio` | theta / beta |
| `common_alpha_theta_ratio` | alpha / theta |
| `common_sigma_beta_ratio` | sigma / beta |
| `common_slow_fast_ratio` | (delta + theta) / (alpha + sigma + beta) |
| `common_spectral_entropy` | Shannon entropy of the 0.5-30 Hz PSD, normalised by log of the number of bins |
| `common_hjorth_activity` | variance of the epoch |
| `common_hjorth_mobility` | sqrt(var(dx) / var(x)) |
| `common_hjorth_complexity` | mobility of dx divided by mobility of x |
| `common_zero_crossing_rate` | fraction of adjacent samples where the sign changes |
| `common_spectral_centroid` | power-weighted mean frequency over 0.5-30 Hz |
| `common_spectral_slope` | slope of log PSD against log frequency over 1-30 Hz |
| `common_sample_entropy` | sample entropy, m = 2, r = 0.2 SD, signal decimated to 384 points |
| `common_peak_frequency` | frequency of the largest PSD bin in 0.5-30 Hz |
| `common_spectral_edge_95` | frequency below which 95 per cent of 0.5-30 Hz power lies |

### DYN, 73 features

All of these are computed from the K = 29 sub-windows. Column prefix `dyn_`,
plus one `smti_`.

**The eight summaries.** For a scalar trajectory z(k) over the K windows, write
Dz(k) = z(k) - z(k-1) for the step, I[.] for the indicator, and let
H(k) = I[z(k) >= the 75th percentile of z within this same epoch].

| Suffix | Definition |
| --- | --- |
| `_mean` | mean of z over the K windows |
| `_std` | standard deviation of z |
| `_diff_std` | standard deviation of Dz |
| `_abs_step` | mean of the absolute value of Dz |
| `_crossing_rate` | fraction of adjacent window pairs where I[z(k) > 0] changes value |
| `_high_state_switching` | fraction of adjacent window pairs where H(k) changes value |
| `_dominance_fraction` | fraction of windows with z(k) > 0 |
| `_mean_run_length` | mean length in windows of the maximal runs on which I[z(k) > 0] stays constant, counting runs of both states, unnormalised so it lies in 1 to 29 |

The high-state threshold is the 75th percentile of that trajectory **within the
same epoch**. It is not global and not fixed.

**Band-ratio trajectories, 4 x 8 = 32.** With p_b(k) the relative power of band
b in window k, and eps = 1e-12 added to numerator and denominator before every
logarithm,

- `theta_alpha` = log((p_theta + eps) / (p_alpha + eps))
- `alpha_sigma` = log((p_alpha + eps) / (p_sigma + eps))
- `sigma_beta` = log((p_sigma + eps) / (p_beta + eps))
- `slow_fast` = log((p_delta + p_theta + eps) / (p_alpha + p_sigma + p_beta + eps))

Each gets all eight summaries, giving names such as
`dyn_theta_alpha_dominance_fraction`. An exact zero counts as non-dominant.

**Spectral-path shape, 5.** With p(k) the five-dimensional relative band power
vector of window k, u(k) = p(k+1) - p(k) and v(k) = p(k) - p(k-1),

| Column | Definition |
| --- | --- |
| `dyn_band_trajectory_length` | sum over k of the L2 norm of u(k), the path length L |
| `dyn_band_trajectory_step_mean` | mean of those step norms |
| `dyn_band_trajectory_step_std` | standard deviation of those step norms |
| `dyn_band_trajectory_turning` | mean over interior k of 1 - cosine between u(k) and v(k), the turning T, with the denominator stabilised by eps and the cosine clipped to [-1, 1] |
| `dyn_band_state_entropy` | Shannon entropy of the epoch-mean relative band power vector, normalised by log 5 |

**Band-power trajectories, 5 x 4 = 20.** Each of the five relative band powers
across windows gets `_power_std`, `_power_diff_std`, `_power_abs_step` and
`_high_state_switching`, giving names such as `dyn_theta_power_std`.

**Window-level descriptors, 3 x 5 = 15.** Each window also gives a spectral
slope over 1-30 Hz, a spectral entropy over 0.5-30 Hz and a spectral centroid
over 0.5-30 Hz. Each of these three trajectories gets `_mean`, `_std`,
`_diff_std`, `_abs_step` and `_high_state_switching`, giving names such as
`dyn_slope_diff_std`.

**Composite instability, 1.**

| Column | Definition |
| --- | --- |
| `smti_spectral_microtrajectory_instability` | `dyn_band_trajectory_turning` times `dyn_band_trajectory_step_mean` |

**Note on collinearity.** `dyn_band_trajectory_length` is `dyn_band_trajectory_step_mean`
multiplied by the number of steps, so the two are proportional and carry the same
information. Both are kept because both are named in the paper, but only 72 of
the 73 are linearly independent.

**Note on the count.** The submitted paper reported 77 features. Four of the
five composite descriptors were exact copies of descriptors already in the set,
so the set has **73 unique features**. Dropping the four duplicates changes AUC
by 0.0001. This code defines the 73.

**Numerical guard.** eps = 1e-12 is added to the numerator and denominator
before every logarithm and to the turning denominator, so a band power of zero
in some window cannot produce a division by zero or a log of zero. It is not
tuned, and any value well below the scale of the data gives the same result. On
these data it never comes into play, since the smallest mean trajectory step
across the 47,357 cassette epochs is 5.65e-3.

---

## 6. Which script produces which table

| Paper item | Command | Output |
| --- | --- | --- |
| Table I, dataset summary | `run_all.py binary` | `results/table1_datasets.csv` |
| Table II, feature inventory | section 5 above | |
| Table III, summaries applied to each window series | section 5 above | |
| Table IV, bootstrap AUC gains | `run_all.py binary` | `results/table3_bootstrap_gains.csv` |
| Table V, main N1/REM evaluation | `run_all.py binary` | `results/table4_main_results.csv` |
| AUC intervals quoted in the Results | `run_all.py binary` | `results/table4_main_results.csv` |
| Table VI, size-matched controls | `run_all.py controls` | `results/table5_controls.csv` |
| UCDDB section | `run_all.py ucddb` | `results/ucddb_models.csv`, `results/ucddb_gains.csv` |
| Table VII, five-class evaluation | `run_all.py fiveclass` | `results/table6_five_class.csv` |
| Fig. 1, pipeline schematic | drawn separately for the paper; `figures.py` writes an equivalent version | `results/fig_pipeline.pdf` |
| Fig. 2, example epochs | `figures.py` | `results/fig_real_epoch_feature_examples.pdf` |
| Fig. 3, single-feature separability | `figures.py` | `results/fig2_feature_separability.png` |

The five-class run uses a 160-point cap for sample entropy rather than 384, so
that the full label set stays tractable. Everything else matches the binary run.

Direction-adjusted AUCs in Fig. 2 are descriptive. They are optimistic by
construction and are not out-of-sample estimates.

---

## 7. Citation

Please cite the paper. Sleep-EDF Expanded and UCDDB carry their own citation
requirements, given on their PhysioNet pages.

---

## 8. Reproduction

Checked against the original analysis three ways.

**Features.** Extracting from the raw EDFs, all 94 columns (5 BP, 16 HC, 73 DYN)
match the original per-epoch tables to better than 1e-9 on Sleep-EDF, with
identical epoch selection: 47,357 cassette epochs / 153 recordings / 78 subjects,
12,002 telemetry / 44 / 22, 6,419 UCDDB / 25 records. Five-class wake trimming
gives the same epochs and labels.

**Evaluation.** At the fold seed used for the submission this code reproduces the
published tables exactly.

| | BP | Alpha/theta | HC | DYN | HC+DYN |
| --- | --- | --- | --- | --- | --- |
| Cassette, published | 0.697 | 0.677 | 0.727 | 0.760 | 0.763 |
| Cassette, this code | 0.697 | 0.677 | 0.727 | 0.760 | 0.763 |
| Telemetry, published | 0.763 | 0.783 | 0.794 | 0.823 | 0.834 |
| Telemetry, this code | 0.763 | 0.783 | 0.794 | 0.823 | 0.834 |

Balanced accuracy, macro F1, the bootstrap gains, the controls and the five-class
table all reproduce to the last published digit, and the worked-example figure
gives the same descriptor values (N1: L = 8.51, T = 1.49, 10 switches; REM:
L = 2.80, T = 1.03, 4 switches).

**End to end.** From the raw EDFs at `evaluate.SEED = 0`:

| | HC | DYN | HC+DYN | HC+DYN vs HC |
| --- | --- | --- | --- | --- |
| Cassette | 0.722 | 0.761 | 0.763 | +0.041 [+0.026, +0.057], p < 0.001 |
| Telemetry | 0.793 | 0.819 | 0.831 | +0.037 [+0.016, +0.056], p = 0.001 |
| UCDDB | 0.789 | 0.750 | 0.800 | +0.011 [-0.005, +0.027], p = 0.165 |

The size-matched controls stay at the baseline, at -0.001 to -0.004, while
HC+DYN gains +0.041 and +0.037.

### Two corrections

UCDDB dynamic features were previously computed on the unfiltered epoch: the
earlier script skipped the detrend, median-centring and 0.3-45 Hz bandpass given
in the Methods. This code applies the filter everywhere. Sleep-EDF and the
five-class run were checked and were never affected.

The UCDDB comparison is not stable across fold assignments. With corrected
features it runs from -0.010 to +0.011 depending on the seed and no interval
excludes zero, so the proposed features neither help nor harm the handcrafted
baseline there.
