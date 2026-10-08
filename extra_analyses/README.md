# Controls and secondary analyses

Additional controls and secondary analyses. Not needed to reproduce the main
tables, but kept here so every number reported alongside the paper can be checked.

Each script imports `features.py` and `evaluate.py` from the parent package and
reads the tables written by `run_all.py extract`, so the settings match the main
analysis.

```bash
cd ..            # the package root
python run_all.py extract        # if you have not already
cd extra_analyses
python r14_decomposition.py      # any script, results land in extra_analyses/results/
```

| Script | Raised by | Question |
| --- | --- | --- |
| `r01_nested_selection.py` | bCwc M2 | Was the feature subset chosen on the same folds it was scored on? |
| `r02_gradient_boosting.py` | bCwc M3, RtY2 M4 | Does the ordering of the feature sets survive a nonlinear classifier? |
| `r03_ucddb_null.py` | bCwc M4 | Why do the features add nothing on UCDDB? Includes the fold-seed spread and the apnea-severity stratification. |
| `r04_window_ablation.py` | bCwc m1, qwPP | Is 2 s with a 1-s stride better than 4 s with a 2-s stride? |
| `r05_prevalence_threshold.py` | bCwc Q4 | How does the comparison look at a prevalence-matched operating point? |
| `r06_holm_and_redundancy.py` | bCwc m5, m6 | Holm correction, and how redundant the 73 features are. |
| `r07_window_order_permutation.py` | qwPP, 3uru W2 | Does the order of the sub-windows carry information? |
| `r08_pz_oz_montage.py` | qwPP, 3uru W4 | Is the improvement ocular? Recomputed on a posterior derivation. |
| `r09_five_class_ci.py` | qwPP | Uncertainty for the five-class evaluation. |
| `r10_transfer.py` | RtY2 M3 | Train on Sleep-EDF, test on UCDDB, with no retraining. |
| `r11_subepoch_baseline.py` | 3uru W1 | Comparison against a generic sub-epoch summary. |
| `r12_stage_boundary.py` | 3uru W3 | Does the improvement come from stage-boundary epochs? |
| `r13_effect_sizes.py` | afFY | Evidence that does not depend on a classifier. |
| `r14_decomposition.py` | afFY | Which part of the representation carries the effect? |

Three scripts re-extract features and are therefore slow. `r04` and `r07`
recompute the proposed features under a different setting, and `r08` recomputes
both feature sets on Pz-Oz. Each caches per recording and can be interrupted, and
each takes a record count as an optional argument for a quick check:

```bash
python r07_window_order_permutation.py 5    # 5 recordings, about a minute
python r07_window_order_permutation.py      # the full cassette set, hours
```

## Two corrections

The published UCDDB dynamic features were computed on the unfiltered epoch. The
earlier extraction script sliced the raw signal and never applied the detrend,
median-centring and 0.3-45 Hz bandpass given in the Methods; the stored values
match the unfiltered epoch to 9e-16. Every path here applies the filter.
Sleep-EDF and the five-class run were checked and are unaffected.

The UCDDB result is not stable across fold assignments. With corrected features,
adding the proposed set to the handcrafted baseline gives -0.010 at the seed used
for the submission and +0.011 at the seed used for the revision, and neither
interval excludes zero. What is supportable is that there is no reliable
difference in either direction, not that there is a decrease.
`r03_ucddb_null.py` prints the spread across 15 seeds.
