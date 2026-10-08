"""Reviewer bCwc, major concern 4. Why do the features add nothing on UCDDB?

Three explanations were suggested: the within-epoch variability is washed out,
the features are redundant with the handcrafted set, or they are not separable
in this cohort. Each is tested. The result is then checked against the fold
assignment, and stratified by apnea severity.
"""
from _common import *
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

uc, ca = load("ucddb"), load("cassette")
y_u, g_u = label_and_group(uc, "ucddb")
dyn = F.dyn_cols(uc)

# 1. is within-epoch variability smaller in UCDDB?
rows = []
for c in dyn:
    for stage in ["N1", "REM"]:
        a = uc.loc[uc.stage_name == stage, c].std()
        b = ca.loc[ca.stage_name == stage, c].std()
        rows.append({"feature": c, "stage": stage, "ucddb_over_cassette": a / (b + F.EPS)})
v = pd.DataFrame(rows)
print("median UCDDB/cassette SD ratio: N1 %.2f  REM %.2f"
      % (v[v.stage == "N1"].ucddb_over_cassette.median(),
         v[v.stage == "REM"].ucddb_over_cassette.median()), flush=True)

# 2. are they redundant with the handcrafted set?
red = []
for name, df in [("ucddb", uc), ("cassette", ca)]:
    hcx = df[F.hc_cols(df)].replace([np.inf, -np.inf], np.nan)
    hcx = hcx.fillna(hcx.median())
    r2 = []
    for c in dyn:
        t = df[c].replace([np.inf, -np.inf], np.nan).fillna(df[c].median())
        r2.append(LinearRegression().fit(hcx, t).score(hcx, t))
    red.append({"dataset": name, "median_r2_of_dyn_on_hc": float(np.median(r2))})
show(red)

# 3. are they separable within a record?
per_rec = []
for rec, sub in uc.groupby("record"):
    yy = (sub.stage_name == "REM").astype(int).values
    if len(np.unique(yy)) < 2:
        continue
    per_rec.append(np.median([E.directional_auc(yy, sub[c].values) for c in dyn]))
print("median within-record single-feature AUC: %.3f" % np.median(per_rec), flush=True)
pooled_u = [E.directional_auc(y_u, uc[c].fillna(uc[c].median()).values) for c in dyn]
y_c = (ca.stage_name == "REM").astype(int).values
pooled_c = [E.directional_auc(y_c, ca[c].fillna(ca[c].median()).values) for c in dyn]
print("features with pooled AUC >= 0.60: UCDDB %d of %d, cassette %d of %d"
      % (sum(a >= 0.60 for a in pooled_u), len(dyn), sum(a >= 0.60 for a in pooled_c), len(dyn)), flush=True)

# 4. how much does the fold assignment matter?
seeds = []
for s in range(15):
    E.SEED = s
    ph = E.out_of_fold_proba(uc, F.hc_cols(uc), y_u, g_u)
    pb = E.out_of_fold_proba(uc, F.hc_cols(uc) + dyn, y_u, g_u)
    seeds.append(roc_auc_score(y_u, pb) - roc_auc_score(y_u, ph))
    print("  seed %2d delta %+0.3f" % (s, seeds[-1]), flush=True)
E.SEED = 0
print("across 15 fold seeds: mean %+0.3f  SD %.3f  range [%+0.3f, %+0.3f]"
      % (np.mean(seeds), np.std(seeds), min(seeds), max(seeds)), flush=True)
save("r03_ucddb_fold_seeds.csv", pd.DataFrame({"seed": range(15), "delta_auc": seeds}))

# 5. apnea severity
details = D.UCDDB_DIR / "SubjectDetails.xls"
if details.exists():
    try:                       # UCDDB ships this as a real .xls workbook
        sd = pd.read_excel(details)
    except Exception as exc:
        print("could not read %s (%s), skipping apnea severity" % (details.name, exc), flush=True)
        sd = None
    if sd is None:
        raise SystemExit(0)
    sd.columns = [str(c).strip().lower() for c in sd.columns]
    idc = next(c for c in sd.columns if "study" in c or "number" in c or c == "id")
    ahic = next(c for c in sd.columns if "ahi" in c)
    sd["record"] = sd[idc].astype(str).str.strip().str.lower()
    ahi = dict(zip(sd.record, pd.to_numeric(sd[ahic], errors="coerce")))
    ph = E.out_of_fold_proba(uc, F.hc_cols(uc), y_u, g_u)
    pb = E.out_of_fold_proba(uc, F.hc_cols(uc) + dyn, y_u, g_u)
    rec_rows = []
    for rec, sub in uc.groupby("record"):
        i = uc.record.values == rec
        yy = y_u[i]
        if len(np.unique(yy)) < 2:
            continue
        rec_rows.append({"record": rec, "ahi": ahi.get(rec, np.nan),
                         "hc": roc_auc_score(yy, ph[i]), "hc_dyn": roc_auc_score(yy, pb[i])})
    r = pd.DataFrame(rec_rows).dropna(subset=["ahi"])
    r["delta"] = r.hc_dyn - r.hc
    r["severity"] = pd.cut(r.ahi, [-1, 5, 15, 30, 1e9], labels=["normal", "mild", "moderate", "severe"])
    print(r.groupby("severity", observed=True).agg(
        records=("record", "size"), median_ahi=("ahi", "median"),
        hc=("hc", "median"), hc_dyn=("hc_dyn", "median"), delta=("delta", "median")).to_string(), flush=True)
    for col, label in [("hc", "HC vs AHI"), ("delta", "improvement vs AHI")]:
        rho, p = stats.spearmanr(r.ahi, r[col])
        print("  %-22s rho %+0.3f  p %.3f" % (label, rho, p), flush=True)
    save("r03_ucddb_apnea_severity.csv", r)
else:
    print("SubjectDetails not found, skipping the apnea severity step", flush=True)
