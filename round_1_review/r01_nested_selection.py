"""Reviewer bCwc, major concern 2. Was the feature subset chosen on the same folds?

The submitted paper fixed the proposed set a priori. Here the subset is instead
chosen inside each training fold by inner subject-grouped cross-validation, so
the held-out fold never influences the choice. If the fixed and nested results
agree, the selection step was not a free parameter.

The inner cross-validation is slow, so each subset's out-of-fold predictions are
cached under results/_oof/ and a rerun continues from there.
"""
from _common import *
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

CANDIDATES = {
    "theta_alpha_only": lambda c: [x for x in c if "theta_alpha" in x],
    "order_free_only": order_free,
    "all_dyn": lambda c: list(c),
}


def fit_subset(df, hc, dyn, y, g, name):
    store = OUT / "_oof" / ("r01_%s.npz" % name)
    store.parent.mkdir(parents=True, exist_ok=True)
    if store.exists():
        z = np.load(store, allow_pickle=True)
        return z["hc"], z["fixed"], z["nested"], list(z["picks"])

    oof_hc, oof_fixed, oof_nested = (np.zeros(len(y)) for _ in range(3))
    picks = []
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=E.SEED)
    for fold, (tr, te) in enumerate(outer.split(df[hc + dyn], y, g), 1):
        sub = df.iloc[tr]
        best, best_auc = None, -1.0
        for cand, pick in CANDIDATES.items():
            cols = hc + pick(dyn)
            inner = StratifiedGroupKFold(3, shuffle=True, random_state=E.SEED + 1)
            p = np.zeros(len(tr))
            for i, j in inner.split(sub[cols], y[tr], g[tr]):
                m = E.make_model().fit(sub[cols].values[i], y[tr][i])
                p[j] = m.predict_proba(sub[cols].values[j])[:, 1]
            auc = roc_auc_score(y[tr], p)
            if auc > best_auc:
                best, best_auc = cand, auc
        picks.append(best)
        for cols, store_arr in [(hc, oof_hc), (hc + dyn, oof_fixed),
                                (hc + CANDIDATES[best](dyn), oof_nested)]:
            m = E.make_model().fit(df[cols].values[tr], y[tr])
            store_arr[te] = m.predict_proba(df[cols].values[te])[:, 1]
        print("  fold %d/5 chose %s" % (fold, best), flush=True)
    np.savez(store, hc=oof_hc, fixed=oof_fixed, nested=oof_nested, picks=np.array(picks))
    return oof_hc, oof_fixed, oof_nested, picks


rows = []
for name in ["cassette", "telemetry"]:
    df = load(name)
    y, g = label_and_group(df, name)
    hc, dyn = F.hc_cols(df), F.dyn_cols(df)
    oof_hc, oof_fixed, oof_nested, picks = fit_subset(df, hc, dyn, y, g, name)
    print("%s: subset chosen per outer fold = %s" % (name, picks), flush=True)
    for tag, p in [("HC", oof_hc), ("subset fixed a priori", oof_fixed),
                   ("subset chosen in training folds", oof_nested)]:
        row = {"subset": name, "model": tag, "auc": roc_auc_score(y, p)}
        if tag != "HC":
            row.update(cached("r01b_%s_%s" % (name, tag.split()[1]),
                              lambda p=p: E.bootstrap_gain(g, y, p, oof_hc)))
        rows.append(row)
        print("  %-32s AUC %.3f%s" % (tag, row["auc"],
              "" if tag == "HC" else "  delta %+0.3f [%+0.3f, %+0.3f]"
              % (row["delta_auc"], row["ci_low"], row["ci_high"])), flush=True)
save("r01_nested_selection.csv", pd.DataFrame(rows))
