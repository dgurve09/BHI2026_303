"""Reviewer afFY. Evidence that does not depend on a classifier.

For each descriptor, the N1 versus REM effect size as Hedges' g, and the number
of individual subjects in whom the direction of the difference holds.
"""
from _common import *

WANTED = [("dyn_theta_alpha_mean", "Theta-alpha mean level"),
          ("dyn_theta_alpha_dominance_fraction", "Theta-alpha dominance fraction"),
          ("dyn_theta_alpha_crossing_rate", "Theta-alpha crossing rate"),
          ("dyn_sigma_beta_std", "Sigma-beta SD"),
          ("dyn_theta_alpha_diff_std", "Theta-alpha step SD"),
          ("dyn_band_trajectory_length", "Path length L"),
          ("dyn_band_trajectory_turning", "Turning T")]


def hedges_g(a, b):
    na, nb = len(a), len(b)
    s = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    d = (a.mean() - b.mean()) / (s + F.EPS)
    return d * (1 - 3 / (4 * (na + nb) - 9))          # small-sample correction


rows = []
for col, label in WANTED:
    row = {"descriptor": label, "column": col}
    for name in ["cassette", "telemetry"]:
        df = load(name)
        n1 = df.loc[df.stage_name == "N1", col].dropna().values
        rem = df.loc[df.stage_name == "REM", col].dropna().values
        g_all = hedges_g(n1, rem)
        agree = 0
        subs = df.subject.unique()
        for s in subs:
            d = df[df.subject == s]
            a = d.loc[d.stage_name == "N1", col].dropna().values
            b = d.loc[d.stage_name == "REM", col].dropna().values
            if len(a) > 1 and len(b) > 1 and np.sign(a.mean() - b.mean()) == np.sign(g_all):
                agree += 1
        row["%s_g" % name] = g_all
        row["%s_subjects" % name] = "%d/%d" % (agree, len(subs))
    rows.append(row)
    print("  %-32s cassette g %+0.2f (%s)   telemetry g %+0.2f (%s)"
          % (label, row["cassette_g"], row["cassette_subjects"],
             row["telemetry_g"], row["telemetry_subjects"]), flush=True)
save("r13_effect_sizes.csv", pd.DataFrame(rows))
