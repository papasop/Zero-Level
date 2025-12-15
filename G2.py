import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# -------------------------
# 0) choose y and X
# -------------------------
dfc = df.copy()

y_col = "F"                 # <- 关键：别用 log(|log|) 了
feat_cols = ["Zc_log", "Zs_entropy"]  # <- 关键：跨族更稳的输入
group_col = "family"

# sanity
for c in [y_col] + feat_cols + [group_col]:
    assert c in dfc.columns, f"missing col: {c}"

# drop NA/inf (should be none)
Xbase = dfc[feat_cols].replace([np.inf, -np.inf], np.nan)
y = dfc[y_col].replace([np.inf, -np.inf], np.nan)
mask = Xbase.notna().all(axis=1) & y.notna()
dfc = dfc.loc[mask].copy()

Xbase = dfc[feat_cols]
y = dfc[y_col].to_numpy(float)
groups = dfc[group_col].astype(str).to_numpy()

print("clean rows:", len(dfc))
print("families:", sorted(dfc[group_col].unique()))

# -------------------------
# 1) Model A: Global only (no family)
# -------------------------
model_global = Pipeline([
    ("scaler", StandardScaler()),
    ("reg", Ridge(alpha=1.0))
])

# -------------------------
# 2) Model B: Family-aware (topological locking)
#     y = f(Zc_log, Zs_entropy) + family_offset
# -------------------------
pre = ColumnTransformer([
    ("num", StandardScaler(), feat_cols),
    ("fam", OneHotEncoder(handle_unknown="ignore"), [group_col])
], remainder="drop")

model_family = Pipeline([
    ("pre", pre),
    ("reg", Ridge(alpha=1.0))
])

# -------------------------
# 3) CV helpers
# -------------------------
def cv_r2(model, Xdf, y, groups=None, n_splits=5, seed=0):
    r2s = []
    if groups is None:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = splitter.split(Xdf)
    else:
        splitter = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
        splits = splitter.split(Xdf, y, groups)

    for tr, te in splits:
        model.fit(Xdf.iloc[tr], y[tr])
        pred = model.predict(Xdf.iloc[te])
        r2s.append(r2_score(y[te], pred))
    return float(np.mean(r2s)), float(np.std(r2s))

Xdf_global = dfc[feat_cols]
Xdf_family = dfc[feat_cols + [group_col]]

print("\n=== Random KFold (mixed) ===")
mu, sd = cv_r2(model_global, Xdf_global, y, groups=None)
print(f"Global-only:   R2_cv={mu:.4f} ± {sd:.4f}")
mu, sd = cv_r2(model_family, Xdf_family, y, groups=None)
print(f"Family-aware:  R2_cv={mu:.4f} ± {sd:.4f}")

print("\n=== GroupKFold (leave-family-out) ===")
mu, sd = cv_r2(model_global, Xdf_global, y, groups=groups)
print(f"Global-only:   R2_cv={mu:.4f} ± {sd:.4f}")
mu, sd = cv_r2(model_family, Xdf_family, y, groups=groups)
print(f"Family-aware:  R2_cv={mu:.4f} ± {sd:.4f}")

print("\n✅ DONE (print end marker)")
