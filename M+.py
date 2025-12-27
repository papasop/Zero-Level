# ==========================================================================================================
# Inject-B parameter tuning: sweep + refine search for (mu_sigma, mu_t)
# Goal: hit target beta (optional) while maximizing R2 and minimizing beta_local_std
# ==========================================================================================================

import numpy as np
import pandas as pd

# ---- tuning knobs ----
TARGET_BETA = None        # e.g. 0.5 or 1.0 or whatever you want; set None to just "best R2 + stable local slope"
W_BETA = 1.0              # weight of beta match penalty
W_STD  = 1.0              # weight of beta_local_std
W_R2   = 20.0             # weight of (1-R2) penalty

TOPK = 15

# coarse sweep ranges (start here)
MU_SIGMA_GRID = np.linspace(-0.03, 0.03, 61)   # includes 0.01
MU_T_GRID     = np.linspace(-0.30, 0.30, 61)   # mu_t often controls curvature response more strongly

# refinement step (optional)
REFINE = True
REFINE_RADIUS_SIGMA = 0.004
REFINE_RADIUS_T     = 0.04
REFINE_POINTS = 41  # per axis in refinement

def score_candidate(beta, r2, beta_local_std, target_beta=None):
    """Lower is better."""
    s = 0.0
    if target_beta is not None:
        s += W_BETA * (beta - target_beta)**2
    s += W_STD * (beta_local_std**2)
    s += W_R2 * ((1.0 - r2)**2)
    return float(s)

def summarize_modeB_output(out: dict) -> dict:
    beta = float(out["beta"])
    r2 = float(out["r2"])
    beta_local_std = float(np.std(out["beta_local"]))
    beta_local_span = float(np.max(out["beta_local"]) - np.min(out["beta_local"]))
    return {
        "beta_fit": beta,
        "R2": r2,
        "beta_local_std": beta_local_std,
        "beta_local_span": beta_local_span,
    }

def sweep_injectB(
    *,
    I_mean: float,
    E_mean: float,
    cfg: Config,
    p_theory: float,
    lam: float,
    mu_sigma_grid,
    mu_t_grid,
    title="SWEEP",
    target_beta=None,
):
    rows = []
    for mu_s in mu_sigma_grid:
        for mu_t in mu_t_grid:
            out = run_modeB_mass_scan(
                title=title,
                I_mean=I_mean,
                E_mean=E_mean,
                cfg=cfg,
                p_theory=p_theory,
                lam=lam,
                inject="B",
                mu_sigma=float(mu_s),
                mu_t=float(mu_t),
            )
            stats = summarize_modeB_output(out)
            rows.append({
                "mu_sigma": float(mu_s),
                "mu_t": float(mu_t),
                **stats,
                "score": score_candidate(stats["beta_fit"], stats["R2"], stats["beta_local_std"], target_beta=target_beta),
            })
    df = pd.DataFrame(rows).sort_values("score", ascending=True).reset_index(drop=True)
    return df

def refine_around_best(
    best_mu_sigma: float,
    best_mu_t: float,
    *,
    I_mean: float,
    E_mean: float,
    cfg: Config,
    p_theory: float,
    lam: float,
    target_beta=None,
):
    mu_s_grid = np.linspace(best_mu_sigma - REFINE_RADIUS_SIGMA, best_mu_sigma + REFINE_RADIUS_SIGMA, REFINE_POINTS)
    mu_t_grid = np.linspace(best_mu_t - REFINE_RADIUS_T,     best_mu_t     + REFINE_RADIUS_T,     REFINE_POINTS)
    df_ref = sweep_injectB(
        I_mean=I_mean,
        E_mean=E_mean,
        cfg=cfg,
        p_theory=p_theory,
        lam=lam,
        mu_sigma_grid=mu_s_grid,
        mu_t_grid=mu_t_grid,
        title="REFINE",
        target_beta=target_beta,
    )
    return df_ref

def tune_injectB_from_latest_manifest(target_beta=None):
    # load baseline settlement
    files = sorted(glob.glob("v6_9_4g_fixed3c6_path_seed[0-9].csv"))
    if len(files) == 0 or (not os.path.exists(MANIFEST_PATH)):
        print("[TUNE] baseline CSV or manifest missing.")
        return None

    mfest = read_manifest()
    H_star = np.array(mfest["H_star"], dtype=float)

    resA = settlement_from_files(
        files=files,
        cfg=cfg0,
        H_star=H_star,
        mode="IMPLY_P_FROM_LAMBDA",
        log10K_target=cfg0.log10K_target,
        target_L_ref=L_REF_LOG10,
        p_theory=None
    )

    cand = candidate_p_values(cfg0)
    if MODEB_P_CHOICE not in cand or isinstance(cand[MODEB_P_CHOICE], dict):
        raise ValueError(f"MODEB_P_CHOICE must be one of {[k for k,v in cand.items() if not isinstance(v, dict)]}")
    p_theory = float(cand[MODEB_P_CHOICE])

    lam = lambda_cal_E1_pos_at_star(cfg0)

    print("\n" + "="*108)
    print(f"[TUNE Inject-B] p={MODEB_P_CHOICE}={p_theory:.12f}  target_beta={target_beta}")
    print("="*108)

    df = sweep_injectB(
        I_mean=resA["I_mean"],
        E_mean=resA["E_mean"],
        cfg=cfg0,
        p_theory=p_theory,
        lam=lam,
        mu_sigma_grid=MU_SIGMA_GRID,
        mu_t_grid=MU_T_GRID,
        title="COARSE",
        target_beta=target_beta,
    )

    print("\n[COARSE TOP]")
    print(df.head(TOPK).to_string(index=False))

    if not REFINE:
        return df

    best = df.iloc[0]
    df_ref = refine_around_best(
        best_mu_sigma=float(best["mu_sigma"]),
        best_mu_t=float(best["mu_t"]),
        I_mean=resA["I_mean"],
        E_mean=resA["E_mean"],
        cfg=cfg0,
        p_theory=p_theory,
        lam=lam,
        target_beta=target_beta,
    )

    print("\n[REFINE TOP]")
    print(df_ref.head(TOPK).to_string(index=False))

    return df, df_ref

# ---- Run tuning ----
# Example 1: just find "most stable Inject-B" (high R2, low local std), no target beta:
# df_coarse, df_ref = tune_injectB_from_latest_manifest(target_beta=None)

# Example 2: force beta near a desired value:
# df_coarse, df_ref = tune_injectB_from_latest_manifest(target_beta=0.50)

df_coarse, df_ref = tune_injectB_from_latest_manifest(target_beta=TARGET_BETA)


