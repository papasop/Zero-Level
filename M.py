# ==========================================================================================================
# FULL ONE-CELL (Runner + Settlement + Hessian-from-TEEHERED-loss + MODE A/B + Ablations)
# FIX: Hessian must come from tethered loss: full_loss + tether_w ||x-x*||^2
# ==========================================================================================================

from __future__ import annotations
import math, os, csv, glob, json, hashlib
from dataclasses import dataclass, asdict, replace
from typing import Callable, Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import mpmath as mp
mp.dps = 50

# ----------------------------------------------------------------------------------------------------------
# GLOBAL KNOBS
# ----------------------------------------------------------------------------------------------------------
VERSION_TAG = "V6.9.4g_fixed3c6+V6.95c_PATCH_DEEPSEEK_VALIDATION_TETHERED_HESSIAN_FIX_2025-12-27"

# IO
CSV_GLOB = "v6_9_4g_fixed3c6_path_seed*.csv"
MANIFEST_PATH = "v6_9_4g_fixed3c6_manifest.json"

# Runner
ENSEMBLE_SEEDS = 5
FORCE_RERUN = True
PRINT_EACH_STEP = False

# Settlement Mode:
#   "IMPLY_P_FROM_LAMBDA"       => implied p from Λ_obs (consistency/inversion; NOT prediction)
#   "PREDICT_LAMBDA_FROM_P"     => predict Λ from fixed p_theory (true prediction test)
SETTLEMENT_MODE = "IMPLY_P_FROM_LAMBDA"

# External ledger target for MODE A or error reference for MODE B:
LAMBDA_OBS_LOG10 = -122.0

# For MODE B: set a fixed external p_theory if you want true prediction
P_THEORY: Optional[float] = None

# Ablation suite
RUN_ABLATIONS = True

# ----------------------------------------------------------------------------------------------------------
# UTIL
# ----------------------------------------------------------------------------------------------------------
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha256_json(obj: dict) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(blob)

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)

def project_box(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.array(
        [clamp(float(x[0]), float(lo[0]), float(hi[0])),
         clamp(float(x[1]), float(lo[1]), float(hi[1]))],
        dtype=float
    )

def write_csv(path: str, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def safe_log10(x: float, floor: float = 1e-300) -> float:
    return math.log10(max(abs(float(x)), floor))

# ----------------------------------------------------------------------------------------------------------
# CORE MODEL
# ----------------------------------------------------------------------------------------------------------
def zeta_abs(sigma: float, t: float) -> float:
    return float(abs(mp.zeta(mp.mpc(sigma, t))))

def veneziano_logabs(sigma: float, t: float, alpha0: float = 0.0, alpha1: float = 1.0) -> float:
    zs = mp.mpc(sigma, t)
    zt = mp.mpc(t, sigma)
    a_s = alpha0 + alpha1 * zs
    a_t = alpha0 + alpha1 * zt
    try:
        lg = mp.loggamma(-a_s) + mp.loggamma(-a_t) - mp.loggamma(-(a_s + a_t))
        return float(mp.re(lg))
    except Exception:
        return float("nan")

def numeric_grad(f: Callable[[np.ndarray], float], x: np.ndarray, h: float = 2e-6) -> np.ndarray:
    g = np.zeros(2, dtype=float)
    for i in range(2):
        xp = x.copy(); xm = x.copy()
        xp[i] += h; xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2.0 * h)
    return g

def numeric_hessian(f: Callable[[np.ndarray], float], x: np.ndarray, h: float = 5e-5) -> np.ndarray:
    H = np.zeros((2,2), dtype=float)
    fx = f(x)

    for i in range(2):
        ei = np.zeros(2); ei[i] = 1.0
        fpp = f(x + h*ei)
        fmm = f(x - h*ei)
        H[i,i] = (fpp - 2.0*fx + fmm) / (h*h)

    e0 = np.array([1.0, 0.0])
    e1 = np.array([0.0, 1.0])
    fpp = f(x + h*e0 + h*e1)
    fpm = f(x + h*e0 - h*e1)
    fmp = f(x - h*e0 + h*e1)
    fmm = f(x - h*e0 - h*e1)
    H01 = (fpp - fpm - fmp + fmm) / (4.0*h*h)
    H[0,1] = H01
    H[1,0] = H01
    return H

# ----------------------------------------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------------------------------------
@dataclass
class Config:
    ppm_proton: float = -70.64
    ppm_neutron: float = -0.21
    ppm_muon: float = 0.20

    sigma_star: float = 0.597496091691
    t_star: float = 16.050352300380

    log10K_target: float = 31.935489

    max_steps: int = 12000
    min_steps: int = 40
    tether_w: float = 120.0

    step_radius: float = 0.03

    box_sigma_rad: float = 0.08
    box_t_rad: float = 0.15

    w_ppm: float = 1.0
    w_K: float = 1.0
    w_V: float = 0.03

@dataclass
class StopCfg:
    I_win: int = 400
    I_rel_tol: float = 1e-7
    I_eps: float = 1e-15
    I_patience: int = 2

    B_win: int = 50
    dist_plateau_tol: float = 1e-12
    wobble_r_tol: float = 0.10

    plateau_win: int = 600
    plateau_dist_tol: float = 1e-12

cfg0 = Config()
stop_cfg0 = StopCfg()

params_star = {
    "eta": 0.050006178519845686,
    "a_nm": 1148692.547863098,
    "a_p": 999.9999999995263,
    "b_p": 0.00027728027428906317
}

# ----------------------------------------------------------------------------------------------------------
# OBSERVABLES / PPM
# ----------------------------------------------------------------------------------------------------------
def observable_proton(sigma: float, t: float, p: Dict[str, float]) -> float:
    z = complex(sigma, t)
    term = 1.0 / (1.0 + z * z)
    return float(p["a_p"] * term.real + p["b_p"] * term.imag)

def observable_neutron(sigma: float, t: float, p: Dict[str, float]) -> float:
    z = complex(sigma, t)
    term = math.log1p(abs(z)) + float(p["eta"]) * (1.0 / (1.0 + z)).real
    return float(p["a_nm"] * term)

def observable_muon(sigma: float, t: float, p: Dict[str, float]) -> float:
    term = (math.e ** (-float(p["eta"]) * sigma)) * math.cos(float(p["eta"]) * t)
    return float(1000.0 * term)

def ppm_residuals_centered(sigma: float, t: float, p: Dict[str, float], cfg: Config) -> Tuple[float, float, float]:
    mp_x = observable_proton(sigma, t, p)
    mn_x = observable_neutron(sigma, t, p)
    mm_x = observable_muon(sigma, t, p)

    mp_s = observable_proton(cfg.sigma_star, cfg.t_star, p)
    mn_s = observable_neutron(cfg.sigma_star, cfg.t_star, p)
    mm_s = observable_muon(cfg.sigma_star, cfg.t_star, p)

    sp = 1.0
    sn = 1e-6
    sm = 1e-3

    ppm_p = cfg.ppm_proton + (mp_x - mp_s) * sp
    ppm_n = cfg.ppm_neutron + (mn_x - mn_s) * sn
    ppm_m = cfg.ppm_muon + (mm_x - mm_s) * sm
    return float(ppm_p), float(ppm_n), float(ppm_m)

def ppm_errors_relative_to_baseline(sigma: float, t: float, p: Dict[str, float], cfg: Config) -> Tuple[float, float, float]:
    ppm_p, ppm_n, ppm_m = ppm_residuals_centered(sigma, t, p, cfg)
    return (ppm_p - cfg.ppm_proton, ppm_n - cfg.ppm_neutron, ppm_m - cfg.ppm_muon)

# ----------------------------------------------------------------------------------------------------------
# K lock
# ----------------------------------------------------------------------------------------------------------
def S_pos(sigma: float, t: float) -> float:
    a = zeta_abs(sigma, t)
    return 1.0 + a * a

def lambda_cal_E1_pos_at_star(cfg: Config) -> float:
    return float(cfg.log10K_target / max(S_pos(cfg.sigma_star, cfg.t_star), 1e-30))

def log10K_pred(sigma: float, t: float, lam: float) -> float:
    return float(lam * S_pos(sigma, t))

# ----------------------------------------------------------------------------------------------------------
# LOSS + METRIC
# ----------------------------------------------------------------------------------------------------------
def full_loss(x: np.ndarray, p: Dict[str, float], lam: float, cfg: Config) -> float:
    sigma, t = float(x[0]), float(x[1])

    ep, en, em = ppm_errors_relative_to_baseline(sigma, t, p, cfg)
    L_ppm = (ep / 80.0) ** 2 + (en / 0.2) ** 2 + (em / 0.2) ** 2

    lk = log10K_pred(sigma, t, lam)
    L_K = (lk - cfg.log10K_target) ** 2

    LV = veneziano_logabs(sigma, t)
    L_V = 10.0 if not np.isfinite(LV) else (LV / 10.0) ** 2

    return float(cfg.w_ppm * L_ppm + cfg.w_K * L_K + cfg.w_V * L_V)

def spd_metric_stable(x: np.ndarray, p: Dict[str, float], lam: float, cfg: Config) -> np.ndarray:
    f = lambda xx: full_loss(xx, p, lam, cfg)
    g = numeric_grad(f, x, h=2e-6)
    M = np.outer(g, g)
    tr = float(np.trace(M))
    if tr <= 1e-30:
        return np.eye(2)
    M = M / tr
    return (np.eye(2) + M + 1e-8 * np.eye(2)).astype(float)

def _unit(v: np.ndarray, eps: float = 1e-300) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
    return v / n

def _mean_resultant_length(dirs: List[np.ndarray]) -> float:
    if len(dirs) == 0:
        return 1.0
    s = np.zeros(2, dtype=float)
    for d in dirs:
        s += _unit(d)
    return float(np.linalg.norm(s) / max(1, len(dirs)))

# ----------------------------------------------------------------------------------------------------------
# PATH RUNNER (STOPFIX)
# ----------------------------------------------------------------------------------------------------------
def run_path_integral_hardened(
    x0: np.ndarray,
    x_star: np.ndarray,
    loss_fn: Callable[[np.ndarray], float],
    grad_fn: Callable[[np.ndarray], np.ndarray],
    metric_fn: Callable[[np.ndarray], np.ndarray],
    lo: np.ndarray,
    hi: np.ndarray,
    max_steps: int,
    min_steps: int,
    tether_w: float,
    step_radius0: float,
    stop_cfg: StopCfg,
    alpha0: float = 1.0,
    backtrack: float = 0.5,
    max_bt: int = 25,
) -> Tuple[List[Dict[str, float]], float, np.ndarray, float, str]:

    path: List[Dict[str, float]] = []
    I_path = 0.0

    I_hist: List[float] = [0.0]
    dist_hist: List[float] = []
    dx_hist: List[np.ndarray] = []

    A_ok_count = 0
    stop_reason = "max_steps"

    x = project_box(np.array(x0, dtype=float), lo, hi)

    def tethered_loss(xx: np.ndarray) -> float:
        return float(loss_fn(xx) + tether_w * float(np.sum((xx - x_star) ** 2)))

    L = float(tethered_loss(x))

    for k in range(max_steps):
        step_radius = step_radius0 * (0.5 + 0.5 * math.exp(-k / 600.0))

        g_raw = grad_fn(x).astype(float)
        g = g_raw + (2.0 * tether_w) * (x - x_star)
        G = metric_fn(x).astype(float) + 1e-12 * np.eye(2)

        try:
            dx_dir = -np.linalg.solve(G, g)
        except np.linalg.LinAlgError:
            dx_dir = -g

        nrm = float(np.linalg.norm(dx_dir))
        if nrm > step_radius:
            dx_dir = dx_dir * (step_radius / nrm)

        alpha = float(alpha0)
        accepted = False
        x_new = x.copy()
        L_new = L

        for _ in range(max_bt):
            cand = project_box(x + alpha * dx_dir, lo, hi)
            cand_L = float(tethered_loss(cand))
            if cand_L <= L + (1e-12 + 1e-10 * max(1.0, abs(L))):
                accepted = True
                x_new, L_new = cand, cand_L
                break
            alpha *= backtrack
            if alpha < 1e-16:
                break

        if not accepted:
            if k < min_steps:
                pull = -0.5 * (x - x_star)
                pn = float(np.linalg.norm(pull))
                if pn > step_radius:
                    pull = pull * (step_radius / pn)
                x_new = project_box(x + pull, lo, hi)
                L_new = float(tethered_loss(x_new))
                stop_reason = "pullback"
            else:
                stop_reason = "no_accept_after_min"
                break

        dx = (x_new - x).astype(float)
        dx_hist.append(dx.copy())

        G_here = metric_fn(x).astype(float) + 1e-12 * np.eye(2)
        ds2 = float(dx.T @ G_here @ dx)
        ds = float(np.sqrt(max(0.0, ds2)))

        I_path += ds
        I_hist.append(I_path)

        x = x_new
        dL = float(L_new - L)
        L = L_new

        dist = float(np.linalg.norm(x - x_star))
        dist_hist.append(dist)

        row = {
            "step": float(k),
            "sigma": float(x[0]),
            "t": float(x[1]),
            "loss": float(L),
            "alpha": float(alpha),
            "ds": float(ds),
            "I_path": float(I_path),
            "dist_to_star": float(dist),
            "grad_norm": float(np.linalg.norm(g)),
            "dx_norm": float(np.linalg.norm(dx)),
            "dL": float(dL),
            "stop_reason_code": float(0.0),
        }
        path.append(row)

        if PRINT_EACH_STEP and (k % 200 == 0):
            print(f"  step={k:5d} I={I_path:.6f} dist={dist:.3e} loss={L:.6e}")

        if k < min_steps:
            continue

        # A) windowed I convergence
        W = stop_cfg.I_win
        if len(I_hist) >= W + 1:
            I_old = I_hist[-(W + 1)]
            I_new = I_hist[-1]
            relW = abs(I_new - I_old) / max(abs(I_old), stop_cfg.I_eps)

            if relW < stop_cfg.I_rel_tol:
                A_ok_count += 1
            else:
                A_ok_count = 0

            if A_ok_count >= stop_cfg.I_patience:
                stop_reason = "I_converged"
                break

        # B) stagnation + wobble (operational)
        BW = stop_cfg.B_win
        if len(dist_hist) >= BW and len(dx_hist) >= BW:
            recent_best = min(dist_hist[-BW:])
            prev_best = min(dist_hist[:-BW]) if len(dist_hist) > BW else recent_best
            no_progress = (recent_best > prev_best - stop_cfg.dist_plateau_tol)

            recent_dirs = [_unit(d) for d in dx_hist[-BW:]]
            r = _mean_resultant_length(recent_dirs)
            wobble = (r < stop_cfg.wobble_r_tol)

            if no_progress and wobble:
                stop_reason = "stagnation_wobble"
                break

        # C) plateau
        PW = stop_cfg.plateau_win
        if len(dist_hist) >= PW:
            best_recent = min(dist_hist[-PW:])
            best_all = min(dist_hist)
            if best_recent > best_all - stop_cfg.plateau_dist_tol:
                stop_reason = "plateau_dist"
                break

    reason_map = {
        "I_converged": 1.0,
        "stagnation_wobble": 2.0,
        "plateau_dist": 2.2,
        "no_accept_after_min": 3.0,
        "pullback": 3.5,
        "max_steps": 4.0,
    }
    if path:
        path[-1]["stop_reason_code"] = reason_map.get(stop_reason, 9.0)

    dist_end = float(np.linalg.norm(x - x_star))
    return path, float(I_path), x, dist_end, stop_reason

# ----------------------------------------------------------------------------------------------------------
# MANIFEST
# ----------------------------------------------------------------------------------------------------------
def build_manifest(extra: dict) -> dict:
    m = {
        "version": VERSION_TAG,
        "mp_dps": int(mp.dps),
        "cfg": asdict(extra["cfg"]),
        "stop_cfg": asdict(extra["stop_cfg"]),
        "params_star": params_star,
        "csv_glob": CSV_GLOB,
        "hessian_source": extra.get("hessian_source", "numeric_hessian(tethered_loss_at_star)"),
        "H_star": extra.get("H_star"),
        **{k:v for k,v in extra.items() if k not in ("cfg","stop_cfg","H_star","hessian_source")},
    }
    m["sha256"] = sha256_json(m)
    return m

def write_manifest(m: dict) -> None:
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, sort_keys=True)

def read_manifest() -> dict:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def validate_csvs(files: List[str]) -> bool:
    required = {"sigma", "t", "I_path", "dist_to_star", "grad_norm", "dx_norm", "dL", "stop_reason_code"}
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception:
            return False
        cols = set([c.lower() for c in df.columns])
        if not required.issubset(cols):
            return False
        if len(df) < 10:
            return False
    return True

# ----------------------------------------------------------------------------------------------------------
# RUNNER (FIXED HESSIAN SOURCE)
# ----------------------------------------------------------------------------------------------------------
def run_v6_9_4g_fixed3c6_full(cfg: Config, stop_cfg: StopCfg, ensemble_seeds: int = 5, tag: str = "") -> dict:
    lam = lambda_cal_E1_pos_at_star(cfg)
    sigma_star, t_star = cfg.sigma_star, cfg.t_star
    x_star = np.array([sigma_star, t_star], dtype=float)

    lo = np.array([sigma_star - cfg.box_sigma_rad, t_star - cfg.box_t_rad], dtype=float)
    hi = np.array([sigma_star + cfg.box_sigma_rad, t_star + cfg.box_t_rad], dtype=float)

    def loss_wrapped(xx: np.ndarray) -> float:
        return full_loss(xx, params_star, lam, cfg)

    def grad_wrapped(xx: np.ndarray) -> np.ndarray:
        return numeric_grad(lambda z: full_loss(z, params_star, lam, cfg), xx, h=2e-6)

    def metric_wrapped(xx: np.ndarray) -> np.ndarray:
        return spd_metric_stable(xx, params_star, lam, cfg)

    # ---------------- FIX: Hessian must come from tethered loss ----------------
    def tethered_loss_for_H(xx: np.ndarray) -> float:
        return float(full_loss(xx, params_star, lam, cfg) + cfg.tether_w * float(np.sum((xx - x_star) ** 2)))

    H_star = numeric_hessian(tethered_loss_for_H, x_star, h=5e-5)
    # -------------------------------------------------------------------------

    # Diagnostics at s*
    absz = zeta_abs(sigma_star, t_star)
    ep, en, em = ppm_errors_relative_to_baseline(sigma_star, t_star, params_star, cfg)
    log10K = log10K_pred(sigma_star, t_star, lam)
    LV = veneziano_logabs(sigma_star, t_star)

    print("\n" + "=" * 108)
    print(f"Runner {VERSION_TAG}{(' :: ' + tag) if tag else ''}")
    print("=" * 108)
    print("[Step 0] s* diagnostics")
    print(f"s* = {sigma_star:.12f} + {t_star:.12f}i")
    print(f"|zeta(s*)| = {absz:.6e}")
    print(f"ppm errors = ({ep:.3f},{en:.3f},{em:.3f}) [0 by construction]")
    print(f"log10K(s*) = {log10K:.6f} target={cfg.log10K_target:.6f} [calibrated]")
    print(f"Veneziano log|A|(s*) = {LV:.6f}")

    w, _ = np.linalg.eigh((H_star + H_star.T)/2.0)
    lam_min, lam_max = float(w[0]), float(w[-1])
    cond = lam_max / max(lam_min, 1e-300)
    print("[Hessian from TETHERED loss @ s*] (numeric)")
    print(np.array2string(H_star, precision=6, suppress_small=False))
    print(f"eigs: lam_min={lam_min:.6e} lam_max={lam_max:.6e}  cond={cond:.6f}")

    rng = np.random.default_rng(12345)
    I_list, dist_list = [], []
    reason_counts: Dict[str, int] = {}

    print("\n[Step 1] ensemble path run (STOPFIX A/B/C)")
    for seed in range(ensemble_seeds):
        jitter = rng.normal(0.0, 0.02, size=(2,))
        x0 = project_box(x_star + jitter, lo, hi)

        path, I_path, x_end, dist_end, stop_reason = run_path_integral_hardened(
            x0=x0,
            x_star=x_star,
            loss_fn=loss_wrapped,
            grad_fn=grad_wrapped,
            metric_fn=metric_wrapped,
            lo=lo,
            hi=hi,
            max_steps=cfg.max_steps,
            min_steps=cfg.min_steps,
            tether_w=cfg.tether_w,
            step_radius0=cfg.step_radius,
            stop_cfg=stop_cfg,
            alpha0=1.0,
        )

        out_csv = f"v6_9_4g_fixed3c6_path_seed{seed}{('_' + tag) if tag else ''}.csv"
        write_csv(out_csv, path)

        I_list.append(I_path)
        dist_list.append(dist_end)
        reason_counts[stop_reason] = reason_counts.get(stop_reason, 0) + 1

        print(f"[seed {seed}] I={I_path:.6f} steps={len(path):5d} dist={dist_end:.3e} stop={stop_reason} x_end=({x_end[0]:.6f},{x_end[1]:.6f})")

    print("\n[Ensemble summary]")
    print(f"I mean={float(np.mean(I_list)):.6f} std={float(np.std(I_list)):.6f}")
    print(f"dist mean={float(np.mean(dist_list)):.3e} std={float(np.std(dist_list)):.3e}")
    print("stop reasons:", reason_counts)

    m = build_manifest({
        "cfg": cfg,
        "stop_cfg": stop_cfg,
        "ensemble_seeds": ensemble_seeds,
        "files_written": [f"v6_9_4g_fixed3c6_path_seed{i}{('_' + tag) if tag else ''}.csv" for i in range(ensemble_seeds)],
        "stop_reasons": reason_counts,
        "H_star": H_star.tolist(),
        "hessian_source": "numeric_hessian(tethered_loss = full_loss + tether_w||x-x*||^2 at s*)",
        "tag": tag,
    })
    write_manifest(m)
    print(f"[Manifest] wrote {MANIFEST_PATH} sha256={m['sha256'][:16]}...")
    return m

# ----------------------------------------------------------------------------------------------------------
# SETTLEMENT
# ----------------------------------------------------------------------------------------------------------
def read_path_csv_strict(fp: str) -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(fp)
    cols = [c.lower() for c in df.columns]
    if "sigma" not in cols or "t" not in cols:
        raise ValueError(f"CSV missing sigma/t columns: {fp}")
    cx = df.columns[cols.index("sigma")]
    cy = df.columns[cols.index("t")]
    x = df[[cx, cy]].astype(float).to_numpy()
    return df, x

def settlement_from_files(
    files: List[str],
    cfg: Config,
    H_star: np.ndarray,
    mode: str,
    log10K_target: float,
    target_log10_lambda: float,
    p_theory: Optional[float],
) -> dict:
    sigma_star, t_star = cfg.sigma_star, cfg.t_star
    x_star = np.array([sigma_star, t_star], dtype=float)

    I_list, dist_end_list, dx_list, stop_codes = [], [], [], []

    for f in files:
        df, x = read_path_csv_strict(f)
        I = float(pd.to_numeric(df["I_path"], errors="coerce").dropna().iloc[-1])
        x_end = x[-1]
        dist_end = float(np.linalg.norm(x_end - x_star))
        dx = (x_end - x_star).astype(float)

        I_list.append(I)
        dist_end_list.append(dist_end)
        dx_list.append(dx)

        if "stop_reason_code" in df.columns:
            stop_codes.append(float(pd.to_numeric(df["stop_reason_code"], errors="coerce").dropna().iloc[-1]))

    I_arr = np.array(I_list, dtype=float)
    dist_end_arr = np.array(dist_end_list, dtype=float)
    dist2_arr = dist_end_arr ** 2

    # Quadratic energy using the (tethered) Hessian
    E_list = [0.5 * float(dx.T @ H_star @ dx) for dx in dx_list]
    E_arr = np.array(E_list, dtype=float)

    I_mean = float(I_arr.mean())
    E_mean = float(E_arr.mean())
    log10I = safe_log10(I_mean)
    log10E = safe_log10(E_mean)
    log10K = float(log10K_target)

    def log10Lambda(p: float) -> float:
        return log10I + log10E - p * log10K

    out: dict = {
        "I_mean": I_mean, "I_std": float(I_arr.std()),
        "E_mean": E_mean, "E_std": float(E_arr.std()),
        "dist_end_mean": float(dist_end_arr.mean()), "dist_end_std": float(dist_end_arr.std()),
        "dist2_mean": float(dist2_arr.mean()),
        "log10I": log10I, "log10E": log10E, "log10K": log10K,
        "stop_codes": stop_codes,
    }

    if mode.upper() == "IMPLY_P_FROM_LAMBDA":
        p_star = (log10I + log10E - target_log10_lambda) / log10K
        out["p_star"] = float(p_star)
        out["log10Lambda_at_p_star"] = float(log10Lambda(p_star))
        out["mode"] = "IMPLY_P_FROM_LAMBDA"
    elif mode.upper() == "PREDICT_LAMBDA_FROM_P":
        if p_theory is None:
            raise ValueError("P_THEORY must be set for prediction mode.")
        pred = float(log10Lambda(float(p_theory)))
        out["p_theory"] = float(p_theory)
        out["log10Lambda_pred"] = pred
        out["delta_dex_vs_obs"] = float(pred - target_log10_lambda)
        out["mode"] = "PREDICT_LAMBDA_FROM_P"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return out

def print_settlement_report(title: str, files: List[str], H_star: np.ndarray, cfg: Config, result: dict) -> None:
    print("\n" + "=" * 108)
    print(f"[SETTLEMENT] {title}  mode={result['mode']}  files={len(files)}")
    print("=" * 108)
    for f in files:
        print(" -", f)

    print("\n[Ensemble stats]")
    print(f"I_mean        = {result['I_mean']:.13g}   std={result['I_std']:.5g}")
    print(f"E_mean        = {result['E_mean']:.13g}   std={result['E_std']:.5g}")
    print(f"dist_end_mean = {result['dist_end_mean']:.13g} std={result['dist_end_std']:.5g}")
    print(f"dist2_mean    = {result['dist2_mean']:.13g}")

    if result.get("stop_codes"):
        u, c = np.unique(np.array(result["stop_codes"], dtype=float), return_counts=True)
        print("[stop_reason_code counts]", dict(zip(u.tolist(), c.tolist())))

    w, _ = np.linalg.eigh((H_star + H_star.T)/2.0)
    lam_min, lam_max = float(w[0]), float(w[-1])
    cond = lam_max / max(lam_min, 1e-300)

    print("\n[Hessian @ s*] (TETHERED; numeric)")
    print(np.array2string(H_star, precision=6, suppress_small=False))
    print(f"eigs: lam_min={lam_min:.6e} lam_max={lam_max:.6e}  cond={cond:.6f}")

    print("\n[LOG-domain ledger]")
    print(f"log10(I)     = {result['log10I']:.6f}")
    print(f"log10(E_res) = {result['log10E']:.6f}")
    print(f"log10(K)     = {result['log10K']:.6f}")

    if result["mode"] == "IMPLY_P_FROM_LAMBDA":
        print(f"log10 Λ_obs  = {LAMBDA_OBS_LOG10:.6f}")
        print(f"p_star       = {result['p_star']:.12f}  (implied; NOT prediction of Λ)")
        print(f"check log10Λ(p_star) = {result['log10Lambda_at_p_star']:.12f}")
    else:
        print(f"p_theory     = {result['p_theory']:.12f}")
        print(f"pred log10Λ  = {result['log10Lambda_pred']:.12f}")
        print(f"Δdex vs obs  = {result['delta_dex_vs_obs']:+.6f}")

# ----------------------------------------------------------------------------------------------------------
# CANDIDATE p values (transparent)
# ----------------------------------------------------------------------------------------------------------
def candidate_p_values(cfg: Config) -> dict:
    sigma, t = cfg.sigma_star, cfg.t_star
    p_geom = 3.5 + (math.pi / 2.0) * (sigma / t)
    alpha_inv = 137.035999
    p_alpha1 = 3.5 + alpha_inv / 2400.0
    alpha = 1.0 / alpha_inv
    p_alpha2 = 3.5 + 8.592 * alpha
    return {"p_geom": p_geom, "p_alpha1": p_alpha1, "p_alpha2": p_alpha2}

# ----------------------------------------------------------------------------------------------------------
# ABLATIONS
# ----------------------------------------------------------------------------------------------------------
@dataclass
class AblationCase:
    name: str
    cfg_patch: Dict[str, Any]
    stop_patch: Dict[str, Any]

def run_ablation_suite(base_cfg: Config, base_stop: StopCfg) -> None:
    cand = candidate_p_values(base_cfg)

    cases = [
        AblationCase("BASELINE", {}, {}),
        AblationCase("WOBBLE_STRICT", {}, {"wobble_r_tol": 0.05}),
        AblationCase("WOBBLE_LOOSE",  {}, {"wobble_r_tol": 0.20}),
        AblationCase("V_WEIGHT_LOW",  {"w_V": 0.01}, {}),
        AblationCase("V_WEIGHT_HIGH", {"w_V": 0.10}, {}),
        AblationCase("TETHER_LOW",  {"tether_w": 60.0}, {}),
        AblationCase("TETHER_HIGH", {"tether_w": 240.0}, {}),
        AblationCase("BOX_TIGHT", {"box_sigma_rad": 0.05, "box_t_rad": 0.10}, {}),
        AblationCase("BOX_WIDE",  {"box_sigma_rad": 0.12, "box_t_rad": 0.22}, {}),
    ]

    print("\n" + "=" * 108)
    print("[ABLATIONS] Sensitivity suite (each case runs fresh CSVs with its own tag)")
    print("=" * 108)
    print("Candidate p values (transparent set):")
    for k,v in cand.items():
        print(f"  {k:9s} = {v:.10f}")

    summary_rows = []

    for case in cases:
        tag = case.name
        cfg = replace(base_cfg, **case.cfg_patch) if case.cfg_patch else base_cfg
        stp = replace(base_stop, **case.stop_patch) if case.stop_patch else base_stop

        for fp in glob.glob(f"v6_9_4g_fixed3c6_path_seed*_{tag}.csv"):
            try: os.remove(fp)
            except: pass

        m = run_v6_9_4g_fixed3c6_full(cfg, stp, ensemble_seeds=ENSEMBLE_SEEDS, tag=tag)
        H_star = np.array(m["H_star"], dtype=float)
        files = sorted(glob.glob(f"v6_9_4g_fixed3c6_path_seed*_{tag}.csv"))

        resA = settlement_from_files(
            files=files,
            cfg=cfg,
            H_star=H_star,
            mode="IMPLY_P_FROM_LAMBDA",
            log10K_target=cfg.log10K_target,
            target_log10_lambda=LAMBDA_OBS_LOG10,
            p_theory=None
        )

        row = {
            "case": tag,
            "I_mean": resA["I_mean"],
            "I_std": resA["I_std"],
            "E_mean": resA["E_mean"],
            "E_std": resA["E_std"],
            "dist_end_mean": resA["dist_end_mean"],
            "p_star_implied": resA["p_star"],
        }

        # Predictions under fixed candidate p's
        for name, pval in cand.items():
            resB = settlement_from_files(
                files=files,
                cfg=cfg,
                H_star=H_star,
                mode="PREDICT_LAMBDA_FROM_P",
                log10K_target=cfg.log10K_target,
                target_log10_lambda=LAMBDA_OBS_LOG10,
                p_theory=pval
            )
            row[f"log10Lambda_pred_{name}"] = resB["log10Lambda_pred"]
            row[f"deltaDex_{name}"] = resB["delta_dex_vs_obs"]

        summary_rows.append(row)

        print("\n" + "-"*108)
        print(f"[CASE {tag}]  I={resA['I_mean']:.6f}±{resA['I_std']:.6f}  "
              f"E={resA['E_mean']:.3e}±{resA['E_std']:.1e}  "
              f"dist={resA['dist_end_mean']:.3e}  p*={resA['p_star']:.6f}")

        for name in cand.keys():
            print(f"  PRED({name:8s}): log10Λ={row[f'log10Lambda_pred_{name}']:.3f}  "
                  f"Δdex={row[f'deltaDex_{name}']:+.3f}")

    df = pd.DataFrame(summary_rows)
    pd.set_option("display.max_columns", 200)
    print("\n" + "=" * 108)
    print("[ABLATIONS SUMMARY TABLE]")
    print("=" * 108)
    print(df.to_string(index=False))

# ----------------------------------------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------------------------------------
print("=" * 108)
print("DeepSeek patch runner (FIXED): tethered Hessian for E_res and stability diagnostics")
print("=" * 108)

existing = sorted(glob.glob(CSV_GLOB))
need_rerun = FORCE_RERUN
if not need_rerun:
    if len(existing) >= ENSEMBLE_SEEDS and validate_csvs(existing[:ENSEMBLE_SEEDS]):
        need_rerun = False
    else:
        need_rerun = True

if need_rerun:
    print("\nCleaning old baseline CSV/manifest...")
    for fp in glob.glob("v6_9_4g_fixed3c6_path_seed*.csv"):
        try: os.remove(fp)
        except: pass
    try:
        if os.path.exists(MANIFEST_PATH):
            os.remove(MANIFEST_PATH)
    except: pass

manifest = run_v6_9_4g_fixed3c6_full(cfg0, stop_cfg0, ensemble_seeds=ENSEMBLE_SEEDS, tag="")

m = read_manifest()
H_star = np.array(m["H_star"], dtype=float)

files = sorted(glob.glob("v6_9_4g_fixed3c6_path_seed[0-9].csv"))
if len(files) == 0:
    files = sorted(glob.glob("v6_9_4g_fixed3c6_path_seed*.csv"))

resA = settlement_from_files(
    files=files,
    cfg=cfg0,
    H_star=H_star,
    mode="IMPLY_P_FROM_LAMBDA",
    log10K_target=cfg0.log10K_target,
    target_log10_lambda=LAMBDA_OBS_LOG10,
    p_theory=None
)
print_settlement_report("BASELINE", files, H_star, cfg0, resA)

cand = candidate_p_values(cfg0)
print("\n[Candidate p transparency]")
for k,v in cand.items():
    print(f"  {k:9s} = {v:.10f}   Δ vs p* = {abs(v - resA['p_star']):.3e}")

print("\n" + "="*108)
print("[PREDICTION MODE DEMO] Predict Λ from fixed p_theory (true test; not an identity)")
print("="*108)
for name, pval in cand.items():
    resB = settlement_from_files(
        files=files,
        cfg=cfg0,
        H_star=H_star,
        mode="PREDICT_LAMBDA_FROM_P",
        log10K_target=cfg0.log10K_target,
        target_log10_lambda=LAMBDA_OBS_LOG10,
        p_theory=pval
    )
    print(f"p_theory={name:9s}={pval:.10f}  pred log10Λ={resB['log10Lambda_pred']:.6f}  Δdex={resB['delta_dex_vs_obs']:+.6f}")

if RUN_ABLATIONS:
    run_ablation_suite(cfg0, stop_cfg0)
