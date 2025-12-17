import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# Font setup (optional, avoid Chinese glyph warnings)
# ============================================================
def setup_cjk_font():
    import matplotlib
    from matplotlib import font_manager
    preferred = [
        "Noto Sans CJK SC", "Noto Sans CJK JP", "Microsoft YaHei",
        "SimHei", "PingFang SC", "WenQuanYi Zen Hei", "Arial Unicode MS"
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            matplotlib.rcParams["font.family"] = name
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    matplotlib.rcParams["axes.unicode_minus"] = False
    return False

setup_cjk_font()


# ============================================================
# Core utilities
# ============================================================
def mean_spacing_from_T(T):
    # Δ(T) ≈ 2π / log(T/2π)
    T = np.asarray(T, dtype=np.float64)
    denom = np.log(np.maximum(T / (2 * np.pi), 1.0000001))
    denom = np.where(denom > 0.1, denom, 0.1)
    return 2.0 * np.pi / denom


def r_stat_from_unfolded(unfolded):
    s = np.asarray(unfolded, dtype=np.float64)
    s = s[np.isfinite(s)]
    if len(s) < 3:
        return np.array([])
    r = np.minimum(s[1:], s[:-1]) / np.maximum(s[1:], s[:-1])
    r = r[np.isfinite(r)]
    return np.clip(r, 0.0, 1.0)


def unfolded_positions_from_spacings(s, x0=0.0):
    s = np.asarray(s, dtype=np.float64)
    s = s[np.isfinite(s)]
    x = np.empty(len(s) + 1, dtype=np.float64)
    x[0] = float(x0)
    x[1:] = x0 + np.cumsum(s)
    return x


# ============================================================
# Rigidity: Σ^2(L) and Δ3(L)
# ============================================================
def number_variance_sigma2(x, L_values, n_windows=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    x.sort()
    if len(x) < 300:
        return np.full_like(L_values, np.nan, dtype=np.float64)

    xmin, xmax = x[0], x[-1]
    sig2 = []
    for L in L_values:
        if xmax - xmin <= L + 1:
            sig2.append(np.nan)
            continue
        starts = rng.uniform(xmin, xmax - L, size=n_windows)
        left = np.searchsorted(x, starts, side="left")
        right = np.searchsorted(x, starts + L, side="right")
        counts = (right - left).astype(np.float64)
        sig2.append(float(np.var(counts, ddof=1)))
    return np.array(sig2, dtype=np.float64)


def delta3_rigidity(x, L_values, n_windows=1200, grid_points=200, seed=1):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    x.sort()
    if len(x) < 500:
        return np.full_like(L_values, np.nan, dtype=np.float64)

    xmin, xmax = x[0], x[-1]
    out = []
    for L in L_values:
        if xmax - xmin <= L + 1:
            out.append(np.nan)
            continue
        starts = rng.uniform(xmin, xmax - L, size=n_windows)
        vals = []
        for s0 in starts:
            t = np.linspace(s0, s0 + L, grid_points)
            Nt = np.searchsorted(x, t, side="right").astype(np.float64)

            A = np.vstack([t, np.ones_like(t)]).T
            a, b = np.linalg.lstsq(A, Nt, rcond=None)[0]

            resid2 = (Nt - (a * t + b))**2
            integral = np.trapz(resid2, t)
            vals.append(integral / L)
        out.append(float(np.mean(vals)))
    return np.array(out, dtype=np.float64)


# ============================================================
# Short-range RMT reference distributions: spacing s and ratio r
# ============================================================
def pdf_spacing_poisson(s):
    s = np.asarray(s, dtype=np.float64)
    return np.exp(-np.maximum(s, 0.0))

def pdf_spacing_goe_wigner(s):
    # GOE Wigner: p(s)=(pi/2)s exp(-pi s^2/4)
    s = np.asarray(s, dtype=np.float64)
    s = np.maximum(s, 0.0)
    return (np.pi/2.0) * s * np.exp(-np.pi * s*s / 4.0)

def pdf_spacing_gue_wigner(s):
    # GUE Wigner: p(s)=(32/pi^2)s^2 exp(-4 s^2/pi)
    s = np.asarray(s, dtype=np.float64)
    s = np.maximum(s, 0.0)
    return (32.0/(np.pi**2)) * (s**2) * np.exp(-4.0*(s**2)/np.pi)

def pdf_r_poisson(r):
    # Poisson r: p(r)=2/(1+r)^2 on [0,1]
    r = np.asarray(r, dtype=np.float64)
    r = np.clip(r, 0.0, 1.0)
    return 2.0 / (1.0 + r)**2

def pdf_r_atas_beta(r, beta):
    # Atas family (unnormalized): (r+r^2)^β / (1+r+r^2)^(1+3β/2)
    r = np.asarray(r, dtype=np.float64)
    r = np.clip(r, 0.0, 1.0)
    num = (r + r*r)**beta
    den = (1.0 + r + r*r)**(1.0 + 1.5*beta)
    return num / np.maximum(den, 1e-300)

def build_normalized_pdf(pdf_raw_func, grid, *args):
    raw = pdf_raw_func(grid, *args)
    raw = np.maximum(raw, 0.0)
    Z = np.trapz(raw, grid)
    if Z <= 0:
        return np.zeros_like(grid)
    return raw / Z

def empirical_cdf(samples, grid):
    samples = np.asarray(samples, dtype=np.float64)
    samples = samples[np.isfinite(samples)]
    samples.sort()
    return np.searchsorted(samples, grid, side="right") / max(len(samples), 1)

def ks_distance_to_pdf(samples, grid, pdf):
    pdf = np.asarray(pdf, dtype=np.float64)
    pdf = np.maximum(pdf, 0.0)
    Z = np.trapz(pdf, grid)
    pdf = pdf / max(Z, 1e-12)
    cdf = np.cumsum((pdf[:-1] + pdf[1:]) * 0.5 * np.diff(grid))
    cdf = np.concatenate([[0.0], cdf])
    cdf = np.clip(cdf, 0.0, 1.0)
    F_emp = empirical_cdf(samples, grid)
    return float(np.max(np.abs(F_emp - cdf))), cdf

def w1_distance(samples, theory_samples):
    samples = np.asarray(samples, dtype=np.float64)
    samples = samples[np.isfinite(samples)]
    theory_samples = np.asarray(theory_samples, dtype=np.float64)
    theory_samples = theory_samples[np.isfinite(theory_samples)]
    if len(samples) < 10 or len(theory_samples) < 10:
        return float("nan")
    return float(stats.wasserstein_distance(samples, theory_samples))

def sample_from_grid_cdf(grid, cdf, size, rng):
    u = rng.random(size)
    return np.interp(u, cdf, grid)


def short_range_rmt_classification(unfolded, rvals, seed=0, n_theory=120000):
    rng = np.random.default_rng(seed)

    # ---- spacing s ----
    s = np.asarray(unfolded, dtype=np.float64)
    s = s[np.isfinite(s)]
    s = s[s >= 0]
    smax = np.percentile(s, 99.5) * 1.6 + 1e-6
    s_grid = np.linspace(0.0, max(smax, 6.0), 5000)

    pdf_s = {
        "Poisson": pdf_spacing_poisson(s_grid),
        "GOE": pdf_spacing_goe_wigner(s_grid),
        "GUE": pdf_spacing_gue_wigner(s_grid),
    }
    s_metrics = {}
    s_cdfs = {}
    for k in ["Poisson", "GOE", "GUE"]:
        ks, cdf = ks_distance_to_pdf(s, s_grid, pdf_s[k])
        s_cdfs[k] = cdf
        samp = sample_from_grid_cdf(s_grid, cdf, n_theory, rng)
        w1 = w1_distance(s, samp)
        s_metrics[k] = {"KS": ks, "W1": w1}
    best_s = sorted([(k, s_metrics[k]["KS"], s_metrics[k]["W1"]) for k in s_metrics],
                    key=lambda x: (x[1], x[2]))[0][0]
    s_metrics["best_fit"] = best_s

    # ---- r-stat ----
    r = np.asarray(rvals, dtype=np.float64)
    r = r[np.isfinite(r)]
    r_grid = np.linspace(0.0, 1.0, 4000)

    pdf_r = {
        "Poisson": pdf_r_poisson(r_grid),
        "GOE": build_normalized_pdf(pdf_r_atas_beta, r_grid, 1),
        "GUE": build_normalized_pdf(pdf_r_atas_beta, r_grid, 2),
    }
    r_metrics = {}
    r_cdfs = {}
    for k in ["Poisson", "GOE", "GUE"]:
        ks, cdf = ks_distance_to_pdf(r, r_grid, pdf_r[k])
        r_cdfs[k] = cdf
        samp = sample_from_grid_cdf(r_grid, cdf, n_theory, rng)
        w1 = w1_distance(r, samp)
        r_metrics[k] = {"KS": ks, "W1": w1}
    best_r = sorted([(k, r_metrics[k]["KS"], r_metrics[k]["W1"]) for k in r_metrics],
                    key=lambda x: (x[1], x[2]))[0][0]
    r_metrics["best_fit"] = best_r

    return {
        "s_grid": s_grid, "pdf_s": pdf_s, "s_metrics": s_metrics,
        "r_grid": r_grid, "pdf_r": pdf_r, "r_metrics": r_metrics
    }


# ============================================================
# GAR-V6 macro smooth model + LOCKED baseline
# ============================================================
class GARV6Smooth:
    def __init__(self):
        self.params = {"S": 1.035, "beta": -1.500, "phi": np.pi/2}
        self.ln2pi = np.log(2*np.pi)

    def gamma_smooth(self, n):
        n = np.asarray(n, dtype=np.float64)
        n = np.maximum(n, 3.0)
        S = self.params["S"]
        beta = self.params["beta"]
        phi = self.params["phi"]

        denom = np.log(n) - self.ln2pi - 1.0
        denom = np.where(denom > 0.1, denom, 0.1)
        main = (2.0*np.pi*n) / denom
        slow = beta * np.log(np.log(n))
        return S * (main + slow + phi)


class GARV6Locked(GARV6Smooth):
    def __init__(self):
        super().__init__()
        self.A2 = 0.800
        self.omega = 1.618

    def gamma(self, n):
        n = np.asarray(n, dtype=np.float64)
        base = self.gamma_smooth(n)
        return base + self.params["S"] * (self.A2 * np.sin(self.omega * n))


# ============================================================
# Folding: build gamma_n from unfolded positions x_n
# ============================================================
def fold_unfolded_positions_to_gamma(model_smooth: GARV6Smooth, start_n: int, x_unfolded: np.ndarray):
    x = np.asarray(x_unfolded, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        raise ValueError("unfolded positions too short")

    M = len(x)
    n = np.arange(start_n, start_n + M, dtype=np.int64)

    gamma = np.empty(M, dtype=np.float64)
    gamma[0] = float(model_smooth.gamma_smooth(np.array([start_n], dtype=np.float64))[0])

    s = np.diff(x)
    for i in range(M - 1):
        Delta = mean_spacing_from_T(gamma[i])
        gamma[i+1] = gamma[i] + Delta * s[i]
    return n, gamma


# ============================================================
# Micro generators: IID_WIGNER / RM_GUE+GOE / CUE / PRIME_PHASE
# ============================================================
def sample_gue_wigner_spacings(size, rng):
    # Rejection from Gamma(3,1/3)
    from math import pi
    def p(s):
        return (32.0/(pi**2)) * s*s * np.exp(-4.0*s*s/pi)
    def q(s):
        return (27.0/2.0) * s*s * np.exp(-3.0*s)
    Menv = 3.5
    out = []
    batch = max(8000, size * 4)
    while len(out) < size:
        s = rng.gamma(shape=3.0, scale=1.0/3.0, size=batch)
        u = rng.random(batch)
        acc = u < (p(s) / (Menv*q(s) + 1e-18))
        out.extend(list(s[acc]))
    return np.array(out[:size], dtype=np.float64)

def gen_unfolded_iid_wigner(M, seed=0):
    rng = np.random.default_rng(seed)
    s = sample_gue_wigner_spacings(M - 1, rng)
    s = s / np.mean(s)
    return unfolded_positions_from_spacings(s, x0=0.0)


def gen_unfolded_from_random_matrix(
    M,
    kind="GUE",
    dim=600,
    ensembles=None,
    seed=0,
    poly_deg=7,
    take_bulk_frac=0.8,
):
    """
    Robust RM generator:
      - stitch multiple medium matrices until >= M bulk unfolded levels
      - unfold each by polynomial fit rank(E)~P(z)
      - concatenate with offsets and global normalize mean spacing=1
    """
    rng = np.random.default_rng(seed)
    kind = kind.upper()

    if ensembles is None:
        bulk_per = max(80, int(dim * take_bulk_frac))
        ensembles = int(np.ceil(M / bulk_per)) + 2

    blocks = []
    offset = 0.0

    for _ in range(ensembles):
        if kind == "GUE":
            A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
            H = (A + A.conj().T) / np.sqrt(2.0 * dim)
            evals = np.linalg.eigvalsh(H).real
        elif kind == "GOE":
            A = rng.normal(size=(dim, dim))
            H = (A + A.T) / np.sqrt(2.0 * dim)
            evals = np.linalg.eigvalsh(H).real
        else:
            raise ValueError("kind must be 'GOE' or 'GUE'")

        evals.sort()

        m = int(dim * take_bulk_frac)
        start = (dim - m) // 2
        bulk = evals[start:start + m].astype(np.float64)

        rank = np.arange(1, m + 1, dtype=np.float64)
        mu, sig = np.mean(bulk), np.std(bulk)
        z = (bulk - mu) / max(sig, 1e-12)

        deg = int(min(poly_deg, m - 1))
        coeff = np.polyfit(z, rank, deg)
        P = np.poly1d(coeff)
        x = P(z).astype(np.float64)
        x.sort()

        x = x - x.min()
        sp = np.diff(x)
        sp = sp[sp > 1e-12]
        mean_sp = np.mean(sp) if len(sp) else 1.0
        x = x / mean_sp

        x = x + offset
        offset = x[-1] + 1.5
        blocks.append(x)

        total = sum(len(b) for b in blocks)
        if total >= M:
            break

    x_all = np.concatenate(blocks)
    x_all.sort()

    if len(x_all) < M:
        raise RuntimeError(
            f"{kind} RM produced only {len(x_all)} levels < M={M}. "
            f"Increase ensembles or dim or take_bulk_frac."
        )

    x_all = x_all[:M]
    sp = np.diff(x_all)
    sp = sp[sp > 1e-12]
    mean_sp = np.mean(sp) if len(sp) else 1.0
    x_all = x_all / mean_sp
    x_all = x_all - x_all[0]
    return x_all


def gen_unfolded_from_cue(M, dim=1200, seed=0):
    """
    Haar unitary (CUE): eigenangles give sine-kernel local statistics.
    Natural unfolding: x = (N/2π)*theta, then normalize mean spacing.
    """
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    Q, R = np.linalg.qr(Z)
    diag = np.diag(R)
    ph = diag / np.abs(diag)
    Q = Q * ph.conj()

    eigvals = np.linalg.eigvals(Q)
    theta = np.mod(np.angle(eigvals), 2*np.pi)
    theta.sort()

    N = len(theta)
    x = (N / (2*np.pi)) * theta
    x = x - x.min()

    sp = np.diff(x)
    sp = sp[sp > 1e-12]
    x = x / (np.mean(sp) if len(sp) else 1.0)

    if len(x) < M:
        raise RuntimeError(f"CUE produced only {len(x)} levels < M={M}. Increase dim.")
    return x[:M]


def primes_up_to(N):
    sieve = np.ones(N + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(N**0.5) + 1):
        if sieve[p]:
            sieve[p*p:N+1:p] = False
    return np.nonzero(sieve)[0]


def gen_unfolded_prime_phase(M, seed=0, pmax=2000, strength=0.35, ar_rho=0.98):
    """
    Arithmetic-like correlated generator on unfolded spacings:
      z_n = Σ a_p cos(2π n log p / scale + φ_p) + AR(1) noise
      s_n = exp(z_n), normalize mean to 1
    """
    rng = np.random.default_rng(seed)
    plist = primes_up_to(pmax)
    logs = np.log(plist.astype(np.float64))
    scale = np.max(logs) * 20.0

    a = 1.0 / np.sqrt(plist.astype(np.float64))
    a = a / np.linalg.norm(a)
    a = a * strength
    phi = rng.uniform(0, 2*np.pi, size=len(plist))

    n = np.arange(M - 1, dtype=np.float64)
    arg = (2*np.pi * np.outer(n, logs) / scale) + phi
    field = np.cos(arg) @ a

    eps = rng.normal(size=M - 1)
    ar = np.empty(M - 1, dtype=np.float64)
    ar[0] = eps[0]
    sigma = np.sqrt(max(1 - ar_rho**2, 1e-12))
    for i in range(1, M - 1):
        ar[i] = ar_rho * ar[i-1] + sigma * eps[i]
    ar = 0.25 * ar

    z = field + ar
    s = np.exp(z)
    s = s / np.mean(s)
    return unfolded_positions_from_spacings(s, x0=0.0)


# ============================================================
# Evaluation & plotting
# ============================================================
def evaluate_gamma(gamma):
    gamma = np.asarray(gamma, dtype=np.float64)
    intervals = np.diff(gamma)
    theo = mean_spacing_from_T(gamma[1:])
    unfolded = intervals / theo
    r = r_stat_from_unfolded(unfolded)
    x = unfolded_positions_from_spacings(unfolded, x0=0.0)
    return {
        "gamma": gamma,
        "intervals": intervals,
        "unfolded": unfolded,
        "r": r,
        "x_unfolded": x,
        "mean_unfolded": float(np.mean(unfolded)),
        "std_unfolded": float(np.std(unfolded)),
        "mean_r": float(np.mean(r)) if len(r) else float("nan"),
        "std_r": float(np.std(r)) if len(r) else float("nan"),
    }


def plot_short_range_overlay(result, sr, save_path="gar_v6_short_range.png"):
    fig = plt.figure(figsize=(14, 6))

    ax1 = plt.subplot(1, 2, 1)
    ax1.hist(result["unfolded"], bins=90, density=True, alpha=0.6, label="model")
    ax1.plot(sr["s_grid"], sr["pdf_s"]["Poisson"], "--", label="Poisson")
    ax1.plot(sr["s_grid"], sr["pdf_s"]["GOE"], "--", label="GOE (Wigner)")
    ax1.plot(sr["s_grid"], sr["pdf_s"]["GUE"], "--", label="GUE (Wigner)")
    ax1.set_title(f"Unfolded spacing s | best_fit={sr['s_metrics']['best_fit']}")
    ax1.set_xlabel("s")
    ax1.set_ylabel("density")
    ax1.grid(True, alpha=0.25)
    ax1.legend()

    ax2 = plt.subplot(1, 2, 2)
    ax2.hist(result["r"], bins=90, density=True, alpha=0.6, label="model")
    ax2.plot(sr["r_grid"], sr["pdf_r"]["Poisson"], "--", label="Poisson (Atas)")
    ax2.plot(sr["r_grid"], sr["pdf_r"]["GOE"], "--", label="GOE (Atas)")
    ax2.plot(sr["r_grid"], sr["pdf_r"]["GUE"], "--", label="GUE (Atas)")
    ax2.set_title(f"r-stat distribution | best_fit={sr['r_metrics']['best_fit']}")
    ax2.set_xlabel("r")
    ax2.set_ylabel("density")
    ax2.grid(True, alpha=0.25)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return save_path


def plot_rigidity(L_values, curves, save_path="gar_v6_rigidity.png"):
    fig = plt.figure(figsize=(14, 6))

    ax1 = plt.subplot(1, 2, 1)
    for name, d in curves.items():
        ax1.plot(L_values, d["sigma2"], label=name)
    ax1.set_title(r"Number variance $\Sigma^2(L)$")
    ax1.set_xlabel("L")
    ax1.set_ylabel(r"$\Sigma^2(L)$")
    ax1.grid(True, alpha=0.25)
    ax1.legend()

    ax2 = plt.subplot(1, 2, 2)
    for name, d in curves.items():
        ax2.plot(L_values, d["delta3"], label=name)
    ax2.set_title(r"Spectral rigidity $\Delta_3(L)$")
    ax2.set_xlabel("L")
    ax2.set_ylabel(r"$\Delta_3(L)$")
    ax2.grid(True, alpha=0.25)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return save_path


# ============================================================
# Main
# ============================================================
def main():
    # =========================================================
    # Choose ONE mode:
    #   "RM_GUE"     (1)  Random-matrix GUE point process (unfolded), then fold to gamma
    #   "CUE_SINE"   (2)  CUE eigenangles (sine-kernel proxy), then fold
    #   "PRIME_PHASE"(3)  prime-like correlated phase field (unfolded), then fold
    #   "IID_WIGNER"      i.i.d Wigner-GUE spacings (short-range only), then fold
    #   "LOCKED"          deterministic phase-locked on gamma axis
    # =========================================================
    MODE = "RM_GUE"

    start_n = 1000
    M = 15000

    # Reference RM settings (if slow, reduce dim to 450; quality decreases a bit)
    REF_DIM = 600
    REF_BULK = 0.8
    REF_POLY = 7

    print("=" * 80)
    print("GAR-V6 FINAL PLATFORM: short-range RMT + rigidity Σ^2(L), Δ3(L)")
    print("=" * 80)
    print(f"MODE={MODE} | start_n={start_n} | M={M}")

    smooth = GARV6Smooth()

    # ---- Generate gamma_n from chosen mode ----
    if MODE == "LOCKED":
        locked = GARV6Locked()
        n = np.arange(start_n, start_n + M, dtype=np.int64)
        gamma = locked.gamma(n)
    else:
        if MODE == "RM_GUE":
            x = gen_unfolded_from_random_matrix(
                M=M, kind="GUE", dim=REF_DIM, seed=7,
                poly_deg=REF_POLY, take_bulk_frac=REF_BULK
            )
        elif MODE == "CUE_SINE":
            x = gen_unfolded_from_cue(M=M, dim=1200, seed=9)
        elif MODE == "PRIME_PHASE":
            x = gen_unfolded_prime_phase(M=M, seed=11, pmax=2000, strength=0.35, ar_rho=0.98)
        elif MODE == "IID_WIGNER":
            x = gen_unfolded_iid_wigner(M=M, seed=42)
        else:
            raise ValueError("Unknown MODE")

        n, gamma = fold_unfolded_positions_to_gamma(smooth, start_n, x)

    # ---- Evaluate ----
    res = evaluate_gamma(gamma)
    print("\n[Model summary]")
    print(f"unfolded mean: {res['mean_unfolded']:.6f} | std: {res['std_unfolded']:.6f}")
    print(f"mean r      : {res['mean_r']:.6f} | std: {res['std_r']:.6f}")

    # ---- Short-range classification ----
    sr = short_range_rmt_classification(res["unfolded"], res["r"], seed=0, n_theory=120000)
    print("\n[Short-range RMT classification]")
    for k in ["Poisson", "GOE", "GUE"]:
        ms = sr["s_metrics"][k]
        mr = sr["r_metrics"][k]
        print(f"{k:7s} | spacing: KS={ms['KS']:.6f}, W1={ms['W1']:.6f} | r: KS={mr['KS']:.6f}, W1={mr['W1']:.6f}")
    print(f"-> best_fit(spacing)={sr['s_metrics']['best_fit']} | best_fit(r)={sr['r_metrics']['best_fit']}")

    out_short = plot_short_range_overlay(res, sr, save_path=f"gar_v6_short_range_{MODE}.png")
    print(f"\nSaved short-range plot -> {out_short}")

    # ---- Rigidity analysis ----
    L_values = np.array([1, 2, 3, 5, 7, 10, 14, 20, 28, 40, 55], dtype=np.float64)

    # Poisson reference
    rng = np.random.default_rng(123)
    s_p = rng.exponential(scale=1.0, size=M - 1)
    x_poisson = unfolded_positions_from_spacings(s_p, x0=0.0)

    # RM references (true GUE and true GOE), unfolded directly in x
    x_ref_gue = gen_unfolded_from_random_matrix(
        M=M, kind="GUE", dim=REF_DIM, seed=101,
        poly_deg=REF_POLY, take_bulk_frac=REF_BULK
    )
    x_ref_goe = gen_unfolded_from_random_matrix(
        M=M, kind="GOE", dim=REF_DIM, seed=202,
        poly_deg=REF_POLY, take_bulk_frac=REF_BULK
    )

    curves = {
        f"{MODE} (model)": {
            "sigma2": number_variance_sigma2(res["x_unfolded"], L_values, n_windows=4000, seed=0),
            "delta3": delta3_rigidity(res["x_unfolded"], L_values, n_windows=1200, grid_points=200, seed=1),
        },
        "Poisson (ref)": {
            "sigma2": number_variance_sigma2(x_poisson, L_values, n_windows=4000, seed=2),
            "delta3": delta3_rigidity(x_poisson, L_values, n_windows=1200, grid_points=200, seed=3),
        },
        "RM-GUE (ref)": {
            "sigma2": number_variance_sigma2(x_ref_gue, L_values, n_windows=4000, seed=4),
            "delta3": delta3_rigidity(x_ref_gue, L_values, n_windows=1200, grid_points=200, seed=5),
        },
        "RM-GOE (ref)": {
            "sigma2": number_variance_sigma2(x_ref_goe, L_values, n_windows=4000, seed=6),
            "delta3": delta3_rigidity(x_ref_goe, L_values, n_windows=1200, grid_points=200, seed=7),
        },
    }

    out_rig = plot_rigidity(L_values, curves, save_path=f"gar_v6_rigidity_{MODE}.png")
    print(f"\nSaved rigidity plot -> {out_rig}")

    # ---- Compact summary at L=10 and L=40 ----
    def pick(arr, L):
        idx = np.where(L_values == L)[0]
        if len(idx) == 0:
            return np.nan
        return float(arr[idx[0]])

    print("\n" + "-" * 80)
    print("Rigidity summary (selected L)")
    print("-" * 80)
    for name, d in curves.items():
        s10 = pick(d["sigma2"], 10); d10 = pick(d["delta3"], 10)
        s40 = pick(d["sigma2"], 40); d40 = pick(d["delta3"], 40)
        print(f"{name:14s} | L=10: Σ^2={s10:.4f}, Δ3={d10:.4f} | L=40: Σ^2={s40:.4f}, Δ3={d40:.4f}")

    print("\nInterpretation guardrail:")
    print("  - 若 best_fit(r)=GUE 但 Σ^2/Δ3 明显偏离 RM-GUE(ref)：只能说短程 GUE-like，不能说全尺度一致。")
    print("  - 若 Σ^2/Δ3 也贴近 RM-GUE(ref)：才接近“点过程同类”（中程刚性一致）。")


if __name__ == "__main__":
    main()

