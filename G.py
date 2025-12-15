# === P6 LOGIC-CHAIN (SINGLE CELL) ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# 0) Global config
# ---------------------------
CFG = dict(
    dt=0.001,
    T=400,
    kappa=1.0,
    noise=0.1,
    deltaK=0.1,
    eps0=1e-3,
    eps_scan=np.array([1e-4,3e-4,8e-4,2.2e-3,6.3e-3,1.77e-2,5e-2], float),
    seeds=list(range(20)),
)

def cov(x):
    x=np.asarray(x,float); m=float(np.mean(x)); s=float(np.std(x,ddof=1)) if len(x)>1 else 0.0
    return np.inf if abs(m)<1e-300 else s/m

# ---------------------------
# 1) Core builders
# ---------------------------
def build_laplacian(N, dim=1):
    Nn = N**dim
    if dim==1:
        D=np.zeros((N-1,N))
        for i in range(N-1):
            D[i,i]=-1.0; D[i,i+1]=1.0
        return D.T@D, Nn
    if dim==2:
        L=np.zeros((Nn,Nn))
        for i in range(Nn):
            r,c=i//N,i%N
            nb=[]
            if c<N-1: nb.append(i+1)
            if c>0:   nb.append(i-1)
            if r<N-1: nb.append(i+N)
            if r>0:   nb.append(i-N)
            L[i,i]=len(nb)
            for j in nb: L[i,j]=-1.0
        return L, Nn
    if dim==3:
        L=np.zeros((Nn,Nn)); Np=N*N
        for i in range(Nn):
            x=i%N; y=(i//N)%N; z=i//Np
            nb=[]
            if x<N-1: nb.append(i+1)
            if x>0:   nb.append(i-1)
            if y<N-1: nb.append(i+N)
            if y>0:   nb.append(i-N)
            if z<N-1: nb.append(i+Np)
            if z>0:   nb.append(i-Np)
            L[i,i]=len(nb)
            for j in nb: L[i,j]=-1.0
        return L, Nn
    raise NotImplementedError

def build_Delta(L, def_type, seed=0):
    np.random.seed(seed)
    n=L.shape[0]
    if def_type=="Lap_Like":
        R=np.random.randn(n,n)*0.05; R=0.5*(R+R.T)
        D=L+R
        return D-D.mean()
    if def_type=="Diag_Local":
        D=np.zeros((n,n))
        k=max(1,int(0.1*n))
        idx=np.random.choice(n,k,replace=False)
        for i in idx: D[i,i]=np.random.randn()
        return D - np.trace(D)/n*np.eye(n)
    if def_type=="Random_Sparse":
        R=np.random.randn(n,n); R=0.5*(R+R.T)
        ones=np.ones((n,1))
        return R - (R@ones)@ones.T/n
    if def_type=="Low_Rank":
        r=max(1,int(np.sqrt(n)//2))
        U=np.random.randn(n,r); D=U@U.T; D=0.5*(D+D.T)
        return D-D.mean()
    raise ValueError("unknown def_type")

def build_B(L, eps, Delta):
    return L + eps*Delta

# ---------------------------
# 2) Dynamics + residual/invariants
# ---------------------------
def simulate_u(B, T=400, dt=0.001, kappa=1.0, noise=0.1, seed=0):
    np.random.seed(seed)
    n=B.shape[0]
    u=np.random.randn(n); out=[u.copy()]
    for _ in range(T-1):
        eta=noise*np.sqrt(dt)*np.random.randn(n)
        u = u + dt*(-kappa*(B@u) + eta)
        out.append(u.copy())
    return np.array(out)

def chi_ts(L,B,u_ts):
    return ((L-B)@u_ts.T).T

def invariants(chi, dt):
    H=np.sum(chi**2,axis=1)
    Phi=np.sum(np.abs(chi),axis=1)
    logH=np.log(H+1e-300); logP=np.log(Phi+1e-300)
    dlogH=np.gradient(logH,dt); dlogP=np.gradient(logP,dt)
    dlogP[np.abs(dlogP)<1e-15]=1e-15
    K=dlogH/dlogP
    Pdot=np.gradient(Phi,dt)
    F=H/(Phi*np.abs(Pdot)+1e-300)
    return H,Phi,Pdot,K,F

def Ward_stat(L,B,u_ts,chi):
    Bu=(B@u_ts.T).T
    num=np.linalg.norm(chi,axis=1)
    den=np.linalg.norm(Bu,axis=1)+1e-300
    return float(np.median(num/den))

def basin_mask(H,Phi,Pdot,K, Kt=2.0, deltaK=0.1):
    return (np.abs(K-Kt)<deltaK) & (H>1e-10) & (Phi>1e-10) & (np.abs(Pdot)>1e-8)

def robust_Kfit(H,Phi,mask,trim=0.2):
    x=np.log(Phi+1e-300)[mask]
    y=np.log(H+1e-300)[mask]
    if len(x)<6: return np.nan
    s1,b1,r1,_,_=linregress(x,y)
    res=np.abs(y-(b1+s1*x))
    k=int((1-trim)*len(res))
    idx=np.argsort(res)[:max(4,k)]
    s2,_,_,_,_=linregress(x[idx],y[idx])
    return float(s2)

def run_one(L, Delta, eps, seed_u):
    B = build_B(L, eps, Delta)
    u = simulate_u(B, T=CFG["T"], dt=CFG["dt"], seed=seed_u,
                   kappa=CFG["kappa"], noise=CFG["noise"])
    chi = chi_ts(L,B,u)
    H,Phi,Pdot,K,F = invariants(chi, CFG["dt"])
    m = basin_mask(H,Phi,Pdot,K, deltaK=CFG["deltaK"])
    if not np.any(m): 
        return None
    return dict(
        Fbar=float(np.median(F[m])),
        Ward=Ward_stat(L,B,u,chi),
        Kfit=robust_Kfit(H,Phi,m),
        basin_ct=int(np.sum(m))
    )

# ---------------------------
# 3) Spectral factor Zs = 1/effrank(Δ)
# ---------------------------
def eigvals_sym(A):
    A=0.5*(A+A.T)
    return np.linalg.eigvalsh(A)

def Zs_effrank(Delta):
    w=np.abs(eigvals_sym(Delta))
    s=w/(np.sum(w)+1e-300)
    H=-np.sum(s*np.log(s+1e-300))
    r_eff=float(np.exp(H))
    return 1.0/(r_eff+1e-300)

def fit_no_intercept(Zc, Zs, F):
    X=np.column_stack([Zc,Zs]).astype(float)
    y=np.asarray(F,float)
    beta, *_ = np.linalg.lstsq(X,y,rcond=None)
    yhat=X@beta
    R2_unc = 1.0 - float(np.sum((y-yhat)**2))/float(np.sum(y**2)+1e-300)
    n=len(y); k=2
    sse=float(np.sum((y-yhat)**2))
    sst0=float(np.sum(y**2)+1e-300)
    Adj = 1.0 - (sse/max(1,n-k))/((sst0/max(1,n))+1e-300)
    return float(beta[0]), float(beta[1]), float(R2_unc), float(Adj)

# ---------------------------
# 4) Families
# ---------------------------
FAMILIES = [
    dict(name="1D Lap_Like",      def_type="Lap_Like",      dim=1, N=32),
    dict(name="2D Diag_Local",    def_type="Diag_Local",    dim=2, N=8),
    dict(name="2D Random_Sparse", def_type="Random_Sparse", dim=2, N=8),
    dict(name="3D Low_Rank",      def_type="Low_Rank",      dim=3, N=6),
]

print("=== P6 LOGIC-CHAIN SINGLE CELL ===")
print("CFG:", CFG)
print("Families:", [f["name"] for f in FAMILIES])

# ============================================================
# I) L0: Precision closure test (L=B => Ward ~ machine epsilon)
# ============================================================
L0_L,_ = build_laplacian(32,1)
L0_B = L0_L.copy()
u0 = simulate_u(L0_B, T=CFG["T"], dt=CFG["dt"], seed=0, kappa=CFG["kappa"], noise=CFG["noise"])
chi0 = chi_ts(L0_L,L0_B,u0)
W0 = Ward_stat(L0_L,L0_B,u0,chi0)
print("\n--- I) L0 PRECISION CLOSURE ---")
print(f"Ward_rel (L=B) = {W0:.3e} | max||chi|| = {np.max(np.linalg.norm(chi0,axis=1)):.3e}")

# ============================================================
# II) K=2 universality: pooled across families/eps/seeds
# ============================================================
rows=[]
for fam in FAMILIES:
    L,_=build_laplacian(fam["N"], fam["dim"])
    for s in CFG["seeds"]:
        Delta = build_Delta(L, fam["def_type"], seed=s)
        for eps in CFG["eps_scan"]:
            r = run_one(L, Delta, eps, seed_u=s)
            if r is None: 
                continue
            rows.append(dict(family=fam["name"], def_type=fam["def_type"], dim=fam["dim"],
                             N_nodes=L.shape[0], seed=s, eps=eps, **r))

df = pd.DataFrame(rows)
k = df["Kfit"].dropna().to_numpy()

print("\n--- II) K=2 KINEMATIC UNIVERSALITY (pooled) ---")
print("K_fit pooled mean =", float(np.mean(k)))
print("K_fit pooled std  =", float(np.std(k, ddof=1)))
print("K_fit pooled min/max =", float(np.min(k)), "/", float(np.max(k)))

# ============================================================
# III) d -> 4 interface (plug your DAG estimator here)
# ============================================================
def estimate_alexandrov_dimension(M=1000, seed=0):
    # TODO: replace with your real estimator
    return dict(d_mean=np.nan, ci_low=np.nan, ci_high=np.nan)

def microcausality_metrics(M=1000, seed=0):
    # TODO: replace with your real metrics
    return dict(strict_leak=np.nan, c_eff=np.nan)

print("\n--- III) d -> 4 GEOMETRY FLOW (INTERFACE) ---")
for M in [1000, 1800]:
    dres = estimate_alexandrov_dimension(M=M, seed=0)
    cres = microcausality_metrics(M=M, seed=0)
    print(f"M={M} | d={dres['d_mean']} (CI [{dres['ci_low']},{dres['ci_high']}]) | leak={cres['strict_leak']} | c_eff={cres['c_eff']}")

# ============================================================
# IV) Topological locking at eps0 (per-family OLS, Zs=effrank)
# ============================================================
amp_rows=[]
for fam in FAMILIES:
    L,_=build_laplacian(fam["N"], fam["dim"])
    Zc_list=[]; Zs_list=[]; F_list=[]
    for s in CFG["seeds"]:
        Delta=build_Delta(L,fam["def_type"],seed=s)
        r = run_one(L, Delta, CFG["eps0"], seed_u=s)
        if r is None: 
            continue
        Fbar, W = r["Fbar"], r["Ward"]
        Zc_list.append(1.0/(W+1e-300))
        Zs_list.append(Zs_effrank(Delta))
        F_list.append(Fbar)
    Zc_arr=np.array(Zc_list); Zs_arr=np.array(Zs_list); F_arr=np.array(F_list)
    if len(F_arr) < 6:
        amp_rows.append(dict(family=fam["name"], AdjR2=np.nan, R2_unc=np.nan, C2=np.nan, C3=np.nan, n=len(F_arr)))
        continue
    C2,C3,R2u,Adj=fit_no_intercept(Zc_arr,Zs_arr,F_arr)
    amp_rows.append(dict(family=fam["name"], AdjR2=Adj, R2_unc=R2u, C2=C2, C3=C3, n=len(F_arr)))

amp_df=pd.DataFrame(amp_rows).sort_values("AdjR2", ascending=False)
print("\n--- IV) TOPOLOGICAL LOCKING @ eps0 (per-family OLS, Zs=effrank) ---")
print(amp_df.to_string(index=False))

# ============================================================
# V) Order-parameter stability: alpha_raw ~ 0 across eps (median over seeds)
# ============================================================
dyn_rows=[]
for fam in FAMILIES:
    sub = df[df["family"]==fam["name"]].copy()
    g = sub.groupby("eps").agg(Fbar=("Fbar","median"), Ward=("Ward","median")).reset_index()
    if len(g) < 3: 
        continue
    g0 = g.iloc[(np.abs(g["eps"].to_numpy()-CFG["eps0"])).argmin()]
    F0=float(g0["Fbar"]); W0=float(g0["Ward"])
    x=np.log(g["Ward"].to_numpy()/(W0+1e-300)+1e-300)
    y=np.log(g["Fbar"].to_numpy()/(F0+1e-300)+1e-300)
    s1,b1,r1,_,_=linregress(x,y)
    alpha=float(-s1); R2=float(r1**2)
    dyn_rows.append(dict(family=fam["name"], alpha_raw=alpha, R2_raw=R2, CoV_F=cov(g["Fbar"].to_numpy())))

dyn_df=pd.DataFrame(dyn_rows)
print("\n--- V) ORDER PARAMETER STABILITY (alpha ~ 0, CoV small) ---")
print(dyn_df.to_string(index=False))

# ============================================================
# VI) One-panel summary plots
# ============================================================
fig = plt.figure(figsize=(10,6))

plt.subplot(2,2,1)
plt.hist(df["Kfit"].dropna().to_numpy(), bins=30)
plt.title("K_fit pooled distribution")
plt.xlabel("K_fit"); plt.ylabel("count")

plt.subplot(2,2,2)
plt.bar(dyn_df["family"], dyn_df["alpha_raw"])
plt.xticks(rotation=30, ha="right")
plt.title("alpha_raw (logF vs logWard slope)")
plt.ylabel("alpha_raw")

plt.subplot(2,2,3)
plt.bar(dyn_df["family"], dyn_df["CoV_F"])
plt.xticks(rotation=30, ha="right")
plt.title("CoV(Fbar) across eps")
plt.ylabel("CoV")

plt.subplot(2,2,4)
plt.bar(amp_df["family"], amp_df["AdjR2"])
plt.xticks(rotation=30, ha="right")
plt.title("Topological locking strength (Adj.R2) @ eps0")
plt.ylim(0,1.05)

plt.tight_layout()

print("\n✅ DONE (print end marker)")

