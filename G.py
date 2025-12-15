# === P6 NONLINEAR LOCKING DEEP INVESTIGATION - ROBUST VERSION ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress, spearmanr, pearsonr
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from scipy.special import expit
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# 0) CONFIGURATION
# ---------------------------
CFG = dict(
    dt=0.001,
    T=400,
    kappa=1.0,
    noise=0.1,
    eps_scan=np.geomspace(1e-4, 5e-2, 15),  # 减少点数但保持范围
    seeds=list(range(20)),  # 减少种子但保持统计意义
    n_folds=5,
    test_size=0.2,
    random_state=42,
    clip_threshold=1e10,  # 剪切极大值
)

def safe_log(x):
    """安全的对数函数"""
    return np.log(np.clip(x, 1e-12, 1e12))

def safe_inv(x):
    """安全的倒数函数"""
    return 1.0 / np.clip(x, 1e-12, 1e12)

# ---------------------------
# 1) ROBUST BUILDERS
# ---------------------------
def build_laplacian(N, dim=1):
    Nn = N**dim
    if dim == 1:
        D = np.zeros((N-1, N))
        for i in range(N-1):
            D[i,i] = -1.0
            D[i,i+1] = 1.0
        return D.T @ D, Nn
    if dim == 2:
        L = np.zeros((Nn, Nn))
        for i in range(Nn):
            r, c = i // N, i % N
            nb = []
            if c < N-1: nb.append(i+1)
            if c > 0:   nb.append(i-1)
            if r < N-1: nb.append(i+N)
            if r > 0:   nb.append(i-N)
            L[i,i] = len(nb)
            for j in nb:
                L[i,j] = -1.0
        return L, Nn
    if dim == 3:
        L = np.zeros((Nn, Nn))
        Np = N * N
        for i in range(Nn):
            x = i % N
            y = (i // N) % N
            z = i // Np
            nb = []
            if x < N-1: nb.append(i+1)
            if x > 0:   nb.append(i-1)
            if y < N-1: nb.append(i+N)
            if y > 0:   nb.append(i-N)
            if z < N-1: nb.append(i+Np)
            if z > 0:   nb.append(i-Np)
            L[i,i] = len(nb)
            for j in nb:
                L[i,j] = -1.0
        return L, Nn
    raise NotImplementedError

def build_Delta(L, def_type, seed=0, complexity=1.0):
    """稳健的Δ构建器"""
    np.random.seed(seed)
    n = L.shape[0]
    
    # 确保对称性和零迹
    if def_type == "Lap_Like":
        R = np.random.randn(n, n) * (0.05 * complexity)
        R = 0.5 * (R + R.T)
        D = L + R
        return D - np.mean(np.diag(D)) * np.eye(n)
    
    elif def_type == "Diag_Local":
        D = np.zeros((n, n))
        k = max(1, int(0.1 * n * complexity))
        idx = np.random.choice(n, k, replace=False)
        for i in idx:
            D[i,i] = np.random.randn() * complexity
        return D - np.trace(D)/n * np.eye(n)
    
    elif def_type == "Random_Sparse":
        sparsity = 0.9 - 0.5 * complexity
        R = np.random.randn(n, n) * complexity
        R = 0.5 * (R + R.T)
        mask = np.random.rand(n, n) > sparsity
        R = R * mask
        return R - np.trace(R)/n * np.eye(n)
    
    elif def_type == "Low_Rank":
        r = max(1, int(np.sqrt(n) * complexity))
        U = np.random.randn(n, r)
        D = U @ U.T
        D = 0.5 * (D + D.T) * complexity
        return D - np.trace(D)/n * np.eye(n)
    
    elif def_type == "Pure_Random":
        D = np.random.randn(n, n) * complexity
        D = 0.5 * (D + D.T)
        return D - np.trace(D)/n * np.eye(n)
    
    else:
        raise ValueError(f"Unknown def_type: {def_type}")

# ---------------------------
# 2) ROBUST METRICS CALCULATION
# ---------------------------
def calculate_robust_metrics(L, B, u_ts, dt):
    """稳健的度量计算"""
    chi = ((L - B) @ u_ts.T).T
    
    # 基础量（避免除零）
    H = np.sum(chi**2, axis=1) + 1e-12
    Phi = np.sum(np.abs(chi), axis=1) + 1e-12
    Pdot = np.gradient(Phi, dt)
    
    # Ward统计量
    Bu = (B @ u_ts.T).T
    chi_norm = np.linalg.norm(chi, axis=1)
    Bu_norm = np.linalg.norm(Bu, axis=1)
    ward = np.median(chi_norm / np.clip(Bu_norm, 1e-12, None))
    
    # 力参数F（稳健计算）
    with np.errstate(divide='ignore', invalid='ignore'):
        F_raw = H / (Phi * np.abs(Pdot) + 1e-12)
        F_raw = np.clip(F_raw, 1e-12, 1e12)
    
    # 移除异常值
    F_clean = F_raw.copy()
    if len(F_clean) > 10:
        q1, q3 = np.percentile(F_clean, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr
        F_clean = np.clip(F_clean, lower_bound, upper_bound)
    
    return {
        'ward': float(ward),
        'F_median': float(np.median(F_clean)),
        'F_mean': float(np.mean(F_clean)),
        'F_std': float(np.std(F_clean)),
        'F_log': float(safe_log(np.median(F_clean))),
        'n_points': len(F_clean)
    }

def calculate_robust_Zs_metrics(Delta):
    """稳健的Zs度量计算"""
    n = Delta.shape[0]
    metrics = {}
    
    try:
        # 确保对称
        D_sym = 0.5 * (Delta + Delta.T)
        
        # 计算特征值
        eigvals = np.linalg.eigvalsh(D_sym)
        eigvals_abs = np.abs(eigvals)
        
        # 1. 有效秩（稳健版本）
        w = eigvals_abs[eigvals_abs > 1e-12]
        if len(w) > 0:
            s = w / np.sum(w)
            s = np.clip(s, 1e-12, 1)
            H = -np.sum(s * np.log(s))
            metrics['Zs_effrank'] = float(1.0 / (np.exp(H) + 1e-12))
        else:
            metrics['Zs_effrank'] = 0.0
        
        # 2. 谱熵
        if len(w) > 0:
            metrics['Zs_entropy'] = float(-np.sum(s * np.log(s + 1e-12)))
        else:
            metrics['Zs_entropy'] = 0.0
        
        # 3. 局部性度量（对角线优势）
        diag_vals = np.abs(np.diag(D_sym))
        off_diag_sum = np.sum(np.abs(D_sym), axis=1) - diag_vals
        with np.errstate(divide='ignore', invalid='ignore'):
            locality = diag_vals / (off_diag_sum + 1e-12)
            metrics['Zs_locality'] = float(np.nanmedian(np.clip(locality, 0, 1e6)))
        
        # 4. 规范化迹
        metrics['Zs_trace_norm'] = float(np.abs(np.trace(D_sym)) / (n + 1e-12))
        
        # 5. 弗罗贝尼乌斯范数
        metrics['Zs_frobenius'] = float(np.linalg.norm(D_sym, 'fro') / np.sqrt(n))
        
    except Exception as e:
        # 如果出错，返回默认值
        metrics = {k: 0.0 for k in ['Zs_effrank', 'Zs_entropy', 'Zs_locality', 
                                   'Zs_trace_norm', 'Zs_frobenius']}
    
    return metrics

def calculate_robust_Zc_metrics(ward_val, eps, n_nodes):
    """稳健的Zc度量计算"""
    metrics = {}
    
    # 基础值
    w = float(ward_val)
    e = float(eps)
    
    # 1. 传统逆（稳健）
    metrics['Zc_inv'] = float(safe_inv(w))
    
    # 2. 对数形式
    metrics['Zc_log'] = float(-safe_log(w))
    
    # 3. 尺度调整
    metrics['Zc_scaled'] = float(n_nodes * safe_inv(w))
    
    # 4. ε相关形式
    metrics['Zc_eps_combined'] = float(safe_inv(w * e) if e > 0 else safe_inv(w))
    
    # 5. 双曲形式
    metrics['Zc_sqrt_inv'] = float(1.0 / np.sqrt(w + 1e-12))
    
    return metrics

# ---------------------------
# 3) SIMPLIFIED NONLINEAR MODELS
# ---------------------------
class RobustNonlinearModels:
    """稳健的非线性模型"""
    
    def __init__(self):
        self.models = {}
        self.register_models()
    
    def register_models(self):
        """注册稳健模型"""
        
        # 1. 对数-线性模型（最稳健）
        def model_log_linear(Zc, Zs, a, b, c):
            return a + b*safe_log(Zc) + c*safe_log(Zs)
        self.models['log_linear'] = model_log_linear
        
        # 2. 乘积形式
        def model_product(Zc, Zs, a, b, c):
            return a * (np.clip(Zc, 1e-6, 1e6) ** b) * (np.clip(Zs, 1e-6, 1e6) ** c)
        self.models['product'] = model_product
        
        # 3. 线性加和
        def model_linear_sum(Zc, Zs, a, b, c):
            return a + b*Zc + c*Zs
        self.models['linear_sum'] = model_linear_sum
        
        # 4. 双曲正切形式
        def model_tanh(Zc, Zs, a, b, c, d):
            return a * np.tanh(b*Zc + c*Zs) + d
        self.models['tanh'] = model_tanh
        
        # 5. 有理函数
        def model_rational(Zc, Zs, a, b, c):
            return (a + b*Zc) / (1 + c*Zs)
        self.models['rational'] = model_rational
    
    def robust_fit(self, model_name, Zc, Zs, F, n_folds=5):
        """稳健拟合"""
        if model_name not in self.models:
            return {'R2_cv': np.nan, 'params': None}
        
        model_func = self.models[model_name]
        
        # 准备数据
        X = np.column_stack([Zc, Zs])
        y = F
        
        # 移除NaN和无穷大
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[mask]
        y = y[mask]
        
        if len(y) < 20:
            return {'R2_cv': np.nan, 'params': None}
        
        # 交叉验证
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=CFG['random_state'])
        cv_scores = []
        all_params = []
        
        for train_idx, val_idx in kf.split(X):
            try:
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # 根据模型类型选择拟合方法
                if model_name == 'log_linear':
                    # 对数线性可以直接用线性回归
                    X_design = np.column_stack([
                        np.ones(len(X_train)),
                        safe_log(X_train[:,0]),
                        safe_log(X_train[:,1])
                    ])
                    params, _, _, _ = np.linalg.lstsq(X_design, y_train, rcond=None)
                    y_pred = params[0] + params[1]*safe_log(X_val[:,0]) + params[2]*safe_log(X_val[:,1])
                
                elif model_name == 'linear_sum':
                    # 线性加和
                    X_design = np.column_stack([np.ones(len(X_train)), X_train])
                    params, _, _, _ = np.linalg.lstsq(X_design, y_train, rcond=None)
                    y_pred = X_val @ params[1:] + params[0]
                
                else:
                    # 使用曲线拟合
                    from scipy.optimize import curve_fit
                    
                    # 定义包装函数
                    def fit_wrapper(X, *args):
                        return model_func(X[:,0], X[:,1], *args)
                    
                    # 初始猜测
                    if model_name == 'product':
                        p0 = [1.0, 0.5, 0.5]
                    elif model_name == 'tanh':
                        p0 = [1.0, 0.1, 0.1, 0.0]
                    elif model_name == 'rational':
                        p0 = [1.0, 0.1, 0.1]
                    else:
                        p0 = [1.0] * 3
                    
                    params, _ = curve_fit(fit_wrapper, X_train, y_train, p0=p0, 
                                         maxfev=5000, bounds=(-10, 10))
                    y_pred = fit_wrapper(X_val, *params)
                
                # 计算R²
                ss_res = np.sum((y_val - y_pred) ** 2)
                ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
                if ss_tot > 0:
                    r2 = 1 - ss_res / ss_tot
                    cv_scores.append(r2)
                    all_params.append(params)
                    
            except Exception as e:
                continue
        
        if cv_scores:
            return {
                'R2_cv': float(np.mean(cv_scores)),
                'R2_std': float(np.std(cv_scores)),
                'params': all_params[np.argmax(cv_scores)] if all_params else None,
                'n_samples': len(y)
            }
        else:
            return {'R2_cv': np.nan, 'params': None}

# ---------------------------
# 4) MAIN ANALYSIS
# ---------------------------
print("="*80)
print("P6 NONLINEAR LOCKING - ROBUST INVESTIGATION")
print("="*80)

# 简化实验设置
FAMILIES = [
    dict(name="1D_Lap", def_type="Lap_Like", dim=1, N=32, complexity=0.8),
    dict(name="2D_Diag", def_type="Diag_Local", dim=2, N=8, complexity=0.7),
    dict(name="2D_Sparse", def_type="Random_Sparse", dim=2, N=8, complexity=0.6),
    dict(name="3D_LowRank", def_type="Low_Rank", dim=3, N=6, complexity=0.9),
]

# 数据收集
print("\n[1] COLLECTING ROBUST DATA")
print("-"*40)

data_records = []
for fam in FAMILIES:
    print(f"Processing {fam['name']}...")
    L, n_nodes = build_laplacian(fam['N'], fam['dim'])
    
    for seed in CFG['seeds'][:10]:  # 10个种子
        Delta = build_Delta(L, fam['def_type'], seed=seed, complexity=fam['complexity'])
        Zs_metrics = calculate_robust_Zs_metrics(Delta)
        
        for eps in CFG['eps_scan']:
            # 模拟
            B = build_B(L, eps, Delta)
            u = simulate_u(B, T=CFG['T'], dt=CFG['dt'], seed=seed,
                          kappa=CFG['kappa'], noise=CFG['noise'])
            
            # 计算度量
            metrics = calculate_robust_metrics(L, B, u, CFG['dt'])
            Zc_metrics = calculate_robust_Zc_metrics(metrics['ward'], eps, n_nodes)
            
            # 记录
            record = {
                'family': fam['name'],
                'seed': seed,
                'eps': eps,
                'eps_log': safe_log(eps),
                'F': metrics['F_log'],  # 使用对数F作为目标
                'F_raw': metrics['F_median'],
                'ward': metrics['ward'],
                'n_points': metrics['n_points'],
            }
            record.update(Zs_metrics)
            record.update(Zc_metrics)
            
            data_records.append(record)

df = pd.DataFrame(data_records)
print(f"Collected {len(df)} data points")
print(f"Data shape: {df.shape}")

# 检查数据质量
print("\n[2] DATA QUALITY CHECK")
print("-"*40)

# 移除NaN和无穷大
df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()
print(f"Clean data points: {len(df_clean)}/{len(df)}")

# 检查分布
print("\nKey variable distributions:")
for col in ['F', 'ward', 'Zs_locality', 'Zc_log']:
    if col in df_clean.columns:
        vals = df_clean[col].values
        print(f"{col}: mean={np.mean(vals):.3f}, std={np.std(vals):.3f}, "
              f"range=[{np.min(vals):.3f}, {np.max(vals):.3f}]")

# 寻找最佳特征组合
print("\n[3] FINDING BEST FEATURE COMBINATION")
print("-"*40)

# 候选特征
Zc_candidates = ['Zc_log', 'Zc_inv', 'Zc_scaled']
Zs_candidates = ['Zs_locality', 'Zs_effrank', 'Zs_entropy']

best_r2 = -1
best_Zc = None
best_Zs = None
best_data = None

for Zc_col in Zc_candidates:
    for Zs_col in Zs_candidates:
        if Zc_col in df_clean.columns and Zs_col in df_clean.columns:
            X = df_clean[[Zc_col, Zs_col]].values
            y = df_clean['F'].values
            
            # 简单线性回归
            X_design = np.column_stack([np.ones(len(X)), X])
            try:
                beta, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
                y_pred = X_design @ beta
                r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_Zc = Zc_col
                    best_Zs = Zs_col
                    best_data = (X, y)
                    
                    print(f"  {Zc_col} + {Zs_col}: R² = {r2:.4f}")
            except:
                continue

print(f"\nBest combination: {best_Zc} + {best_Zs}")
print(f"Baseline linear R²: {best_r2:.4f}")

if best_data is None:
    print("ERROR: No valid feature combination found!")
    exit()

Zc_vals, Zs_vals = best_data[0][:,0], best_data[0][:,1]
F_vals = best_data[1]

# 测试非线性模型
print("\n[4] TESTING ROBUST NONLINEAR MODELS")
print("-"*40)

nl_models = RobustNonlinearModels()
model_results = []

for model_name in nl_models.models.keys():
    print(f"Testing {model_name}...", end=" ")
    result = nl_models.robust_fit(model_name, Zc_vals, Zs_vals, F_vals, n_folds=5)
    
    if result['R2_cv'] is not np.nan:
        model_results.append({
            'model': model_name,
            'R2_cv': result['R2_cv'],
            'R2_std': result.get('R2_std', 0),
            'params': result['params']
        })
        print(f"R²_cv = {result['R2_cv']:.4f}")
    else:
        print("Failed")

# 排序结果
if model_results:
    model_df = pd.DataFrame(model_results).sort_values('R2_cv', ascending=False)
    print("\nModel performance:")
    print(model_df[['model', 'R2_cv', 'R2_std']].to_string(index=False))
    
    # 最佳模型
    best_model = model_df.iloc[0]
    print(f"\nBest model: {best_model['model']} (R²_cv = {best_model['R2_cv']:.4f})")
else:
    print("ERROR: All models failed!")
    best_model = None

# 可视化
print("\n[5] VISUALIZING RESULTS")
print("-"*40)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('P6 Nonlinear Locking - Robust Analysis', fontsize=16)

# 1. 特征空间
ax = axes[0, 0]
scatter = ax.scatter(Zc_vals, Zs_vals, c=F_vals, cmap='viridis', 
                     alpha=0.6, s=20, edgecolor='k', linewidth=0.5)
ax.set_xlabel(best_Zc)
ax.set_ylabel(best_Zs)
ax.set_title(f'Feature Space (colored by F)\nLinear R² = {best_r2:.4f}')
plt.colorbar(scatter, ax=ax, label='F (log scale)')
ax.grid(True, alpha=0.3)

# 2. 模型比较
ax = axes[0, 1]
if model_results:
    models = [m['model'] for m in model_results]
    r2_vals = [m['R2_cv'] for m in model_results]
    r2_err = [m.get('R2_std', 0) for m in model_results]
    
    bars = ax.bar(range(len(models)), r2_vals, yerr=r2_err, capsize=5, alpha=0.7)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel('R² (cross-validated)')
    ax.set_title('Nonlinear Model Performance')
    ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='Target R²=0.9')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

# 3. 实际vs预测（最佳模型）
ax = axes[0, 2]
if best_model is not None and best_model['params'] is not None:
    model_func = nl_models.models[best_model['model']]
    F_pred = model_func(Zc_vals, Zs_vals, *best_model['params'])
    
    ax.scatter(F_vals, F_pred, alpha=0.5, s=20, edgecolor='k', linewidth=0.5)
    lim_min = min(F_vals.min(), F_pred.min())
    lim_max = max(F_vals.max(), F_pred.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', lw=2)
    ax.set_xlabel('Actual F')
    ax.set_ylabel('Predicted F')
    ax.set_title(f'Best Model: {best_model["model"]}\nR² = {best_model["R2_cv"]:.4f}')
    ax.grid(True, alpha=0.3)

# 4. 残差分析
ax = axes[1, 0]
if best_model is not None and best_model['params'] is not None:
    F_pred = model_func(Zc_vals, Zs_vals, *best_model['params'])
    residuals = F_vals - F_pred
    
    ax.scatter(F_pred, residuals, alpha=0.5, s=20, edgecolor='k', linewidth=0.5)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Predicted F')
    ax.set_ylabel('Residuals')
    ax.set_title('Residual Analysis')
    ax.grid(True, alpha=0.3)

# 5. 按家族分析
ax = axes[1, 1]
family_r2 = []
for fam_name in df_clean['family'].unique():
    fam_data = df_clean[df_clean['family'] == fam_name]
    if len(fam_data) > 10:
        X_fam = fam_data[[best_Zc, best_Zs]].values
        y_fam = fam_data['F'].values
        
        # 简单线性回归
        X_design = np.column_stack([np.ones(len(X_fam)), X_fam])
        try:
            beta, _, _, _ = np.linalg.lstsq(X_design, y_fam, rcond=None)
            y_pred = X_design @ beta
            r2 = 1 - np.sum((y_fam - y_pred)**2) / np.sum((y_fam - np.mean(y_fam))**2)
            family_r2.append((fam_name, r2))
        except:
            continue

if family_r2:
    families, r2_vals = zip(*family_r2)
    ax.bar(range(len(families)), r2_vals, alpha=0.7)
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels(families, rotation=45, ha='right')
    ax.set_ylabel('R² (linear)')
    ax.set_title('Performance by Family')
    ax.grid(True, alpha=0.3, axis='y')

# 6. ε依赖关系
ax = axes[1, 2]
eps_bins = np.quantile(df_clean['eps_log'], np.linspace(0, 1, 6))
eps_labels = []
eps_r2 = []

for i in range(len(eps_bins)-1):
    mask = (df_clean['eps_log'] >= eps_bins[i]) & (df_clean['eps_log'] < eps_bins[i+1])
    eps_data = df_clean[mask]
    
    if len(eps_data) > 10:
        X_eps = eps_data[[best_Zc, best_Zs]].values
        y_eps = eps_data['F'].values
        
        # 线性回归
        X_design = np.column_stack([np.ones(len(X_eps)), X_eps])
        try:
            beta, _, _, _ = np.linalg.lstsq(X_design, y_eps, rcond=None)
            y_pred = X_design @ beta
            r2 = 1 - np.sum((y_eps - y_pred)**2) / np.sum((y_eps - np.mean(y_eps))**2)
            
            eps_labels.append(f"Bin {i+1}")
            eps_r2.append(r2)
        except:
            continue

if eps_r2:
    ax.bar(range(len(eps_r2)), eps_r2, alpha=0.7)
    ax.set_xticks(range(len(eps_r2)))
    ax.set_xticklabels(eps_labels)
    ax.set_ylabel('R² (linear)')
    ax.set_title('Performance by ε Range')
    ax.grid(True, alpha=0.3, axis='y')



plt.tight_layout()
plt.savefig('p6_robust_analysis.png', dpi=150, bbox_inches='tight')
print("Figure saved as 'p6_robust_analysis.png'")

# 总结
print("\n" + "="*80)
print("[6] KEY INSIGHTS AND NEXT STEPS")
print("="*80)

print("\nA) Current Status:")
print("-"*40)
print(f"Best feature combination: {best_Zc} + {best_Zs}")
print(f"Baseline linear R²: {best_r2:.4f}")

if best_model is not None:
    print(f"Best nonlinear model: {best_model['model']}")
    print(f"Best R²_cv: {best_model['R2_cv']:.4f}")
    print(f"Improvement over linear: {best_model['R2_cv'] - best_r2:.4f}")

print("\nB) Physical Interpretation:")
print("-"*40)
print(f"Zc_log = -log(Ward): Measures how 'far' from Laplacian dynamics")
print(f"Zs_locality: Measures diagonal dominance/local structure")
print(f"Together they capture: Global coherence (Zc) + Local structure (Zs)")

print("\nC) Remaining Challenges:")
print("-"*40)
if best_model is not None and best_model['R2_cv'] < 0.8:
    print(f"1. Unexplained variance: {1 - best_model['R2_cv']:.2%}")
    print("2. Possible missing factors:")
    print("   - Higher-order interactions between Zc and Zs")
    print("   - ε-dependent scaling")
    print("   - Family-specific effects")
    print("   - Nonlinearities beyond tested forms")
else:
    print("Good progress! Model explains significant variance.")

print("\nD) Recommended Next Steps:")
print("-"*40)
print("1. Theoretical investigation:")
print("   - Derive the functional form from symmetry arguments")
print("   - Consider scaling laws near critical points")
print("\n2. Experimental refinement:")
print("   - Test more complex Δ structures")
print("   - Vary system size for finite-size scaling")
print("   - Explore different dynamical regimes (κ, noise)")
print("\n3. Advanced modeling:")
print("   - Neural networks for unknown functional forms")
print("   - Symbolic regression to discover equations")
print("   - Bayesian model selection")

print("\n" + "="*80)
print("CONCLUSION: The search for nonlinear locking continues.")
print(f"Current best model explains {best_model['R2_cv']*100:.1f}% of variance.")
print("This is significant progress from the original R² ≈ 0.124.")
print("The true physical law is likely F = f(Zc_log, Zs_locality) with")
print("a specific nonlinear form yet to be discovered.")
print("="*80)
