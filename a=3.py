# ==========================================
# Closure Perturbation Falsification Test
# ==========================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

# --- 1. 基准值（闭合点）
G_CODATA = 6.67430e-11
Ward0 = 9.640468e-03
eps0 = 1e-3

# --- 2. 构造扰动系列 ε（打破闭合）
eps_list = np.logspace(-4, -1, 20)  # 从1e-4到1e-1

# --- 3. 模拟 Ward（假设与 ε 近似线性）
Ward = Ward0 * (eps_list / eps0)

# --- 4. 定义幂律响应模型：G ~ Ward^{-α}
def G_model(W, a, alpha):
    return G_CODATA * np.exp(a) * (W / Ward0) ** (-alpha)

# --- 5. 模拟实际观测（带微小噪声）
alpha_true = 3.0
a_true = 0.0
G_sim = G_model(Ward, a_true, alpha_true) * (1 + np.random.normal(0, 0.03, len(Ward)))

# --- 6. 拟合 log–log 幂律
logW = np.log(Ward / Ward0)
logG = np.log(G_sim / G_CODATA)
popt, pcov = curve_fit(lambda x, a, b: a + b*x, logW, logG)
a_fit, b_fit = popt
alpha_fit = -b_fit

# --- 7. 输出结果表
df = pd.DataFrame({
    "eps": eps_list,
    "Ward": Ward,
    "G_sim": G_sim,
    "G_ratio": G_sim / G_CODATA
})
display(df.head(8))

# --- 8. 打印拟合结果
print("=== Global Power-Law Test ===")
print(f"log(G/G0) ≈ a + b log(W/W0)")
print(f"a ≈ {a_fit:.4f},  b ≈ {b_fit:.4f}  → α ≈ {alpha_fit:.4f}")

# --- 9. 绘制验证图
plt.figure(figsize=(6,4))
plt.loglog(Ward/Ward0, G_sim/G_CODATA, 'o', label='Simulated data')
plt.loglog(Ward/Ward0, np.exp(a_fit)*(Ward/Ward0)**b_fit, '--', label=f'Fit α≈{alpha_fit:.2f}')
plt.xlabel("Ward / Ward0 (closure residual ratio)")
plt.ylabel("G / G_CODATA")
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.title("Falsification of Power-Law Relation: G ∝ Ward^{-α}")
plt.show()
