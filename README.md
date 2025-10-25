# MIN-QG: Zero-Level Supersymmetry from Closure
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/YOUR_COLAB_LINK_HERE) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) [![arXiv](https://img.shields.io/badge/arXiv-250X.xxxxx-red)](https://arxiv.org/abs/250X.xxxxx)

> **One file. One universe.**  
> `Zero-Level.py` generates **Ward closure**, **K=2.001**, **4D spacetime**, **background independence** — all in <4000 s.

---

## Run Now (Colab)

[Open `Zero-Level.py` in Colab → Run All](https://colab.research.google.com/drive/YOUR_COLAB_LINK_HERE)

- No install · No GPU  
- Full RED-TEAM P0–P8 in **<65 min**  
- Fast mode: `cfg.geom_samples = 4` → **<15 min**

---

## Results (Auto-Generated)

| Metric | Value | Meaning |
|-------|-------|--------|
| **Ward** | `2.40e-16` | Machine closure |
| **K_time** | `2.001` | Lorentz fixed point |
| **d_eff** | `4.00` | 4D emerges |
| **z** | `16.35` | Stable density |
| **ΔK** | `1.0e-3` | Background-free |

---

## Local Run

```bash
wget https://raw.githubusercontent.com/yourname/min-qg/main/Zero-Level.py
pip install numpy scipy networkx scikit-learn tqdm matplotlib
python Zero-Level.py
