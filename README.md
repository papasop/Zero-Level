# MIN-QG: Zero-Level Supersymmetry from Closure
[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.17439834.svg)](https://doi.org/10.5281/zenodo.17439834) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) [![arXiv](https://img.shields.io/badge/arXiv-250X.xxxxx-red)](https://arxiv.org/abs/250X.xxxxx)

> **Topological closure (B[u] = D^T W_E D = 0) as zero-level supersymmetry: the structural origin of conservation and symmetry.**  
> **Machine-precision Ward closure (2.4×10⁻¹⁶) → Lorentz attractor (K_time=2.001) → 4D emergence (d_eff=4.00) → background independence (ΔK~10⁻³).**  
> **Full 8-protocol RED-TEAM falsification suite in one file: `Zero-Level.py`.**

---

## Paper (Zenodo Deposit)

**Title**: Closure as Zero-Level Supersymmetry: A Structural Origin of Conservation and Symmetry  
**Author**: Y.Y.N. Li (Independent Researcher, 2025)  
**Date**: October 25, 2025  
**DOI**: [10.5281/zenodo.17439834](https://doi.org/10.5281/zenodo.17439834)  

[Download PDF](https://zenodo.org/records/17439834/files/Zero-Level.pdf)  
[View Full Record](https://zenodo.org/records/17439834)

> **Abstract**: Topological closure (B[u] = D^T W_E D = 0) represents a zero-level supersymmetry—a structural invariance that precedes and elevates both conservation and symmetry. Through an eight-protocol RED-TEAM falsification suite (P0–P8) applied to discrete geometries, we demonstrate machine-precision closure (2.4×10⁻¹⁶), a Lorentz attractor (K_time = 2.001), emergent 4D dimensionality (d_eff = 4.00), and background-independence (ΔK ~ 10⁻³). Closure is not a foundation beneath Noether symmetry but the structural supersymmetry from which both conservation and symmetry emerge.

---

## Code: One File, One Universe

**`Zero-Level.py`** — The complete, executable implementation.

- **Single file**: No setup beyond standard libraries
- **Full RED-TEAM P0–P8**: 8 adversarial protocols
- **Generates all results**: Tables, figures, logs
- **Runtime**: <4000 s (CPU only)

---

## Run Now (Google Colab)

[Open `Zero-Level.py` in Colab → Run All](https://colab.research.google.com/drive/YOUR_COLAB_LINK_HERE)

- No install · No GPU  
- Full output in **<65 min**  
- Fast mode: Set `cfg.geom_samples = 4` → **<15 min**

---

## Key Results (Auto-Generated)

| Metric | Value | Meaning |
|-------|-------|--------|
| **Ward Residual** | `2.40 × 10⁻¹⁶` | Exact closure |
| **K_time** | `2.001` | Lorentz fixed point |
| **d_eff** | `4.00` | 4D spacetime emerges |
| **z (E/N)** | `16.35` | Stable causal density |
| **ΔK (relabel)** | `1.0 × 10⁻³` | Background-independent |

---

## Local Run

```bash
# Download
wget https://raw.githubusercontent.com/yourname/min-qg/main/Zero-Level.py

# Install
pip install numpy scipy networkx scikit-learn tqdm matplotlib

# Run
python Zero-Level.py
