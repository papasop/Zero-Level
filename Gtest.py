# ==============================================
# CRITICAL TEST: Is C_L₀ a Fixed Point?
# Test if systematic offset converges to fixed value vs tends to zero
# ==============================================
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize
import pandas as pd
from tqdm import tqdm

np.random.seed(42)

class FixedPointTest:
    """Test if C_L₀ is a true topological fixed point"""
    
    def __init__(self):
        self.G_CODATA = 6.67430e-11
        self.c = 299792458.0
        self.hbar = 1.054571817e-34
        
        # Planck units
        self.l_pl = np.sqrt(self.hbar * self.G_CODATA / self.c**3)
        self.t_pl = self.l_pl / self.c
        self.m_pl = np.sqrt(self.hbar * self.c / self.G_CODATA)
        
    def generate_system(self, N):
        """Generate N systems with structural parameters"""
        # Generate in natural units (Planck units)
        sigma = 0.15  # Realistic variation
        
        L_L_nat = np.random.lognormal(mean=0, sigma=sigma, size=N)
        L_M_nat = np.random.lognormal(mean=0, sigma=sigma, size=N)
        
        # Convert to SI
        L_L_SI = L_L_nat * self.l_pl
        L_M_SI = L_M_nat * self.m_pl
        
        # Structural term in SI
        structural_SI = self.c**2 * L_L_SI / L_M_SI
        
        # Calculate mean and statistics
        mean_struct = np.mean(structural_SI)
        std_struct = np.std(structural_SI)
        
        # Calculate C_L₀ that would give perfect prediction
        C_L0_ideal = self.G_CODATA / mean_struct
        
        # Prediction errors
        # Without C_L₀
        G_without = structural_SI
        error_without = np.mean(np.abs(G_without - self.G_CODATA) / self.G_CODATA)
        
        # With ideal C_L₀
        G_with_ideal = C_L0_ideal * structural_SI
        error_with_ideal = np.mean(np.abs(G_with_ideal - self.G_CODATA) / self.G_CODATA)
        
        # Statistical uncertainty
        se_mean = std_struct / np.sqrt(N)
        rel_uncertainty = se_mean / mean_struct
        
        return {
            'N': N,
            'mean_struct': mean_struct,
            'std_struct': std_struct,
            'C_L0_ideal': C_L0_ideal,
            'error_without': error_without,
            'error_with_ideal': error_with_ideal,
            'rel_uncertainty': rel_uncertainty,
            'log_offset': np.log(mean_struct) - np.log(self.G_CODATA)
        }

def run_fixed_point_analysis():
    """Main analysis: test convergence with increasing N"""
    print("="*80)
    print("CRITICAL TEST: Is C_L₀ a True Topological Fixed Point?")
    print("="*80)
    print("\nKey question: As N → ∞, does systematic offset → 0 or → fixed value?")
    print("If C_L₀ is topological fixed point: offset → fixed non-zero value")
    print("If C_L₀ is statistical mean: offset → 0 (law of large numbers)")
    print("-"*80)
    
    tester = FixedPointTest()
    
    # Test increasing N
    N_values = [10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]
    
    results = []
    print("\nRunning convergence test with increasing N:")
    print("-"*60)
    
    for N in tqdm(N_values):
        result = tester.generate_system(N)
        results.append(result)
        
        if N <= 1000 or N % 10000 == 0:
            print(f"N = {N:6d}: C_L₀ = {result['C_L0_ideal']:.6f}, "
                  f"Offset = {result['error_without']:.3%}, "
                  f"Uncertainty = {result['rel_uncertainty']:.3%}")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # ============================================
    # ANALYSIS 1: Convergence of C_L₀
    # ============================================
    print("\n" + "="*80)
    print("ANALYSIS 1: Convergence of C_L₀ with N")
    print("="*80)
    
    # Fit convergence model
    logN = np.log(df['N'])
    C_L0_values = df['C_L0_ideal']
    
    # Fit power law: C_L₀(N) = C_inf + a × N^(-b)
    def convergence_model(N, C_inf, a, b):
        return C_inf + a * (N ** (-b))
    
    try:
        popt, _ = optimize.curve_fit(convergence_model, df['N'], C_L0_values,
                                    p0=[0.97, 0.1, 0.5], maxfev=5000)
        C_inf, a, b = popt
        
        print(f"\nConvergence fit: C_L₀(N) = {C_inf:.6f} + {a:.3f} × N^(-{b:.3f})")
        print(f"As N → ∞, C_L₀ → {C_inf:.6f}")
        print(f"Convergence exponent b = {b:.3f}")
        
        # Compare with expectation for statistical mean
        # For statistical mean, b should be ~0.5 (central limit theorem)
        print(f"\nInterpretation:")
        print(f"  If b ≈ 0.5: Statistical convergence (law of large numbers)")
        print(f"  If b ≈ 0: Fixed point (independent of N)")
        print(f"  Our b = {b:.3f} → {'Fixed point behavior' if b < 0.3 else 'Statistical convergence'}")
        
    except:
        print("Convergence fit failed - using direct analysis")
        C_inf = np.mean(C_L0_values[-3:])  # Average of last 3 points
    
    # ============================================
    # ANALYSIS 2: Systematic offset convergence
    # ============================================
    print("\n" + "="*80)
    print("ANALYSIS 2: Systematic Offset Convergence")
    print("="*80)
    
    # Fit offset convergence
    offsets = df['error_without']
    
    def offset_model(N, offset_inf, c, d):
        return offset_inf + c * (N ** (-d))
    
    try:
        popt_offset, _ = optimize.curve_fit(offset_model, df['N'], offsets,
                                           p0=[0.17, 0.5, 0.5], maxfev=5000)
        offset_inf, c, d = popt_offset
        
        print(f"\nOffset convergence: Offset(N) = {offset_inf:.3%} + {c:.3f} × N^(-{d:.3f})")
        print(f"As N → ∞, systematic offset → {offset_inf:.3%}")
        print(f"Paper's claimed offset: ~1171% = 11.71")
        print(f"Our asymptotic offset: {offset_inf:.3f} (factor)")
        
        if offset_inf > 0.01:  # More than 1% offset
            print("\n✅ RESULT: Offset converges to NON-ZERO fixed value")
            print("This supports C_L₀ as topological fixed point")
        else:
            print("\n⚠ RESULT: Offset converges toward zero")
            print("This suggests statistical rather than topological nature")
            
    except:
        print("Offset fit failed")
    
    # ============================================
    # ANALYSIS 3: Statistical vs Topological Test
    # ============================================
    print("\n" + "="*80)
    print("ANALYSIS 3: Statistical Significance Test")
    print("="*80)
    
    # Test if C_L₀ values are consistent with random sampling
    # If C_L₀ is topological, values should cluster around fixed point
    # If statistical, variance should decrease as 1/√N
    
    # Calculate expected statistical variance
    expected_std = df['std_struct'] / (df['mean_struct'] * np.sqrt(df['N']))
    actual_std = np.std([r['C_L0_ideal'] for r in results[:5]])  # Use first few for baseline
    
    print(f"\nVariance analysis:")
    print(f"Expected statistical std (N→∞): {expected_std.iloc[-1]:.6f}")
    print(f"Actual variation in C_L₀: {np.std(df['C_L0_ideal']):.6f}")
    
    # Bayesian test: Is there evidence for fixed point?
    from scipy import stats
    
    # Fit two models:
    # Model 1: Fixed point (constant)
    # Model 2: Statistical mean (converging to some value)
    
    # Simple test: check if last few values are consistent with constant
    last_values = df['C_L0_ideal'].values[-5:]
    mean_last = np.mean(last_values)
    std_last = np.std(last_values)
    
    # Test if variation is less than expected from statistics
    expected_variation = expected_std.iloc[-5:].mean()
    
    print(f"\nLast 5 C_L₀ values: {last_values}")
    print(f"Mean: {mean_last:.6f}, Std: {std_last:.6f}")
    print(f"Expected statistical variation: {expected_variation:.6f}")
    
    if std_last < expected_variation:
        print("✅ Evidence for fixed point: Variation less than statistical expectation")
    else:
        print("⚠ Consistent with statistical variation")
    
    # ============================================
    # VISUALIZATION
    # ============================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. C_L₀ convergence
    ax = axes[0, 0]
    ax.loglog(df['N'], df['C_L0_ideal'], 'bo-', linewidth=2, markersize=6)
    ax.axhline(y=C_inf if 'C_inf' in locals() else np.mean(df['C_L0_ideal']), 
               color='r', linestyle='--', label=f'Asymptote: {C_inf:.4f}' if 'C_inf' in locals() else 'Mean')
    ax.set_xlabel('Number of Systems (N)', fontsize=11)
    ax.set_ylabel('C_L₀ Estimate', fontsize=11)
    ax.set_title('C_L₀ Convergence with N', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Systematic offset convergence
    ax = axes[0, 1]
    ax.loglog(df['N'], df['error_without'], 'ro-', linewidth=2, markersize=6)
    if 'offset_inf' in locals():
        ax.axhline(y=offset_inf, color='r', linestyle='--', 
                  label=f'Asymptote: {offset_inf:.3%}')
    ax.set_xlabel('Number of Systems (N)', fontsize=11)
    ax.set_ylabel('Systematic Offset', fontsize=11)
    ax.set_title('Systematic Offset Convergence', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Uncertainty reduction
    ax = axes[0, 2]
    ax.loglog(df['N'], df['rel_uncertainty'], 'go-', linewidth=2, markersize=6)
    # Theoretical 1/√N line
    theoretical = 0.15 / np.sqrt(df['N'])  # Based on sigma=0.15
    ax.loglog(df['N'], theoretical, 'k--', label='Theoretical: 1/√N')
    ax.set_xlabel('Number of Systems (N)', fontsize=11)
    ax.set_ylabel('Relative Uncertainty', fontsize=11)
    ax.set_title('Statistical Uncertainty Reduction', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Distribution of structural term (log scale)
    ax = axes[1, 0]
    # Generate one large sample for distribution
    large_N = 100000
    L_L_large = np.random.lognormal(mean=0, sigma=0.15, size=large_N)
    L_M_large = np.random.lognormal(mean=0, sigma=0.15, size=large_N)
    structural_large = L_L_large / L_M_large
    
    ax.hist(np.log(structural_large), bins=50, alpha=0.7, density=True)
    ax.axvline(x=0, color='r', linestyle='--', label='Expected: log(1) = 0')
    ax.set_xlabel('log(structural term)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'Distribution of Structural Term (N={large_N:,})', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. C_L₀ vs N on linear scale
    ax = axes[1, 1]
    ax.plot(df['N'], df['C_L0_ideal'], 'bo-', linewidth=2, markersize=6)
    if 'C_inf' in locals():
        ax.axhline(y=C_inf, color='r', linestyle='--', 
                  label=f'Asymptote: {C_inf:.6f}')
        # Add confidence band
        asymptotic_std = np.std(df['C_L0_ideal'].values[-10:])
        ax.fill_between(df['N'], C_inf - 2*asymptotic_std, C_inf + 2*asymptotic_std,
                       alpha=0.2, color='r', label='95% confidence band')
    ax.set_xlabel('Number of Systems (N)', fontsize=11)
    ax.set_ylabel('C_L₀', fontsize=11)
    ax.set_title('C_L₀ Convergence (Linear Scale)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Bayesian evidence plot
    ax = axes[1, 2]
    
    # Simulate Bayesian update
    prior_mean = 1.0  # Prior: C_L₀ should be ~1
    prior_std = 0.1
    
    N_bayes = [10, 100, 1000, 10000]
    posterior_means = []
    posterior_stds = []
    
    for N in N_bayes:
        sample = tester.generate_system(N)
        data_mean = sample['C_L0_ideal']
        data_std = sample['rel_uncertainty']
        
        # Bayesian update (conjugate prior)
        posterior_precision = 1/prior_std**2 + N/data_std**2
        posterior_mean = (prior_mean/prior_std**2 + N*data_mean/data_std**2) / posterior_precision
        posterior_std = 1/np.sqrt(posterior_precision)
        
        posterior_means.append(posterior_mean)
        posterior_stds.append(posterior_std)
    
    ax.errorbar(N_bayes, posterior_means, yerr=np.array(posterior_stds)*2, 
               fmt='o-', capsize=5, capthick=2, label='Posterior ± 2σ')
    ax.axhline(y=1.0, color='r', linestyle='--', label='Expected: 1.0')
    ax.set_xscale('log')
    ax.set_xlabel('Number of Systems (N)', fontsize=11)
    ax.set_ylabel('C_L₀ Posterior Mean', fontsize=11)
    ax.set_title('Bayesian Evidence Accumulation', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # ============================================
    # FINAL ASSESSMENT
    # ============================================
    print("\n" + "="*80)
    print("FINAL ASSESSMENT: Is C_L₀ a Topological Fixed Point?")
    print("="*80)
    
    # Collect evidence
    evidence_fixed = []
    evidence_statistical = []
    
    # Evidence 1: Convergence rate
    if 'b' in locals() and b < 0.3:
        evidence_fixed.append(f"Slow convergence (b={b:.3f} < 0.3)")
    else:
        evidence_statistical.append(f"Fast convergence (b={b:.3f} ≈ 0.5)")
    
    # Evidence 2: Asymptotic offset
    if 'offset_inf' in locals() and offset_inf > 0.01:
        evidence_fixed.append(f"Non-zero asymptotic offset ({offset_inf:.3%})")
    else:
        evidence_statistical.append(f"Offset tends to zero")
    
    # Evidence 3: Variance analysis
    if 'std_last' in locals() and 'expected_variation' in locals():
        if std_last < expected_variation:
            evidence_fixed.append("Variance less than statistical expectation")
        else:
            evidence_statistical.append("Variance consistent with statistics")
    
    print("\nEvidence for FIXED POINT (topological):")
    if evidence_fixed:
        for ev in evidence_fixed:
            print(f"  ✓ {ev}")
    else:
        print("  (No strong evidence)")
    
    print("\nEvidence for STATISTICAL MEAN:")
    if evidence_statistical:
        for ev in evidence_statistical:
            print(f"  ⚠ {ev}")
    else:
        print("  (Contradicts pure statistics)")
    
    # Overall conclusion
    print("\n" + "-"*80)
    if len(evidence_fixed) > len(evidence_statistical):
        print("CONCLUSION: C_L₀ shows characteristics of a TOPOLOGICAL FIXED POINT")
        print("The systematic offset converges to a non-zero fixed value")
        print("This supports the paper's claim of topological origin")
    else:
        print("CONCLUSION: C_L₀ behaves more like a STATISTICAL MEAN")
        print("The offset tends to zero with large N")
        print("This suggests a statistical rather than topological interpretation")
    
    print("-"*80)
    
    return df

# ============================================
# ADDITIONAL TEST: Finite Size Scaling
# ============================================

def finite_size_scaling_test():
    """Test finite size scaling to distinguish critical fixed point"""
    print("\n" + "="*80)
    print("ADDITIONAL TEST: Finite Size Scaling Analysis")
    print("="*80)
    print("\nIf C_L₀ is a critical fixed point, it should show scaling behavior")
    print("with characteristic exponents.")
    
    tester = FixedPointTest()
    
    # Test different system sizes (not just N, but physical size)
    # Simulate by varying the "correlation length" or domain size
    
    scales = [0.1, 0.3, 1.0, 3.0, 10.0]  # Scale factors
    N_per_scale = 1000
    
    scale_results = []
    
    for scale in scales:
        # Scale affects the variation
        effective_sigma = 0.15 / scale  # Larger scale → smaller relative variations
        
        L_L_scaled = np.random.lognormal(mean=0, sigma=effective_sigma, size=N_per_scale)
        L_M_scaled = np.random.lognormal(mean=0, sigma=effective_sigma, size=N_per_scale)
        
        structural_scaled = L_L_scaled / L_M_scaled
        mean_struct = np.mean(structural_scaled)
        C_L0_scaled = tester.G_CODATA / (mean_struct * tester.c**2 * 
                                       tester.l_pl / tester.m_pl)
        
        scale_results.append({
            'scale': scale,
            'C_L0': C_L0_scaled,
            'variation': effective_sigma,
            'mean_struct': mean_struct
        })
        
        print(f"Scale = {scale:4.1f}: C_L₀ = {C_L0_scaled:.6f}, "
              f"σ = {effective_sigma:.3f}, mean(S) = {mean_struct:.3f}")
    
    # Check scaling behavior
    scales_arr = np.array([r['scale'] for r in scale_results])
    C_L0_arr = np.array([r['C_L0'] for r in scale_results])
    
    # Fit scaling law: C_L₀(ξ) = C_∞ + A × ξ^(-ν)
    def scaling_law(x, C_inf, A, nu):
        return C_inf + A * (x ** (-nu))
    
    try:
        popt, _ = optimize.curve_fit(scaling_law, scales_arr, C_L0_arr,
                                    p0=[0.97, 0.1, 1.0])
        C_inf_scaling, A_scaling, nu_scaling = popt
        
        print(f"\nFinite size scaling fit:")
        print(f"C_L₀(ξ) = {C_inf_scaling:.6f} + {A_scaling:.3f} × ξ^(-{nu_scaling:.3f})")
        print(f"Critical exponent ν = {nu_scaling:.3f}")
        
        if nu_scaling > 0.5:
            print("Large exponent ν suggests critical fixed point behavior")
        else:
            print("Small exponent ν suggests non-critical behavior")
            
    except:
        print("\nScaling analysis inconclusive")
    
    return scale_results

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("FIXED POINT TEST: Is C_L₀ Topological or Statistical?")
    print("Testing convergence with N → ∞")
    print("="*80)
    
    # Run main fixed point analysis
    df_results = run_fixed_point_analysis()
    
    # Run finite size scaling
    scale_results = finite_size_scaling_test()
    
    print("\n" + "="*80)
    print("SUMMARY FOR PAPER REVISION")
    print("="*80)
    
    print("\nBased on this analysis, the paper should:")
    print("1. Clarify whether C_L₀ is claimed as topological fixed point or statistical mean")
    print("2. Provide evidence for non-zero asymptotic systematic offset")
    print("3. Show convergence behavior with increasing system count")
    print("4. Distinguish between statistical fluctuations and topological invariance")
    
    print("\nKey finding from our test:")
    last_row = df_results.iloc[-1]
    print(f"With N = {last_row['N']:,} systems:")
    print(f"  C_L₀ estimate: {last_row['C_L0_ideal']:.6f}")
    print(f"  Systematic offset: {last_row['error_without']:.3%}")
    print(f"  This offset is {'non-zero' if last_row['error_without'] > 0.01 else 'small'}")
    
    if last_row['error_without'] > 0.01:
        print("\n✅ Supports paper's claim: True systematic offset exists")
        print("   C_L₀ appears to be a topological fixed point")
    else:
        print("\n⚠ Challenges paper's claim: Offset diminishes with large N")
        print("   C_L₀ appears statistical rather than topological")
