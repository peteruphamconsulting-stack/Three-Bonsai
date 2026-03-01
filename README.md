# Goldbach–Lemoine Descent Graphs

Code repository for the paper *Goldbach–Lemoine Descent Graphs* by Peter Upham.

Given an integer $N$, a **Descent Graph** is a directed acyclic graph constructed by iterated Lemoine ($M = p + 2q$) and Goldbach ($N = a + b$) decomposition with witness selection constrained to the window $q \in [\lceil M/5 \rceil, \lfloor M/3 \rfloor]$. The recursion terminates at a universal set of 10 (or 12) sink primes. The normalized sink-weight vector is an **arithmetic signature** of $N$, which lives on a simplex and is analyzed via isometric log-ratio (ILR) embedding.

---

## Pipeline Overview

The analysis proceeds in five stages. Each stage reads from the output of the previous one.

```
┌─────────────────────────────────┐
│ 1. descent_graph_sink_weights   │  Build descent graphs, output raw sink weights
│    → sink_weights.csv           │
└──────────────┬──────────────────┘
               │
       ┌───────┼────────────────────────────────┐
       ▼       ▼                                ▼
┌────────────────┐ ┌──────────────────────────┐ ┌──────────────────────────────┐
│ 2. ilr_        │ │ 3. persistence_homology   │ │ 4. modular_arithmetic_       │
│ intrinsic_     │ │    → TDA stats CSV +      │ │    analysis                  │
│ dimension      │ │      persistence diagrams │ │    → parity/mod-3/mod-6      │
│ → dim CSVs     │ └──────────────────────────┘ │      fingerprints, MI tables  │
│   + plots      │                              └──────────────────────────────┘
└───────┬────────┘
        ▼
┌────────────────────────────────┐
│ 5. scale_invariance_dimension  │
│    → d_hat vs log10(N) bands   │
└────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────┐
│ 6. serial_dependence_analysis                     │
│    → entropy, ACF, Hurst exponents, twin primes  │
└──────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────┐
│ 7. pseudocount_sensitivity_and_hurst_crossval     │
│    → ε-sensitivity sweep, R/S cross-validation   │
└──────────────────────────────────────────────────┘

Standalone (reads stage 1 machinery, no CSV input):
┌──────────────────────────────────────────────────┐
│ 8. coarsening_matrix_verification                 │
│    → coarsening matrix C(S12,S10), LaTeX table   │
└──────────────────────────────────────────────────┘
```

---

## Files

| Script | Purpose | Input | Key outputs |
|---|---|---|---|
| `descent_graph_sink_weights.py` | Construct descent graphs under each policy; compute raw sink-weight vectors | CLI parameters (N range, policies, sink model) | CSV with columns `N`, `parity`, `is_prime`, per-policy sink weights, witness traces, branching stats |
| `ilr_intrinsic_dimension.py` | ILR transform → kNN-MLE and correlation dimension estimation | Sink-weight CSV from stage 1 | Dimension summary CSVs, ILR `.npy` arrays, comparison plots |
| `persistence_homology.py` | Persistent homology (Vietoris–Rips, H₁ and H₂) on ILR/CLR embeddings | Sink-weight CSV from stage 1 | Per-replicate stats CSV, persistence diagram plots |
| `modular_arithmetic_analysis.py` | Modular structure of raw sink coefficients: parity skeleton, mod-3 cascade, mutual information, cross-policy agreement | Sink-weight CSV from stage 1 | Console tables, fingerprint heatmaps, MI summary |
| `scale_invariance_dimension.py` | Intrinsic dimension stability across log₁₀ magnitude bands | Sink-weight CSV from stage 1 | Band × policy dimension table, heatmap, multi-k plots, summary CSV |
| `serial_dependence_analysis.py` | Entropy stratification, ACF/spectrum, DFA Hurst exponents, prime/twin-prime signatures | Sink-weight CSV from stage 1 | Console summary + 3 figure PNGs |
| `pseudocount_sensitivity_and_hurst_crossval.py` | Pseudocount ε-sensitivity sweep; R/S cross-validation of Hurst; primorial spectral check | Sink-weight CSV from stage 1 | Console summary + 3 figure PNGs |
| `coarsening_matrix_verification.py` | Compute coarsening matrix C(S₁₂, S₁₀) and verify the identity W\_S₁₀ = C · W\_S₁₂ | Imports `descent_graph_sink_weights.py` directly | LaTeX table, matrix CSV, verification CSV, summary |

---

## Requirements

**Python ≥ 3.9**

Core (all scripts):
```
numpy
pandas
matplotlib
```

Additional (specific scripts):
```
scikit-learn     # ilr_intrinsic_dimension.py, scale_invariance_dimension.py, persistence_homology.py (optional PCA)
ripser           # persistence_homology.py
persim           # persistence_homology.py (diagram plotting only)
```

Install all at once:
```bash
pip install numpy pandas matplotlib scikit-learn ripser persim
```

---

## Usage

### Stage 1 — Generate sink weights

```bash
# 10-sink, consecutive band of 5000 integers near 5×10⁶, all parities, three policies
python descent_graph_sink_weights.py \
    --sinks 10 \
    --start 5000001 --end 5005000 \
    --sample_n 0 \
    --include_even 1 \
    --policies down,up,quarter \
    --out_csv sink_weights_5M.csv

# 12-sink, stratified random sample of 10k integers, with center policy at 4/15
python descent_graph_sink_weights.py \
    --sinks 12 \
    --start 50000 --end 500000000 \
    --sample_n 10000 \
    --include_even 1 \
    --policies down,up,quarter,center \
    --center 4/15 \
    --out_csv sink_weights_12sink.csv
```

Key flags:
- `--sinks 10|12` — terminal basis (10-sink: {2,3,5,19,29,37,47,59,73,97}; 12-sink: {11,13,...,97})
- `--sample_n 0` — use all integers in range (consecutive); `>0` — random sample of that size
- `--include_even 1` — include even integers (decomposed via Goldbach at the root)
- `--center FRAC` — center fraction for the `center` policy, strictly between 1/5 and 1/3 (e.g., `4/15` or `0.2667`)
- `--trace_witnesses 1` — include depth-1/depth-2 witness columns (default: on)

### Stage 2 — Intrinsic dimension estimation

```bash
python ilr_intrinsic_dimension.py sink_weights_5M.csv
```

Reads the CSV, auto-detects policies from `W_{policy}_{sink}` columns, applies the ILR transform with pseudocount ε = 0.5 (configurable in the `CONFIG` dict), and writes per-policy dimension estimates and plots to a `*_dim/` directory alongside the input CSV.

### Stage 3 — Persistent homology

```bash
# 12-sink ILR, H₂, 800-point subsamples, 10 replicates
python persistence_homology.py sink_weights_12sink.csv \
    --sinks 12 \
    --space ilr \
    --maxdim 2 \
    --samples 800 \
    --reps 10 \
    --eps 0.5 \
    --plot \
    --stats_csv tda_stats.csv
```

Key flags:
- `--space ilr|clr` — compositional transform (ILR recommended)
- `--maxdim 1|2` — compute through H₁ or H₂
- `--samples N` — subsample size per replicate (paper uses 800)
- `--reps K` — number of subsampling replicates for stability
- `--eps` — pseudocount (use 0.5 to match the paper; default is 1e-12)

### Stage 4 — Modular arithmetic analysis

```bash
python modular_arithmetic_analysis.py sink_weights_5M.csv --outdir figures/
```

Analyzes the raw integer-valued sink coefficients modulo small integers, exploiting the reconstruction identity Σ s·W_s(N) = N and the Lemoine recurrence W_s(N) = W_s(p) + 2·W_s(q). Produces parity fingerprints, mod-3 and mod-6 concentration tables, mutual information estimates, cross-policy modular agreement, and coefficient growth scaling by residue class.

### Stage 5 — Dimensional contraction across magnitude

```bash
python scale_invariance_dimension.py sink_weights_logstrat.csv \
    --outdir ./scale_figs \
    --policies down,up,quarter,center,random \
    --k_values 10,15,20,25,30
```

Requires a **log-stratified** sample (use `--sampling log_stratified` in stage 1). Splits the data into log₁₀ magnitude bands and computes kNN-based intrinsic dimension within each band per policy. Produces a band × policy summary table, heatmaps, and multi-k diagnostic plots.

### Stage 6 — Serial dependence analysis

```bash
python serial_dependence_analysis.py sink_weights_5M.csv --outdir figures/
```

Requires **consecutive** integers in the input CSV (not random samples). Auto-detects policies. Produces:
- `fig_entropy_by_residue.png` — Shannon entropy colored by N mod 3
- `fig_acf_spectrum.png` — ACF and power spectrum (raw + mod-6 detrended)
- `fig_hurst_dfa.png` — DFA log-log plots (raw + detrended)

Experiments 5–6 (prime vs. composite signatures, twin prime proximity) require the `is_prime` column in the CSV, which is included by default in `descent_graph_sink_weights.py`.

### Stage 7 — Pseudocount sensitivity & Hurst cross-validation

```bash
python pseudocount_sensitivity_and_hurst_crossval.py sink_weights_5M.csv --outdir figures/
```

Produces:
- `fig_epsilon_sensitivity.png` — dimension, PCA concentration, and Hurst vs. ε
- `fig_spectral_primorial.png` — spectra after mod-6 vs. mod-30 detrending
- `fig_hurst_crossval.png` — DFA(1) vs. R/S, mod-6 vs. mod-30, all ILR coordinates

### Stage 8 — Coarsening matrix verification

```bash
python coarsening_matrix_verification.py \
    --start 5000001 --end 5050000 --sample_n 5000 \
    --include_even 1 \
    --policies down,up,quarter,center --center 4/15 \
    --out_prefix coarsening_results
```

Computes the 10×12 coarsening matrix C(S₁₂, S₁₀) by running the descent of each S₁₂-sink against S₁₀, then verifies W\_S₁₀(N) = C · W\_S₁₂(N) exactly for a sample of integers across all policies. Imports `descent_graph_sink_weights.py` directly (no CSV input needed). Outputs:
- `*_matrix.tex` — LaTeX table for the paper
- `*_matrix.csv` — machine-readable matrix
- `*_verification.csv` — per-(N, policy) pass/fail
- `*_summary.txt` — human-readable summary

---

## Output CSV format (Stage 1)

The sink-weight CSV has the following column groups:

| Column(s) | Description |
|---|---|
| `N` | The integer |
| `parity` | `odd` or `even` |
| `is_prime` | `1` if N is prime, `0` otherwise |
| `ok_{policy}` | `1` if decomposition succeeded, `0` if not |
| `W_{policy}_{sink}` | Raw integer sink weight for each terminal prime under each policy |
| `recon_{policy}` | Reconstruction check: Σ(W × sink) should equal N |
| `T_{policy}_{field}` | Depth-1/2 witness trace (if `--trace_witnesses 1`) |
| `B_{policy}_{stat}` | Branching statistics (max depth, node counts, mass flow by depth) |

The sink weights are **unnormalized integer coefficients**. To obtain the compositional signature, normalize each policy's weight vector to sum to 1. To obtain ILR coordinates, apply the Helmert-basis ILR transform after pseudocount replacement (ε = 0.5) and closure — this is handled automatically by all downstream scripts.

---

## Policies

| Policy | Witness selection rule |
|---|---|---|
| **down** | Scan q downward from ⌊M/3⌋ | 
| **up** | Scan q upward from ⌈M/5⌉ |
| **quarter** | Scan outward from ⌊M/4⌋ | 
| **center** | Scan outward from ⌊f·M⌋ for a chosen f ∈ (1/5, 1/3) | 

---

## Primality testing

All scripts use deterministic Miller–Rabin with bases {2, 3, 5, 7, 11}, which is provably correct for all n < 3.2 × 10¹⁴. This exceeds the paper's computational range (up to 5 × 10⁸) by a factor of >600,000. Results are memoized for performance.

---

## Citation

```bibtex
@article{upham2026gldg,
  title   = {Goldbach--Lemoine Descent Graphs},
  author  = {Upham, Peter},
  year    = {2026},
  note    = {Preprint, ThreeBons.ai}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
