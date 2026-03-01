#!/usr/bin/env python3
"""
modular_arithmetic_analysis.py
==============================

Analyse the modular arithmetic structure of raw (integer-valued) sink
coefficients from Goldbach-Lemoine Descent Graphs.

The ILR / simplex analysis works with normalised log-ratios and deliberately
erases absolute magnitudes.  This script works with the raw integers W_s(N)
and exploits two properties invisible to the compositional lens:

    (A) Reconstruction identity:  sum_s  s * W_s(N)  =  N
    (B) Recurrence:  W_s(N) = W_s(p) + 2*W_s(q)  for odd N = p + 2q

Taking (A) and (B) modulo small integers yields modular constraints that
relate the residue class of N to the residues of individual coefficients.

Experiments
-----------
  1. Reconstruction sanity check
  2. Parity fingerprint: W_s(N) mod 2  vs  N mod 6
  3. Mod-3 fingerprint:  W_s(N) mod 3  vs  N mod 6
  4. Mod-6 fingerprint:  W_s(N) mod 6  vs  N mod 6  (full period)
  5. Mutual information:  I( N mod m ; W-vector mod k )  for several (m, k)
  6. Cross-sink GCD structure, conditioned on N mod 6
  7. Coefficient growth scaling by residue class
  8. Cross-policy modular agreement: do policies agree on W_s mod k?
  9. Modular reconstruction constraints (analytic + empirical)
 10. Total-mass (sum W_s) modular structure

Input:  CSV with columns  N  and  W_{policy}_{sink}
        (same format as descent_graph_sink_weights.py output)

Output: Console tables + figures in --outdir

Usage:
    python modular_arithmetic_analysis.py data.csv [--outdir ./modular_figs]
           [--policies down,up,quarter,center,random]

Requirements: numpy, pandas, matplotlib
"""

import argparse
import math
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# ────────────────────────────────────────────────────────────────
# Column detection  (same regex as serial_dependence_analysis.py)
# ────────────────────────────────────────────────────────────────

WEIGHT_RE = re.compile(r"^W_(?P<pol>[a-zA-Z]+)_(?P<s>\d+)$")


def detect_policies(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Return {policy: [col_names sorted by sink]}."""
    groups: Dict[str, List[Tuple[int, str]]] = {}
    for c in df.columns:
        m = WEIGHT_RE.match(c)
        if m:
            pol = m.group("pol").lower()
            sink = int(m.group("s"))
            groups.setdefault(pol, []).append((sink, c))
    return {p: [c for _, c in sorted(v)] for p, v in groups.items()}


def detect_sinks(cols: List[str]) -> List[int]:
    """Extract sink integers from column names like W_down_2."""
    sinks = []
    for c in cols:
        m = WEIGHT_RE.match(c)
        if m:
            sinks.append(int(m.group("s")))
    return sorted(set(sinks))


def get_weight_matrix(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    """Return T x D integer weight matrix."""
    return df[cols].values.astype(np.int64)


# ────────────────────────────────────────────────────────────────
# Utility: contingency table, mutual information, GCD
# ────────────────────────────────────────────────────────────────

def contingency(a: np.ndarray, b: np.ndarray,
                n_a: int, n_b: int) -> np.ndarray:
    """Contingency table counts[i, j] = #{a == i and b == j}."""
    C = np.zeros((n_a, n_b), dtype=np.int64)
    for i in range(n_a):
        mask = a == i
        for j in range(n_b):
            C[i, j] = np.sum(b[mask] == j)
    return C


def mutual_info_discrete(a: np.ndarray, b: np.ndarray,
                         n_a: int, n_b: int) -> float:
    """Mutual information I(A; B) in bits from two integer arrays."""
    C = contingency(a, b, n_a, n_b).astype(np.float64)
    N = C.sum()
    if N == 0:
        return 0.0
    p_joint = C / N
    p_a = p_joint.sum(axis=1, keepdims=True)
    p_b = p_joint.sum(axis=0, keepdims=True)
    denom = p_a * p_b
    mask = (p_joint > 0) & (denom > 0)
    mi = np.sum(p_joint[mask] * np.log2(p_joint[mask] / denom[mask]))
    return float(mi)


def entropy_discrete(a: np.ndarray, n_a: int) -> float:
    """Shannon entropy H(A) in bits."""
    counts = np.bincount(a, minlength=n_a).astype(np.float64)
    p = counts / counts.sum()
    p = p[p > 0]
    return -float(np.sum(p * np.log2(p)))


def vec_gcd(a: int, b: int) -> int:
    return math.gcd(abs(int(a)), abs(int(b)))


# ────────────────────────────────────────────────────────────────
# Experiment implementations
# ────────────────────────────────────────────────────────────────

def exp1_reconstruction(df, groups, sinks_by_pol, N_vals, outdir):
    """Verify reconstruction identity: sum_s  s * W_s(N)  =  N."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Reconstruction identity verification")
    print("=" * 70)

    for pol, cols in groups.items():
        W = get_weight_matrix(df, cols)
        sinks = sinks_by_pol[pol]
        recon = W @ np.array(sinks, dtype=np.int64)
        mismatches = np.sum(recon != N_vals)
        print(f"  {pol.upper():>8s}: {mismatches} mismatches out of {len(N_vals)} "
              f"({'PASS' if mismatches == 0 else 'FAIL'})")
        if mismatches > 0:
            bad = np.where(recon != N_vals)[0][:5]
            for idx in bad:
                print(f"    N={N_vals[idx]}: recon={recon[idx]}, diff={recon[idx]-N_vals[idx]}")


def exp2_parity_fingerprint(df, groups, sinks_by_pol, N_vals, outdir):
    """W_s(N) mod 2 vs N mod 6 contingency tables."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Parity fingerprint — W_s(N) mod 2 vs N mod 6")
    print("=" * 70)

    mod6 = N_vals % 6

    for pol, cols in groups.items():
        W = get_weight_matrix(df, cols)
        sinks = sinks_by_pol[pol]
        D = len(sinks)
        print(f"\n  {pol.upper()}:")

        # For each sink, compute the conditional distribution P(W_s even | N mod 6 = r)
        print(f"    {'Sink':>6s}", end="")
        for r in range(6):
            print(f"  r={r:d}(%even)", end="")
        print("   deterministic?")

        n_deterministic = 0
        for j in range(D):
            w_mod2 = W[:, j] % 2
            fracs = []
            is_det = True
            for r in range(6):
                mask = mod6 == r
                frac_even = np.mean(w_mod2[mask] == 0)
                fracs.append(frac_even)
                if 0.001 < frac_even < 0.999:
                    is_det = False
            n_deterministic += int(is_det)
            det_str = "YES" if is_det else "no"
            print(f"    {sinks[j]:6d}", end="")
            for f in fracs:
                print(f"  {f:10.3f}", end="")
            print(f"   {det_str}")

        print(f"    Deterministic sinks: {n_deterministic}/{D}")


def exp3_mod3_fingerprint(df, groups, sinks_by_pol, N_vals, outdir):
    """W_s(N) mod 3 vs N mod 6 contingency heatmaps."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Mod-3 fingerprint — W_s(N) mod 3 vs N mod 6")
    print("=" * 70)

    mod6 = N_vals % 6
    policies = sorted(groups.keys())

    for pol in policies:
        cols = groups[pol]
        W = get_weight_matrix(df, cols)
        sinks = sinks_by_pol[pol]
        D = len(sinks)

        print(f"\n  {pol.upper()}:")
        print(f"    {'Sink':>6s}", end="")
        for r in range(6):
            print(f"   r={r}(0/1/2)", end="")
        print()

        for j in range(D):
            w_mod3 = W[:, j] % 3
            print(f"    {sinks[j]:6d}", end="")
            for r in range(6):
                mask = mod6 == r
                counts = np.bincount(w_mod3[mask], minlength=3)
                pcts = counts / counts.sum() * 100
                print(f"   {pcts[0]:3.0f}/{pcts[1]:3.0f}/{pcts[2]:3.0f}", end="")
            print()

    # Heatmap figure: for each policy, show a (D x 6) grid of dominant W_s mod 3 class
    n_pol = len(policies)
    fig, axes = plt.subplots(1, n_pol, figsize=(4 * n_pol, 5))
    if n_pol == 1:
        axes = [axes]

    for pi, pol in enumerate(policies):
        cols = groups[pol]
        W = get_weight_matrix(df, cols)
        sinks = sinks_by_pol[pol]
        D = len(sinks)

        # For each (sink, residue class), compute entropy of W_s mod 3
        H_grid = np.zeros((D, 6))
        for j in range(D):
            w_mod3 = W[:, j] % 3
            for r in range(6):
                mask = mod6 == r
                counts = np.bincount(w_mod3[mask], minlength=3).astype(float)
                p = counts / counts.sum()
                p = p[p > 0]
                H_grid[j, r] = -np.sum(p * np.log2(p))

        ax = axes[pi]
        im = ax.imshow(H_grid, aspect="auto", cmap="YlOrRd_r",
                       vmin=0, vmax=np.log2(3))
        ax.set_xticks(range(6))
        ax.set_xticklabels([f"{r}" for r in range(6)])
        ax.set_xlabel("N mod 6")
        ax.set_yticks(range(D))
        ax.set_yticklabels([str(s) for s in sinks], fontsize=8)
        ax.set_ylabel("Sink")
        ax.set_title(f"{pol.upper()}", fontsize=10)
        plt.colorbar(im, ax=ax, label="H(W_s mod 3) bits", shrink=0.8)

    fig.suptitle("Entropy of W_s mod 3 conditioned on N mod 6\n"
                 "(0 = deterministic, log₂3 ≈ 1.58 = uniform)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_mod3_entropy_heatmap.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print(f"\n  -> fig_mod3_entropy_heatmap.png")


def exp4_mod6_fingerprint(df, groups, sinks_by_pol, N_vals, outdir):
    """W_s(N) mod 6 vs N mod 6 — full-period contingency."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Mod-6 fingerprint — W_s(N) mod 6 vs N mod 6")
    print("=" * 70)
    print("  (Combined parity + mod-3 structure)")

    mod6 = N_vals % 6
    policies = sorted(groups.keys())

    # For each policy, compute a single summary: average entropy of
    # W_s mod 6 | N mod 6, across sinks.
    print(f"\n  {'Policy':>8s}  {'Mean H(Ws mod 6|N mod 6)':>26s}  "
          f"{'Max possible':>14s}  {'Efficiency':>12s}")
    max_H = np.log2(6)

    for pol in policies:
        cols = groups[pol]
        W = get_weight_matrix(df, cols)
        sinks = sinks_by_pol[pol]
        D = len(sinks)

        entropies = []
        for j in range(D):
            w_mod6 = W[:, j] % 6
            for r in range(6):
                mask = mod6 == r
                counts = np.bincount(w_mod6[mask], minlength=6).astype(float)
                p = counts / counts.sum()
                p = p[p > 0]
                entropies.append(-np.sum(p * np.log2(p)))

        mean_H = np.mean(entropies)
        print(f"  {pol.upper():>8s}  {mean_H:26.4f}  {max_H:14.4f}  "
              f"{mean_H / max_H * 100:11.1f}%")


def exp5_mutual_information(df, groups, sinks_by_pol, N_vals, outdir):
    """Mutual information I(N mod m ; W-vector mod k) for several (m, k)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Mutual information I(N mod m ; W_vec mod k)")
    print("=" * 70)
    print("  How much does the coefficient residue vector tell you about N's residue class?")

    policies = sorted(groups.keys())
    mk_pairs = [(2, 2), (3, 3), (6, 2), (6, 3), (6, 6)]

    for m_mod, k_mod in mk_pairs:
        print(f"\n  --- m={m_mod}, k={k_mod} ---")
        H_N = entropy_discrete(N_vals % m_mod, m_mod)
        print(f"  H(N mod {m_mod}) = {H_N:.4f} bits")
        print(f"  {'Policy':>8s}  {'I(N;W_vec)':>12s}  {'NMI':>8s}  "
              f"{'per-sink MI (top 3)':>40s}")

        for pol in policies:
            cols = groups[pol]
            W = get_weight_matrix(df, cols)
            sinks = sinks_by_pol[pol]
            D = len(sinks)

            # Joint MI: encode the full W mod k vector as a single integer
            # (for small k and D, this is tractable; for large, we subsample)
            N_mod = N_vals % m_mod
            W_mod = W % k_mod

            # Per-sink MI
            per_sink_mi = []
            for j in range(D):
                mi_j = mutual_info_discrete(N_mod, W_mod[:, j], m_mod, k_mod)
                per_sink_mi.append((sinks[j], mi_j))
            per_sink_mi.sort(key=lambda x: -x[1])
            top3 = ", ".join(f"s={s}: {mi:.4f}" for s, mi in per_sink_mi[:3])

            # Joint MI via hashing (if k^D is tractable)
            if k_mod ** D <= 100_000:
                # Encode vector as base-k integer
                multipliers = k_mod ** np.arange(D)
                W_hash = (W_mod * multipliers[None, :]).sum(axis=1)
                n_states = k_mod ** D
                joint_mi = mutual_info_discrete(N_mod, W_hash, m_mod, n_states)
            else:
                # Use sum of per-sink MI as lower bound
                joint_mi = sum(mi for _, mi in per_sink_mi)

            nmi = joint_mi / H_N if H_N > 0 else 0.0
            print(f"  {pol.upper():>8s}  {joint_mi:12.4f}  {nmi:8.3f}  {top3:>40s}")


def exp6_gcd_structure(df, groups, sinks_by_pol, N_vals, outdir):
    """Cross-sink GCD patterns conditioned on N mod 6."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Cross-sink GCD structure")
    print("=" * 70)
    print("  Mean gcd(W_s, W_{s'}) conditioned on N mod 6, across all sink pairs")

    mod6 = N_vals % 6
    policies = sorted(groups.keys())
    T = len(N_vals)

    # Subsample for speed if T is large (GCD is per-row)
    max_rows = min(T, 20_000)
    if T > max_rows:
        rng = np.random.default_rng(42)
        idx = rng.choice(T, max_rows, replace=False)
        idx.sort()
        print(f"  (Subsampled {max_rows} of {T} rows for GCD computation)")
    else:
        idx = np.arange(T)

    for pol in policies:
        cols = groups[pol]
        W = get_weight_matrix(df, cols)[idx]
        sinks = sinks_by_pol[pol]
        D = len(sinks)
        mod6_sub = mod6[idx]

        # For each residue class, compute mean GCD across all pairs
        print(f"\n  {pol.upper()}:")
        print(f"    {'Pair':>12s}", end="")
        for r in range(6):
            print(f"  r={r:d}", end="")
        print("   overall")

        # Pick a few representative pairs (first×last, adjacent, etc.)
        # Plus global average
        pair_list = []
        if D >= 2:
            pair_list.append((0, D - 1))       # first × last sink
            pair_list.append((0, 1))            # first two sinks
            pair_list.append((D // 2, D // 2 + 1))  # middle pair
        # Also report the overall average
        global_gcds = {r: [] for r in range(6)}
        global_gcds["all"] = []

        for j1 in range(D):
            for j2 in range(j1 + 1, D):
                row_gcds = np.array([vec_gcd(W[i, j1], W[i, j2])
                                     for i in range(len(W))])
                for r in range(6):
                    mask = mod6_sub == r
                    if mask.any():
                        global_gcds[r].extend(row_gcds[mask].tolist())
                global_gcds["all"].extend(row_gcds.tolist())

                if (j1, j2) in pair_list:
                    label = f"({sinks[j1]},{sinks[j2]})"
                    print(f"    {label:>12s}", end="")
                    for r in range(6):
                        mask = mod6_sub == r
                        mean_g = np.mean(row_gcds[mask]) if mask.any() else 0
                        print(f"  {mean_g:4.1f}", end="")
                    print(f"   {np.mean(row_gcds):4.1f}")

        # Overall average
        print(f"    {'AVERAGE':>12s}", end="")
        for r in range(6):
            print(f"  {np.mean(global_gcds[r]):4.1f}" if global_gcds[r] else "   ---", end="")
        print(f"   {np.mean(global_gcds['all']):4.1f}")

        # Distribution of GCDs
        all_g = np.array(global_gcds["all"])
        gcd_vals, gcd_counts = np.unique(all_g, return_counts=True)
        top_idx = np.argsort(-gcd_counts)[:8]
        top_str = ", ".join(f"gcd={gcd_vals[i]}: {gcd_counts[i]/len(all_g)*100:.1f}%"
                            for i in top_idx)
        print(f"    GCD distribution (top 8): {top_str}")


def exp7_growth_scaling(df, groups, sinks_by_pol, N_vals, outdir):
    """Coefficient growth rates by residue class."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 7: Coefficient growth scaling by residue class")
    print("=" * 70)
    print("  E[W_s(N)] / N  as a function of N, stratified by N mod 6")

    mod6 = N_vals % 6
    policies = sorted(groups.keys())

    # For consecutive-integer blocks, bin by N and compute conditional means
    # For wide-range samples, sort by N and use sliding windows
    N_sorted_idx = np.argsort(N_vals)
    N_sorted = N_vals[N_sorted_idx]

    n_pol = len(policies)
    # Pick representative sinks: smallest, largest, and the one carrying most mass
    for pol in policies:
        cols = groups[pol]
        W = get_weight_matrix(df, cols)
        sinks = sinks_by_pol[pol]
        D = len(sinks)

        # Mean proportion of each sink (mass share = s * W_s / N)
        mass_share = np.zeros(D)
        for j in range(D):
            mass_share[j] = np.mean(sinks[j] * W[:, j].astype(float) / N_vals)

        top_j = np.argsort(-mass_share)[:3]
        print(f"\n  {pol.upper()} — top 3 sinks by mass share: "
              + ", ".join(f"s={sinks[j]} ({mass_share[j]*100:.1f}%)" for j in top_j))

        # For each residue class, compute mean W_s / (N / s) ≈ share
        print(f"    {'Sink':>6s}  {'share':>6s}", end="")
        for r in range(6):
            print(f"  r={r:d}", end="")
        print()

        for j in range(D):
            ratio = W[:, j].astype(float) * sinks[j] / N_vals
            print(f"    {sinks[j]:6d}  {mass_share[j]:6.3f}", end="")
            for r in range(6):
                mask = mod6 == r
                print(f"  {np.mean(ratio[mask]):.3f}", end="")
            print()

    # Figure: for one policy (first), plot W_s/N vs N for top 3 sinks, colored by mod 6
    pol0 = policies[0]
    cols0 = groups[pol0]
    W0 = get_weight_matrix(df, cols0)
    sinks0 = sinks_by_pol[pol0]
    D0 = len(sinks0)
    mass0 = np.array([np.mean(sinks0[j] * W0[:, j].astype(float) / N_vals) for j in range(D0)])
    top3 = np.argsort(-mass0)[:3]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors6 = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]

    for ai, j in enumerate(top3):
        ax = axes[ai]
        ratio = W0[:, j].astype(float) * sinks0[j] / N_vals
        for r in range(6):
            mask = mod6 == r
            order = np.argsort(N_vals[mask])
            ax.plot(N_vals[mask][order], ratio[mask][order],
                    ".", markersize=0.5, alpha=0.3, color=colors6[r],
                    label=f"r={r}" if ai == 0 else None)
        ax.set_xlabel("N")
        ax.set_ylabel(f"s·W_s / N")
        ax.set_title(f"sink {sinks0[j]} (share={mass0[j]:.3f})")
        ax.grid(True, alpha=0.2)

    if len(axes) > 0:
        axes[0].legend(fontsize=7, markerscale=5, ncol=3, title="N mod 6")
    fig.suptitle(f"Mass share evolution — {pol0.upper()}", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_growth_scaling.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print(f"\n  -> fig_growth_scaling.png")


def exp8_cross_policy_agreement(df, groups, sinks_by_pol, N_vals, outdir):
    """Do different policies agree on W_s(N) mod k?"""
    print("\n" + "=" * 70)
    print("EXPERIMENT 8: Cross-policy modular agreement")
    print("=" * 70)
    print("  Fraction of integers where W_s(N) mod k agrees across policy pairs")

    policies = sorted(groups.keys())
    if len(policies) < 2:
        print("  (Need at least 2 policies; skipping)")
        return

    # Find common sinks across all policies
    all_sinks = [set(sinks_by_pol[p]) for p in policies]
    common_sinks = sorted(set.intersection(*all_sinks))
    if not common_sinks:
        print("  No common sinks across policies; skipping")
        return

    print(f"  Common sinks: {common_sinks}")

    for k_mod in [2, 3, 6]:
        print(f"\n  --- mod {k_mod} ---")

        # Build W mod k matrices for each policy (using common sinks only)
        W_mod = {}
        for pol in policies:
            cols = groups[pol]
            sinks = sinks_by_pol[pol]
            W_full = get_weight_matrix(df, cols)
            # Extract columns for common sinks
            common_idx = [sinks.index(s) for s in common_sinks]
            W_mod[pol] = W_full[:, common_idx] % k_mod

        # Pairwise agreement
        print(f"    {'Pair':>20s}  {'exact match':>12s}  {'per-sink agreement':>20s}")
        for i, p1 in enumerate(policies):
            for p2 in policies[i + 1:]:
                exact_match = np.all(W_mod[p1] == W_mod[p2], axis=1).mean()
                per_sink = np.mean(W_mod[p1] == W_mod[p2], axis=0)
                per_sink_str = ", ".join(f"{common_sinks[j]}:{per_sink[j]:.3f}"
                                        for j in range(min(4, len(common_sinks))))
                print(f"    {p1+'/'+p2:>20s}  {exact_match:12.4f}  {per_sink_str}")


def exp9_modular_constraints(df, groups, sinks_by_pol, N_vals, outdir):
    """
    Analytic + empirical modular reconstruction constraints.

    From sum_s s * W_s = N, taking mod k gives:
        sum_s (s mod k) * (W_s mod k)  ≡  N mod k   (mod k)

    For k=2: the even sinks (2) contribute 0, odd sinks contribute W_s mod 2.
    For k=3: sink residues mod 3 give a linear constraint on W_s mod 3.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 9: Modular reconstruction constraints")
    print("=" * 70)
    print("  From Σ s·W_s = N (mod k), derive and verify constraints")

    policies = sorted(groups.keys())

    for pol in policies:
        cols = groups[pol]
        W = get_weight_matrix(df, cols)
        sinks = sinks_by_pol[pol]
        D = len(sinks)
        sinks_arr = np.array(sinks, dtype=np.int64)

        print(f"\n  {pol.upper()}:")

        for k in [2, 3, 5, 6]:
            # Analytic: s mod k for each sink
            s_mod_k = sinks_arr % k
            # Empirical: check (sum (s mod k) * (W_s mod k)) mod k == N mod k
            lhs = np.zeros(len(N_vals), dtype=np.int64)
            for j in range(D):
                lhs = (lhs + s_mod_k[j] * (W[:, j] % k)) % k
            rhs = N_vals % k
            agreement = np.mean(lhs == rhs)

            # Which sinks vanish mod k? (s ≡ 0 mod k → contributes nothing)
            vanishing = [sinks[j] for j in range(D) if s_mod_k[j] == 0]
            active = [(sinks[j], int(s_mod_k[j])) for j in range(D) if s_mod_k[j] != 0]

            print(f"    mod {k}: agreement = {agreement:.6f}  "
                  f"(expect 1.000000 by identity)")
            print(f"      vanishing sinks (s≡0 mod {k}): {vanishing}")
            print(f"      active sinks (s mod {k}): "
                  + ", ".join(f"{s}→{r}" for s, r in active[:8])
                  + ("..." if len(active) > 8 else ""))

            # For mod 2: how many active sinks have deterministic parity?
            if k == 2:
                mod6 = N_vals % 6
                n_det = 0
                for j in range(D):
                    if s_mod_k[j] == 0:
                        continue
                    w_mod2 = W[:, j] % 2
                    # Check if parity is determined by N mod 6
                    determined = True
                    for r in range(6):
                        mask = mod6 == r
                        vals = np.unique(w_mod2[mask])
                        if len(vals) > 1:
                            determined = False
                            break
                    if determined:
                        n_det += 1
                n_active = sum(1 for j in range(D) if s_mod_k[j] != 0)
                print(f"      parity determined by N mod 6: {n_det}/{n_active} active sinks")


def exp10_total_mass(df, groups, sinks_by_pol, N_vals, outdir):
    """Total mass (sum of all W_s) modular structure."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 10: Total mass Σ W_s — modular structure")
    print("=" * 70)
    print("  Total mass = sum of all coefficients (unweighted by sink value)")

    mod6 = N_vals % 6
    policies = sorted(groups.keys())

    for k in [2, 3, 6]:
        print(f"\n  --- Σ W_s mod {k} vs N mod 6 ---")
        print(f"    {'Policy':>8s}", end="")
        for r in range(6):
            labels = [str(v) for v in range(k)]
            print(f"  r={r}({'/'.join(labels)})", end="")
        print()

        for pol in policies:
            cols = groups[pol]
            W = get_weight_matrix(df, cols)
            total = W.sum(axis=1)
            total_mod_k = total % k

            print(f"    {pol.upper():>8s}", end="")
            for r in range(6):
                mask = mod6 == r
                counts = np.bincount(total_mod_k[mask], minlength=k)
                pcts = counts / counts.sum() * 100
                pct_str = "/".join(f"{p:.0f}" for p in pcts)
                print(f"  {pct_str:>{4+3*(k-1)}s}", end="")
            print()

    # Figure: distribution of total mass by policy and residue class
    n_pol = len(policies)
    fig, axes = plt.subplots(1, n_pol, figsize=(4 * n_pol, 4))
    if n_pol == 1:
        axes = [axes]
    colors6 = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]

    for pi, pol in enumerate(policies):
        cols = groups[pol]
        W = get_weight_matrix(df, cols)
        total = W.sum(axis=1).astype(float)
        # Normalise by N to get "branching density"
        density = total / N_vals

        ax = axes[pi]
        for r in range(6):
            mask = mod6 == r
            ax.hist(density[mask], bins=60, alpha=0.4, color=colors6[r],
                    label=f"r={r}", density=True)
        ax.set_xlabel("Σ W_s / N")
        ax.set_ylabel("density")
        ax.set_title(f"{pol.upper()}", fontsize=10)
        if pi == 0:
            ax.legend(fontsize=6, title="N mod 6")
        ax.grid(True, alpha=0.2)

    fig.suptitle("Total-mass density Σ W_s / N by residue class", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_total_mass_density.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print(f"\n  -> fig_total_mass_density.png")


# ────────────────────────────────────────────────────────────────
# Summary figure: mod-2 determinism heatmap across all policies
# ────────────────────────────────────────────────────────────────

def summary_parity_heatmap(df, groups, sinks_by_pol, N_vals, outdir):
    """Heatmap: P(W_s even | N mod 6) for all policies and sinks."""
    mod6 = N_vals % 6
    policies = sorted(groups.keys())

    n_pol = len(policies)
    fig, axes = plt.subplots(1, n_pol, figsize=(4 * n_pol, 5))
    if n_pol == 1:
        axes = [axes]

    for pi, pol in enumerate(policies):
        cols = groups[pol]
        W = get_weight_matrix(df, cols)
        sinks = sinks_by_pol[pol]
        D = len(sinks)

        grid = np.zeros((D, 6))
        for j in range(D):
            w_mod2 = W[:, j] % 2
            for r in range(6):
                mask = mod6 == r
                grid[j, r] = np.mean(w_mod2[mask] == 0)

        ax = axes[pi]
        im = ax.imshow(grid, aspect="auto", cmap="RdYlBu", vmin=0, vmax=1)
        ax.set_xticks(range(6))
        ax.set_xticklabels([str(r) for r in range(6)])
        ax.set_xlabel("N mod 6")
        ax.set_yticks(range(D))
        ax.set_yticklabels([str(s) for s in sinks], fontsize=8)
        ax.set_ylabel("Sink")
        ax.set_title(f"{pol.upper()}", fontsize=10)

        # Annotate cells near 0 or 1 (deterministic)
        for j in range(D):
            for r in range(6):
                v = grid[j, r]
                if v > 0.99 or v < 0.01:
                    ax.text(r, j, f"{v:.0f}", ha="center", va="center",
                            fontsize=7, fontweight="bold",
                            color="white" if v < 0.5 else "black")

        plt.colorbar(im, ax=ax, label="P(W_s even)", shrink=0.8)

    fig.suptitle("Parity determinism: P(W_s even | N mod 6)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_parity_heatmap.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print(f"\n  -> fig_parity_heatmap.png")


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Modular arithmetic analysis of raw Descent Graph "
                    "sink coefficients.")
    parser.add_argument("input_csv",
                        help="CSV with columns N and W_{policy}_{sink}")
    parser.add_argument("--outdir", default=".",
                        help="Output directory for figures (default: cwd)")
    parser.add_argument("--policies", default=None,
                        help="Comma-separated subset of policies to analyse "
                             "(default: all detected)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load
    df = pd.read_csv(args.input_csv)
    if "N" not in df.columns:
        sys.exit("ERROR: CSV must contain a column named 'N'.")

    df = df.sort_values("N").reset_index(drop=True)
    N_vals = df["N"].values.astype(np.int64)

    all_groups = detect_policies(df)
    if args.policies:
        keep = {p.strip().lower() for p in args.policies.split(",")}
        all_groups = {p: v for p, v in all_groups.items() if p in keep}

    if not all_groups:
        sys.exit("ERROR: No matching policy columns found.")

    # Filter to rows where all selected policies have ok == 1
    for pol in list(all_groups.keys()):
        ok_col = f"ok_{pol}"
        if ok_col in df.columns:
            mask = df[ok_col] == 1
            if not mask.all():
                n_drop = (~mask).sum()
                print(f"  Filtering {n_drop} rows where ok_{pol} != 1")
                df = df[mask].reset_index(drop=True)
                N_vals = df["N"].values.astype(np.int64)

    # Detect sinks per policy
    sinks_by_pol = {}
    for pol, cols in all_groups.items():
        sinks_by_pol[pol] = detect_sinks(cols)

    policies_str = ", ".join(f"{p.upper()} ({len(sinks_by_pol[p])} sinks)"
                             for p in sorted(all_groups.keys()))
    print(f"Loaded {len(N_vals)} rows: N = {N_vals[0]:,} .. {N_vals[-1]:,}")
    print(f"Policies: {policies_str}")
    print(f"Output:   {args.outdir}")

    # Run experiments
    exp1_reconstruction(df, all_groups, sinks_by_pol, N_vals, args.outdir)
    exp2_parity_fingerprint(df, all_groups, sinks_by_pol, N_vals, args.outdir)
    exp3_mod3_fingerprint(df, all_groups, sinks_by_pol, N_vals, args.outdir)
    exp4_mod6_fingerprint(df, all_groups, sinks_by_pol, N_vals, args.outdir)
    exp5_mutual_information(df, all_groups, sinks_by_pol, N_vals, args.outdir)
    exp6_gcd_structure(df, all_groups, sinks_by_pol, N_vals, args.outdir)
    exp7_growth_scaling(df, all_groups, sinks_by_pol, N_vals, args.outdir)
    exp8_cross_policy_agreement(df, all_groups, sinks_by_pol, N_vals, args.outdir)
    exp9_modular_constraints(df, all_groups, sinks_by_pol, N_vals, args.outdir)
    exp10_total_mass(df, all_groups, sinks_by_pol, N_vals, args.outdir)

    # Summary figures
    summary_parity_heatmap(df, all_groups, sinks_by_pol, N_vals, args.outdir)

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
