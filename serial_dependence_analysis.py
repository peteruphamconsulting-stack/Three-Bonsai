#!/usr/bin/env python3
"""
serial_dependence_analysis.py
=============================

Analyzes serial dependence structure in Goldbach-Lemoine Descent Graph
sink-weight sequences over consecutive integers.

Experiments:
  1. Shannon entropy by residue class (mod 3 and mod 6)
  2. Autocorrelation and power spectrum of ILR coordinates (raw + mod-6 detrended)
  3. Hurst exponent via DFA (raw + mod-6 detrended)
  4. Cross-policy entropy correlation
  5. Prime vs composite signatures
  6. Twin prime ILR proximity
  7. Secular drift diagnostic (sliding-window median of ILR coordinates)
  8. Three-stage DFA decomposition (raw / mod6 / fully-detrended)
  9. ACF decay classification (power-law vs exponential on fully-detrended)
 10. Within-class shuffle null test for Hurst baseline
 11. Multifractal DFA (MF-DFA): generalized Hurst exponents h(q)
     and singularity spectrum f(alpha)

Input:  CSV with columns N, W_{policy}_{sink} (e.g., W_down_2, W_up_97, ...)
Output: Figure PNGs + console summary

Usage:
    python serial_dependence_analysis.py <path_to_csv> [--outdir DIR]
           [--n_shuffles 200]

Requirements: numpy, pandas, matplotlib
"""

import sys
import math
import re
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────────
# ILR Transform (Helmert basis, Aitchison framework)
# ────────────────────────────────────────────────────────────────

def _helmert_submatrix(D):
    """Helmert contrast matrix of dimension (D-1) x D."""
    H = np.zeros((D - 1, D))
    for k in range(1, D):
        d = math.sqrt(k * (k + 1))
        H[k - 1, :k] = 1.0 / d
        H[k - 1, k] = -k / d
    return H


def ilr_transform(X_raw, eps=0.5):
    """
    Isometric log-ratio transform.
    X_raw: (n, D) array of non-negative compositions.
    Returns: (n, D-1) array of ILR coordinates.
    """
    X = X_raw.astype(np.float64, copy=True)
    X[X == 0] = eps
    s = X.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    X = X / s
    logX = np.log(np.maximum(X, 1e-300))
    clr = logX - logX.mean(axis=1, keepdims=True)
    H = _helmert_submatrix(X.shape[1])
    return clr @ H.T


# ────────────────────────────────────────────────────────────────
# Shannon Entropy
# ────────────────────────────────────────────────────────────────

def shannon_entropy(w):
    """Shannon entropy of probability vector w (base 2, bits)."""
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    w = w[w > 0]
    return -np.sum(w * np.log2(w))


# ────────────────────────────────────────────────────────────────
# DFA (Detrended Fluctuation Analysis)
# ────────────────────────────────────────────────────────────────

def dfa_hurst(x, min_box=10, max_box=None, n_pts=25):
    """
    Estimate Hurst exponent via DFA(1) -- linear detrending.
    Returns (H, box_sizes, fluctuations).
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    y = np.cumsum(x)
    N = len(y)
    if max_box is None:
        max_box = N // 4
    boxes = np.unique(
        np.logspace(np.log10(min_box), np.log10(max_box), n_pts).astype(int)
    )

    fluct = []
    for n in boxes:
        n_seg = N // n
        if n_seg < 2:
            continue
        rms_vals = []
        for i in range(n_seg):
            seg = y[i * n:(i + 1) * n]
            t = np.arange(n)
            coeffs = np.polyfit(t, seg, 1)
            rms_vals.append(np.sqrt(np.mean((seg - np.polyval(coeffs, t))**2)))
        fluct.append((n, np.mean(rms_vals)))

    if len(fluct) < 4:
        return np.nan, np.array([]), np.array([])

    bs = np.array([f[0] for f in fluct])
    fs = np.array([f[1] for f in fluct])
    mask = fs > 0
    coeffs = np.polyfit(np.log(bs[mask]), np.log(fs[mask]), 1)
    return coeffs[0], bs, fs


# ────────────────────────────────────────────────────────────────
# MF-DFA (Multifractal Detrended Fluctuation Analysis)
# ────────────────────────────────────────────────────────────────

def mfdfa(x, q_list=None, min_box=10, max_box=None, n_pts=25, order=1):
    """
    Multifractal DFA.

    Parameters
    ----------
    x : 1D array -- the time series
    q_list : list of floats -- moment orders (default: dense grid from -5 to 5)
    min_box, max_box, n_pts : scale range and sampling
    order : polynomial detrending order (1 = linear)

    Returns
    -------
    dict with keys:
        q       : array of q values
        scales  : array of box sizes used
        Fq      : (len(q), len(scales)) array of q-th order fluctuations
        hq      : array of generalized Hurst exponents h(q)
        tauq    : array of scaling exponents tau(q) = q*h(q) - 1
        alpha   : array of singularity strengths (Hoelder exponents)
        falpha  : array of singularity spectrum f(alpha)
        width   : Delta_alpha = alpha_max - alpha_min  (multifractal width)
    """
    if q_list is None:
        q_list = [-5.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
    q_arr = np.array(q_list, dtype=float)

    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    y = np.cumsum(x)   # profile
    N = len(y)

    if max_box is None:
        max_box = N // 4
    scales = np.unique(
        np.logspace(np.log10(min_box), np.log10(max_box), n_pts).astype(int)
    )

    # For each scale, compute local variance in each segment
    Fq = np.full((len(q_arr), len(scales)), np.nan)

    for si, s in enumerate(scales):
        n_seg = N // s
        if n_seg < 2:
            continue

        # Non-overlapping segments from both ends (standard MF-DFA protocol)
        local_var = []
        for direction in [0, 1]:
            for v in range(n_seg):
                if direction == 0:
                    seg = y[v * s:(v + 1) * s]
                else:
                    seg = y[N - (v + 1) * s:N - v * s]
                t = np.arange(s)
                coeffs = np.polyfit(t, seg, order)
                residual = seg - np.polyval(coeffs, t)
                var_v = np.mean(residual**2)
                if var_v > 0:
                    local_var.append(var_v)

        if len(local_var) < 2:
            continue

        local_var = np.array(local_var)

        for qi, q in enumerate(q_arr):
            if abs(q) < 1e-10:
                # q ~ 0: geometric mean (log average)
                Fq[qi, si] = np.exp(0.5 * np.mean(np.log(local_var)))
            else:
                Fq[qi, si] = np.mean(local_var**(q / 2.0))**(1.0 / q)

    # Fit log-log slopes to get h(q) for each q
    log_s = np.log(scales.astype(float))
    hq = np.full(len(q_arr), np.nan)
    for qi in range(len(q_arr)):
        valid = ~np.isnan(Fq[qi]) & (Fq[qi] > 0)
        if valid.sum() < 4:
            continue
        log_f = np.log(Fq[qi, valid])
        coeffs = np.polyfit(log_s[valid], log_f, 1)
        hq[qi] = coeffs[0]

    # Multifractal scaling exponent tau(q) = q*h(q) - 1
    tauq = q_arr * hq - 1.0

    # Singularity spectrum via Legendre transform:
    # alpha(q) = d(tau)/dq,  f(alpha) = q*alpha - tau(q)
    alpha = np.gradient(tauq, q_arr)
    falpha = q_arr * alpha - tauq

    # Multifractal width
    valid_alpha = alpha[~np.isnan(alpha)]
    width = (valid_alpha.max() - valid_alpha.min()) if len(valid_alpha) > 1 else 0.0

    return {
        'q': q_arr,
        'scales': scales,
        'Fq': Fq,
        'hq': hq,
        'tauq': tauq,
        'alpha': alpha,
        'falpha': falpha,
        'width': width,
    }


# ────────────────────────────────────────────────────────────────
# Column detection
# ────────────────────────────────────────────────────────────────

WEIGHT_RE = re.compile(r"^W_(?P<pol>[a-zA-Z]+)_(?P<s>\d+)$")

def detect_policies(df):
    """Parse W_{policy}_{sink} columns. Returns {policy: [col_names]}."""
    groups = {}
    for c in df.columns:
        m = WEIGHT_RE.match(c)
        if m:
            pol = m.group("pol").lower()
            sink = int(m.group("s"))
            groups.setdefault(pol, []).append((sink, c))
    return {p: [c for _, c in sorted(v)] for p, v in groups.items()}


# ────────────────────────────────────────────────────────────────
# Detrending
# ────────────────────────────────────────────────────────────────

def detrend_mod6(y, mod6):
    """Subtract per-class means for residue classes 0..5."""
    y_det = y.copy()
    for r in range(6):
        mask = mod6 == r
        y_det[mask] -= y[mask].mean()
    return y_det


def running_median(y, window):
    """Running median using pandas (C-optimized)."""
    n = len(y)
    if n < window:
        return np.full(n, np.median(y))
    return pd.Series(y).rolling(window, center=True, min_periods=window // 4).median().values


def detrend_full(y, mod6, secular_window=1001):
    """
    Three-stage detrending:
      1. Separate by mod-6 residue class
      2. Within each class, subtract running median (secular trend)
      3. Reassemble in original order

    secular_window: window width in *within-class* indices.
        With 100K total rows, each class has ~16.7K points,
        so window=1001 spans ~6000 consecutive N values.
    """
    y_out = np.empty_like(y)
    for r in range(6):
        mask = mod6 == r
        y_class = y[mask]
        if len(y_class) == 0:
            continue
        # subtract class mean (stage 1)
        y_class = y_class - y_class.mean()
        # subtract running median of secular drift (stage 2)
        w = min(secular_window, len(y_class) // 3 * 2 + 1)
        if w < 3:
            w = 3
        trend = running_median(y_class, w)
        # fill any remaining NaNs from edges
        nans = np.isnan(trend)
        if nans.any():
            trend[nans] = 0.0
        y_out[mask] = y_class - trend
    return y_out


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Serial dependence analysis for Goldbach-Lemoine "
                    "Descent Graph sink-weight sequences.")
    parser.add_argument("input_csv",
                        help="CSV with columns N and W_{policy}_{sink}")
    parser.add_argument("--outdir", default=".",
                        help="Output directory for figures (default: cwd)")
    parser.add_argument("--n_shuffles", type=int, default=200,
                        help="Number of shuffles for Experiment 10 null test (default: 200)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    csv_path = args.input_csv
    df = pd.read_csv(csv_path).sort_values("N").reset_index(drop=True)
    N_vals = df["N"].values
    mod3 = N_vals % 3
    mod6 = N_vals % 6
    n_rows = len(df)

    print(f"Loaded {n_rows} rows: N = {N_vals[0]:,} .. {N_vals[-1]:,}")

    groups = detect_policies(df)
    policies = sorted(groups.keys())
    print(f"Policies: {policies}\n")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 1: Shannon Entropy by residue class
    # ══════════════════════════════════════════════════════════
    print("=" * 65)
    print("EXPERIMENT 1: Shannon Entropy by Residue Class")
    print("=" * 65)

    fig1, axes1 = plt.subplots(len(policies), 1,
                                figsize=(14, 3 * len(policies)), sharex=True)
    if len(policies) == 1:
        axes1 = [axes1]

    for i, pol in enumerate(policies):
        X = df[groups[pol]].values.astype(float)
        ent = np.array([shannon_entropy(row) for row in X])
        max_H = np.log2(X.shape[1])

        print(f"\n  {pol.upper()}: mean={ent.mean():.4f}  std={ent.std():.4f}"
              f"  max_possible={max_H:.4f}  efficiency={ent.mean()/max_H*100:.1f}%")
        for r in range(3):
            m = mod3 == r
            print(f"    N=={r} (mod 3): mean={ent[m].mean():.4f}  std={ent[m].std():.4f}")
        for r in range(6):
            m = mod6 == r
            print(f"    N=={r} (mod 6): mean={ent[m].mean():.4f}  std={ent[m].std():.4f}")

        ax = axes1[i]
        colors = {0: '#1f77b4', 1: '#2ca02c', 2: '#ff7f0e'}
        for r in range(3):
            m = mod3 == r
            ax.plot(N_vals[m], ent[m], '.', markersize=1.2, alpha=0.5,
                    color=colors[r], label=f'N=={r} (mod 3)')
        ax.axhline(max_H, color='red', ls='--', alpha=0.3, label=f'max={max_H:.2f}')
        ax.set_ylabel(f'{pol.upper()}\nH (bits)')
        ax.legend(loc='upper right', fontsize=7, markerscale=4)
        ax.set_title(f'Shannon Entropy -- {pol.upper()}', fontsize=10)

    axes1[-1].set_xlabel('N')
    fig1.tight_layout()
    fig1.savefig(os.path.join(args.outdir, 'fig_entropy_by_residue.png'), dpi=150)
    print(f"\n  -> fig_entropy_by_residue.png")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 2: ACF & Power Spectrum (raw + mod-6 detrended)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("EXPERIMENT 2: ACF & Power Spectrum")
    print("=" * 65)

    max_lag = min(120, n_rows // 4)

    fig2, axes2 = plt.subplots(len(policies), 4,
                                figsize=(20, 4 * len(policies)))
    if len(policies) == 1:
        axes2 = axes2.reshape(1, -1)

    for i, pol in enumerate(policies):
        X = df[groups[pol]].values.astype(float)
        ilr_mat = ilr_transform(X)
        y_raw = ilr_mat[:, 0]
        y_det = detrend_mod6(y_raw, mod6)

        for j, (y, label) in enumerate([(y_raw, "raw"), (y_det, "detrended")]):
            y_c = y - y.mean()
            acf_full = np.correlate(y_c, y_c, 'full')
            acf = acf_full[len(acf_full) // 2:]
            acf /= acf[0]

            col = j * 2

            ax = axes2[i, col]
            ax.bar(range(max_lag), acf[:max_lag], width=0.8,
                   color='steelblue', alpha=0.7)
            ax.axhline(0, color='k', lw=0.5)
            sig = 1.96 / np.sqrt(len(y))
            ax.axhline(sig, color='red', ls='--', alpha=0.4)
            ax.axhline(-sig, color='red', ls='--', alpha=0.4)
            for lag_mark in [6, 12, 30, 60]:
                if lag_mark < max_lag:
                    ax.axvline(lag_mark, color='gray', ls=':', alpha=0.2)
            ax.set_title(f'{pol.upper()} ACF ({label})', fontsize=9)
            ax.set_xlabel('Lag')
            ax.set_ylabel('ACF')

            ax2 = axes2[i, col + 1]
            freqs = np.fft.rfftfreq(len(y))
            power = np.abs(np.fft.rfft(y_c))**2
            ax2.semilogy(freqs[1:len(freqs) // 2], power[1:len(power) // 2],
                         color='darkorange', lw=0.6, alpha=0.7)
            for period, lbl in [(2, '1/2'), (3, '1/3'), (6, '1/6'), (30, '1/30')]:
                f = 1.0 / period
                if f < freqs[-1]:
                    ax2.axvline(f, color='gray', ls='--', alpha=0.3)
            ax2.set_title(f'{pol.upper()} Spectrum ({label})', fontsize=9)
            ax2.set_xlabel('Frequency (cycles/integer)')

            print(f"\n  {pol.upper()} ({label}) ILR[0]:")
            for lag in [1, 2, 3, 6, 12, 30, 50, 100]:
                if lag < len(acf):
                    print(f"    lag {lag:3d}: {acf[lag]:+.4f}")

    fig2.tight_layout()
    fig2.savefig(os.path.join(args.outdir, 'fig_acf_spectrum.png'), dpi=150)
    print(f"\n  -> fig_acf_spectrum.png")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 3: Hurst Exponent via DFA (raw + detrended)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("EXPERIMENT 3: Hurst Exponent (DFA)")
    print("=" * 65)

    fig3, axes3 = plt.subplots(2, len(policies),
                                figsize=(5 * len(policies), 8))
    if len(policies) == 1:
        axes3 = axes3.reshape(-1, 1)

    for i, pol in enumerate(policies):
        X = df[groups[pol]].values.astype(float)
        ilr_mat = ilr_transform(X)
        n_coords = min(ilr_mat.shape[1], 6)

        for row_idx, label in enumerate(["raw", "mod6-detrended"]):
            print(f"\n  {pol.upper()} ({label}):")
            ax = axes3[row_idx, i]
            hursts = []

            for j in range(n_coords):
                y = ilr_mat[:, j].copy()
                if label == "mod6-detrended":
                    y = detrend_mod6(y, mod6)

                H, bs, fs = dfa_hurst(y, min_box=10, max_box=len(y) // 4)
                hursts.append(H)
                tag = ""
                if H > 0.65:
                    tag = " *** LONG MEMORY ***"
                elif H > 0.55:
                    tag = " * trending"
                elif H < 0.35:
                    tag = " (anti-persistent)"
                print(f"    ILR[{j}]: H = {H:.4f}{tag}")

                if j == 0 and len(bs) > 0:
                    mask = np.array(fs) > 0
                    ax.loglog(np.array(bs)[mask], np.array(fs)[mask], 'o-',
                              markersize=3, label=f'data (H={H:.3f})')
                    lb = np.log(np.array(bs)[mask])
                    lf = np.log(np.array(fs)[mask])
                    fit = np.polyfit(lb, lf, 1)
                    ax.loglog(np.array(bs)[mask],
                              np.exp(np.polyval(fit, lb)),
                              '--', color='red', alpha=0.6)
                    bsr = np.array(bs)[mask]
                    ax.loglog(bsr, fs[0] * (bsr / bsr[0])**0.5,
                              ':', color='gray', alpha=0.4, label='H=0.5')

            ax.set_title(f'{pol.upper()} DFA ({label})', fontsize=9)
            ax.set_xlabel('Box size')
            ax.set_ylabel('F(n)')
            ax.legend(fontsize=7)
            print(f"    Mean H: {np.mean(hursts):.4f}")

    fig3.tight_layout()
    fig3.savefig(os.path.join(args.outdir, 'fig_hurst_dfa.png'), dpi=150)
    print(f"\n  -> fig_hurst_dfa.png")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 4: Cross-policy entropy correlation
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("EXPERIMENT 4: Cross-Policy Entropy Correlation")
    print("=" * 65)

    ent_dict = {}
    for pol in policies:
        X = df[groups[pol]].values.astype(float)
        ent_dict[pol] = np.array([shannon_entropy(row) for row in X])
    print("\n  Correlation matrix:")
    print(pd.DataFrame(ent_dict).corr().to_string())

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 5: Prime vs composite signatures
    # ══════════════════════════════════════════════════════════
    if 'is_prime' in df.columns:
        print("\n" + "=" * 65)
        print("EXPERIMENT 5: Prime vs Composite Signatures")
        print("=" * 65)

        is_p = df['is_prime'].values.astype(bool)
        print(f"\n  Primes in range: {is_p.sum()}")

        for pol in policies:
            X = df[groups[pol]].values.astype(float)
            ent = np.array([shannon_entropy(row) for row in X])
            ilr_mat = ilr_transform(X)
            cent = ilr_mat.mean(axis=0)

            print(f"\n  {pol.upper()}:")
            print(f"    Prime entropy:     mean={ent[is_p].mean():.4f}")
            print(f"    Composite entropy: mean={ent[~is_p].mean():.4f}")
            print(f"    Difference:        {ent[is_p].mean() - ent[~is_p].mean():.4f}")

            d_p = np.sqrt(((ilr_mat[is_p] - cent)**2).sum(axis=1))
            d_c = np.sqrt(((ilr_mat[~is_p] - cent)**2).sum(axis=1))
            print(f"    ILR centroid dist (prime):     {d_p.mean():.4f}")
            print(f"    ILR centroid dist (composite): {d_c.mean():.4f}")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 6: Twin prime proximity
    # ══════════════════════════════════════════════════════════
    if 'is_prime' in df.columns:
        print("\n" + "=" * 65)
        print("EXPERIMENT 6: Twin Prime ILR Proximity")
        print("=" * 65)

        is_p = df['is_prime'].values.astype(bool)
        N_set = set(N_vals.astype(int))
        n_to_idx = {int(N_vals[i]): i for i in range(len(N_vals))}

        for pol in policies:
            X = df[groups[pol]].values.astype(float)
            ilr_mat = ilr_transform(X)

            twin_dists = []
            np.random.seed(42)

            for idx in range(len(N_vals)):
                n_val = int(N_vals[idx])
                if is_p[idx] and (n_val + 2) in N_set:
                    idx2 = n_to_idx[n_val + 2]
                    if is_p[idx2]:
                        d = np.linalg.norm(ilr_mat[idx] - ilr_mat[idx2])
                        twin_dists.append(d)

            random_dists = []
            for _ in range(min(5000, len(df) - 2)):
                idx = np.random.randint(0, len(df) - 2)
                d = np.linalg.norm(ilr_mat[idx] - ilr_mat[idx + 2])
                random_dists.append(d)

            twin_dists = np.array(twin_dists)
            random_dists = np.array(random_dists)
            print(f"\n  {pol.upper()}: {len(twin_dists)} twin pairs")
            if len(twin_dists) > 0:
                ratio = twin_dists.mean() / random_dists.mean()
                print(f"    Twin ILR dist:   mean={twin_dists.mean():.4f}  "
                      f"median={np.median(twin_dists):.4f}")
                print(f"    Random lag-2:    mean={random_dists.mean():.4f}  "
                      f"median={np.median(random_dists):.4f}")
                print(f"    Ratio:           {ratio:.4f}"
                      f"  ({'CLOSER' if ratio < 0.9 else 'FARTHER' if ratio > 1.1 else 'SIMILAR'})")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 7: Secular Drift Diagnostic
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("EXPERIMENT 7: Secular Drift Diagnostic")
    print("=" * 65)

    sw_width = min(2001, n_rows // 10)

    fig7, axes7 = plt.subplots(len(policies), 2,
                                figsize=(16, 3.5 * len(policies)))
    if len(policies) == 1:
        axes7 = axes7.reshape(1, -1)

    for i, pol in enumerate(policies):
        X = df[groups[pol]].values.astype(float)
        ilr_mat = ilr_transform(X)
        n_coords = min(ilr_mat.shape[1], 6)

        print(f"\n  {pol.upper()}: sliding window width = {sw_width}")
        drift_range = np.zeros(n_coords)

        ax_drift = axes7[i, 0]
        ax_resid = axes7[i, 1]

        for j in range(n_coords):
            y = ilr_mat[:, j]
            sw_mean = pd.Series(y).rolling(sw_width, center=True,
                                           min_periods=sw_width // 4).median()
            sw_arr = sw_mean.values
            valid = ~np.isnan(sw_arr)
            if valid.any():
                drift_range[j] = np.nanmax(sw_arr) - np.nanmin(sw_arr)
            if j < 3:
                ax_drift.plot(N_vals[valid], sw_arr[valid], lw=1.0,
                              alpha=0.8, label=f'ILR[{j}]')

        print(f"    Drift range per ILR coord: "
              + ", ".join(f"[{j}]={drift_range[j]:.4f}" for j in range(n_coords)))
        total_std = np.array([ilr_mat[:, j].std() for j in range(n_coords)])
        drift_frac = drift_range / np.maximum(total_std, 1e-12)
        print(f"    Drift/std ratio per coord: "
              + ", ".join(f"[{j}]={drift_frac[j]:.3f}" for j in range(n_coords)))

        ax_drift.set_title(f'{pol.upper()} -- Secular Drift (sliding median)', fontsize=10)
        ax_drift.set_xlabel('N')
        ax_drift.set_ylabel('ILR coordinate')
        ax_drift.legend(fontsize=7)

        y0 = ilr_mat[:, 0]
        y0_mod6 = detrend_mod6(y0, mod6)
        y0_full = detrend_full(y0, mod6, secular_window=1001)
        ax_resid.plot(N_vals[::20], y0_mod6[::20], '.', markersize=1, alpha=0.3,
                      color='blue', label='mod6-detrended')
        ax_resid.plot(N_vals[::20], y0_full[::20], '.', markersize=1, alpha=0.3,
                      color='red', label='fully-detrended')
        ax_resid.set_title(f'{pol.upper()} -- ILR[0]: mod6 vs full detrend', fontsize=10)
        ax_resid.set_xlabel('N')
        ax_resid.legend(fontsize=7, markerscale=6)

    fig7.tight_layout()
    fig7.savefig(os.path.join(args.outdir, 'fig_secular_drift.png'), dpi=150)
    print(f"\n  -> fig_secular_drift.png")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 8: Three-Stage DFA Decomposition
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("EXPERIMENT 8: Three-Stage DFA Decomposition")
    print("=" * 65)
    print("  [raw  /  mod6-detrended  /  fully-detrended (mod6 + secular)]")

    fig8, axes8 = plt.subplots(len(policies), 2,
                                figsize=(14, 4 * len(policies)))
    if len(policies) == 1:
        axes8 = axes8.reshape(1, -1)

    for i, pol in enumerate(policies):
        X = df[groups[pol]].values.astype(float)
        ilr_mat = ilr_transform(X)
        n_coords = min(ilr_mat.shape[1], 6)

        print(f"\n  {pol.upper()}:")
        hdr = f"    {'Coord':>6s}  {'H_raw':>7s}  {'H_mod6':>7s}  {'H_full':>7s}  {'delta':>7s}"
        print(hdr)
        print(f"    {'---':>6s}  {'---':>7s}  {'---':>7s}  {'---':>7s}  {'---':>7s}")

        h_raw_all, h_mod6_all, h_full_all = [], [], []
        ax_bar = axes8[i, 0]
        ax_dfa = axes8[i, 1]

        for j in range(n_coords):
            y = ilr_mat[:, j]
            y_m6 = detrend_mod6(y, mod6)
            y_full = detrend_full(y, mod6, secular_window=1001)

            H_raw, _, _ = dfa_hurst(y)
            H_m6, _, _ = dfa_hurst(y_m6)
            H_full, bs_f, fs_f = dfa_hurst(y_full)

            h_raw_all.append(H_raw)
            h_mod6_all.append(H_m6)
            h_full_all.append(H_full)

            delta = H_m6 - H_full
            print(f"    ILR[{j}]  {H_raw:7.4f}  {H_m6:7.4f}  {H_full:7.4f}  {delta:+7.4f}")

            if j == 0 and len(bs_f) > 0:
                for stage_y, stage_lbl, stage_clr in [
                    (y, 'raw', '#1f77b4'),
                    (y_m6, 'mod6', '#ff7f0e'),
                    (y_full, 'full', '#2ca02c')
                ]:
                    H_s, bs_s, fs_s = dfa_hurst(stage_y)
                    msk = np.array(fs_s) > 0
                    if msk.any():
                        ax_dfa.loglog(np.array(bs_s)[msk], np.array(fs_s)[msk],
                                      'o-', markersize=3, color=stage_clr,
                                      alpha=0.7, label=f'{stage_lbl} H={H_s:.3f}')
                bs_ref = np.array(bs_f)
                msk = np.array(fs_f) > 0
                if msk.any():
                    ax_dfa.loglog(bs_ref[msk],
                                  fs_f[msk][0] * (bs_ref[msk] / bs_ref[msk][0])**0.5,
                                  ':', color='gray', alpha=0.5, label='H=0.5 ref')

        mean_raw = np.mean(h_raw_all)
        mean_m6 = np.mean(h_mod6_all)
        mean_full = np.mean(h_full_all)
        print(f"    {'MEAN':>6s}  {mean_raw:7.4f}  {mean_m6:7.4f}  {mean_full:7.4f}  "
              f"{mean_m6 - mean_full:+7.4f}")

        x_pos = np.arange(n_coords)
        w = 0.25
        ax_bar.bar(x_pos - w, h_raw_all, w, label='raw', color='#1f77b4', alpha=0.8)
        ax_bar.bar(x_pos, h_mod6_all, w, label='mod6', color='#ff7f0e', alpha=0.8)
        ax_bar.bar(x_pos + w, h_full_all, w, label='full', color='#2ca02c', alpha=0.8)
        ax_bar.axhline(0.5, color='gray', ls='--', alpha=0.5, label='H=0.5')
        ax_bar.set_xticks(x_pos)
        ax_bar.set_xticklabels([f'ILR[{j}]' for j in range(n_coords)], fontsize=7)
        ax_bar.set_ylabel('Hurst exponent')
        ax_bar.set_title(f'{pol.upper()} -- H by detrending stage', fontsize=10)
        ax_bar.legend(fontsize=7)
        ax_bar.set_ylim(0, max(max(h_mod6_all) + 0.1, 1.1))

        ax_dfa.set_title(f'{pol.upper()} -- DFA ILR[0] three stages', fontsize=10)
        ax_dfa.set_xlabel('Box size')
        ax_dfa.set_ylabel('F(n)')
        ax_dfa.legend(fontsize=7)

    fig8.tight_layout()
    fig8.savefig(os.path.join(args.outdir, 'fig_three_stage_dfa.png'), dpi=150)
    print(f"\n  -> fig_three_stage_dfa.png")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 9: ACF Decay Classification (fully-detrended)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("EXPERIMENT 9: ACF Decay Classification (fully-detrended)")
    print("=" * 65)
    print("  Fits ACF(k) to power-law k^beta and exponential exp(-k/tau)")

    fit_max_lag = min(500, n_rows // 10)

    fig9, axes9 = plt.subplots(len(policies), 2,
                                figsize=(14, 3.5 * len(policies)))
    if len(policies) == 1:
        axes9 = axes9.reshape(1, -1)

    for i, pol in enumerate(policies):
        X = df[groups[pol]].values.astype(float)
        ilr_mat = ilr_transform(X)

        print(f"\n  {pol.upper()}:")
        print(f"    {'Coord':>6s}  {'H_full':>7s}  {'ACF(1)':>7s}  {'ACF(10)':>8s}"
              f"  {'ACF(50)':>8s}  {'ACF(100)':>9s}  {'beta':>7s}  {'tau':>7s}"
              f"  {'R2_pow':>7s}  {'R2_exp':>7s}  model")

        ax_acf = axes9[i, 0]
        ax_fit = axes9[i, 1]

        for j in range(min(ilr_mat.shape[1], 6)):
            y_full = detrend_full(ilr_mat[:, j], mod6, secular_window=1001)
            y_c = y_full - y_full.mean()
            acf_full = np.correlate(y_c, y_c, 'full')
            acf = acf_full[len(acf_full) // 2:]
            if acf[0] > 0:
                acf = acf / acf[0]
            else:
                continue

            H_full, _, _ = dfa_hurst(y_full)

            acf_vals = {}
            for lag in [1, 10, 50, 100]:
                acf_vals[lag] = acf[lag] if lag < len(acf) else np.nan

            lags_fit = np.arange(2, min(fit_max_lag, len(acf)))
            acf_fit = acf[lags_fit]
            pos_mask = acf_fit > 0.01

            beta_pow = np.nan
            r2_pow = 0.0
            tau_exp = np.nan
            r2_exp = 0.0
            coeffs_p = None

            if pos_mask.sum() > 10:
                log_lags = np.log(lags_fit[pos_mask])
                log_acf = np.log(acf_fit[pos_mask])

                try:
                    coeffs_p = np.polyfit(log_lags, log_acf, 1)
                    beta_pow = coeffs_p[0]
                    pred_p = np.polyval(coeffs_p, log_lags)
                    ss_res = np.sum((log_acf - pred_p)**2)
                    ss_tot = np.sum((log_acf - log_acf.mean())**2)
                    r2_pow = 1.0 - ss_res / max(ss_tot, 1e-15)
                except Exception:
                    pass

                try:
                    coeffs_e = np.polyfit(lags_fit[pos_mask], log_acf, 1)
                    tau_exp = -1.0 / coeffs_e[0] if coeffs_e[0] < 0 else np.nan
                    pred_e = np.polyval(coeffs_e, lags_fit[pos_mask])
                    ss_res_e = np.sum((log_acf - pred_e)**2)
                    r2_exp = 1.0 - ss_res_e / max(ss_tot, 1e-15)
                except Exception:
                    pass

            winner = "power" if r2_pow > r2_exp + 0.02 else \
                     "expon" if r2_exp > r2_pow + 0.02 else "ambig"
            print(f"    ILR[{j}]  {H_full:7.4f}  {acf_vals[1]:+7.4f}  "
                  f"{acf_vals[10]:+8.4f}  {acf_vals[50]:+8.4f}  {acf_vals[100]:+9.4f}  "
                  f"{beta_pow:7.3f}  {tau_exp:7.1f}  "
                  f"{r2_pow:7.4f}  {r2_exp:7.4f}  {winner}")

            if j < 3:
                ax_acf.plot(range(min(200, len(acf))), acf[:min(200, len(acf))],
                            lw=0.8, alpha=0.7, label=f'ILR[{j}]')
                if pos_mask.sum() > 10 and coeffs_p is not None and not np.isnan(beta_pow):
                    ll = lags_fit[pos_mask]
                    ax_fit.loglog(ll, acf_fit[pos_mask], '.', markersize=2,
                                  alpha=0.5, label=f'ILR[{j}] data')
                    ax_fit.loglog(ll, np.exp(np.polyval(coeffs_p, np.log(ll))),
                                  '--', lw=1, alpha=0.6,
                                  label=f'beta={beta_pow:.2f}')

        ax_acf.axhline(0, color='k', lw=0.5)
        sig = 1.96 / np.sqrt(n_rows)
        ax_acf.axhline(sig, color='red', ls='--', alpha=0.3)
        ax_acf.axhline(-sig, color='red', ls='--', alpha=0.3)
        ax_acf.set_title(f'{pol.upper()} -- ACF fully-detrended', fontsize=10)
        ax_acf.set_xlabel('Lag')
        ax_acf.set_ylabel('ACF')
        ax_acf.legend(fontsize=6)

        ax_fit.set_title(f'{pol.upper()} -- ACF log-log + power-law fit', fontsize=10)
        ax_fit.set_xlabel('Lag')
        ax_fit.set_ylabel('ACF(k)')
        ax_fit.legend(fontsize=6)

    fig9.tight_layout()
    fig9.savefig(os.path.join(args.outdir, 'fig_acf_decay.png'), dpi=150)
    print(f"\n  -> fig_acf_decay.png")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 10: Shuffle Null Test for Hurst Baseline
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("EXPERIMENT 10: Shuffle Null Test for Hurst Baseline")
    print("=" * 65)
    print("  Within-class shuffle: permute compositions within each mod-6 class,")
    print("  reassemble, compute DFA. Repeat n_shuffles times for null distribution.")

    n_shuffles = args.n_shuffles
    rng = np.random.default_rng(seed=2025)

    fig10, axes10 = plt.subplots(1, len(policies),
                                  figsize=(4 * len(policies), 4))
    if len(policies) == 1:
        axes10 = [axes10]

    for i, pol in enumerate(policies):
        X = df[groups[pol]].values.astype(float)
        ilr_mat = ilr_transform(X)
        n_coords = min(ilr_mat.shape[1], 6)

        print(f"\n  {pol.upper()}:")

        h_obs = []
        for j in range(n_coords):
            y_full = detrend_full(ilr_mat[:, j], mod6, secular_window=1001)
            H, _, _ = dfa_hurst(y_full)
            h_obs.append(H)
        h_obs_mean = np.mean(h_obs)

        h_null_all = []
        for shuf in range(n_shuffles):
            if (shuf + 1) % 20 == 0 or shuf == 0:
                print(f"      shuffle {shuf + 1}/{n_shuffles} ...", flush=True)
            ilr_shuffled = np.empty_like(ilr_mat)
            for r in range(6):
                mask = mod6 == r
                idx_class = np.where(mask)[0]
                perm = rng.permutation(len(idx_class))
                ilr_shuffled[idx_class] = ilr_mat[idx_class[perm]]

            h_shuf = []
            for j in range(n_coords):
                y_full_s = detrend_full(ilr_shuffled[:, j], mod6, secular_window=1001)
                H_s, _, _ = dfa_hurst(y_full_s)
                h_shuf.append(H_s)
            h_null_all.append(np.mean(h_shuf))

        h_null = np.array(h_null_all)
        p_value = np.mean(h_null >= h_obs_mean)

        print(f"    Observed mean H (full detrend): {h_obs_mean:.4f}")
        print(f"    Shuffle null: mean={h_null.mean():.4f}  "
              f"std={h_null.std():.4f}  "
              f"[{np.percentile(h_null, 2.5):.4f}, {np.percentile(h_null, 97.5):.4f}] 95% CI")
        print(f"    Excess H = {h_obs_mean - h_null.mean():.4f}")
        z = (h_obs_mean - h_null.mean()) / max(h_null.std(), 1e-10)
        print(f"    Z-score = {z:.2f},  p(null >= obs) = {p_value:.4f}")
        for j in range(n_coords):
            print(f"    ILR[{j}]: H_obs={h_obs[j]:.4f}")

        ax = axes10[i]
        ax.hist(h_null, bins=15, alpha=0.6, color='steelblue',
                edgecolor='navy', label=f'shuffle null (n={n_shuffles})')
        ax.axvline(h_obs_mean, color='red', lw=2, label=f'observed H={h_obs_mean:.3f}')
        ax.axvline(h_null.mean(), color='blue', ls='--', lw=1,
                   label=f'null mean={h_null.mean():.3f}')
        ax.axvline(0.5, color='gray', ls=':', alpha=0.5, label='H=0.5')
        ax.set_title(f'{pol.upper()} -- Shuffle Null', fontsize=10)
        ax.set_xlabel('Mean H')
        ax.legend(fontsize=6)

    fig10.tight_layout()
    fig10.savefig(os.path.join(args.outdir, 'fig_shuffle_null.png'), dpi=150)
    print(f"\n  -> fig_shuffle_null.png")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 11: Multifractal DFA (MF-DFA)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("EXPERIMENT 11: Multifractal DFA (MF-DFA)")
    print("=" * 65)
    print("  Generalized Hurst exponents h(q) and singularity spectrum f(alpha)")
    print("  on fully-detrended ILR[0] for each policy.")
    print("  Monofractal: Delta_alpha ~ 0, h(q) ~ const.")
    print("  Multifractal: Delta_alpha >> 0, h(q) decreasing in q.")

    q_list = [-5.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0]

    fig11, axes11 = plt.subplots(len(policies), 3,
                                  figsize=(18, 4.5 * len(policies)))
    if len(policies) == 1:
        axes11 = axes11.reshape(1, -1)

    # Summary table header
    print(f"\n  {'Policy':>8s}  {'h(2)':>6s}  {'h(-2)':>6s}  {'dh':>6s}  "
          f"{'d_alpha':>7s}  {'a_peak':>7s}  {'f_peak':>7s}  interp")
    print(f"  {'---':>8s}  {'---':>6s}  {'---':>6s}  {'---':>6s}  "
          f"{'---':>7s}  {'---':>7s}  {'---':>7s}  {'---':>12s}")

    for i, pol in enumerate(policies):
        X = df[groups[pol]].values.astype(float)
        ilr_mat = ilr_transform(X)

        # Run MF-DFA on fully-detrended ILR[0]
        y_full = detrend_full(ilr_mat[:, 0], mod6, secular_window=1001)
        result = mfdfa(y_full, q_list=q_list, min_box=10, n_pts=30)

        q = result['q']
        hq = result['hq']
        tauq = result['tauq']
        alpha_arr = result['alpha']
        falpha = result['falpha']
        width = result['width']

        # Extract key values
        idx_q2 = q_list.index(2.0)
        idx_qm2 = q_list.index(-2.0)
        h2 = hq[idx_q2]
        hm2 = hq[idx_qm2]
        delta_h = hm2 - h2 if not (np.isnan(hm2) or np.isnan(h2)) else np.nan

        # Peak of f(alpha) spectrum
        valid_fa = ~np.isnan(falpha)
        if valid_fa.any():
            peak_idx = np.nanargmax(falpha)
            alpha_peak = alpha_arr[peak_idx]
            f_peak = falpha[peak_idx]
        else:
            alpha_peak = np.nan
            f_peak = np.nan

        # Interpretation
        if width > 0.3:
            interp = "MULTIFRACTAL"
        elif width > 0.15:
            interp = "weakly multi"
        else:
            interp = "monofractal"

        print(f"  {pol.upper():>8s}  {h2:6.3f}  {hm2:6.3f}  {delta_h:6.3f}  "
              f"{width:7.3f}  {alpha_peak:7.3f}  {f_peak:7.3f}  {interp}")

        # Also run MF-DFA on additional ILR coords
        for j in range(1, min(ilr_mat.shape[1], 4)):
            y_j = detrend_full(ilr_mat[:, j], mod6, secular_window=1001)
            res_j = mfdfa(y_j, q_list=q_list, min_box=10, n_pts=30)
            hq_j = res_j['hq']
            h2_j = hq_j[idx_q2]
            hm2_j = hq_j[idx_qm2]
            print(f"    ILR[{j}]: h(2)={h2_j:.3f}  h(-2)={hm2_j:.3f}  "
                  f"d_alpha={res_j['width']:.3f}")

        # -- Plot 1: h(q) spectrum --
        ax_hq = axes11[i, 0]
        valid_hq = ~np.isnan(hq)
        ax_hq.plot(q[valid_hq], hq[valid_hq], 'o-', color='#2ca02c',
                   markersize=5, lw=1.5, label='ILR[0]')
        if not np.isnan(h2):
            ax_hq.axhline(h2, color='gray', ls='--', alpha=0.4,
                          label=f'h(2)={h2:.3f}')
        ax_hq.axhline(0.5, color='red', ls=':', alpha=0.3, label='H=0.5')
        ax_hq.set_xlabel('q')
        ax_hq.set_ylabel('h(q)')
        ax_hq.set_title(f'{pol.upper()} -- Generalized Hurst h(q)', fontsize=10)
        ax_hq.legend(fontsize=7)
        ax_hq.grid(True, alpha=0.2)

        # -- Plot 2: tau(q) spectrum --
        ax_tau = axes11[i, 1]
        valid_tau = ~np.isnan(tauq)
        ax_tau.plot(q[valid_tau], tauq[valid_tau], 's-', color='#1f77b4',
                    markersize=4, lw=1.5)
        if not np.isnan(h2):
            tau_mono = q * h2 - 1.0
            ax_tau.plot(q, tau_mono, '--', color='gray', alpha=0.5,
                        label=f'mono (H={h2:.3f})')
        ax_tau.set_xlabel('q')
        ax_tau.set_ylabel('tau(q)')
        ax_tau.set_title(f'{pol.upper()} -- Scaling exponent tau(q)', fontsize=10)
        ax_tau.legend(fontsize=7)
        ax_tau.grid(True, alpha=0.2)

        # -- Plot 3: f(alpha) singularity spectrum --
        ax_fa = axes11[i, 2]
        valid_spec = ~np.isnan(alpha_arr) & ~np.isnan(falpha)
        if valid_spec.any():
            ax_fa.plot(alpha_arr[valid_spec], falpha[valid_spec], 'D-',
                       color='#d62728', markersize=5, lw=1.5)
            if not np.isnan(alpha_peak):
                ax_fa.plot(alpha_peak, f_peak, '*', color='gold',
                           markersize=15, zorder=5)
            ax_fa.annotate(f'd_alpha = {width:.3f}',
                          xy=(0.05, 0.05), xycoords='axes fraction',
                          fontsize=9, color='#d62728',
                          bbox=dict(boxstyle='round', fc='white', alpha=0.8))
        ax_fa.set_xlabel('alpha (Hoelder exponent)')
        ax_fa.set_ylabel('f(alpha)')
        ax_fa.set_title(f'{pol.upper()} -- Singularity spectrum f(alpha)', fontsize=10)
        ax_fa.grid(True, alpha=0.2)

    fig11.tight_layout()
    fig11.savefig(os.path.join(args.outdir, 'fig_mfdfa.png'), dpi=150)
    print(f"\n  -> fig_mfdfa.png")

    # -- MF-DFA comparison: observed vs shuffle null --
    print("\n  MF-DFA Shuffle Null Comparison (5 shuffles, quick check):")
    print(f"  {'Policy':>8s}  {'da_obs':>7s}  {'da_null':>7s}  {'dh_obs':>7s}  {'dh_null':>7s}")
    print(f"  {'---':>8s}  {'---':>7s}  {'---':>7s}  {'---':>7s}  {'---':>7s}")

    rng_mf = np.random.default_rng(seed=2026)
    n_mf_shuf = 5

    for i, pol in enumerate(policies):
        X = df[groups[pol]].values.astype(float)
        ilr_mat = ilr_transform(X)
        y_full = detrend_full(ilr_mat[:, 0], mod6, secular_window=1001)

        res_obs = mfdfa(y_full, q_list=q_list, min_box=10, n_pts=30)
        width_obs = res_obs['width']
        h2_obs = res_obs['hq'][idx_q2]
        hm2_obs = res_obs['hq'][idx_qm2]
        dh_obs = hm2_obs - h2_obs

        widths_null = []
        dh_null_list = []
        for _ in range(n_mf_shuf):
            ilr_shuf = np.empty_like(ilr_mat)
            for r in range(6):
                mask = mod6 == r
                idx_class = np.where(mask)[0]
                perm = rng_mf.permutation(len(idx_class))
                ilr_shuf[idx_class] = ilr_mat[idx_class[perm]]
            y_shuf = detrend_full(ilr_shuf[:, 0], mod6, secular_window=1001)
            res_shuf = mfdfa(y_shuf, q_list=q_list, min_box=10, n_pts=30)
            widths_null.append(res_shuf['width'])
            h2_s = res_shuf['hq'][idx_q2]
            hm2_s = res_shuf['hq'][idx_qm2]
            dh_null_list.append(hm2_s - h2_s)

        mean_width_null = np.mean(widths_null)
        mean_dh_null = np.mean(dh_null_list)
        print(f"  {pol.upper():>8s}  {width_obs:7.3f}  {mean_width_null:7.3f}  "
              f"{dh_obs:7.3f}  {mean_dh_null:7.3f}")

    print("\n" + "=" * 65)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
