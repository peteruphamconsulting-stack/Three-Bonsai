#!/usr/bin/env python3
"""
scale_invariance_dimension.py

Assess whether intrinsic dimension estimates are stable across orders of
magnitude.  Reads the same CSV produced by descent_graph_sink_weights.py,
splits the data into log10 magnitude bands, and computes kNN-based intrinsic
dimension estimates within each band, per policy.

Produces:
  1. Console table of d_hat by band x policy
  2. fig_scale_invariance.png  — d_hat vs log10(N) per policy
  3. fig_scale_invariance_heatmap.png — band x policy heatmap
  4. fig_scale_invariance_multi_k.png — d_hat vs k per band per policy
  5. scale_invariance_summary.csv

Usage:
  python3 scale_invariance_dimension.py embeddings_logstrat_10sink_5pol.csv \
      --outdir ./scale_figs \
      --policies down,up,quarter,center,random \
      --k_values 10,15,20,25,30

Requires: numpy, pandas, matplotlib, scikit-learn
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except ImportError:
    HAVE_PLT = False

try:
    from sklearn.neighbors import NearestNeighbors
    HAVE_SK = True
except ImportError:
    HAVE_SK = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_K_VALUES = [10, 15, 20, 25, 30]
PSEUDOCOUNT_EPS = 0.5
KNN_TRIM_FRACTION = 0.02
WEIGHT_COL_REGEX = r"^[Ww]_(?:(?P<policy>[A-Za-z0-9]+)_)?(?P<sink>\d+)$"
MIN_BAND_SIZE = 80  # minimum points per band to attempt dimension estimation


# ---------------------------------------------------------------------------
# ILR transform (standalone)
# ---------------------------------------------------------------------------

def helmert_submatrix(D: int) -> np.ndarray:
    """Orthonormal (D-1) x D Helmert sub-matrix."""
    H = np.zeros((D - 1, D), dtype=np.float64)
    for k in range(1, D):
        denom = math.sqrt(k * (k + 1))
        H[k - 1, :k] = 1.0 / denom
        H[k - 1, k] = -k / denom
    return H


def closure_rows(X: np.ndarray) -> np.ndarray:
    s = X.sum(axis=1, keepdims=True)
    s = np.where(s == 0, 1.0, s)
    return X / s


def apply_pseudocount(X: np.ndarray, eps: float = 0.5) -> np.ndarray:
    X2 = X.astype(np.float64, copy=True)
    X2[X2 == 0] = eps
    return X2


def ilr_transform(X: np.ndarray, eps: float = 0.5) -> np.ndarray:
    """Raw counts -> ILR coordinates. Adds pseudocount, closes, transforms."""
    X2 = apply_pseudocount(X, eps)
    X_closed = closure_rows(X2)
    logX = np.log(np.maximum(X_closed, 1e-300))
    clr = logX - logX.mean(axis=1, keepdims=True)
    H = helmert_submatrix(X.shape[1])
    return clr @ H.T


# ---------------------------------------------------------------------------
# kNN-MLE dimension (Levina-Bickel)
# ---------------------------------------------------------------------------

def knn_dimension(X: np.ndarray, k_list: List[int],
                  trim_frac: float = 0.02) -> Dict:
    """
    Returns dict with keys:
      'summary': list of dicts {k, d_mean, d_median, d_trimmed, n_valid, median_r}
      'local': dict k -> array of per-point estimates
    """
    if not HAVE_SK:
        raise RuntimeError("scikit-learn required for kNN dimension estimation")

    n = X.shape[0]
    max_k = max(k_list)
    if max_k >= n:
        max_k = n - 1
        k_list = [k for k in k_list if k < n]

    nn = NearestNeighbors(n_neighbors=max_k + 1, algorithm="auto",
                          metric="euclidean")
    nn.fit(X)
    dists, _ = nn.kneighbors(X, return_distance=True)
    dists = dists[:, 1:]  # drop self

    results = []
    local_by_k = {}

    for k in k_list:
        Tk = dists[:, k - 1]
        Tj = dists[:, :k - 1]
        valid = (Tk > 0) & (Tj.min(axis=1) > 0)

        di = np.full(n, np.nan)
        if valid.any():
            ratio = Tk[valid, None] / Tj[valid]
            inv = np.mean(np.log(ratio), axis=1)
            good = inv > 0
            tmp = np.full(inv.shape[0], np.nan)
            tmp[good] = 1.0 / inv[good]
            di[valid] = tmp

        local_by_k[k] = di
        vals = di[np.isfinite(di)]
        n_valid = len(vals)

        if n_valid == 0:
            results.append({"k": k, "d_mean": np.nan, "d_median": np.nan,
                            "d_trimmed": np.nan, "n_valid": 0, "median_r": np.nan})
            continue

        d_mean = float(np.mean(vals))
        d_median = float(np.median(vals))

        # trimmed mean
        if trim_frac > 0 and trim_frac < 0.49 and n_valid > 10:
            lo = np.quantile(vals, trim_frac)
            hi = np.quantile(vals, 1 - trim_frac)
            trimmed = vals[(vals >= lo) & (vals <= hi)]
            d_trimmed = float(np.mean(trimmed)) if len(trimmed) > 0 else d_mean
        else:
            d_trimmed = d_mean

        Tk_pos = Tk[Tk > 0]
        median_r = float(np.median(Tk_pos)) if len(Tk_pos) > 0 else np.nan

        results.append({
            "k": k, "d_mean": d_mean, "d_median": d_median,
            "d_trimmed": d_trimmed, "n_valid": n_valid, "median_r": median_r
        })

    return {"summary": results, "local": local_by_k}


# ---------------------------------------------------------------------------
# Magnitude band utilities
# ---------------------------------------------------------------------------

def build_magnitude_bands(N_vals: np.ndarray) -> List[Tuple[int, int, str]]:
    """
    Build log10 decade bands spanning [min(N), max(N)].
    Returns list of (lo, hi, label) tuples.
    """
    lo_all = int(N_vals.min())
    hi_all = int(N_vals.max())

    # Find powers of 10 in range
    boundaries = [lo_all]
    p = 10 ** len(str(lo_all))
    while p <= hi_all:
        if p > lo_all:
            boundaries.append(p)
        p *= 10
    boundaries.append(hi_all + 1)

    bands = []
    for i in range(len(boundaries) - 1):
        blo = boundaries[i]
        bhi = boundaries[i + 1] - 1
        if blo <= bhi:
            # Label
            if blo >= 1e9:
                label = f"{blo/1e9:.1f}B-{bhi/1e9:.1f}B"
            elif blo >= 1e6:
                label = f"{blo/1e6:.0f}M-{bhi/1e6:.0f}M"
            elif blo >= 1e3:
                label = f"{blo/1e3:.0f}k-{bhi/1e3:.0f}k"
            else:
                label = f"{blo}-{bhi}"
            bands.append((blo, bhi, label))
    return bands


# ---------------------------------------------------------------------------
# Detect weight columns (same logic as main script)
# ---------------------------------------------------------------------------

def detect_weight_columns(df: pd.DataFrame,
                          policies_filter: Optional[List[str]] = None
                          ) -> Dict[str, List[str]]:
    """
    Returns dict: policy -> list of weight column names, sorted by sink.
    """
    pat = re.compile(WEIGHT_COL_REGEX)
    groups: Dict[str, List[Tuple[int, str]]] = {}
    for c in df.columns:
        m = pat.match(c)
        if not m:
            continue
        sink = int(m.group("sink"))
        policy = m.groupdict().get("policy") or "ALL"
        groups.setdefault(policy, []).append((sink, c))

    out = {}
    for policy, pairs in groups.items():
        if policies_filter and policy not in policies_filter:
            continue
        pairs.sort(key=lambda t: t[0])
        out[policy] = [c for _, c in pairs]
    return out


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_scale_analysis(csv_path: str,
                       outdir: str,
                       policies: Optional[List[str]] = None,
                       k_values: Optional[List[int]] = None,
                       reference_k: int = 20):
    """
    Main entry point. Loads CSV, splits by magnitude band, computes
    kNN dimension per band x policy.
    """
    k_list = k_values or DEFAULT_K_VALUES
    outpath = Path(outdir)
    outpath.mkdir(parents=True, exist_ok=True)

    # Load
    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Rows={len(df):,}  Cols={len(df.columns):,}")

    if "N" not in df.columns:
        raise ValueError("CSV must have an 'N' column for magnitude banding")

    N_vals = df["N"].values.astype(np.int64)

    # Detect policies
    groups = detect_weight_columns(df, policies)
    print(f"Policies: {sorted(groups.keys())}")

    # Build bands
    bands = build_magnitude_bands(N_vals)
    print(f"\nMagnitude bands ({len(bands)}):")
    for blo, bhi, label in bands:
        n_in = int(np.sum((N_vals >= blo) & (N_vals <= bhi)))
        print(f"  {label:>20s}  n={n_in:,}")

    # Also include "ALL" as a band for reference
    bands_with_all = bands + [(int(N_vals.min()), int(N_vals.max()), "ALL")]

    # --- Compute dimension per band x policy ---
    all_rows = []

    for policy, cols in sorted(groups.items()):
        print(f"\n{'='*60}")
        print(f"Policy: {policy.upper()}")
        print(f"{'='*60}")

        # Extract weight matrix for this policy
        W_full = df[cols].apply(pd.to_numeric, errors="coerce").values.astype(np.float64)

        # Drop zero-variance columns
        var = np.var(W_full, axis=0)
        keep_mask = var > 0
        if not np.all(keep_mask):
            dropped = [cols[i] for i in range(len(cols)) if not keep_mask[i]]
            print(f"  Dropping {len(dropped)} zero-variance columns: {dropped}")
        W_full = W_full[:, keep_mask]
        D = W_full.shape[1]

        if D < 2:
            print(f"  Skipping {policy}: fewer than 2 non-degenerate sinks")
            continue

        for blo, bhi, label in bands_with_all:
            mask = (N_vals >= blo) & (N_vals <= bhi)
            W_band = W_full[mask]

            # Drop zero-sum rows
            row_sums = W_band.sum(axis=1)
            W_band = W_band[row_sums > 0]
            n_pts = W_band.shape[0]

            if n_pts < MIN_BAND_SIZE:
                print(f"  {label:>20s}: n={n_pts:,} (< {MIN_BAND_SIZE}, skipping)")
                for k in k_list:
                    all_rows.append({
                        "policy": policy, "band": label,
                        "band_lo": blo, "band_hi": bhi,
                        "n_points": n_pts, "k": k,
                        "d_trimmed": np.nan, "d_median": np.nan,
                        "d_mean": np.nan, "median_r": np.nan,
                        "note": "too_few_points"
                    })
                continue

            # ILR transform
            ilr = ilr_transform(W_band, eps=PSEUDOCOUNT_EPS)

            # kNN dimension
            usable_k = [k for k in k_list if k < n_pts - 1]
            if not usable_k:
                print(f"  {label:>20s}: n={n_pts:,} (too few for any k)")
                continue

            res = knn_dimension(ilr, usable_k, trim_frac=KNN_TRIM_FRACTION)

            # Print summary at reference k
            ref_entry = None
            for entry in res["summary"]:
                if entry["k"] == reference_k:
                    ref_entry = entry
                    break
            if ref_entry is None and res["summary"]:
                ref_entry = res["summary"][len(res["summary"]) // 2]

            if ref_entry:
                print(f"  {label:>20s}: n={n_pts:>5,}  "
                      f"d_hat(k={ref_entry['k']})={ref_entry['d_trimmed']:.2f}  "
                      f"(median={ref_entry['d_median']:.2f}, "
                      f"mean={ref_entry['d_mean']:.2f})")

            for entry in res["summary"]:
                all_rows.append({
                    "policy": policy, "band": label,
                    "band_lo": blo, "band_hi": bhi,
                    "n_points": n_pts, "k": entry["k"],
                    "d_trimmed": entry["d_trimmed"],
                    "d_median": entry["d_median"],
                    "d_mean": entry["d_mean"],
                    "median_r": entry["median_r"],
                    "note": ""
                })

    summary_df = pd.DataFrame(all_rows)
    summary_csv = outpath / "scale_invariance_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nWrote: {summary_csv}")

    # --- Console summary table at reference k ---
    print(f"\n{'='*70}")
    print(f"SCALE INVARIANCE SUMMARY  (d_hat trimmed mean, k={reference_k})")
    print(f"{'='*70}")

    ref_df = summary_df[summary_df["k"] == reference_k].copy()
    if ref_df.empty:
        closest_k = min(k_list, key=lambda k: abs(k - reference_k))
        ref_df = summary_df[summary_df["k"] == closest_k].copy()
        print(f"  (using k={closest_k} as fallback)")

    policy_names = sorted(groups.keys())

    # Header
    header = f"{'Band':>20s}  {'n':>6s}"
    for pol in policy_names:
        header += f"  {pol:>10s}"
    print(header)
    print("-" * len(header))

    for _, _, label in bands_with_all:
        band_data = ref_df[ref_df["band"] == label]
        if band_data.empty:
            continue
        n_pts = int(band_data.iloc[0]["n_points"])
        row_str = f"{label:>20s}  {n_pts:>6,}"
        for pol in policy_names:
            pol_row = band_data[band_data["policy"] == pol]
            if pol_row.empty or np.isnan(pol_row.iloc[0]["d_trimmed"]):
                row_str += f"  {'---':>10s}"
            else:
                d = pol_row.iloc[0]["d_trimmed"]
                row_str += f"  {d:>10.2f}"
        print(row_str)

    # --- Stability statistics ---
    print(f"\n{'='*70}")
    print("STABILITY: coefficient of variation (CV) across bands, excluding ALL")
    print(f"{'='*70}")
    for pol in policy_names:
        pol_bands = ref_df[(ref_df["policy"] == pol) & (ref_df["band"] != "ALL")]
        vals = pol_bands["d_trimmed"].dropna().values
        if len(vals) >= 2:
            mu = np.mean(vals)
            sd = np.std(vals, ddof=1)
            cv = sd / mu if mu > 0 else np.nan
            rng = np.max(vals) - np.min(vals)
            print(f"  {pol:>10s}: mean={mu:.2f}  sd={sd:.2f}  CV={cv:.3f}  "
                  f"range=[{np.min(vals):.2f}, {np.max(vals):.2f}]  span={rng:.2f}")
        else:
            print(f"  {pol:>10s}: insufficient bands")

    # --- Plots ---
    if HAVE_PLT:
        _plot_line(summary_df, reference_k, bands, policy_names,
                   outpath / "fig_scale_invariance.png")
        _plot_heatmap(summary_df, reference_k, bands, policy_names,
                      outpath / "fig_scale_invariance_heatmap.png")
        _plot_multi_k(summary_df, k_list, bands, policy_names,
                      outpath / "fig_scale_invariance_multi_k.png")

    print(f"\nDone. Outputs in: {outpath}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_line(df: pd.DataFrame, ref_k: int,
               bands: List[Tuple[int, int, str]],
               policies: List[str], out_png: Path):
    """d_hat vs magnitude band for each policy, at reference k."""
    ref = df[(df["k"] == ref_k) & (df["band"] != "ALL")]

    fig, ax = plt.subplots(figsize=(10, 6))

    band_labels = [label for _, _, label in bands]
    x_pos = np.arange(len(band_labels))

    colors = plt.cm.Set2(np.linspace(0, 1, max(len(policies), 3)))

    for i, pol in enumerate(policies):
        pol_data = ref[ref["policy"] == pol]
        y_vals = []
        for _, _, label in bands:
            row = pol_data[pol_data["band"] == label]
            if row.empty or np.isnan(row.iloc[0]["d_trimmed"]):
                y_vals.append(np.nan)
            else:
                y_vals.append(row.iloc[0]["d_trimmed"])

        ax.plot(x_pos, y_vals, marker="o", linewidth=2, markersize=8,
                label=pol, color=colors[i])

    ax.set_xticks(x_pos)
    ax.set_xticklabels(band_labels, rotation=30, ha="right", fontsize=9)
    ax.set_xlabel("Magnitude band", fontsize=11)
    ax.set_ylabel(f"d_hat (kNN trimmed mean, k={ref_k})", fontsize=11)
    ax.set_title("Intrinsic dimension across orders of magnitude", fontsize=13)
    ax.legend(fontsize=10)
    ax.axhline(y=9, color="gray", linestyle=":", alpha=0.5, label="ambient (9)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"  -> {out_png.name}")


def _plot_heatmap(df: pd.DataFrame, ref_k: int,
                  bands: List[Tuple[int, int, str]],
                  policies: List[str], out_png: Path):
    """Band x policy heatmap of d_hat."""
    ref = df[(df["k"] == ref_k) & (df["band"] != "ALL")]

    band_labels = [label for _, _, label in bands]
    matrix = np.full((len(band_labels), len(policies)), np.nan)

    for i, (_, _, label) in enumerate(bands):
        for j, pol in enumerate(policies):
            row = ref[(ref["band"] == label) & (ref["policy"] == pol)]
            if not row.empty:
                matrix[i, j] = row.iloc[0]["d_trimmed"]

    fig, ax = plt.subplots(figsize=(8, max(4, len(band_labels) * 0.7)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis",
                   vmin=np.nanmin(matrix) * 0.9,
                   vmax=np.nanmax(matrix) * 1.1)

    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(policies, fontsize=10)
    ax.set_yticks(range(len(band_labels)))
    ax.set_yticklabels(band_labels, fontsize=9)
    ax.set_xlabel("Policy", fontsize=11)
    ax.set_ylabel("Magnitude band", fontsize=11)
    ax.set_title(f"d_hat heatmap (k={ref_k})", fontsize=13)

    # Annotate cells
    for i in range(len(band_labels)):
        for j in range(len(policies)):
            v = matrix[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="white" if v < np.nanmean(matrix) else "black")

    fig.colorbar(im, ax=ax, label="d_hat", shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"  -> {out_png.name}")


def _plot_multi_k(df: pd.DataFrame, k_list: List[int],
                  bands: List[Tuple[int, int, str]],
                  policies: List[str], out_png: Path):
    """For each policy, plot d_hat vs k for each band (stability across k)."""
    n_pol = len(policies)
    fig, axes = plt.subplots(1, n_pol, figsize=(4 * n_pol, 5), sharey=True)
    if n_pol == 1:
        axes = [axes]

    band_labels = [label for _, _, label in bands]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(bands), 3)))

    for ax, pol in zip(axes, policies):
        for i, (_, _, label) in enumerate(bands):
            band_data = df[(df["policy"] == pol) & (df["band"] == label)]
            if band_data.empty:
                continue
            band_data = band_data.sort_values("k")
            ax.plot(band_data["k"], band_data["d_trimmed"],
                    marker=".", linewidth=1.5, label=label, color=colors[i])

        ax.set_xlabel("k", fontsize=10)
        ax.set_title(pol, fontsize=11)
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("d_hat (trimmed mean)", fontsize=10)
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Dimension stability across k and magnitude", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"  -> {out_png.name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Scale-invariance analysis of intrinsic dimension")
    ap.add_argument("csv", help="Embedding CSV from descent_graph_sink_weights.py")
    ap.add_argument("--outdir", default=None,
                    help="Output directory (default: <csv>_scale)")
    ap.add_argument("--policies", default=None,
                    help="Comma-separated policy filter (default: all detected)")
    ap.add_argument("--k_values", default="10,15,20,25,30",
                    help="Comma-separated k values for kNN (default: 10,15,20,25,30)")
    ap.add_argument("--reference_k", type=int, default=20,
                    help="Reference k for summary table (default: 20)")
    args = ap.parse_args()

    csv_path = args.csv
    outdir = args.outdir or str(Path(csv_path).with_suffix("")) + "_scale"
    policies = [p.strip() for p in args.policies.split(",")] if args.policies else None
    k_values = [int(k.strip()) for k in args.k_values.split(",")]

    run_scale_analysis(csv_path, outdir, policies, k_values, args.reference_k)


if __name__ == "__main__":
    main()
