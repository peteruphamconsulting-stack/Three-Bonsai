#!/usr/bin/env python3
"""
run_tda_persistence.py

Compute persistent homology statistics for sink-weight embeddings across policies,
using compositional geometry (CLR or ILR) with subsampling replicates.

Key fixes vs prior scripts:
  - CLR is correctly named CLR.
  - True ILR is provided (Helmert orthonormal basis), not mislabeled CLR.
  - No forced PCA(2) unless you explicitly request it.
  - Subsampling runs multiple replicates and reports mean/std stability.

Expected CSV columns:
  - Weight columns: W_{policy}_{sink} for each sink in the chosen sink set
      e.g., W_down_2, W_down_3, ... for 10-sink
  - Optional validity filter per policy: ok_{policy} == 1

Policies searched (auto-detected): down, up, quarter, center
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Optional plotting
try:
    import matplotlib.pyplot as plt
    from persim import plot_diagrams
    PLOTTING_AVAILABLE = True
except Exception:
    PLOTTING_AVAILABLE = False

# TDA
try:
    from ripser import ripser
except Exception as e:
    print("ERROR: ripser not found. Install with: pip install ripser", file=sys.stderr)
    raise

# Optional PCA
try:
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


SINKS_10 = [2, 3, 5, 19, 29, 37, 47, 59, 73, 97]
SINKS_12 = [11, 13, 17, 19, 23, 29, 31, 37, 47, 59, 73, 97]
DEFAULT_POLICIES = ["down", "up", "quarter", "center", "random"]


# ----------------------------
# Compositional transforms
# ----------------------------
def normalize_to_simplex(X: np.ndarray) -> np.ndarray:
    """Row-normalize to sum 1; if a row sums to 0, leave it as all-zeros."""
    row_sums = X.sum(axis=1, keepdims=True)
    safe = row_sums.copy()
    safe[safe == 0] = 1.0
    return X / safe


def replace_zeros_and_renormalize(X: np.ndarray, eps: float) -> np.ndarray:
    """
    Replace zeros with eps, then renormalize rows to sum 1.
    This is a standard pseudocount strategy for log-ratio transforms.
    """
    Xc = X.copy()
    Xc[Xc <= 0] = eps
    return normalize_to_simplex(Xc)


def clr_transform(X_simplex: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    CLR transform:
      clr(x) = log(x) - mean(log(x))  (row-wise)
    """
    Xc = replace_zeros_and_renormalize(X_simplex, eps=eps)
    logX = np.log(Xc)
    return logX - logX.mean(axis=1, keepdims=True)


def helmert_basis(D: int) -> np.ndarray:
    """
    Construct a D x (D-1) Helmert sub-matrix with orthonormal columns,
    spanning the subspace orthogonal to the all-ones vector.

    This yields a valid ILR basis: ilr(x) = clr(x) @ H
    and preserves Aitchison distances (isometry).
    """
    H = np.zeros((D, D - 1), dtype=float)
    for j in range(1, D):  # j = 1..D-1
        # First j entries: 1/sqrt(j*(j+1)), entry j+1: -j/sqrt(j*(j+1))
        denom = np.sqrt(j * (j + 1.0))
        H[:j, j - 1] = 1.0 / denom
        H[j, j - 1] = -float(j) / denom
    return H


def ilr_transform(X_simplex: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    ILR transform via Helmert basis:
      ilr(x) = clr(x) @ H   where H is D x (D-1) orthonormal basis
    """
    X_clr = clr_transform(X_simplex, eps=eps)
    D = X_clr.shape[1]
    H = helmert_basis(D)
    return X_clr @ H


def transform_points(
    X_simplex: np.ndarray,
    space: str,
    eps: float,
    pca_dim: Optional[int] = None,
    pca_seed: int = 0,
) -> np.ndarray:
    """
    Apply CLR/ILR, then optional PCA reduction.
    """
    space = space.lower()
    if space == "clr":
        Z = clr_transform(X_simplex, eps=eps)
    elif space == "ilr":
        Z = ilr_transform(X_simplex, eps=eps)
    else:
        raise ValueError(f"Unknown space '{space}'. Use 'clr' or 'ilr'.")

    if pca_dim is not None:
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn not available, but --pca_dim was requested.")
        if pca_dim <= 0 or pca_dim > Z.shape[1]:
            raise ValueError(f"--pca_dim must be in [1, {Z.shape[1]}]. Got {pca_dim}.")
        pca = PCA(n_components=pca_dim, random_state=pca_seed)
        Z = pca.fit_transform(Z)

    return Z


# ----------------------------
# Persistence helpers
# ----------------------------
def finite_lifetimes(diagram: np.ndarray) -> np.ndarray:
    """
    Return finite lifetimes (death - birth), excluding inf deaths.
    diagram shape: (k, 2) or empty.
    """
    if diagram is None or len(diagram) == 0:
        return np.array([], dtype=float)
    births = diagram[:, 0]
    deaths = diagram[:, 1]
    mask = np.isfinite(deaths) & np.isfinite(births)
    lifetimes = deaths[mask] - births[mask]
    lifetimes = lifetimes[np.isfinite(lifetimes)]
    return lifetimes


def diagram_stats(diagram: np.ndarray) -> Dict[str, float]:
    """
    Compute average and max lifetime for a diagram; NaN if no finite lifetimes.
    """
    lt = finite_lifetimes(diagram)
    if lt.size == 0:
        return {"avg": np.nan, "max": np.nan, "count": 0}
    return {"avg": float(np.mean(lt)), "max": float(np.max(lt)), "count": int(lt.size)}


# ----------------------------
# Data extraction
# ----------------------------
def detect_policies(df: pd.DataFrame, sinks: List[int]) -> List[str]:
    """
    A policy is valid if all required weight columns exist.
    """
    valid = []
    for p in DEFAULT_POLICIES:
        cols = [f"W_{p}_{s}" for s in sinks]
        if all(c in df.columns for c in cols):
            valid.append(p)
    return valid


def get_policy_matrix(df: pd.DataFrame, policy: str, sinks: List[int]) -> np.ndarray:
    """
    Extract weights for a policy, applying optional ok_{policy} filter if present.
    """
    ok_col = f"ok_{policy}"
    if ok_col in df.columns:
        sub = df[df[ok_col] == 1].copy()
    else:
        sub = df

    cols = [f"W_{policy}_{s}" for s in sinks]
    if not all(c in sub.columns for c in cols):
        missing = [c for c in cols if c not in sub.columns]
        raise KeyError(f"Missing columns for policy '{policy}': {missing}")

    X = sub[cols].to_numpy(dtype=float)
    X = normalize_to_simplex(X)
    return X


# ----------------------------
# Main analysis
# ----------------------------
def run(
    csv_file: str,
    sinks_mode: str,
    space: str,
    maxdim: int,
    samples: int,
    reps: int,
    seed: int,
    eps: float,
    pca_dim: Optional[int],
    plot: bool,
    plot_rep: int,
    out_png: Optional[str],
    stats_csv: Optional[str],
) -> None:
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV not found: {csv_file}")

    df = pd.read_csv(csv_file)

    sinks = SINKS_12 if sinks_mode == "12" else SINKS_10
    print(f"--- Loading {csv_file} ---")
    print(f"Sink model: {sinks_mode}-sink  sinks={sinks}")
    print(f"Space: {space.upper()}   maxdim=H{maxdim}   eps={eps:g}")
    if pca_dim is not None:
        print(f"PCA reduction: {pca_dim} dims")
    else:
        print("PCA reduction: none (recommended for true geometry)")

    policies = detect_policies(df, sinks)
    if not policies:
        print("ERROR: No valid policies found. Expected columns like W_down_<sink>.", file=sys.stderr)
        return
    print(f"Policies detected: {', '.join(policies)}")

    rng = np.random.default_rng(seed)

    # Storage: rows = (rep, policy, dim)
    records = []

    # For plotting: store diagrams for chosen replicate
    plot_diagrams_by_policy = {}

    for rep_idx in range(reps):
        rep_seed = int(rng.integers(0, 2**31 - 1))
        rep_rng = np.random.default_rng(rep_seed)

        print(f"\n=== Rep {rep_idx+1}/{reps} (seed={rep_seed}) ===")
        for policy in policies:
            X = get_policy_matrix(df, policy, sinks)
            n = X.shape[0]
            if n == 0:
                print(f"  {policy.upper()}: no rows after filter.")
                continue

            if samples is not None and samples > 0 and n > samples:
                idx = rep_rng.choice(n, size=samples, replace=False)
                Xs = X[idx]
                print(f"  {policy.upper()}: subsample {samples}/{n}")
            else:
                Xs = X
                print(f"  {policy.upper()}: using all {n}")

            Z = transform_points(
                Xs,
                space=space,
                eps=eps,
                pca_dim=pca_dim,
                pca_seed=rep_seed,
            )

            # Run ripser
            res = ripser(Z, maxdim=maxdim)
            dgms = res["dgms"]

            # Capture for plotting on a chosen replicate
            if plot and rep_idx == plot_rep:
                plot_diagrams_by_policy[policy] = dgms

            # Stats by homology dimension (H1..Hmaxdim)
            for d in range(1, maxdim + 1):
                st = diagram_stats(dgms[d] if d < len(dgms) else np.empty((0, 2)))
                records.append(
                    {
                        "rep": rep_idx,
                        "policy": policy,
                        "H": d,
                        "avg_lifetime": st["avg"],
                        "max_lifetime": st["max"],
                        "count": st["count"],
                        "n_points": int(Z.shape[0]),
                        "space": space,
                        "sinks_mode": sinks_mode,
                        "pca_dim": -1 if pca_dim is None else int(pca_dim),
                        "rep_seed": rep_seed,
                    }
                )

            # Print quick per-policy summary for this rep
            if maxdim >= 1:
                s1 = diagram_stats(dgms[1] if len(dgms) > 1 else np.empty((0, 2)))
                print(f"    H1: avg={s1['avg']:.6g}  max={s1['max']:.6g}  count={s1['count']}")
            if maxdim >= 2:
                s2 = diagram_stats(dgms[2] if len(dgms) > 2 else np.empty((0, 2)))
                print(f"    H2: avg={s2['avg']:.6g}  max={s2['max']:.6g}  count={s2['count']}")

    stats_df = pd.DataFrame(records)

    # Aggregate stability summary
    print("\n=== Aggregate summary (mean ± std over reps) ===")
    for policy in policies:
        for d in range(1, maxdim + 1):
            sub = stats_df[(stats_df["policy"] == policy) & (stats_df["H"] == d)]
            if len(sub) == 0:
                continue
            mean_avg = np.nanmean(sub["avg_lifetime"])
            std_avg = np.nanstd(sub["avg_lifetime"])
            mean_max = np.nanmean(sub["max_lifetime"])
            std_max = np.nanstd(sub["max_lifetime"])
            mean_cnt = float(np.mean(sub["count"]))
            std_cnt = float(np.std(sub["count"]))
            appear = float(np.mean(sub["count"].to_numpy() > 0))
            print(
                f"{policy.upper():8s} H{d}: "
                f"avg_life={mean_avg:.6g} ± {std_avg:.6g}   "
                f"max_life={mean_max:.6g} ± {std_max:.6g}   "
                f"count={mean_cnt:.3g} ± {std_cnt:.3g}   "
                f"appear={appear:.3g}"
            )

    # Save per-rep stats CSV if requested
    if stats_csv:
        stats_df.to_csv(stats_csv, index=False)
        print(f"\nWrote per-replicate stats to: {stats_csv}")

    # Plot diagrams for chosen replicate
    if plot:
        if not PLOTTING_AVAILABLE:
            print("Plot requested, but matplotlib/persim not available.", file=sys.stderr)
        else:
            n_pol = len(plot_diagrams_by_policy)
            if n_pol == 0:
                print("No diagrams captured for plotting (check plot_rep and policies).", file=sys.stderr)
            else:
                cols = 2
                rows = (n_pol + 1) // 2
                fig, axes = plt.subplots(rows, cols, figsize=(14, 6 * rows))
                axes = np.array(axes).ravel()

                for i, policy in enumerate(sorted(plot_diagrams_by_policy.keys())):
                    dgms = plot_diagrams_by_policy[policy]
                    ax = axes[i]
                    plot_diagrams(dgms, show=False, ax=ax)
                    ax.set_title(
                        f"{policy.upper()}  ({space.upper()}, "
                        f"{sinks_mode}-sink, maxdim=H{maxdim}"
                        + (f", PCA={pca_dim}" if pca_dim is not None else "")
                        + f")  rep={plot_rep}"
                    )

                for j in range(i + 1, len(axes)):
                    axes[j].axis("off")

                plt.tight_layout()
                if out_png:
                    plt.savefig(out_png, dpi=150)
                    print(f"Saved diagram plot: {out_png}")
                else:
                    out_png_auto = f"tda_{sinks_mode}sink_{space}_H{maxdim}" + (f"_pca{pca_dim}" if pca_dim else "") + ".png"
                    plt.savefig(out_png_auto, dpi=150)
                    print(f"Saved diagram plot: {out_png_auto}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_file", help="Input embeddings CSV (with W_{policy}_{sink} columns).")

    ap.add_argument("--sinks", choices=["10", "12"], default="10", help="Choose sink model (10 or 12).")
    ap.add_argument("--space", choices=["clr", "ilr"], default="clr", help="Compositional space for TDA.")
    ap.add_argument("--maxdim", type=int, default=1, choices=[1, 2], help="Max homology dimension (1 or 2).")

    ap.add_argument("--samples", type=int, default=2000, help="Subsample size per policy per replicate.")
    ap.add_argument("--reps", type=int, default=1, help="Number of subsampling replicates.")
    ap.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    ap.add_argument("--eps", type=float, default=1e-12, help="Pseudocount for zeros before log-ratio.")

    ap.add_argument("--pca_dim", type=int, default=None,
                    help="Optional PCA dimension after CLR/ILR (use sparingly; changes topology).")

    ap.add_argument("--plot", action="store_true", help="Plot persistence diagrams for one replicate.")
    ap.add_argument("--plot_rep", type=int, default=0, help="Which replicate index to plot (0-based).")
    ap.add_argument("--out_png", type=str, default=None, help="Output PNG filename for diagram plot.")
    ap.add_argument("--stats_csv", type=str, default=None, help="Write per-replicate stats to CSV.")

    args = ap.parse_args()

    if args.pca_dim is not None and args.pca_dim > 0 and not SKLEARN_AVAILABLE:
        print("ERROR: scikit-learn is required for --pca_dim. Install with: pip install scikit-learn", file=sys.stderr)
        sys.exit(1)

    if args.plot and args.plot_rep < 0:
        print("ERROR: --plot_rep must be >= 0.", file=sys.stderr)
        sys.exit(1)

    run(
        csv_file=args.csv_file,
        sinks_mode=args.sinks,
        space=args.space,
        maxdim=args.maxdim,
        samples=args.samples,
        reps=args.reps,
        seed=args.seed,
        eps=args.eps,
        pca_dim=args.pca_dim,
        plot=args.plot,
        plot_rep=args.plot_rep,
        out_png=args.out_png,
        stats_csv=args.stats_csv,
    )


if __name__ == "__main__":
    main()
