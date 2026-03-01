#!/usr/bin/env python3
"""
embedding_ilr_pca_viz.py

Ingest out_10sink.csv or out_12sink.csv produced by HolisticEmbedding* scripts.
For each policy:
  1) Extract sink-weight/count columns (W_{policy}_{sink})
  2) Row-normalize to simplex (divide by row sum)
  3) ILR (or CLR) transform in Aitchison geometry
  4) PCA on ILR coordinates
  5) Produce user-directed visualizations (2D and 3D), with optional coloring.

Defaults are set in CONFIG below; override via CLI flags.

Notes
-----
- This script is intended for visualization and exploratory analysis.
- PCA (dim-reduction) is fine for visuals; do NOT use PCA-reduced coordinates to make
  persistent homology (topology) claims.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# PCA
try:
    from sklearn.decomposition import PCA
except Exception as e:
    print("ERROR: scikit-learn not available. Install with: pip install scikit-learn", file=sys.stderr)
    raise

# Plotting
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
except Exception:
    pass


# -----------------------------
# CONFIG (defaults)
# -----------------------------
@dataclass
class Config:
    eps: float = 1e-12                  # pseudocount for zeros before log-ratio
    sample_n: int = 0                   # 0 => no subsample; otherwise random subsample per policy
    seed: int = 42

    # Transform
    space: str = "ilr"                  # "ilr" or "clr"

    # PCA
    pca_components: int = 6             # compute at least up to PC3; can be larger for coloring/regime

    # Outputs
    outdir: str = "viz_out"
    dpi: int = 160

    # Regime heuristic
    regime_mode: str = "none"           # "none", "pc1_terciles", "kmeans3"
    regime_kmeans_k: int = 3


    regime_kmax: int = 6

    # Additional continuous color modes
    # - density_knn: local density proxy via kNN distance in PCA space
    # - scale_resid: scale-invariant proxy via residual norm after smoothing PCs vs log10(N)
    density_knn_k: int = 10              # k for kNN distance (higher => smoother)
    density_pca_dims: int = 5            # number of PCA dims to use for density (<=0 => all computed PCs)
    density_standardize: bool = True     # standardize PCA coords before kNN
    scale_resid_dims: int = 3            # number of PCs to de-trend vs log10(N)
    scale_resid_window: int = 0          # rolling window (odd). 0 => auto based on N

CONFIG = Config()

# Known sink sets
SINKS_10 = [2, 3, 5, 19, 29, 37, 47, 59, 73, 97]
SINKS_12 = [11, 13, 17, 19, 23, 29, 31, 37, 47, 59, 73, 97]
DEFAULT_POLICIES = ["down", "up", "quarter", "center"]


# -----------------------------
# Helpers: compositional transforms
# -----------------------------
def normalize_to_simplex(X: np.ndarray) -> np.ndarray:
    row_sums = X.sum(axis=1, keepdims=True)
    safe = row_sums.copy()
    safe[safe == 0] = 1.0
    return X / safe


def replace_zeros_and_renormalize(X: np.ndarray, eps: float) -> np.ndarray:
    Xc = X.copy()
    Xc[Xc <= 0] = eps
    return normalize_to_simplex(Xc)


def clr_transform(X_simplex: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    Xc = replace_zeros_and_renormalize(X_simplex, eps=eps)
    logX = np.log(Xc)
    return logX - logX.mean(axis=1, keepdims=True)


def helmert_basis(D: int) -> np.ndarray:
    # D x (D-1) orthonormal basis orthogonal to 1-vector
    H = np.zeros((D, D - 1), dtype=float)
    for j in range(1, D):  # 1..D-1
        denom = np.sqrt(j * (j + 1.0))
        H[:j, j - 1] = 1.0 / denom
        H[j, j - 1] = -float(j) / denom
    return H


def ilr_transform(X_simplex: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X_clr = clr_transform(X_simplex, eps=eps)
    D = X_clr.shape[1]
    H = helmert_basis(D)
    return X_clr @ H


# -----------------------------
# Column/policy detection
# -----------------------------
def detect_sinks_mode(df: pd.DataFrame) -> Tuple[str, List[int]]:
    # Prefer explicit detection by presence of W_*_<sink> columns
    def has_all(sinks: List[int]) -> bool:
        for p in DEFAULT_POLICIES:
            cols = [f"W_{p}_{s}" for s in sinks]
            if all(c in df.columns for c in cols):
                return True
        return False

    if has_all(SINKS_12):
        return "12", SINKS_12
    if has_all(SINKS_10):
        return "10", SINKS_10

    # Fall back: infer sinks by parsing column names like W_down_97
    sinks_found = set()
    for c in df.columns:
        m = re.match(r"^W_([A-Za-z]+)_(\d+)$", c)
        if m:
            sinks_found.add(int(m.group(2)))
    if len(sinks_found) >= 8:
        sinks = sorted(sinks_found)
        # if user has a custom set, keep it
        return "auto", sinks

    raise ValueError("Could not detect sink set. Expected columns like W_down_2 ...")


def detect_policies(df: pd.DataFrame, sinks: List[int]) -> List[str]:
    valid = []
    for p in DEFAULT_POLICIES:
        cols = [f"W_{p}_{s}" for s in sinks]
        if all(c in df.columns for c in cols):
            valid.append(p)
    return valid


def find_integer_column(df: pd.DataFrame) -> Optional[str]:
    # Common candidates seen in these workflows
    candidates = ["N", "n", "M", "m", "integer", "value", "root", "start"]
    for c in candidates:
        if c in df.columns:
            return c
    # Otherwise: first integer-like column with large values
    for c in df.columns:
        if pd.api.types.is_integer_dtype(df[c]) or pd.api.types.is_numeric_dtype(df[c]):
            # heuristic: mostly whole numbers and reasonably large
            s = df[c].dropna()
            if len(s) == 0:
                continue
            frac = np.mean(np.isclose(s.values, np.round(s.values)))
            if frac > 0.98 and np.nanmedian(s.values) > 1000:
                return c
    return None


def get_policy_matrix(df: pd.DataFrame, policy: str, sinks: List[int]) -> np.ndarray:
    ok_col = f"ok_{policy}"
    sub = df[df[ok_col] == 1].copy() if ok_col in df.columns else df
    cols = [f"W_{policy}_{s}" for s in sinks]
    missing = [c for c in cols if c not in sub.columns]
    if missing:
        raise KeyError(f"Missing columns for policy '{policy}': {missing}")
    X = sub[cols].to_numpy(dtype=float)
    X = normalize_to_simplex(X)
    return X, sub


# -----------------------------
# Coloring / regimes
# -----------------------------
def compute_magnitude(ilr_or_clr: np.ndarray) -> np.ndarray:
    # Euclidean norm in ILR/CLR space is a reasonable "distance from uniform"
    return np.linalg.norm(ilr_or_clr, axis=1)


def compute_knn_distance(
    pcs: np.ndarray,
    k: int = 10,
    dims: int = 5,
    standardize: bool = True,
    metric: str = "euclidean",
) -> np.ndarray:
    """kNN distance proxy for local density.

    Returns the distance to the k-th nearest neighbor in (optionally standardized) PCA space.
    Smaller values indicate higher local density.
    """
    try:
        from sklearn.neighbors import NearestNeighbors
    except Exception:
        # Fallback: no sklearn => return NaNs so caller can degrade gracefully
        return np.full((pcs.shape[0],), np.nan, dtype=float)

    X = pcs
    if dims and dims > 0:
        X = X[:, : min(int(dims), X.shape[1])]

    if standardize:
        try:
            from sklearn.preprocessing import StandardScaler
            X = StandardScaler().fit_transform(X)
        except Exception:
            pass

    n = X.shape[0]
    if n <= 1:
        return np.zeros((n,), dtype=float)

    k_eff = int(max(2, min(int(k), max(2, n))))
    nn = NearestNeighbors(n_neighbors=k_eff, metric=metric)
    nn.fit(X)
    dists, _ = nn.kneighbors(X)
    # distance to the k-th neighbor (last column)
    return dists[:, -1].astype(float)


def compute_scale_residual_norm(
    pcs: np.ndarray,
    logn: np.ndarray,
    dims: int = 3,
    window: int = 0,
) -> np.ndarray:
    """Scale-invariant proxy: residual norm after smoothing PCs vs log10(N).

    Sorts points by log10(N), applies a centered rolling median to each of the first `dims` PCs,
    subtracts the trend, and returns the residual vector norm per point.
    """
    n = pcs.shape[0]
    if n == 0:
        return np.array([], dtype=float)

    dims_eff = min(int(dims), pcs.shape[1])
    if dims_eff <= 0:
        return np.zeros((n,), dtype=float)

    order = np.argsort(logn)
    resid = np.zeros((n, dims_eff), dtype=float)

    # Choose a robust window if not provided; ensure it is odd and within [3, n]
    if window and window > 0:
        win = int(window)
    else:
        # heuristic: ~5% of points, minimum 51, maximum n (odd)
        win = max(51, int(max(1, n // 20)) * 2 + 1)

    win = max(3, min(win, n if (n % 2 == 1) else max(3, n - 1)))
    if win % 2 == 0:
        win += 1
        if win > n:
            win = n if (n % 2 == 1) else max(3, n - 1)

    for j in range(dims_eff):
        y = pcs[order, j]
        # rolling median over sorted-by-scale points (robust to outliers)
        sm = pd.Series(y).rolling(window=win, center=True, min_periods=1).median().to_numpy()
        resid[order, j] = y - sm

    return np.linalg.norm(resid, axis=1).astype(float)


def compute_regime(labels_source: np.ndarray, mode: str, seed: int, k: int = 3, kmax: int = 6) -> np.ndarray:
    """
    Returns integer labels in {0..K-1}. If the selected/fit K==1, returns an all-zero vector.
    """
    mode = mode.lower()
    if mode == "none":
        return np.array([], dtype=int)

    if mode == "pc1_terciles":
        x = labels_source
        qs = np.quantile(x, [1/3, 2/3])
        return np.digitize(x, qs, right=False)

    if mode == "kmeans3":
        # fixed K (default 3)
        try:
            from sklearn.cluster import KMeans
        except Exception:
            raise RuntimeError("kmeans requested but scikit-learn clustering not available.")
        X = labels_source.reshape(-1, 1) if labels_source.ndim == 1 else labels_source
        km = KMeans(n_clusters=k, random_state=seed, n_init=20)
        return km.fit_predict(X)

    if mode == "auto_kmeans":
        # Choose K in [1..kmax] by silhouette score (prefers simpler models if tie/near-tie).
        # If best K==1, caller can treat this as "single regime".
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
        except Exception:
            raise RuntimeError("auto_kmeans requested but scikit-learn clustering not available.")
        X = labels_source.reshape(-1, 1) if labels_source.ndim == 1 else labels_source
        n = X.shape[0]
        # Need at least 2 points for K=1, and at least K points for K clusters
        kmax_eff = max(1, min(int(kmax), max(1, n - 1)))
        best_k = 1
        best_score = -np.inf
        best_labels = np.zeros(n, dtype=int)

        # Compute for K>=2 only; silhouette undefined for K=1. We still allow K=1 baseline.
        for kk in range(2, kmax_eff + 1):
            km = KMeans(n_clusters=kk, random_state=seed, n_init=20)
            labels = km.fit_predict(X)
            # If clustering collapses (rare), skip
            if len(np.unique(labels)) < 2:
                continue
            try:
                score = silhouette_score(X, labels, metric="euclidean")
            except Exception:
                continue
            # Prefer smaller K unless score improves meaningfully
            if score > best_score + 1e-3:
                best_score = score
                best_k = kk
                best_labels = labels

        # If no valid K>=2 beats baseline, return all zeros (single regime)
        return best_labels if best_k > 1 else np.zeros(n, dtype=int)

    if mode == "gmm_bic":
        # Choose K in [1..kmax] by BIC using GaussianMixture on features (e.g., PCs).
        try:
            from sklearn.mixture import GaussianMixture
        except Exception:
            raise RuntimeError("gmm_bic requested but scikit-learn mixture not available.")
        X = labels_source.reshape(-1, 1) if labels_source.ndim == 1 else labels_source
        n = X.shape[0]
        kmax_eff = max(1, min(int(kmax), max(1, n)))
        best_bic = np.inf
        best = np.zeros(n, dtype=int)
        for kk in range(1, kmax_eff + 1):
            gm = GaussianMixture(n_components=kk, covariance_type="full", random_state=seed, n_init=5)
            gm.fit(X)
            bic = gm.bic(X)
            if bic < best_bic:
                best_bic = bic
                best = gm.predict(X)
        return best

    raise ValueError(f"Unknown regime_mode={mode}")
    if mode == "pc1_terciles":
        x = labels_source
        qs = np.quantile(x, [1/3, 2/3])
        return np.digitize(x, qs, right=False)

    if mode == "kmeans3":
        try:
            from sklearn.cluster import KMeans
        except Exception:
            raise RuntimeError("kmeans3 requested but scikit-learn clustering not available.")
        X = labels_source.reshape(-1, 1) if labels_source.ndim == 1 else labels_source
        km = KMeans(n_clusters=k, random_state=seed, n_init=20)
        return km.fit_predict(X)

    raise ValueError(f"Unknown regime_mode={mode}")


def make_color_vector(
    df_sub: pd.DataFrame,
    int_col: Optional[str],
    Z: np.ndarray,
    pcs: np.ndarray,
    color_by: str,
    regime_mode: str,
    seed: int,
    mod_base: int,
    residue_value: int,
) -> Tuple[Optional[np.ndarray], Optional[str], bool]:
    """
    Returns: (color_vector, label, is_categorical)
    """
    color_by = color_by.lower()
    if color_by in ("none", "", "off"):
        return None, None, False

    if color_by in ("n", "logn", "scale", "magnitude_n"):
        if int_col is None:
            return None, None, False
        v = df_sub[int_col].to_numpy(dtype=float)
        return np.log10(np.maximum(v, 1.0)), f"log10({int_col})", False

    if color_by in ("parity", "evenodd", "mod2"):
        if int_col is None:
            return None, None, False
        v = df_sub[int_col].to_numpy(dtype=int)
        return (v % 2), "parity (mod 2)", True

    if color_by in ("magnitude", "norm", "distance"):
        v = compute_magnitude(Z)
        return v, f"||{CONFIG.space.upper()}||", False

    if color_by in ("pc1", "pc2", "pc3"):
        idx = int(color_by[-1]) - 1
        if pcs.shape[1] <= idx:
            return None, None, False
        return pcs[:, idx], color_by.upper(), False


    if color_by in ("density_knn", "knn_density", "local_density", "density"):
        # Local density proxy via kNN distance in PCA space (smaller distance => denser).
        k = int(getattr(CONFIG, "density_knn_k", 10))
        dims = int(getattr(CONFIG, "density_pca_dims", 5))
        d = compute_knn_distance(
            pcs,
            k=k,
            dims=dims,
            standardize=bool(getattr(CONFIG, "density_standardize", True)),
        )
        if np.all(np.isnan(d)):
            return None, None, False
        # Convert to "higher = denser" for more intuitive coloring.
        dens = -np.log10(np.maximum(d, 1e-12))
        return dens, f"-log10(kNN dist; k={k})", False

    if color_by in ("scale_resid", "scale_residual", "residual", "residual_norm"):
        # Scale-invariant proxy: residual norm after smoothing PCs vs log10(N).
        if int_col is None:
            return None, None, False
        nvals = df_sub[int_col].to_numpy(dtype=float)
        logn = np.log10(np.maximum(nvals, 1.0))
        dims = int(getattr(CONFIG, "scale_resid_dims", 3))
        win = int(getattr(CONFIG, "scale_resid_window", 0))
        v = compute_scale_residual_norm(pcs, logn, dims=dims, window=win)
        dims_eff = min(max(1, dims), pcs.shape[1])
        return v, f"scale residual ||r|| (PC1..PC{dims_eff})", False



    if color_by in ("regime", "cluster"):
        # Regime labeling:
        # - pc1_terciles: fixed 3 bins by PC1 quantiles
        # - kmeans3: fixed K=3 (fit in full ILR/CLR space)
        # - auto_kmeans: choose K by silhouette (may yield a single regime => no coloring)
        # - gmm_bic: choose K by BIC (may yield a single regime => no coloring)
        if regime_mode == "none":
            regime_mode = "pc1_terciles"

        if regime_mode.lower() == "pc1_terciles":
            labels = compute_regime(pcs[:, 0], "pc1_terciles", seed)
        elif regime_mode.lower() == "kmeans3":
            use = Z  # full ILR/CLR space (most principled for compositional geometry)
            labels = compute_regime(use, "kmeans3", seed, k=3)
        elif regime_mode.lower() == "auto_kmeans":
            use = Z  # full ILR/CLR space (most principled for compositional geometry)
            labels = compute_regime(use, "auto_kmeans", seed, kmax=CONFIG.regime_kmax)
        elif regime_mode.lower() == "gmm_bic":
            use = Z  # full ILR/CLR space (most principled for compositional geometry)
            labels = compute_regime(use, "gmm_bic", seed, kmax=CONFIG.regime_kmax)
        else:
            labels = compute_regime(pcs[:, 0], regime_mode, seed)

        # If the chosen model yields a single cluster, do not color as if multiple regimes exist.
        if len(np.unique(labels)) <= 1:
            return None, None, False

        return labels, f"regime ({regime_mode})", True

    if color_by in ("residue", "mod", "modm", "modclass", "residue_class"):
        # Color by residue class N mod m (default m=30). Uses a numeric colorbar.
        if int_col is None:
            return None, None, False
        mbase = int(mod_base) if int(mod_base) > 0 else 30
        v = df_sub[int_col].to_numpy(dtype=int) % mbase
        return v.astype(float), f"{int_col} mod {mbase}", False

    if color_by in ("residue_bin", "divisible", "factor", "isresidue"):
        # Binary residue coloring: highlight whether N mod m == r (default r=0 => divisibility by m).
        if int_col is None:
            return None, None, False
        mbase = int(mod_base) if int(mod_base) > 0 else 30
        r = int(residue_value) % mbase
        v = df_sub[int_col].to_numpy(dtype=int) % mbase
        labels = (v == r).astype(int)  # 1 = matches, 0 = other
        return labels, f"1 if {int_col} mod {mbase} == {r}", True





    # Unknown
    return None, None, False


# -----------------------------
# Plotting
# -----------------------------
def _apply_coloring(ax, cvals, clabel, categorical: bool):
    if cvals is None:
        return None

    if categorical:
        # use discrete colormap; legend with unique values
        uniq = np.unique(cvals)
        # map to 0..K-1
        mapping = {u: i for i, u in enumerate(uniq)}
        cmapped = np.array([mapping[u] for u in cvals], dtype=int)
        sc = ax.scatter([], [], alpha=0.0)  # placeholder
        # Return mapping for legend use
        return mapping

    else:
        # Continuous colormap with colorbar
        return None


def plot_2d(
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    outpath: str,
    cvals: Optional[np.ndarray],
    clabel: Optional[str],
    categorical: bool,
    dpi: int,
):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)

    if cvals is None:
        ax.scatter(x, y, s=10, alpha=0.65)
    else:
        if categorical:
            uniq = np.unique(cvals)
            for u in uniq:
                mask = (cvals == u)
                ax.scatter(x[mask], y[mask], s=10, alpha=0.70, label=str(u))
            ax.legend(title=clabel, loc="best", frameon=True)
        else:
            sc = ax.scatter(x, y, s=10, alpha=0.65, c=cvals)
            cb = fig.colorbar(sc, ax=ax)
            cb.set_label(clabel)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def plot_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    zlabel: str,
    outpath: str,
    cvals: Optional[np.ndarray],
    clabel: Optional[str],
    categorical: bool,
    dpi: int,
):
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    if cvals is None:
        ax.scatter(x, y, z, s=8, alpha=0.65)
    else:
        if categorical:
            uniq = np.unique(cvals)
            for u in uniq:
                mask = (cvals == u)
                ax.scatter(x[mask], y[mask], z[mask], s=8, alpha=0.70, label=str(u))
            ax.legend(title=clabel, loc="best", frameon=True)
        else:
            sc = ax.scatter(x, y, z, s=8, alpha=0.65, c=cvals)
            cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.08)
            cb.set_label(clabel)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
import re  # placed here so detect_sinks_mode fallback works


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_file", help="out_10sink.csv or out_12sink.csv")
    ap.add_argument("--outdir", default=CONFIG.outdir, help="Output directory for figures")
    ap.add_argument("--out_format", choices=["png", "eps"], default="png",
                    help="Output figure format (png or eps). Default: png")
    ap.add_argument("--space", choices=["ilr", "clr"], default=CONFIG.space, help="CLR or ILR coordinates before PCA")
    ap.add_argument("--sinks", choices=["auto", "10", "12"], default="auto", help="Sink set (auto detects if possible)")
    ap.add_argument("--policies", default="all", help="Comma list like down,up,quarter,center or 'all'")
    ap.add_argument("--sample_n", type=int, default=CONFIG.sample_n, help="Subsample per policy for plotting (0=all)")
    ap.add_argument("--seed", type=int, default=CONFIG.seed, help="RNG seed")
    ap.add_argument("--eps", type=float, default=CONFIG.eps, help="Pseudocount for zeros before log-ratio")
    ap.add_argument("--pca_components", type=int, default=CONFIG.pca_components, help="Number of PCA components to compute")

    ap.add_argument("--plots", default="pc1pc2,pc1pc2pc3",
                    help="Comma list of plots: pc1pc2, pc2pc3, pc1pc2pc3, pc1pc2mag, pc1pc3mag, pc2pc3mag")
    ap.add_argument("--color_by", default="logN",
                    help="Color: none|logN|magnitude|parity|pc1|pc2|pc3|density_knn|scale_resid|regime|residue (N mod m)|residue_bin (N mod m == r)")
    ap.add_argument("--mod_base", type=int, default=30,
                    help="Modulus base for residue coloring when --color_by residue/mod (e.g., 30)")
    ap.add_argument("--residue_value", type=int, default=0,
                    help="Target residue r for binary residue coloring (default 0). Used when --color_by residue_bin/divisible.")
    ap.add_argument("--density_knn_k", type=int, default=CONFIG.density_knn_k,
                    help="k for density_knn: distance to k-th nearest neighbor (default 10)")
    ap.add_argument("--density_pca_dims", type=int, default=CONFIG.density_pca_dims,
                    help="Dims for density_knn: number of PCs used (<=0 => all computed PCs). Default 5.")
    ap.add_argument("--no_density_standardize", action="store_true",
                    help="Disable standardization before density_knn (not recommended).")
    ap.add_argument("--scale_resid_dims", type=int, default=CONFIG.scale_resid_dims,
                    help="Dims for scale_resid: number of PCs to de-trend vs log10(N) (default 3).")
    ap.add_argument("--scale_resid_window", type=int, default=CONFIG.scale_resid_window,
                    help="Rolling median window (odd) for scale_resid (0 => auto).")

    ap.add_argument("--regime_mode", default=CONFIG.regime_mode, choices=["none", "pc1_terciles", "kmeans3", "auto_kmeans", "gmm_bic"],
                    help="Regime heuristic if color_by=regime")
    ap.add_argument("--regime_kmax", type=int, default=CONFIG.regime_kmax,
                    help="Max clusters for auto_kmeans or gmm_bic (k from 1..kmax)")
    args = ap.parse_args()

    CONFIG.regime_kmax = args.regime_kmax
    CONFIG.density_knn_k = args.density_knn_k
    CONFIG.density_pca_dims = args.density_pca_dims
    CONFIG.density_standardize = (not args.no_density_standardize)
    CONFIG.scale_resid_dims = args.scale_resid_dims
    CONFIG.scale_resid_window = args.scale_resid_window


    csv_file = args.csv_file
    if not os.path.exists(csv_file):
        raise FileNotFoundError(csv_file)

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(csv_file)

    # Detect sink set and policies
    if args.sinks == "10":
        sinks_mode, sinks = "10", SINKS_10
    elif args.sinks == "12":
        sinks_mode, sinks = "12", SINKS_12
    else:
        sinks_mode, sinks = detect_sinks_mode(df)

    policies = detect_policies(df, sinks)
    if args.policies.lower() != "all":
        want = [p.strip().lower() for p in args.policies.split(",") if p.strip()]
        policies = [p for p in policies if p in want]

    if not policies:
        raise ValueError("No policies found. Expected columns like W_down_<sink> etc.")

    int_col = find_integer_column(df)

    # RNG for subsampling
    rng = np.random.default_rng(args.seed)

    print(f"Loaded: {csv_file}")
    print(f"Sinks: {sinks_mode}  (D={len(sinks)} => ILR dim={len(sinks)-1})")
    print(f"Policies: {', '.join(policies)}")
    if int_col:
        print(f"Integer column: {int_col}")
    else:
        print("Integer column: (not detected) -> parity/logN coloring disabled")

    plot_specs = [p.strip().lower() for p in args.plots.split(",") if p.strip()]

    # For each policy: compute ILR/CLR, PCA, then plots
    for policy in policies:
        X, sub = get_policy_matrix(df, policy, sinks)

        # Optional subsample for plotting speed
        if args.sample_n and args.sample_n > 0 and X.shape[0] > args.sample_n:
            idx = rng.choice(X.shape[0], size=args.sample_n, replace=False)
            X = X[idx]
            sub = sub.iloc[idx].copy()

        # Transform
        if args.space == "clr":
            Z = clr_transform(X, eps=args.eps)
        else:
            Z = ilr_transform(X, eps=args.eps)

        # PCA
        ncomp = min(args.pca_components, Z.shape[1])
        pca = PCA(n_components=ncomp, random_state=args.seed)
        pcs = pca.fit_transform(Z)
        # "Magnitude" axis for visuals:
        # By your convention, magnitude = log10(N) (scale) if an integer column is available.
        # If not available, fall back to ||ILR|| / ||CLR|| as a distance-from-uniform proxy.
        if int_col is not None:
            mag = np.log10(np.maximum(sub[int_col].to_numpy(dtype=float), 1.0))
            mag_label = f"log10({int_col})"
        else:
            mag = compute_magnitude(Z)
            mag_label = f"||{args.space.upper()}||"

        # Determine coloring
        cvals, clabel, categorical = make_color_vector(
            df_sub=sub,
            int_col=int_col,
            Z=Z,
            pcs=pcs,
            color_by=args.color_by,
            regime_mode=args.regime_mode,
            seed=args.seed,
        mod_base=args.mod_base,
        residue_value=args.residue_value,
    )# Make plots
        for spec in plot_specs:
            outname = f"{os.path.splitext(os.path.basename(csv_file))[0]}_{sinks_mode}sink_{policy}_{args.space}_{spec}"
            if args.color_by.lower() != "none":
                outname += f"_color-{args.color_by.lower()}"
            outpath = os.path.join(args.outdir, outname + f".{args.out_format}")

            if spec == "pc1pc2":
                plot_2d(
                    pcs[:, 0], pcs[:, 1],
                    title=f"{policy.upper()}  {args.space.upper()}→PCA  ({sinks_mode}-sink)",
                    xlabel="PC1", ylabel="PC2",
                    outpath=outpath,
                    cvals=cvals, clabel=clabel, categorical=categorical, dpi=CONFIG.dpi
                )

            elif spec == "pc2pc3":
                if pcs.shape[1] < 3:
                    continue
                plot_2d(
                    pcs[:, 1], pcs[:, 2],
                    title=f"{policy.upper()}  {args.space.upper()}→PCA  ({sinks_mode}-sink)",
                    xlabel="PC2", ylabel="PC3",
                    outpath=outpath,
                    cvals=cvals, clabel=clabel, categorical=categorical, dpi=CONFIG.dpi
                )

            elif spec == "pc1pc2pc3":
                if pcs.shape[1] < 3:
                    continue
                plot_3d(
                    pcs[:, 0], pcs[:, 1], pcs[:, 2],
                    title=f"{policy.upper()}  {args.space.upper()}→PCA  ({sinks_mode}-sink)",
                    xlabel="PC1", ylabel="PC2", zlabel="PC3",
                    outpath=outpath,
                    cvals=cvals, clabel=clabel, categorical=categorical, dpi=CONFIG.dpi
                )

            elif spec == "pc1pc2mag":
                plot_3d(
                    pcs[:, 0], pcs[:, 1], mag,
                    title=f"{policy.upper()}  {args.space.upper()}→PCA  ({sinks_mode}-sink)",
                    xlabel="PC1", ylabel="PC2", zlabel=mag_label,
                    outpath=outpath,
                    cvals=cvals, clabel=clabel, categorical=categorical, dpi=CONFIG.dpi
                )

            elif spec == "pc1pc3mag":
                if pcs.shape[1] < 3:
                    continue
                plot_3d(
                    pcs[:, 0], pcs[:, 2], mag,
                    title=f"{policy.upper()}  {args.space.upper()}→PCA  ({sinks_mode}-sink)",
                    xlabel="PC1", ylabel="PC3", zlabel=mag_label,
                    outpath=outpath,
                    cvals=cvals, clabel=clabel, categorical=categorical, dpi=CONFIG.dpi
                )

            elif spec == "pc2pc3mag":
                if pcs.shape[1] < 3:
                    continue
                plot_3d(
                    pcs[:, 1], pcs[:, 2], mag,
                    title=f"{policy.upper()}  {args.space.upper()}→PCA  ({sinks_mode}-sink)",
                    xlabel="PC2", ylabel="PC3", zlabel=mag_label,
                    outpath=outpath,
                    cvals=cvals, clabel=clabel, categorical=categorical, dpi=CONFIG.dpi
                )

            else:
                print(f"WARNING: unknown plot spec '{spec}' (skipping)")
                continue

            print(f"Wrote: {outpath}")

        # Print explained variance ratios for first few PCs
        ev = pca.explained_variance_ratio_
        head = ", ".join([f"PC{i+1}={ev[i]:.4f}" for i in range(min(5, len(ev)))])
        print(f"{policy.upper()} explained variance: {head}")

    print("Done.")


if __name__ == "__main__":
    main()
