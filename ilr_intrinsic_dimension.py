#!/usr/bin/env python3
"""
Estimate intrinsic/fractal dimension from compositional sink-coefficient data.

Per policy group (if present):
  1) Detect sink weight columns in a CSV (e.g., W_down_2 ... W_down_97).
  2) Drop zero-variance columns (e.g., all-zero sinks).
  3) Apply a pseudocount to remaining zeros, close rows to the simplex.
  4) Transform to ILR coordinates (Helmert basis; Euclidean).
  5) Estimate dimension via:
        - kNN-MLE (Levina–Bickel), multiple k values (local estimator)
        - Correlation dimension (Grassberger–Procaccia style) using random-pair sampling (global estimator)
        - Correlation dimension *matched to kNN scale*: for each k, fit corr-dim slope in a window centered
          around the median kNN radius (helps apples-to-apples comparisons).
  6) Write summaries + optional plots.

Usage (minimal CLI):
  python3 estimate_ilr_intrinsic_dimension_matched.py path/to/your.csv

Config is in CONFIG below.
"""

from __future__ import annotations

import re
import sys
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# --- Optional deps (script works without plotting) ---
try:
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False

try:
    from sklearn.neighbors import NearestNeighbors
    HAVE_SK = True
except Exception:
    HAVE_SK = False


# =========================
# CONFIG (edit as needed)
# =========================
CONFIG = {
    # Column detection:
    # Accept both "W_down_2" and "w_down_2" (and "W_2" if no policy segment).
    "WEIGHT_COL_REGEX": r"^[Ww]_(?:(?P<policy>[A-Za-z0-9]+)_)?(?P<sink>\d+)$",

    # If you prefer explicit columns, set a list here (otherwise auto-detect):
    "WEIGHT_COLS_EXPLICIT": None,  # e.g. ["W_down_2","W_down_3",...]

    # Zero handling: add pseudocount ONLY to zero entries (not to all entries).
    "PSEUDOCOUNT_EPS": 0.5,

    # Drop columns that are all zeros or have zero variance.
    "DROP_ZERO_VARIANCE_COLS": True,

    # Drop rows whose selected weight columns sum to 0 (after col filtering).
    "DROP_ZERO_SUM_ROWS": True,

    # kNN-MLE settings
    "K_LIST": [10, 15, 20, 25, 30, 40, 50],
    "KNN_TRIM_FRACTION": 0.02,  # trim tails of local estimates; set 0 to disable

    # Correlation dimension settings (random-pair sampling)
    "CORR_N_PAIRS": 1_500_000,
    "CORR_R_GRID_SIZE": 40,
    "CORR_R_MIN_Q": 0.02,   # avoid very small scales (zeros/pseudocount artifacts)
    "CORR_R_MAX_Q": 0.50,   # avoid very large scales (saturation)
    "CORR_MIN_COUNTS": 200,
    "CORR_WINDOW": 9,
    "CORR_MIN_R2": 0.995,

    # Correlation dimension matched to kNN scale:
    "CORR_MIN_LOG_SPAN": 0.35,  # minimum log(r) span for a candidate scaling window
    # For each k, build an r-grid in [r_k/alpha, r_k*alpha] and fit a slope window there.
    "CORR_MATCH_ALPHA": 2.0,       # widen/narrow the band around median kNN radius
    "CORR_MATCH_GRID_SIZE": 28,    # per-k grid size (smaller than global to keep it quick)
    "CORR_MATCH_WINDOW": 9,
    "CORR_MATCH_MIN_R2": 0.99,     # slightly looser; these windows can be small
    "CORR_MATCH_MIN_LOG_SPAN": 0.35,  # minimum log(r) span for matched-to-k scaling window
    "CORR_MATCH_MIN_R_MODE": "knn_quantile",  # "knn_quantile" (recommended) or "global_quantile"
    "CORR_MATCH_MIN_R_K": 10,               # k whose neighbor radii define the minimum trusted scale
    "CORR_MATCH_MIN_R_Q": 0.05,             # quantile of that kNN radius distribution used as floor
    "CORR_MATCH_MIN_COUNTS": 200,  # still require enough sampled pairs below r

    "RANDOM_SEED": 17,

    # Output
    "WRITE_ILR_NPY": True,
    "MAKE_PLOTS": True,  # auto-disables if matplotlib missing
}


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_log(x: np.ndarray, eps: float = 1e-300) -> np.ndarray:
    return np.log(np.maximum(x, eps))


def helmert_submatrix(D: int) -> np.ndarray:
    """
    Orthonormal (D-1) x D Helmert sub-matrix H.
    For row vectors: ilr = clr @ H.T
    """
    if D < 2:
        raise ValueError("D must be >= 2 for ILR.")
    H = np.zeros((D - 1, D), dtype=np.float64)
    for k in range(1, D):  # 1..D-1
        denom = math.sqrt(k * (k + 1))
        H[k - 1, :k] = 1.0 / denom
        H[k - 1, k] = -k / denom
    return H


def closure_rows(X: np.ndarray) -> np.ndarray:
    s = X.sum(axis=1, keepdims=True)
    s = np.where(s == 0, 1.0, s)
    return X / s


def apply_pseudocount_to_zeros(X: np.ndarray, eps: float) -> np.ndarray:
    if eps <= 0:
        raise ValueError("PSEUDOCOUNT_EPS must be > 0 when zeros exist.")
    X2 = X.astype(np.float64, copy=True)
    mask = (X2 == 0)
    if mask.any():
        X2[mask] = eps
    return X2


def ilr_transform_closed(X_closed: np.ndarray) -> np.ndarray:
    """
    X_closed: n x D composition (rows sum to 1, positive).
    Returns: n x (D-1) ILR coordinates.
    """
    logX = safe_log(X_closed)
    clr = logX - logX.mean(axis=1, keepdims=True)
    H = helmert_submatrix(X_closed.shape[1])
    return clr @ H.T


def detect_weight_columns(df: pd.DataFrame, regex: str, explicit: Optional[List[str]]):
    """
    Returns:
      groups: Dict[str, List[str]] mapping policy -> weight columns
              policy is 'ALL' if not present.
    """
    if explicit:
        cols = [c for c in explicit if c in df.columns]
        if not cols:
            raise ValueError("WEIGHT_COLS_EXPLICIT provided but none found in CSV.")
        return {"ALL": cols}

    pat = re.compile(regex)
    groups: Dict[str, List[Tuple[int, str]]] = {}
    for c in df.columns:
        m = pat.match(c)
        if not m:
            continue
        sink = int(m.group("sink"))
        policy = m.groupdict().get("policy") or "ALL"
        groups.setdefault(policy, []).append((sink, c))

    if not groups:
        raise ValueError("No weight columns detected. Update CONFIG['WEIGHT_COL_REGEX'] or WEIGHT_COLS_EXPLICIT.")

    out: Dict[str, List[str]] = {}
    for policy, pairs in groups.items():
        pairs_sorted = sorted(pairs, key=lambda t: t[0])
        out[policy] = [c for _, c in pairs_sorted]
    return out


def knn_mle_dimension(X: np.ndarray, k_list: List[int], trim_fraction: float = 0.0):
    """
    Levina–Bickel kNN-MLE dimension estimator.

    Returns:
      summary_df with columns: k, d_hat_mean, d_hat_median, d_hat_trimmed_mean, n_valid, median_r_k
      local_by_k: dict k -> local dimension array (n,)
      median_r_by_k: dict k -> median distance to k-th NN (across points)
      radii_by_k: dict k -> distance-to-kNN array Tk (n,)
    """
    if not HAVE_SK:
        raise RuntimeError("scikit-learn is required for kNN-MLE.")

    n = X.shape[0]
    max_k = max(k_list)
    nn = NearestNeighbors(n_neighbors=max_k + 1, algorithm="auto", metric="euclidean")
    nn.fit(X)
    dists, _ = nn.kneighbors(X, return_distance=True)
    dists = dists[:, 1:]  # drop self

    local_by_k = {}
    median_r_by_k = {}
    radii_by_k = {}
    rows = []

    for k in k_list:
        if k < 3:
            raise ValueError("k should be >= 3.")
        Tk = dists[:, k - 1]
        Tj = dists[:, : k - 1]
        valid = (Tk > 0) & (Tj.min(axis=1) > 0)

        di = np.full(n, np.nan, dtype=np.float64)
        if valid.any():
            ratio = Tk[valid][:, None] / Tj[valid]
            inv = np.mean(np.log(ratio), axis=1)
            good = inv > 0
            tmp = np.full(inv.shape[0], np.nan, dtype=np.float64)
            tmp[good] = 1.0 / inv[good]
            di[valid] = tmp

        local_by_k[k] = di
        vals = di[np.isfinite(di)]

        # robust "scale" associated with k
        Tk_pos = Tk[Tk > 0]
        median_r = float(np.median(Tk_pos)) if Tk_pos.size else float("nan")
        median_r_by_k[k] = median_r
        radii_by_k[k] = Tk.astype(np.float64, copy=True)

        if vals.size == 0:
            rows.append({"k": k, "d_hat_mean": np.nan, "d_hat_median": np.nan,
                         "d_hat_trimmed_mean": np.nan, "n_valid": 0,
                         "median_r_k": median_r})
            continue

        mean = float(np.mean(vals))
        med = float(np.median(vals))
        if trim_fraction and 0 < trim_fraction < 0.49:
            lo = np.quantile(vals, trim_fraction)
            hi = np.quantile(vals, 1 - trim_fraction)
            trimmed = vals[(vals >= lo) & (vals <= hi)]
            tmean = float(np.mean(trimmed)) if trimmed.size else np.nan
        else:
            tmean = mean

        rows.append({"k": k, "d_hat_mean": mean, "d_hat_median": med,
                     "d_hat_trimmed_mean": tmean, "n_valid": int(vals.size),
                     "median_r_k": median_r})

    return pd.DataFrame(rows), local_by_k, median_r_by_k, radii_by_k


def sample_pairwise_distances(X: np.ndarray, n_pairs: int, rng: np.random.Generator) -> np.ndarray:
    n = X.shape[0]
    i = rng.integers(0, n, size=n_pairs, endpoint=False)
    j = rng.integers(0, n, size=n_pairs, endpoint=False)
    same = (i == j)
    if same.any():
        j[same] = (j[same] + 1) % n
    diff = X[i] - X[j]
    return np.sqrt(np.sum(diff * diff, axis=1))


class CorrSampler:
    """
    Hold a sampled set of pairwise distances so we can compute C(r) cheaply for many r grids.
    """
    def __init__(self, X: np.ndarray, n_pairs: int, seed: int):
        rng = np.random.default_rng(seed)
        d = sample_pairwise_distances(X, n_pairs=n_pairs, rng=rng)
        self.d_sorted = np.sort(d.astype(np.float64))
        self.m = self.d_sorted.size

    def C_of_r(self, r_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        counts = np.searchsorted(self.d_sorted, r_grid, side="right")
        C = counts / float(self.m)
        return counts, C

    def quantile(self, q: float) -> float:
        q = float(q)
        if not (0.0 <= q <= 1.0):
            raise ValueError("quantile q must be in [0,1]")
        idx = int(round(q * (self.m - 1)))
        return float(self.d_sorted[idx])


def _fit_best_window(log_r: np.ndarray, log_C: np.ndarray, ok: np.ndarray, window: int, min_r2: float, min_log_span: float = 0.0):
    """
    Slide a fixed-length window and find a best linear fit slope in log-log space.
    Returns: slopes, r2s, best dict
    """
    grid_size = log_r.size
    slopes = np.full(grid_size, np.nan, dtype=np.float64)
    r2s = np.full(grid_size, np.nan, dtype=np.float64)

    if ok.sum() < window + 2:
        best = {"slope": np.nan, "r2": np.nan, "start_idx": None, "end_idx": None,
                "note": "Not enough usable r points in grid."}
        return slopes, r2s, best

    best_slope = np.nan
    best_r2 = -np.inf
    best_start = None

    for start in range(0, grid_size - window + 1):
        end = start + window
        if not ok[start:end].all():
            continue
        x = log_r[start:end]
        if float(min_log_span) > 0.0:
            span = float(x[-1] - x[0])
            if not np.isfinite(span) or span < float(min_log_span):
                continue
        y = log_C[start:end]
        x0 = x.mean(); y0 = y.mean()
        xx = np.sum((x - x0) ** 2)
        if xx <= 0:
            continue
        b = np.sum((x - x0) * (y - y0)) / xx
        a = y0 - b * x0
        y_hat = a + b * x
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y0) ** 2)
        r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else np.inf)

        center = start + window // 2
        slopes[center] = b
        r2s[center] = r2

        if r2 >= min_r2 and r2 > best_r2:
            best_r2 = r2
            best_slope = b
            best_start = start

    if best_start is None:
        # fallback to best available r2 (still useful)
        best_r2 = -np.inf
        for start in range(0, grid_size - window + 1):
            end = start + window
            if not ok[start:end].all():
                continue
            x = log_r[start:end]
            if float(min_log_span) > 0.0:
                span = float(x[-1] - x[0])
                if not np.isfinite(span) or span < float(min_log_span):
                    continue
            y = log_C[start:end]
            x0 = x.mean(); y0 = y.mean()
            xx = np.sum((x - x0) ** 2)
            if xx <= 0:
                continue
            b = np.sum((x - x0) * (y - y0)) / xx
            a = y0 - b * x0
            y_hat = a + b * x
            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - y0) ** 2)
            r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else np.inf)
            if r2 > best_r2:
                best_r2 = r2
                best_slope = b
                best_start = start
        note = "No window met min_r2; using best available window by R^2."
    else:
        note = "Best window met min_r2."

    best = {
        "slope": float(best_slope) if np.isfinite(best_slope) else np.nan,
        "r2": float(best_r2) if np.isfinite(best_r2) else np.nan,
        "start_idx": int(best_start) if best_start is not None else None,
        "end_idx": int(best_start + window) if best_start is not None else None,
        "note": note,
    }
    return slopes, r2s, best


def correlation_dimension_from_sampler(sampler: CorrSampler,
                                      r_lo: float,
                                      r_hi: float,
                                      grid_size: int,
                                      min_counts: int,
                                      window: int,
                                      min_r2: float,
                                      min_log_span: float = 0.0):
    """
    Correlation dimension on a specified [r_lo, r_hi] range, using an existing CorrSampler.

    Returns:
      df with r, C(r), log_r, log_C, ok, slope_local, r2_local
      best dict describing chosen scaling window and slope.
    """
    r_lo = float(max(r_lo, np.finfo(np.float64).tiny))
    r_hi = float(max(r_hi, r_lo * 1.01))
    r_grid = np.exp(np.linspace(np.log(r_lo), np.log(r_hi), int(grid_size)))

    counts, C = sampler.C_of_r(r_grid)
    ok = (counts >= int(min_counts)) & (C <= 0.95) & (C > 0)

    log_r = np.full_like(r_grid, np.nan, dtype=np.float64)
    log_C = np.full_like(C, np.nan, dtype=np.float64)
    log_r[ok] = np.log(r_grid[ok])
    log_C[ok] = np.log(C[ok])

    slopes, r2s, best = _fit_best_window(log_r, log_C, ok, window=int(window), min_r2=float(min_r2), min_log_span=float(min_log_span))

    df = pd.DataFrame({
        "r": r_grid,
        "count": counts,
        "C": C,
        "ok": ok,
        "log_r": log_r,
        "log_C": log_C,
        "slope_local": slopes,
        "r2_local": r2s,
    })

    if best.get("start_idx") is not None:
        s = best["start_idx"]; e = best["end_idx"]
        best.update({
            "r_start": float(r_grid[s]),
            "r_end": float(r_grid[e - 1]),
            "log_span": float(log_r[e - 1] - log_r[s]) if (np.isfinite(log_r[e - 1]) and np.isfinite(log_r[s])) else np.nan,
        })
    else:
        best.update({"r_start": None, "r_end": None, "log_span": None})

    best.update({
        "grid_size": int(grid_size),
        "window": int(window),
        "min_r2": float(min_r2),
        "min_log_span": float(min_log_span),
        "min_counts": int(min_counts),
    })
    return df, best


def correlation_dimension_global(X: np.ndarray,
                                n_pairs: int,
                                grid_size: int,
                                r_min_q: float,
                                r_max_q: float,
                                min_counts: int,
                                window: int,
                                min_r2: float,
                                min_log_span: float,
                                seed: int):
    """
    Backwards-compatible "global" correlation dimension (quantile-ranged).
    """
    sampler = CorrSampler(X, n_pairs=int(n_pairs), seed=int(seed))
    r_lo = max(sampler.quantile(r_min_q), np.finfo(np.float64).tiny)
    r_hi = max(sampler.quantile(r_max_q), r_lo * 10)
    df, best = correlation_dimension_from_sampler(
        sampler, r_lo=r_lo, r_hi=r_hi, grid_size=int(grid_size),
        min_counts=int(min_counts), window=int(window), min_r2=float(min_r2), min_log_span=float(min_log_span),
    )
    best.update({
        "pairs": int(n_pairs),
        "r_min_q": float(r_min_q),
        "r_max_q": float(r_max_q),
        "note_global_range": "Range chosen by distance quantiles; see r_min_q/r_max_q.",
    })
    return df, best, sampler


def maybe_plot_corr(df: pd.DataFrame, out_png: Path, title: str, best: dict):
    if not HAVE_PLT or not CONFIG.get("MAKE_PLOTS", True):
        return
    ok = df["ok"].values.astype(bool)
    x = df.loc[ok, "log_r"].values
    y = df.loc[ok, "log_C"].values

    plt.figure()
    plt.scatter(x, y, s=8)
    plt.xlabel("log r")
    plt.ylabel("log C(r)")
    plt.title(title)

    if best.get("start_idx") is not None:
        s = best["start_idx"]; e = best["end_idx"]
        if df["ok"].iloc[s:e].all():
            xw = df["log_r"].iloc[s:e].values
            yw = df["log_C"].iloc[s:e].values
            b = float(best["slope"])
            a = yw.mean() - b * xw.mean()
            xs = np.array([xw.min(), xw.max()])
            ys = a + b * xs
            plt.plot(xs, ys, linewidth=2)
            plt.text(xs.mean(), ys.mean(),
                     f"slope≈{best['slope']:.3f}\nR²≈{best['r2']:.4f}",
                     ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def maybe_plot_matched(knn_summary: pd.DataFrame, matched_df: pd.DataFrame, out_png: Path, title: str):
    if not HAVE_PLT or not CONFIG.get("MAKE_PLOTS", True):
        return
    # k vs dimension
    plt.figure()
    x = knn_summary["k"].values
    y_knn = knn_summary["d_hat_trimmed_mean"].values
    plt.plot(x, y_knn, marker="o", linestyle="-", label="kNN–MLE (trimmed mean)")

    if matched_df is not None and len(matched_df):
        y_cd = matched_df["corr_slope"].values
        plt.plot(x, y_cd, marker="o", linestyle="-", label="Corr-dim matched to median r_k")

    plt.xlabel("k (nearest neighbors)")
    plt.ylabel("dimension estimate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 estimate_ilr_intrinsic_dimension_matched.py path/to/data.csv")
        sys.exit(2)

    csv_path = Path(sys.argv[1]).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    outdir = Path(str(csv_path.with_suffix("")) + "_dim")
    ensure_dir(outdir)

    (outdir / "config_used.json").write_text(json.dumps(CONFIG, indent=2))

    print(f"[{_now()}] Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[{_now()}] Rows={len(df):,}  Cols={len(df.columns):,}")

    groups = detect_weight_columns(df, CONFIG["WEIGHT_COL_REGEX"], CONFIG["WEIGHT_COLS_EXPLICIT"])
    print(f"[{_now()}] Detected policy groups: {list(groups.keys())}")

    for policy, cols in groups.items():
        print(f"\n[{_now()}] === Policy group: {policy} ===")
        sub = df[cols].copy()

        # Coerce to numeric; drop NaNs
        for c in cols:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
        n0 = len(sub)
        sub = sub.dropna(axis=0, how="any")
        if len(sub) != n0:
            print(f"[{_now()}] Dropped {n0 - len(sub):,} rows with NaNs in weight columns.")

        # Drop zero-variance columns (including all-zero)
        if CONFIG["DROP_ZERO_VARIANCE_COLS"]:
            var = sub.var(axis=0, ddof=0)
            keep_cols = var[var > 0].index.tolist()
            dropped = [c for c in cols if c not in keep_cols]
            if dropped:
                print(f"[{_now()}] Dropping {len(dropped)} zero-variance columns: {dropped[:10]}{'...' if len(dropped) > 10 else ''}")
            sub = sub[keep_cols]

        if sub.shape[1] < 2:
            print(f"[{_now()}] Not enough non-degenerate sinks after filtering (D={sub.shape[1]}). Skipping.")
            continue

        X = sub.to_numpy(dtype=np.float64)

        # Drop zero-sum rows
        if CONFIG["DROP_ZERO_SUM_ROWS"]:
            rs = X.sum(axis=1)
            keep = rs > 0
            if not np.all(keep):
                print(f"[{_now()}] Dropping {(~keep).sum():,} rows with zero sum across selected sinks.")
            X = X[keep]

        X = apply_pseudocount_to_zeros(X, eps=float(CONFIG["PSEUDOCOUNT_EPS"]))
        X_closed = closure_rows(X)
        ilr = ilr_transform_closed(X_closed)

        print(f"[{_now()}] ILR shape: {ilr.shape[0]:,} x {ilr.shape[1]} (retained sinks D={X_closed.shape[1]})")
        if CONFIG["WRITE_ILR_NPY"]:
            np.save(outdir / f"ilr_{policy}.npy", ilr)

        # --- kNN-MLE (and median kNN radii) ---
        knn_summary = None
        median_r_by_k = None
        if HAVE_SK:
            knn_summary, local_by_k, median_r_by_k, radii_by_k = knn_mle_dimension(
                ilr,
                k_list=list(CONFIG["K_LIST"]),
                trim_fraction=float(CONFIG["KNN_TRIM_FRACTION"]),
            )
            knn_summary.to_csv(outdir / f"knn_mle_summary_{policy}.csv", index=False)
            print(knn_summary)

            local_df = pd.DataFrame({f"d_hat_k{k}": local_by_k[k] for k in CONFIG["K_LIST"]})
            local_df.to_csv(outdir / f"knn_mle_local_{policy}.csv", index=False)
        else:
            print(f"[{_now()}] WARNING: scikit-learn not available; skipping kNN-MLE (and corr-dim matching).")

        # --- Correlation dimension (global) ---
        corr_df, best, sampler = correlation_dimension_global(
            ilr,
            n_pairs=int(CONFIG["CORR_N_PAIRS"]),
            grid_size=int(CONFIG["CORR_R_GRID_SIZE"]),
            r_min_q=float(CONFIG["CORR_R_MIN_Q"]),
            r_max_q=float(CONFIG["CORR_R_MAX_Q"]),
            min_counts=int(CONFIG["CORR_MIN_COUNTS"]),
            window=int(CONFIG["CORR_WINDOW"]),
            min_r2=float(CONFIG["CORR_MIN_R2"]),
            min_log_span=float(CONFIG.get("CORR_MIN_LOG_SPAN", 0.0)),
            seed=int(CONFIG["RANDOM_SEED"]) + 1,
        )
        corr_df.to_csv(outdir / f"corr_dimension_{policy}.csv", index=False)
        (outdir / f"corr_dimension_best_{policy}.json").write_text(json.dumps(best, indent=2))
        print(f"[{_now()}] Best corr-dim (global) slope≈{best.get('slope')}  R²≈{best.get('r2')}  note={best.get('note')}")

        if HAVE_PLT and CONFIG.get("MAKE_PLOTS", True):
            maybe_plot_corr(corr_df, outdir / f"corr_dimension_{policy}.png",
                            title=f"Correlation dimension (policy={policy}) [global]",
                            best=best)

        # --- Correlation dimension matched to kNN scale ---
        matched_df = None
        if HAVE_SK and median_r_by_k:
            alpha = float(CONFIG["CORR_MATCH_ALPHA"])

            # Minimum trusted scale for "matched" corr-dim fits.
            # Using a GLOBAL pairwise-distance quantile can be far too conservative for spread-out clouds
            # (it can exceed the kNN radii), causing matched ranges to collapse. Instead, we default to a
            # POLICY-SPECIFIC floor derived from the distribution of kNN radii.
            match_mode = str(CONFIG.get("CORR_MATCH_MIN_R_MODE", "knn_quantile")).strip().lower()
            if match_mode not in ("knn_quantile", "global_quantile"):
                match_mode = "knn_quantile"

            min_r_guard = None
            if match_mode == "global_quantile":
                min_r_guard = float(sampler.quantile(float(CONFIG["CORR_R_MIN_Q"])))
            else:
                k0 = int(CONFIG.get("CORR_MATCH_MIN_R_K", min(CONFIG["K_LIST"])))
                q0 = float(CONFIG.get("CORR_MATCH_MIN_R_Q", 0.05))
                Tk0 = radii_by_k.get(k0)
                if Tk0 is None:
                    # fall back: use the smallest available k
                    k0 = int(min(radii_by_k.keys())) if radii_by_k else int(min(CONFIG["K_LIST"]))
                    Tk0 = radii_by_k.get(k0)
                if Tk0 is None:
                    # final fall back
                    min_r_guard = float(sampler.quantile(float(CONFIG["CORR_R_MIN_Q"])))
                else:
                    Tk_pos = Tk0[np.isfinite(Tk0) & (Tk0 > 0)]
                    if Tk_pos.size < 50:
                        min_r_guard = float(sampler.quantile(float(CONFIG["CORR_R_MIN_Q"])))
                    else:
                        min_r_guard = float(np.quantile(Tk_pos, min(max(q0, 0.0), 1.0)))

            min_r_guard = max(float(min_r_guard), np.finfo(np.float64).tiny)

            # For the *upper* guard, we still prefer a large pairwise-distance quantile to avoid saturation.
            max_r_guard = float(sampler.quantile(min(0.98, max(float(CONFIG["CORR_R_MAX_Q"]), 0.80))))

            rows = []
            for k in CONFIG["K_LIST"]:
                r_k = float(median_r_by_k.get(k, float("nan")))
                if not np.isfinite(r_k) or r_k <= 0:
                    rows.append({"k": k, "median_r_k": r_k, "corr_slope": np.nan, "corr_r2": np.nan,
                                 "r_lo": None, "r_hi": None, "r_start": None, "r_end": None, "note": "No valid median_r_k"})
                    continue

                r_lo = max(r_k / alpha, min_r_guard)
                r_hi = min(r_k * alpha, max_r_guard)
                if r_hi <= r_lo * 1.01:
                    rows.append({"k": k, "median_r_k": r_k, "corr_slope": np.nan, "corr_r2": np.nan,
                                 "r_lo": r_lo, "r_hi": r_hi, "r_start": None, "r_end": None,
                                 "note": "Matched range collapsed after guards; adjust CORR_MATCH_ALPHA or guards"})
                    continue

                df_k, best_k = correlation_dimension_from_sampler(
                    sampler,
                    r_lo=r_lo,
                    r_hi=r_hi,
                    grid_size=int(CONFIG["CORR_MATCH_GRID_SIZE"]),
                    min_counts=int(CONFIG["CORR_MATCH_MIN_COUNTS"]),
                    window=int(CONFIG["CORR_MATCH_WINDOW"]),
                    min_r2=float(CONFIG["CORR_MATCH_MIN_R2"]),
                    min_log_span=float(CONFIG.get("CORR_MATCH_MIN_LOG_SPAN", 0.0)),
                )
                # store per-k full df optionally (can be large); keep concise by default
                out_k_csv = outdir / f"corr_dimension_matched_policy_{policy}_k{k}.csv"
                df_k.to_csv(out_k_csv, index=False)

                rows.append({
                    "k": int(k),
                    "median_r_k": r_k,
                    "r_lo": r_lo,
                    "r_hi": r_hi,
                    "corr_slope": best_k.get("slope"),
                    "corr_r2": best_k.get("r2"),
                    "r_start": best_k.get("r_start"),
                    "r_end": best_k.get("r_end"),
                    "note": best_k.get("note"),
                    "grid_size": best_k.get("grid_size"),
                    "window": best_k.get("window"),
                })

            matched_df = pd.DataFrame(rows)
            matched_df.to_csv(outdir / f"corr_dimension_matched_{policy}.csv", index=False)
            (outdir / f"corr_dimension_matched_meta_{policy}.json").write_text(json.dumps({
                "alpha": alpha,
                "min_r_guard": float(min_r_guard),
                "max_r_guard": float(max_r_guard),
                "pairs": int(CONFIG["CORR_N_PAIRS"]),
                "seed": int(CONFIG["RANDOM_SEED"]) + 1,
            }, indent=2))
            print(f"[{_now()}] Wrote corr-dim matched-to-kNN summaries: corr_dimension_matched_{policy}.csv")

            if HAVE_PLT and CONFIG.get("MAKE_PLOTS", True) and knn_summary is not None:
                maybe_plot_matched(
                    knn_summary,
                    matched_df,
                    outdir / f"dim_compare_knn_vs_corr_matched_{policy}.png",
                    title=f"Dimension estimates (policy={policy})",
                )

    print(f"\n[{_now()}] Done. Outputs in: {outdir}")


if __name__ == "__main__":
    main()
