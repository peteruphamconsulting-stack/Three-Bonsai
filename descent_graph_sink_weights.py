#!/usr/bin/env python3
"""
iterated_decomp_embeddings.py

One unified script to generate iterated Goldbach/Lemoine sink-coefficient embeddings
for either the 10-sink or 12-sink system, with:

- CLI toggle: --sinks 10 | 12
- Policies: down / up / quarter / center (with --center fraction)
- Output:
  (a) sink-weight coefficients per policy
  (b) depth-1 and depth-2 witness tracing per policy
  (c) depth-wise branching / mass-flow summaries per policy

Notes:
- Odd nodes: Lemoine M = p + 2q with q in [ceil(M/5), floor(M/3)]
- Even nodes: Goldbach N = a + b, scanning a downward from floor(N/2)
- Recursion continues until reaching sinks (terminals).
- We track branching stats while traversing the tree (not just using cached weights).

Example:
  # 12-sink wide-range sample of 10k odds, include evens too, center at 2/7
  python3 iterated_decomp_embeddings.py --sinks 12 --start 5000001 --end 800000000 \
      --sample_n 10000 --include_even 1 --policies down,up,center --center 2/7 \
      --out_csv out_12sink.csv

  # 10-sink consecutive band, odds only, all policies with center at 0.27
  python3 iterated_decomp_embeddings.py --sinks 10 --start 5000001 --end 5050000 \
      --sample_n 0 --include_even 0 --policies down,up,quarter,center --center 0.27 \
      --out_csv out_10sink_band.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import multiprocessing
import random
import time
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

import numpy as np

# ------------------------------------------------------------
# Sink sets (10 vs 12)
# ------------------------------------------------------------

SINKS_10: Tuple[int, ...] = (2, 3, 5, 19, 29, 37, 47, 59, 73, 97)
SINKS_12: Tuple[int, ...] = (11, 13, 17, 19, 23, 29, 31, 37, 47, 59, 73, 97)

# ------------------------------------------------------------
# Prime utilities
# ------------------------------------------------------------

SMALL_PRIMES: Tuple[int, ...] = (
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67,
    71, 73, 79, 83, 89, 97
)

# Primality memoization (massive speed-up for repeated tests)
_PRIME_CACHE: Dict[int, bool] = {}


# Deterministic MR bases that are plenty for your ranges (>= 1e9).
# If you push far beyond ~3e14, upgrade bases accordingly.
MR_BASES: Tuple[int, ...] = (2, 3, 5, 7, 11)

def _powmod(a: int, d: int, n: int) -> int:
    return pow(a, d, n)

def is_prime(n: int) -> bool:
    v = _PRIME_CACHE.get(n)
    if v is not None:
        return v
    if n < 2:
        _PRIME_CACHE[n] = False
        return False
    for p in SMALL_PRIMES:
        if n == p:
            _PRIME_CACHE[n] = True
            return True
        if n % p == 0:
            _PRIME_CACHE[n] = False
            return False

    # Miller-Rabin
    d = n - 1
    s = 0
    while (d & 1) == 0:
        s += 1
        d >>= 1

    for a in MR_BASES:
        if a % n == 0:
            continue
        x = _powmod(a, d, n)
        if x == 1 or x == n - 1:
            continue
        composite = True
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                composite = False
                break
        if composite:
            _PRIME_CACHE[n] = False
            return False
    _PRIME_CACHE[n] = True
    return True

def prev_prime(n: int) -> int:
    """Largest prime <= n, or -1 if none."""
    if n < 2:
        return -1
    if n == 2:
        return 2
    if (n & 1) == 0:
        n -= 1
    while n >= 2:
        if is_prime(n):
            return n
        n -= 2
    return -1

def next_prime(n: int) -> int:
    """Smallest prime >= n, or -1 if none in int range."""
    if n <= 2:
        return 2
    if (n & 1) == 0:
        n += 1
    while True:
        if is_prime(n):
            return n
        n += 2

# ------------------------------------------------------------
# Decomposition rules
# ------------------------------------------------------------

def lemoine_bounds(M: int) -> Tuple[int, int]:
    """q bounds for M = p + 2q with q prime."""
    lo = (M + 4) // 5  # ceil(M/5)
    hi = M // 3        # floor(M/3)
    return lo, hi

# Cache the chosen witness for each odd M and each policy
_LEMOINE_CACHE: Dict[tuple, Optional[Tuple[int, int]]] = {}
# Cache Goldbach witness for even N
_GOLDBACH_CACHE: Dict[object, Optional[Tuple[int, int]]] = {}

# Global to hold the center fraction value for the "center" policy
_CENTER_FRACTION: Optional[float] = None

def set_center_fraction(value: float) -> None:
    """Set the global center fraction for the 'center' policy."""
    global _CENTER_FRACTION
    _CENTER_FRACTION = value

# Global(s) for the 'alpha' / 'alpha_rand' policies
_CENTER_ALPHA: Optional[float] = None   # in [0,1], position within [ceil(M/5), floor(M/3)]
_ALPHA_SEED: int = 0                    # used for per-integer deterministic randomness
_ALPHA_JITTER: float = 0.0              # if >0, alpha_rand draws within [alpha±jitter] *per integer*
_ROOT_ALPHA_CACHE: Dict[Tuple[int, int, int, int], float] = {}  # (seed, rootN, alpha_scaled, jitter_scaled)->alpha_root

# Global seed for the 'random' policy (uniform-random witness selection)
_RANDOM_WITNESS_SEED: int = 42

def set_random_witness_seed(seed: int) -> None:
    """Set RNG seed used by 'random' policy for reproducible uniform witness selection."""
    global _RANDOM_WITNESS_SEED
    _RANDOM_WITNESS_SEED = int(seed)

def set_center_alpha(alpha: float) -> None:
    """Set the global alpha in [0,1] for the 'alpha' / 'alpha_rand' policies."""
    global _CENTER_ALPHA
    _CENTER_ALPHA = alpha

def set_alpha_seed(seed: int) -> None:
    """Set RNG seed used by 'alpha_rand' to choose a per-integer start point."""
    global _ALPHA_SEED
    _ALPHA_SEED = int(seed)

def set_alpha_jitter(jitter: float) -> None:
    """Set jitter in [0,1] used by 'alpha_rand' around the base alpha."""
    global _ALPHA_JITTER
    _ALPHA_JITTER = float(jitter)

def _alpha_for_root(rootN: int) -> float:
    """Deterministic per-integer alpha for the alpha_rand policy.

    Key requirement: randomness is tied to the *root integer* (the sampled N), not to each
    intermediate node M in the decomposition. This ensures every node in a given integer's
    descent uses the same relative start position within its [lo,hi] q-window.
    """
    base_scaled = int(round((float(_CENTER_ALPHA) if _CENTER_ALPHA is not None else 0.0) * 1_000_000))
    jit_scaled = int(round(float(_ALPHA_JITTER or 0.0) * 1_000_000))
    key = (int(_ALPHA_SEED), int(rootN), base_scaled, jit_scaled)
    if key in _ROOT_ALPHA_CACHE:
        return _ROOT_ALPHA_CACHE[key]

    rng = random.Random((int(_ALPHA_SEED) << 1) ^ int(rootN))
    u = rng.random()  # in [0,1)

    if _CENTER_ALPHA is None:
        alpha_root = u
    else:
        base = float(_CENTER_ALPHA)
        if _ALPHA_JITTER and _ALPHA_JITTER > 0:
            alpha_root = max(0.0, min(1.0, base + (2.0 * u - 1.0) * float(_ALPHA_JITTER)))
        else:
            alpha_root = base

    _ROOT_ALPHA_CACHE[key] = alpha_root
    return alpha_root


def _lemoine_cache_key(policy: str, M: int, rootN: Optional[int] = None):
    """Cache key for Lemoine witness lookup.

    Deterministic policies (down/up/quarter/center/alpha) produce the same
    witness for a given M regardless of root, so the key is just (policy, M).
    Per-root policies (random/alpha_rand) include rootN.
    """
    if policy in ("alpha_rand",):
        a = int(round((_CENTER_ALPHA or 0.0) * 1_000_000))
        j = int(round(float(_ALPHA_JITTER or 0.0) * 1_000_000))
        return (policy, M, a, j, int(_ALPHA_SEED), int(rootN or 0))
    if policy == "random":
        return (policy, M, int(_RANDOM_WITNESS_SEED), int(rootN or 0))
    if policy == "alpha":
        a = int(round((_CENTER_ALPHA or 0.0) * 1_000_000))
        return (policy, M, a)
    # deterministic: down, up, quarter, center
    return (policy, M)


def find_lemoine_pair(M: int, policy: str, rootN: Optional[int] = None) -> Optional[Tuple[int, int]]:
    """
    Return (p, q) such that M = p + 2q with primes p,q and q in [ceil(M/5), floor(M/3)].
    policy in {"down","up","quarter","center","alpha","alpha_rand"}.
    """
    key = _lemoine_cache_key(policy, M, rootN)
    if key in _LEMOINE_CACHE:
        return _LEMOINE_CACHE[key]

    if M < 7 or (M & 1) == 0:
        _LEMOINE_CACHE[key] = None
        return None

    lo, hi = lemoine_bounds(M)
    if hi < lo:
        _LEMOINE_CACHE[key] = None
        return None

    def ok(q: int) -> bool:
        p = M - 2 * q
        return (p >= 2 and is_prime(p))

    if policy == "down":
        q = prev_prime(hi)
        while q != -1 and q >= lo:
            if ok(q):
                _LEMOINE_CACHE[key] = (M - 2 * q, q)
                return _LEMOINE_CACHE[key]
            q = prev_prime(q - 1)
        _LEMOINE_CACHE[key] = None
        return None

    if policy == "up":
        q = next_prime(lo)
        while q != -1 and q <= hi:
            if ok(q):
                _LEMOINE_CACHE[key] = (M - 2 * q, q)
                return _LEMOINE_CACHE[key]
            q = next_prime(q + 1)
        _LEMOINE_CACHE[key] = None
        return None

    if policy == "quarter":
        # Center at floor(M/4) and expand outward over primes in-range.
        center = M // 4
        qd = prev_prime(min(center, hi))
        qu = next_prime(max(center, lo))
        seen = set()

        # Expand outward until we exhaust both directions.
        while True:
            progressed = False

            # push qd down until it's an unseen prime within [lo, hi] (or exhausted)
            while qd != -1 and (qd < lo or qd > hi or qd in seen):
                qd = prev_prime(qd - 1)

            # push qu up until it's an unseen prime within [lo, hi] (or exhausted)
            while qu != -1 and (qu < lo or qu > hi or qu in seen):
                # if we've already stepped above hi, stop the upward search
                if qu > hi:
                    qu = -1
                    break
                qu = next_prime(qu + 1)

            if qd != -1 and lo <= qd <= hi and qd not in seen:
                progressed = True
                seen.add(qd)
                if ok(qd):
                    _LEMOINE_CACHE[key] = (M - 2 * qd, qd)
                    return _LEMOINE_CACHE[key]
                qd = prev_prime(qd - 1)

            if qu != -1 and lo <= qu <= hi and qu not in seen:
                progressed = True
                seen.add(qu)
                if ok(qu):
                    _LEMOINE_CACHE[key] = (M - 2 * qu, qu)
                    return _LEMOINE_CACHE[key]
                qu = next_prime(qu + 1)

            if not progressed:
                break

        _LEMOINE_CACHE[key] = None
        return None

    if policy == "center":
        # Center at floor(M * _CENTER_FRACTION) and expand outward over primes in-range.
        if _CENTER_FRACTION is None:
            raise ValueError("center policy requires --center fraction to be set")
        
        center = int(M * _CENTER_FRACTION)
        qd = prev_prime(min(center, hi))
        qu = next_prime(max(center, lo))
        seen = set()

        # Expand outward until we exhaust both directions.
        while True:
            progressed = False

            # push qd down until it's an unseen prime within [lo, hi] (or exhausted)
            while qd != -1 and (qd < lo or qd > hi or qd in seen):
                qd = prev_prime(qd - 1)

            # push qu up until it's an unseen prime within [lo, hi] (or exhausted)
            while qu != -1 and (qu < lo or qu > hi or qu in seen):
                # if we've already stepped above hi, stop the upward search
                if qu > hi:
                    qu = -1
                    break
                qu = next_prime(qu + 1)

            if qd != -1 and lo <= qd <= hi and qd not in seen:
                progressed = True
                seen.add(qd)
                if ok(qd):
                    _LEMOINE_CACHE[key] = (M - 2 * qd, qd)
                    return _LEMOINE_CACHE[key]
                qd = prev_prime(qd - 1)

            if qu != -1 and lo <= qu <= hi and qu not in seen:
                progressed = True
                seen.add(qu)
                if ok(qu):
                    _LEMOINE_CACHE[key] = (M - 2 * qu, qu)
                    return _LEMOINE_CACHE[key]
                qu = next_prime(qu + 1)

            if not progressed:
                break

        _LEMOINE_CACHE[key] = None
        return None


    if policy == "alpha":
        # Center within the *q-window* [lo, hi] using alpha in [0,1], then expand outward.
        if _CENTER_ALPHA is None:
            raise ValueError("alpha policy requires --alpha to be set")
        alpha = float(_CENTER_ALPHA)
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError(f"alpha must be in [0,1], got {alpha}")
        center = int(round(lo + alpha * (hi - lo)))
        qd = prev_prime(min(center, hi))
        qu = next_prime(max(center, lo))
        seen = set()

        while True:
            progressed = False

            while qd != -1 and (qd < lo or qd > hi or qd in seen):
                qd = prev_prime(qd - 1)

            while qu != -1 and (qu < lo or qu > hi or qu in seen):
                if qu > hi:
                    qu = -1
                    break
                qu = next_prime(qu + 1)

            if qd != -1 and lo <= qd <= hi and qd not in seen:
                progressed = True
                seen.add(qd)
                if ok(qd):
                    _LEMOINE_CACHE[key] = (M - 2 * qd, qd)
                    return _LEMOINE_CACHE[key]
                qd = prev_prime(qd - 1)

            if qu != -1 and lo <= qu <= hi and qu not in seen:
                progressed = True
                seen.add(qu)
                if ok(qu):
                    _LEMOINE_CACHE[key] = (M - 2 * qu, qu)
                    return _LEMOINE_CACHE[key]
                qu = next_prime(qu + 1)

            if not progressed:
                break

        _LEMOINE_CACHE[key] = None
        return None

    if policy == "alpha_rand":
        # Choose a per-*integer* center inside [lo, hi] (deterministic under --alpha_seed),
        # optionally jittered around the base --alpha, then expand outward.
        if rootN is None:
            raise ValueError("alpha_rand policy requires rootN (the original sampled integer) to be provided")
        alpha_root = _alpha_for_root(int(rootN))
        center = int(round(lo + alpha_root * (hi - lo)))
        qd = prev_prime(min(center, hi))
        qu = next_prime(max(center, lo))
        seen = set()

        while True:
            progressed = False

            while qd != -1 and (qd < lo or qd > hi or qd in seen):
                qd = prev_prime(qd - 1)

            while qu != -1 and (qu < lo or qu > hi or qu in seen):
                if qu > hi:
                    qu = -1
                    break
                qu = next_prime(qu + 1)

            if qd != -1 and lo <= qd <= hi and qd not in seen:
                progressed = True
                seen.add(qd)
                if ok(qd):
                    _LEMOINE_CACHE[key] = (M - 2 * qd, qd)
                    return _LEMOINE_CACHE[key]
                qd = prev_prime(qd - 1)

            if qu != -1 and lo <= qu <= hi and qu not in seen:
                progressed = True
                seen.add(qu)
                if ok(qu):
                    _LEMOINE_CACHE[key] = (M - 2 * qu, qu)
                    return _LEMOINE_CACHE[key]
                qu = next_prime(qu + 1)

            if not progressed:
                break

        _LEMOINE_CACHE[key] = None
        return None

    if policy == "random":
        # Pick a random starting point in [lo, hi] and expand outward —
        # same O(few) cost as center/quarter, but with a random anchor.
        # RNG seeded deterministically on (M, rootN, global_seed).
        rng = random.Random(
            (int(_RANDOM_WITNESS_SEED) << 32) ^ (int(rootN or 0) << 16) ^ int(M)
        )
        center = rng.randint(lo, hi)
        qd = prev_prime(min(center, hi))
        qu = next_prime(max(center, lo))
        seen = set()

        while True:
            progressed = False

            while qd != -1 and (qd < lo or qd > hi or qd in seen):
                qd = prev_prime(qd - 1)

            while qu != -1 and (qu < lo or qu > hi or qu in seen):
                if qu > hi:
                    qu = -1
                    break
                qu = next_prime(qu + 1)

            if qd != -1 and lo <= qd <= hi and qd not in seen:
                progressed = True
                seen.add(qd)
                if ok(qd):
                    _LEMOINE_CACHE[key] = (M - 2 * qd, qd)
                    return _LEMOINE_CACHE[key]
                qd = prev_prime(qd - 1)

            if qu != -1 and lo <= qu <= hi and qu not in seen:
                progressed = True
                seen.add(qu)
                if ok(qu):
                    _LEMOINE_CACHE[key] = (M - 2 * qu, qu)
                    return _LEMOINE_CACHE[key]
                qu = next_prime(qu + 1)

            if not progressed:
                break

        _LEMOINE_CACHE[key] = None
        return None

    raise ValueError(f"Unknown policy: {policy}")

def find_goldbach_pair(N: int, policy: str = "", rootN: Optional[int] = None) -> Optional[Tuple[int, int]]:
    """
    Return primes (a,b) such that N=a+b by scanning a downward from floor(N/2).

    All policies—including 'random'—use the same deterministic N/2-down scan
    for even integers.  This ensures that the first decomposition step for even
    roots is identical across policies; the 'random' policy randomises only the
    Lemoine witness selection at odd nodes further down the tree.  Keeping the
    Goldbach step uniform eliminates a confound when attributing serial-dependence
    differences to the Lemoine selection rule alone.

    Cache key is just N (policy-independent).
    """
    cache_key = N
    if cache_key in _GOLDBACH_CACHE:
        return _GOLDBACH_CACHE[cache_key]
    if N < 4 or (N & 1) == 1:
        _GOLDBACH_CACHE[cache_key] = None
        return None

    # Deterministic scan downward from N/2 (all policies)
    a = N // 2
    if (a & 1) == 0:
        a -= 1
    while a >= 2:
        if is_prime(a):
            b = N - a
            if is_prime(b):
                _GOLDBACH_CACHE[cache_key] = (a, b)
                return _GOLDBACH_CACHE[cache_key]
        a -= 2

    _GOLDBACH_CACHE[cache_key] = None
    return None

# ------------------------------------------------------------
# Depth-wise branching stats + witness trace
# ------------------------------------------------------------

TRACE_FIELDS = [
    # root witness
    "root_kind",          # "L" or "G" or "S" or ""
    "root_a", "root_b",   # L: a=p, b=q ; G: a=a, b=b ; S: a=sink, b=""
    "root_lo", "root_hi", # Lemoine q-bounds for root (else "")
    "root_gap_hi", "root_gap_lo",

    # depth-2 witness for child A = root_a
    "A_kind", "A_a", "A_b", "A_lo", "A_hi", "A_gap_hi", "A_gap_lo",
    # depth-2 witness for child B = root_b
    "B_kind", "B_a", "B_b", "B_lo", "B_hi", "B_gap_hi", "B_gap_lo",
]

@dataclass
class BranchStats:
    # These are variable-length (grow as needed during recursion)
    nodes: List[int]
    internal: List[int]
    sinks: List[int]
    mass: List[float]

    def _ensure(self, depth: int) -> None:
        while len(self.nodes) <= depth:
            self.nodes.append(0)
            self.internal.append(0)
            self.sinks.append(0)
            self.mass.append(0.0)

    def record_node(self, depth: int, coeff: float, is_sink: bool) -> None:
        self._ensure(depth)
        self.nodes[depth] += 1
        self.mass[depth] += float(coeff)
        if is_sink:
            self.sinks[depth] += 1
        else:
            self.internal[depth] += 1

    def max_depth(self) -> int:
        return max(0, len(self.nodes) - 1)

    def totals(self) -> Dict[str, float]:
        return {
            "total_nodes": int(sum(self.nodes)),
            "total_internal": int(sum(self.internal)),
            "total_sinks": int(sum(self.sinks)),
            "total_mass": float(sum(self.mass)),
        }

def trace_depth2_witnesses(
    N: int,
    policy: str,
    include_even: bool,
    sink_set: set
) -> Dict[str, object]:
    out: Dict[str, object] = {k: "" for k in TRACE_FIELDS}

    def trace_lemoine(M: int, prefix: str) -> None:
        if M in sink_set:
            out[f"{prefix}_kind"] = "S"
            out[f"{prefix}_a"] = M
            return
        if M < 7 or (M & 1) == 0:
            return
        pair = find_lemoine_pair(M, policy, N)
        if pair is None:
            return
        p, q = pair
        lo, hi = lemoine_bounds(M)
        out[f"{prefix}_kind"] = "L"
        out[f"{prefix}_a"] = p
        out[f"{prefix}_b"] = q
        out[f"{prefix}_lo"] = lo
        out[f"{prefix}_hi"] = hi
        out[f"{prefix}_gap_hi"] = q - hi
        out[f"{prefix}_gap_lo"] = q - lo

    # Root
    if N in sink_set:
        out["root_kind"] = "S"
        out["root_a"] = N
        return out

    if (N & 1) == 1:
        pair = find_lemoine_pair(N, policy, N)
        if pair is None:
            return out
        p, q = pair
        lo, hi = lemoine_bounds(N)
        out["root_kind"] = "L"
        out["root_a"] = p
        out["root_b"] = q
        out["root_lo"] = lo
        out["root_hi"] = hi
        out["root_gap_hi"] = q - hi
        out["root_gap_lo"] = q - lo

        trace_lemoine(p, "A")
        trace_lemoine(q, "B")
        return out

    # Even root
    if not include_even:
        return out
    gb = find_goldbach_pair(N, policy=policy, rootN=N)
    if gb is None:
        return out
    a, b = gb
    out["root_kind"] = "G"
    out["root_a"] = a
    out["root_b"] = b

    trace_lemoine(a, "A")
    trace_lemoine(b, "B")
    return out

# ------------------------------------------------------------
# Resolve to sink-weight vector + branching stats
# ------------------------------------------------------------

# Global cache: (policy, node) -> unit-coefficient W vector (np.int64 array)
# For deterministic policies only. After processing a few hundred integers
# this cache covers most encountered sub-primes, eliminating the vast
# majority of recursive sub-tree traversals for subsequent integers.
_W_RESOLVE_CACHE: Dict[Tuple, Optional[np.ndarray]] = {}

@dataclass
class ResolveResult:
    ok: bool
    W: Optional[np.ndarray]
    reason: str = ""
    fail_node: Optional[int] = None

def unit_W(sink: int, sink_index: Dict[int, int], D: int) -> np.ndarray:
    W = np.zeros(D, dtype=np.int64)
    W[sink_index[sink]] = 1
    return W

# Pre-compute unit vectors for sinks (avoids np.zeros per leaf)
_SINK_UNIT_VECTORS: Dict[int, np.ndarray] = {}

def _ensure_sink_unit_vectors(sinks: Tuple[int, ...], sink_index: Dict[int, int]) -> None:
    global _SINK_UNIT_VECTORS
    if not _SINK_UNIT_VECTORS:
        D = len(sinks)
        for s in sinks:
            v = np.zeros(D, dtype=np.int64)
            v[sink_index[s]] = 1
            _SINK_UNIT_VECTORS[s] = v

def resolve_with_branching(
    N: int,
    policy: str,
    include_even: bool,
    sinks: Tuple[int, ...],
    sink_index: Dict[int, int],
    sink_set: set,
    max_depth_guard: int = 128,
    collect_stats: bool = True,
) -> Tuple[ResolveResult, BranchStats]:
    """
    Returns (ResolveResult, BranchStats).

    Uses in-place accumulation into a plain Python list instead of creating
    and adding numpy arrays at every node.  This is ~3-5× faster for D=10
    because it eliminates hundreds of thousands of numpy.zeros + array
    addition operations per root integer.

    The W vector is accumulated as W[sink_index] += coeff at each sink leaf.
    Only converted to np.ndarray at the very end.
    """
    D = len(sinks)

    # Pre-allocate stats lists to avoid repeated _ensure / append calls
    stats_nodes   = [0] * (max_depth_guard + 1)
    stats_internal = [0] * (max_depth_guard + 1)
    stats_sinks   = [0] * (max_depth_guard + 1)
    stats_mass    = [0.0] * (max_depth_guard + 1)
    max_depth_seen = 0

    # In-place accumulation target
    W = [0] * D
    failed = False

    def rec(node: int, coeff: int, depth: int) -> None:
        nonlocal failed, max_depth_seen
        if failed:
            return
        if depth > max_depth_guard:
            failed = True
            return

        if depth > max_depth_seen:
            max_depth_seen = depth

        # --- Sink (leaf) ---
        if node in sink_set:
            if collect_stats:
                stats_nodes[depth] += 1
                stats_sinks[depth] += 1
                stats_mass[depth] += coeff
            W[sink_index[node]] += coeff
            return

        if collect_stats:
            stats_nodes[depth] += 1
            stats_internal[depth] += 1
            stats_mass[depth] += coeff

        # --- Odd node → Lemoine ---
        if (node & 1) == 1:
            pair = find_lemoine_pair(node, policy, N)
            if pair is None:
                failed = True
                return
            p, q = pair
            rec(p, coeff, depth + 1)
            if not failed:
                rec(q, coeff * 2, depth + 1)
            return

        # --- Even node → Goldbach ---
        if not include_even:
            failed = True
            return

        gb = find_goldbach_pair(node, policy=policy, rootN=N)
        if gb is None:
            failed = True
            return
        a, b = gb
        rec(a, coeff, depth + 1)
        if not failed:
            rec(b, coeff, depth + 1)

    rec(N, 1, 0)

    # Trim stats to actual depth used
    trim = max_depth_seen + 1
    stats = BranchStats(
        nodes=stats_nodes[:trim],
        internal=stats_internal[:trim],
        sinks=stats_sinks[:trim],
        mass=stats_mass[:trim],
    )

    if failed:
        return ResolveResult(False, None, reason="resolve_failed", fail_node=N), stats
    return ResolveResult(True, np.array(W, dtype=np.int64), reason=""), stats

# ------------------------------------------------------------
# Multiprocessing worker for Phase 2 (random / per-root policies)
# ------------------------------------------------------------

def _worker_init(prime_cache_snapshot: Dict[int, bool],
                 goldbach_cache_snapshot: Dict,
                 global_settings: Dict) -> None:
    """
    Initializer for each worker process.  Installs the warm prime and
    Goldbach caches built during Phase 1, and restores module-level
    settings so that find_lemoine_pair / find_goldbach_pair / is_prime
    work correctly.
    """
    global _PRIME_CACHE, _GOLDBACH_CACHE, _LEMOINE_CACHE
    global _RANDOM_WITNESS_SEED, _CENTER_FRACTION
    global _CENTER_ALPHA, _ALPHA_SEED, _ALPHA_JITTER, _ROOT_ALPHA_CACHE

    _PRIME_CACHE = dict(prime_cache_snapshot)
    _GOLDBACH_CACHE = dict(goldbach_cache_snapshot)
    _LEMOINE_CACHE = {}                       # fresh per worker (per-root keys)
    _ROOT_ALPHA_CACHE = {}                    # fresh per worker

    _RANDOM_WITNESS_SEED = global_settings.get("random_witness_seed", 42)
    _CENTER_FRACTION = global_settings.get("center_fraction", None)
    _CENTER_ALPHA = global_settings.get("center_alpha", None)
    _ALPHA_SEED = global_settings.get("alpha_seed", 0)
    _ALPHA_JITTER = global_settings.get("alpha_jitter", 0.0)


def _worker_process_chunk(args_tuple: Tuple) -> List[Tuple]:
    """
    Process a chunk of integers for the given policies.

    Args (unpacked from tuple for Pool.map compatibility):
        chunk:           list of ints (the N values)
        rng_policies:    list of policy strings to compute
        sinks:           tuple of sink primes
        include_even:    bool
        trace_witnesses: bool

    Returns:
        List of (N, pol, rr_ok, rr_W, rr_reason, bs_nodes, bs_internal,
                 bs_sinks, bs_mass, trace_dict) tuples.
    """
    chunk, rng_policies, sinks, include_even, trace_witnesses = args_tuple

    sink_index = {p: i for i, p in enumerate(sinks)}
    sink_set = set(sinks)
    results = []

    for N in chunk:
        for pol in rng_policies:
            rr, bs = resolve_with_branching(
                N=N, policy=pol, include_even=include_even,
                sinks=sinks, sink_index=sink_index, sink_set=sink_set
            )
            tr = trace_depth2_witnesses(N, pol, include_even, sink_set) if trace_witnesses else {}

            # Serialize to plain types for pickling
            results.append((
                N, pol,
                rr.ok,
                rr.W.tolist() if (rr.ok and rr.W is not None) else None,
                rr.reason,
                list(bs.nodes), list(bs.internal), list(bs.sinks),
                list(bs.mass),
                dict(tr),
            ))

    return results


# ------------------------------------------------------------
# CSV generation
# ------------------------------------------------------------

def generate_numbers(
    start: int,
    end: int,
    sample_n: int,
    seed: int,
    sampling: str = "uniform",
) -> List[int]:
    """Generate a list of integers from [start, end].

    Args:
        start, end: inclusive range bounds
        sample_n:   0 => consecutive [start, end]; >0 => draw this many
        seed:       RNG seed for reproducibility
        sampling:   one of
            "uniform"         – plain uniform random sample (original behaviour)
            "log_stratified"  – equal count per log₁₀ decade; uniform within each
            "stratified_bands"– equal count per predefined magnitude band; bands are
                                [start, 10^k₁), [10^k₁, 10^k₂), …, [10^k_last, end]
                                where k₁…k_last are the powers of 10 inside the range

    Returns:
        Sorted list of distinct integers.
    """
    if start > end:
        raise ValueError("start must be <= end")
    if sample_n <= 0:
        return list(range(start, end + 1))

    rng = random.Random(seed)
    span = end - start + 1

    if sampling == "uniform":
        n = min(sample_n, span)
        return sorted(rng.sample(range(start, end + 1), n))

    # --- Build magnitude bands ---
    # Band boundaries are powers of 10 that fall strictly inside (start, end).
    # First band: [start, 10^k₁);  last band: [10^k_last, end].
    boundaries = [start]
    p = 10 ** (len(str(start)))          # first power of 10 > start
    while p <= end:
        if p > start:
            boundaries.append(p)
        p *= 10
    boundaries.append(end + 1)           # exclusive upper for last band

    bands: List[Tuple[int, int]] = []    # (lo_inclusive, hi_inclusive)
    for i in range(len(boundaries) - 1):
        lo = boundaries[i]
        hi = boundaries[i + 1] - 1
        if lo <= hi:
            bands.append((lo, hi))

    n_bands = len(bands)
    if n_bands == 0:
        return sorted(rng.sample(range(start, end + 1), min(sample_n, span)))

    if sampling == "log_stratified":
        # Equal count per band (remainder distributed to largest bands first)
        base_per_band = sample_n // n_bands
        remainder = sample_n % n_bands
        # Give the remainder to the *largest* bands (sorted by size descending)
        band_sizes = [(hi - lo + 1, idx) for idx, (lo, hi) in enumerate(bands)]
        band_sizes.sort(key=lambda x: -x[0])
        alloc = [base_per_band] * n_bands
        for i in range(remainder):
            alloc[band_sizes[i][1]] += 1

    elif sampling == "stratified_bands":
        # Allocate proportional to log-width of each band (so each decade
        # gets roughly equal representation regardless of linear span).
        log_widths = []
        for lo, hi in bands:
            # log-width of band; use max(lo,1) to avoid log(0)
            lw = math.log10(max(hi, 1)) - math.log10(max(lo, 1))
            log_widths.append(max(lw, 0.01))  # floor for tiny bands
        total_lw = sum(log_widths)
        raw_alloc = [sample_n * lw / total_lw for lw in log_widths]
        # Round with remainder preservation
        alloc = [int(a) for a in raw_alloc]
        remainder = sample_n - sum(alloc)
        fracs = [(raw_alloc[i] - alloc[i], i) for i in range(n_bands)]
        fracs.sort(key=lambda x: -x[0])
        for i in range(remainder):
            alloc[fracs[i][1]] += 1

    else:
        raise ValueError(f"Unknown sampling strategy: {sampling!r}. "
                         f"Choose from: uniform, log_stratified, stratified_bands")

    # Sample within each band
    result: List[int] = []
    band_report: List[str] = []
    for (lo, hi), n_draw in zip(bands, alloc):
        band_span = hi - lo + 1
        n_draw = min(n_draw, band_span)
        if n_draw > 0:
            result.extend(rng.sample(range(lo, hi + 1), n_draw))
        band_report.append(f"    [{lo:>12,}, {hi:>12,}]  span={band_span:>12,}  n={n_draw:>5,}")

    print(f"Sampling strategy: {sampling}  ({len(bands)} bands, {len(result)} total)")
    for line in band_report:
        print(line)

    return sorted(result)

def write_embeddings_csv(
        sinks_mode: int,
    start: int,
    end: int,
    sample_n: int,
    seed: int,
    include_even: bool,
    policies: List[str],
    out_csv: str,
    max_report_depth: int,
    trace_witnesses: bool,
    branching_stats: bool = False,
    progress_every: int = 1000,
    nums_override: Optional[List[int]] = None,
    workers: int = 1,
    sampling: str = "uniform",
) -> None:
    if sinks_mode not in (10, 12):
        raise ValueError("--sinks must be 10 or 12")

    sinks = SINKS_10 if sinks_mode == 10 else SINKS_12
    sink_index = {p: i for i, p in enumerate(sinks)}
    sink_set = set(sinks)
    D = len(sinks)

    if nums_override is not None:
        nums = list(nums_override)
    else:
        nums = generate_numbers(start, end, sample_n, seed, sampling=sampling)
    if not include_even:
        nums = [n for n in nums if (n & 1) == 1]

    # --- Split policies into deterministic (cacheable) and per-root (random) ---
    # Deterministic policies benefit massively from cross-root Lemoine caching;
    # processing them first also warms the prime cache for the random pass.
    det_policies = [p for p in policies if p not in ("random", "alpha_rand")]
    rng_policies = [p for p in policies if p in ("random", "alpha_rand")]

    # --- Storage for per-(N, policy) results ---
    # Keyed by integer N, value is dict pol -> (ResolveResult, BranchStats, trace_dict)
    results: Dict[int, Dict[str, Tuple]] = {N: {} for N in nums}

    t0 = time.time()

    # === PHASE 1: Deterministic policies (high cache reuse) ===
    if det_policies:
        print(f"\n--- Phase 1: deterministic policies ({', '.join(p.upper() for p in det_policies)}) ---",
              flush=True)
        phase1_t0 = time.time()
        for idx, N in enumerate(nums, 1):
            if progress_every and (idx == 1 or (idx % progress_every) == 0):
                now = time.time()
                elapsed = now - phase1_t0
                rate = idx / max(elapsed, 1e-9)
                remaining = max(len(nums) - idx, 0)
                eta_sec = remaining / max(rate, 1e-9)
                pcache = len(_PRIME_CACHE)
                lcache = len(_LEMOINE_CACHE)
                print(f"  [Phase 1] [{idx:,}/{len(nums):,}] N={N:,}  "
                      f"rate={rate:,.1f}/s  elapsed={elapsed/60:.1f}m  ETA={eta_sec/60:.1f}m  "
                      f"primes={pcache:,}  lemoine={lcache:,}",
                      flush=True)

            for pol in det_policies:
                rr, bs = resolve_with_branching(
                    N=N, policy=pol, include_even=include_even,
                    sinks=sinks, sink_index=sink_index, sink_set=sink_set
                )
                tr = trace_depth2_witnesses(N, pol, include_even, sink_set) if trace_witnesses else {}
                results[N][pol] = (rr, bs, tr)

        dt1 = time.time() - phase1_t0
        print(f"  Phase 1 done in {dt1:.1f}s.  Prime cache: {len(_PRIME_CACHE):,} entries.  "
              f"Lemoine cache: {len(_LEMOINE_CACHE):,} entries.\n", flush=True)

    # === PHASE 2: Random/per-root policies (no cross-root Lemoine reuse, ===
    # ===          but prime cache is now warm from phase 1)              ===
    if rng_policies:
        n_workers = max(1, min(workers, multiprocessing.cpu_count() or 1))
        print(f"--- Phase 2: per-root policies ({', '.join(p.upper() for p in rng_policies)}) ---",
              flush=True)
        print(f"  (Prime cache warm: {len(_PRIME_CACHE):,} entries — "
              f"primality tests will be fast)", flush=True)

        if n_workers > 1:
            # --- Parallel path ---
            print(f"  Using {n_workers} worker processes", flush=True)

            # Snapshot caches built during Phase 1
            global_settings = {
                "random_witness_seed": _RANDOM_WITNESS_SEED,
                "center_fraction": _CENTER_FRACTION,
                "center_alpha": _CENTER_ALPHA,
                "alpha_seed": _ALPHA_SEED,
                "alpha_jitter": _ALPHA_JITTER,
            }

            # Split nums into roughly equal chunks
            chunk_size = max(1, (len(nums) + n_workers - 1) // n_workers)
            chunks = [nums[i:i + chunk_size] for i in range(0, len(nums), chunk_size)]

            work_args = [
                (chunk, rng_policies, sinks, include_even, trace_witnesses)
                for chunk in chunks
            ]

            phase2_t0 = time.time()
            print(f"  Distributing {len(nums):,} integers across {len(chunks)} chunks ...",
                  flush=True)

            with multiprocessing.Pool(
                processes=n_workers,
                initializer=_worker_init,
                initargs=(_PRIME_CACHE, _GOLDBACH_CACHE, global_settings),
            ) as pool:
                chunk_results = pool.map(_worker_process_chunk, work_args)

            # Unpack worker results back into the results dict
            for chunk_res in chunk_results:
                for (N, pol, ok, W_list, reason,
                     bs_nodes, bs_internal, bs_sinks, bs_mass, tr) in chunk_res:
                    W_arr = np.array(W_list, dtype=np.int64) if W_list is not None else None
                    rr = ResolveResult(ok=ok, W=W_arr, reason=reason)
                    bs = BranchStats(nodes=bs_nodes, internal=bs_internal,
                                     sinks=bs_sinks, mass=bs_mass)
                    results[N][pol] = (rr, bs, tr)

            dt2 = time.time() - phase2_t0
            print(f"  Phase 2 done in {dt2:.1f}s "
                  f"({len(nums)/max(dt2,0.01):,.0f} integers/s across {n_workers} workers).\n",
                  flush=True)

        else:
            # --- Sequential path (single-threaded) ---
            phase2_t0 = time.time()
            for idx, N in enumerate(nums, 1):
                if progress_every and (idx == 1 or (idx % progress_every) == 0):
                    now = time.time()
                    elapsed = now - phase2_t0
                    rate = idx / max(elapsed, 1e-9)
                    remaining = max(len(nums) - idx, 0)
                    eta_sec = remaining / max(rate, 1e-9)
                    print(f"  [Phase 2] [{idx:,}/{len(nums):,}] N={N:,}  "
                          f"rate={rate:,.1f}/s  elapsed={elapsed/60:.1f}m  ETA={eta_sec/60:.1f}m",
                          flush=True)

                for pol in rng_policies:
                    rr, bs = resolve_with_branching(
                        N=N, policy=pol, include_even=include_even,
                        sinks=sinks, sink_index=sink_index, sink_set=sink_set
                    )
                    tr = trace_depth2_witnesses(N, pol, include_even, sink_set) if trace_witnesses else {}
                    results[N][pol] = (rr, bs, tr)

            dt2 = time.time() - phase2_t0
            print(f"  Phase 2 done in {dt2:.1f}s.\n", flush=True)

    # === PHASE 3: Write CSV from stored results ===
    print(f"--- Writing CSV: {out_csv} ---", flush=True)

    # Header
    header: List[str] = ["N", "parity"]

    # Per-policy OK flags + reason
    for pol in policies:
        header += [f"ok_{pol}", f"reason_{pol}"]

    # Per-policy sink weights
    for pol in policies:
        for s in sinks:
            header.append(f"W_{pol}_{s}")
        header.append(f"recon_{pol}")

    # Per-policy witness trace (depth 1–2)
    if trace_witnesses:
        for pol in policies:
            for k in TRACE_FIELDS:
                header.append(f"T_{pol}_{k}")

    # Per-policy branching stats
    for pol in policies:
        header += [
            f"B_{pol}_max_depth",
            f"B_{pol}_total_nodes",
            f"B_{pol}_total_internal",
            f"B_{pol}_total_sinks",
            f"B_{pol}_total_mass",
        ]
        for d in range(max_report_depth + 1):
            header += [
                f"B_{pol}_d{d}_nodes",
                f"B_{pol}_d{d}_internal",
                f"B_{pol}_d{d}_sinks",
                f"B_{pol}_d{d}_mass",
            ]

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)

        for N in nums:
            parity = "odd" if (N & 1) == 1 else "even"
            row: List[object] = [N, parity]

            pol_data = results[N]

            # ok/reason columns
            for pol in policies:
                rr, bs, tr = pol_data.get(pol, (ResolveResult(False, None, "not_computed"), BranchStats([], [], [], []), {}))
                row.append(1 if rr.ok else 0)
                row.append("" if rr.ok else rr.reason)

            # weights + recon
            for pol in policies:
                rr, bs, tr = pol_data.get(pol, (ResolveResult(False, None), BranchStats([], [], [], []), {}))
                if not rr.ok or rr.W is None:
                    row.extend([""] * D)
                    row.append("")
                else:
                    Wv = rr.W
                    for k in range(D):
                        row.append(int(Wv[k]))
                    recon = int(sum(int(Wv[i]) * sinks[i] for i in range(D)))
                    row.append(recon)

            # witness trace columns
            if trace_witnesses:
                for pol in policies:
                    rr, bs, tr = pol_data.get(pol, (None, None, {}))
                    for k in TRACE_FIELDS:
                        row.append(tr.get(k, ""))

            # branching stats columns
            for pol in policies:
                rr, bs, tr = pol_data.get(pol, (None, BranchStats([], [], [], []), {}))
                tot = bs.totals()
                row += [
                    bs.max_depth(),
                    tot["total_nodes"],
                    tot["total_internal"],
                    tot["total_sinks"],
                    f"{tot['total_mass']:.6g}",
                ]
                for d in range(max_report_depth + 1):
                    if d < len(bs.nodes):
                        row += [
                            bs.nodes[d],
                            bs.internal[d],
                            bs.sinks[d],
                            f"{bs.mass[d]:.6g}",
                        ]
                    else:
                        row += [0, 0, 0, "0"]

            w.writerow(row)

    dt = time.time() - t0
    print(f"Done. Wrote {len(nums)} rows to {out_csv} in {dt:.1f}s")

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def parse_center_fraction(value: str) -> float:
    """
    Parse a center fraction from string (e.g., '2/7' or '0.2857').
    Returns float value and validates it's strictly in (1/5, 1/3).
    """
    try:
        # Try parsing as fraction first
        if '/' in value:
            frac = Fraction(value)
            result = float(frac)
        else:
            result = float(value)
        
        # Validate strictly between 1/5 and 1/3
        lower_bound = 1.0 / 5.0
        upper_bound = 1.0 / 3.0
        
        if result <= lower_bound or result >= upper_bound:
            raise ValueError(
                f"Center fraction must be strictly between 1/5 ({lower_bound:.10f}) "
                f"and 1/3 ({upper_bound:.10f}), got {result:.10f}"
            )
        
        return result
    except (ValueError, ZeroDivisionError) as e:
        raise ValueError(f"Invalid center fraction '{value}': {e}")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sinks", type=int, default=10, choices=[10, 12],
                    help="Choose sink system: 10 or 12")
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--n_list_csv", type=str, default=None,
                    help="Optional CSV containing an explicit list of integers to embed. If set, --start/--end/--sample_n are ignored.")
    ap.add_argument("--n_list_col", type=str, default="N",
                    help="Column name in --n_list_csv to use (default: N). If not found, uses the first column.")
    ap.add_argument("--n_list_n", type=int, default=0,
                    help="Optional subsample size from --n_list_csv (0 => use all). Subsampling is without replacement.")
    ap.add_argument("--n_list_seed", type=int, default=17,
                    help="Seed for subsampling from --n_list_csv (default: 17).")
    ap.add_argument("--sample_n", type=int, default=0,
                    help="0 => consecutive [start,end], else random sample size")
    ap.add_argument("--sample_seed", type=int, default=12345)
    ap.add_argument("--sampling", type=str, default="uniform",
                    choices=["uniform", "log_stratified", "stratified_bands"],
                    help="Sampling strategy: 'uniform' (default) draws uniformly from [start,end]; "
                         "'log_stratified' splits range into log10 decades and draws equal count "
                         "per decade; 'stratified_bands' allocates proportional to log-width "
                         "of each decade band.")
    ap.add_argument("--progress_every", type=int, default=1000,
                    help="Print progress every K integers (0 disables)")
    ap.add_argument("--include_even", type=int, default=0, choices=[0, 1],
                    help="1 => allow even roots via Goldbach; 0 => odds only")
    ap.add_argument("--policies", type=str, default="down,up,quarter",
                    help="Comma-separated policies: down,up,quarter,center,alpha,alpha_rand,random")
    ap.add_argument("--center", type=str, default=None,
                    help="Center fraction for 'center' policy (e.g., '2/7' or '0.2857'), "
                         "must be strictly between 1/5 and 1/3")
    ap.add_argument("--alpha", type=float, default=None,
                    help="For policy 'alpha': start search at a fixed position alpha in [0,1] "
                         "within the q-window [ceil(M/5), floor(M/3)] (0=lo, 1=hi).")
    ap.add_argument("--alpha_seed", type=int, default=0,
                    help="For policy 'alpha_rand': seed controlling per-integer randomized start points.")
    ap.add_argument("--alpha_jitter", type=float, default=0.0,
                    help="For policy 'alpha_rand' with --alpha set: per-integer alpha is jittered "
                         "uniformly within [alpha - jitter, alpha + jitter] (clipped).")
    ap.add_argument("--random_seed", type=int, default=42,
                    help="RNG seed for 'random' policy (uniform witness selection). Default: 42.")
    ap.add_argument("--max_report_depth", type=int, default=12,
                    help="Depth buckets to report in branching stats (root depth=0)")
    ap.add_argument("--trace_witnesses", type=int, default=1, choices=[0, 1],
                    help="1 => include depth-1/2 witness trace columns")
    ap.add_argument("--branching_stats", type=int, default=0, choices=[0, 1],
                    help="1 => include depth-wise branching stat columns (slower; "
                         "disables sub-tree W-vector caching). Default: 0")
    ap.add_argument("--workers", type=int, default=1,
                    help="Number of parallel worker processes for random/per-root policies. "
                         "Default: 1 (sequential). Set to number of CPU cores for speedup.")
    ap.add_argument("--out_csv", type=str, required=True)

    args = ap.parse_args()

    # Explicit N-list support (for fixed point-tracking across frames)
    def load_n_list(path: str, col: str) -> list[int]:
        import csv as _csv
        nums: list[int] = []
        with open(path, "r", newline="") as f:
            r = _csv.reader(f)
            header = next(r, None)
            if header is None:
                return nums
            header_l = [h.strip() for h in header]
            try:
                j = header_l.index(col)
            except ValueError:
                j = 0  # fall back to first column
            for row in r:
                if not row:
                    continue
                try:
                    nums.append(int(float(row[j])))
                except Exception:
                    continue
        return nums

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]

    # Build the integer list
    nums_override = None
    if args.n_list_csv:
        nums = load_n_list(args.n_list_csv, args.n_list_col)
        if args.n_list_n and args.n_list_n > 0 and len(nums) > args.n_list_n:
            import random as _random
            rng = _random.Random(args.n_list_seed)
            nums = rng.sample(nums, args.n_list_n)
        # Respect include_even flag for the *root* integers list
        if not args.include_even:
            nums = [n for n in nums if (n & 1) == 1]
        nums_override = nums
    
    # Validate policies
    for p in policies:
        if p not in ("down", "up", "quarter", "center", "alpha", "alpha_rand", "random"):
            raise ValueError(f"Unknown policy: {p}")
    
    # If 'center' policy is used, validate and set the center fraction
    if "center" in policies:
        if args.center is None:
            raise ValueError("--center fraction is required when using 'center' policy")
        center_value = parse_center_fraction(args.center)
        set_center_fraction(center_value)
        print(f"Using center fraction: {center_value:.10f} (input: {args.center})")
    elif args.center is not None:
        print("Warning: --center specified but 'center' not in --policies, ignoring")


    # If 'alpha'/'alpha_rand' policy is used, validate and set alpha controls
    if "alpha" in policies:
        if args.alpha is None:
            raise ValueError("--alpha is required when using policy 'alpha'")
        if args.alpha < 0.0 or args.alpha > 1.0:
            raise ValueError(f"--alpha must be in [0,1], got {args.alpha}")
        set_center_alpha(float(args.alpha))
        print(f"Using alpha: {args.alpha:.6f} (0=lo, 1=hi within [M/5, M/3])")

    if "alpha_rand" in policies:
        # If --alpha is provided, alpha_rand will jitter around it; otherwise it is fully uniform.
        if args.alpha is not None:
            if args.alpha < 0.0 or args.alpha > 1.0:
                raise ValueError(f"--alpha must be in [0,1], got {args.alpha}")
            set_center_alpha(float(args.alpha))
        set_alpha_seed(int(args.alpha_seed))
        set_alpha_jitter(float(args.alpha_jitter))
        if args.alpha is None:
            print(f"Using alpha_rand: uniform alpha in [0,1] with seed={args.alpha_seed}")
        else:
            print(f"Using alpha_rand: base alpha={args.alpha:.6f} jitter={args.alpha_jitter} seed={args.alpha_seed}")

    if ("alpha" not in policies) and ("alpha_rand" not in policies):
        if args.alpha is not None or args.alpha_seed != 0 or (args.alpha_jitter and args.alpha_jitter != 0.0):
            print("Warning: --alpha/--alpha_seed/--alpha_jitter specified but no alpha policy in --policies; ignoring")

    if "random" in policies:
        set_random_witness_seed(args.random_seed)
        print(f"Using random witness policy with seed={args.random_seed}")

    write_embeddings_csv(
        sinks_mode=args.sinks,
        start=args.start,
        end=args.end,
        sample_n=args.sample_n,
        seed=args.sample_seed,
        include_even=bool(args.include_even),
        policies=policies,
        out_csv=args.out_csv,
        nums_override=nums_override,
        max_report_depth=args.max_report_depth,
        trace_witnesses=bool(args.trace_witnesses),
        branching_stats=bool(args.branching_stats),
        progress_every=args.progress_every,
        workers=args.workers,
        sampling=args.sampling,
    )

if __name__ == "__main__":
    main()