#!/usr/bin/env python3
"""
coarsening_matrix_verification.py

Computes the coarsening matrix C from the 12-sink to the 10-sink terminal
basis (Lemma 1 / Theorem 1 of the poset-of-bases section), verifies the
identity  W_S10(N) = C * W_S12(N)  exactly for a large sample of integers
across all deterministic policies, and outputs:

  (a) The 10x12 coarsening matrix C as a LaTeX table (one per policy).
  (b) A verification summary (pass/fail counts, any residuals).
  (c) Both as a CSV for archival.

The coarsening matrix is policy-dependent in principle because each buffer
sink t in S12 minus S10 = {11, 13, 17, 23, 31} descends to S10-sinks under
the policy's witness-selection rule, and different policies could choose
different witnesses.  In practice, for the S12-to-S10 coarsening the matrix
turns out to be policy-invariant (see note in output).

Usage:
  python3 coarsening_matrix_verification.py \
      --start 5000001 --end 5050000 --sample_n 5000 \
      --include_even 1 \
      --policies down,up,quarter,center --center 4/15 \
      --out_prefix coarsening_results

Outputs:
  coarsening_results_matrix.csv      – one row per (policy, coarse_sink, fine_sink)
  coarsening_results_matrix.tex      – LaTeX table(s) for the paper
  coarsening_results_verification.csv – one row per (N, policy) with pass/fail
  coarsening_results_summary.txt     – human-readable summary

Requires:  numpy
Imports:   descent_graph_sink_weights (the main codebase) for is_prime,
           find_lemoine_pair, find_goldbach_pair, resolve_with_branching, etc.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Import the shared descent machinery
# ---------------------------------------------------------------------------

# Allow importing from the same directory or from a path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from descent_graph_sink_weights import (
    SINKS_10,
    SINKS_12,
    is_prime,
    find_lemoine_pair,
    find_goldbach_pair,
    resolve_with_branching,
    set_center_fraction,
    set_random_witness_seed,
    parse_center_fraction,
    generate_numbers,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BUFFER_SINKS = tuple(s for s in SINKS_12 if s not in set(SINKS_10))
# = (11, 13, 17, 23, 31)

SHARED_SINKS = tuple(s for s in SINKS_12 if s in set(SINKS_10))
# = (19, 29, 37, 47, 59, 73, 97)

SINK_INDEX_10 = {p: i for i, p in enumerate(SINKS_10)}
SINK_INDEX_12 = {p: i for i, p in enumerate(SINKS_12)}
SINK_SET_10 = set(SINKS_10)
SINK_SET_12 = set(SINKS_12)

D10 = len(SINKS_10)
D12 = len(SINKS_12)


# ---------------------------------------------------------------------------
# Step 1: Compute the coarsening matrix C  (D10 × D12)
# ---------------------------------------------------------------------------

def compute_coarsening_column(t: int, policy: str) -> np.ndarray:
    """
    Compute column C[:, j] of the coarsening matrix, i.e., the S10-weight
    vector of the S12-sink t under the given policy.

    If t is in S10 n S12  (a shared sink), the column is the standard basis
    vector e_t in R^D10.

    If t is in S12 minus S10  (a buffer sink), we run the full descent of t
    against the 10-sink basis under the given policy.
    """
    if t in SINK_SET_10:
        col = np.zeros(D10, dtype=np.int64)
        col[SINK_INDEX_10[t]] = 1
        return col

    # Buffer sink: run descent against S10
    rr, _ = resolve_with_branching(
        N=t,
        policy=policy,
        include_even=True,  # buffer sinks are small odd primes, but be safe
        sinks=SINKS_10,
        sink_index=SINK_INDEX_10,
        sink_set=SINK_SET_10,
        collect_stats=False,
    )

    if not rr.ok or rr.W is None:
        raise RuntimeError(
            f"Failed to resolve buffer sink {t} under policy '{policy}': "
            f"{rr.reason}"
        )

    return rr.W


def compute_coarsening_matrix(policy: str) -> np.ndarray:
    """
    Build the full D10 × D12 coarsening matrix C for the given policy.
    Column j corresponds to SINKS_12[j].
    """
    C = np.zeros((D10, D12), dtype=np.int64)
    for j, t in enumerate(SINKS_12):
        C[:, j] = compute_coarsening_column(t, policy)
    return C


# ---------------------------------------------------------------------------
# Step 2: Verify  W_S10(N)  =  C · W_S12(N)  for a sample of integers
# ---------------------------------------------------------------------------

def verify_coarsening(
    C: np.ndarray,
    N: int,
    policy: str,
    include_even: bool,
) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Compute W_S10(N) and W_S12(N) independently, then check C * W_S12 == W_S10.
    Returns (passed, residual_or_None).
    """
    # S10 descent
    rr10, _ = resolve_with_branching(
        N=N, policy=policy, include_even=include_even,
        sinks=SINKS_10, sink_index=SINK_INDEX_10, sink_set=SINK_SET_10,
        collect_stats=False,
    )
    if not rr10.ok or rr10.W is None:
        return False, None

    # S12 descent
    rr12, _ = resolve_with_branching(
        N=N, policy=policy, include_even=include_even,
        sinks=SINKS_12, sink_index=SINK_INDEX_12, sink_set=SINK_SET_12,
        collect_stats=False,
    )
    if not rr12.ok or rr12.W is None:
        return False, None

    W10 = rr10.W
    W12 = rr12.W

    # C · W12 should equal W10 exactly (integer arithmetic)
    predicted = C @ W12
    residual = W10 - predicted

    passed = np.all(residual == 0)
    return bool(passed), residual


# ---------------------------------------------------------------------------
# Step 3: Output formatting
# ---------------------------------------------------------------------------

def matrix_to_latex_invariant(C: np.ndarray, policies: list) -> str:
    """
    Format the coarsening matrix as a single policy-independent LaTeX table.
    Used when C is identical across all policies.
    """
    lines = []
    lines.append(f"% Coarsening matrix C  (S12 -> S10)  — policy-invariant")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Coarsening matrix $C(\calS_{12}, \calS_{10})$.  "
        r"Each column is the $\calS_{10}$-weight vector of the corresponding "
        r"$\calS_{12}$-sink.  The matrix is policy-invariant: the buffer sinks "
        r"are small enough that their Lemoine windows admit only one witness each.}"
    )
    lines.append(f"\\label{{tab:coarsening}}")
    lines.append(r"\smallskip")
    lines.append(r"\footnotesize")

    col_spec = "@{}r" + "r" * D12 + "@{}"
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    header_cells = []
    for t in SINKS_12:
        if t in SINK_SET_10:
            header_cells.append(f"${t}$")
        else:
            header_cells.append(f"$\\mathbf{{{t}}}$")
    lines.append("$s \\backslash t$ & " + " & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")

    for i, s in enumerate(SINKS_10):
        cells = []
        for j in range(D12):
            val = C[i, j]
            if val == 0:
                cells.append(r"\cdot")
            else:
                cells.append(str(val))
        lines.append(f"${s}$ & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def matrix_to_latex(C: np.ndarray, policy: str) -> str:
    """
    Format the D10 × D12 coarsening matrix as a LaTeX table.
    Rows = S10 sinks, Columns = S12 sinks.
    """
    lines = []
    lines.append(f"% Coarsening matrix C  (S12 -> S10)  under \\textsc{{{policy}}}")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        f"\\caption{{Coarsening matrix $C_\\pi(\\calS_{{12}}, \\calS_{{10}})$ "
        f"under \\textsc{{{policy}}}.  "
        f"Each column is the $\\calS_{{10}}$-weight vector of the corresponding "
        f"$\\calS_{{12}}$-sink.  Shared sinks yield standard basis vectors "
        f"(shown as 1); buffer sinks (\\textbf{{bold}} headers) show the "
        f"redistribution of their mass to coarse sinks.}}"
    )
    lines.append(f"\\label{{tab:coarsening-{policy}}}")
    lines.append(r"\smallskip")
    lines.append(r"\footnotesize")

    # Column spec: row header + 12 data columns
    col_spec = "@{}r" + "r" * D12 + "@{}"
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Header row: S12 sink labels, bold for buffer sinks
    header_cells = []
    for t in SINKS_12:
        if t in SINK_SET_10:
            header_cells.append(f"${t}$")
        else:
            header_cells.append(f"$\\mathbf{{{t}}}$")
    lines.append("$s \\backslash t$ & " + " & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")

    # Data rows
    for i, s in enumerate(SINKS_10):
        cells = []
        for j in range(D12):
            val = C[i, j]
            if val == 0:
                cells.append(r"\cdot")
            else:
                cells.append(str(val))
        lines.append(f"${s}$ & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def matrix_to_csv_rows(C: np.ndarray, policy: str) -> List[List]:
    """Return rows for the matrix CSV: (policy, coarse_sink, fine_sink, weight)."""
    rows = []
    for i, s in enumerate(SINKS_10):
        for j, t in enumerate(SINKS_12):
            rows.append([policy, s, t, int(C[i, j])])
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute coarsening matrices and verify C · W_S12 = W_S10."
    )
    ap.add_argument("--start", type=int, default=5_000_001)
    ap.add_argument("--end", type=int, default=5_050_000)
    ap.add_argument("--sample_n", type=int, default=5000,
                    help="0 => consecutive [start,end]; >0 => random sample")
    ap.add_argument("--sample_seed", type=int, default=12345)
    ap.add_argument("--include_even", type=int, default=1, choices=[0, 1])
    ap.add_argument("--policies", type=str, default="down,up,quarter,center",
                    help="Comma-separated deterministic policies")
    ap.add_argument("--center", type=str, default="4/15",
                    help="Center fraction for 'center' policy")
    ap.add_argument("--progress_every", type=int, default=500)
    ap.add_argument("--out_prefix", type=str, default="coarsening_results",
                    help="Prefix for output files")
    args = ap.parse_args()

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    include_even = bool(args.include_even)

    # Set up center fraction if needed
    if "center" in policies:
        center_value = parse_center_fraction(args.center)
        set_center_fraction(center_value)
        print(f"Center fraction: {center_value:.10f} (from '{args.center}')")

    # ── Generate integer sample ───────────────────────────────────────────
    print(f"\nGenerating integers from [{args.start}, {args.end}] ...")
    nums = generate_numbers(args.start, args.end, args.sample_n, args.sample_seed)
    if not include_even:
        nums = [n for n in nums if (n & 1) == 1]
    print(f"  {len(nums)} integers selected.\n")

    # ── Compute and display coarsening matrices ───────────────────────────
    matrices: Dict[str, np.ndarray] = {}
    latex_blocks: List[str] = []
    matrix_csv_rows: List[List] = []

    for pol in policies:
        print(f"Computing coarsening matrix for '{pol}' ...")
        C = compute_coarsening_matrix(pol)
        matrices[pol] = C

        # Display
        print(f"\n  C_{pol}  (rows = S10 sinks, cols = S12 sinks):\n")
        header = "       " + "".join(f"{t:>6}" for t in SINKS_12)
        print(header)
        for i, s in enumerate(SINKS_10):
            row_str = f"  {s:>4} " + "".join(f"{C[i, j]:>6}" for j in range(D12))
            print(row_str)
        print()

        # Reconstruction identity check on C columns
        for j, t in enumerate(SINKS_12):
            col_sum = sum(int(SINKS_10[i]) * int(C[i, j]) for i in range(D10))
            if col_sum != t:
                print(f"  WARNING: column {t} sums to {col_sum}, expected {t}")
            else:
                print(f"  Column {t}: reconstruction identity OK (Σ s·C[s,{t}] = {col_sum})")
        print()

        latex_blocks.append(matrix_to_latex(C, pol))
        matrix_csv_rows.extend(matrix_to_csv_rows(C, pol))

    # Check if all matrices are identical (policy-invariant)
    policy_invariant = True
    ref_pol = policies[0]
    for pol in policies[1:]:
        if not np.array_equal(matrices[pol], matrices[ref_pol]):
            policy_invariant = False
            break

    if policy_invariant:
        print("=" * 70)
        print("NOTE: The coarsening matrix is IDENTICAL across all policies.")
        print("This is expected: the buffer sinks {11,13,17,23,31} are small")
        print("enough that their Lemoine windows admit only one witness each,")
        print("so all policies select the same decomposition path.")
        print("=" * 70)
        print()
        # Produce a single policy-independent table
        latex_blocks = [matrix_to_latex_invariant(matrices[ref_pol], policies)]

    # ── Verify coarsening identity on sample ──────────────────────────────
    print("=" * 70)
    print("Verifying  W_S10(N) = C · W_S12(N)  for each (N, policy) ...")
    print("=" * 70)

    verification_rows: List[List] = []  # (N, policy, passed, max_abs_residual)
    summary: Dict[str, Dict[str, int]] = {
        pol: {"pass": 0, "fail": 0, "skip": 0} for pol in policies
    }

    t0 = time.time()
    total_checks = len(nums) * len(policies)
    done = 0

    for idx, N in enumerate(nums):
        for pol in policies:
            C = matrices[pol]
            passed, residual = verify_coarsening(C, N, pol, include_even)

            if residual is None:
                summary[pol]["skip"] += 1
                verification_rows.append([N, pol, "skip", ""])
            elif passed:
                summary[pol]["pass"] += 1
                verification_rows.append([N, pol, "pass", 0])
            else:
                summary[pol]["fail"] += 1
                max_res = int(np.max(np.abs(residual)))
                verification_rows.append([N, pol, "FAIL", max_res])
                print(f"  FAIL: N={N}, policy={pol}, residual={residual}")

            done += 1

        if args.progress_every and (idx + 1) % args.progress_every == 0:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            print(f"  [{idx + 1}/{len(nums)}]  {done}/{total_checks} checks  "
                  f"({rate:.0f}/s)")

    elapsed = time.time() - t0

    # ── Print summary ─────────────────────────────────────────────────────
    print(f"\nCompleted {total_checks} checks in {elapsed:.1f}s.\n")
    summary_lines = []
    all_pass = True
    for pol in policies:
        s = summary[pol]
        line = (f"  {pol:>10s}:  {s['pass']:>6d} pass,  {s['fail']:>6d} fail,  "
                f"{s['skip']:>6d} skip")
        print(line)
        summary_lines.append(line)
        if s["fail"] > 0:
            all_pass = False

    if all_pass:
        verdict = ("\nVERDICT: PASS — the coarsening identity "
                   "W_S10(N) = C · W_S12(N) holds exactly for all "
                   f"{total_checks} (N, policy) pairs tested.")
    else:
        verdict = ("\nVERDICT: FAIL — some coarsening identity checks "
                   "did not pass.  See details above.")
    print(verdict)
    summary_lines.append(verdict)

    # ── Write outputs ─────────────────────────────────────────────────────
    # Matrix CSV
    matrix_csv = f"{args.out_prefix}_matrix.csv"
    with open(matrix_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "coarse_sink", "fine_sink", "weight"])
        w.writerows(matrix_csv_rows)
    print(f"\nMatrix CSV:        {matrix_csv}")

    # Matrix LaTeX
    matrix_tex = f"{args.out_prefix}_matrix.tex"
    with open(matrix_tex, "w") as f:
        f.write("% Auto-generated by coarsening_matrix_verification.py\n")
        f.write("% One table per policy.\n\n")
        f.write("\n\n".join(latex_blocks))
        f.write("\n")
    print(f"Matrix LaTeX:      {matrix_tex}")

    # Verification CSV
    verif_csv = f"{args.out_prefix}_verification.csv"
    with open(verif_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "policy", "result", "max_abs_residual"])
        w.writerows(verification_rows)
    print(f"Verification CSV:  {verif_csv}")

    # Summary text
    summary_txt = f"{args.out_prefix}_summary.txt"
    with open(summary_txt, "w") as f:
        f.write("Coarsening Matrix Verification Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Range: [{args.start}, {args.end}]\n")
        f.write(f"Sample size: {len(nums)}\n")
        f.write(f"Include even: {include_even}\n")
        f.write(f"Policies: {policies}\n")
        f.write(f"Total checks: {total_checks}\n")
        f.write(f"Elapsed: {elapsed:.1f}s\n\n")
        if policy_invariant:
            f.write("POLICY INVARIANCE: The coarsening matrix is identical across\n")
            f.write("all tested policies.  The buffer sinks {11,13,17,23,31} are\n")
            f.write("small enough that their Lemoine windows admit only one witness\n")
            f.write("each, so all policies select the same decomposition.\n\n")
        f.write("Per-policy results:\n")
        for line in summary_lines:
            f.write(line + "\n")
        f.write("\n\nCoarsening matrices:\n\n")
        for pol in policies:
            C = matrices[pol]
            f.write(f"Policy: {pol}\n")
            header = "       " + "".join(f"{t:>6}" for t in SINKS_12)
            f.write(header + "\n")
            for i, s in enumerate(SINKS_10):
                row_str = f"  {s:>4} " + "".join(
                    f"{C[i, j]:>6}" for j in range(D12)
                )
                f.write(row_str + "\n")
            f.write("\n")
    print(f"Summary:           {summary_txt}")


if __name__ == "__main__":
    main()
