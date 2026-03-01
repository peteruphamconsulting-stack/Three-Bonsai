#!/usr/bin/env python3
"""
render_ancestry_policy_panels.py

Render a node-by-node (DAG) illustration of a single integer's iterated
Goldbach/Lemoine decomposition under multiple policies, using the *exact*
decomposition logic from HolisticEmbeddingE.py (the embedding generator).

Output:
  - One PNG per policy (Graphviz dot)
  - One combined 2x2 panel PNG (for 4 policies)

Usage examples:
  python3 render_ancestry_policy_panels.py --N 5001 --sinks 10 --outdir ancestry_viz
  python3 render_ancestry_policy_panels.py --N 5000 --sinks 10 --include_even 1 --outdir ancestry_viz
  python3 render_ancestry_policy_panels.py --N 5001 --center 4/15 --policies down,up,quarter,center

Notes:
- Node label "m=" is the total inbound coefficient mass at that node (aggregated over merged paths).
- Sink label "w=" is the final sink weight (embedding coefficient) accumulated at that sink.
- Edge labels are the multiplier applied along that branch:
    * Lemoine: M -> p has 1×, M -> q has 2× (since M = p + 2q)
    * Goldbach: N -> a and N -> b both have 1× (since N = a + b)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import shutil
import math
from collections import defaultdict, deque
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from PIL import Image, ImageDraw, ImageFont

# Import the exact logic from your embedding script.
import descent_graph_sink_weights as DG


def parse_fraction(s: str) -> float:
    try:
        return float(Fraction(s))
    except Exception as e:
        raise ValueError(f"Could not parse fraction '{s}' (e.g. 4/15): {e}")


def get_sink_tuple(sinks_mode: int) -> Tuple[int, ...]:
    if sinks_mode == 10:
        return DG.SINKS_10
    if sinks_mode == 12:
        return DG.SINKS_12
    raise ValueError("--sinks must be 10 or 12")


def children(node: int, policy: str, include_even: bool, sink_set: Set[int], rootN: int) -> Optional[List[Tuple[int, int]]]:
    if node in sink_set:
        return []
    # odd => Lemoine
    if (node & 1) == 1:
        pair = DG.find_lemoine_pair(node, policy, rootN=rootN)
        if pair is None:
            return None
        p, q = pair
        return [(p, 1), (q, 2)]
    # even => Goldbach
    if not include_even:
        return None
    gb = DG.find_goldbach_pair(node, policy=policy, rootN=rootN)
    if gb is None:
        return None
    a, b = gb
    return [(a, 1), (b, 1)]


def build_dag(root: int, policy: str, include_even: bool, sink_set: Set[int]):
    adj: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    nodes: Set[int] = {root}
    q: deque[int] = deque([root])

    while q:
        u = q.popleft()
        ch = children(u, policy, include_even, sink_set, rootN=root)
        if ch is None:
            raise RuntimeError(f"Failed to decompose node={u} under policy='{policy}' (include_even={include_even})")
        for v, mult in ch:
            adj[u].append((v, mult))
            if v not in nodes:
                nodes.add(v)
                q.append(v)

    # depths (min distance from root)
    depth: Dict[int, int] = {root: 0}
    dq: deque[int] = deque([root])
    while dq:
        u = dq.popleft()
        for v, _ in adj.get(u, []):
            nd = depth[u] + 1
            if v not in depth or nd < depth[v]:
                depth[v] = nd
                dq.append(v)

    # coefficient mass flow (aggregate over merged paths)
    # NOTE: Nodes can have multiple parents (merged paths). Using a simple
    # depth-sorted sweep is NOT correct because a node's outgoing edges may be
    # processed before all of its incoming mass is known. We therefore compute
    # mass via a true topological order over the DAG.
    mass: Dict[int, int] = defaultdict(int)
    mass[root] = 1

    indeg: Dict[int, int] = {n: 0 for n in nodes}
    for u, outs in adj.items():
        for v, _mult in outs:
            indeg[v] = indeg.get(v, 0) + 1

    # Kahn's algorithm (deterministic tie-break by (depth, value))
    ready = [n for n, d in indeg.items() if d == 0]
    ready.sort(key=lambda x: (depth.get(x, 10**9), x))
    dq: deque[int] = deque(ready)

    processed = 0
    while dq:
        u = dq.popleft()
        processed += 1
        for v, mult in adj.get(u, []):
            mass[v] += mass[u] * mult
            indeg[v] -= 1
            if indeg[v] == 0:
                dq.append(v)
        if dq:
            dq = deque(sorted(dq, key=lambda x: (depth.get(x, 10**9), x)))

    if processed != len(nodes):
        # Should never happen (the decomposition should strictly reduce values),
        # but guard anyway in case upstream logic changes.
        raise RuntimeError(
            f"Mass-flow topo pass did not process all nodes (processed={processed}, total={len(nodes)}). "
            "This suggests a cycle or disconnected component."
        )

    return nodes, adj, depth, mass


def pad_to(im: Image.Image, w: int, h: int) -> Image.Image:
    new = Image.new("RGB", (w, h), (255, 255, 255))
    new.paste(im, ((w - im.size[0]) // 2, (h - im.size[1]) // 2))
    return new


def make_dot(nodes, adj, depth, mass, sink_set, title: str) -> str:
    by_depth: Dict[int, List[int]] = defaultdict(list)
    for n in nodes:
        by_depth[depth.get(n, 0)].append(n)
    maxd = max(by_depth) if by_depth else 0

    lines: List[str] = []
    lines.append("digraph G {")
    lines.append("  rankdir=TB;")
    lines.append(f'  graph [fontname="Helvetica", labelloc="t", label="{title}", fontsize=18, nodesep=0.4, ranksep=0.6];')
    lines.append('  node [fontname="Helvetica", shape=ellipse, style="filled", fillcolor="white", fontsize=11];')
    lines.append('  edge [fontname="Helvetica", fontsize=9];')

    for d in range(maxd + 1):
        if d not in by_depth:
            continue
        lines.append("  { rank=same;")
        for n in sorted(by_depth[d]):
            nid = f"n{n}"
            if n in sink_set:
                lbl = f"{n}\\nw={mass[n]}"
                shape = "box"
                fill = '"#D0D0D0"'
                periph = 2
            else:
                lbl = f"{n}\\nm={mass[n]}"
                shape = "ellipse"
                fill = '"white"'
                periph = 1
            lines.append(f'    {nid} [label="{lbl}", shape={shape}, fillcolor={fill}, peripheries={periph}];')
        lines.append("  }")

    # Merge parallel edges: (u,v) -> total multiplier
    # This handles cases like 5001 = 1667 + 2*1667 (p == q)
    merged_edges: Dict[Tuple[int, int], int] = {}
    for u, outs in adj.items():
        for v, mult in outs:
            key = (u, v)
            merged_edges[key] = merged_edges.get(key, 0) + mult

    for (u, v), total_mult in merged_edges.items():
        if total_mult == 1:
            lines.append(f'  n{u} -> n{v};')
        else:
            lines.append(f'  n{u} -> n{v} [label=" x{total_mult}", fontcolor="red"];')
    lines.append("}")
    return "\n".join(lines)



def _arrow(draw: ImageDraw.ImageDraw, x0: float, y0: float, x1: float, y1: float, head_len: float = 10, head_w: float = 6):
    """Draw a simple arrow from (x0,y0) -> (x1,y1)."""
    draw.line((x0, y0, x1, y1), fill=(0, 0, 0), width=2)
    ang = math.atan2(y1 - y0, x1 - x0)
    # two head points
    hx1 = x1 - head_len * math.cos(ang) + head_w * math.sin(ang)
    hy1 = y1 - head_len * math.sin(ang) - head_w * math.cos(ang)
    hx2 = x1 - head_len * math.cos(ang) - head_w * math.sin(ang)
    hy2 = y1 - head_len * math.sin(ang) + head_w * math.cos(ang)
    draw.polygon([(x1, y1), (hx1, hy1), (hx2, hy2)], fill=(0, 0, 0))


def render_png_pil(nodes, adj, depth, mass, sink_set, title: str, png_path: Path) -> None:
    """Pure-Python fallback renderer (no Graphviz required)."""
    # Font (best-effort)
    try:
        font = ImageFont.truetype("Helvetica.ttf", 14)
        font_title = ImageFont.truetype("Helvetica.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
        font_title = ImageFont.load_default()

    by_depth: Dict[int, List[int]] = defaultdict(list)
    for n in nodes:
        by_depth[depth.get(n, 0)].append(n)
    maxd = max(by_depth) if by_depth else 0
    layers = []
    max_layer = 0
    for d in range(maxd + 1):
        layer = sorted(by_depth.get(d, []))
        layers.append(layer)
        max_layer = max(max_layer, len(layer))

    # Layout parameters
    x_gap = 160
    y_gap = 130
    node_w = 110
    node_h = 56
    margin_x = 60
    margin_y = 70
    title_h = 50

    W = max(900, margin_x * 2 + max_layer * x_gap)
    H = margin_y * 2 + title_h + (maxd + 1) * y_gap

    im = Image.new("RGB", (int(W), int(H)), (255, 255, 255))
    draw = ImageDraw.Draw(im)

    # Title
    draw.text((margin_x, 20), title, fill=(0, 0, 0), font=font_title)

    # Node positions
    pos: Dict[int, Tuple[float, float]] = {}
    for d, layer in enumerate(layers):
        if not layer:
            continue
        y = margin_y + title_h + d * y_gap
        # center nodes in this layer
        total_w = (len(layer) - 1) * x_gap
        x0 = (W - total_w) / 2.0
        for i, n in enumerate(layer):
            x = x0 + i * x_gap
            pos[n] = (x, y)

    # Draw edges first
    for u in nodes:
        for v, mult in adj.get(u, []):
            if u not in pos or v not in pos:
                continue
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            # start at bottom of u, end at top of v
            sx, sy = x0, y0 + node_h / 2
            tx, ty = x1, y1 - node_h / 2
            _arrow(draw, sx, sy, tx, ty)
            # edge label
            mx, my = (sx + tx) / 2, (sy + ty) / 2
            if mult != 1:
                draw.text((mx + 4, my - 10), f"×{mult}", fill=(0, 0, 0), font=font)

    # Draw nodes
    for n, (x, y) in pos.items():
        x0 = x - node_w / 2
        y0 = y - node_h / 2
        x1 = x + node_w / 2
        y1 = y + node_h / 2

        is_sink = n in sink_set
        fill = (220, 220, 220) if is_sink else (255, 255, 255)

        if is_sink:
            draw.rectangle((x0, y0, x1, y1), outline=(0, 0, 0), width=2, fill=fill)
            label = f"{n}\nw={mass[n]}"
        else:
            draw.ellipse((x0, y0, x1, y1), outline=(0, 0, 0), width=2, fill=fill)
            label = f"{n}\nm={mass[n]}"

        # label centered (compat across Pillow versions)
        try:
            bbox = draw.multiline_textbbox((0, 0), label, font=font, align="center")
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except Exception:
            # Older Pillow: fall back to approximate size
            tw, th = draw.multiline_textsize(label, font=font)
        draw.multiline_text((x - tw / 2, y - th / 2), label, fill=(0, 0, 0), font=font, align="center")

    im.save(png_path)
def render(root: int, sinks_mode: int, include_even: bool, policies: List[str], center: float, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    sinks = get_sink_tuple(sinks_mode)
    sink_set = set(sinks)

    DG.set_center_fraction(center)

    pngs: List[Path] = []
    titles = {
        "down": "N/3-down (DOWN)",
        "up": "N/5-up (UP)",
        "quarter": "N/4-outward (QUARTER)",
        "center": f"{Fraction(center).limit_denominator()}-centered (CENTER)",
    }

    for pol in policies:
        nodes, adj, depth, mass = build_dag(root, pol, include_even, sink_set)
        # Sanity check: the recursive linear identity should close exactly:
        # root == sum_{s in sinks} (mass[s] * s)
        sink_value = sum(int(mass.get(s, 0)) * int(s) for s in sink_set)
        if sink_value != root:
            diff = sink_value - root
            print(f"[warn] {pol}: sink-weighted sum = {sink_value} (diff {diff:+d}) != root {root}. "
                  "This indicates upstream decomposition did not fully terminate in the sink set, "
                  "or a cycle/merge issue in mass propagation.")
        title = f"Policy: {titles.get(pol, pol)} | Integer N={root}"
        dot = make_dot(nodes, adj, depth, mass, sink_set, title)
        dot_path = outdir / f"tree_{root}_{pol}.dot"
        png_path = outdir / f"tree_{root}_{pol}.png"
        dot_path.write_text(dot, encoding="utf-8")
        dot_exe = shutil.which("dot")
        if dot_exe:
            subprocess.run([dot_exe, "-Tpng", str(dot_path), "-o", str(png_path)], check=True)
        else:
            # Fallback: render with a pure-Python PIL routine so the script works even without Graphviz.
            # If you prefer Graphviz output quality, install it and ensure `dot` is on your PATH.
            render_png_pil(nodes, adj, depth, mass, sink_set, title, png_path)
        pngs.append(png_path)

    # combine into 2x2 if we have exactly 4 policies
    if len(pngs) == 4:
        ims = [Image.open(p) for p in pngs]
        max_w = max(im.size[0] for im in ims)
        max_h = max(im.size[1] for im in ims)
        ims = [pad_to(im, max_w, max_h) for im in ims]

        grid = Image.new("RGB", (2 * max_w, 2 * max_h), (255, 255, 255))
        grid.paste(ims[0], (0, 0))
        grid.paste(ims[1], (max_w, 0))
        grid.paste(ims[2], (0, max_h))
        grid.paste(ims[3], (max_w, max_h))

        grid_path = outdir / f"tree_{root}_all_policies.png"
        grid.save(grid_path)
        print(f"Wrote: {grid_path}")

    for p in pngs:
        print(f"Wrote: {p}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=5003, help="Integer to decompose.")
    ap.add_argument("--sinks", type=int, default=10, choices=[10, 12], help="Sink set size (10 or 12).")
    ap.add_argument("--include_even", type=int, default=1, help="Allow Goldbach splits for even nodes (0/1).")
    ap.add_argument("--policies", type=str, default="down,up,quarter,center",
                    help="Comma-separated policies: down,up,quarter,center")
    ap.add_argument("--center", type=str, default="4/15", help="Center fraction for 'center' policy, e.g. 4/15.")
    ap.add_argument("--outdir", type=str, default="ancestry_viz", help="Output directory.")
    args = ap.parse_args()

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    center = parse_fraction(args.center)
    render(
        root=args.N,
        sinks_mode=args.sinks,
        include_even=bool(args.include_even),
        policies=policies,
        center=center,
        outdir=Path(args.outdir),
    )


if __name__ == "__main__":
    main()
