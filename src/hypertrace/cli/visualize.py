"""Visualise trained concept spaces (Poincaré and/or Euclidean).

    hypertrace-visualize --run $HT_OUTPUT_ROOT/runs/main_poincare_masked_latent_f0_s0
    hypertrace-visualize --run <poincare run> --run <euclidean run> --color-by margin_spiculated
    hypertrace-visualize --run <run> --color-by malignancy --split test --out figs/
    hypertrace-visualize --run <run> --labels          # also write concept names next to the points

Writes to --out (default: ./concept_space_figures):
    concept_space.png        one row per run: radial depth view | disk view | node depth vs tree depth |
                             lesion depth for concept present vs absent
    <run>.html               interactive radial + disk views (hover: nodule id, split, depth, label;
                             click a legend entry to hide a concept or a whole group)
    <run>_coords.csv         2-D coordinates of every node and lesion, for your own figures

Product runs (one factor per concept group) get one row per factor, each in its own geometry
(Poincaré disk, sphere in stereographic view, or flat), with the colours of the whole tree.

Concept encoding: hue = concept group (size, shape, margin, texture, calcification; categorical
slots in fixed order), marker shape + shade = concept within its group; names are in the legend,
not on the plot (``--labels`` adds them back). Hue alone cannot separate five groups for every
reader (the colour-vision check fails past three hues on a scatter), so each concept is
identified by hue AND marker, and the legend is always drawn. Unscored nodes are hollow; trunk
nodes (chest ... nodule) are grey hexagons. Nodules are a grey ramp of the --color-by label, so they do
not compete with the concept colours.

Needs the run folder (config.yaml, checkpoints/best.pt) and the embedding cache it was trained on.
See hypertrace/viz.py for exactly how each layout is computed.
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from hypertrace.viz import SpaceView, build_views

log = logging.getLogger("hypertrace.visualize")

# Reference palette (dataviz skill, light mode).
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#e5e4e0"
SERIES_1, SERIES_2 = "#2a78d6", "#eb6834"  # categorical slots 1-2 (histogram: present / absent)
MISSING = "#c3c2b7"
# Categorical slots in fixed order: one per concept group.
GROUP_HUES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CONCEPT_MARKERS = ["o", "s", "^", "D", "v", "P", "p", "X"]
PLOTLY_SYMBOLS = {"o": "circle", "s": "square", "^": "triangle-up", "D": "diamond", "v": "triangle-down",
                  "P": "cross", "h": "hexagon", "p": "pentagon", "X": "x"}
# Neutral sequential ramp for the nodules (light = label absent, dark = present).
GREY_RAMP = ["#dcdbd5", "#c9c8c1", "#b4b3ac", "#94938c", "#75746e", "#52514e", "#2e2d2b"]


def _short(name: str) -> str:
    if name.startswith("size_") and (name[5:7] in ("lt", "ge") or "to" in name[5:]):
        return name[5:].replace("lt", "<").replace("ge", "≥").replace("to", "–") + " mm"
    for p in ("margin_", "texture_", "calcification_", "size_", "shape_"):
        if name.startswith(p):
            return name[len(p):].replace("_", " ")
    return name.replace("pulmonary_lesion", "pulm. lesion").replace("_", " ")


def _title(group: str, display: str | None = None) -> str:
    if display:  # ontology display term, without a parenthetical ("size (mean diameter)" -> "Size")
        display = display.split(" (")[0]
        return display[:1].upper() + display[1:]
    return {"texture": "Nodule type"}.get(group, group.replace("_", " ").capitalize())


def _mix(a: str, b: str, t: float) -> str:
    """Blend hex colour a towards b by t in [0, 1]."""
    ca = np.array([int(a[i:i + 2], 16) for i in (1, 3, 5)], float)
    cb = np.array([int(b[i:i + 2], 16) for i in (1, 3, 5)], float)
    c = np.round(ca + (cb - ca) * t).astype(int)
    return "#" + "".join(f"{v:02x}" for v in c)


@dataclass
class ConceptStyle:
    color: str
    marker: str
    kind: str          # trunk | group | concept
    group: str | None
    label: str
    scored: bool
    group_title: str = ""


def concept_styles(v: SpaceView) -> list[ConceptStyle]:
    disp = v.node_display or [None] * len(v.node_names)
    name_of = {n: d for n, d in zip(v.node_names, disp, strict=True)}

    def label(i: int) -> str:
        return disp[i] or _short(v.node_names[i])

    groups = list(dict.fromkeys(g for g in v.node_group if g is not None))
    order = v.group_order or groups  # all groups of the tree: a group keeps its colour in every factor row
    hue = {g: GROUP_HUES[order.index(g) % len(GROUP_HUES)] for g in groups}
    members = {g: [i for i, (n, gg) in enumerate(zip(v.node_names, v.node_group, strict=True))
                   if gg == g and n != g] for g in groups}
    styles: list[ConceptStyle | None] = [None] * len(v.node_names)
    for g in groups:
        idx = members[g]
        shades = np.linspace(-0.3, 0.45, len(idx)) if len(idx) > 1 else [0.0]
        for j, (i, t) in enumerate(zip(idx, shades, strict=True)):
            color = _mix(hue[g], SURFACE, -t) if t < 0 else _mix(hue[g], INK, t)
            styles[i] = ConceptStyle(color, CONCEPT_MARKERS[j % len(CONCEPT_MARKERS)], "concept", g,
                                     label(i), v.node_scored[i])
    for i, g in enumerate(v.node_group):
        if styles[i] is None:
            if g is not None:  # the group node itself (e.g. "size")
                styles[i] = ConceptStyle(hue[g], "o", "group", g, _title(g, name_of.get(g)), v.node_scored[i])
            else:
                styles[i] = ConceptStyle(INK_2, "h", "trunk", None, label(i), v.node_scored[i])
    for st in styles:  # group titles in the legend come from the group node's display term
        st.group_title = _title(st.group, name_of.get(st.group)) if st.group else "Tree"
    return styles  # type: ignore[return-value]


def safe_name(run_name: str) -> str:
    """File name for a view ("run · size (euclidean)" -> "run__size")."""
    return re.sub(r"\s*\(.*?\)", "", run_name).replace(" · ", "__").replace(" ", "_")


def geometry_words(v: SpaceView) -> tuple[str, str]:
    """(geometry label, what the radius / depth axis means)."""
    if v.geometry == "poincare":
        return "Poincaré", "hyperbolic depth d(0,x)"
    if v.geometry == "euclidean":
        return "Euclidean", "Euclidean norm |x|"
    label = "Sphere" if v.geometry == "sphere" else "κ-stereographic"
    return f"{label} (κ = {v.kappa:.2g})", "geodesic depth d(0,x)"


def _color_values(v: SpaceView, color_by: str) -> tuple[np.ndarray, str, tuple[float, float]]:
    if color_by == "malignancy":
        return v.lesions["malignancy_mean"].to_numpy(), "reader-mean malignancy (1-5)", (1.0, 5.0)
    col = f"soft_{color_by}"
    if col not in v.lesions:
        opts = [c[5:] for c in v.lesions.columns if c.startswith("soft_")] + ["malignancy"]
        raise SystemExit(f"--color-by {color_by!r} unknown. Options: {', '.join(opts)}")
    return v.lesions[col].to_numpy(), f"{color_by} (soft label)", (0.0, 1.0)


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=7)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _scatter_nodes(ax, xy: np.ndarray, styles: list[ConceptStyle]) -> None:
    for (x, y), st in zip(xy, styles, strict=True):
        if st.kind == "trunk":
            ax.scatter([x], [y], s=60, marker="h", color=INK_2, edgecolors=SURFACE, linewidths=1.2, zorder=3)
        elif st.scored:
            ax.scatter([x], [y], s=70, marker=st.marker, color=st.color, edgecolors=SURFACE, linewidths=1.2,
                       zorder=4)
        else:  # unscored: hollow
            ax.scatter([x], [y], s=54, marker=st.marker, facecolors=SURFACE, edgecolors=st.color, linewidths=1.8,
                       zorder=4)


def _draw_space(ax, v: SpaceView, nodes: np.ndarray, lesions: np.ndarray, colors: np.ndarray, rng, cmap,
                labels: bool, mask: np.ndarray, styles: list[ConceptStyle], lim: float | None = None) -> object:
    import matplotlib.patheffects as pe

    idx = {n: i for i, n in enumerate(v.node_names)}
    for i, parent in enumerate(v.node_parents):
        if parent is not None:
            a, b = nodes[idx[parent]], nodes[i]
            ax.plot([a[0], b[0]], [a[1], b[1]], color=MUTED, linewidth=0.9, zorder=2)
    finite = np.isfinite(colors) & mask
    ax.scatter(lesions[mask & ~finite, 0], lesions[mask & ~finite, 1], s=9, color=MISSING, linewidths=0, zorder=1)
    sc = ax.scatter(lesions[finite, 0], lesions[finite, 1], c=colors[finite], cmap=cmap, vmin=rng[0], vmax=rng[1],
                    s=10, linewidths=0, alpha=0.8, zorder=1)
    _scatter_nodes(ax, nodes, styles)
    if lim is not None:  # fixed square window (disk view)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
    else:
        ax.set_aspect("equal", adjustable="datalim")
    if labels:
        place_labels(ax, nodes, [st.label for st in styles], halo=pe.withStroke(linewidth=2.5, foreground=SURFACE))
    return sc


_OFFSETS = [(5, 4), (5, -10), (-5, 4), (-5, -10), (0, 9), (0, -14), (10, -3), (-10, -3),
            (14, 8), (-14, 8), (14, -16), (-14, -16), (0, 18), (0, -22), (20, -3), (-20, -3)]


def place_labels(ax, xy: np.ndarray, texts: list[str], halo, fontsize: float = 6.5) -> None:
    """Greedy collision-avoiding labels: try offsets around each point, keep the first that overlaps no
    earlier label; if none fits, use the least-overlapping one. Leader lines are drawn for far offsets."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placed: list = []
    order = np.argsort(-np.linalg.norm(xy, axis=1))  # outermost (most specific) first: they matter most
    for i in order:
        best, best_overlap = None, None
        for dx, dy in _OFFSETS:
            ha = "left" if dx > 0 else "right" if dx < 0 else "center"
            t = ax.annotate(texts[i], xy[i], xytext=(dx, dy), textcoords="offset points", fontsize=fontsize,
                            color=INK, zorder=5, ha=ha, path_effects=[halo])
            bb = t.get_window_extent(renderer).expanded(1.05, 1.15)
            overlap = sum(max(0, min(bb.x1, o.x1) - max(bb.x0, o.x0)) * max(0, min(bb.y1, o.y1) - max(bb.y0, o.y0))
                          for o in placed)
            if best_overlap is None or overlap < best_overlap:
                if best is not None:
                    best[0].remove()
                best, best_overlap = (t, bb, dx, dy), overlap
            else:
                t.remove()
            if overlap == 0:
                break
        t, bb, dx, dy = best
        if abs(dx) + abs(dy) > 16:  # far offset: leader line back to the point
            t.arrow_patch = None
            ax.annotate("", xy[i], xytext=(dx, dy), textcoords="offset points", zorder=4,
                        arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.6, shrinkA=0, shrinkB=2))
        placed.append(bb)


def legend_columns(styles: list[ConceptStyle]) -> tuple[list, list[str], set[int]]:
    """Handles and labels for a grouped figure legend: one column per concept group (header + its
    concepts) and a last column for the tree. Returns (handles, labels, header positions)."""
    from matplotlib.lines import Line2D

    def mk(st: ConceptStyle):
        if st.scored:
            return Line2D([], [], marker=st.marker, linestyle="", color=st.color, markeredgecolor=SURFACE,
                          markersize=7.5)
        return Line2D([], [], marker=st.marker, linestyle="", markerfacecolor=SURFACE, markeredgecolor=st.color,
                      markeredgewidth=1.6, markersize=7)

    blank = Line2D([], [], linestyle="", marker="")
    cols: list[list[tuple]] = []
    for g in dict.fromkeys(st.group for st in styles if st.group is not None):
        col = [(blank, next(st.group_title for st in styles if st.group == g), True)]
        col += [(mk(st), st.label, False)
                for st in styles if st.group == g and st.kind == "concept"]
        cols.append(col)
    cols.append([(blank, "Tree", True),
                 (Line2D([], [], marker="h", linestyle="", color=INK_2, markersize=7.5), "trunk concept", False),
                 (Line2D([], [], marker="o", linestyle="", markerfacecolor=SURFACE, markeredgecolor=INK_2,
                         markeredgewidth=1.6, markersize=7), "hollow = unscored", False),
                 (Line2D([], [], color=MUTED), "parent → child", False),
                 (Line2D([], [], marker="o", linestyle="", color=GREY_RAMP[4], markersize=4), "nodule", False)])
    rows = max(len(c) for c in cols)
    handles, labels, headers = [], [], set()
    for c in cols:  # matplotlib fills legend columns top to bottom
        c = c + [(blank, "", False)] * (rows - len(c))
        for h, lbl, is_header in c:
            if is_header:
                headers.add(len(labels))
            handles.append(h)
            labels.append(lbl)
    return handles, labels, headers


def plot_png(views: list[SpaceView], color_by: str, split: str, labels: bool, out: Path,
             full_disk: bool = False) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Circle

    cmap = LinearSegmentedColormap.from_list("seq_grey", GREY_RAMP)
    seen: dict[tuple, ConceptStyle] = {}  # legend over every view (product runs: one group per row)
    for v in views:
        for st in concept_styles(v):
            seen.setdefault((st.group, st.label, st.kind), st)
    order = views[0].group_order or []
    styles0 = sorted(seen.values(), key=lambda st: order.index(st.group) if st.group in order else -1)
    handles, leg_labels, headers = legend_columns(styles0)
    n_cols = len(set(st.group for st in styles0 if st.group is not None)) + 1
    leg_rows = len(leg_labels) // n_cols
    leg_h = 0.19 * leg_rows + 0.35  # inches
    fig_h = 4.6 * len(views) + leg_h
    fig, grid = plt.subplots(len(views), 5, figsize=(18, fig_h), squeeze=False,
                             gridspec_kw={"width_ratios": [1.15, 1.15, 0.045, 0.9, 0.9], "wspace": 0.36})
    fig.patch.set_facecolor(SURFACE)
    for grow, v in zip(grid, views, strict=True):
        styles = concept_styles(v)
        row = [grow[0], grow[1], grow[3], grow[4]]
        cax = grow[2]
        colors, clabel, rng = _color_values(v, color_by)
        mask = np.ones(len(v.lesions), bool) if split == "all" else (v.lesions["split"] == split).to_numpy()
        geo, depth_word = geometry_words(v)

        ax = row[0]
        _style(ax)
        rmax = max(v.node_depth.max(), v.lesions["depth"].max()) * 1.05
        for r in np.linspace(0, rmax, 5)[1:]:
            ax.add_patch(Circle((0, 0), r, fill=False, color=GRID, linewidth=0.8, zorder=0))
        sc = _draw_space(ax, v, v.radial_nodes, v.radial_lesions, colors, rng, cmap, labels, mask, styles)
        ax.set_title(f"{v.run_name}\n{geo} · radial view (radius = {depth_word})", fontsize=8.5, color=INK, loc="left")

        ax = row[1]
        _style(ax)
        lim = None
        if v.disk_radius is not None:
            ax.add_patch(Circle((0, 0), v.disk_radius, fill=False, color=MUTED, linewidth=1, linestyle="--"))
            if v.disk_kind == "sphere":
                title = (f"Sphere, stereographic (tangent PCA → exp map, κ = {v.kappa:.2g})\n"
                         "dashed = equator; outside it = far hemisphere")
            else:
                title = f"Poincaré disk (tangent PCA → exp map, κ = {v.kappa:.2g})"
            extent = np.abs(np.concatenate([v.disk_nodes, v.disk_lesions[mask]])).max() * 1.25
            if v.disk_kind == "sphere":
                lim = max(extent, v.disk_radius * 1.05)
            elif full_disk or extent >= 0.8 * v.disk_radius:
                lim = v.disk_radius * 1.05
            else:
                lim = extent
                title += f"\nzoomed to data; boundary at r = {v.disk_radius:.2g} (use --full-disk)"
        else:
            title = "PCA of the Euclidean concept space"
        _draw_space(ax, v, v.disk_nodes, v.disk_lesions, colors, rng, cmap, labels, mask, styles, lim)
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
        cb = fig.colorbar(sc, cax=cax)
        short = "malignancy\n(reader mean)" if color_by == "malignancy" else f"nodules:\n{_short(color_by)}"
        cb.ax.set_title(short, fontsize=7, color=INK_2, loc="left", pad=6)
        cb.ax.tick_params(labelsize=7, colors=INK_2)
        cb.outline.set_edgecolor(GRID)

        ax = row[2]
        _style(ax)
        jit = (np.arange(len(v.node_names)) % 5 - 2) * 0.05
        x = v.node_tree_depth + jit
        _scatter_nodes(ax, np.c_[x, v.node_depth], styles)
        if labels:
            import matplotlib.patheffects as pe
            place_labels(ax, np.c_[x, v.node_depth], [st.label for st in styles],
                         halo=pe.withStroke(linewidth=2.5, foreground=SURFACE), fontsize=6)
        rho = spearmanr(v.node_tree_depth, v.node_depth).statistic
        ax.set_xlabel("depth in concept tree", fontsize=7.5, color=INK_2)
        ax.set_ylabel(depth_word, fontsize=7.5, color=INK_2)
        ax.set_title(f"Concept depth vs tree depth (Spearman ρ = {rho:.2f})", fontsize=8.5, color=INK, loc="left")
        ax.margins(x=0.15, y=0.12)

        ax = row[3]
        _style(ax)
        d = v.lesions["depth"].to_numpy()[mask]
        cv = colors[mask]
        thr = 3.0 if color_by == "malignancy" else 0.5
        bins = np.linspace(d.min(), d.max(), 30)
        pos_lbl, neg_lbl = (("malignancy ≥ 3", "malignancy < 3") if color_by == "malignancy"
                            else (f"{_short(color_by)} present", f"{_short(color_by)} absent"))
        for sel, col, lbl in ((cv >= thr, SERIES_1, pos_lbl), (cv < thr, SERIES_2, neg_lbl)):
            if sel.any():
                ax.hist(d[sel], bins=bins, histtype="step", linewidth=2, color=col, density=True,
                        label=f"{lbl} (n={sel.sum()})")
        ax.margins(y=0.25)
        ax.set_xlabel(f"lesion {depth_word}", fontsize=7.5, color=INK_2)
        ax.set_ylabel("density", fontsize=7.5, color=INK_2)
        ax.set_title(f"Lesion depth ({split} nodules)", fontsize=8.5, color=INK, loc="left")
        ax.legend(fontsize=7, frameon=False, labelcolor=INK_2, loc="upper right")

    leg = fig.legend(handles=handles, labels=leg_labels, loc="lower center", ncol=n_cols, frameon=False,
                     fontsize=7.5, labelcolor=INK_2, columnspacing=2.2, handletextpad=0.4,
                     bbox_to_anchor=(0.5, 0.0))
    for i, txt in enumerate(leg.get_texts()):
        if i in headers:
            txt.set_fontweight("bold")
            txt.set_color(INK)
    fig.subplots_adjust(left=0.04, right=0.985, top=1 - 0.55 / fig_h, bottom=(leg_h + 0.45) / fig_h,
                        hspace=0.38)
    path = out / "concept_space.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_html(v: SpaceView, color_by: str, split: str, out: Path, labels: bool = False) -> Path | None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        log.warning("plotly not installed (pip install plotly): skipping the interactive HTML")
        return None
    colors, clabel, rng = _color_values(v, color_by)
    mask = np.ones(len(v.lesions), bool) if split == "all" else (v.lesions["split"] == split).to_numpy()
    les = v.lesions[mask]
    styles = concept_styles(v)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("radial view (radius = depth)", "disk / PCA view"))
    scale = [[i / (len(GREY_RAMP) - 1), c] for i, c in enumerate(GREY_RAMP)]
    idx = {n: i for i, n in enumerate(v.node_names)}
    for col, (nodes, lesions) in enumerate(((v.radial_nodes, v.radial_lesions), (v.disk_nodes, v.disk_lesions)), 1):
        ex, ey = [], []
        for i, p in enumerate(v.node_parents):
            if p is not None:
                ex += [nodes[idx[p], 0], nodes[i, 0], None]
                ey += [nodes[idx[p], 1], nodes[i, 1], None]
        fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color=MUTED, width=1), hoverinfo="skip",
                                 showlegend=False), 1, col)
        hover = [f"{r.nodule_id}<br>split: {r.split}<br>depth: {r.depth:.3f}<br>{clabel}: {c:.2f}"
                 for r, c in zip(les.itertuples(), colors[mask], strict=True)]
        fig.add_trace(go.Scatter(
            x=lesions[mask, 0], y=lesions[mask, 1], mode="markers", text=hover, hoverinfo="text", name="nodules",
            legendgroup="nodules",
            marker=dict(size=5, color=np.nan_to_num(colors[mask], nan=rng[0]), colorscale=scale, cmin=rng[0],
                        cmax=rng[1], opacity=0.85, colorbar=dict(title=clabel) if col == 2 else None,
                        showscale=col == 2),
            showlegend=col == 1), 1, col)
        seen: set = set()
        for i, st in enumerate(styles):
            group = st.group_title
            first = group not in seen
            seen.add(group)
            hollow = st.kind != "trunk" and not st.scored
            fig.add_trace(go.Scatter(
                x=[nodes[i, 0]], y=[nodes[i, 1]], mode="markers+text" if labels else "markers",
                text=[st.label] if labels else None, textposition="top right",
                textfont=dict(size=10, color=INK),
                hovertext=[f"<b>{v.node_names[i]}</b><br>group: {group}<br>depth: {v.node_depth[i]:.3f}"
                           f"<br>tree depth: {int(v.node_tree_depth[i])}"], hoverinfo="text",
                marker=dict(size=12 if st.scored else 10, symbol=PLOTLY_SYMBOLS[st.marker],
                            color=SURFACE if hollow else st.color,
                            line=dict(color=st.color if hollow else SURFACE, width=2 if hollow else 1.2)),
                name=st.label + ("" if st.scored or st.kind == "trunk" else " (unscored)"),
                # one legend entry per concept; clicking it hides that concept in both panels
                legendgroup=f"node-{i}", legendgrouptitle_text=group if (first and col == 1) else None,
                showlegend=col == 1), 1, col)
        fig.update_xaxes(scaleanchor=f"y{'' if col == 1 else col}", row=1, col=col, gridcolor=GRID, zeroline=False)
        fig.update_yaxes(row=1, col=col, gridcolor=GRID, zeroline=False)
    if v.disk_radius is not None:
        extent = np.abs(np.concatenate([v.disk_nodes, v.disk_lesions[mask]])).max() * 1.25
        lim = v.disk_radius * 1.05 if extent >= 0.8 * v.disk_radius else extent  # open zoomed, like the PNG
        if v.disk_kind == "sphere":
            lim = max(extent, v.disk_radius * 1.05)
        fig.update_xaxes(range=[-lim, lim], row=1, col=2)
        fig.update_yaxes(range=[-lim, lim], row=1, col=2)
        t = np.linspace(0, 2 * np.pi, 200)
        fig.add_trace(go.Scatter(x=v.disk_radius * np.cos(t), y=v.disk_radius * np.sin(t), mode="lines",
                                 line=dict(color=MUTED, dash="dash", width=1), hoverinfo="skip", showlegend=False),
                      1, 2)
    fig.update_layout(title=f"{v.run_name} ({v.geometry})", plot_bgcolor=SURFACE, paper_bgcolor=SURFACE,
                      font=dict(color=INK, size=12), height=680, width=1450,
                      legend=dict(groupclick="toggleitem", tracegroupgap=6, font=dict(size=11)))
    path = out / f"{safe_name(v.run_name)}.html"
    fig.write_html(path, include_plotlyjs=True)  # self-contained: opens offline
    return path


def write_coords(v: SpaceView, out: Path) -> Path:
    nodes = pd.DataFrame({"kind": "concept", "id": v.node_names, "parent": v.node_parents, "scored": v.node_scored,
                          "group": v.node_group, "tree_depth": v.node_tree_depth, "depth": v.node_depth,
                          "radial_x": v.radial_nodes[:, 0], "radial_y": v.radial_nodes[:, 1],
                          "disk_x": v.disk_nodes[:, 0], "disk_y": v.disk_nodes[:, 1]})
    les = v.lesions.rename(columns={"nodule_id": "id"}).assign(
        kind="nodule", radial_x=v.radial_lesions[:, 0], radial_y=v.radial_lesions[:, 1],
        disk_x=v.disk_lesions[:, 0], disk_y=v.disk_lesions[:, 1])
    path = out / f"{safe_name(v.run_name)}_coords.csv"
    pd.concat([nodes, les], ignore_index=True).to_csv(path, index=False)
    return path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, type=Path, help="run folder (repeat to compare)")
    ap.add_argument("--color-by", default="margin_spiculated", help="scored concept name, or 'malignancy'")
    ap.add_argument("--split", default="all", choices=["all", "train", "val", "test"], help="which nodules to show")
    ap.add_argument("--out", type=Path, default=Path("concept_space_figures"))
    ap.add_argument("--no-html", action="store_true")
    ap.add_argument("--labels", action="store_true", help="also write concept names next to the points")
    ap.add_argument("--no-labels", action="store_true", help=argparse.SUPPRESS)  # old flag: now the default
    ap.add_argument("--full-disk", action="store_true", help="always show the whole Poincaré disk (default: zoom)")
    ap.add_argument("--name", default="concept_space", help="PNG file name (without .png)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    a.out.mkdir(parents=True, exist_ok=True)

    views = []
    for run in a.run:
        if not (run / "checkpoints" / "best.pt").exists():
            raise SystemExit(f"{run}: no checkpoints/best.pt. Is this a finished hypertrace-train run folder?")
        for v in build_views(run):
            views.append(v)
            log.info("%s: %d concepts, %d nodules", v.run_name, len(v.node_names), len(v.lesions))
    png = plot_png(views, a.color_by, a.split, a.labels and not a.no_labels, a.out, a.full_disk)
    if a.name != "concept_space":
        png = png.rename(png.with_name(f"{a.name}.png"))
    log.info("wrote %s", png)
    for v in views:
        log.info("wrote %s", write_coords(v, a.out))
        if not a.no_html:
            p = plot_html(v, a.color_by, a.split, a.out, a.labels)
            if p:
                log.info("wrote %s", p)


if __name__ == "__main__":
    main()
