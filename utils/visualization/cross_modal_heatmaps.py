"""
Cross-modal attention heatmaps.

Visualizes Signal G: the [num_prototypes × num_pathways] attention matrix
from cross-attention fusion, where prototypes query pathways.

Provides:
    - Per-risk-group averaged heatmaps with difference panel
    - Top prototype-pathway pairs bar chart
    - Individual prototype pathway profiles
"""

import os
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

logger = logging.getLogger(__name__)

COLOR_LOW = '#2196F3'
COLOR_HIGH = '#E53935'


def plot_cross_modal_heatmap(
    attention_by_patient: Dict[str, Dict],
    pathway_names: List[str],
    prototype_names: List[str],
    output_path: str,
    title: str = 'Cross-Modal Attention (Population Average)',
    figsize: Tuple[int, int] = None,
    cmap: str = 'YlOrRd',
    top_n_pathways: int = 30,
):
    """
    Single heatmap of cross-modal attention averaged across all patients.

    Args:
        attention_by_patient: Dict pid → {'cross_modal_attention': [N, P], ...}
        pathway_names: Pathway name list (length P).
        prototype_names: Prototype name list (length N).
        output_path: Save path.
        title: Plot title.
        figsize: Figure size (auto-computed if None).
        cmap: Colormap.
        top_n_pathways: Show only top pathways by max attention.
    """
    avg_attn = _compute_group_average(attention_by_patient)

    if avg_attn is None:
        logger.warning("No cross-modal attention data available")
        return

    # Select top pathways by maximum attention across any prototype
    avg_attn, sel_pathway_names = _select_top_pathways(
        avg_attn, pathway_names, top_n_pathways
    )

    _render_single_heatmap(
        avg_attn, sel_pathway_names, prototype_names,
        title, output_path, figsize, cmap
    )


def plot_cross_modal_comparison(
    attention_by_patient: Dict[str, Dict],
    pathway_names: List[str],
    prototype_names: List[str],
    output_dir: str,
    significance_mask: Optional[Dict[int, np.ndarray]] = None,
    top_n_pathways: int = 25,
    cmap_groups: str = 'YlOrRd',
    cmap_diff: str = 'RdBu_r',
):
    """
    Three-panel heatmap: Low Risk | High Risk | Difference.

    The difference panel highlights where cross-modal patterns diverge
    between risk groups. Optionally overlays significance markers from
    per-prototype rank analysis.

    Args:
        attention_by_patient: Dict pid → {'cross_modal_attention': [N, P],
                                          'risk_group': str, ...}
        pathway_names: Pathway names.
        prototype_names: Prototype names.
        output_dir: Output directory.
        significance_mask: Optional dict proto_idx → boolean array [P]
            indicating which pathways are significantly different for
            that prototype. From per-prototype rank analysis.
        top_n_pathways: Number of pathways to show.
        cmap_groups: Colormap for group averages.
        cmap_diff: Colormap for difference panel (diverging).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Compute per-group averages
    low_attn = _compute_group_average(attention_by_patient, group='Low Risk')
    high_attn = _compute_group_average(attention_by_patient, group='High Risk')

    if low_attn is None or high_attn is None:
        logger.warning("Insufficient data for cross-modal comparison")
        return

    diff_attn = high_attn - low_attn

    # Select top pathways by max difference or max attention
    combined_importance = np.maximum(
        np.abs(diff_attn).max(axis=0),
        np.maximum(low_attn.max(axis=0), high_attn.max(axis=0))
    )
    top_idx = np.argsort(combined_importance)[-top_n_pathways:][::-1]

    low_sel = low_attn[:, top_idx]
    high_sel = high_attn[:, top_idx]
    diff_sel = diff_attn[:, top_idx]
    sel_pw_names = [pathway_names[i] for i in top_idx]

    # Build significance overlay if available
    sig_overlay = None
    if significance_mask is not None:
        n_proto = low_sel.shape[0]
        sig_overlay = np.zeros_like(diff_sel, dtype=bool)
        for proto_idx in range(n_proto):
            if proto_idx in significance_mask:
                full_sig = significance_mask[proto_idx]
                sig_overlay[proto_idx, :] = full_sig[top_idx]

    # Figure setup
    n_pathways = len(sel_pw_names)
    n_proto = low_sel.shape[0]
    cell_h = max(0.5, 8 / n_proto)
    cell_w = max(0.3, 12 / n_pathways)
    fig_w = n_pathways * cell_w + 8  # Extra for labels and colorbars
    fig_h = n_proto * cell_h + 3

    fig = plt.figure(figsize=(min(fig_w, 30), min(fig_h, 12)))
    gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.15)

    # Shared scale for group heatmaps
    vmax_groups = max(low_sel.max(), high_sel.max())
    vmin_groups = 0

    # Diverging scale for difference
    vmax_diff = max(abs(diff_sel.min()), abs(diff_sel.max()))

    # Truncate pathway names
    display_pw = [
        n[:35] + '...' if len(n) > 35 else n for n in sel_pw_names
    ]

    # Panel 1: Low Risk
    ax1 = fig.add_subplot(gs[0])
    im1 = ax1.imshow(
        low_sel, aspect='auto', cmap=cmap_groups,
        vmin=vmin_groups, vmax=vmax_groups
    )
    ax1.set_title('Low Risk', fontsize=12, fontweight='bold', color=COLOR_LOW)
    ax1.set_yticks(range(n_proto))
    ax1.set_yticklabels(prototype_names, fontsize=9)
    ax1.set_xticks(range(n_pathways))
    ax1.set_xticklabels(display_pw, fontsize=7, rotation=90, ha='center')

    # Panel 2: High Risk
    ax2 = fig.add_subplot(gs[1])
    im2 = ax2.imshow(
        high_sel, aspect='auto', cmap=cmap_groups,
        vmin=vmin_groups, vmax=vmax_groups
    )
    ax2.set_title('High Risk', fontsize=12, fontweight='bold', color=COLOR_HIGH)
    ax2.set_yticks(range(n_proto))
    ax2.set_yticklabels([], fontsize=9)
    ax2.set_xticks(range(n_pathways))
    ax2.set_xticklabels(display_pw, fontsize=7, rotation=90, ha='center')

    # Panel 3: Difference (High - Low)
    ax3 = fig.add_subplot(gs[2])
    im3 = ax3.imshow(
        diff_sel, aspect='auto', cmap=cmap_diff,
        vmin=-vmax_diff, vmax=vmax_diff
    )
    ax3.set_title('Δ (High − Low)', fontsize=12, fontweight='bold')
    ax3.set_yticks(range(n_proto))
    ax3.set_yticklabels([], fontsize=9)
    ax3.set_xticks(range(n_pathways))
    ax3.set_xticklabels(display_pw, fontsize=7, rotation=90, ha='center')

    # Overlay significance markers
    if sig_overlay is not None:
        for i in range(n_proto):
            for j in range(n_pathways):
                if sig_overlay[i, j]:
                    ax3.text(
                        j, i, '★', ha='center', va='center',
                        fontsize=8, color='black', fontweight='bold'
                    )

    # Colorbar for difference panel
    cax = fig.add_subplot(gs[3])
    plt.colorbar(im3, cax=cax, label='Attention Δ')

    fig.suptitle(
        'Cross-Modal Attention: Prototypes → Pathways',
        fontsize=14, fontweight='bold', y=1.02
    )

    output_path = os.path.join(output_dir, 'cross_modal_comparison.pdf')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved cross-modal comparison to {output_path}")

    # Also save individual group heatmaps
    _render_single_heatmap(
        low_sel, sel_pw_names, prototype_names,
        'Cross-Modal Attention (Low Risk)',
        os.path.join(output_dir, 'cross_modal_low_risk.pdf'),
        cmap=cmap_groups
    )
    _render_single_heatmap(
        high_sel, sel_pw_names, prototype_names,
        'Cross-Modal Attention (High Risk)',
        os.path.join(output_dir, 'cross_modal_high_risk.pdf'),
        cmap=cmap_groups
    )


def plot_top_prototype_pathway_pairs(
    attention_by_patient: Dict[str, Dict],
    pathway_names: List[str],
    prototype_names: List[str],
    output_path: str,
    top_k: int = 25,
    title: str = 'Top Prototype-Pathway Attention Pairs',
    by_risk_group: bool = True,
):
    """
    Bar chart of highest-attention prototype-pathway pairs.

    Optionally shows separate bars for each risk group to highlight
    which pairs are differentially attended.

    Args:
        attention_by_patient: Dict pid → attention data.
        pathway_names: Pathway names.
        prototype_names: Prototype names.
        output_path: Save path.
        top_k: Number of top pairs.
        title: Plot title.
        by_risk_group: Show grouped bars for low/high risk.
    """
    if by_risk_group:
        low_attn = _compute_group_average(attention_by_patient, group='Low Risk')
        high_attn = _compute_group_average(attention_by_patient, group='High Risk')
        if low_attn is None or high_attn is None:
            by_risk_group = False

    avg_attn = _compute_group_average(attention_by_patient)
    if avg_attn is None:
        return

    n_proto, n_pathways = avg_attn.shape

    # Find top pairs by overall average
    flat_idx = np.argsort(avg_attn.ravel())[-top_k:][::-1]
    pairs = []
    for idx in flat_idx:
        pi, pj = np.unravel_index(idx, (n_proto, n_pathways))
        pw_name = pathway_names[pj]
        pr_name = prototype_names[pi]
        label = f"{pr_name}: {pw_name[:30]}{'...' if len(pw_name) > 30 else ''}"

        entry = {
            'proto_idx': pi,
            'pathway_idx': pj,
            'label': label,
            'attention': avg_attn[pi, pj],
        }
        if by_risk_group:
            entry['low_risk'] = low_attn[pi, pj]
            entry['high_risk'] = high_attn[pi, pj]
        pairs.append(entry)

    # Plot
    fig_height = max(8, top_k * 0.4)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    y_pos = np.arange(len(pairs))
    labels = [p['label'] for p in pairs]

    if by_risk_group:
        width = 0.35
        bars_low = ax.barh(
            y_pos + width / 2,
            [p['low_risk'] for p in pairs],
            width, label='Low Risk', color=COLOR_LOW, alpha=0.8
        )
        bars_high = ax.barh(
            y_pos - width / 2,
            [p['high_risk'] for p in pairs],
            width, label='High Risk', color=COLOR_HIGH, alpha=0.8
        )
        ax.legend(fontsize=9, loc='lower right')
    else:
        ax.barh(
            y_pos,
            [p['attention'] for p in pairs],
            color='steelblue', alpha=0.8
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Mean Attention Weight', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    output_path = _ensure_pdf(output_path)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved top pairs plot to {output_path}")


# ---- Internal helpers ----

def _compute_group_average(
    attention_by_patient: Dict[str, Dict],
    group: Optional[str] = None,
    key: str = 'cross_modal_attention'
) -> Optional[np.ndarray]:
    """
    Average cross-modal attention matrices across patients.

    Args:
        attention_by_patient: Patient attention data.
        group: If provided, only average patients in this risk group.
        key: Key for the attention matrix.

    Returns:
        Averaged matrix [N, P] or None if no data.
    """
    matrices = []
    for pid, data in attention_by_patient.items():
        if key not in data:
            continue
        if group is not None and data.get('risk_group') != group:
            continue
        matrices.append(np.asarray(data[key]))

    if not matrices:
        return None

    return np.mean(matrices, axis=0)


def _select_top_pathways(
    attn_matrix: np.ndarray,
    pathway_names: List[str],
    top_n: int
) -> Tuple[np.ndarray, List[str]]:
    """Select top pathways by max attention across prototypes."""
    max_attn = attn_matrix.max(axis=0)  # [P]
    top_idx = np.argsort(max_attn)[-top_n:][::-1]

    return attn_matrix[:, top_idx], [pathway_names[i] for i in top_idx]


def _render_single_heatmap(
    attn: np.ndarray,
    pathway_names: List[str],
    prototype_names: List[str],
    title: str,
    output_path: str,
    figsize: Optional[Tuple] = None,
    cmap: str = 'YlOrRd',
):
    """Render a single attention heatmap."""
    n_proto, n_pathways = attn.shape

    if figsize is None:
        fig_w = max(8, n_pathways * 0.4 + 3)
        fig_h = max(4, n_proto * 0.6 + 2)
        figsize = (min(fig_w, 20), min(fig_h, 10))

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(attn, aspect='auto', cmap=cmap)
    plt.colorbar(im, ax=ax, shrink=0.8, label='Attention Weight')

    display_pw = [
        n[:35] + '...' if len(n) > 35 else n for n in pathway_names
    ]
    ax.set_xticks(range(n_pathways))
    ax.set_xticklabels(display_pw, fontsize=7, rotation=90, ha='center')
    ax.set_yticks(range(n_proto))
    ax.set_yticklabels(prototype_names, fontsize=9)

    ax.set_xlabel('Pathways', fontsize=11)
    ax.set_ylabel('Prototypes', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')

    plt.tight_layout()
    output_path = _ensure_pdf(output_path)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved heatmap to {output_path}")


def _ensure_pdf(path):
    base, ext = os.path.splitext(path)
    if ext.lower() != '.pdf':
        return base + '.pdf'
    return path