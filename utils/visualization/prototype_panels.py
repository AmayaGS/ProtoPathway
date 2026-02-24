"""
Prototype Visualization Panels.

Publication-quality figures for prototype interpretability:

    1. Prototype importance bar charts — shows the top-K most important
       prototypes by gate weight (Signal E) or fusion-attended weight
       (Signal H). Three panels: overall, high-risk, low-risk.

    2. Prototype exemplar patches — for each top prototype, extracts the
       most similar tissue patches (by cosine similarity from the WSI
       encoder). Creates a grid figure for the paper.

    3. Per-overlay exemplar strips — standalone figures showing the
       representative patches for each prototype visible in a spatial
       overlay, separated from the overlay itself for cleaner layout.

Data sources:
    - patch_assignments['gate_weights'] [K] — Signal E
    - fusion_gate_weights [K] — Signal H
    - patch_assignments['similarities'] [P, K] — cosine sim (pre-softmax)
    - patch_assignments['hard_assignments'] [P] — prototype labels
    - WSI features .pt files — coords for patch extraction
    - WSI .svs files or canvas — actual patch images

Usage:
    from utils.visualization.prototype_panels import (
        plot_prototype_importance,
        plot_prototype_exemplars,
        plot_overlay_exemplar_strip,
    )
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize

logger = logging.getLogger(__name__)

# Try importing OpenSlide (optional dependency for WSI reading)
try:
    import openslide
    HAS_OPENSLIDE = True
except ImportError:
    HAS_OPENSLIDE = False

# ── Palette ──────────────────────────────────────────────────────────
COLOR_LOW = '#2196F3'
COLOR_HIGH = '#E53935'
COLOR_OVERALL = '#616161'

_FORMATS = ['pdf', 'svg', 'png']


# =====================================================================
# 1. Prototype importance bar charts
# =====================================================================

def plot_prototype_importance(
    attention_by_patient: Dict[str, Dict],
    output_dir: str,
    top_k: int = 5,
    use_fusion_gate: bool = True,
    dpi: int = 300,
):
    """
    Bar charts of prototype importance: overall, high-risk, low-risk.

    For each group, averages the gate weights across patients and shows
    the top-K prototypes. Produces three separate figures (one per group)
    saved in PDF, SVG, and PNG.

    Args:
        attention_by_patient: Dict pid -> {
            'patch_assignments': {'gate_weights': [K], ...},
            'fusion_gate_weights': [K],  # optional
            'risk_group': str,
        }
        output_dir: Output directory for figures.
        top_k: Number of top prototypes to show.
        use_fusion_gate: If True, use Signal H (fusion_gate_weights)
            when available; otherwise fall back to Signal E (gate_weights).
        dpi: Output DPI for PNG.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect gate weights by risk group
    groups = {'Overall': [], 'High Risk': [], 'Low Risk': []}

    for pid, attn in attention_by_patient.items():
        weights = _get_importance_weights(attn, use_fusion_gate)
        if weights is None:
            continue

        risk = attn.get('risk_group', 'Unknown')
        groups['Overall'].append(weights)
        if 'High' in risk:
            groups['High Risk'].append(weights)
        elif 'Low' in risk:
            groups['Low Risk'].append(weights)

    # Generate one figure per group
    for group_name, weight_list in groups.items():
        if len(weight_list) == 0:
            logger.warning(f"No patients in {group_name}, skipping")
            continue

        mean_weights = np.stack(weight_list).mean(axis=0)  # [K]
        _plot_importance_bars(
            mean_weights, group_name, top_k, output_dir, dpi,
            use_fusion_gate,
        )


def _get_importance_weights(
    attn: Dict, use_fusion_gate: bool
) -> Optional[np.ndarray]:
    """Extract gate weights from patient attention dict."""
    if use_fusion_gate and 'fusion_gate_weights' in attn:
        w = np.asarray(attn['fusion_gate_weights'])
        if w.ndim > 0 and len(w) > 0:
            return w

    pa = attn.get('patch_assignments', {})
    w = pa.get('gate_weights')
    if w is not None:
        w = np.asarray(w)
        if w.ndim > 0 and len(w) > 0:
            return w

    return None


def _plot_importance_bars(
    mean_weights: np.ndarray,
    group_name: str,
    top_k: int,
    output_dir: Path,
    dpi: int,
    use_fusion_gate: bool,
):
    """Single bar chart for one group."""
    K = len(mean_weights)
    top_k = min(top_k, K)

    # Sort descending, take top-K
    order = np.argsort(mean_weights)[::-1][:top_k]
    values = mean_weights[order]
    labels = [f'Proto {i}' for i in order]

    # Color by group
    color_map = {
        'Overall': COLOR_OVERALL,
        'High Risk': COLOR_HIGH,
        'Low Risk': COLOR_LOW,
    }
    color = color_map.get(group_name, COLOR_OVERALL)

    fig, ax = plt.subplots(figsize=(max(4, top_k * 0.9), 3.5))
    bars = ax.barh(
        range(top_k), values, color=color, alpha=0.85,
        edgecolor='white', linewidth=0.5,
    )
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()

    signal_label = 'Fusion Gate (H)' if use_fusion_gate else 'WSI Gate (E)'
    ax.set_xlabel(f'Mean {signal_label} Weight', fontsize=10)
    ax.set_title(
        f'Top {top_k} Prototypes — {group_name}',
        fontsize=11, fontweight='bold',
    )
    ax.grid(axis='x', alpha=0.3)

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f'{val:.4f}', ha='left', va='center', fontsize=8,
        )

    plt.tight_layout()

    slug = group_name.lower().replace(' ', '_')
    base = output_dir / f'prototype_importance_{slug}'
    for fmt in _FORMATS:
        fig.savefig(
            str(base) + f'.{fmt}', dpi=dpi,
            bbox_inches='tight', facecolor='white',
        )
    plt.close(fig)
    logger.info(f"Saved prototype importance ({group_name}) to {base}.*")


# =====================================================================
# 2. Prototype exemplar patches
# =====================================================================

def extract_exemplar_patches(
    patient_id: str,
    attention_data: Dict,
    coords: np.ndarray,
    canvas: np.ndarray,
    downsample: int,
    coord_spacing: int,
    top_k_protos: int = 5,
    n_patches_per_proto: int = 5,
    use_fusion_gate: bool = True,
    wsi_path: Optional[str] = None,
    patch_size: int = 256,
) -> Dict[int, List[np.ndarray]]:
    """
    Extract the most similar patch images for the top-K prototypes.

    Uses the cosine similarity matrix from the WSI encoder to find the
    patches most similar to each prototype. Reads patch images from the
    WSI file if available, otherwise crops from the pre-built canvas.

    Args:
        patient_id: Patient ID (for logging).
        attention_data: Patient attention dict with patch_assignments.
        coords: [N, 2] patch coordinates at level 0.
        canvas: Pre-built RGB canvas [H, W, 3] (fallback if no WSI).
        downsample: Downsample factor used for canvas.
        coord_spacing: Patch footprint in coordinate space.
        top_k_protos: Number of top prototypes to extract for.
        n_patches_per_proto: Number of exemplar patches per prototype.
        use_fusion_gate: Use fusion gate for importance ranking.
        wsi_path: Optional path to WSI for high-res patch extraction.
        patch_size: Patch size at extraction level.

    Returns:
        Dict[proto_idx -> List[np.ndarray]] where each array is an
        RGB patch image. Keys are the top-K prototype indices sorted
        by importance.
    """
    pa = attention_data.get('patch_assignments', {})
    similarities = pa.get('similarities')  # [P, K] cosine similarity

    if similarities is None:
        logger.warning(
            f"{patient_id}: no similarities in attention data, "
            f"cannot extract exemplar patches"
        )
        return {}

    similarities = np.asarray(similarities)
    if similarities.ndim != 2:
        logger.warning(f"{patient_id}: unexpected similarities shape")
        return {}

    # Determine top-K prototypes by importance
    weights = _get_importance_weights(attention_data, use_fusion_gate)
    if weights is None:
        # Fall back to counting assignments
        hard = np.asarray(pa.get('hard_assignments', pa.get('assignments', [])))
        if len(hard) == 0:
            return {}
        from collections import Counter
        counts = Counter(hard.astype(int).tolist())
        K = similarities.shape[1]
        weights = np.array([counts.get(i, 0) for i in range(K)], dtype=float)

    top_k_protos = min(top_k_protos, len(weights))
    top_proto_indices = np.argsort(weights)[::-1][:top_k_protos]

    # Try WSI-based extraction first
    use_wsi = wsi_path is not None and HAS_OPENSLIDE and os.path.exists(wsi_path)

    if use_wsi:
        wsi = openslide.OpenSlide(wsi_path)
    else:
        wsi = None

    effective_patch = int(coord_spacing / downsample)
    exemplars = {}

    for proto_idx in top_proto_indices:
        proto_idx = int(proto_idx)
        # Sort patches by similarity to this prototype (descending)
        sims = similarities[:, proto_idx]
        top_patch_indices = np.argsort(sims)[::-1][:n_patches_per_proto]

        patches = []
        for patch_i in top_patch_indices:
            patch_img = _extract_single_patch(
                patch_i, coords, canvas, downsample,
                coord_spacing, effective_patch, wsi, patch_size,
            )
            if patch_img is not None:
                patches.append(patch_img)

        exemplars[proto_idx] = patches

    if wsi is not None:
        wsi.close()

    logger.info(
        f"Extracted exemplars for {len(exemplars)} prototypes "
        f"({patient_id})"
    )
    return exemplars


def _extract_single_patch(
    patch_idx: int,
    coords: np.ndarray,
    canvas: np.ndarray,
    downsample: int,
    coord_spacing: int,
    effective_patch: int,
    wsi,  # openslide.OpenSlide or None
    patch_size: int,
) -> Optional[np.ndarray]:
    """Extract a single patch image, preferring WSI over canvas."""
    x, y = int(coords[patch_idx, 0]), int(coords[patch_idx, 1])

    if wsi is not None:
        try:
            from PIL import Image as PILImage
            region = wsi.read_region((x, y), 0, (coord_spacing, coord_spacing))
            patch_rgb = np.array(region.convert('RGB'))
            # Resize to a reasonable display size (e.g., 256x256)
            target = min(256, coord_spacing)
            if patch_rgb.shape[0] != target:
                pil = PILImage.fromarray(patch_rgb)
                pil = pil.resize((target, target), PILImage.LANCZOS)
                patch_rgb = np.array(pil)
            return patch_rgb
        except Exception as e:
            logger.debug(f"WSI patch extraction failed at ({x},{y}): {e}")

    # Fallback: crop from canvas
    cx = int(x / downsample)
    cy = int(y / downsample)
    h, w = canvas.shape[:2]
    y1 = max(0, cy)
    x1 = max(0, cx)
    y2 = min(h, cy + effective_patch)
    x2 = min(w, cx + effective_patch)

    crop = canvas[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return None

    # Resize to uniform size
    from PIL import Image as PILImage
    target = min(256, effective_patch)
    pil = PILImage.fromarray(crop)
    pil = pil.resize((target, target), PILImage.LANCZOS)
    return np.array(pil)


def plot_prototype_exemplars(
    exemplars: Dict[int, List[np.ndarray]],
    importance_weights: np.ndarray,
    output_dir: str,
    slide_id: str = '',
    proto_colors: Optional[Dict[int, Tuple]] = None,
    dpi: int = 300,
    title_prefix: str = '',
):
    """
    Grid figure showing exemplar patches for each top prototype.

    Layout: one row per prototype, columns are exemplar patches.
    Left margin shows prototype index, importance weight, and a
    colored indicator matching the spatial overlay.

    Args:
        exemplars: Dict[proto_idx -> List[patch_images]].
        importance_weights: [K] gate weights for annotation.
        output_dir: Save directory.
        slide_id: Slide ID for filename.
        proto_colors: Optional color dict from render_prototype_overlay.
        dpi: Output DPI.
        title_prefix: Optional prefix for the title.
    """
    if not exemplars:
        logger.warning("No exemplars to plot")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort prototypes by importance
    sorted_protos = sorted(
        exemplars.keys(),
        key=lambda p: float(importance_weights[p]) if p < len(importance_weights) else 0,
        reverse=True,
    )

    n_protos = len(sorted_protos)
    max_patches = max(len(v) for v in exemplars.values()) if exemplars else 0
    if max_patches == 0:
        return

    # Figure layout
    patch_display_size = 1.8  # inches per patch cell
    label_width = 2.0
    fig_w = label_width + max_patches * patch_display_size
    fig_h = n_protos * patch_display_size + 0.8

    fig, axes = plt.subplots(
        n_protos, max_patches,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )

    for row, proto_idx in enumerate(sorted_protos):
        patches = exemplars[proto_idx]
        weight = (
            float(importance_weights[proto_idx])
            if proto_idx < len(importance_weights) else 0.0
        )

        for col in range(max_patches):
            ax = axes[row, col]
            ax.axis('off')

            if col < len(patches) and patches[col] is not None:
                img = patches[col]
                ax.imshow(img)

                # Colored border if we have prototype colors
                if proto_colors and proto_idx in proto_colors:
                    color_rgb = proto_colors[proto_idx]
                    color_norm = tuple(c / 255.0 for c in color_rgb)
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_color(color_norm)
                        spine.set_linewidth(3)

        # Row label (leftmost cell annotation)
        label = f'Proto {proto_idx}\n(w={weight:.4f})'
        axes[row, 0].set_title(
            label, fontsize=8, fontweight='bold',
            loc='left', pad=2,
        )

    title = f'{title_prefix}Prototype Exemplar Patches'
    if slide_id:
        title += f' — {slide_id}'
    fig.suptitle(title, fontsize=11, fontweight='bold', y=1.01)

    plt.tight_layout()

    base = output_dir / f'{slide_id}_prototype_exemplars' if slide_id else output_dir / 'prototype_exemplars'
    for fmt in _FORMATS:
        fig.savefig(
            str(base) + f'.{fmt}', dpi=dpi,
            bbox_inches='tight', facecolor='white',
        )
    plt.close(fig)
    logger.info(f"Saved prototype exemplars to {base}.*")


# =====================================================================
# 3. Per-overlay exemplar strips
# =====================================================================

def plot_overlay_exemplar_strip(
    exemplars: Dict[int, List[np.ndarray]],
    visible_protos: List[int],
    importance_weights: np.ndarray,
    overlay_title: str,
    output_dir: str,
    slide_id: str = '',
    proto_colors: Optional[Dict[int, Tuple]] = None,
    n_patches: int = 4,
    dpi: int = 300,
):
    """
    Standalone strip showing exemplar patches for prototypes visible
    in a specific overlay panel.

    Designed to be placed next to the corresponding spatial overlay in
    a composite figure. Each prototype gets a column of patches with a
    colored header matching the overlay.

    Args:
        exemplars: Full exemplar dict from extract_exemplar_patches.
        visible_protos: Prototype indices present in this overlay.
        importance_weights: [K] gate weights.
        overlay_title: Title slug for filename (e.g., 'prototype_assignments').
        output_dir: Save directory.
        slide_id: Slide ID for filename.
        proto_colors: Color dict from overlay rendering.
        n_patches: Max patches to show per prototype.
        dpi: Output DPI.
    """
    if not exemplars or not visible_protos:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort by importance
    sorted_protos = sorted(
        visible_protos,
        key=lambda p: float(importance_weights[p]) if p < len(importance_weights) else 0,
        reverse=True,
    )

    # Limit to prototypes that have exemplars
    sorted_protos = [p for p in sorted_protos if p in exemplars and len(exemplars[p]) > 0]
    if not sorted_protos:
        return

    n_cols = len(sorted_protos)
    n_rows = min(n_patches, max(len(exemplars[p]) for p in sorted_protos))

    cell_size = 1.5
    fig_w = n_cols * cell_size + 0.5
    fig_h = n_rows * cell_size + 1.2

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )

    for col, proto_idx in enumerate(sorted_protos):
        patches = exemplars[proto_idx][:n_rows]

        # Column header
        weight = (
            float(importance_weights[proto_idx])
            if proto_idx < len(importance_weights) else 0.0
        )
        header = f'P{proto_idx}\n({weight:.3f})'

        if proto_colors and proto_idx in proto_colors:
            color_rgb = proto_colors[proto_idx]
            color_norm = tuple(c / 255.0 for c in color_rgb)
        else:
            color_norm = (0.4, 0.4, 0.4)

        axes[0, col].set_title(
            header, fontsize=7, fontweight='bold',
            color=color_norm, pad=3,
        )

        for row in range(n_rows):
            ax = axes[row, col]
            ax.axis('off')

            if row < len(patches) and patches[row] is not None:
                ax.imshow(patches[row])
                # Thin colored border
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color(color_norm)
                    spine.set_linewidth(2)

    slug = overlay_title.lower().replace(' ', '_').replace('(', '').replace(')', '')
    fig.suptitle(
        f'Exemplar Patches — {overlay_title}',
        fontsize=9, fontweight='bold', y=1.02,
    )

    plt.tight_layout()

    name = f'{slide_id}_{slug}_exemplars' if slide_id else f'{slug}_exemplars'
    base = output_dir / name
    for fmt in _FORMATS:
        fig.savefig(
            str(base) + f'.{fmt}', dpi=dpi,
            bbox_inches='tight', facecolor='white',
        )
    plt.close(fig)
    logger.info(f"Saved overlay exemplar strip to {base}.*")


# =====================================================================
# 4. Cohort-level prototype exemplars (across patients)
# =====================================================================

def plot_cohort_prototype_exemplars(
    attention_by_patient: Dict[str, Dict],
    wsi_features_dir: str,
    output_dir: str,
    top_k_protos: int = 5,
    n_patches_per_proto: int = 8,
    wsi_dir: Optional[str] = None,
    downsample: int = 4,
    patch_size: int = 256,
    use_fusion_gate: bool = True,
    dpi: int = 300,
):
    """
    Cohort-level exemplar visualization.

    For each of the top-K most important prototypes (averaged across
    all patients), finds the globally most similar patches across the
    cohort. Produces three figures: overall, high-risk, low-risk.

    This is the population-level analogue of the per-patient exemplars.

    Args:
        attention_by_patient: Full attention dict.
        wsi_features_dir: Directory with .pt files.
        output_dir: Output directory.
        top_k_protos: Number of top prototypes.
        n_patches_per_proto: Patches per prototype to show.
        wsi_dir: Optional WSI directory for high-res extraction.
        downsample: Downsample factor.
        patch_size: Patch extraction size.
        use_fusion_gate: Use Signal H for importance.
        dpi: Output DPI.
    """
    import torch
    from utils.visualization.spatial_heatmaps import (
        infer_coord_spacing,
        build_canvas_from_wsi,
        build_canvas_from_coords,
        _find_wsi_for_slide,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = {
        'overall': {'pids': [], 'weights': []},
        'high_risk': {'pids': [], 'weights': []},
        'low_risk': {'pids': [], 'weights': []},
    }

    # Collect importance weights
    for pid, attn in attention_by_patient.items():
        w = _get_importance_weights(attn, use_fusion_gate)
        if w is None:
            continue

        risk = attn.get('risk_group', 'Unknown')
        groups['overall']['pids'].append(pid)
        groups['overall']['weights'].append(w)

        if 'High' in risk:
            groups['high_risk']['pids'].append(pid)
            groups['high_risk']['weights'].append(w)
        elif 'Low' in risk:
            groups['low_risk']['pids'].append(pid)
            groups['low_risk']['weights'].append(w)

    for group_name, group_data in groups.items():
        if len(group_data['weights']) == 0:
            continue

        mean_weights = np.stack(group_data['weights']).mean(axis=0)
        top_proto_indices = np.argsort(mean_weights)[::-1][:top_k_protos]

        logger.info(
            f"Collecting cohort exemplars for {group_name} "
            f"({len(group_data['pids'])} patients, "
            f"protos: {top_proto_indices.tolist()})"
        )

        # Collect (similarity, patient_id, patch_idx) for each prototype
        proto_candidates = {int(p): [] for p in top_proto_indices}

        for pid in group_data['pids']:
            attn = attention_by_patient[pid]
            pa = attn.get('patch_assignments', {})
            sims = pa.get('similarities')
            if sims is None:
                continue
            sims = np.asarray(sims)

            for proto_idx in top_proto_indices:
                proto_idx = int(proto_idx)
                if proto_idx >= sims.shape[1]:
                    continue
                col = sims[:, proto_idx]
                # Keep top candidates from this patient
                top_n = min(n_patches_per_proto, len(col))
                best_patches = np.argsort(col)[::-1][:top_n]
                for pi in best_patches:
                    proto_candidates[proto_idx].append(
                        (float(col[pi]), pid, int(pi))
                    )

        # Sort globally and take top-N per prototype
        exemplars = {}
        for proto_idx in [int(p) for p in top_proto_indices]:
            candidates = sorted(
                proto_candidates[proto_idx], key=lambda x: -x[0]
            )[:n_patches_per_proto]

            patches = []
            for sim_val, pid, patch_i in candidates:
                patch_img = _extract_patch_from_pt(
                    pid, patch_i, wsi_features_dir, wsi_dir,
                    downsample, patch_size,
                )
                if patch_img is not None:
                    patches.append(patch_img)

            exemplars[proto_idx] = patches

        # Plot
        plot_prototype_exemplars(
            exemplars=exemplars,
            importance_weights=mean_weights,
            output_dir=str(output_dir),
            slide_id=f'cohort_{group_name}',
            dpi=dpi,
            title_prefix=f'{group_name.replace("_", " ").title()} — ',
        )


def _extract_patch_from_pt(
    patient_id: str,
    patch_idx: int,
    wsi_features_dir: str,
    wsi_dir: Optional[str],
    downsample: int,
    patch_size: int,
) -> Optional[np.ndarray]:
    """Extract a single patch for cohort-level exemplar viz."""
    import torch
    from utils.visualization.spatial_heatmaps import (
        infer_coord_spacing,
        build_canvas_from_coords,
        _find_wsi_for_slide,
    )

    pt_path = os.path.join(wsi_features_dir, f'{patient_id}.pt')
    if not os.path.exists(pt_path):
        return None

    data = torch.load(pt_path, map_location='cpu', weights_only=False)
    if isinstance(data, torch.Tensor) or 'coords' not in data:
        return None

    coords = np.asarray(data['coords'])
    if patch_idx >= len(coords):
        return None

    cs = infer_coord_spacing(coords)
    x, y = int(coords[patch_idx, 0]), int(coords[patch_idx, 1])

    # Try WSI
    if wsi_dir:
        slide_info = data.get('slide_info', [])
        for si in slide_info:
            sid = si.get('slide_id', '')
            wsi_path = _find_wsi_for_slide(sid, wsi_dir, None)
            if wsi_path and HAS_OPENSLIDE:
                try:
                    wsi = openslide.OpenSlide(wsi_path)
                    from PIL import Image as PILImage
                    region = wsi.read_region((x, y), 0, (cs, cs))
                    patch_rgb = np.array(region.convert('RGB'))
                    wsi.close()
                    target = min(256, cs)
                    if patch_rgb.shape[0] != target:
                        pil = PILImage.fromarray(patch_rgb)
                        pil = pil.resize((target, target), PILImage.LANCZOS)
                        patch_rgb = np.array(pil)
                    return patch_rgb
                except Exception:
                    pass

    # Fallback: coordinate-based placeholder (grey with coords)
    # Not ideal but at least shows something
    effective = int(cs / downsample)
    canvas = build_canvas_from_coords(
        coords[patch_idx:patch_idx+1], patch_size,
        downsample=downsample, coord_spacing=cs,
    )
    if canvas.size > 0:
        from PIL import Image as PILImage
        pil = PILImage.fromarray(canvas)
        pil = pil.resize((256, 256), PILImage.LANCZOS)
        return np.array(pil)

    return None