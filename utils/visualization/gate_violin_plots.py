"""
Violin/box plots for gate importance distributions by risk group.

Shows the full distribution of gate weights (pathway, prototype)
across patients, split by risk group. This reveals whether importance
differences are driven by consistent shifts or outliers.

Consumes patient-level data from ImportanceAnalyzer.
"""

import os
import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logger = logging.getLogger(__name__)

COLOR_LOW = '#2196F3'
COLOR_HIGH = '#E53935'


def plot_gate_importance_violin(
    analyzer,
    title: str,
    output_path: str,
    n: int = 15,
    sort_by: str = 'effect_size',
    show_significance: bool = True,
    figsize_per_entity: float = 0.6,
    min_fig_width: float = 10,
    orientation: str = 'horizontal',
):
    """
    Violin plot of gate importance distributions by risk group.

    For each entity, shows paired violin/box plots comparing the
    importance distribution in low vs high risk patients.

    Args:
        analyzer: ImportanceAnalyzer instance (with data loaded).
        title: Plot title.
        output_path: Path to save figure.
        n: Number of top entities to show (by effect size).
        sort_by: 'effect_size' or 'rank_difference'.
        show_significance: Mark FDR-significant entities.
        figsize_per_entity: Height per entity row.
        min_fig_width: Minimum figure width.
        orientation: 'horizontal' for entities on y-axis (recommended
            for long pathway names), 'vertical' for x-axis.
    """
    # Get results for ordering
    results = analyzer.rank_analysis()

    if sort_by == 'effect_size':
        top_entities = results.sort_values(
            'abs_rank_biserial_r', ascending=False
        ).head(n)['entity'].tolist()
    else:
        top_entities = results.sort_values(
            'rank_difference', key=abs, ascending=False
        ).head(n)['entity'].tolist()

    # Get patient-level data
    patient_df = analyzer.get_patient_level_data()

    # Build long-form dataframe for selected entities
    entity_idx = {name: i for i, name in enumerate(analyzer.entity_names)}
    selected_idx = [entity_idx[e] for e in top_entities if e in entity_idx]
    selected_names = [e for e in top_entities if e in entity_idx]

    # Significance lookup
    sig_map = dict(zip(results['entity'], results['significant']))
    effect_map = dict(zip(results['entity'], results['rank_biserial_r']))

    if orientation == 'horizontal':
        _plot_horizontal_violin(
            patient_df, selected_names, sig_map, effect_map,
            title, output_path, figsize_per_entity, min_fig_width,
            show_significance
        )
    else:
        _plot_vertical_violin(
            patient_df, selected_names, sig_map, effect_map,
            title, output_path, figsize_per_entity, show_significance
        )


def _plot_horizontal_violin(
    patient_df, entity_names, sig_map, effect_map,
    title, output_path, figsize_per_entity, min_fig_width,
    show_significance
):
    """Horizontal violin: entities on y-axis, importance on x-axis."""
    n_entities = len(entity_names)
    fig_height = max(6, n_entities * figsize_per_entity)
    fig, ax = plt.subplots(figsize=(min_fig_width, fig_height))

    positions = np.arange(n_entities)
    width = 0.35

    low_data = []
    high_data = []
    low_mask = patient_df['risk_group'] == 'Low Risk'
    high_mask = patient_df['risk_group'] == 'High Risk'

    for entity in entity_names:
        low_data.append(patient_df.loc[low_mask, entity].values)
        high_data.append(patient_df.loc[high_mask, entity].values)

    # Violin plots — low risk
    vp_low = ax.violinplot(
        low_data,
        positions=positions - width / 2,
        vert=False,
        showmeans=True,
        showextrema=False,
        widths=width * 0.9
    )
    _style_violins(vp_low, COLOR_LOW, alpha=0.6)

    # Violin plots — high risk
    vp_high = ax.violinplot(
        high_data,
        positions=positions + width / 2,
        vert=False,
        showmeans=True,
        showextrema=False,
        widths=width * 0.9
    )
    _style_violins(vp_high, COLOR_HIGH, alpha=0.6)

    # Labels and significance markers
    labels = []
    for i, entity in enumerate(entity_names):
        label = entity[:45] + '...' if len(entity) > 45 else entity
        if show_significance and sig_map.get(entity, False):
            r = effect_map.get(entity, 0)
            label = f"★ {label} (r={r:.2f})"
        labels.append(label)

    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()

    ax.set_xlabel('Importance Score', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_LOW, alpha=0.6, label='Low Risk'),
        mpatches.Patch(facecolor=COLOR_HIGH, alpha=0.6, label='High Risk'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    output_path = _ensure_pdf(output_path)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved violin plot to {output_path}")


def _plot_vertical_violin(
    patient_df, entity_names, sig_map, effect_map,
    title, output_path, figsize_per_entity, show_significance
):
    """Vertical violin: entities on x-axis (better for short names like prototypes)."""
    n_entities = len(entity_names)
    fig_width = max(8, n_entities * figsize_per_entity + 2)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    positions = np.arange(n_entities)
    width = 0.35

    low_data = []
    high_data = []
    low_mask = patient_df['risk_group'] == 'Low Risk'
    high_mask = patient_df['risk_group'] == 'High Risk'

    for entity in entity_names:
        low_data.append(patient_df.loc[low_mask, entity].values)
        high_data.append(patient_df.loc[high_mask, entity].values)

    vp_low = ax.violinplot(
        low_data,
        positions=positions - width / 2,
        vert=True,
        showmeans=True,
        showextrema=False,
        widths=width * 0.9
    )
    _style_violins(vp_low, COLOR_LOW, alpha=0.6)

    vp_high = ax.violinplot(
        high_data,
        positions=positions + width / 2,
        vert=True,
        showmeans=True,
        showextrema=False,
        widths=width * 0.9
    )
    _style_violins(vp_high, COLOR_HIGH, alpha=0.6)

    # Significance markers
    for i, entity in enumerate(entity_names):
        if show_significance and sig_map.get(entity, False):
            y_max = max(
                np.max(low_data[i]) if len(low_data[i]) else 0,
                np.max(high_data[i]) if len(high_data[i]) else 0
            )
            ax.text(
                i, y_max * 1.05, '★',
                ha='center', va='bottom', fontsize=12, color='#FF6F00'
            )

    labels = [e[:20] + '...' if len(e) > 20 else e for e in entity_names]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9, rotation=45, ha='right')

    ax.set_ylabel('Importance Score', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    legend_elements = [
        mpatches.Patch(facecolor=COLOR_LOW, alpha=0.6, label='Low Risk'),
        mpatches.Patch(facecolor=COLOR_HIGH, alpha=0.6, label='High Risk'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.tight_layout()
    output_path = _ensure_pdf(output_path)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved violin plot to {output_path}")


def plot_assignment_frequency_violin(
    analyzer,
    title: str = 'Prototype Assignment Frequency by Risk Group',
    output_path: str = 'assignment_frequency_violin.pdf',
    **kwargs
):
    """
    Convenience wrapper for prototype assignment frequency violins.

    Uses vertical orientation (better for prototype labels).
    """
    plot_gate_importance_violin(
        analyzer=analyzer,
        title=title,
        output_path=output_path,
        orientation='vertical',
        n=kwargs.pop('n', 20),
        **kwargs
    )


def _style_violins(vp, color, alpha=0.6):
    """Apply consistent styling to violin plot collection."""
    for body in vp['bodies']:
        body.set_facecolor(color)
        body.set_alpha(alpha)
        body.set_edgecolor('white')
        body.set_linewidth(0.5)
    if 'cmeans' in vp:
        vp['cmeans'].set_color(color)
        vp['cmeans'].set_linewidth(1.5)


def _ensure_pdf(path):
    base, ext = os.path.splitext(path)
    if ext.lower() != '.pdf':
        return base + '.pdf'
    return path