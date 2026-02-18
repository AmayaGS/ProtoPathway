"""
Publication-quality Kaplan-Meier survival curves.

Features:
    - 2-group (median split) and 4-group (quartile) stratification
    - Confidence intervals with censorship markers
    - Log-rank test (2-group) / multivariate log-rank (4-group)
    - RMST for censored-only groups
    - Scientific notation p-value with significance asterisk
    - PDF output at 300dpi with transparent background
"""

import os
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

logger = logging.getLogger(__name__)

from utils.analysis.fold_aggregation import stratify_risk_scores, GROUP_NAMES_2, GROUP_NAMES_4

# Consistent style
COLORS_2 = ['#2ca02c', '#d62728']               # green, red
COLORS_4 = ['#1a6e1a', '#66bb6a', '#ff9800', '#d62728']  # dk green, lt green, orange, red


def plot_kaplan_meier(
    times: np.ndarray,
    events: np.ndarray,
    risk_scores: np.ndarray,
    output_path: str,
    n_groups: int = 2,
    figsize: Tuple[int, int] = (10, 7),
    show_ci: bool = True,
    show_censors: bool = True,
    title: Optional[str] = None,
):
    """
    Plot publication-quality Kaplan-Meier curves stratified by risk.

    Args:
        times: Survival times [n_patients].
        events: Event indicators (1=event, 0=censored) [n_patients].
        risk_scores: Predicted risk scores [n_patients].
        output_path: Path to save figure (PDF recommended).
        n_groups: Number of risk groups (2 or 4).
        figsize: Figure size.
        show_ci: Whether to show confidence intervals.
        show_censors: Whether to show censorship markers.
        title: Custom title. If None, uses p-value.

    Returns:
        Dict with 'p_value' and 'mean_survivals'.
    """
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test, multivariate_logrank_test
    except ImportError:
        logger.warning("lifelines not installed, skipping KM plot")
        return {}

    # Stratify into risk groups
    groups, group_names, colors = _stratify(risk_scores, n_groups)

    fig, ax = plt.subplots(figsize=figsize)

    kmf = KaplanMeierFitter()
    mean_survivals = {}

    for i, (name, color) in enumerate(zip(group_names, colors)):
        mask = groups == i
        n_group = mask.sum()
        if n_group == 0:
            continue

        kmf.fit(
            times[mask],
            events[mask],
            label=f"{name} (n={n_group})"
        )

        kmf.plot_survival_function(
            ax=ax,
            ci_show=show_ci,
            color=color,
            show_censors=show_censors,
            censor_styles={'ms': 6, 'marker': '|'},
            linewidth=2.0,
        )

        # Calculate mean survival
        mean_survivals[name] = _compute_mean_survival(
            times[mask], events[mask], kmf
        )

    # Log-rank test
    p_value = _compute_logrank(times, events, groups, n_groups)

    # Format title with p-value
    if title is None:
        p_str = _format_pvalue(p_value)
        sig_marker = '*' if p_value < 0.05 else ''
        title = f"Log-rank p = {p_str}{sig_marker}"

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (months)', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Custom legend with line handles
    legend_elements = [
        Line2D([0], [0], color=colors[i], lw=2, label=group_names[i])
        for i in range(len(group_names))
        if (groups == i).sum() > 0
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=10,
              framealpha=0.9)

    # Ensure PDF output
    output_path = _ensure_pdf(output_path)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    fig.savefig(
        output_path,
        bbox_inches='tight',
        transparent=True,
        dpi=300,
        metadata={'Creator': 'ProtoPathway'}
    )
    plt.close(fig)

    logger.info(f"Saved KM plot ({n_groups}-group) to {output_path}")

    return {
        'p_value': p_value,
        'mean_survivals': mean_survivals
    }


def plot_kaplan_meier_both(
    times: np.ndarray,
    events: np.ndarray,
    risk_scores: np.ndarray,
    output_dir: str,
    **kwargs
) -> Dict:
    """Convenience: generate both 2-group and 4-group KM plots."""
    results = {}
    results['2_group'] = plot_kaplan_meier(
        times, events, risk_scores,
        os.path.join(output_dir, 'kaplan_meier_2group.pdf'),
        n_groups=2, **kwargs
    )
    results['4_group'] = plot_kaplan_meier(
        times, events, risk_scores,
        os.path.join(output_dir, 'kaplan_meier_4group.pdf'),
        n_groups=4, **kwargs
    )
    return results


# ---- Internal helpers ----

def _stratify(risk_scores, n_groups):
    """Assign patients to risk groups (delegates to shared implementation)."""
    groups, group_names = stratify_risk_scores(risk_scores, n_groups)
    colors = COLORS_2 if n_groups == 2 else COLORS_4
    return groups, group_names, colors


def _compute_logrank(times, events, groups, n_groups):
    """Compute log-rank test p-value."""
    from lifelines.statistics import logrank_test, multivariate_logrank_test

    if n_groups == 2:
        result = logrank_test(
            times[groups == 0], times[groups == 1],
            events[groups == 0], events[groups == 1]
        )
        return result.p_value
    else:
        return multivariate_logrank_test(
            np.array(times, dtype=float),
            np.array(events, dtype=float),
            np.array(groups, dtype=float)
        ).p_value


def _compute_mean_survival(times, events, kmf):
    """Compute mean survival time (observed or RMST for censored groups)."""
    observed_times = times[events.astype(bool)]

    if len(observed_times) > 0:
        return {
            'mean': float(np.mean(observed_times)),
            'std': float(np.std(observed_times)),
            'method': 'observed'
        }
    else:
        # Use restricted mean survival time (area under KM curve)
        sf = kmf.survival_function_
        t = sf.index.values
        p = sf.values.flatten()
        rmst = float(np.trapz(p, t))
        return {
            'mean': rmst,
            'std': None,
            'method': 'RMST'
        }


def _format_pvalue(p):
    """Format p-value with scientific notation."""
    if p < 0.001:
        return f"{p:.2e}"
    elif p < 0.01:
        return f"{p:.4f}"
    else:
        return f"{p:.3f}"


def _ensure_pdf(path):
    """Ensure output path has .pdf extension."""
    base, ext = os.path.splitext(path)
    if ext.lower() != '.pdf':
        return base + '.pdf'
    return path