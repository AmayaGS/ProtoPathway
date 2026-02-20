"""
Diagnostic: Cross-modal attention entropy by risk group.

Tests whether the persistent blue dominance in the heatmap reflects a
real biological signal (low-risk = peaked attention, high-risk = diffuse)
rather than a normalization artifact.

For each patient and each prototype row, computes the Shannon entropy
of the softmax attention distribution. Higher entropy = more uniform
attention across pathways.

Usage:
    Add to fold_stratified_analysis.py or run standalone after pool_cv_results.
"""

import numpy as np
from scipy.stats import entropy, mannwhitneyu


def diagnose_crossmodal_entropy(
    attention_by_patient: dict,
    predictions,  # DataFrame with patient_id, risk_group
    logger=None,
):
    """
    Compare cross-modal attention entropy between risk groups.

    For each patient, computes mean entropy across prototype rows.
    Then tests whether high-risk patients have significantly different
    entropy than low-risk patients.

    Args:
        attention_by_patient: Dict pid → {'cross_modal_attention': [N, P],
                                          'risk_group': str, ...}
        predictions: DataFrame with patient_id and risk_group columns.

    Returns:
        Dict with diagnostic results.
    """
    risk_map = dict(
        zip(predictions['patient_id'], predictions['risk_group'])
    )

    low_entropies = []
    high_entropies = []

    low_per_proto = []  # [n_patients_low, n_protos]
    high_per_proto = []

    for pid, attn in attention_by_patient.items():
        if 'cross_modal_attention' not in attn:
            continue
        risk = risk_map.get(pid)
        if risk is None:
            continue

        cm = np.asarray(attn['cross_modal_attention'], dtype=np.float64)
        # cm is [n_protos, n_pathways], each row sums to ~1 (softmax)

        # Per-prototype entropy
        proto_entropies = []
        for row in cm:
            # Ensure valid probability distribution
            row_safe = np.clip(row, 1e-12, None)
            row_safe = row_safe / row_safe.sum()
            proto_entropies.append(entropy(row_safe))

        proto_entropies = np.array(proto_entropies)
        mean_ent = proto_entropies.mean()

        if risk == 'High Risk':
            high_entropies.append(mean_ent)
            high_per_proto.append(proto_entropies)
        elif risk == 'Low Risk':
            low_entropies.append(mean_ent)
            low_per_proto.append(proto_entropies)

    low_entropies = np.array(low_entropies)
    high_entropies = np.array(high_entropies)

    # Overall comparison
    u_stat, p_val = mannwhitneyu(
        low_entropies, high_entropies, alternative='two-sided'
    )
    r_rb = 1.0 - (2.0 * u_stat) / (len(low_entropies) * len(high_entropies))

    # Max possible entropy (uniform over n_pathways)
    n_pathways = cm.shape[1]
    max_entropy = np.log(n_pathways)

    results = {
        'low_risk': {
            'n': len(low_entropies),
            'mean_entropy': float(low_entropies.mean()),
            'std_entropy': float(low_entropies.std()),
            'median_entropy': float(np.median(low_entropies)),
        },
        'high_risk': {
            'n': len(high_entropies),
            'mean_entropy': float(high_entropies.mean()),
            'std_entropy': float(high_entropies.std()),
            'median_entropy': float(np.median(high_entropies)),
        },
        'max_entropy': float(max_entropy),
        'normalised_low': float(low_entropies.mean() / max_entropy),
        'normalised_high': float(high_entropies.mean() / max_entropy),
        'mann_whitney_U': float(u_stat),
        'p_value': float(p_val),
        'rank_biserial_r': float(r_rb),
        'direction': (
            'high-risk MORE uniform (higher entropy)'
            if high_entropies.mean() > low_entropies.mean()
            else 'low-risk MORE uniform (higher entropy)'
        ),
    }

    # Per-prototype breakdown
    if low_per_proto and high_per_proto:
        low_stack = np.stack(low_per_proto)   # [n_low, n_protos]
        high_stack = np.stack(high_per_proto)  # [n_high, n_protos]
        n_protos = low_stack.shape[1]

        proto_results = []
        for p in range(n_protos):
            u, pv = mannwhitneyu(
                low_stack[:, p], high_stack[:, p], alternative='two-sided'
            )
            r = 1.0 - (2.0 * u) / (low_stack.shape[0] * high_stack.shape[0])
            proto_results.append({
                'prototype': p,
                'low_mean_entropy': float(low_stack[:, p].mean()),
                'high_mean_entropy': float(high_stack[:, p].mean()),
                'diff': float(high_stack[:, p].mean() - low_stack[:, p].mean()),
                'p_value': float(pv),
                'rank_biserial_r': float(r),
            })
        results['per_prototype'] = proto_results

    # Print summary
    _log = logger.info if logger else print
    _log("\n" + "=" * 60)
    _log("DIAGNOSTIC: Cross-Modal Attention Entropy")
    _log("=" * 60)
    _log(f"Max entropy (uniform over {n_pathways} pathways): {max_entropy:.3f}")
    _log(f"")
    _log(f"Low Risk  (n={results['low_risk']['n']}): "
         f"mean={results['low_risk']['mean_entropy']:.4f} "
         f"± {results['low_risk']['std_entropy']:.4f}  "
         f"({results['normalised_low']:.1%} of max)")
    _log(f"High Risk (n={results['high_risk']['n']}): "
         f"mean={results['high_risk']['mean_entropy']:.4f} "
         f"± {results['high_risk']['std_entropy']:.4f}  "
         f"({results['normalised_high']:.1%} of max)")
    _log(f"")
    _log(f"Direction: {results['direction']}")
    _log(f"Mann-Whitney U = {u_stat:.1f}, p = {p_val:.2e}")
    _log(f"Rank-biserial r = {r_rb:.3f}")
    _log(f"")

    if 'per_prototype' in results:
        _log("Per-prototype breakdown:")
        for pr in results['per_prototype']:
            sig = "***" if pr['p_value'] < 0.001 else "**" if pr['p_value'] < 0.01 else "*" if pr['p_value'] < 0.05 else ""
            _log(f"  Proto {pr['prototype']:2d}: "
                 f"low={pr['low_mean_entropy']:.4f}  "
                 f"high={pr['high_mean_entropy']:.4f}  "
                 f"Δ={pr['diff']:+.4f}  "
                 f"p={pr['p_value']:.3e}  "
                 f"r={pr['rank_biserial_r']:+.3f} {sig}")

    return results