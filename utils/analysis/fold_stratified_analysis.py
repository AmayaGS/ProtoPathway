"""
Fold-Stratified Meta-Analysis for ProtoPathway Importance.

Runs ImportanceAnalyzer independently within each CV fold, then combines
per-fold results using Stouffer's weighted Z method. This eliminates
cross-fold confounding from different models learning different attention
patterns.

Drop-in replacement for the pooled analysis in run_importance_analysis().

Why this matters:
    Different fold models learn different relative importance orderings.
    When pooling patients across folds and comparing High vs Low risk,
    inter-model variance drowns out genuine within-model risk-associated
    signal. Fold-stratified analysis isolates the within-fold signal,
    then combines it with proper meta-analytic methods.

Usage:
    # In experiments/visualize.py, replace:
    #   analyzers = run_importance_analysis(...)
    # With:
    #   analyzers = run_fold_stratified_importance_analysis(...)
"""

import os
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm, mannwhitneyu
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests

from utils.analysis.importance_analyzer import (
    ImportanceAnalyzer,
    PrototypeShiftAnalyzer,
    _compute_ranks,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Core: fold-stratified rank analysis
# ============================================================================


def _per_fold_mann_whitney(
    importance_matrix: np.ndarray,
    groups: np.ndarray,
    n_entities: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Run Mann-Whitney U per entity within a single fold.

    Args:
        importance_matrix: [n_patients, n_entities] raw importance scores.
        groups: [n_patients] array of 'low'/'high' labels.
        n_entities: Number of entities.

    Returns:
        p_values: [n_entities] raw p-values (NaN if test failed).
        effect_sizes: [n_entities] rank-biserial r (-1 to 1).
        rank_diffs: [n_entities] mean rank difference (high - low).
        n_low: Number of low-risk patients.
        n_high: Number of high-risk patients.
    """
    # Within-patient ranking
    rank_matrix = np.zeros_like(importance_matrix)
    for i in range(importance_matrix.shape[0]):
        rank_matrix[i] = rankdata(importance_matrix[i], method='average')

    low_mask = groups == 'low'
    high_mask = groups == 'high'

    n_low = low_mask.sum()
    n_high = high_mask.sum()

    low_ranks = rank_matrix[low_mask]
    high_ranks = rank_matrix[high_mask]

    p_values = np.full(n_entities, np.nan)
    effect_sizes = np.full(n_entities, np.nan)
    rank_diffs = np.full(n_entities, np.nan)

    if n_low < 2 or n_high < 2:
        return p_values, effect_sizes, rank_diffs, n_low, n_high

    for j in range(n_entities):
        low_r = low_ranks[:, j]
        high_r = high_ranks[:, j]

        try:
            u_stat, p_val = mannwhitneyu(
                low_r, high_r, alternative='two-sided'
            )
            # Rank-biserial correlation: r = 1 - 2U/(n1*n2)
            r_rb = 1.0 - (2.0 * u_stat) / (n_low * n_high)

            p_values[j] = p_val
            effect_sizes[j] = r_rb
            rank_diffs[j] = high_r.mean() - low_r.mean()
        except Exception:
            pass

    return p_values, effect_sizes, rank_diffs, n_low, n_high


def stouffer_combine(
    per_fold_pvalues: List[np.ndarray],
    per_fold_effects: List[np.ndarray],
    per_fold_n: List[int],
    fdr_method: str = 'fdr_bh',
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Combine per-fold p-values and effect sizes using Stouffer's weighted Z.

    Stouffer's method:
        z_i = Φ^{-1}(1 - p_i/2) * sign(effect_i)   [directional]
        w_i = sqrt(n_i)
        Z   = Σ(w_i * z_i) / sqrt(Σ w_i²)
        p   = 2 * (1 - Φ(|Z|))

    Args:
        per_fold_pvalues: List of [n_entities] p-value arrays.
        per_fold_effects: List of [n_entities] effect size arrays.
        per_fold_n: List of total patients per fold.
        fdr_method: Multiple testing correction.
        alpha: Significance threshold.

    Returns:
        combined_p: [n_entities] combined raw p-values.
        combined_p_fdr: [n_entities] FDR-corrected p-values.
        combined_effect: [n_entities] weighted mean effect size.
        significant: [n_entities] boolean significance flags.
    """
    n_entities = per_fold_pvalues[0].shape[0]
    n_folds = len(per_fold_pvalues)

    # Weights proportional to sqrt(sample size)
    weights = np.array([np.sqrt(n) for n in per_fold_n])

    combined_p = np.full(n_entities, np.nan)
    combined_effect = np.full(n_entities, np.nan)

    for j in range(n_entities):
        fold_z = []
        fold_w = []
        fold_eff = []
        fold_wt = []

        for k in range(n_folds):
            p_jk = per_fold_pvalues[k][j]
            e_jk = per_fold_effects[k][j]

            if np.isnan(p_jk) or np.isnan(e_jk):
                continue

            # Clamp p-value away from 0 and 1 for numerical stability
            p_clamped = np.clip(p_jk, 1e-15, 1.0 - 1e-15)

            # Convert to directional z-score
            z = norm.ppf(1.0 - p_clamped / 2.0) * np.sign(e_jk)

            fold_z.append(z)
            fold_w.append(weights[k])
            fold_eff.append(e_jk)
            fold_wt.append(weights[k])

        if len(fold_z) < 2:
            # Need at least 2 folds for meaningful combination
            continue

        fold_z = np.array(fold_z)
        fold_w = np.array(fold_w)
        fold_eff = np.array(fold_eff)
        fold_wt = np.array(fold_wt)

        # Stouffer's Z
        z_combined = np.sum(fold_w * fold_z) / np.sqrt(np.sum(fold_w ** 2))
        p_combined = 2.0 * norm.sf(np.abs(z_combined))

        # Weighted mean effect size
        eff_combined = np.sum(fold_wt * fold_eff) / np.sum(fold_wt)

        combined_p[j] = p_combined
        combined_effect[j] = eff_combined

    # FDR correction (only on non-NaN entries)
    valid_mask = ~np.isnan(combined_p)
    combined_p_fdr = np.full(n_entities, np.nan)
    significant = np.zeros(n_entities, dtype=bool)

    if valid_mask.sum() > 0:
        reject, pvals_corrected, _, _ = multipletests(
            combined_p[valid_mask], alpha=alpha, method=fdr_method
        )
        combined_p_fdr[valid_mask] = pvals_corrected
        significant[valid_mask] = reject

    return combined_p, combined_p_fdr, combined_effect, significant


# ============================================================================
# Fold-stratified ImportanceAnalyzer wrapper
# ============================================================================

class FoldStratifiedAnalyzer:
    """
    Wraps ImportanceAnalyzer to run per-fold then combine via Stouffer's.

    Same external interface as ImportanceAnalyzer for downstream consumers
    (bar plots, violin plots, etc.), but internally runs separate analyses
    per fold and combines results.

    The rank_analysis() output DataFrame has the same columns as
    ImportanceAnalyzer.rank_analysis(), plus:
        - n_contributing_folds: How many folds contributed to this entity
        - per_fold_effects: Serialized per-fold effect sizes
    """

    def __init__(
        self,
        entity_names: List[str],
        analysis_name: str,
        group_names: Dict[str, str] = None,
    ):
        self.entity_names = list(entity_names)
        self.n_entities = len(entity_names)
        self.analysis_name = analysis_name
        self.group_names = group_names or {
            'low': 'Low Risk',
            'high': 'High Risk',
        }

        # Per-patient storage (same as ImportanceAnalyzer, plus fold)
        self.patient_ids: List[str] = []
        self.importance_vectors: List[np.ndarray] = []
        self.groups: List[str] = []
        self.folds: List[int] = []

        self._results: Optional[pd.DataFrame] = None

    def add_patient(
        self,
        patient_id: str,
        importance: np.ndarray,
        risk_group: str,
        fold: int,
    ):
        """Add a patient with fold assignment."""
        group = self._normalize_group(risk_group)
        importance = np.asarray(importance, dtype=np.float64).flatten()

        if len(importance) != self.n_entities:
            raise ValueError(
                f"Importance vector length {len(importance)} != "
                f"n_entities {self.n_entities}"
            )

        self.patient_ids.append(patient_id)
        self.importance_vectors.append(importance)
        self.groups.append(group)
        self.folds.append(fold)
        self._results = None

    def _normalize_group(self, group: str) -> str:
        g = str(group).lower().strip()
        if g in ('low', 'low risk', '0'):
            return 'low'
        elif g in ('high', 'high risk', '1'):
            return 'high'
        else:
            raise ValueError(f"Unknown risk group '{group}'")

    @property
    def n_patients(self) -> int:
        return len(self.patient_ids)

    @property
    def n_low(self) -> int:
        return sum(1 for g in self.groups if g == 'low')

    @property
    def n_high(self) -> int:
        return sum(1 for g in self.groups if g == 'high')

    def rank_analysis(
        self,
        fdr_method: str = 'fdr_bh',
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """
        Fold-stratified rank analysis with Stouffer combination.

        Returns DataFrame with same columns as ImportanceAnalyzer.rank_analysis().
        """
        if self._results is not None:
            return self._results

        importance_matrix = np.stack(self.importance_vectors)
        groups = np.array(self.groups)
        folds = np.array(self.folds)
        unique_folds = sorted(set(self.folds))

        per_fold_pvalues = []
        per_fold_effects = []
        per_fold_rank_diffs = []
        per_fold_n = []
        fold_details = []

        for fold_idx in unique_folds:
            fold_mask = folds == fold_idx
            fold_imp = importance_matrix[fold_mask]
            fold_groups = groups[fold_mask]

            n_low_fold = (fold_groups == 'low').sum()
            n_high_fold = (fold_groups == 'high').sum()

            if n_low_fold < 2 or n_high_fold < 2:
                logger.warning(
                    f"  Fold {fold_idx}: skipping ({n_low_fold} low, "
                    f"{n_high_fold} high) — need ≥2 per group"
                )
                continue

            p_vals, effects, rdiffs, n_l, n_h = _per_fold_mann_whitney(
                fold_imp, fold_groups, self.n_entities
            )

            per_fold_pvalues.append(p_vals)
            per_fold_effects.append(effects)
            per_fold_rank_diffs.append(rdiffs)
            per_fold_n.append(n_l + n_h)
            fold_details.append({
                'fold': fold_idx,
                'n_low': n_l,
                'n_high': n_h,
                'n_total': n_l + n_h,
            })

            logger.info(
                f"  Fold {fold_idx}: {n_l} low + {n_h} high = "
                f"{n_l + n_h} patients"
            )

        if len(per_fold_pvalues) < 2:
            logger.warning(
                f"[{self.analysis_name}] Only {len(per_fold_pvalues)} "
                f"usable folds — falling back to pooled analysis"
            )
            # Fall back to standard pooled analysis
            fallback = ImportanceAnalyzer(
                self.entity_names, self.analysis_name, self.group_names
            )
            for pid, imp, grp in zip(
                self.patient_ids, self.importance_vectors, self.groups
            ):
                fallback.add_patient(pid, imp, grp)
            self._results = fallback.rank_analysis(fdr_method, alpha)
            return self._results

        # Combine across folds
        combined_p, combined_p_fdr, combined_effect, significant = (
            stouffer_combine(
                per_fold_pvalues, per_fold_effects, per_fold_n,
                fdr_method, alpha,
            )
        )

        # Weighted mean rank difference for display
        weights = np.array([np.sqrt(n) for n in per_fold_n])
        rdiff_stack = np.stack(per_fold_rank_diffs)  # [n_folds, n_entities]
        # Replace NaN with 0 for weighted average
        rdiff_clean = np.nan_to_num(rdiff_stack, nan=0.0)
        weighted_rdiff = (
            np.sum(weights[:, None] * rdiff_clean, axis=0)
            / np.sum(weights)
        )

        # Per-group mean importance (pooled, for display only)
        low_mask = groups == 'low'
        high_mask = groups == 'high'
        low_imp = importance_matrix[low_mask]
        high_imp = importance_matrix[high_mask]

        # Count contributing folds per entity
        n_contributing = np.zeros(self.n_entities, dtype=int)
        for pv in per_fold_pvalues:
            n_contributing += (~np.isnan(pv)).astype(int)

        # Build results DataFrame
        results = pd.DataFrame({
            'entity': self.entity_names,
            'mean_rank_low': [np.nan] * self.n_entities,  # not meaningful pooled
            'mean_rank_high': [np.nan] * self.n_entities,
            'rank_difference': weighted_rdiff,
            'mean_importance_low': low_imp.mean(axis=0) if low_mask.any() else np.nan,
            'mean_importance_high': high_imp.mean(axis=0) if high_mask.any() else np.nan,
            'u_statistic': np.nan,  # not applicable for combined
            'p_value': combined_p,
            'p_value_fdr': combined_p_fdr,
            'rank_biserial_r': combined_effect,
            'abs_rank_biserial_r': np.abs(combined_effect),
            'significant': significant,
            'higher_in': [
                self.group_names['high'] if e > 0
                else self.group_names['low']
                for e in combined_effect
            ],
            'n_contributing_folds': n_contributing,
        })

        results = results.sort_values(
            'abs_rank_biserial_r', ascending=False
        ).reset_index(drop=True)

        self._results = results

        n_sig = significant.sum()
        logger.info(
            f"[{self.analysis_name}] Stouffer meta-analysis: "
            f"{n_sig}/{self.n_entities} significant (FDR < {alpha}), "
            f"combined from {len(per_fold_pvalues)} folds"
        )

        return results

    def top_differential(
        self,
        k: int = 20,
        significant_only: bool = False,
        min_effect_size: float = 0.0,
    ) -> pd.DataFrame:
        """Top entities by combined effect size."""
        results = self.rank_analysis()
        filtered = results.copy()
        if significant_only:
            filtered = filtered[filtered['significant']]
        if min_effect_size > 0:
            filtered = filtered[
                filtered['abs_rank_biserial_r'] >= min_effect_size
            ]
        return filtered.head(k)

    def per_group_summary(self, k: int = 20) -> Dict[str, pd.DataFrame]:
        """Top entities per risk group (pooled, for display)."""
        importance_matrix = np.stack(self.importance_vectors)
        groups = np.array(self.groups)
        summaries = {}
        for group_key, group_label in self.group_names.items():
            mask = groups == group_key
            if not mask.any():
                continue
            group_imp = importance_matrix[mask]
            normed = group_imp / (
                group_imp.sum(axis=1, keepdims=True) + 1e-8
            )
            mean_normed = normed.mean(axis=0)
            top_idx = np.argsort(mean_normed)[-k:][::-1]
            summaries[group_key] = pd.DataFrame({
                'entity': [self.entity_names[i] for i in top_idx],
                'mean_normalized_importance': mean_normed[top_idx],
                'mean_raw_importance': group_imp.mean(axis=0)[top_idx],
                'n_patients': mask.sum(),
            })
        return summaries

    def get_patient_level_data(self) -> pd.DataFrame:
        """Full patient-level data including fold."""
        df = pd.DataFrame(
            np.stack(self.importance_vectors),
            columns=self.entity_names,
        )
        df.insert(0, 'patient_id', self.patient_ids)
        df.insert(1, 'risk_group', [
            self.group_names[g] for g in self.groups
        ])
        df.insert(2, 'fold', self.folds)
        return df

    def save_results(self, output_dir: str, save_patient_level: bool = True):
        """Save analysis results (same format as ImportanceAnalyzer)."""
        os.makedirs(output_dir, exist_ok=True)
        prefix = self.analysis_name

        results = self.rank_analysis()
        results.to_csv(
            os.path.join(output_dir, f'{prefix}_rank_analysis.csv'),
            index=False,
        )

        top = self.top_differential(k=40)
        top.to_csv(
            os.path.join(output_dir, f'{prefix}_top_differential.csv'),
            index=False,
        )

        summaries = self.per_group_summary(k=40)
        for group_key, df in summaries.items():
            df.to_csv(
                os.path.join(
                    output_dir, f'{prefix}_top_{group_key}_risk.csv'
                ),
                index=False,
            )

        if save_patient_level:
            patient_df = self.get_patient_level_data()
            patient_df.to_csv(
                os.path.join(output_dir, f'{prefix}_patient_data.csv'),
                index=False,
            )


# ============================================================================
# Drop-in replacement for run_importance_analysis
# ============================================================================

def run_fold_stratified_importance_analysis(
    predictions: pd.DataFrame,
    attention_by_patient: Dict[str, Dict],
    entity_names: Dict[str, List[str]],
    output_dir: str,
    pathways_of_interest: Optional[List[str]] = None,
    top_k_pathways: int = 10,
    skip_prototype_signals: bool = False,
) -> Dict[str, FoldStratifiedAnalyzer]:
    """
    Drop-in replacement for run_importance_analysis() using fold stratification.

    Same args, same return type structure. Uses FoldStratifiedAnalyzer
    instead of ImportanceAnalyzer internally.

    Requires predictions DataFrame to have a 'fold' column (added by
    pool_cv_results).
    """
    os.makedirs(output_dir, exist_ok=True)

    gene_names = entity_names.get('gene_names', [])
    pathway_names = entity_names.get('pathway_names', [])

    # Build risk group + fold lookup
    risk_map = dict(
        zip(predictions['patient_id'], predictions['risk_group'])
    )
    fold_map = dict(
        zip(predictions['patient_id'], predictions['fold'])
    )

    valid_patients = [
        pid for pid in attention_by_patient
        if pid in risk_map
        and risk_map[pid] is not None
        and pid in fold_map
    ]

    logger.info(
        f"Running fold-stratified importance analysis on "
        f"{len(valid_patients)} patients"
    )

    # Detect available signals
    sample_attn = attention_by_patient[valid_patients[0]]
    has_gene_pathway = 'gene_pathway_attention' in sample_attn
    has_pathway_gate = 'pathway_importance' in sample_attn
    has_wsi_gate = (
        'patch_assignments' in sample_attn
        and isinstance(sample_attn['patch_assignments'], dict)
        and 'gate_weights' in sample_attn['patch_assignments']
    )
    has_fusion_gate = 'fusion_gate_weights' in sample_attn
    has_cross_modal = 'cross_modal_attention' in sample_attn

    n_protos = None
    if has_wsi_gate:
        n_protos = len(sample_attn['patch_assignments']['gate_weights'])
    elif has_cross_modal:
        n_protos = sample_attn['cross_modal_attention'].shape[0]
    proto_names = [
        f'Prototype {i}' for i in range(n_protos)
    ] if n_protos else []

    analyzers = {}

    # Helper to create and populate a FoldStratifiedAnalyzer
    def _make_analyzer(names, analysis_name, extract_fn):
        analyzer = FoldStratifiedAnalyzer(names, analysis_name)
        for pid in valid_patients:
            attn = attention_by_patient[pid]
            importance = extract_fn(attn)
            analyzer.add_patient(
                pid, importance, risk_map[pid], fold_map[pid]
            )
        analyzer.save_results(output_dir)
        return analyzer

    # ---- 1. Pathway gate importance (Signal B) ----
    if has_pathway_gate and pathway_names:
        logger.info("  [Fold-stratified] Pathway gate importance...")
        analyzers['pathway_gate'] = _make_analyzer(
            pathway_names, 'pathway_gate',
            lambda a: a['pathway_importance'],
        )

    # ---- 2. Gene importance: average and sum (Signal C) ----
    if has_gene_pathway and has_pathway_gate and gene_names:
        logger.info("  [Fold-stratified] Gene importance (avg + sum)...")

        def _gene_avg(attn):
            A = attn['gene_pathway_attention']
            participation = np.maximum((A > 0).sum(axis=1).astype(float), 1.0)
            return A.sum(axis=1) / participation

        def _gene_sum(attn):
            return attn['gene_pathway_attention'].sum(axis=1)

        analyzers['gene_average'] = _make_analyzer(
            gene_names, 'gene_average', _gene_avg,
        )
        analyzers['gene_sum'] = _make_analyzer(
            gene_names, 'gene_sum', _gene_sum,
        )

    # ---- 3. Raw prototype importance (Signal E) ----
    if not skip_prototype_signals:
        if has_wsi_gate and proto_names:
            logger.info("  [Fold-stratified] Prototype raw importance...")
            analyzers['prototype_raw'] = _make_analyzer(
                proto_names, 'prototype_raw',
                lambda a: a['patch_assignments']['gate_weights'],
            )

        # ---- 4. Pathway-attended prototype importance (Signal H) ----
        if has_fusion_gate and proto_names:
            logger.info("  [Fold-stratified] Prototype attended importance...")
            analyzers['prototype_attended'] = _make_analyzer(
                proto_names, 'prototype_attended',
                lambda a: a['fusion_gate_weights'],
            )

        # ---- 5. Prototype shift (E → H) ----
        # Keep as-is (PrototypeShiftAnalyzer uses paired Wilcoxon, less
        # affected by cross-fold issues since it's within-patient)
        if has_wsi_gate and has_fusion_gate and proto_names:
            logger.info("  Prototype importance shift (E → H)...")
            shift_analyzer = PrototypeShiftAnalyzer(proto_names)
            for pid in valid_patients:
                attn = attention_by_patient[pid]
                shift_analyzer.add_patient(
                    pid,
                    wsi_gate=attn['patch_assignments']['gate_weights'],
                    fusion_gate=attn['fusion_gate_weights'],
                    risk_group=risk_map[pid],
                )
            shift_analyzer.save_results(output_dir)
            analyzers['prototype_shift'] = shift_analyzer

        # ---- 6. Cross-modal per-prototype pathway ranking (Signal G) ----
        if has_cross_modal and pathway_names and proto_names:
            logger.info("  [Fold-stratified] Per-prototype pathway attention...")
            for proto_idx in range(n_protos):
                name = f'crossmodal_proto_{proto_idx}'
                analyzers[name] = _make_analyzer(
                    pathway_names, name,
                    lambda a, idx=proto_idx: a['cross_modal_attention'][idx],
                )

    # ---- 7. Within-pathway gene analysis ----
    if has_gene_pathway and gene_names and pathway_names:
        if pathways_of_interest is None and 'pathway_gate' in analyzers:
            pw_results = analyzers['pathway_gate'].top_differential(
                k=top_k_pathways
            )
            pathways_of_interest = pw_results['entity'].tolist()
            logger.info(
                f"  Auto-selected {len(pathways_of_interest)} pathways"
            )

        if pathways_of_interest:
            logger.info(
                f"  [Fold-stratified] Genes within "
                f"{len(pathways_of_interest)} pathways..."
            )
            for pw_name in pathways_of_interest:
                if pw_name not in pathway_names:
                    continue
                pw_idx = pathway_names.index(pw_name)

                # Find participating genes
                gene_mask = np.zeros(len(gene_names), dtype=bool)
                for pid in valid_patients:
                    col = attention_by_patient[pid][
                        'gene_pathway_attention'
                    ][:, pw_idx]
                    gene_mask |= (col > 0)

                participating = [
                    gene_names[i]
                    for i in range(len(gene_names)) if gene_mask[i]
                ]
                gene_indices = [
                    i for i in range(len(gene_names)) if gene_mask[i]
                ]

                if len(participating) < 2:
                    continue

                safe_pw = (
                    pw_name[:60]
                    .replace(' ', '_')
                    .replace('/', '_')
                    .replace(':', '_')
                )
                aname = f'genes_in_{safe_pw}'

                pw_analyzer = FoldStratifiedAnalyzer(participating, aname)
                for pid in valid_patients:
                    col = attention_by_patient[pid][
                        'gene_pathway_attention'
                    ][:, pw_idx]
                    pw_analyzer.add_patient(
                        pid, col[gene_indices],
                        risk_map[pid], fold_map[pid],
                    )
                pw_analyzer.save_results(output_dir)
                analyzers[aname] = pw_analyzer

    # ---- Summary ----
    logger.info(f"\nFold-stratified importance analysis complete:")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Analyses: {len(analyzers)}")
    for name, analyzer in analyzers.items():
        if isinstance(analyzer, (FoldStratifiedAnalyzer, ImportanceAnalyzer)):
            results = analyzer.rank_analysis()
            n_sig = results['significant'].sum()
            logger.info(
                f"    {name}: {len(results)} entities, {n_sig} significant"
            )
        elif isinstance(analyzer, PrototypeShiftAnalyzer):
            results = analyzer.analyze_shift()
            n_sig = results['significant'].sum()
            logger.info(
                f"    {name}: {len(results)} prototypes, "
                f"{n_sig} significant shift"
            )

    return analyzers