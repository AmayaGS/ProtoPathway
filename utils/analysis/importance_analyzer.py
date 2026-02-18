"""
Rank-Based Importance Analysis for ProtoPathway.

Provides nonparametric statistical comparison of model importance signals
between risk groups (high vs low risk, from median-split risk scores).

Statistical framework:
    - Mann-Whitney U test (nonparametric group comparison)
    - Rank-biserial correlation (nonparametric effect size)
    - Benjamini-Hochberg FDR correction (multiple testing)

Handles all per-patient importance vectors from ProtoPathway:
    - Pathway gate importance (Signal B)
    - Gene importance: average and sum variants (Signal C)
    - Raw prototype importance (Signal E, WSI gate)
    - Pathway-attended prototype importance (Signal H, fusion gate)
    - Prototype assignment proportions (derived from Signal F)
    - Within-pathway gene attention (Signal A, per pathway)
    - Per-prototype pathway attention ranking (Signal G, per prototype)
"""

import os
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)


class ImportanceAnalyzer:
    """
    Rank-based importance analysis for any per-patient importance vector.

    Compares entity importance between two groups (high/low risk) using:
    - Within-patient ranking (converts to relative importance)
    - Mann-Whitney U test per entity
    - Rank-biserial correlation as effect size
    - Benjamini-Hochberg FDR correction

    Usage:
        analyzer = ImportanceAnalyzer(
            entity_names=['TP53', 'BRCA1', ...],
            analysis_name='gene_average'
        )
        for pid, importance, group in patient_data:
            analyzer.add_patient(pid, importance, group)
        results = analyzer.rank_analysis()
    """

    def __init__(
        self,
        entity_names: List[str],
        analysis_name: str,
        group_names: Dict[str, str] = None
    ):
        """
        Args:
            entity_names: Names for each entity (genes, pathways, prototypes).
                Order must match importance vector indices.
            analysis_name: Identifier for this analysis
                (e.g. 'pathway_gate', 'gene_average', 'prototype_raw').
            group_names: Display names for groups.
                Defaults to {'low': 'Low Risk', 'high': 'High Risk'}.
        """
        self.entity_names = list(entity_names)
        self.n_entities = len(entity_names)
        self.analysis_name = analysis_name
        self.group_names = group_names or {
            'low': 'Low Risk',
            'high': 'High Risk'
        }

        # Per-patient storage
        self.patient_ids: List[str] = []
        self.importance_vectors: List[np.ndarray] = []
        self.groups: List[str] = []  # 'low' or 'high'

        # Results cache
        self._results: Optional[pd.DataFrame] = None

    def add_patient(
        self,
        patient_id: str,
        importance: np.ndarray,
        risk_group: str
    ):
        """
        Add a patient's importance vector and risk group assignment.

        Args:
            patient_id: Patient identifier.
            importance: Importance vector [n_entities]. Must match entity_names length.
            risk_group: Risk group label. Accepts 'High Risk'/'Low Risk' or 'high'/'low'.
        """
        # Normalize group label
        group = self._normalize_group(risk_group)

        # Validate
        importance = np.asarray(importance, dtype=np.float64).flatten()
        if len(importance) != self.n_entities:
            raise ValueError(
                f"Importance vector length {len(importance)} does not match "
                f"n_entities {self.n_entities} for analysis '{self.analysis_name}'"
            )

        self.patient_ids.append(patient_id)
        self.importance_vectors.append(importance)
        self.groups.append(group)
        self._results = None  # Invalidate cache

    def _normalize_group(self, group: str) -> str:
        """Normalize group labels to 'low' or 'high'."""
        g = str(group).lower().strip()
        if g in ('low', 'low risk', '0'):
            return 'low'
        elif g in ('high', 'high risk', '1'):
            return 'high'
        else:
            raise ValueError(
                f"Unknown risk group '{group}'. "
                f"Expected 'High Risk'/'Low Risk' or 'high'/'low'."
            )

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
        alpha: float = 0.05
    ) -> pd.DataFrame:
        """
        Core analysis: rank-based comparison between risk groups.

        For each patient, importance scores are converted to ranks (higher
        importance → higher rank). Then for each entity, ranks are compared
        between risk groups using Mann-Whitney U with rank-biserial
        correlation as effect size.

        Args:
            fdr_method: Multiple testing correction method for statsmodels.
                Default 'fdr_bh' (Benjamini-Hochberg).
            alpha: Significance threshold after FDR correction.

        Returns:
            DataFrame sorted by |rank_biserial_r| with columns:
                - entity: Entity name
                - mean_rank_low, mean_rank_high: Mean rank per group
                - rank_difference: mean_rank_high - mean_rank_low
                - mean_importance_low, mean_importance_high: Raw importance means
                - u_statistic: Mann-Whitney U statistic
                - p_value: Raw p-value
                - p_value_fdr: FDR-corrected p-value
                - rank_biserial_r: Effect size (-1 to 1)
                - abs_rank_biserial_r: Absolute effect size
                - significant: Whether p_value_fdr < alpha
                - higher_in: Which group has higher ranks
        """
        if self._results is not None:
            return self._results

        if self.n_patients < 4:
            raise ValueError(
                f"Need at least 4 patients for rank analysis, "
                f"got {self.n_patients}"
            )
        if self.n_low < 2 or self.n_high < 2:
            raise ValueError(
                f"Need at least 2 patients per group. "
                f"Got {self.n_low} low, {self.n_high} high."
            )

        # Stack importance vectors: [n_patients, n_entities]
        importance_matrix = np.stack(self.importance_vectors)
        groups = np.array(self.groups)

        # Convert to within-patient ranks (higher importance → higher rank)
        rank_matrix = np.zeros_like(importance_matrix)
        for i in range(self.n_patients):
            rank_matrix[i] = _compute_ranks(importance_matrix[i])

        # Split by group
        low_mask = groups == 'low'
        high_mask = groups == 'high'

        low_ranks = rank_matrix[low_mask]       # [n_low, n_entities]
        high_ranks = rank_matrix[high_mask]     # [n_high, n_entities]
        low_importance = importance_matrix[low_mask]
        high_importance = importance_matrix[high_mask]

        n1 = low_ranks.shape[0]
        n2 = high_ranks.shape[0]

        # Per-entity statistical tests
        results = []
        for j in range(self.n_entities):
            low_r = low_ranks[:, j]
            high_r = high_ranks[:, j]

            # Mann-Whitney U test
            try:
                u_stat, p_val = mannwhitneyu(
                    high_r, low_r, alternative='two-sided'
                )
            except ValueError:
                # All values identical
                u_stat = n1 * n2 / 2
                p_val = 1.0

            # Rank-biserial correlation: r = 1 - (2U) / (n1 * n2)
            r_rb = 1.0 - (2.0 * u_stat) / (n1 * n2)

            results.append({
                'entity': self.entity_names[j],
                'mean_rank_low': low_r.mean(),
                'mean_rank_high': high_r.mean(),
                'rank_difference': high_r.mean() - low_r.mean(),
                'mean_importance_low': low_importance[:, j].mean(),
                'mean_importance_high': high_importance[:, j].mean(),
                'std_importance_low': low_importance[:, j].std(),
                'std_importance_high': high_importance[:, j].std(),
                'u_statistic': u_stat,
                'p_value': p_val,
                'rank_biserial_r': r_rb,
                'abs_rank_biserial_r': abs(r_rb),
            })

        df = pd.DataFrame(results)

        # FDR correction
        if len(df) > 0:
            reject, p_fdr, _, _ = multipletests(
                df['p_value'].values,
                alpha=alpha,
                method=fdr_method
            )
            df['p_value_fdr'] = p_fdr
            df['significant'] = reject
        else:
            df['p_value_fdr'] = []
            df['significant'] = []

        # Determine which group has higher ranks
        df['higher_in'] = np.where(
            df['rank_difference'] > 0,
            self.group_names['high'],
            self.group_names['low']
        )

        # Sort by absolute effect size
        df = df.sort_values('abs_rank_biserial_r', ascending=False)
        df = df.reset_index(drop=True)

        self._results = df
        return df

    def top_differential(
        self,
        k: int = 20,
        fdr_threshold: float = 0.05,
        min_effect_size: float = 0.0,
        significant_only: bool = False
    ) -> pd.DataFrame:
        """
        Get top-K entities by |rank_biserial_r|.

        Args:
            k: Maximum number of entities to return.
            fdr_threshold: FDR significance threshold.
            min_effect_size: Minimum |rank_biserial_r| to include.
            significant_only: If True, only include FDR-significant entities.

        Returns:
            Filtered and sorted DataFrame.
        """
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
        """
        Top entities per risk group by mean normalized importance.

        Args:
            k: Number of top entities per group.

        Returns:
            Dict with 'low_risk' and 'high_risk' DataFrames.
        """
        importance_matrix = np.stack(self.importance_vectors)
        groups = np.array(self.groups)

        summaries = {}
        for group_key, group_label in self.group_names.items():
            mask = groups == group_key
            if not mask.any():
                continue

            group_imp = importance_matrix[mask]

            # Normalize within patient, then average across patients
            normed = group_imp / (
                group_imp.sum(axis=1, keepdims=True) + 1e-8
            )
            mean_normed = normed.mean(axis=0)

            top_idx = np.argsort(mean_normed)[-k:][::-1]
            summaries[group_key] = pd.DataFrame({
                'entity': [self.entity_names[i] for i in top_idx],
                'mean_normalized_importance': mean_normed[top_idx],
                'mean_raw_importance': group_imp.mean(axis=0)[top_idx],
                'n_patients': mask.sum()
            })

        return summaries

    def get_patient_level_data(self) -> pd.DataFrame:
        """
        Get full patient-level importance matrix for external analysis.

        Returns:
            DataFrame with patient_id, risk_group, and one column per entity.
        """
        df = pd.DataFrame(
            np.stack(self.importance_vectors),
            columns=self.entity_names
        )
        df.insert(0, 'patient_id', self.patient_ids)
        df.insert(1, 'risk_group', [
            self.group_names[g] for g in self.groups
        ])
        return df

    def save_results(self, output_dir: str, save_patient_level: bool = True):
        """
        Save analysis results to CSV files.

        Args:
            output_dir: Directory to save outputs.
            save_patient_level: Whether to also save patient-level data.

        Produces:
            {analysis_name}_rank_analysis.csv      - Full rank analysis results
            {analysis_name}_top_differential.csv    - Top 40 by effect size
            {analysis_name}_patient_data.csv        - Patient-level importance
            {analysis_name}_summary.csv             - Per-group top entities
        """
        os.makedirs(output_dir, exist_ok=True)
        prefix = self.analysis_name

        # Full results
        results = self.rank_analysis()
        results.to_csv(
            os.path.join(output_dir, f'{prefix}_rank_analysis.csv'),
            index=False
        )

        # Top differential
        top = self.top_differential(k=40)
        top.to_csv(
            os.path.join(output_dir, f'{prefix}_top_differential.csv'),
            index=False
        )

        # Per-group summary
        summaries = self.per_group_summary(k=40)
        for group_key, df in summaries.items():
            df.to_csv(
                os.path.join(
                    output_dir,
                    f'{prefix}_top_{group_key}_risk.csv'
                ),
                index=False
            )

        # Patient-level data
        if save_patient_level:
            patient_df = self.get_patient_level_data()
            patient_df.to_csv(
                os.path.join(output_dir, f'{prefix}_patient_data.csv'),
                index=False
            )

        n_sig = results['significant'].sum()
        logger.info(
            f"[{self.analysis_name}] Saved results: "
            f"{len(results)} entities, {n_sig} significant (FDR < 0.05)"
        )


class PrototypeShiftAnalyzer:
    """
    Analyze prototype importance shift: before vs after pathway attendance.

    Compares WSI gate weights (Signal E) with fusion gate weights (Signal H)
    using Wilcoxon signed-rank test (paired within-patient).

    This reveals how pathway context changes the model's morphological
    priorities.
    """

    def __init__(
        self,
        prototype_names: List[str],
        group_names: Dict[str, str] = None
    ):
        """
        Args:
            prototype_names: Names/IDs for each prototype.
            group_names: Display names for risk groups.
        """
        self.prototype_names = list(prototype_names)
        self.n_prototypes = len(prototype_names)
        self.group_names = group_names or {
            'low': 'Low Risk',
            'high': 'High Risk'
        }

        self.patient_ids: List[str] = []
        self.wsi_gate_weights: List[np.ndarray] = []     # Signal E
        self.fusion_gate_weights: List[np.ndarray] = []   # Signal H
        self.groups: List[str] = []

    def add_patient(
        self,
        patient_id: str,
        wsi_gate: np.ndarray,
        fusion_gate: np.ndarray,
        risk_group: str
    ):
        """Add a patient's paired gate weights."""
        group = risk_group.lower().replace(' risk', '').strip()
        if group not in ('low', 'high'):
            if group in ('0',):
                group = 'low'
            elif group in ('1',):
                group = 'high'

        self.patient_ids.append(patient_id)
        self.wsi_gate_weights.append(
            np.asarray(wsi_gate, dtype=np.float64).flatten()
        )
        self.fusion_gate_weights.append(
            np.asarray(fusion_gate, dtype=np.float64).flatten()
        )
        self.groups.append(group)

    def analyze_shift(
        self,
        fdr_method: str = 'fdr_bh',
        alpha: float = 0.05
    ) -> pd.DataFrame:
        """
        Analyze per-prototype importance shift across all patients.

        For each prototype, tests whether its importance systematically
        changes after pathway attention using Wilcoxon signed-rank test.

        Returns:
            DataFrame with columns:
                - prototype: Prototype name
                - mean_wsi_gate: Mean Signal E weight
                - mean_fusion_gate: Mean Signal H weight
                - mean_shift: Mean (H - E) across patients
                - std_shift: Std of shift
                - wilcoxon_stat: Test statistic
                - p_value, p_value_fdr: Raw and corrected p-values
                - significant: FDR significance
                - direction: 'gained' or 'lost' importance
        """
        E = np.stack(self.wsi_gate_weights)   # [n_patients, n_proto]
        H = np.stack(self.fusion_gate_weights)  # [n_patients, n_proto]
        shift = H - E

        results = []
        for j in range(self.n_prototypes):
            e_vals = E[:, j]
            h_vals = H[:, j]
            s_vals = shift[:, j]

            # Wilcoxon signed-rank test
            try:
                stat, p_val = wilcoxon(h_vals, e_vals, alternative='two-sided')
            except ValueError:
                # All differences are zero
                stat, p_val = 0.0, 1.0

            results.append({
                'prototype': self.prototype_names[j],
                'mean_wsi_gate': e_vals.mean(),
                'mean_fusion_gate': h_vals.mean(),
                'mean_shift': s_vals.mean(),
                'std_shift': s_vals.std(),
                'median_shift': np.median(s_vals),
                'wilcoxon_stat': stat,
                'p_value': p_val,
            })

        df = pd.DataFrame(results)

        # FDR correction
        if len(df) > 0:
            reject, p_fdr, _, _ = multipletests(
                df['p_value'].values, alpha=alpha, method=fdr_method
            )
            df['p_value_fdr'] = p_fdr
            df['significant'] = reject

        df['direction'] = np.where(
            df['mean_shift'] > 0, 'gained', 'lost'
        )

        df = df.sort_values('mean_shift', key=abs, ascending=False)
        df = df.reset_index(drop=True)

        return df

    def analyze_shift_by_risk(
        self,
        fdr_method: str = 'fdr_bh',
        alpha: float = 0.05
    ) -> Dict[str, pd.DataFrame]:
        """
        Run shift analysis separately for each risk group.

        Returns:
            Dict with 'low' and 'high' DataFrames (same format as
            analyze_shift).
        """
        E = np.stack(self.wsi_gate_weights)
        H = np.stack(self.fusion_gate_weights)
        groups = np.array(self.groups)

        results_by_group = {}
        for group_key in ('low', 'high'):
            mask = groups == group_key
            if mask.sum() < 3:
                logger.warning(
                    f"Too few patients ({mask.sum()}) in {group_key} "
                    f"group for shift analysis"
                )
                continue

            e_group = E[mask]
            h_group = H[mask]
            shift = h_group - e_group

            rows = []
            for j in range(self.n_prototypes):
                try:
                    stat, p_val = wilcoxon(
                        h_group[:, j], e_group[:, j],
                        alternative='two-sided'
                    )
                except ValueError:
                    stat, p_val = 0.0, 1.0

                rows.append({
                    'prototype': self.prototype_names[j],
                    'mean_wsi_gate': e_group[:, j].mean(),
                    'mean_fusion_gate': h_group[:, j].mean(),
                    'mean_shift': shift[:, j].mean(),
                    'std_shift': shift[:, j].std(),
                    'wilcoxon_stat': stat,
                    'p_value': p_val,
                })

            df = pd.DataFrame(rows)
            if len(df) > 0:
                reject, p_fdr, _, _ = multipletests(
                    df['p_value'].values, alpha=alpha, method=fdr_method
                )
                df['p_value_fdr'] = p_fdr
                df['significant'] = reject

            df['direction'] = np.where(
                df['mean_shift'] > 0, 'gained', 'lost'
            )
            df = df.sort_values('mean_shift', key=abs, ascending=False)
            df = df.reset_index(drop=True)

            results_by_group[group_key] = df

        return results_by_group

    def save_results(self, output_dir: str):
        """Save shift analysis results."""
        os.makedirs(output_dir, exist_ok=True)

        # Overall shift
        shift_df = self.analyze_shift()
        shift_df.to_csv(
            os.path.join(output_dir, 'prototype_shift_analysis.csv'),
            index=False
        )

        # Per-risk-group shift
        by_risk = self.analyze_shift_by_risk()
        for group_key, df in by_risk.items():
            df.to_csv(
                os.path.join(
                    output_dir,
                    f'prototype_shift_{group_key}_risk.csv'
                ),
                index=False
            )

        n_sig = shift_df['significant'].sum()
        logger.info(
            f"[prototype_shift] {n_sig}/{self.n_prototypes} prototypes "
            f"with significant importance shift"
        )


# ============================================================================
# Orchestration: run all importance analyses from pooled data
# ============================================================================

def run_importance_analysis(
    predictions: pd.DataFrame,
    attention_by_patient: Dict[str, Dict],
    entity_names: Dict[str, List[str]],
    output_dir: str,
    pathways_of_interest: Optional[List[str]] = None,
    top_k_pathways: int = 10,
) -> Dict[str, ImportanceAnalyzer]:
    """
    Run the full suite of importance analyses on pooled CV data.

    This is the main entry point. It instantiates analyzers for each
    signal type, populates them from pooled attention data, runs
    analysis, and saves results.

    Args:
        predictions: Pooled predictions DataFrame (must include 'risk_group').
        attention_by_patient: Dict from pool_cv_results().
        entity_names: Dict with 'gene_names', 'pathway_names' lists.
        output_dir: Directory for all analysis outputs.
        pathways_of_interest: Optional list of pathway names for
            within-pathway gene analysis. If None, auto-selects top
            pathways from pathway gate analysis.
        top_k_pathways: If pathways_of_interest is None, analyse genes
            within the top-K most differential pathways.

    Returns:
        Dict mapping analysis_name → ImportanceAnalyzer instance.
    """
    os.makedirs(output_dir, exist_ok=True)

    gene_names = entity_names.get('gene_names', [])
    pathway_names = entity_names.get('pathway_names', [])

    # Build risk group lookup
    risk_map = dict(
        zip(predictions['patient_id'], predictions['risk_group'])
    )

    # Filter to patients with both attention and risk data
    valid_patients = [
        pid for pid in attention_by_patient
        if pid in risk_map and risk_map[pid] is not None
    ]

    logger.info(
        f"Running importance analysis on {len(valid_patients)} patients"
    )

    # Detect available signals from first patient
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
    has_hard_assignments = (
        'patch_assignments' in sample_attn
        and isinstance(sample_attn['patch_assignments'], dict)
        and 'hard_assignments' in sample_attn['patch_assignments']
    )

    # Infer prototype names
    n_protos = None
    if has_wsi_gate:
        n_protos = len(
            sample_attn['patch_assignments']['gate_weights']
        )
    elif has_cross_modal:
        n_protos = sample_attn['cross_modal_attention'].shape[0]
    proto_names = [f'Prototype {i}' for i in range(n_protos)] if n_protos else []

    analyzers = {}

    # ---- 1. Pathway gate importance (Signal B) ----
    if has_pathway_gate and pathway_names:
        logger.info("  Analyzing pathway gate importance (Signal B)...")
        pw_analyzer = ImportanceAnalyzer(pathway_names, 'pathway_gate')

        for pid in valid_patients:
            attn = attention_by_patient[pid]
            pw_analyzer.add_patient(
                pid, attn['pathway_importance'], risk_map[pid]
            )

        pw_analyzer.save_results(output_dir)
        analyzers['pathway_gate'] = pw_analyzer

    # ---- 2. Gene importance: average and sum (Signal C) ----
    if has_gene_pathway and has_pathway_gate and gene_names:
        logger.info(
            "  Analyzing gene importance — average and sum (Signal C)..."
        )

        gene_avg_analyzer = ImportanceAnalyzer(gene_names, 'gene_average')
        gene_sum_analyzer = ImportanceAnalyzer(gene_names, 'gene_sum')

        for pid in valid_patients:
            attn = attention_by_patient[pid]
            A = attn['gene_pathway_attention']   # [G, P]
            B = attn['pathway_importance']       # [P]

            # Sum: total importance across all pathways
            # I_sum_g = sum_p A_gp
            gene_sum = A.sum(axis=1)  # [G]

            # Average: mean importance per participating pathway
            # I_avg_g = sum_p A_gp / sum_p I(A_gp > 0)
            participation = (A > 0).sum(axis=1).astype(np.float64)  # [G]
            participation = np.maximum(participation, 1.0)  # avoid div by 0
            gene_avg = A.sum(axis=1) / participation  # [G]

            gene_avg_analyzer.add_patient(pid, gene_avg, risk_map[pid])
            gene_sum_analyzer.add_patient(pid, gene_sum, risk_map[pid])

        gene_avg_analyzer.save_results(output_dir)
        gene_sum_analyzer.save_results(output_dir)
        analyzers['gene_average'] = gene_avg_analyzer
        analyzers['gene_sum'] = gene_sum_analyzer

    # ---- 3. Raw prototype importance (Signal E: WSI gate) ----
    if has_wsi_gate and proto_names:
        logger.info("  Analyzing raw prototype importance (Signal E)...")
        proto_raw = ImportanceAnalyzer(proto_names, 'prototype_raw')

        for pid in valid_patients:
            attn = attention_by_patient[pid]
            gate_w = attn['patch_assignments']['gate_weights']
            proto_raw.add_patient(pid, gate_w, risk_map[pid])

        proto_raw.save_results(output_dir)
        analyzers['prototype_raw'] = proto_raw

    # ---- 4. Pathway-attended prototype importance (Signal H) ----
    if has_fusion_gate and proto_names:
        logger.info(
            "  Analyzing attended prototype importance (Signal H)..."
        )
        proto_attended = ImportanceAnalyzer(
            proto_names, 'prototype_attended'
        )

        for pid in valid_patients:
            attn = attention_by_patient[pid]
            proto_attended.add_patient(
                pid, attn['fusion_gate_weights'], risk_map[pid]
            )

        proto_attended.save_results(output_dir)
        analyzers['prototype_attended'] = proto_attended

    # ---- 5. Prototype importance shift (Signal E vs H) ----
    if has_wsi_gate and has_fusion_gate and proto_names:
        logger.info("  Analyzing prototype importance shift (E → H)...")
        shift_analyzer = PrototypeShiftAnalyzer(proto_names)

        for pid in valid_patients:
            attn = attention_by_patient[pid]
            shift_analyzer.add_patient(
                pid,
                wsi_gate=attn['patch_assignments']['gate_weights'],
                fusion_gate=attn['fusion_gate_weights'],
                risk_group=risk_map[pid]
            )

        shift_analyzer.save_results(output_dir)
        analyzers['prototype_shift'] = shift_analyzer

    # ---- 6. Prototype assignment frequency (Signal F) ----
    if has_hard_assignments and proto_names:
        logger.info("  Analyzing prototype assignment frequency...")
        assign_analyzer = ImportanceAnalyzer(
            proto_names, 'prototype_assignment_freq'
        )

        for pid in valid_patients:
            attn = attention_by_patient[pid]
            hard = attn['patch_assignments']['hard_assignments']
            if isinstance(hard, np.ndarray):
                hard_t = hard.astype(int)
            else:
                hard_t = np.array(hard, dtype=int)

            proportions = np.bincount(
                hard_t, minlength=n_protos
            ).astype(np.float64)
            proportions /= (proportions.sum() + 1e-8)

            assign_analyzer.add_patient(pid, proportions, risk_map[pid])

        assign_analyzer.save_results(output_dir)
        analyzers['prototype_assignment_freq'] = assign_analyzer

    # ---- 7. Per-prototype pathway ranking (Signal G) ----
    if has_cross_modal and pathway_names and proto_names:
        logger.info(
            "  Analyzing per-prototype pathway attention (Signal G)..."
        )

        for proto_idx in range(n_protos):
            proto_name = proto_names[proto_idx]
            safe_name = f'crossmodal_proto_{proto_idx}'

            cm_analyzer = ImportanceAnalyzer(
                pathway_names, safe_name
            )

            for pid in valid_patients:
                attn = attention_by_patient[pid]
                # Row proto_idx of cross-modal attention: [P]
                cm_row = attn['cross_modal_attention'][proto_idx]
                cm_analyzer.add_patient(pid, cm_row, risk_map[pid])

            cm_analyzer.save_results(output_dir)
            analyzers[safe_name] = cm_analyzer

    # ---- 8. Within-pathway gene attention (Signal A columns) ----
    if has_gene_pathway and gene_names and pathway_names:
        # Determine which pathways to analyze
        if pathways_of_interest is None and 'pathway_gate' in analyzers:
            # Auto-select top pathways from gate analysis
            pw_results = analyzers['pathway_gate'].top_differential(
                k=top_k_pathways
            )
            pathways_of_interest = pw_results['entity'].tolist()
            logger.info(
                f"  Auto-selected {len(pathways_of_interest)} pathways "
                f"for within-pathway gene analysis"
            )

        if pathways_of_interest:
            logger.info(
                f"  Analyzing genes within {len(pathways_of_interest)} "
                f"pathways..."
            )

            for pw_name in pathways_of_interest:
                if pw_name not in pathway_names:
                    logger.warning(
                        f"  Pathway '{pw_name}' not found, skipping"
                    )
                    continue

                pw_idx = pathway_names.index(pw_name)

                # Identify genes participating in this pathway
                # (non-zero attention in at least one patient)
                gene_mask = np.zeros(len(gene_names), dtype=bool)
                for pid in valid_patients:
                    col = attention_by_patient[pid][
                        'gene_pathway_attention'
                    ][:, pw_idx]
                    gene_mask |= (col > 0)

                participating_genes = [
                    gene_names[i]
                    for i in range(len(gene_names)) if gene_mask[i]
                ]
                gene_indices = [
                    i for i in range(len(gene_names)) if gene_mask[i]
                ]

                if len(participating_genes) < 2:
                    continue

                # Sanitize pathway name for filename
                safe_pw = (
                    pw_name[:60]
                    .replace(' ', '_')
                    .replace('/', '_')
                    .replace(':', '_')
                )
                analysis_name = f'genes_in_{safe_pw}'

                pw_gene_analyzer = ImportanceAnalyzer(
                    participating_genes, analysis_name
                )

                for pid in valid_patients:
                    col = attention_by_patient[pid][
                        'gene_pathway_attention'
                    ][:, pw_idx]
                    pw_gene_analyzer.add_patient(
                        pid, col[gene_indices], risk_map[pid]
                    )

                pw_gene_analyzer.save_results(output_dir)
                analyzers[analysis_name] = pw_gene_analyzer

    # ---- Summary ----
    logger.info(f"\nImportance analysis complete:")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Analyses run: {len(analyzers)}")
    for name, analyzer in analyzers.items():
        if isinstance(analyzer, ImportanceAnalyzer):
            results = analyzer.rank_analysis()
            n_sig = results['significant'].sum()
            logger.info(
                f"    {name}: {len(results)} entities, "
                f"{n_sig} significant"
            )
        elif isinstance(analyzer, PrototypeShiftAnalyzer):
            results = analyzer.analyze_shift()
            n_sig = results['significant'].sum()
            logger.info(
                f"    {name}: {len(results)} prototypes, "
                f"{n_sig} with significant shift"
            )

    return analyzers


# ============================================================================
# Utilities
# ============================================================================

def _compute_ranks(scores: np.ndarray) -> np.ndarray:
    """
    Convert importance scores to ranks within a patient.

    Higher score → higher rank.
    Ties are handled by average rank (consistent with Mann-Whitney U
    assumptions).

    Args:
        scores: 1D array of importance scores.

    Returns:
        1D array of ranks (1-based, higher = more important).
    """
    from scipy.stats import rankdata
    return rankdata(scores, method='average')