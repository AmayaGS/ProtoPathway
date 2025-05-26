# utils/gene_importance_analyzer.py
# utils/vis_results.py

import os
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.stats import ttest_ind, mannwhitneyu


class GenePathwayAnalyzer:
    """
    Unified analyzer for gene, pathway, and cross-modal importance analysis
    """

    def __init__(self, gene_idx, pathway_idx, analysis_type='gene'):
        """
        Initialize analyzer

        Args:
            gene_idx: Dictionary mapping gene names to indices
            pathway_idx: Dictionary mapping pathway names to indices
            analysis_type: 'gene', 'pathway', or 'crossmodal'
        """
        self.analysis_type = analysis_type
        self.idx_to_gene = {v.item(): k for k, v in gene_idx.items()}
        self.idx_to_pathway = {v.item(): k for k, v in pathway_idx.items()}
        self.patient_data = {}
        self.class_data = defaultdict(list)

        # Set analysis-specific parameters
        self._setup_analysis_params()

    def _setup_analysis_params(self):
        """Setup analysis-specific parameters"""
        if self.analysis_type == 'gene':
            self.entity_name = 'gene'
            self.idx_to_entity = self.idx_to_gene
            self.aggregate_func = lambda x: x.sum(dim=1)  # Sum across pathways
        elif self.analysis_type == 'pathway':
            self.entity_name = 'pathway'
            self.idx_to_entity = self.idx_to_pathway
            self.aggregate_func = lambda x: x.sum(dim=0)  # Sum across genes
        elif self.analysis_type == 'crossmodal':
            self.entity_name = 'pathway'
            self.idx_to_entity = self.idx_to_pathway
            self.aggregate_func = lambda x: x.sum(dim=1).squeeze()  # Sum across prototypes
        else:
            raise ValueError("analysis_type must be 'gene', 'pathway', or 'crossmodal'")

    def add_patient(self, patient_id, attention_data, label=None):
        """
        Add patient data to analyzer

        Args:
            patient_id: Patient identifier
            attention_data: Attention tensor (gene_pathway_attn or cross_modal_attn)
            label: Class label tensor
        """
        # Calculate importance scores
        if self.analysis_type == 'crossmodal':
            importance_raw = attention_data.sum(dim=1).squeeze().cpu().numpy()
        else:
            importance_raw = self.aggregate_func(attention_data).cpu().numpy()

        # Normalize within patient (relative importance)
        importance_norm = importance_raw / (importance_raw.sum() + 1e-8)

        self.patient_data[patient_id] = {
            'importance_raw': importance_raw,
            'importance_norm': importance_norm,
            'label': label.item() if label is not None else None
        }

        if label is not None:
            self.class_data[label.item()].append((patient_id, importance_norm))

    def top_entities(self, patient_id, k=10000, use_normalized=True):
        """Get top entities for a patient"""
        scores = (self.patient_data[patient_id]['importance_norm'] if use_normalized
                  else self.patient_data[patient_id]['importance_raw'])
        top_idx = np.argsort(scores)[-k:][::-1]

        return pd.DataFrame({
            self.entity_name: [self.idx_to_entity[i] for i in top_idx],
            'score': scores[top_idx]
        })

    def save_patient_results(self, output_dir):
        """Save individual patient results to CSV files"""
        os.makedirs(output_dir, exist_ok=True)

        for pid, data in self.patient_data.items():
            df = pd.DataFrame({
                self.entity_name: [self.idx_to_entity[i] for i in range(len(data['importance_raw']))],
                'importance': data['importance_raw'],
                'importance_norm': data['importance_norm']
            })
            filename = f"patient_{pid}_label_{data['label']}_{self.analysis_type}_{self.entity_name}s.csv"
            df.to_csv(os.path.join(output_dir, filename), index=False)

    def class_aggregation(self, k=10000, use_normalized=True):
        """Get top entities by class with mean importance scores"""
        results = {}

        for label, patient_list in self.class_data.items():
            if use_normalized:
                scores = np.stack([self.patient_data[pid]['importance_norm']
                                   for pid, _ in patient_list])
            else:
                scores = np.stack([scores for _, scores in patient_list])

            mean_scores = scores.mean(axis=0)
            top_idx = np.argsort(mean_scores)[-k:][::-1]

            results[label] = pd.DataFrame({
                self.entity_name: [self.idx_to_entity[i] for i in top_idx],
                'mean_importance': mean_scores[top_idx],
                'n_patients': len(patient_list)
            })

        return results

    def class_differences(self, k=10000, use_normalized=True):
        """Find entities with biggest differences between classes"""
        if len(self.class_data) != 2:
            raise ValueError("Need exactly 2 classes for comparison")

        labels = sorted(list(self.class_data.keys()))

        if use_normalized:
            class0_scores = np.stack([self.patient_data[pid]['importance_norm']
                                      for pid, _ in self.class_data[labels[0]]])
            class1_scores = np.stack([self.patient_data[pid]['importance_norm']
                                      for pid, _ in self.class_data[labels[1]]])
        else:
            class0_scores = np.stack([scores for _, scores in self.class_data[labels[0]]])
            class1_scores = np.stack([scores for _, scores in self.class_data[labels[1]]])

        mean0 = class0_scores.mean(axis=0)
        mean1 = class1_scores.mean(axis=0)
        diff = mean1 - mean0
        fold_change = mean1 / (mean0 + 1e-8)

        # Calculate Cohen's d (effect size)
        pooled_std = np.sqrt(((class0_scores.var(axis=0) * (len(class0_scores) - 1)) +
                              (class1_scores.var(axis=0) * (len(class1_scores) - 1))) /
                             (len(class0_scores) + len(class1_scores) - 2))
        cohens_d = (mean1 - mean0) / (pooled_std + 1e-8)

        # T-test for each entity
        p_values = []
        for i in range(len(mean0)):
            _, p = ttest_ind(class0_scores[:, i], class1_scores[:, i])
            p_values.append(p)

        # Create results dataframe
        results = pd.DataFrame({
            self.entity_name: [self.idx_to_entity[i] for i in range(len(mean0))],
            f'mean_class_{labels[0]}': mean0,
            f'mean_class_{labels[1]}': mean1,
            'difference': diff,
            'cohens_d': cohens_d,
            'abs_cohens_d': np.abs(cohens_d),
            'fold_change': fold_change,
            'log2_fold_change': np.log2((mean1 + 1e-8) / (mean0 + 1e-8)),
            'p_value': p_values,
            'dominant_class': np.where(mean1 > mean0, labels[1], labels[0])
        })

        results['significant'] = results['p_value'] < 0.05
        results = results.sort_values('abs_cohens_d', ascending=False)

        return results

    def class_specific_drivers(self, k=100, effect_size_threshold=0.2):
        """Get entities that are significantly more important in each class"""
        diff_results = self.class_differences(k=1000, use_normalized=True)
        significant = diff_results[
            (diff_results['significant'] == True) &
            (diff_results['abs_cohens_d'] > effect_size_threshold)
            ]

        labels = sorted(list(self.class_data.keys()))

        class0_drivers = significant[significant['dominant_class'] == labels[0]]
        class1_drivers = significant[significant['dominant_class'] == labels[1]]

        return {
            f'class_{labels[0]}_drivers': class0_drivers,
            f'class_{labels[1]}_drivers': class1_drivers,
            'summary': {
                f'class_{labels[0]}_{self.entity_name}s': len(class0_drivers),
                f'class_{labels[1]}_{self.entity_name}s': len(class1_drivers),
                'total_significant': len(significant)
            }
        }

    def rank_based_analysis(self, k=1000):
        """Compare entity rankings within patients between classes"""
        labels = sorted(list(self.class_data.keys()))

        # Get ranks for each patient using normalized scores
        class0_ranks = []
        class1_ranks = []

        for label in labels:
            for pid, _ in self.class_data[label]:
                scores = self.patient_data[pid]['importance_norm']
                ranks = len(scores) - np.argsort(np.argsort(scores))  # Higher score = higher rank

                if label == labels[0]:
                    class0_ranks.append(ranks)
                else:
                    class1_ranks.append(ranks)

        class0_ranks = np.array(class0_ranks)
        class1_ranks = np.array(class1_ranks)

        # Compare ranks for each entity
        results = []
        for i in range(class0_ranks.shape[1]):
            statistic, p_value = mannwhitneyu(
                class1_ranks[:, i], class0_ranks[:, i],
                alternative='two-sided'
            )

            mean_rank_0 = class0_ranks[:, i].mean()
            mean_rank_1 = class1_ranks[:, i].mean()

            results.append({
                self.entity_name: self.idx_to_entity[i],
                f'mean_rank_class_{labels[0]}': mean_rank_0,
                f'mean_rank_class_{labels[1]}': mean_rank_1,
                'rank_difference': mean_rank_1 - mean_rank_0,
                'p_value': p_value,
                'higher_in_class': labels[1] if mean_rank_1 > mean_rank_0 else labels[0]
            })

        rank_df = pd.DataFrame(results)
        rank_df['significant'] = rank_df['p_value'] < 0.05
        rank_df = rank_df.sort_values('rank_difference', key=abs, ascending=False)

        return rank_df

    def consensus_analysis(self, output_dir, k_per_method=10000):
        """Create consensus rankings and save to CSV files"""
        # Get results from all methods
        diff_results = self.class_differences(k=k_per_method, use_normalized=True)
        enhanced_drivers = self.class_specific_drivers(k=k_per_method, effect_size_threshold=0.2)
        rank_results = self.rank_based_analysis(k=k_per_method)

        labels = sorted(list(self.class_data.keys()))

        for class_label in labels:
            # Get entities from each method
            method1_entities = set(diff_results[
                                       (diff_results['significant'] == True) &
                                       (diff_results['dominant_class'] == class_label)
                                       ][self.entity_name])

            driver_key = f'class_{class_label}_drivers'
            method2_entities = (set(enhanced_drivers[driver_key][self.entity_name])
                                if driver_key in enhanced_drivers else set())

            method3_entities = set(rank_results[
                                       (rank_results['significant'] == True) &
                                       (rank_results['higher_in_class'] == class_label)
                                       ][self.entity_name])

            # Count method agreement
            all_entities = method1_entities | method2_entities | method3_entities
            consensus_data = []

            for entity in all_entities:
                method_count = sum([
                    entity in method1_entities,
                    entity in method2_entities,
                    entity in method3_entities
                ])

                confidence = 'high' if method_count == 3 else 'medium' if method_count == 2 else 'exploratory'

                consensus_data.append({
                    self.entity_name: entity,
                    'method_count': method_count,
                    'confidence': confidence,
                    'in_statistical': entity in method1_entities,
                    'in_enhanced': entity in method2_entities,
                    'in_rank': entity in method3_entities
                })

            # Save to CSV
            consensus_df = pd.DataFrame(consensus_data)
            consensus_df = consensus_df.sort_values(['method_count', self.entity_name], ascending=[False, True])

            output_file = os.path.join(output_dir,
                                       f'class_{class_label}_consensus_{self.analysis_type}_{self.entity_name}s.csv')
            consensus_df.to_csv(output_file, index=False)

        return f"{self.analysis_type.capitalize()} {self.entity_name} consensus analysis saved to {output_dir}"