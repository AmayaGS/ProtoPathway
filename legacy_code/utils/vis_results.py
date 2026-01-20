# utils/gene_importance_analyzer.py
# utils/vis_results.py

import os
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.stats import ttest_ind, mannwhitneyu
import matplotlib.pyplot as plt



class GenePathwayAnalyzer:
    """
    Unified analyzer for gene, pathway, and cross-modal importance analysis
    """

    def __init__(self, gene_idx, pathway_idx, analysis_type='gene_average'):
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
        if self.analysis_type == 'gene_sum':
            self.entity_name = 'gene_sum'
            self.idx_to_entity = self.idx_to_gene
            self.aggregate_func = lambda x: x.sum(dim=1)
        elif self.analysis_type == 'gene_average':
            self.entity_name = 'gene_average'
            self.idx_to_entity = self.idx_to_gene
            self.aggregate_func = self._gene_average_importance
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
        # # Calculate importance scores
        # if self.analysis_type == 'crossmodal':
        #     importance_raw = attention_data.sum(dim=1).squeeze().cpu().numpy()
        # else:
        #     importance_raw = self.aggregate_func(attention_data).cpu().numpy()

        importance_raw = self.aggregate_func(attention_data).cpu().numpy()

        # Normalize within patient (relative importance)
        importance_norm = importance_raw / (importance_raw.sum() + 1e-8)

        self.patient_data[patient_id] = {
            'importance_raw': importance_raw,
            'importance_norm': importance_norm,
            'label': label.item() if label is not None else None,
            'raw_attention_matrix': attention_data.cpu().numpy()
        }

        if label is not None:
            self.class_data[label.item()].append((patient_id, importance_norm))

    def _gene_average_importance(self, gene_pathway_matrix):
        """Calculate average gene importance across pathways where gene participates"""
        eps = 1e-8

        # Count non-zero connections (pathways each gene participates in)
        nonzero_mask = gene_pathway_matrix > eps
        pathway_counts = nonzero_mask.sum(dim=1).float()  # [num_genes]

        # Sum across pathways
        total_attention = gene_pathway_matrix.sum(dim=1)  # [num_genes]

        # Average attention per pathway participation
        avg_attention = total_attention / (pathway_counts + eps)

        return avg_attention

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

    def plot_pathway_gene_class_differences(self, pathway_name, top_k=15, output_path=None,
                                            plot_type='bar', figsize=(12, 8)):
        """
        Plot gene attention differences between classes for a specific pathway
        """

        if self.analysis_type not in ['gene_sum', 'gene_average']:
            raise ValueError("This function only works with gene analysis types")

        # Find pathway index
        pathway_idx = None
        for idx, name in self.idx_to_pathway.items():
            if name == pathway_name:
                pathway_idx = idx
                break

        if pathway_idx is None:
            available_pathways = list(self.idx_to_pathway.values())[:10]  # Show first 10
            raise ValueError(f"Pathway '{pathway_name}' not found. Available pathways include: {available_pathways}...")

        # Collect gene attention data for this pathway by class
        class_gene_data = {0: [], 1: []}  # Assuming binary classification

        for patient_id, patient_data in self.patient_data.items():
            label = patient_data['label']
            if label is None:
                continue

            # Get gene attention for this pathway from raw matrix
            raw_matrix = patient_data['raw_attention_matrix']  # [num_genes, num_pathways]
            pathway_gene_attention = raw_matrix[:, pathway_idx]  # [num_genes]

            class_gene_data[label].append(pathway_gene_attention)

        # Convert to numpy arrays and calculate means
        class_means = {}
        class_stds = {}

        for label in [0, 1]:
            if len(class_gene_data[label]) > 0:
                class_array = np.array(class_gene_data[label])  # [n_patients, n_genes]
                class_means[label] = class_array.mean(axis=0)  # [n_genes]
                class_stds[label] = class_array.std(axis=0)  # [n_genes]
            else:
                print(f"⚠️ No patients found for class {label}")
                return None

        # Calculate differences and find top genes
        mean_diff = class_means[1] - class_means[0]  # Class 1 - Class 0
        top_gene_indices = np.argsort(np.abs(mean_diff))[-top_k:][::-1]

        # Prepare data for plotting
        gene_names = [self.idx_to_gene[i] for i in top_gene_indices]
        class_0_values = class_means[0][top_gene_indices]
        class_1_values = class_means[1][top_gene_indices]
        differences = mean_diff[top_gene_indices]

        # Create plot
        plt.figure(figsize=figsize)

        if plot_type == 'bar':
            x = np.arange(len(gene_names))
            width = 0.35

            plt.bar(x - width / 2, class_0_values, width, label='Class 0', alpha=0.8, color='steelblue')
            plt.bar(x + width / 2, class_1_values, width, label='Class 1', alpha=0.8, color='coral')

            plt.xlabel('Genes', fontsize=12)
            plt.ylabel('Mean Attention Score', fontsize=12)
            plt.title(f'Gene Attention Patterns in {pathway_name}\nClass Comparison (Top {top_k} genes)', fontsize=14)
            plt.xticks(x, [name[:20] + '...' if len(name) > 20 else name for name in gene_names],
                       rotation=45, ha='right')
            plt.legend()
            plt.grid(axis='y', alpha=0.3)

        elif plot_type == 'difference':
            # Plot the differences directly
            colors = ['steelblue' if d < 0 else 'coral' for d in differences]
            bars = plt.barh(range(len(gene_names)), differences, color=colors, alpha=0.7)

            plt.yticks(range(len(gene_names)), [name[:25] + '...' if len(name) > 25 else name for name in gene_names])
            plt.xlabel('Attention Difference (Class 1 - Class 0)', fontsize=12)
            plt.title(f'Gene Attention Differences in {pathway_name}\n(Positive = Higher in High Inflammatory)', fontsize=14)
            plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            plt.grid(axis='x', alpha=0.3)

            # # Add value labels
            # for i, (bar, diff) in enumerate(zip(bars, differences)):
            #     plt.text(diff + (0.001 if diff > 0 else -0.001), bar.get_y() + bar.get_height() / 2,
            #              f'{diff:.3f}', ha='left' if diff > 0 else 'right', va='center', fontsize=9)

        plt.tight_layout()

        # Save if path provided
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            print(f"Saved plot to {output_path}")

        plt.show()

        # Return summary statistics
        return {
            'pathway': pathway_name,
            'pathway_idx': pathway_idx,
            'n_class_0_patients': len(class_gene_data[0]),
            'n_class_1_patients': len(class_gene_data[1]),
            'top_genes': gene_names,
            'class_0_means': class_0_values,
            'class_1_means': class_1_values,
            'differences': differences,
            'max_difference_gene': gene_names[0],
            'max_difference_value': differences[0]
        }

class PathwayGateAnalyzer:
    """
    Analyzer for actual pathway gating weights (not gene-pathway aggregation)
    """

    def __init__(self, pathway_idx):
        """
        Initialize pathway gate analyzer

        Args:
            pathway_idx: Dictionary mapping pathway names to indices
        """
        self.idx_to_pathway = {v.item(): k for k, v in pathway_idx.items()}
        self.patient_data = {}
        self.class_data = defaultdict(list)

    def add_patient(self, patient_id, pathway_gates, label=None):
        """
        Add patient pathway gating data

        Args:
            patient_id: Patient identifier
            pathway_gates: Pathway gating weights tensor [num_pathways]
            label: Class label
        """
        gates_squeezed = pathway_gates.squeeze()
        gates_raw = gates_squeezed.cpu().numpy()

        # # Normalize (these should already be softmax normalized, but just in case)
        # gates_norm = gates_raw / (gates_raw.sum() + 1e-8)

        self.patient_data[patient_id] = {
            'gates_raw': gates_raw,
            # 'gates_norm': gates_norm,
            'label': label.item() if label is not None else None
        }

        if label is not None:
            self.class_data[label.item()].append((patient_id, gates_raw))

    def top_pathways(self, patient_id, k=15):
        """Get top pathways for a patient"""
        gates = self.patient_data[patient_id]['gates_norm']
        top_idx = np.argsort(gates)[-k:][::-1]

        return pd.DataFrame({
            'pathway': [self.idx_to_pathway[i] for i in top_idx],
            'importance': gates[top_idx]
        })

    def class_aggregation(self, k=15):
        """Get top pathways by class"""
        results = {}

        for label, patient_list in self.class_data.items():
            if len(patient_list) == 0:
                continue

            # Average pathway gates across patients in this class
            all_gates = np.array([gates for _, gates in patient_list])
            mean_gates = all_gates.mean(axis=0)
            std_gates = all_gates.std(axis=0)

            # Get top k pathways
            top_idx = np.argsort(mean_gates)[-k:][::-1]

            results[label] = pd.DataFrame({
                'pathway': [self.idx_to_pathway[i] for i in top_idx],
                'mean_importance': mean_gates[top_idx],
                'std_importance': std_gates[top_idx],
                'n_patients': len(patient_list)
            })

        return results

    def calculate_statistical_differences(self, alpha=0.05):
        """Calculate Cohen's d and p-values for pathway differences between classes"""
        from scipy import stats
        import numpy as np

        if len(self.class_data[0]) == 0 or len(self.class_data[1]) == 0:
            print("Need both classes for statistical comparison")
            return None

        # Get data for both classes
        class_0_gates = np.array([gates for _, gates in self.class_data[0]])  # [n_patients_0, n_pathways]
        class_1_gates = np.array([gates for _, gates in self.class_data[1]])  # [n_patients_1, n_pathways]

        n_pathways = class_0_gates.shape[1]

        results = []
        for pathway_idx in range(n_pathways):
            # Get data for this pathway
            class_0_values = class_0_gates[:, pathway_idx]
            class_1_values = class_1_gates[:, pathway_idx]

            # Calculate statistics
            mean_0, std_0 = class_0_values.mean(), class_0_values.std()
            mean_1, std_1 = class_1_values.mean(), class_1_values.std()

            # Cohen's d (effect size)
            pooled_std = np.sqrt(((len(class_0_values) - 1) * std_0 ** 2 +
                                  (len(class_1_values) - 1) * std_1 ** 2) /
                                 (len(class_0_values) + len(class_1_values) - 2))
            cohens_d = (mean_1 - mean_0) / (pooled_std + 1e-8)

            # Statistical test (Mann-Whitney U for non-parametric)
            statistic, p_value = stats.mannwhitneyu(class_0_values, class_1_values,
                                                    alternative='two-sided')

            # t-test as alternative
            t_stat, t_pvalue = stats.ttest_ind(class_0_values, class_1_values)

            results.append({
                'pathway_idx': pathway_idx,
                'pathway_name': self.idx_to_pathway[pathway_idx],
                'class_0_mean': mean_0,
                'class_1_mean': mean_1,
                'class_0_std': std_0,
                'class_1_std': std_1,
                'mean_difference': mean_1 - mean_0,
                'cohens_d': cohens_d,
                'mannwhitney_p': p_value,
                'ttest_p': t_pvalue,
                'significant': p_value < alpha,
                'effect_size_category': self._categorize_effect_size(abs(cohens_d))
            })

        return pd.DataFrame(results)

    def _categorize_effect_size(self, abs_cohens_d):
        """Categorize Cohen's d effect size"""
        if abs_cohens_d < 0.2:
            return 'negligible'
        elif abs_cohens_d < 0.5:
            return 'small'
        elif abs_cohens_d < 0.8:
            return 'medium'
        else:
            return 'large'

    def plot_statistical_pathway_differences(self, top_k=15, output_path=None,
                                             plot_type='cohens_d', alpha=0.05):
        """
        Plot pathway differences using statistical measures

        Args:
            plot_type: 'cohens_d', 'fold_change', or 'rank_difference'
        """

        # Calculate statistical differences
        stats_df = self.calculate_statistical_differences(alpha=alpha)
        if stats_df is None:
            return None

        # Sort by different criteria based on plot type
        if plot_type == 'cohens_d':
            stats_df_sorted = stats_df.sort_values('cohens_d', key=abs, ascending=False)
            metric_col = 'cohens_d'
            title = "Pathway Importance: Cohen's D Effect Sizes"
            xlabel = "Cohen's D (Standardized Effect Size)"

        elif plot_type == 'fold_change':
            # Calculate fold change (Class 1 / Class 0)
            stats_df['fold_change'] = np.log2((stats_df['class_1_mean'] + 1e-8) /
                                              (stats_df['class_0_mean'] + 1e-8))
            stats_df_sorted = stats_df.sort_values('fold_change', key=abs, ascending=False)
            metric_col = 'fold_change'
            title = "Pathway Importance: Log2 Fold Changes"
            xlabel = "Log2 Fold Change (Class 1 / Class 0)"

        elif plot_type == 'rank_difference':
            # Calculate relative ranks within each class
            class_0_ranks = stats_df['class_0_mean'].rank(ascending=False)
            class_1_ranks = stats_df['class_1_mean'].rank(ascending=False)
            stats_df['rank_difference'] = class_1_ranks - class_0_ranks     # Positive = higher rank in class 0
            stats_df_sorted = stats_df.sort_values('rank_difference', key=abs, ascending=False)
            metric_col = 'rank_difference'
            title = "Pathway Importance: Rank Differences Between Classes"
            xlabel = "Rank Difference (Class 1 Rank - Class 0 Rank)"

        # Get top pathways
        top_pathways = stats_df_sorted.head(top_k)

        # Create plot
        plt.figure(figsize=(12, max(8, top_k * 0.4)))

        # Get plot data
        pathway_names = top_pathways['pathway_name'].values
        metric_values = top_pathways[metric_col].values
        p_values = top_pathways['mannwhitney_p'].values

        # Color by significance and direction
        colors = []
        # for val, p_val in zip(metric_values, p_values):
        #     if p_val < alpha:  # Significant
        #         colors.append('darkgreen' if val > 0 else 'darkred')
        #     else:  # Not significant
        #         colors.append('lightgreen' if val > 0 else 'lightcoral')

        for val, p_val in zip(metric_values, p_values):
            colors.append('coral' if val > 0 else 'steelblue')

        # Horizontal bar plot
        bars = plt.barh(range(len(pathway_names)), metric_values, color=colors, alpha=0.8)

        # Formatting
        plt.yticks(range(len(pathway_names)),
                   [name[:40] + '...' if len(name) > 40 else name for name in pathway_names])
        plt.xlabel(xlabel, fontsize=12)
        plt.title(f'{title}', fontsize=14)
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        plt.grid(axis='x', alpha=0.3)

        # # Add significance indicators
        # for i, (bar, p_val, cohens_d) in enumerate(zip(bars, p_values, top_pathways['cohens_d'])):
        #     if p_val < alpha:
        #         plt.text(bar.get_width() + 0.01 * max(abs(metric_values)),
        #                  bar.get_y() + bar.get_height() / 2,
        #                  f'**', ha='left', va='center', fontsize=12, fontweight='bold')

        # # Add legend
        # from matplotlib.patches import Patch
        # legend_elements = [
        #     Patch(facecolor='darkgreen', alpha=0.8, label=f'Higher in Class 1 (p<{alpha})'),
        #     Patch(facecolor='darkred', alpha=0.8, label=f'Higher in Class 0 (p<{alpha})'),
        #     Patch(facecolor='lightgreen', alpha=0.8, label='Higher in Class 1 (n.s.)'),
        #     Patch(facecolor='lightcoral', alpha=0.8, label='Higher in Class 0 (n.s.)')
        # ]
        # plt.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1, 0))

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            print(f"Saved statistical pathway plot to {output_path}")

        plt.show()

        return {
            'statistics': stats_df_sorted,
            'top_pathways': top_pathways,
            'significant_pathways': top_pathways[top_pathways['significant']],
            'n_significant': sum(top_pathways['significant'])
        }

    def plot_pathway_class_differences(self, top_k=15, output_path=None, plot_type='difference'):
        """
        Plot pathway gating weight differences between classes
        """
        # Calculate class means
        class_means = {}
        class_stds = {}

        for label in [0, 1]:
            if label in self.class_data and len(self.class_data[label]) > 0:
                all_gates = np.array([gates for _, gates in self.class_data[label]])
                class_means[label] = all_gates.mean(axis=0)
                class_stds[label] = all_gates.std(axis=0)
            else:
                print(f"⚠️ No patients found for class {label}")
                return None

        # Calculate differences and get top pathways
        mean_diff = class_means[1] - class_means[0]  # Class 1 - Class 0
        top_pathway_indices = np.argsort(np.abs(mean_diff))[-top_k:][::-1].astype(int)

        # Prepare plot data
        pathway_names = [self.idx_to_pathway[i] for i in top_pathway_indices]
        differences = mean_diff[top_pathway_indices]
        class_0_values = class_means[0][top_pathway_indices]
        class_1_values = class_means[1][top_pathway_indices]

        # Create plot
        plt.figure(figsize=(12, max(8, top_k * 0.4)))

        if plot_type == 'difference':
            colors = ['steelblue' if d < 0 else 'coral' for d in differences]
            bars = plt.barh(range(len(pathway_names)), differences, color=colors, alpha=0.7)

            plt.yticks(range(len(pathway_names)),
                       [name[:40] + '...' if len(name) > 40 else name for name in pathway_names])
            plt.xlabel('Pathway Gating Weight Difference (Class 1 - Class 0)', fontsize=12)
            plt.title(f'Pathway Importance Differences Between Classes\n(Top {top_k} Most Different Pathways)',
                      fontsize=14)
            plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            plt.grid(axis='x', alpha=0.3)

        elif plot_type == 'bar':
            x = np.arange(len(pathway_names))
            width = 0.35

            plt.bar(x - width / 2, class_0_values, width, label='Class 0', alpha=0.8, color='steelblue')
            plt.bar(x + width / 2, class_1_values, width, label='Class 1', alpha=0.8, color='coral')

            plt.xlabel('Pathways', fontsize=12)
            plt.ylabel('Mean Gating Weight', fontsize=12)
            plt.title(f'Pathway Gating Weights by Class\n(Top {top_k} Most Different)', fontsize=14)
            plt.xticks(x, [name[:25] + '...' if len(name) > 25 else name for name in pathway_names],
                       rotation=45, ha='right')
            plt.legend()
            plt.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved pathway gate plot to {output_path}")

        plt.show()

        return {
            'top_pathways': pathway_names,
            'differences': differences,
            'class_0_means': class_0_values,
            'class_1_means': class_1_values,
            'max_difference_pathway': pathway_names[0],
            'max_difference_value': differences[0]
        }

    def pathway_rank_differences(self, k=20):
        """Simple rank difference calculation and return DataFrame for saving"""

        if len(self.class_data[0]) == 0 or len(self.class_data[1]) == 0:
            print("Need both classes for comparison")
            return None

        # Get mean importance for each class
        class_0_gates = np.array([gates for _, gates in self.class_data[0]])
        class_1_gates = np.array([gates for _, gates in self.class_data[1]])

        class_0_mean = class_0_gates.mean(axis=0)
        class_1_mean = class_1_gates.mean(axis=0)

        # Calculate ranks (1 = most important)
        class_0_ranks = pd.Series(class_0_mean).rank(ascending=False)
        class_1_ranks = pd.Series(class_1_mean).rank(ascending=False)

        # Create results DataFrame
        results = pd.DataFrame({
            'pathway': [self.idx_to_pathway[i] for i in range(len(class_0_mean))],
            'class_0_mean': class_0_mean,
            'class_1_mean': class_1_mean,
            'class_0_rank': class_0_ranks.values,
            'class_1_rank': class_1_ranks.values,
            'rank_difference': (class_1_ranks - class_0_ranks).values
        })

        # Sort by absolute rank difference
        results = results.sort_values('rank_difference', key=abs, ascending=False)

        return results.head(k)



def plot_pathway_gates_from_csv(csv_path, top_k=15, plot_type='rank_difference',
                                output_path=None, figsize=(12, 8)):
    """
    Create pathway gate plots directly from CSV file

    Args:
        csv_path: Path to pathway_gates_rank_differences.csv
        top_k: Number of top pathways to show
        plot_type: 'rank_difference', 'bar_comparison', or 'both'
        output_path: Where to save plot
        figsize: Figure size
    """

    # Read the CSV
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    print(f"📊 Loaded {len(df)} pathways from {csv_path}")

    # Get top pathways
    df_sorted = df.reindex(df['rank_difference'].abs().sort_values(ascending=False).index)
    top_pathways = df_sorted.head(top_k).copy()

    top_pathways = top_pathways.sort_values('rank_difference', ascending=True)

    pathway_names = top_pathways['pathway'].values
    rank_differences = top_pathways['rank_difference'].values
    class_0_means = top_pathways['class_0_mean'].values
    class_1_means = top_pathways['class_1_mean'].values

    if plot_type == 'rank_difference':
        # Create rank difference plot
        plt.figure(figsize=figsize)

        # Color by direction
        colors = ['steelblue' if d < 0 else 'coral' for d in rank_differences]

        bars = plt.barh(range(len(pathway_names)), rank_differences, color=colors, alpha=0.8)

        plt.yticks(range(len(pathway_names)),
                   [name[:40] + '...' if len(name) > 40 else name for name in pathway_names])
        plt.xlabel(
            '← Higher Priority in Class 0  |  Higher Priority in Class 1 →',
            fontsize=12)
        plt.title(f'Pathway Gate Importance Rank Differences', fontsize=14)
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        plt.grid(axis='x', alpha=0.3)

        # # Add legend
        # from matplotlib.patches import Patch
        # legend_elements = [
        #     Patch(facecolor='darkgreen', alpha=0.8, label='Higher Priority in Class 1'),
        #     Patch(facecolor='darkred', alpha=0.8, label='Higher Priority in Class 0')
        # ]
        # plt.legend(handles=legend_elements, loc='lower right')

    elif plot_type == 'bar_comparison':
        # Create side-by-side comparison
        plt.figure(figsize=figsize)

        x = np.arange(len(pathway_names))
        width = 0.35

        plt.bar(x - width / 2, class_0_means, width, label='Class 0', alpha=0.8, color='steelblue')
        plt.bar(x + width / 2, class_1_means, width, label='Class 1', alpha=0.8, color='coral')

        plt.xlabel('Pathways', fontsize=12)
        plt.ylabel('Mean Gating Weight', fontsize=12)
        plt.title(f'Pathway Gating Weights by Class\n(Top {top_k} Most Different)', fontsize=14)
        plt.xticks(x, [name[:25] + '...' if len(name) > 25 else name for name in pathway_names],
                   rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)

    elif plot_type == 'both':
        # Create subplot with both plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(figsize[0], figsize[1] * 1.5))

        # Top plot: Rank differences
        colors = ['darkred' if d < 0 else 'darkgreen' for d in rank_differences]
        ax1.barh(range(len(pathway_names)), rank_differences, color=colors, alpha=0.8)
        ax1.set_yticks(range(len(pathway_names)))
        ax1.set_yticklabels([name[:30] + '...' if len(name) > 30 else name for name in pathway_names])
        ax1.set_xlabel('Rank Difference (Class 1 - Class 0)')
        ax1.set_title('Pathway Rank Differences')
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax1.grid(axis='x', alpha=0.3)

        # Bottom plot: Mean comparison
        x = np.arange(len(pathway_names))
        width = 0.35
        ax2.bar(x - width / 2, class_0_means, width, label='Class 0', alpha=0.8, color='steelblue')
        ax2.bar(x + width / 2, class_1_means, width, label='Class 1', alpha=0.8, color='coral')
        ax2.set_xticks(x)
        ax2.set_xticklabels([name[:20] + '...' if len(name) > 20 else name for name in pathway_names],
                            rotation=45, ha='right')
        ax2.set_ylabel('Mean Gating Weight')
        ax2.set_title('Mean Gating Weights by Class')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved plot to {output_path}")

    plt.show()

    # Return summary stats
    return {
        'top_pathways': pathway_names.tolist(),
        'rank_differences': rank_differences.tolist(),
        'max_difference': max(abs(rank_differences)),
        'class_1_higher': sum(rank_differences > 0),
        'class_0_higher': sum(rank_differences < 0)
    }


class PrototypeGateAnalyzer:
    """
    Analyzer for prototype gating weights
    """

    def __init__(self, num_prototypes):
        """
        Initialize prototype gate analyzer

        Args:
            num_prototypes: Number of prototypes in the model
        """
        self.num_prototypes = num_prototypes
        self.patient_data = {}
        self.class_data = defaultdict(list)

    def add_patient(self, patient_id, prototype_gates, label=None):
        """
        Add patient prototype gating data

        Args:
            patient_id: Patient identifier
            prototype_gates: Prototype gating weights tensor [num_prototypes]
            label: Class label
        """
        # Squeeze and convert to numpy
        gates_squeezed = prototype_gates.squeeze()
        gates_raw = gates_squeezed.cpu().numpy()

        # Normalize
        # gates_norm = ((gates_raw / (gates_raw.sum() + 1e-8)) * 10 ) + 0.154
        gates_norm = ((gates_raw / (gates_raw.sum() + 1e-8)) * 10)

        self.patient_data[patient_id] = {
            'gates_norm': gates_norm,
            'label': label.item() if label is not None else None
        }

        if label is not None:
            self.class_data[label.item()].append((patient_id, gates_norm))

    def global_prototype_importance(self, k=20):
        """Get globally most important prototypes across all patients"""

        if len(self.patient_data) == 0:
            return None

        # Average across all patients
        all_gates = np.array([data['gates_norm'] for data in self.patient_data.values()])
        global_mean = all_gates.mean(axis=0)
        global_std = all_gates.std(axis=0)

        # Get top prototypes
        top_indices = np.argsort(global_mean)[-k:][::-1]

        results = pd.DataFrame({
            'prototype': [f'Prototype {i}' for i in top_indices],
            'prototype_id': top_indices,
            'mean_importance': global_mean[top_indices],
            'std_importance': global_std[top_indices],
            'n_patients': len(self.patient_data)
        })

        return results

    def prototype_rank_differences(self, k=20, sort_by='rank_difference'):
        """
        Calculate prototype differences between classes

        Args:
            k: Number of top results to return
            sort_by: 'rank_difference', 'percentage_difference', or 'fold_change'
        """

        if len(self.class_data[0]) == 0 or len(self.class_data[1]) == 0:
            print("Need both classes for comparison")
            return None

        # Get mean importance for each class
        class_0_gates = np.array([gates for _, gates in self.class_data[0]])
        class_1_gates = np.array([gates for _, gates in self.class_data[1]])

        class_0_mean = class_0_gates.mean(axis=0)
        class_1_mean = class_1_gates.mean(axis=0)

        # Calculate ranks (1 = most important)
        class_0_ranks = pd.Series(class_0_mean).rank(ascending=False)
        class_1_ranks = pd.Series(class_1_mean).rank(ascending=False)

        # Calculate percentage difference: (Class1 - Class0) / Class0 * 100
        percentage_diff = ((class_1_mean - class_0_mean) / (class_0_mean + 1e-8)) * 100

        # Calculate fold change: log2(Class1 / Class0)
        fold_change = np.log2((class_1_mean + 1e-8) / (class_0_mean + 1e-8))

        # Create results DataFrame
        results = pd.DataFrame({
            'prototype': [f'Prototype {i}' for i in range(len(class_0_mean))],
            'prototype_id': range(len(class_0_mean)),
            'class_0_mean': class_0_mean,
            'class_1_mean': class_1_mean,
            'class_0_rank': class_0_ranks.values,
            'class_1_rank': class_1_ranks.values,
            'rank_difference': (class_1_ranks - class_0_ranks).values,
            'percentage_difference': percentage_diff,
            'fold_change': fold_change
        })

        # Sort by chosen metric
        if sort_by == 'rank_difference':
            results = results.sort_values('rank_difference', key=abs, ascending=False)
        elif sort_by == 'percentage_difference':
            results = results.sort_values('percentage_difference', key=abs, ascending=False)
        elif sort_by == 'fold_change':
            results = results.sort_values('fold_change', key=abs, ascending=False)
        else:
            raise ValueError("sort_by must be 'rank_difference', 'percentage_difference', or 'fold_change'")

        return results.head(k)

    def prepare_violin_data(self, prototype_ids=None, top_k=15):
        """
        Prepare data for violin plots showing prototype importance distributions

        Args:
            prototype_ids: Specific prototype IDs to include (optional)
            top_k: If prototype_ids not provided, use top k by global importance
        """

        if prototype_ids is None:
            # Get top prototypes by global importance
            all_importance = []
            for class_label in [0, 1]:
                if class_label in self.class_data:
                    class_gates = np.array([gates for _, gates in self.class_data[class_label]])
                    all_importance.append(class_gates)

            if len(all_importance) > 0:
                global_mean = np.concatenate(all_importance).mean(axis=0)
                prototype_ids = np.argsort(global_mean)[-top_k:][::-1]
            else:
                prototype_ids = list(range(min(top_k, self.num_prototypes)))

        # Prepare long-format data for violin plots
        violin_data = []

        for class_label in [0, 1]:
            if class_label not in self.class_data:
                continue

            for patient_id, gates in self.class_data[class_label]:
                for proto_id in prototype_ids:
                    violin_data.append({
                        'prototype_id': proto_id,
                        'prototype_name': f'P{proto_id}',
                        'class': f'Class {class_label}',
                        'class_numeric': class_label,
                        'importance': gates[proto_id],
                        'patient_id': patient_id
                    })

        return pd.DataFrame(violin_data)

    def plot_prototype_violin(self, prototype_ids=None, top_k=10, figsize=(16, 8), output_path=None):
        """
        Create violin plots with each prototype colored differently, ordered by gate value
        """

        violin_df = self.prepare_violin_data(prototype_ids, top_k)

        if len(violin_df) == 0:
            print("❌ No data available for violin plot")
            return None

        # 🔥 Calculate mean gate value for each prototype (across both classes) for ordering
        prototype_means = violin_df.groupby('prototype_id')['importance'].mean().sort_values(ascending=False)
        unique_prototypes = prototype_means.index.tolist()  # Now ordered by gate value!

        print("Prototype ordering (highest to lowest gate value):")
        for proto_id in unique_prototypes:
            print(f"  P{proto_id}: {prototype_means[proto_id]:.4f}")

        # Create subplots - one for each class
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Generate colors for each prototype (same color across classes)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_prototypes)))
        prototype_colors = dict(zip(unique_prototypes, colors))

        # Class 0 violin plot
        class_0_data = violin_df[violin_df['class_numeric'] == 0]
        if len(class_0_data) > 0:
            # Prepare data in the correct order
            class_0_data_ordered = []
            class_0_labels = []
            class_0_colors_ordered = []

            for proto_id in unique_prototypes:
                proto_data = class_0_data[class_0_data['prototype_id'] == proto_id]['importance'].values
                if len(proto_data) > 0:  # Only include if data exists
                    class_0_data_ordered.append(proto_data)
                    class_0_labels.append(f'P{proto_id}')
                    class_0_colors_ordered.append(prototype_colors[proto_id])

            if class_0_data_ordered:
                violin_parts1 = ax1.violinplot(
                    class_0_data_ordered,
                    positions=range(len(class_0_data_ordered)),
                    showmeans=True,
                    showmedians=True,
                    showextrema=True
                )

                # Color each violin by prototype
                for pc, color in zip(violin_parts1['bodies'], class_0_colors_ordered):
                    pc.set_facecolor(color)
                    pc.set_alpha(0.7)
                    pc.set_edgecolor('black')
                    pc.set_linewidth(1)

                ax1.set_xticks(range(len(class_0_labels)))
                ax1.set_xticklabels(class_0_labels, rotation=45)
                ax1.set_title('Class 0 - Highest Ranked Prototypes',
                              fontsize=14, color='steelblue', fontweight='bold')
                ax1.set_ylabel('Gating Weight', fontsize=12)
                ax1.grid(axis='y', alpha=0.3)

        # Class 1 violin plot (same ordering)
        class_1_data = violin_df[violin_df['class_numeric'] == 1]
        if len(class_1_data) > 0:
            # Prepare data in the correct order
            class_1_data_ordered = []
            class_1_labels = []
            class_1_colors_ordered = []

            for proto_id in unique_prototypes:
                proto_data = class_1_data[class_1_data['prototype_id'] == proto_id]['importance'].values
                if len(proto_data) > 0:  # Only include if data exists
                    class_1_data_ordered.append(proto_data)
                    class_1_labels.append(f'P{proto_id}')
                    class_1_colors_ordered.append(prototype_colors[proto_id])

            if class_1_data_ordered:
                violin_parts2 = ax2.violinplot(
                    class_1_data_ordered,
                    positions=range(len(class_1_data_ordered)),
                    showmeans=True,
                    showmedians=True,
                    showextrema=True
                )

                # Color each violin by prototype (same colors as Class 0)
                for pc, color in zip(violin_parts2['bodies'], class_1_colors_ordered):
                    pc.set_facecolor(color)
                    pc.set_alpha(0.7)
                    pc.set_edgecolor('black')
                    pc.set_linewidth(1)

                ax2.set_xticks(range(len(class_1_labels)))
                ax2.set_xticklabels(class_1_labels, rotation=45)
                ax2.set_title('Class 1 - Highest Ranked Prototypes',
                              fontsize=14, color='coral', fontweight='bold')
                ax2.set_ylabel('Gating Weight', fontsize=12)
                ax2.grid(axis='y', alpha=0.3)

        # Set same y-axis scale for easy comparison
        all_values = violin_df['importance'].values
        y_min, y_max = all_values.min(), all_values.max()
        y_range = y_max - y_min
        ax1.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
        ax2.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

        plt.suptitle(f'Rank Difference between Prototypes by Class',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved ordered colorful violin plot to {output_path}")

        plt.show()

        return violin_df, prototype_means  # Return both data and ordering info

def plot_prototype_importance(prototype_df, plot_type='global', top_k=15,
                              output_path=None, figsize=(12, 8)):
    """
    Plot prototype importance from DataFrame

    Args:
        prototype_df: DataFrame from global_prototype_importance() or prototype_rank_differences()
        plot_type: 'global', 'class_comparison', or 'rank_difference'
    """

    top_data = prototype_df.head(top_k)

    plt.figure(figsize=figsize)

    if plot_type == 'global':
        # Simple bar plot of global importance
        prototype_names = top_data['prototype'].values
        importance_values = top_data['mean_importance'].values

        bars = plt.bar(range(len(prototype_names)), importance_values,
                       color='steelblue', alpha=0.7)

        plt.xticks(range(len(prototype_names)),
                   [f'P{top_data.iloc[i]["prototype_id"]}' for i in range(len(prototype_names))],
                   rotation=45)
        plt.ylabel('Mean Importance Score', fontsize=12)
        plt.title(f'Top {top_k} Most Important Prototypes (Global)', fontsize=14)
        plt.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, val in zip(bars, importance_values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    elif plot_type == 'class_comparison':
        # Side-by-side comparison
        prototype_names = top_data['prototype'].values
        class_0_values = top_data['class_0_mean'].values
        class_1_values = top_data['class_1_mean'].values

        x = np.arange(len(prototype_names))
        width = 0.35

        plt.bar(x - width / 2, class_0_values, width, label='Class 0', alpha=0.8, color='steelblue')
        plt.bar(x + width / 2, class_1_values, width, label='Class 1', alpha=0.8, color='coral')

        plt.xlabel('Prototypes', fontsize=12)
        plt.ylabel('Mean Gating Weight', fontsize=12)
        plt.title(f'Prototype Importance by Class\n(Top {top_k} Most Different)', fontsize=14)
        plt.xticks(x, [f'P{top_data.iloc[i]["prototype_id"]}' for i in range(len(prototype_names))],
                   rotation=45)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)

    elif plot_type == 'rank_difference':
        # Rank difference plot (horizontal bars)
        prototype_names = top_data['prototype'].values
        rank_differences = top_data['rank_difference'].values

        # Color by direction
        colors = ['darkred' if d < 0 else 'darkgreen' for d in rank_differences]

        bars = plt.barh(range(len(prototype_names)), rank_differences, color=colors, alpha=0.8)

        plt.yticks(range(len(prototype_names)),
                   [f'P{top_data.iloc[i]["prototype_id"]}' for i in range(len(prototype_names))])
        plt.xlabel(
            'Rank Difference (Class 1 Rank - Class 0 Rank)\n← Higher Priority in Class 0  |  Higher Priority in Class 1 →',
            fontsize=12)
        plt.title(f'Prototype Importance Rank Differences\n(Top {top_k} Most Different)', fontsize=14)
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        plt.grid(axis='x', alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='darkgreen', alpha=0.8, label='Higher Priority in Class 1'),
            Patch(facecolor='darkred', alpha=0.8, label='Higher Priority in Class 0')
        ]
        plt.legend(handles=legend_elements, loc='lower right')

    elif plot_type == 'percentage_difference':
        # Show percentage difference: (Class1 - Class0) / Class0 * 100
        prototype_names = top_data['prototype'].values
        class_0_values = top_data['class_0_mean'].values
        class_1_values = top_data['class_1_mean'].values

        # Calculate percentage difference
        pct_diff = ((class_1_values - class_0_values) / (class_0_values + 1e-8)) * 100

        # Color by direction
        colors = ['darkred' if d < 0 else 'darkgreen' for d in pct_diff]

        bars = plt.barh(range(len(prototype_names)), pct_diff, color=colors, alpha=0.8)

        plt.yticks(range(len(prototype_names)),
                   [f'P{top_data.iloc[i]["prototype_id"]}' for i in range(len(prototype_names))])
        plt.xlabel('Percentage Difference (%)\n← Lower in Class 1  |  Higher in Class 1 →', fontsize=12)
        plt.title(f'Prototype Importance: Percentage Differences\n(Class 1 vs Class 0)', fontsize=14)
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        plt.grid(axis='x', alpha=0.3)

        # # Add value labels
        # for bar, val in zip(bars, pct_diff):
        #     plt.text(bar.get_width() + (1 if val > 0 else -1), bar.get_y() + bar.get_height() / 2,
        #              f'{val:.1f}%', ha='left' if val > 0 else 'right', va='center', fontsize=9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved prototype plot to {output_path}")

    plt.show()

    return True




def plot_prototype_importance_by_class(prototype_df, top_k=15, output_path=None, figsize=(16, 8)):
    """
    Create two side-by-side bar plots showing top prototypes for each class

    Args:
        prototype_df: DataFrame with class_0_mean and class_1_mean columns
        top_k: Number of top prototypes to show per class
        output_path: Where to save plot
        figsize: Figure size
    """

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Class 0 - sort by class_0_mean
    class_0_sorted = prototype_df.sort_values('class_0_mean', ascending=False).head(top_k)
    prototype_ids_0 = class_0_sorted['prototype_id'].values
    importance_0 = class_0_sorted['class_0_mean'].values

    bars1 = ax1.bar(range(len(prototype_ids_0)), importance_0,
                    color='darkred', alpha=0.8)
    ax1.set_xticks(range(len(prototype_ids_0)))
    ax1.set_xticklabels([f'P{pid}' for pid in prototype_ids_0], rotation=45)
    ax1.set_ylabel('Mean Gating Weight', fontsize=12)
    ax1.set_title(f'Top {top_k} Prototypes in Class 0', fontsize=14, color='darkred')
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars1, importance_0):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(importance_0) * 0.01,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=9, rotation=90)

    # Class 1 - sort by class_1_mean
    class_1_sorted = prototype_df.sort_values('class_1_mean', ascending=False).head(top_k)
    prototype_ids_1 = class_1_sorted['prototype_id'].values
    importance_1 = class_1_sorted['class_1_mean'].values

    bars2 = ax2.bar(range(len(prototype_ids_1)), importance_1,
                    color='darkgreen', alpha=0.8)
    ax2.set_xticks(range(len(prototype_ids_1)))
    ax2.set_xticklabels([f'P{pid}' for pid in prototype_ids_1], rotation=45)
    ax2.set_ylabel('Mean Gating Weight', fontsize=12)
    ax2.set_title(f'Top {top_k} Prototypes in Class 1', fontsize=14, color='darkgreen')
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars2, importance_1):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(importance_1) * 0.01,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=9, rotation=90)

    # Set same y-axis scale for easy comparison
    max_val = max(max(importance_0), max(importance_1))
    ax1.set_ylim(0, max_val * 1.15)
    ax2.set_ylim(0, max_val * 1.15)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved class comparison plot to {output_path}")

    plt.show()

    return {
        'class_0_top': class_0_sorted[['prototype_id', 'class_0_mean']].head(top_k),
        'class_1_top': class_1_sorted[['prototype_id', 'class_1_mean']].head(top_k)
    }


from PIL import Image


class PrototypePatchAnalyzer:
    """
    Analyzer for visualizing representative patches of discriminative prototypes
    """

    def __init__(self, extracted_patches_path):
        """
        Initialize analyzer

        Args:
            extracted_patches_path: Path to extracted_patches.csv with patch locations
        """
        self.patches_df = pd.read_csv(extracted_patches_path)
        self.patch_to_location = {}
        self.patient_assignment_data = {}

        # Create patch name to file location mapping
        for _, row in self.patches_df.iterrows():
            self.patch_to_location[row['Patch_name']] = row['File_location']

        self.prototype_patches = defaultdict(lambda: defaultdict(list))  # [prototype_id][class][patches]

    def add_patient_data(self, patient_id, patch_names, patch_assignments, similarities, class_label, prediction):
        """
        Add patient patch data with similarity information

        Args:
            predictions:
            patient_id: Patient identifier
            patch_names: List of patch names
            patch_assignments: Hard prototype assignments [num_patches]
            similarities: Similarity matrix [num_patches, num_prototypes]
            class_label: Patient class label
        """

        # Convert to numpy if needed
        if hasattr(patch_assignments, 'cpu'):
            assignments = patch_assignments.cpu().numpy().squeeze()
        if hasattr(similarities, 'cpu'):
            sim_matrix = similarities.cpu().numpy()

        if len(sim_matrix.shape) == 3:
            sim_matrix = sim_matrix.squeeze(0)

        # proto_counts_total = {}
        # for proto_id in assignments:
        #     proto_counts_total[int(proto_id)] = proto_counts_total.get(int(proto_id), 0) + 1
        #
        # print("Prototype assignments (before file filtering):")
        # for proto_id in sorted(proto_counts_total.keys()):
        #     print(f"  Prototype {proto_id}: {proto_counts_total[proto_id]} patches")

        # skipped_patches = 0
        # Group patches by prototype with similarity scores
        for i, (patch_name_tuple, proto_id) in enumerate(zip(patch_names, assignments)):
            patch_name = patch_name_tuple[0] if isinstance(patch_name_tuple, (list, tuple)) else patch_name_tuple

            if patch_name in self.patch_to_location:
                file_path = self.patch_to_location[patch_name]
                if os.path.exists(file_path):
                    # Get similarity score for this patch to its assigned prototype
                    similarity_score = sim_matrix[i, int(proto_id)]

                    self.prototype_patches[int(proto_id)][int(class_label)].append({
                        'patient_id': patient_id,
                        'patch_name': patch_name,
                        'file_path': file_path,
                        'similarity_score': similarity_score,
                        'all_similarities': sim_matrix[i, :]  # For finding cross-prototype similarities
                    })
        #         else:
        #             print(f"Warning: File not found for patch {patch_name} at {file_path}")
        #             skipped_patches += 1
        #     else:
        #         print(f"Warning: No file location mapping found for patch {patch_name}")
        #         skipped_patches += 1
        #
        # # Report summary
        # if skipped_patches > 0:
        #     print(f"Total patches skipped for patient {patient_id}: {skipped_patches}/{len(patch_names)}")
        #
        # # Debug: See what prototypes are actually assigned
        # unique_prototypes = np.unique(assignments)
        # print(f"Patient {patient_id}: Assigned prototypes: {sorted(unique_prototypes)}")
        #
        # # Count assignments per prototype
        # proto_counts = {}
        # for proto_id in assignments:
        #     proto_counts[int(proto_id)] = proto_counts.get(int(proto_id), 0) + 1
        #
        # print(f"Patient {patient_id}: Prototype assignment counts: {dict(sorted(proto_counts.items()))}")

        # Count assignments per prototype
        proto_counts = {}
        for proto_id in assignments:
            proto_counts[int(proto_id)] = proto_counts.get(int(proto_id), 0) + 1

        print(f"Patient {patient_id}: Prototype assignment counts: {dict(sorted(proto_counts.items()))}")

        # Store assignment frequencies for threshold analysis
        self.patient_assignment_data[patient_id] = {
            'true_class': int(class_label),
            'predicted_class': prediction,
            'assignment_counts': proto_counts,
            'total_patches': len(assignments),
            'assignment_frequencies': {proto_id: count / len(assignments)
                                       for proto_id, count in proto_counts.items()}
        }

    def plot_prototype_patches(self, prototype_ids, patches_per_prototype=6, selection_method='top_similarity',
                               patches_per_class=3, figsize=(20, 12), output_path=None):
        """
        Create a grid showing representative patches for each discriminative prototype

        Args:
            prototype_ids: List of prototype IDs to visualize
            patches_per_prototype: Total patches to show per prototype
            patches_per_class: Patches per class (should be ≤ patches_per_prototype/2)
            figsize: Figure size
            output_path: Where to save plot
        """

        n_prototypes = len(prototype_ids)
        n_cols = patches_per_prototype
        n_rows = n_prototypes

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for row, proto_id in enumerate(prototype_ids):

            class_0_patches = self.prototype_patches[proto_id].get(0, [])
            class_1_patches = self.prototype_patches[proto_id].get(1, [])

            print(
                f"Prototype {proto_id}: {len(class_0_patches)} class 0 patches, {len(class_1_patches)} class 1 patches")

            # Intelligent patch selection
            if selection_method == 'top_similarity':
                # Select patches with highest similarity to this prototype
                class_0_sample = sorted(class_0_patches, key=lambda x: x['similarity_score'], reverse=True)[
                                 :patches_per_class]
                class_1_sample = sorted(class_1_patches, key=lambda x: x['similarity_score'], reverse=True)[
                                 :patches_per_class]

            elif selection_method == 'diverse_similarity':
                # Select patches with diverse similarity scores (high, medium, low)
                class_0_sample = self._select_diverse_patches(class_0_patches, patches_per_class)
                class_1_sample = self._select_diverse_patches(class_1_patches, patches_per_class)

            else:  # random
                class_0_sample = np.random.choice(class_0_patches, min(patches_per_class, len(class_0_patches)),
                                                  replace=False) if class_0_patches else []
                class_1_sample = np.random.choice(class_1_patches, min(patches_per_class, len(class_1_patches)),
                                                  replace=False) if class_1_patches else []

            # Combine and pad if needed
            all_patches = list(class_0_sample) + list(class_1_sample)

            for col in range(n_cols):
                ax = axes[row, col]

                if col < len(all_patches):
                    patch_info = all_patches[col]

                    try:
                        # Load and display patch
                        img = Image.open(patch_info['file_path'])
                        ax.imshow(img)

                        # Determine class for coloring
                        if col < len(class_0_sample):
                            class_label = "Class 0"
                            border_color = 'steelblue'
                        else:
                            class_label = "Class 1"
                            border_color = 'coral'

                        # Get similarity score
                        similarity = patch_info.get('similarity_score', 0)

                        # Set title with all info
                        ax.set_title(f'{class_label}\nPt: {patch_info["patient_id"]}\nSim: {similarity:.3f}',
                                     fontsize=9, color=border_color, fontweight='bold')

                        # Add colored border
                        for spine in ax.spines.values():
                            spine.set_edgecolor(border_color)
                            spine.set_linewidth(3)

                    except Exception as e:
                        ax.text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center')
                        print(f"Error loading {patch_info['file_path']}: {e}")
                else:
                    # Empty subplot
                    ax.text(0.5, 0.5, 'No Patch\nAvailable', ha='center', va='center')

                ax.set_xticks([])
                ax.set_yticks([])

            # Add prototype ID label on the left
            axes[row, 0].set_ylabel(f'Prototype {proto_id}', fontsize=14, fontweight='bold')

            # Add similarity scores to titles
            for col in range(n_cols):
                if col < len(all_patches):
                    patch_info = all_patches[col]
                    similarity = patch_info.get('similarity_score', 0)

                    ax.set_title(f'{class_label}\nPt: {patch_info["patient_id"]}\nSim: {similarity:.3f}',
                                 fontsize=9, color=border_color, fontweight='bold')

        plt.suptitle('Representative Patches for Highest Attended Prototypes', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved prototype patches visualization to {output_path}")

        plt.show()

        return True

    def _select_diverse_patches(self, patches, n_patches):
        """Select patches with diverse similarity scores"""
        if len(patches) <= n_patches:
            return patches

        # Sort by similarity and select from different percentiles
        sorted_patches = sorted(patches, key=lambda x: x['similarity_score'], reverse=True)

        indices = []
        for i in range(n_patches):
            # Select from different parts of the distribution
            percentile = (i / (n_patches - 1)) if n_patches > 1 else 0
            idx = int(percentile * (len(sorted_patches) - 1))
            indices.append(idx)

        return [sorted_patches[i] for i in indices]

    def get_discriminative_prototypes(self, prototype_results_df, top_k=5):
        """
        Get most discriminative prototypes from analysis results

        Args:
            prototype_results_df: DataFrame from PrototypeGateAnalyzer
            top_k: Number of top discriminative prototypes
        """

        # Sort by absolute percentage difference or rank difference
        if 'percentage_difference' in prototype_results_df.columns:
            sort_col = 'percentage_difference'
        else:
            sort_col = 'rank_difference'

        top_discriminative = prototype_results_df.sort_values(sort_col, key=abs, ascending=False).head(top_k)

        return top_discriminative['prototype_id'].tolist()


    def create_prototype_summary_report(self, prototype_ids, output_dir):
        """
        Create detailed summary report for each prototype
        """

        summary_data = []

        for proto_id in prototype_ids:
            class_0_count = len(self.prototype_patches[proto_id].get(0, []))
            class_1_count = len(self.prototype_patches[proto_id].get(1, []))
            total_count = class_0_count + class_1_count

            if total_count > 0:
                class_0_pct = (class_0_count / total_count) * 100
                class_1_pct = (class_1_count / total_count) * 100
            else:
                class_0_pct = class_1_pct = 0

            # Get unique patients
            all_patches = (self.prototype_patches[proto_id].get(0, []) +
                           self.prototype_patches[proto_id].get(1, []))
            unique_patients = len(set(p['patient_id'] for p in all_patches))

            summary_data.append({
                'prototype_id': proto_id,
                'total_patches': total_count,
                'class_0_patches': class_0_count,
                'class_1_patches': class_1_count,
                'class_0_percentage': class_0_pct,
                'class_1_percentage': class_1_pct,
                'unique_patients': unique_patients
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(output_dir, 'prototype_patch_summary.csv'), index=False)

        return summary_df

    def analyze_prototype_thresholds(self, prototype_ids=None, output_path=None):
        """
        Analyze assignment frequency patterns that reveal diagnostic thresholds
        """
        if prototype_ids is None:
            # Get most commonly assigned prototypes
            all_protos = set()
            for data in self.patient_assignment_data.values():
                all_protos.update(data['assignment_counts'].keys())
            prototype_ids = sorted(list(all_protos))[:8]  # Top 8 prototypes

        n_prototypes = len(prototype_ids)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        for i, proto_id in enumerate(prototype_ids):
            ax = axes[i]

            # Collect assignment frequencies by class and correctness
            class_0_correct = []
            class_0_incorrect = []
            class_1_correct = []
            class_1_incorrect = []

            for patient_id, data in self.patient_assignment_data.items():
                if data['predicted_class'] is None:
                    continue

                freq = data['assignment_frequencies'].get(proto_id, 0)

                if data['true_class'] == 0:
                    if data['predicted_class'] == 0:
                        class_0_correct.append(freq)
                    else:
                        class_0_incorrect.append(freq)
                else:  # class 1
                    if data['predicted_class'] == 1:
                        class_1_correct.append(freq)
                    else:
                        class_1_incorrect.append(freq)

            # Create box plots
            box_data = []
            labels = []
            colors = []

            if class_0_correct:
                box_data.append(class_0_correct)
                # labels.append(f'Class 0\nCorrect\n(n={len(class_0_correct)})')
                labels.append(f'Class 0\nCorrect)')
                colors.append('lightblue')

            if class_0_incorrect:
                box_data.append(class_0_incorrect)
                # labels.append(f'Class 0\nIncorrect\n(n={len(class_0_incorrect)})')
                labels.append(f'Class 0\nIncorrect')
                colors.append('lightcoral')

            if class_1_correct:
                box_data.append(class_1_correct)
                # labels.append(f'Class 1\nCorrect\n(n={len(class_1_correct)})')
                labels.append(f'Class 1\nCorrect')
                colors.append('lightgreen')

            if class_1_incorrect:
                box_data.append(class_1_incorrect)
                # labels.append(f'Class 1\nIncorrect\n(n={len(class_1_incorrect)})')
                labels.append(f'Class 1\nIncorrect')
                colors.append('orange')

            if box_data:
                box_plot = ax.boxplot(box_data, labels=labels, patch_artist=True)
                for patch, color in zip(box_plot['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

            ax.set_title(f'Prototype {proto_id}\nAssignment Frequency Thresholds', fontweight='bold')
            ax.set_ylabel('Assignment Frequency')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)

        plt.suptitle('Prototype Assignment Frequency Analysis: Diagnostic Thresholds', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            print(f"Saved threshold analysis to {output_path}")

        plt.show()

    def plot_assignment_scatter(self, prototype_id=25, output_path=None):
        """
        Create scatter plot showing assignment frequency vs prediction patterns
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Collect data
        class_0_data = {'freq': [], 'correct': [], 'patient_ids': []}
        class_1_data = {'freq': [], 'correct': [], 'patient_ids': []}

        for patient_id, data in self.patient_assignment_data.items():
            if data['predicted_class'] is None:
                continue

            freq = data['assignment_frequencies'].get(prototype_id, 0)
            is_correct = data['true_class'] == data['predicted_class']

            if data['true_class'] == 0:
                class_0_data['freq'].append(freq)
                class_0_data['correct'].append(is_correct)
                class_0_data['patient_ids'].append(patient_id)
            else:
                class_1_data['freq'].append(freq)
                class_1_data['correct'].append(is_correct)
                class_1_data['patient_ids'].append(patient_id)

        # Plot Class 0
        for i, (freq, correct, pid) in enumerate(
                zip(class_0_data['freq'], class_0_data['correct'], class_0_data['patient_ids'])):
            color = 'blue' if correct else 'red'
            marker = 'o' if correct else 'x'
            ax1.scatter(freq, i, c=color, marker=marker, s=60, alpha=0.7)

        ax1.set_xlabel(f'Prototype {prototype_id} Assignment Frequency')
        ax1.set_ylabel('Patient Index')
        ax1.set_title(f'Class 0 Patients\nBlue=Correct, Red=Incorrect')
        ax1.grid(True, alpha=0.3)

        # Plot Class 1
        for i, (freq, correct, pid) in enumerate(
                zip(class_1_data['freq'], class_1_data['correct'], class_1_data['patient_ids'])):
            color = 'green' if correct else 'orange'
            marker = 'o' if correct else 'x'
            ax2.scatter(freq, i, c=color, marker=marker, s=60, alpha=0.7)

        ax2.set_xlabel(f'Prototype {prototype_id} Assignment Frequency')
        ax2.set_ylabel('Patient Index')
        ax2.set_title(f'Class 1 Patients\nGreen=Correct, Orange=Incorrect')
        ax2.grid(True, alpha=0.3)

        plt.suptitle(f'Prototype {prototype_id}: Assignment Frequency vs Classification Accuracy', fontsize=14,
                     fontweight='bold')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved scatter plot to {output_path}")

        plt.show()

    def plot_diagnostic_thresholds_summary(self, key_prototypes=[25, 0, 21, 4], output_path=None):
        """
        Create summary visualization showing diagnostic threshold patterns for key prototypes
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 12))
        axes = axes.flatten()

        for i, proto_id in enumerate(key_prototypes):
            ax = axes[i]

            # Calculate mean assignment frequencies by class and correctness
            means = {'C0_correct': [], 'C0_incorrect': [], 'C1_correct': [], 'C1_incorrect': []}

            for patient_id, data in self.patient_assignment_data.items():
                if data['predicted_class'] is None:
                    continue

                freq = data['assignment_frequencies'].get(proto_id, 0)

                if data['true_class'] == 0:
                    if data['predicted_class'] == 0:
                        means['C0_correct'].append(freq)
                    else:
                        means['C0_incorrect'].append(freq)
                else:
                    if data['predicted_class'] == 1:
                        means['C1_correct'].append(freq)
                    else:
                        means['C1_incorrect'].append(freq)

            # Calculate means and stds
            categories = ['Class 0\nCorrect', 'Class 0\nIncorrect', 'Class 1\nCorrect', 'Class 1\nIncorrect']
            values = [np.mean(means['C0_correct']) if means['C0_correct'] else 0,
                      np.mean(means['C0_incorrect']) if means['C0_incorrect'] else 0,
                      np.mean(means['C1_correct']) if means['C1_correct'] else 0,
                      np.mean(means['C1_incorrect']) if means['C1_incorrect'] else 0]

            errors = [np.std(means['C0_correct']) if means['C0_correct'] else 0,
                      np.std(means['C0_incorrect']) if means['C0_incorrect'] else 0,
                      np.std(means['C1_correct']) if means['C1_correct'] else 0,
                      np.std(means['C1_incorrect']) if means['C1_incorrect'] else 0]

            colors = ['lightblue', 'lightcoral', 'lightgreen', 'orange']

            bars = ax.bar(categories, values, yerr=errors, color=colors, alpha=0.7, capsize=6)
            ax.set_title(f'Prototype {proto_id}\nMean Assignment Frequency', fontweight='bold')
            ax.set_ylabel('Assignment Frequency')
            ax.tick_params(axis='x', rotation=45)

            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.suptitle('Diagnostic Threshold Summary: Assignment Frequencies by Classification Outcome',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            print(f"Saved threshold summary to {output_path}")

        plt.show()



