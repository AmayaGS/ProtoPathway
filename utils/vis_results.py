# utils/gene_importance_analyzer.py

import os
import pandas as pd
import numpy as np
from collections import defaultdict


class GeneImportanceAnalyzer:
    def __init__(self, gene_idx, pathway_idx):
        self.idx_to_gene = {v.item(): k for k, v in gene_idx.items()}
        self.idx_to_pathway = {v.item(): k for k, v in pathway_idx.items()}
        self.patient_data = {}
        self.class_data = defaultdict(list)

    def add_patient(self, patient_id, gene_pathway_attn, label=None):
        gene_importance = gene_pathway_attn.sum(dim=1).cpu().numpy()

        self.patient_data[patient_id] = {
            'gene_importance': gene_importance,
            'label': label.item()
        }

        if label is not None:
            self.class_data[label.item()].append((patient_id, gene_importance))

    def top_genes(self, patient_id, k=500):
        scores = self.patient_data[patient_id]['gene_importance']
        top_idx = np.argsort(scores)[-k:][::-1]

        return pd.DataFrame({
            'gene': [self.idx_to_gene[i] for i in top_idx],
            'score': scores[top_idx]
        })

    def save_patient_results(self, output_dir):
        """Save individual patient results to CSV files"""
        os.makedirs(output_dir, exist_ok=True)

        for pid, data in self.patient_data.items():
            df = pd.DataFrame({
                'gene': [self.idx_to_gene[i] for i in range(len(data['gene_importance']))],
                'importance': data['gene_importance']
            })
            filename = f"patient_{pid}_genes.csv"
            df.to_csv(f"{output_dir}/{filename}", index=False)


    def class_aggregation(self, k=500):
        """Get top genes by class with mean importance scores"""
        results = {}

        for label, patient_list in self.class_data.items():
            # Stack all importance scores for this class
            scores = np.stack([scores for _, scores in patient_list])
            mean_scores = scores.mean(axis=0)

            top_idx = np.argsort(mean_scores)[-k:][::-1]

            results[label] = pd.DataFrame({
                'gene': [self.idx_to_gene[i] for i in top_idx],
                'mean_importance': mean_scores[top_idx],
                'n_patients': len(patient_list)
            })


        return results


    def class_differences(self, k=500):
        """Find genes with biggest differences between classes"""
        from scipy.stats import ttest_ind

        if len(self.class_data) != 2:
            raise ValueError("Need exactly 2 classes for comparison")

        labels = sorted(list(self.class_data.keys()))
        class0_scores = np.stack([scores for _, scores in self.class_data[labels[0]]])
        class1_scores = np.stack([scores for _, scores in self.class_data[labels[1]]])

        # Calculate means and differences
        mean0 = class0_scores.mean(axis=0)
        mean1 = class1_scores.mean(axis=0)
        diff = mean1 - mean0
        fold_change = mean1 / (mean0 + 1e-8)  # avoid division by zero

        # T-test for each gene
        p_values = []
        for i in range(len(mean0)):
            _, p = ttest_ind(class0_scores[:, i], class1_scores[:, i])
            p_values.append(p)

        # Create results dataframe
        results = pd.DataFrame({
            'gene': [self.idx_to_gene[i] for i in range(len(mean0))],
            f'mean_class_{labels[0]}': mean0,
            f'mean_class_{labels[1]}': mean1,
            'difference': diff,
            'fold_change': fold_change,
            'p_value': p_values
        })

        # Add enrichment info
        results['enriched_in_class'] = np.where(diff > 0, labels[1], labels[0])
        results['abs_diff'] = np.abs(diff)

        # Sort by absolute difference
        results = results.sort_values('abs_diff', ascending=False)

        # Add significance flag
        results['significant'] = results['p_value'] < 0.05

        return results.head(k)

    def class_specific_drivers(self, k=100):
        """Get top driver genes for each class separately"""
        diff_results = self.class_differences(k=1000)  # Get many genes first
        significant = diff_results[diff_results['significant'] == True]

        labels = list(self.class_data.keys())

        # Split by enriched class
        class0_drivers = significant[significant['enriched_in_class'] == labels[0]].head(k)
        class1_drivers = significant[significant['enriched_in_class'] == labels[1]].head(k)

        return {
            f'class_{labels[0]}_drivers': class0_drivers,
            f'class_{labels[1]}_drivers': class1_drivers
        }


class PathwayImportanceAnalyzer:

    def __init__(self, gene_idx, pathway_idx):
        self.idx_to_gene = {v.item(): k for k, v in gene_idx.items()}
        self.idx_to_pathway = {v.item(): k for k, v in pathway_idx.items()}
        self.patient_data = {}
        self.class_data = defaultdict(list)

    def add_patient(self, patient_id, gene_pathway_attn, label=None):
        pathway_importance = gene_pathway_attn.sum(dim=0).cpu().numpy()

        self.patient_data[patient_id] = {
            'pathway_importance': pathway_importance,
            'label': label.item()
        }

        if label is not None:
            self.class_data[label.item()].append((patient_id, pathway_importance))

    def top_pathways(self, patient_id, k=10):
        scores = self.patient_data[patient_id]['pathway_importance']
        top_idx = np.argsort(scores)[-k:][::-1]

        return pd.DataFrame({
            'pathway': [self.idx_to_pathway[i] for i in top_idx],
            'score': scores[top_idx]
        })

    def save_patient_results(self, output_dir):
        import os
        os.makedirs(output_dir, exist_ok=True)

        for pid, data in self.patient_data.items():
            df = pd.DataFrame({
                'pathway': [self.idx_to_pathway[i] for i in range(len(data['pathway_importance']))],
                'importance': data['pathway_importance']
            })
            filename = f"patient_{pid}_pathways.csv"
            df.to_csv(f"{output_dir}/{filename}", index=False)

    def class_aggregation(self, k=20):
        results = {}

        for label, patient_list in self.class_data.items():
            scores = np.stack([scores for _, scores in patient_list])
            mean_scores = scores.mean(axis=0)

            top_idx = np.argsort(mean_scores)[-k:][::-1]

            results[label] = pd.DataFrame({
                'pathway': [self.idx_to_pathway[i] for i in top_idx],
                'mean_importance': mean_scores[top_idx],
                'n_patients': len(patient_list)
            })

        return results

    def class_differences(self, k=50):
        from scipy.stats import ttest_ind

        if len(self.class_data) != 2:
            raise ValueError("Need exactly 2 classes for comparison")

        labels = sorted(list(self.class_data.keys()))
        class0_scores = np.stack([scores for _, scores in self.class_data[labels[0]]])
        class1_scores = np.stack([scores for _, scores in self.class_data[labels[1]]])

        mean0 = class0_scores.mean(axis=0)
        mean1 = class1_scores.mean(axis=0)
        diff = mean1 - mean0
        fold_change = mean1 / (mean0 + 1e-8)

        p_values = []
        for i in range(len(mean0)):
            _, p = ttest_ind(class0_scores[:, i], class1_scores[:, i])
            p_values.append(p)

        results = pd.DataFrame({
            'pathway': [self.idx_to_pathway[i] for i in range(len(mean0))],
            f'mean_class_{labels[0]}': mean0,
            f'mean_class_{labels[1]}': mean1,
            'difference': diff,
            'fold_change': fold_change,
            'p_value': p_values
        })

        results['enriched_in_class'] = np.where(diff > 0, labels[1], labels[0])
        results['abs_diff'] = np.abs(diff)
        results = results.sort_values('abs_diff', ascending=False)
        results['significant'] = results['p_value'] < 0.05

        return results.head(k)

    def class_specific_drivers(self, k=25):
        diff_results = self.class_differences(k=1000)
        significant = diff_results[diff_results['significant'] == True]

        labels = list(self.class_data.keys())

        class0_drivers = significant[significant['enriched_in_class'] == labels[0]].head(k)
        class1_drivers = significant[significant['enriched_in_class'] == labels[1]].head(k)

        return {
            f'class_{labels[0]}_drivers': class0_drivers,
            f'class_{labels[1]}_drivers': class1_drivers
        }


class CrossModalPathwayAnalyzer:
    def __init__(self, gene_idx, pathway_idx):
        self.idx_to_gene = {v.item(): k for k, v in gene_idx.items()}
        self.idx_to_pathway = {v.item(): k for k, v in pathway_idx.items()}
        self.patient_data = {}
        self.class_data = defaultdict(list)

    def add_patient(self, patient_id, cross_modal_attn, label=None):
        # Method 1: Sum across prototypes to get pathway importance
        pathway_importance = cross_modal_attn.sum(dim=1).squeeze().cpu().numpy()  # [N_pathways]
        pathway_importance_norm = pathway_importance / (pathway_importance.sum() + 1e-8)

        self.patient_data[patient_id] = {
            'pathway_importance': pathway_importance,
            'pathway_importance_norm': pathway_importance_norm,
            'label': label.item()
        }

        if label is not None:
            self.class_data[label.item()].append((patient_id, pathway_importance_norm))

    def top_pathways(self, patient_id, k=10):
        scores = self.patient_data[patient_id]['pathway_importance']
        top_idx = np.argsort(scores)[-k:][::-1]

        return pd.DataFrame({
            'pathway': [self.idx_to_pathway[i] for i in top_idx],
            'cross_modal_score': scores[top_idx]
        })

    def save_patient_results(self, output_dir):
        import os
        os.makedirs(output_dir, exist_ok=True)

        for pid, data in self.patient_data.items():
            df = pd.DataFrame({
                'pathway': [self.idx_to_pathway[i] for i in range(len(data['pathway_importance']))],
                'cross_modal_importance': data['pathway_importance']
            })
            filename = f"patient_{pid}_label_{data['label']}_crossmodal_pathways.csv"
            df.to_csv(f"{output_dir}/{filename}", index=False)

    def class_aggregation(self, k=500):
        results = {}

        for label, patient_list in self.class_data.items():
            scores = np.stack([scores for _, scores in patient_list])
            mean_scores = scores.mean(axis=0)

            top_idx = np.argsort(mean_scores)[-k:][::-1]

            results[label] = pd.DataFrame({
                'pathway': [self.idx_to_pathway[i] for i in top_idx],
                'mean_cross_modal_importance': mean_scores[top_idx],
                'n_patients': len(patient_list)
            })

        return results

    def class_differences(self, k=500):
        from scipy.stats import ttest_ind

        if len(self.class_data) != 2:
            raise ValueError("Need exactly 2 classes for comparison")

        labels = sorted(list(self.class_data.keys()))
        class0_scores = np.stack([self.patient_data[pid]['pathway_importance_norm']
                                for pid, _ in self.class_data[labels[0]]])
        class1_scores = np.stack([self.patient_data[pid]['pathway_importance_norm']
                                for pid, _ in self.class_data[labels[1]]])

        mean0 = class0_scores.mean(axis=0)
        mean1 = class1_scores.mean(axis=0)
        diff = mean1 - mean0
        fold_change = mean1 / (mean0 + 1e-8)

        # Calculate Cohen's d (effect size)
        pooled_std = np.sqrt(((class0_scores.var(axis=0) * (len(class0_scores) - 1)) +
                              (class1_scores.var(axis=0) * (len(class1_scores) - 1))) /
                             (len(class0_scores) + len(class1_scores) - 2))
        cohens_d = (mean1 - mean0) / (pooled_std + 1e-8)

        p_values = []
        for i in range(len(mean0)):
            _, p = ttest_ind(class0_scores[:, i], class1_scores[:, i])
            p_values.append(p)

        results = pd.DataFrame({
            'pathway': [self.idx_to_pathway[i] for i in range(len(mean0))],
            f'mean_class_{labels[0]}': mean0,
            f'mean_class_{labels[1]}': mean1,
            'difference': diff,
            'cohens_d': cohens_d,
            'abs_cohens_d': np.abs(cohens_d),
            'fold_change': fold_change,
            'p_value': p_values,
            'dominant_class': np.where(mean1 > mean0, labels[1], labels[0]),
            'log2_fold_change': np.log2((mean1 + 1e-8) / (mean0 + 1e-8))
        })

        results['enriched_in_class'] = np.where(diff > 0, labels[1], labels[0])
        results['abs_diff'] = np.abs(diff)
        # results = results.sort_values('abs_diff', ascending=False)
        results['significant'] = results['p_value'] < 0.05

        results = results.sort_values('abs_cohens_d', ascending=False)

        return results.head(k)

    def class_specific_drivers(self, k=100):
        diff_results = self.class_differences(k=100)
        significant = diff_results[diff_results['significant'] == True]

        labels = sorted(list(self.class_data.keys()))

        class0_drivers = significant[significant['dominant_class'] == labels[0]].head(k)
        class1_drivers = significant[significant['dominant_class'] == labels[1]].head(k)

        return {
            f'class_{labels[0]}_drivers': class0_drivers,
            f'class_{labels[1]}_drivers': class1_drivers
        }

    def rank_based_analysis(self, k=100):
        """
        Compare pathway rankings within patients between classes
        """
        from scipy.stats import mannwhitneyu

        labels = sorted(list(self.class_data.keys()))

        # Get ranks for each patient
        class0_ranks = []
        class1_ranks = []

        for label in labels:
            for pid, _ in self.class_data[label]:
                scores = self.patient_data[pid]['pathway_importance_norm']
                ranks = len(scores) - np.argsort(np.argsort(scores))  # Higher score = higher rank

                if label == labels[0]:
                    class0_ranks.append(ranks)
                else:
                    class1_ranks.append(ranks)

        class0_ranks = np.array(class0_ranks)
        class1_ranks = np.array(class1_ranks)

        # Compare ranks for each pathway
        results = []
        for i in range(class0_ranks.shape[1]):
            statistic, p_value = mannwhitneyu(
                class1_ranks[:, i], class0_ranks[:, i],
                alternative='two-sided'
            )

            mean_rank_0 = class0_ranks[:, i].mean()
            mean_rank_1 = class1_ranks[:, i].mean()

            results.append({
                'pathway': self.idx_to_pathway[i],
                f'mean_rank_class_{labels[0]}': mean_rank_0,
                f'mean_rank_class_{labels[1]}': mean_rank_1,
                'rank_difference': mean_rank_1 - mean_rank_0,
                'p_value': p_value,
                'higher_in_class': labels[1] if mean_rank_1 > mean_rank_0 else labels[0]
            })

        rank_df = pd.DataFrame(results)
        rank_df['significant'] = rank_df['p_value'] < 0.05
        rank_df = rank_df.sort_values('rank_difference', key=abs, ascending=False)

        return rank_df.head(k)

    def consensus_pathway_analysis(self, output_dir, k_per_method=100):
        """
        Create consensus rankings based on agreement across methods
        """

        # Get results from all three methods
        diff_results = self.class_differences(k=k_per_method)
        drivers = self.class_specific_drivers(k=k_per_method)
        rank_results = self.rank_based_analysis(k=k_per_method)

        # Extract significant pathways from each method by class
        labels = sorted(list(self.class_data.keys()))

        consensus_results = {}

        for class_idx, class_label in enumerate(labels):
            # Method 1: Statistical differences (filter by significance)
            method1_pathways = set(diff_results[
                                       (diff_results['significant'] == True) &
                                       (diff_results['dominant_class'] == class_label)
                                       ]['pathway'])

            # Method 2: Enhanced drivers
            driver_key = f'class_{class_label}_drivers'

            if driver_key in drivers:
                method2_pathways = set(drivers[driver_key]['pathway'])
            else:
                method2_pathways = set()

            # Method 3: Rank-based (filter by significance)
            method3_pathways = set(rank_results[
                                       (rank_results['significant'] == True) &
                                       (rank_results['higher_in_class'] == class_label)
                                       ]['pathway'])

            # Count method agreement
            all_pathways = method1_pathways | method2_pathways | method3_pathways
            consensus_data = []

            for pathway in all_pathways:
                method_count = sum([
                    pathway in method1_pathways,
                    pathway in method2_pathways,
                    pathway in method3_pathways
                ])

                confidence = 'high' if method_count == 3 else 'medium' if method_count == 2 else 'exploratory'

                consensus_data.append({
                    'pathway': pathway,
                    'method_count': method_count,
                    'confidence': confidence,
                    'in_statistical': pathway in method1_pathways,
                    'in_enhanced': pathway in method2_pathways,
                    'in_rank': pathway in method3_pathways
                })

            # Save to CSV
            consensus_df = pd.DataFrame(consensus_data)
            consensus_df = consensus_df.sort_values(['method_count', 'pathway'], ascending=[False, True])

            output_file = os.path.join(output_dir, f'class_{class_label}_consensus_pathways.csv')
            consensus_df.to_csv(output_file, index=False)

        return f"Consensus analysis saved to {output_dir}"