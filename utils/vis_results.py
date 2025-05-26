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

    def class_specific_drivers(self, k=100):
        diff_results = self.class_differences(k=100)
        significant = diff_results[diff_results['significant'] == True]

        labels = list(self.class_data.keys())

        class0_drivers = significant[significant['enriched_in_class'] == labels[0]].head(k)
        class1_drivers = significant[significant['enriched_in_class'] == labels[1]].head(k)

        return {
            f'class_{labels[0]}_drivers': class0_drivers,
            f'class_{labels[1]}_drivers': class1_drivers
        }