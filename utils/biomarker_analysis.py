import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from collections import defaultdict

# Fix for non-interactive environments
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend


class BiomarkerAnalysis:
    """
    Comprehensive biomarker analysis framework for the PathwayEmbeddingModel.
    Analyzes both patient-level and population-level patterns linked to target labels.
    """

    def __init__(self, model, gene_names=None, pathway_names=None, label_dict=None):
        """
        Initialize the biomarker analysis framework.

        Args:
            model: The PathwayEmbeddingModel instance
            gene_names: List of gene names (will use model.gene_names if None)
            pathway_names: List of pathway names (will use model.pathway_names if None)
            label_dict: Dictionary mapping numeric labels to class names
        """
        self.model = model
        self.gene_names = gene_names or model.gene_names
        self.pathway_names = pathway_names or model.pathway_names
        self.label_dict = label_dict or {0: "Class_0", 1: "Class_1"}

        # Storage for patient-level importance scores
        self.patient_data = {
            "patient_id": [],
            "true_label": [],
            "pred_label": [],
            "pathway_importance": [],
            "gene_pathway_attention": [],
            "gene_importance": []
        }

        # Storage for population-level importance scores
        self.population_data = defaultdict(dict)

        # Results storage
        self.biomarker_results = None

        # Parameters for biomarker analysis
        self.min_fold_change = 1.5
        self.max_pvalue = 0.05

    def collect_patient_importance(self, test_loader, device):
        """
        Collect importance scores for all patients in the test dataset.

        Args:
            test_loader: PyTorch Geometric DataLoader with test samples
            device: The computation device

        Returns:
            Dictionary with patient-level importance data
        """
        self.model.eval()

        with torch.no_grad():
            for batch in test_loader:
                batch.to(device)
                patient_id = batch.patient_id
                target = batch.y

                # Forward pass with importance calculation
                outputs = self.model(batch, return_importance=True)

                # Get predictions
                probs = torch.nn.functional.softmax(outputs, dim=1)
                pred = outputs.argmax(dim=1)

                # Store importance scores for this patient
                self.patient_data["patient_id"].append(patient_id)
                self.patient_data["true_label"].append(target.item())
                self.patient_data["pred_label"].append(pred.item())
                self.patient_data["pathway_importance"].append(self.model.pathway_importance.cpu().clone())
                self.patient_data["gene_pathway_attention"].append(self.model.gene_pathway_attention.cpu().clone())
                self.patient_data["gene_importance"].append(self.model.get_gene_importance().cpu().clone())

        return self.patient_data

    def _get_patient_report(self, patient_idx, top_k=10):
        """
        Generate importance report for a specific patient.

        Args:
            patient_idx: Index in the patient_data structures
            top_k: Number of top elements to return

        Returns:
            Dictionary with importance reports
        """
        # Temporarily set the model's importance values to this patient's values
        self.model.pathway_importance = self.patient_data["pathway_importance"][patient_idx]
        self.model.gene_pathway_attention = self.patient_data["gene_pathway_attention"][patient_idx]

        # Generate report
        return self.model.generate_importance_report(
            top_pathways=top_k,
            top_genes=top_k,
            top_genes_per_pathway=top_k
        )

    def analyze_by_group(self):
        """
        Group patients by their true label and analyze importance patterns within each group.

        Returns:
            Dictionary with group-level importance data
        """
        # Create groups by true label
        label_groups = defaultdict(list)
        for i, label in enumerate(self.patient_data["true_label"]):
            label_groups[label].append(i)

        # Analyze each group
        group_results = {}

        for label, indices in label_groups.items():
            label_name = self.label_dict.get(label, f"Class_{label}")

            # Stack importance tensors for this group
            pathway_importance = torch.stack([self.patient_data["pathway_importance"][i] for i in indices])
            gene_pathway_attention = torch.stack([self.patient_data["gene_pathway_attention"][i] for i in indices])
            gene_importance = torch.stack([self.patient_data["gene_importance"][i] for i in indices])

            # Calculate mean and std importance
            mean_pathway_imp = pathway_importance.mean(dim=0)
            std_pathway_imp = pathway_importance.std(dim=0)
            mean_gene_imp = gene_importance.mean(dim=0)
            std_gene_imp = gene_importance.std(dim=0)

            # Store in population data
            self.population_data[label_name] = {
                "pathway_importance": {
                    "mean": mean_pathway_imp,
                    "std": std_pathway_imp
                },
                "gene_importance": {
                    "mean": mean_gene_imp,
                    "std": std_gene_imp
                },
                "patient_indices": indices,
                "count": len(indices)
            }

            # Temporarily set model importance to this group's average
            self.model.pathway_importance = mean_pathway_imp
            self.model.gene_pathway_attention = gene_pathway_attention.mean(dim=0)

            # Generate report for this group
            group_results[label_name] = self.model.generate_importance_report()

        return group_results

    def identify_differential_biomarkers(self, min_fold_change=1.5, max_pvalue=0.05):
        """
        Identify genes and pathways with significantly different importance between groups.

        Args:
            min_fold_change: Minimum fold-change in importance to consider
            max_pvalue: Maximum p-value to consider significant

        Returns:
            DataFrame with differential biomarkers
        """
        # Store parameters for later use in visualization
        self.min_fold_change = min_fold_change
        self.max_pvalue = max_pvalue

        # Only applicable for binary classification currently
        if len(self.population_data) != 2:
            raise ValueError("Differential biomarker analysis requires exactly 2 classes")

        # Get class names
        classes = list(self.population_data.keys())

        # Prepare results storage
        gene_results = []
        pathway_results = []

        # Analysis for pathway biomarkers
        class0_idx = self.population_data[classes[0]]["patient_indices"]
        class1_idx = self.population_data[classes[1]]["patient_indices"]

        # For each pathway, calculate importance difference
        for p_idx in range(len(self.pathway_names)):
            path_name = self.pathway_names[p_idx]

            # Get importance values for this pathway across all patients in each class
            class0_values = torch.stack([
                self.patient_data["pathway_importance"][i][p_idx] for i in class0_idx
            ]).numpy()

            class1_values = torch.stack([
                self.patient_data["pathway_importance"][i][p_idx] for i in class1_idx
            ]).numpy()

            # Calculate statistics
            mean0 = np.mean(class0_values)
            mean1 = np.mean(class1_values)
            fold_change = mean1 / mean0 if mean0 > 0 else float('inf')

            # T-test for significance
            t_stat, p_value = stats.ttest_ind(class0_values, class1_values, equal_var=False)

            # Store result
            pathway_results.append({
                "pathway_name": path_name,
                "importance_class0": mean0,
                "importance_class1": mean1,
                "fold_change": fold_change,
                "log2_fold_change": np.log2(fold_change) if fold_change > 0 else float('inf'),
                "p_value": p_value,
                "significant": (
                                           fold_change >= min_fold_change or fold_change <= 1 / min_fold_change) and p_value <= max_pvalue,
                "enriched_in": classes[1] if fold_change > 1 else classes[0]
            })

        # Similar analysis for genes
        for g_idx in range(len(self.gene_names)):
            gene_name = self.gene_names[g_idx]

            # Get importance values for this gene across all patients in each class
            class0_values = torch.stack([
                self.patient_data["gene_importance"][i][g_idx] for i in class0_idx
            ]).numpy()

            class1_values = torch.stack([
                self.patient_data["gene_importance"][i][g_idx] for i in class1_idx
            ]).numpy()

            # Calculate statistics
            mean0 = np.mean(class0_values)
            mean1 = np.mean(class1_values)
            fold_change = mean1 / mean0 if mean0 > 0 else float('inf')

            # T-test for significance
            t_stat, p_value = stats.ttest_ind(class0_values, class1_values, equal_var=False)

            # Store result
            gene_results.append({
                "gene_name": gene_name,
                "importance_class0": mean0,
                "importance_class1": mean1,
                "fold_change": fold_change,
                "log2_fold_change": np.log2(fold_change) if fold_change > 0 else float('inf'),
                "p_value": p_value,
                "significant": (
                                           fold_change >= min_fold_change or fold_change <= 1 / min_fold_change) and p_value <= max_pvalue,
                "enriched_in": classes[1] if fold_change > 1 else classes[0]
            })

        # Convert to DataFrames
        pathway_df = pd.DataFrame(pathway_results)
        gene_df = pd.DataFrame(gene_results)

        # Sort by significance and fold change
        pathway_df = pathway_df.sort_values(by=["significant", "p_value", "fold_change"],
                                            ascending=[False, True, False])
        gene_df = gene_df.sort_values(by=["significant", "p_value", "fold_change"],
                                      ascending=[False, True, False])

        # Store results
        self.biomarker_results = {
            "pathway_biomarkers": pathway_df,
            "gene_biomarkers": gene_df,
            "classes": classes
        }

        return self.biomarker_results

    def get_patient_importance(self, patient_id=None, patient_idx=None, top_k=10):
        """
        Get importance scores for a specific patient.

        Args:
            patient_id: Optional patient ID to lookup
            patient_idx: Optional index in the patient_data
            top_k: Number of top elements to return

        Returns:
            Dictionary with patient importance data
        """
        if patient_id is not None:
            try:
                idx = self.patient_data["patient_id"].index(patient_id)
            except ValueError:
                raise ValueError(f"Patient ID {patient_id} not found")
        elif patient_idx is not None:
            idx = patient_idx
        else:
            raise ValueError("Either patient_id or patient_idx must be provided")

        return self._get_patient_report(idx, top_k)

    def visualize_biomarkers(self, output_dir="./biomarker_figures"):
        """
        Create visualizations for identified biomarkers.

        Args:
            output_dir: Directory to save the figures

        Returns:
            List of generated figure paths
        """
        try:
            if self.biomarker_results is None:
                raise ValueError("Run identify_differential_biomarkers() first")

            os.makedirs(output_dir, exist_ok=True)
            figure_paths = []

            # Get results
            pathway_df = self.biomarker_results["pathway_biomarkers"]
            gene_df = self.biomarker_results["gene_biomarkers"]
            classes = self.biomarker_results["classes"]

            # 1. Volcano plot for pathways
            fig, ax = plt.subplots(figsize=(12, 8))

            # Create volcano plot
            significant = pathway_df['significant']

            # Plot points
            ax.scatter(
                pathway_df.loc[~significant, 'log2_fold_change'],
                -np.log10(pathway_df.loc[~significant, 'p_value']),
                alpha=0.5, s=30, color='gray', label='Not Significant'
            )

            ax.scatter(
                pathway_df.loc[significant, 'log2_fold_change'],
                -np.log10(pathway_df.loc[significant, 'p_value']),
                alpha=0.8, s=50, color='red', label='Significant'
            )

            # Add labels for top significant pathways
            top_pathways = pathway_df.loc[significant].sort_values('p_value').head(10)
            for _, row in top_pathways.iterrows():
                ax.annotate(
                    row['pathway_name'],
                    xy=(row['log2_fold_change'], -np.log10(row['p_value'])),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3)
                )

            # Add threshold lines - using the class variables now
            ax.axhline(-np.log10(self.max_pvalue), linestyle='--', color='gray', alpha=0.6)
            ax.axvline(np.log2(self.min_fold_change), linestyle='--', color='gray', alpha=0.6)
            ax.axvline(-np.log2(self.min_fold_change), linestyle='--', color='gray', alpha=0.6)

            # Labels and title
            ax.set_xlabel('Log2 Fold Change', fontsize=12)
            ax.set_ylabel('-Log10 P-value', fontsize=12)
            ax.set_title(f'Pathway Importance: {classes[0]} vs {classes[1]}', fontsize=14)
            ax.legend()

            # Save figure
            path = os.path.join(output_dir, 'pathway_volcano_plot.png')
            fig.savefig(path, dpi=300, bbox_inches='tight')
            figure_paths.append(path)
            plt.close(fig)

            # 2. Bar plot of top differential pathways
            top_diff_pathways = pathway_df[pathway_df['significant']].head(15)
            if len(top_diff_pathways) > 0:
                fig, ax = plt.subplots(figsize=(12, 10))

                # Prepare data for grouped bar plot
                pathway_names = top_diff_pathways['pathway_name'].tolist()
                class0_values = top_diff_pathways['importance_class0'].tolist()
                class1_values = top_diff_pathways['importance_class1'].tolist()

                # Plot bars
                x = np.arange(len(pathway_names))
                width = 0.35

                ax.barh(x - width / 2, class0_values, width, label=classes[0], color='skyblue')
                ax.barh(x + width / 2, class1_values, width, label=classes[1], color='salmon')

                # Add labels and formatting
                ax.set_yticks(x)
                ax.set_yticklabels(pathway_names)
                ax.invert_yaxis()  # Labels read top-to-bottom

                ax.set_xlabel('Importance Score', fontsize=12)
                ax.set_title('Top Differential Pathways', fontsize=14)
                ax.legend()

                # Save figure
                path = os.path.join(output_dir, 'top_differential_pathways.png')
                fig.savefig(path, dpi=300, bbox_inches='tight')
                figure_paths.append(path)
                plt.close(fig)

            # 3. Similar volcano plot for genes
            fig, ax = plt.subplots(figsize=(12, 8))

            # Create volcano plot
            significant = gene_df['significant']

            # Plot points
            ax.scatter(
                gene_df.loc[~significant, 'log2_fold_change'],
                -np.log10(gene_df.loc[~significant, 'p_value']),
                alpha=0.5, s=20, color='gray', label='Not Significant'
            )

            ax.scatter(
                gene_df.loc[significant, 'log2_fold_change'],
                -np.log10(gene_df.loc[significant, 'p_value']),
                alpha=0.8, s=30, color='blue', label='Significant'
            )

            # Add labels for top significant genes
            top_genes = gene_df.loc[significant].sort_values('p_value').head(10)
            for _, row in top_genes.iterrows():
                ax.annotate(
                    row['gene_name'],
                    xy=(row['log2_fold_change'], -np.log10(row['p_value'])),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3)
                )

            # Add threshold lines - using the class variables
            ax.axhline(-np.log10(self.max_pvalue), linestyle='--', color='gray', alpha=0.6)
            ax.axvline(np.log2(self.min_fold_change), linestyle='--', color='gray', alpha=0.6)
            ax.axvline(-np.log2(self.min_fold_change), linestyle='--', color='gray', alpha=0.6)

            # Labels and title
            ax.set_xlabel('Log2 Fold Change', fontsize=12)
            ax.set_ylabel('-Log10 P-value', fontsize=12)
            ax.set_title(f'Gene Importance: {classes[0]} vs {classes[1]}', fontsize=14)
            ax.legend()

            # Save figure
            path = os.path.join(output_dir, 'gene_volcano_plot.png')
            fig.savefig(path, dpi=300, bbox_inches='tight')
            figure_paths.append(path)
            plt.close(fig)

            # 4. Generate heatmap for top differential genes by class
            top_diff_genes = gene_df[gene_df['significant']].head(25)
            if len(top_diff_genes) > 0:
                fig, ax = plt.subplots(figsize=(10, 12))

                # Prepare data for heatmap
                gene_names = top_diff_genes['gene_name'].tolist()

                # Create matrix: rows=genes, cols=classes
                heatmap_data = np.zeros((len(gene_names), 2))
                for i, gene_name in enumerate(gene_names):
                    row = top_diff_genes[top_diff_genes['gene_name'] == gene_name].iloc[0]
                    heatmap_data[i, 0] = row['importance_class0']
                    heatmap_data[i, 1] = row['importance_class1']

                # Plot heatmap
                sns.heatmap(
                    heatmap_data,
                    annot=True,
                    fmt=".3f",
                    yticklabels=gene_names,
                    xticklabels=classes,
                    cmap="YlOrRd",
                    ax=ax
                )

                # Add labels and title
                ax.set_title('Top Differential Genes by Class', fontsize=14)

                # Save figure
                path = os.path.join(output_dir, 'gene_importance_heatmap.png')
                fig.savefig(path, dpi=300, bbox_inches='tight')
                figure_paths.append(path)
                plt.close(fig)

            print(f"Successfully created {len(figure_paths)} visualizations in {output_dir}")
            return figure_paths

        except Exception as e:
            print(f"Error generating visualizations: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def generate_complete_report(self, output_dir="./biomarker_report"):
        """
        Generate a complete biomarker analysis report with all results.

        Args:
            output_dir: Directory to save the report files

        Returns:
            Dictionary with report paths
        """
        os.makedirs(output_dir, exist_ok=True)

        # Analyze by group if not done already
        if not self.population_data:
            self.analyze_by_group()

        # Identify biomarkers if not done already
        if self.biomarker_results is None:
            self.identify_differential_biomarkers()

        # Save biomarker results to CSV
        pathway_csv = os.path.join(output_dir, "pathway_biomarkers.csv")
        gene_csv = os.path.join(output_dir, "gene_biomarkers.csv")

        self.biomarker_results["pathway_biomarkers"].to_csv(pathway_csv, index=False)
        self.biomarker_results["gene_biomarkers"].to_csv(gene_csv, index=False)

        # Generate visualizations
        figure_dir = os.path.join(output_dir, "figures")
        figure_paths = self.visualize_biomarkers(figure_dir)

        # Create summary markdown report
        report_path = os.path.join(output_dir, "biomarker_report.md")

        with open(report_path, 'w') as f:
            f.write("# Biomarker Analysis Report\n\n")

            # Dataset summary
            f.write("## Dataset Summary\n\n")
            patient_count = len(self.patient_data["patient_id"])
            f.write(f"Total patients analyzed: {patient_count}\n\n")

            # Class distribution
            f.write("### Class Distribution\n\n")
            for label_name, data in self.population_data.items():
                f.write(f"- {label_name}: {data['count']} patients\n")
            f.write("\n")

            # Differential biomarkers
            f.write("## Differential Biomarkers\n\n")

            # Pathways
            significant_pathways = self.biomarker_results["pathway_biomarkers"][
                self.biomarker_results["pathway_biomarkers"]["significant"]
            ]
            f.write(f"### Significant Differential Pathways: {len(significant_pathways)}\n\n")

            if len(significant_pathways) > 0:
                f.write("| Pathway | Enriched In | Fold Change | P-value |\n")
                f.write("|---------|------------|-------------|--------|\n")

                for _, row in significant_pathways.head(15).iterrows():
                    f.write(
                        f"| {row['pathway_name']} | {row['enriched_in']} | {row['fold_change']:.2f} | {row['p_value']:.4f} |\n")

                if len(significant_pathways) > 15:
                    f.write(f"| ... | ... | ... | ... |\n")
            else:
                f.write("No significant differential pathways found.\n")

            f.write("\n")

            # Genes
            significant_genes = self.biomarker_results["gene_biomarkers"][
                self.biomarker_results["gene_biomarkers"]["significant"]
            ]
            f.write(f"### Significant Differential Genes: {len(significant_genes)}\n\n")

            if len(significant_genes) > 0:
                f.write("| Gene | Enriched In | Fold Change | P-value |\n")
                f.write("|------|------------|-------------|--------|\n")

                for _, row in significant_genes.head(15).iterrows():
                    f.write(
                        f"| {row['gene_name']} | {row['enriched_in']} | {row['fold_change']:.2f} | {row['p_value']:.4f} |\n")

                if len(significant_genes) > 15:
                    f.write(f"| ... | ... | ... | ... |\n")
            else:
                f.write("No significant differential genes found.\n")

            f.write("\n")

            # Figures
            f.write("## Visualizations\n\n")

            for fig_path in figure_paths:
                fig_name = os.path.basename(fig_path)
                rel_path = os.path.join("figures", fig_name)
                f.write(f"![{fig_name}]({rel_path})\n\n")

        return {
            "report": report_path,
            "pathway_biomarkers": pathway_csv,
            "gene_biomarkers": gene_csv,
            "figures": figure_paths
        }


def integrate_with_evaluate_model(model, test_loader, config, device, output_dir="./biomarker_analysis"):
    """
    Integrate biomarker analysis with the evaluate_model function.

    Args:
        model: The PathwayEmbeddingModel instance
        test_loader: PyTorch Geometric DataLoader with test samples
        config: Configuration dictionary
        device: Computation device
        output_dir: Directory to save results

    Returns:
        Dictionary with evaluation metrics and biomarker analysis results
    """
    model.eval()

    # Regular model evaluation metrics
    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_targets = []
    all_probs = []
    all_patient_ids = []

    # Initialize biomarker analysis
    analyzer = BiomarkerAnalysis(
        model=model,
        label_dict=config.get('label_dict', {0: "Class_0", 1: "Class_1"})
    )

    with torch.no_grad():
        for batch in test_loader:
            batch.to(device)
            target = batch.y
            patient_id = batch.patient_id

            # Forward pass with importance calculation
            outputs = model(batch, return_importance=True)

            # Calculate loss
            loss = torch.nn.functional.cross_entropy(outputs, target)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            pred = outputs.argmax(dim=1)

            # Update metrics
            total_loss += loss.item()
            correct += (pred == target).sum().item()
            total += target.size(0)

            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())
            all_probs.append(probs.cpu())
            all_patient_ids.append(patient_id)

            # Store patient data for biomarker analysis
            analyzer.patient_data["patient_id"].append(patient_id)
            analyzer.patient_data["true_label"].append(target.item())
            analyzer.patient_data["pred_label"].append(pred.item())
            analyzer.patient_data["pathway_importance"].append(model.pathway_importance.cpu().clone())
            analyzer.patient_data["gene_pathway_attention"].append(model.gene_pathway_attention.cpu().clone())
            analyzer.patient_data["gene_importance"].append(model.get_gene_importance().cpu().clone())

    # Calculate metrics
    try:
        # Convert lists to arrays for proper metrics
        if all_preds:
            all_preds = torch.cat(all_preds, dim=0).numpy()
        else:
            all_preds = np.array([])

        if all_targets:
            all_targets = torch.cat(all_targets, dim=0).numpy()
        else:
            all_targets = np.array([])

        if all_probs:
            all_probs = torch.cat(all_probs, dim=0).numpy()
        else:
            all_probs = np.array([])

        metrics = {
            'loss': total_loss / len(test_loader) if len(test_loader) > 0 else 0,
            'acc': 100. * correct / total if total > 0 else 0,
            'all_preds': all_preds,
            'all_targets': all_targets,
            'all_probs': all_probs,
            'patient_ids': all_patient_ids
        }

        # Perform biomarker analysis
        print("Analyzing patient groups...")
        analyzer.analyze_by_group()

        print("Identifying differential biomarkers...")
        analyzer.identify_differential_biomarkers()

        print(f"Generating biomarker report in {output_dir}...")
        report_paths = analyzer.generate_complete_report(output_dir)

        # Debugging info
        print(f"Biomarker report generated at: {report_paths['report']}")
        print(f"Figure paths: {report_paths['figures']}")

        # Add biomarker results to metrics
        metrics['biomarker_analysis'] = {
            'analyzer': analyzer,
            'report_paths': report_paths
        }

        return metrics
    except Exception as e:
        print(f"Error in evaluate_model: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'loss': 0,
            'acc': 0,
            'error': str(e)
        }