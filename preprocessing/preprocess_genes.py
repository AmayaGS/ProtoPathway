"""
Gene Expression Preprocessing (Stage 2 - Per Dataset)

Takes the Reactome base pathways and creates dataset-specific outputs:
1. Filtered gene expression matrix
2. Dataset-specific pathway set (only pathways with genes in this dataset)
3. Bipartite graph structure for the GATv2 model

Pipeline:
1. Load gene expression data
2. Load Reactome base pathways (from Stage 1)
3. Optionally filter expression data (for non-TCGA datasets)
4. Filter pathways to those with sufficient genes in expression data
5. Apply pathway variance filtering (remove uninformative pathways)
6. Build bipartite graph structure
7. Save all outputs

Usage:
    python main.py preprocess genes --config configs/preprocessing/preprocess_genes.yaml
    python main.py preprocess genes --config configs/preprocessing/preprocess_genes.yaml dataset=HNSC
"""

import os
import json
import logging
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

from utils.genes import (
    load_expression_data,
    extract_protein_coding_genes,
    filter_expression_data,
    compute_pathway_variance,
    filter_pathways_by_variance
)
from utils.graphs import (
    build_bipartite_graph,
    filter_pathways_by_gene_coverage,
    compute_graph_statistics,
    visualize_bipartite_graph,
    visualize_hypergraph
)


def plot_gene_coverage(pathway_df, output_path):
    """Plot distribution of gene coverage per pathway."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Coverage percentage
    axes[0].hist(pathway_df['coverage'] * 100, bins=30, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Gene Coverage (%)', fontsize=12)
    axes[0].set_ylabel('Number of Pathways', fontsize=12)
    axes[0].set_title('Pathway Gene Coverage in Expression Data')
    axes[0].axvline(x=pathway_df['coverage'].median() * 100, color='red',
                    linestyle='--', label=f"Median: {pathway_df['coverage'].median():.1%}")
    axes[0].legend()

    # Absolute gene counts
    axes[1].hist(pathway_df['genes_present_count'], bins=30, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Genes Present', fontsize=12)
    axes[1].set_ylabel('Number of Pathways', fontsize=12)
    axes[1].set_title('Number of Pathway Genes in Expression Data')
    axes[1].axvline(x=pathway_df['genes_present_count'].median(), color='red',
                    linestyle='--', label=f"Median: {pathway_df['genes_present_count'].median():.0f}")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    logging.info(f"Saved gene coverage plot to {output_path}")


def plot_pathway_variance(pathway_df, threshold, output_path):
    """Plot pathway variance distribution."""
    plt.figure(figsize=(10, 5))
    plt.hist(pathway_df['variance'], bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=threshold, color='red', linestyle='--',
                label=f'Threshold: {threshold:.4f}')
    plt.xlabel('Pathway Activity Variance', fontsize=12)
    plt.ylabel('Number of Pathways', fontsize=12)
    plt.title('Distribution of Pathway Variance')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    logging.info(f"Saved pathway variance plot to {output_path}")


def plot_expression_distributions(expr_df, output_path, sample_n=5):
    """Plot expression distributions for a few sample patients."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Sample patients
    sample_patients = expr_df.index[:sample_n]

    # Per-patient distribution
    for patient in sample_patients:
        axes[0].hist(expr_df.loc[patient], bins=50, alpha=0.5, label=patient[:10])
    axes[0].set_xlabel('Expression Value', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title(f'Expression Distribution (sample of {sample_n} patients)')
    axes[0].legend(fontsize=8)

    # Overall distribution (flatten)
    axes[1].hist(expr_df.values.flatten(), bins=100, alpha=0.7, color='steelblue')
    axes[1].set_xlabel('Expression Value', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Overall Expression Distribution')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    logging.info(f"Saved expression distribution plot to {output_path}")


def run(cfg):
    """
    Run the gene expression preprocessing pipeline.

    Args:
        cfg: OmegaConf configuration object

    Returns:
        dict: Paths to output files
    """
    dataset = cfg['dataset']

    logging.info("=" * 60)
    logging.info(f"Gene Expression Preprocessing: {dataset}")
    logging.info("=" * 60)

    # Create output directories
    output_dir = cfg['output']['dir']
    figures_dir = cfg['output']['figures_dir']
    generate_figures = cfg['output'].get('generate_figures', True)

    os.makedirs(output_dir, exist_ok=True)
    if generate_figures:
        os.makedirs(figures_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Load gene expression data
    # -------------------------------------------------------------------------
    logging.info("\n[Step 1/6] Loading gene expression data...")
    expr_df = load_expression_data(cfg['input']['expression'])

    original_genes = set(expr_df.columns)
    original_patients = set(expr_df.index)

    # -------------------------------------------------------------------------
    # Step 2: Optionally filter expression data
    # -------------------------------------------------------------------------
    expr_filter_cfg = cfg['expression_filtering']
    if expr_filter_cfg['enabled']:
        logging.info("\n[Step 2/6] Filtering expression data...")

        # Load protein-coding genes if GTF provided
        protein_coding = None
        if cfg['input'].get('gtf'):
            protein_coding = extract_protein_coding_genes(cfg['input']['gtf'])

        expr_df = filter_expression_data(
            expr_df,
            protein_coding_genes=protein_coding,
            min_expression=expr_filter_cfg['min_expression'],
            min_proportion=expr_filter_cfg['min_proportion'],
            variance_proportion=expr_filter_cfg['variance_proportion'],
            log_transform=expr_filter_cfg['log_transform'],
            center=expr_filter_cfg['center']
        )
    else:
        logging.info("\n[Step 2/6] Skipping expression filtering (using preprocessed data)")

    # Plot expression distributions
    if generate_figures:
        plot_expression_distributions(
            expr_df,
            os.path.join(figures_dir, 'expression_distributions.pdf')
        )

    # -------------------------------------------------------------------------
    # Step 3: Load Reactome base pathways
    # -------------------------------------------------------------------------
    logging.info("\n[Step 3/6] Loading Reactome base pathways...")

    with open(cfg['input']['reactome_base'], 'rb') as f:
        reactome_data = pickle.load(f)

    pathway_df = reactome_data['pathways_df'].copy()
    logging.info(f"Loaded {len(pathway_df)} base pathways")
    logging.info(f"  Parameters: {reactome_data.get('parameter_string', 'unknown')}")

    # -------------------------------------------------------------------------
    # Step 4: Filter pathways by gene coverage
    # -------------------------------------------------------------------------
    logging.info("\n[Step 4/6] Filtering pathways by gene coverage...")

    pathway_filter_cfg = cfg['pathway_filtering']
    available_genes = set(expr_df.columns)
    pathway_df = filter_pathways_by_gene_coverage(
        pathway_df,
        available_genes,
        min_genes=pathway_filter_cfg['min_genes_present']
    )

    # Plot gene coverage
    if generate_figures:
        plot_gene_coverage(
            pathway_df,
            os.path.join(figures_dir, 'gene_coverage.pdf')
        )

    # -------------------------------------------------------------------------
    # Step 5: Apply pathway variance filtering
    # -------------------------------------------------------------------------
    variance_filter_cfg = pathway_filter_cfg['variance_filter']
    if variance_filter_cfg['enabled']:
        logging.info("\n[Step 5/6] Filtering pathways by variance...")

        # Build pathway_genes dict using genes_present (genes actually in expression data)
        pathway_genes = {}
        for _, row in pathway_df.iterrows():
            pathway_genes[row['pathway_id']] = row['genes_present']

        # Compute variances
        pathway_variances = compute_pathway_variance(expr_df, pathway_genes)

        # Filter
        keep_percentile = variance_filter_cfg['keep_percentile']
        threshold = np.percentile(list(pathway_variances.values()), 100 - keep_percentile)

        pathway_df = filter_pathways_by_variance(
            pathway_df,
            pathway_variances,
            keep_percentile=keep_percentile
        )

        # Plot variance distribution
        if generate_figures:
            plot_pathway_variance(
                pathway_df,
                threshold,
                os.path.join(figures_dir, 'pathway_variance.pdf')
            )
    else:
        logging.info("\n[Step 5/6] Skipping pathway variance filtering")

    # -------------------------------------------------------------------------
    # Step 6: Build bipartite graph
    # -------------------------------------------------------------------------
    logging.info("\n[Step 6/6] Building bipartite graph...")

    # Get genes used by remaining pathways
    genes_in_pathways = set()
    pathway_genes_final = {}
    for _, row in pathway_df.iterrows():
        genes = row['genes_present']
        genes_in_pathways.update(genes)
        pathway_genes_final[row['pathway_id']] = genes

    # Filter expression to only genes in pathways
    genes_to_keep = list(genes_in_pathways & set(expr_df.columns))
    expr_df_filtered = expr_df[genes_to_keep]

    logging.info(f"Expression matrix filtered to {len(genes_to_keep)} genes in pathways")

    # Build graph
    graph_data = build_bipartite_graph(genes_to_keep, pathway_genes_final)
    graph_stats = compute_graph_statistics(graph_data)

    if generate_figures:
        logging.info("Generating graph visualizations...")

        # Get visualization config (with defaults)
        vis_cfg = cfg.get('visualization', {})
        bipartite_cfg = vis_cfg.get('bipartite', {})
        hypergraph_cfg = vis_cfg.get('hypergraph', {})

        # Bipartite graph visualization
        visualize_bipartite_graph(
            graph_data,
            pathway_df=pathway_df,
            max_pathways=bipartite_cfg.get('max_pathways', 30),
            max_genes_per_pathway=bipartite_cfg.get('max_genes_per_pathway', 15),
            output_path=os.path.join(figures_dir, 'bipartite_graph.pdf')
        )

        # Hypergraph visualization
        visualize_hypergraph(
            graph_data,
            pathway_df=pathway_df,
            max_pathways=hypergraph_cfg.get('max_pathways', 20),
            output_path=os.path.join(figures_dir, 'hypergraph.pdf')
        )

    # -------------------------------------------------------------------------
    # Summary statistics
    # -------------------------------------------------------------------------
    logging.info("\n" + "=" * 60)
    logging.info("Preprocessing Summary")
    logging.info("=" * 60)
    logging.info(f"  Dataset: {dataset}")
    logging.info(f"  Patients: {len(expr_df_filtered)}")
    logging.info(f"  Genes (in pathways): {graph_stats['num_genes']}")
    logging.info(f"  Pathways: {graph_stats['num_pathways']}")
    logging.info(f"  Graph edges: {graph_stats['num_edges']}")
    logging.info(f"  Mean genes/pathway: {graph_stats['mean_genes_per_pathway']:.1f}")
    logging.info(f"  Mean pathways/gene: {graph_stats['mean_pathways_per_gene']:.1f}")

    # Source breakdown
    source_counts = pathway_df['source'].value_counts()
    for source, count in source_counts.items():
        logging.info(f"  {source.capitalize()} pathways: {count}")

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------
    logging.info("\n" + "=" * 60)
    logging.info("Saving outputs...")
    logging.info("=" * 60)

    # 1. Gene expression matrix
    expr_path = os.path.join(output_dir, 'gene_expression.csv')
    expr_df_filtered.to_csv(expr_path)
    logging.info(f"Saved expression matrix to {expr_path}")

    # 2. Pathway DataFrame
    pathway_path = os.path.join(output_dir, 'pathways.csv')
    pathway_df.to_csv(pathway_path, index=False)
    logging.info(f"Saved pathways to {pathway_path}")

    # 3. Bipartite graph
    graph_path = os.path.join(output_dir, 'bipartite_graph.pt')
    torch.save(graph_data, graph_path)
    logging.info(f"Saved bipartite graph to {graph_path}")

    # 4. Preprocessing info
    info = {
        'dataset': dataset,
        'original_patients': len(original_patients),
        'original_genes': len(original_genes),
        'final_patients': len(expr_df_filtered),
        'final_genes': len(genes_to_keep),
        'final_pathways': len(pathway_df),
        'expression_filtering_enabled': expr_filter_cfg['enabled'],
        'variance_filtering_enabled': variance_filter_cfg['enabled'],
        'variance_keep_percentile': variance_filter_cfg['keep_percentile'] if variance_filter_cfg['enabled'] else None,
        'reactome_base': cfg['input']['reactome_base'],
        'graph_statistics': graph_stats
    }

    info_path = os.path.join(output_dir, 'preprocessing_info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    logging.info(f"Saved preprocessing info to {info_path}")

    # 5. Config snapshot
    from omegaconf import OmegaConf
    config_path = os.path.join(output_dir, 'preprocess_genes_config.yaml')
    with open(config_path, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    logging.info(f"Saved config to {config_path}")

    logging.info("\n" + "=" * 60)
    logging.info("Gene expression preprocessing complete!")
    logging.info("=" * 60)

    return {
        'expression': expr_path,
        'pathways': pathway_path,
        'graph': graph_path,
        'info': info_path
    }