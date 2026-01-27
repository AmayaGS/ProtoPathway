"""
Reactome Base Preprocessing

Creates a filtered, non-redundant pathway set combining Reactome pathways
with MSigDB Hallmark gene sets. This runs once and produces a base pathway
file that can be used across multiple datasets.

Pipeline:
1. Load Reactome pathways from GMT
2. Build hierarchy and compute depths
3. Filter by depth (remove overly broad top-level pathways)
4. Filter by category (remove housekeeping, irrelevant categories)
5. Filter by size (min/max genes per pathway)
6. Load and integrate Hallmark gene sets (for gap-filling)
7. Remove redundant pathways (high Jaccard similarity)
8. Save outputs with parameter-encoded filename

Usage:
    python main.py preprocess reactome --config configs/preprocessing/reactome_base.yaml
"""

import os
import logging
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from utils.pathways import (
    parse_gmt,
    build_reactome_hierarchy,
    compute_pathway_depths,
    get_pathway_categories,
    filter_by_depth,
    filter_by_category,
    filter_by_size,
    load_hallmark_csv,
    integrate_hallmarks,
    remove_redundant_pathways,
    compute_gene_pathway_statistics,
    generate_parameter_string
)


def plot_depth_histogram(pathway_df, target_depth, output_path):
    """Plot distribution of pathway depths."""
    plt.figure(figsize=(8, 5))
    plt.hist(pathway_df['depth'].dropna(), bins=range(int(pathway_df['depth'].max()) + 2),
             edgecolor='black', alpha=0.7)
    plt.axvline(x=target_depth, color='red', linestyle='--',
                label=f'Max depth threshold ({target_depth})')
    plt.xlabel('Pathway Depth', fontsize=12)
    plt.ylabel('Number of Pathways', fontsize=12)
    plt.title('Distribution of Reactome Pathway Depths')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    logging.info(f"Saved depth histogram to {output_path}")


def plot_size_histogram(pathway_df, min_genes, max_genes, output_path, title_suffix=""):
    """Plot distribution of pathway sizes."""
    plt.figure(figsize=(10, 5))
    plt.hist(pathway_df['gene_count'], bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=min_genes, color='red', linestyle='--', label=f'Min ({min_genes})')
    plt.axvline(x=max_genes, color='red', linestyle='--', label=f'Max ({max_genes})')
    plt.xlabel('Number of Genes per Pathway', fontsize=12)
    plt.ylabel('Number of Pathways', fontsize=12)
    plt.title(f'Distribution of Pathway Sizes{title_suffix}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    logging.info(f"Saved size histogram to {output_path}")


def plot_category_distribution(pathway_df, output_path):
    """Plot distribution of pathways across categories."""
    category_counts = pathway_df['category'].value_counts()

    plt.figure(figsize=(12, 6))
    bars = plt.barh(range(len(category_counts)), category_counts.values)
    plt.yticks(range(len(category_counts)), category_counts.index)
    plt.xlabel('Number of Pathways', fontsize=12)
    plt.title('Pathways per Category (after filtering)')

    # Add count labels
    for bar, count in zip(bars, category_counts.values):
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 str(count), va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    logging.info(f"Saved category distribution to {output_path}")


def plot_source_composition(pathway_df, output_path):
    """Plot Reactome vs Hallmark composition."""
    source_counts = pathway_df['source'].value_counts()

    plt.figure(figsize=(6, 6))
    colors = ['#2ecc71', '#3498db']  # Green for Reactome, blue for Hallmark
    plt.pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90)
    plt.title(f'Pathway Sources (n={len(pathway_df)})')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    logging.info(f"Saved source composition to {output_path}")


def run(cfg):
    """
    Run the Reactome preprocessing pipeline.

    Args:
        cfg: OmegaConf configuration object

    Returns:
        dict: Paths to output files
    """
    logging.info("=" * 60)
    logging.info("Starting Reactome Base Preprocessing")
    logging.info("=" * 60)

    # Create output directories
    os.makedirs(cfg['output']['pathways'], exist_ok=True)
    if cfg['output'].get('generate_figures', True):
        os.makedirs(cfg['output']['figures_dir'], exist_ok=True)

    # Generate parameter string for output filename
    param_str = generate_parameter_string(cfg)
    logging.info(f"Parameter string: {param_str}")

    # -------------------------------------------------------------------------
    # Step 1: Load Reactome pathways
    # -------------------------------------------------------------------------
    logging.info("\n[Step 1/7] Loading Reactome pathways...")
    reactome_dict, reactome_df = parse_gmt(cfg['input']['reactome_gmt'])

    # -------------------------------------------------------------------------
    # Step 2: Build hierarchy and compute depths
    # -------------------------------------------------------------------------
    logging.info("\n[Step 2/7] Building Reactome hierarchy...")
    hierarchy_graph, pathway_names_df = build_reactome_hierarchy(
        cfg['input']['reactome_relations'],
        cfg['input']['reactome_pathways']
    )

    depths = compute_pathway_depths(hierarchy_graph)
    reactome_df['depth'] = reactome_df['pathway_id'].map(depths)

    # Get categories for all pathways
    pathway_categories = get_pathway_categories(hierarchy_graph, pathway_names_df)
    reactome_df['category'] = reactome_df['pathway_id'].map(pathway_categories)

    # Plot initial depth distribution
    if cfg['output'].get('generate_figures', True):
        plot_depth_histogram(
            reactome_df,
            cfg['filtering']['target_depth'],
            os.path.join(cfg['output']['figures_dir'], 'depth_histogram_initial.pdf')
        )

    # -------------------------------------------------------------------------
    # Step 3: Filter by depth
    # -------------------------------------------------------------------------
    logging.info("\n[Step 3/7] Filtering by depth...")
    filtered_df = filter_by_depth(
        reactome_df,
        hierarchy_graph,
        depths,
        cfg['filtering']['target_depth']
    )

    # -------------------------------------------------------------------------
    # Step 4: Filter by category
    # -------------------------------------------------------------------------
    logging.info("\n[Step 4/7] Filtering by category...")
    filtered_df = filter_by_category(
        filtered_df,
        pathway_categories,
        cfg['filtering']['exclude_categories']
    )

    # Plot category distribution after filtering
    if cfg['output'].get('generate_figures', True):
        plot_category_distribution(
            filtered_df,
            os.path.join(cfg['output']['figures_dir'], 'category_distribution.pdf')
        )

    # -------------------------------------------------------------------------
    # Step 5: Filter by size
    # -------------------------------------------------------------------------
    logging.info("\n[Step 5/7] Filtering by size...")

    # Plot size distribution before filtering
    if cfg['output'].get('generate_figures', True):
        plot_size_histogram(
            filtered_df,
            cfg['filtering']['min_genes'],
            cfg['filtering']['max_genes'],
            os.path.join(cfg['output']['figures_dir'], 'size_histogram_before.pdf'),
            title_suffix=" (before size filter)"
        )

    filtered_df = filter_by_size(
        filtered_df,
        cfg['filtering']['min_genes'],
        cfg['filtering']['max_genes']
    )

    # Plot size distribution after filtering
    if cfg['output'].get('generate_figures', True):
        plot_size_histogram(
            filtered_df,
            cfg['filtering']['min_genes'],
            cfg['filtering']['max_genes'],
            os.path.join(cfg['output']['figures_dir'], 'size_histogram_after.pdf'),
            title_suffix=" (after size filter)"
        )

    # -------------------------------------------------------------------------
    # Step 6: Load and integrate Hallmark gene sets
    # -------------------------------------------------------------------------
    logging.info("\n[Step 6/7] Integrating Hallmark gene sets...")

    hallmark_df = load_hallmark_csv(
        cfg['input']['hallmark_csv'],
        cfg['filtering'].get('hallmark_additions', None)
    )

    combined_df, hallmark_displacement_report = integrate_hallmarks(
        filtered_df,
        hallmark_df,
        jaccard_threshold=cfg['filtering']['redundancy']['jaccard_threshold']
    )

    # -------------------------------------------------------------------------
    # Step 7: Remove redundant pathways
    # -------------------------------------------------------------------------
    logging.info("\n[Step 7/7] Removing redundant pathways...")
    # final_df, redundancy_report = remove_redundant_pathways(
    #     combined_df,
    #     jaccard_threshold=cfg['filtering']['redundancy']['jaccard_threshold']
    # )
    final_df, redundancy_report = remove_redundant_pathways(
        combined_df,
        jaccard_threshold=cfg['filtering']['redundancy']['jaccard_threshold'],
        hierarchy_graph=hierarchy_graph,
        depths=depths
    )

    # Plot final composition
    if cfg['output'].get('generate_figures', True):
        plot_source_composition(
            final_df,
            os.path.join(cfg['output']['figures_dir'], 'source_composition.pdf')
        )

        plot_size_histogram(
            final_df,
            cfg['filtering']['min_genes'],
            cfg['filtering']['max_genes'],
            os.path.join(cfg['output']['figures_dir'], 'size_histogram_final.pdf'),
            title_suffix=" (final)"
        )

    # -------------------------------------------------------------------------
    # Compute and log statistics
    # -------------------------------------------------------------------------
    stats = compute_gene_pathway_statistics(final_df)

    logging.info("\n" + "=" * 60)
    logging.info("Final Pathway Set Statistics:")
    logging.info("=" * 60)
    logging.info(f"  Total pathways: {stats['total_pathways']}")
    logging.info(f"  Total unique genes: {stats['total_unique_genes']}")
    logging.info(f"  Mean genes per pathway: {stats['mean_genes_per_pathway']:.1f}")
    logging.info(f"  Median genes per pathway: {stats['median_genes_per_pathway']:.1f}")
    logging.info(f"  Genes in single pathway: {stats['genes_in_single_pathway']}")
    logging.info(f"  Genes in 10+ pathways: {stats['genes_in_10plus_pathways']}")
    logging.info(f"  Mean pathways per gene: {stats['mean_pathways_per_gene']:.2f}")

    # Source breakdown
    source_counts = final_df['source'].value_counts()
    for source, count in source_counts.items():
        logging.info(f"  {source.capitalize()} pathways: {count}")

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------
    logging.info("\n" + "=" * 60)
    logging.info("Saving outputs...")
    logging.info("=" * 60)

    # Main output: pathway DataFrame as CSV
    output_csv = os.path.join(cfg['output']['pathways'], f'pathways_base_{param_str}.csv')
    final_df.to_csv(output_csv, index=False)
    logging.info(f"Saved pathway CSV to {output_csv}")

    # Pickle with additional metadata
    output_pkl = os.path.join(cfg['output']['pathways'], f'pathways_base_{param_str}.pkl')
    output_data = {
        'pathways_df': final_df,
        'statistics': stats,
        'config': dict(cfg),
        'parameter_string': param_str
    }
    with open(output_pkl, 'wb') as f:
        pickle.dump(output_data, f)
    logging.info(f"Saved pathway pickle to {output_pkl}")

    # Save reports
    if len(redundancy_report) > 0:
        redundancy_path = os.path.join(cfg['output']['pathways'], f'redundancy_report_{param_str}.csv')
        redundancy_report.to_csv(redundancy_path, index=False)
        logging.info(f"Saved redundancy report to {redundancy_path}")

    if len(hallmark_displacement_report) > 0:
        displacement_path = os.path.join(cfg['output']['pathways'], f'hallmark_displacement_{param_str}.csv')
        hallmark_displacement_report.to_csv(displacement_path, index=False)
        logging.info(f"Saved Hallmark displacement report to {displacement_path}")

    # Save statistics as JSON for easy inspection
    import json
    stats_path = os.path.join(cfg['output']['pathways'], f'statistics_{param_str}.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logging.info(f"Saved statistics to {stats_path}")

    logging.info("\n" + "=" * 60)
    logging.info("Reactome preprocessing complete!")
    logging.info("=" * 60)

    return {
        'pathways_csv': output_csv,
        'pathways_pkl': output_pkl,
        'statistics': stats_path
    }