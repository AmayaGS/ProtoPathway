# runners/gene_expression_run.py

import os
import pandas as pd
import numpy as np

from utils.helpers import ensure_directory
from utils.expression_utils import extract_gene_annotations, load_expression_data, filter_expression_data
from utils.pathway_utils import (
    load_reactome_pathways, build_reactome_hierarchy, select_pathways_by_depth,
    filter_pathways_by_size, plot_pathway_depth_histogram,
    plot_pathway_size_histograms, plot_filtered_pathway_histogram
)
from utils.pathway_similarity_utils import (
    calculate_pathway_similarities, identify_redundant_pathways, generate_removal_report
)
from utils.hypergraph_utils import build_pathway_hypergraph, visualize_pathway_hypergraph


def gene_expression_preprocessing(config):
    """
    Run the complete gene expression preprocessing pipeline.

    This function performs the following steps:
    1. Extract gene annotations from GTF file
    2. Load and filter gene expression data
    3. Process Reactome pathways and build hierarchy
    4. Select relevant pathways and filter by size
    5. Identify and remove redundant pathways
    6. Build and visualize hypergraph representation

    Args:
        config: Configuration dictionary with pipeline parameters

    Returns:
        Tuple of (filtered_genes, final_pathway_dict)
    """
    logger = config.get('logger', None)
    if logger:
        logger.logger.info("Starting Gene Expression and Reactome Pathways preprocessing")
    else:
        print("Starting Gene Expression and Reactome Pathways preprocessing")

    # Create output directories
    ensure_directory(config['output']['data']['dir'])
    ensure_directory(config['output']['figures']['dir'])

    # Step 1: Extract gene annotations from GTF file
    annotations_df, protein_coding_genes = extract_gene_annotations(
        config['gene_expression']['input']['gtf_file'],
        config['output']['data']['gene_annotations']
    )

    # Step 2: Process gene expression data
    gene_df = load_expression_data(
        config['gene_expression']['input']['gene_expression']
    )

    filtered_gene_df = filter_expression_data(
        gene_df,
        protein_coding_genes,
        min_expression=config['gene_expression']['parameters']['threshold'],
        min_proportion=config['gene_expression']['parameters']['min_proportion'],
        variance_proportion=config['gene_expression']['parameters']['variance_proportion'],
        output_path=config['output']['data']['filtered_genes']
    )

    # Get filtered genes as a set for later use
    filtered_genes = set(filtered_gene_df.T.index)

    if logger:
        logger.logger.info(f"Final filtered gene set contains {len(filtered_genes)} genes")
    else:
        print(f"Final filtered gene set contains {len(filtered_genes)} genes")

    # Step 3: Process Reactome pathways
    pathway_dict, pathway_df = load_reactome_pathways(
        config['gene_expression']['input']['reactome_gmt'],
        config['output']['data']['reactome_pathways']
    )

    # Step 4: Build Reactome hierarchy and select pathways
    hierarchy_graph, depth_df = build_reactome_hierarchy(
        config['gene_expression']['input']['reactome_relations'],
        config['gene_expression']['input']['reactome_pathways']
    )

    # Visualize pathway depths
    plot_pathway_depth_histogram(
        depth_df,
        config['gene_expression']['pathway']['target_depth'],
        config['output']['figures']['dir']
    )

    # Select pathways based on hierarchy
    selected_pathways = select_pathways_by_depth(
        hierarchy_graph,
        depth_df,
        target_depth=config['gene_expression']['pathway']['target_depth']
    )

    # Filter pathway DataFrame to selected pathways
    selected_pathway_df = pathway_df[pathway_df['pathway_id'].isin(selected_pathways)].copy()
    selected_pathway_df['gene_count'] = selected_pathway_df['genes'].apply(len)
    selected_pathway_df = selected_pathway_df.merge(
        depth_df, how='inner', on=['pathway_name', 'pathway_id']
    )

    if logger:
        logger.logger.info(f"Selected {len(selected_pathway_df)} pathways based on hierarchy")
    else:
        print(f"Selected {len(selected_pathway_df)} pathways based on hierarchy")

    # Step 5: Visualize and filter pathways by min/max size
    plot_pathway_size_histograms(
        selected_pathway_df,
        config['gene_expression']['pathway']['min_genes'],
        config['gene_expression']['pathway']['max_genes'],
        config['output']['figures']['dir']
    )

    filtered_pathway_df = filter_pathways_by_size(
        selected_pathway_df,
        min_genes=config['gene_expression']['pathway']['min_genes'],
        max_genes=config['gene_expression']['pathway']['max_genes']
    )

    # Step 6: Analyze pathway similarity and redundancy
    # Create gene sets from filtered pathways
    pathway_gene_sets = {}
    for idx, row in filtered_pathway_df.iterrows():
        pathway_gene_sets[row['pathway_name']] = set(row['genes'])

    # Calculate similarities and find redundant pathways
    similarities, redundant_pairs = calculate_pathway_similarities(pathway_gene_sets)

    # Step 7: Identify and remove redundant pathways
    pathway_id_to_name = dict(zip(
        filtered_pathway_df['pathway_name'],
        filtered_pathway_df['pathway_id']
    ))

    removals = identify_redundant_pathways(
        redundant_pairs,
        pathway_gene_sets,
        pathway_id_to_name
    )

    generate_removal_report(
        removals,
        pathway_gene_sets,
        pathway_id_to_name,
        config['output']['reports']['removed_pathways'],
        config['output']['reports']['removed_pathways_md']
    )

    # Step 8: Create final filtered pathway set
    final_filtered_pathways = filtered_pathway_df[~filtered_pathway_df['pathway_name'].isin(
        set(removals.keys())
    )]

    plot_filtered_pathway_histogram(
        final_filtered_pathways,
        config['gene_expression']['pathway']['min_genes'],
        config['gene_expression']['pathway']['max_genes'],
        config['output']['figures']['dir']
    )

    # Save final filtered pathways
    final_filtered_pathways.to_csv(config['output']['data']['final_pathways'], index=False)

    if logger:
        logger.logger.info(
            f"Saved {len(final_filtered_pathways)} final pathways to {config['output']['data']['final_pathways']}")
    else:
        print(f"Saved {len(final_filtered_pathways)} final pathways to {config['output']['data']['final_pathways']}")

    # Step 9: Create pathway dictionary for hypergraph
    final_pathway_dict = {}
    for idx, row in final_filtered_pathways.iterrows():
        final_pathway_dict[row['pathway_name']] = row['genes']

    # Step 10: Build and visualize hypergraph
    hypergraph = build_pathway_hypergraph(filtered_genes, final_pathway_dict)

    visualize_pathway_hypergraph(
        hypergraph,
        final_pathway_dict,
        max_pathways=config['gene_expression']['pathway']['max_visualization'],
        output_path=config['output']['figures']['dir']
    )

    if logger:
        logger.logger.info("Gene Expression preprocessing pipeline completed successfully!")
    else:
        print("Gene Expression preprocessing pipeline completed successfully!")

    # Return processed data for downstream machine learning pipeline
    return filtered_genes, final_pathway_dict