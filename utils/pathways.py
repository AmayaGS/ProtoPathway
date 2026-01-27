"""
Pathway utilities for Reactome and MSigDB processing.

Consolidates functionality for:
- GMT file parsing (Reactome and MSigDB Hallmarks)
- Reactome hierarchy construction and depth computation
- Category-based filtering
- Size-based filtering
- Jaccard similarity and redundancy removal
"""

import os
import logging
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter


def parse_gmt(file_path):
    """
    Parse a GMT file (works for both Reactome and MSigDB).

    GMT format: pathway_id<TAB>description<TAB>gene1<TAB>gene2<TAB>...

    Returns:
        dict: {pathway_name: list of genes}
        DataFrame: columns [pathway_id, pathway_name, genes]
    """
    pathway_dict = {}
    records = []

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue

            pathway_id = parts[1]
            pathway_name = parts[0]
            genes = parts[2:]

            pathway_dict[pathway_id] = genes
            records.append({
                'pathway_id': pathway_id,
                'pathway_name': pathway_name,
                'genes': genes,
                'gene_count': len(genes)
            })

    df = pd.DataFrame(records)
    logging.info(f"Loaded {len(df)} pathways from {file_path}")

    return pathway_dict, df


def build_reactome_hierarchy(relations_path, pathways_path):
    """
    Build Reactome pathway hierarchy as a directed graph.

    Args:
        relations_path: Path to ReactomePathwaysRelation.txt
        pathways_path: Path to ReactomePathways.txt (for names)

    Returns:
        G: NetworkX DiGraph of pathway hierarchy
        pathway_names_df: DataFrame with pathway_id, pathway_name, species
    """
    logging.info("Building Reactome hierarchy...")

    # Load hierarchy relationships
    relations = pd.read_csv(
        relations_path,
        sep="\t",
        header=None,
        names=["parent", "child"]
    )

    # Filter for human-only (R-HSA) pathways
    relations = relations[
        relations["parent"].str.startswith("R-HSA") &
        relations["child"].str.startswith("R-HSA")
    ]

    # Load pathway names
    pathway_names_df = pd.read_csv(
        pathways_path,
        sep="\t",
        header=None,
        names=["pathway_id", "pathway_name", "species"]
    )
    pathway_names_df = pathway_names_df[pathway_names_df['species'] == 'Homo sapiens']

    # Build directed graph
    G = nx.DiGraph()
    G.add_edges_from(relations.itertuples(index=False, name=None))

    logging.info(f"Hierarchy has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    return G, pathway_names_df


def compute_pathway_depths(G):
    """
    Compute depth of each pathway from root nodes.

    Args:
        G: NetworkX DiGraph of pathway hierarchy

    Returns:
        dict: {pathway_id: depth}
    """
    # Find root nodes (no incoming edges)
    roots = [n for n in G.nodes if G.in_degree(n) == 0]
    logging.info(f"Found {len(roots)} root categories")

    # Compute shortest path from any root (minimum depth)
    depths = {}
    for root in roots:
        lengths = nx.single_source_shortest_path_length(G, root)
        for node, depth in lengths.items():
            if node not in depths or depth < depths[node]:
                depths[node] = depth

    return depths


def get_pathway_categories(G, pathway_names_df):
    """
    Map each pathway to its top-level (root) category.

    Args:
        G: NetworkX DiGraph of pathway hierarchy
        pathway_names_df: DataFrame with pathway names

    Returns:
        dict: {pathway_id: top_level_category_name}
    """
    # Find roots
    roots = [n for n in G.nodes if G.in_degree(n) == 0]

    # Create ID to name mapping
    id_to_name = dict(zip(pathway_names_df['pathway_id'], pathway_names_df['pathway_name']))

    # For each node, find which root it descends from
    pathway_categories = {}

    for node in G.nodes:
        # Find all roots that can reach this node
        for root in roots:
            if nx.has_path(G, root, node) or node == root:
                pathway_categories[node] = id_to_name.get(root, root)
                break

    return pathway_categories


def filter_by_depth(pathway_df, hierarchy_graph, depths, target_depth):
    """
    Select pathways at target_depth, or leaf nodes if branch is shallower.
    """
    # Find leaf nodes (no children in hierarchy)
    leaves = {n for n in hierarchy_graph.nodes() if hierarchy_graph.out_degree(n) == 0}

    selected = []
    for pathway_id in pathway_df['pathway_id']:
        depth = depths.get(pathway_id)
        if depth is None:
            continue

        if depth == target_depth:
            # At target depth - keep
            selected.append(pathway_id)
        elif depth < target_depth and pathway_id in leaves:
            # Shallower but it's a leaf (branch doesn't go deeper) - keep
            selected.append(pathway_id)
        # depth > target_depth: skip (too granular)
        # depth < target_depth and not leaf: skip (has more specific children)

    return pathway_df[pathway_df['pathway_id'].isin(selected)]


def filter_by_category(pathway_df, pathway_categories, exclude_categories):
    """
    Remove pathways belonging to excluded top-level categories.

    Args:
        pathway_df: DataFrame with pathway_id column
        pathway_categories: dict {pathway_id: category_name}
        exclude_categories: list of category names to exclude

    Returns:
        Filtered DataFrame
    """
    pathway_df = pathway_df.copy()
    pathway_df['category'] = pathway_df['pathway_id'].map(pathway_categories)

    before = len(pathway_df)
    exclude_set = set(exclude_categories)
    filtered = pathway_df[~pathway_df['category'].isin(exclude_set)].copy()

    # Log what was removed per category
    removed = pathway_df[pathway_df['category'].isin(exclude_set)]
    for cat in exclude_categories:
        n_removed = len(removed[removed['category'] == cat])
        if n_removed > 0:
            logging.info(f"  Excluded '{cat}': {n_removed} pathways")

    logging.info(f"Category filter: {before} → {len(filtered)} pathways")

    return filtered


def filter_by_size(pathway_df, min_genes, max_genes):
    """
    Filter pathways by gene count.

    Args:
        pathway_df: DataFrame with gene_count column
        min_genes: Minimum genes per pathway
        max_genes: Maximum genes per pathway

    Returns:
        Filtered DataFrame
    """
    before = len(pathway_df)
    filtered = pathway_df[
        (pathway_df['gene_count'] >= min_genes) &
        (pathway_df['gene_count'] <= max_genes)
    ].copy()

    logging.info(f"Size filter ({min_genes}-{max_genes} genes): {before} → {len(filtered)} pathways")

    return filtered


def jaccard_similarity(set1, set2):
    """Compute Jaccard similarity between two sets."""
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0


def calculate_pairwise_similarities(pathway_gene_dict):
    """
    Calculate pairwise Jaccard similarities between all pathways.

    Args:
        pathway_gene_dict: dict {pathway_name: set of genes}

    Returns:
        similarities: list of all pairwise similarities
        redundant_pairs: list of (p1, p2, similarity, size1, size2) for high-similarity pairs
    """
    pathways = list(pathway_gene_dict.keys())
    n = len(pathways)
    similarities = []

    logging.info(f"Computing pairwise similarities for {n} pathways...")

    for i in range(n):
        if i % 100 == 0 and i > 0:
            logging.info(f"  Processed {i}/{n} pathways...")

        for j in range(i + 1, n):
            sim = jaccard_similarity(
                pathway_gene_dict[pathways[i]],
                pathway_gene_dict[pathways[j]]
            )
            similarities.append((pathways[i], pathways[j], sim))

    return similarities


def remove_redundant_pathways(pathway_df, jaccard_threshold=0.8, hierarchy_graph=None, depths=None):
    """
    Remove redundant pathways based on Jaccard similarity.

    When two pathways have similarity >= threshold, selection priority:
    1. Prefer leaf nodes (no children in hierarchy)
    2. Prefer deeper pathways (more specific)
    3. Prefer larger gene sets
    4. Alphabetically first (tie-breaker)

    Args:
        pathway_df: DataFrame with pathway_id, pathway_name, genes columns
        jaccard_threshold: Similarity threshold for redundancy
        hierarchy_graph: NetworkX DiGraph of Reactome hierarchy (optional)
        depths: dict {pathway_id: depth} (optional)

    Returns:
        filtered_df: DataFrame with redundant pathways removed
        removal_report: DataFrame documenting what was removed and why
    """
    # Build gene sets and ID-to-name mapping
    pathway_gene_sets = {}
    id_to_name = dict(zip(pathway_df['pathway_id'], pathway_df['pathway_name']))
    for _, row in pathway_df.iterrows():
        genes = row['genes'] if isinstance(row['genes'], list) else eval(row['genes'])
        pathway_gene_sets[row['pathway_id']] = set(genes)

    # Identify leaf nodes if hierarchy provided
    if hierarchy_graph is not None:
        leaves = {n for n in hierarchy_graph.nodes() if hierarchy_graph.out_degree(n) == 0}
    else:
        leaves = set()

    # Default depths if not provided
    if depths is None:
        depths = {}

    def pathway_sort_key(p):
        """
        Sort key for selecting which pathway to keep.
        Returns tuple for sorting (lower = better to keep):
        - is_leaf: 0 if leaf, 1 if not (prefer leaves)
        - neg_depth: -depth (prefer deeper/more specific)
        - neg_size: -gene_count (prefer larger)
        - name: alphabetical tie-breaker
        """
        is_leaf = 0 if p in leaves else 1
        depth = depths.get(p, 0)
        size = len(pathway_gene_sets[p])
        return (is_leaf, -depth, -size, p)

    # Calculate similarities
    similarities = calculate_pairwise_similarities(pathway_gene_sets)

    # Find redundant pairs
    redundant_pairs = [(p1, p2, sim) for p1, p2, sim in similarities if sim >= jaccard_threshold]
    logging.info(f"Found {len(redundant_pairs)} pathway pairs with Jaccard ≥ {jaccard_threshold}")

    if not redundant_pairs:
        return pathway_df, pd.DataFrame()

    # Build graph of redundant relationships
    G = nx.Graph()
    for p1, p2, sim in redundant_pairs:
        G.add_edge(p1, p2, similarity=sim)

    # For each connected component, keep one representative
    removals = []
    for component in nx.connected_components(G):
        component = list(component)
        if len(component) == 1:
            continue

        # Sort by leaf status, depth, size, then alphabetically
        component_sorted = sorted(component, key=pathway_sort_key)

        kept = component_sorted[0]
        for removed in component_sorted[1:]:
            sim = G[kept][removed]['similarity'] if G.has_edge(kept, removed) else jaccard_threshold
            removals.append({
                'removed_pathway_id': removed,
                'removed_pathway_name': id_to_name.get(removed, removed),
                'kept_pathway_id': kept,
                'kept_pathway_name': id_to_name.get(kept, kept),
                'similarity': sim,
                'removed_size': len(pathway_gene_sets[removed]),
                'kept_size': len(pathway_gene_sets[kept]),
                'kept_is_leaf': kept in leaves,
                'kept_depth': depths.get(kept, None)
            })

    removal_report = pd.DataFrame(removals)
    removed_ids = set(removal_report['removed_pathway_id']) if len(removal_report) > 0 else set()

    filtered_df = pathway_df[~pathway_df['pathway_id'].isin(removed_ids)].copy()

    logging.info(f"Redundancy removal: {len(pathway_df)} → {len(filtered_df)} pathways ({len(removed_ids)} removed)")

    return filtered_df, removal_report

# def remove_redundant_pathways(pathway_df, jaccard_threshold=0.8):
#     """
#     Remove redundant pathways based on Jaccard similarity.
#
#     When two pathways have similarity >= threshold, keep the larger one
#     (or alphabetically first if same size).
#
#     Args:
#         pathway_df: DataFrame with pathway_id, pathway_name, genes columns
#         jaccard_threshold: Similarity threshold for redundancy
#
#     Returns:
#         filtered_df: DataFrame with redundant pathways removed
#         removal_report: DataFrame documenting what was removed and why
#     """
#     # Build gene sets
#     pathway_gene_sets = {}
#     for _, row in pathway_df.iterrows():
#         genes = row['genes'] if isinstance(row['genes'], list) else eval(row['genes'])
#         pathway_gene_sets[row['pathway_id']] = set(genes)
#
#     # Calculate similarities
#     similarities = calculate_pairwise_similarities(pathway_gene_sets)
#
#     # Find redundant pairs
#     redundant_pairs = [(p1, p2, sim) for p1, p2, sim in similarities if sim >= jaccard_threshold]
#     logging.info(f"Found {len(redundant_pairs)} pathway pairs with Jaccard ≥ {jaccard_threshold}")
#
#     if not redundant_pairs:
#         return pathway_df, pd.DataFrame()
#
#     # Build graph of redundant relationships
#     G = nx.Graph()
#     for p1, p2, sim in redundant_pairs:
#         G.add_edge(p1, p2, similarity=sim)
#
#     reactome_id_to_name = dict(zip(pathway_df['pathway_id'], pathway_df['pathway_name']))
#
#     # For each connected component, keep one representative
#     removals = []
#     for component in nx.connected_components(G):
#         component = list(component)
#         if len(component) == 1:
#             continue
#
#         # Sort by size (descending), then alphabetically
#         component_sorted = sorted(
#             component,
#             key=lambda p: (-len(pathway_gene_sets[p]), p)
#         )
#
#         kept = component_sorted[0]
#         for removed in component_sorted[1:]:
#             sim = G[kept][removed]['similarity'] if G.has_edge(kept, removed) else jaccard_threshold
#             removals.append({
#                 'removed_pathway': removed,
#                 'removed_name': reactome_id_to_name.get(removed, removed),
#                 'kept_pathway': kept,
#                 'kept_name': reactome_id_to_name.get(kept, kept),
#                 'similarity': sim,
#                 'removed_size': len(pathway_gene_sets[removed]),
#                 'kept_size': len(pathway_gene_sets[kept])
#             })
#
#     removal_report = pd.DataFrame(removals)
#     removed_ids = set(removal_report['removed_pathway']) if len(removal_report) > 0 else set()
#
#     filtered_df = pathway_df[~pathway_df['pathway_id'].isin(removed_ids)].copy()
#
#     logging.info(f"Redundancy removal: {len(pathway_df)} → {len(filtered_df)} pathways ({len(removed_ids)} removed)")
#
#     return filtered_df, removal_report


def load_hallmark_csv(csv_path, selected_hallmarks=None):
    """
    Load MSigDB Hallmark gene sets from CSV format.

    Expected CSV format:
        - Column headers are Hallmark names (e.g., HALLMARK_HYPOXIA)
        - Rows contain gene names (variable length per column)

    Args:
        csv_path: Path to Hallmark CSV file
        selected_hallmarks: List of Hallmark names to include (None = all)

    Returns:
        DataFrame with pathway_id, pathway_name, genes, gene_count, source columns
    """
    # Read CSV
    raw_df = pd.read_csv(csv_path)

    records = []
    for col in raw_df.columns:
        # Get non-null genes for this Hallmark
        genes = raw_df[col].dropna().tolist()
        genes = [g for g in genes if isinstance(g, str) and g.strip()]

        records.append({
            'pathway_id': col,
            'pathway_name': col,
            'genes': genes,
            'gene_count': len(genes)
        })

    df = pd.DataFrame(records)
    logging.info(f"Loaded {len(df)} Hallmark gene sets from {csv_path}")

    if selected_hallmarks:
        df = df[df['pathway_id'].isin(selected_hallmarks)].copy()

        # Check for missing hallmarks
        found = set(df['pathway_id'])
        missing = set(selected_hallmarks) - found
        if missing:
            logging.warning(f"Hallmarks not found in CSV: {missing}")

        logging.info(f"Selected {len(df)} Hallmark gene sets")

    # Add source column to distinguish from Reactome
    df['source'] = 'hallmark'

    return df


def integrate_hallmarks(reactome_df, hallmark_df, jaccard_threshold=0.8):
    """
    Integrate Hallmark gene sets with Reactome, removing high-overlap Reactome pathways.

    Args:
        reactome_df: Filtered Reactome pathways DataFrame
        hallmark_df: Selected Hallmark gene sets DataFrame
        jaccard_threshold: If a Reactome pathway has Jaccard >= threshold with a
                          Hallmark set, remove the Reactome pathway

    Returns:
        combined_df: DataFrame with Reactome + Hallmarks
        displaced_report: DataFrame of Reactome pathways removed due to Hallmark overlap
    """
    reactome_df = reactome_df.copy()
    reactome_df['source'] = 'reactome'

    # Build gene sets
    reactome_genes = {}
    for _, row in reactome_df.iterrows():
        genes = row['genes'] if isinstance(row['genes'], list) else eval(row['genes'])
        reactome_genes[row['pathway_id']] = set(genes)

    hallmark_genes = {}
    for _, row in hallmark_df.iterrows():
        genes = row['genes'] if isinstance(row['genes'], list) else eval(row['genes'])
        hallmark_genes[row['pathway_id']] = set(genes)

    # Create ID to name mapping for Reactome
    reactome_id_to_name = dict(zip(reactome_df['pathway_id'], reactome_df['pathway_name']))

    # Find Reactome pathways that overlap heavily with Hallmarks
    displaced = []
    for r_id, r_genes in reactome_genes.items():
        for h_id, h_genes in hallmark_genes.items():
            sim = jaccard_similarity(r_genes, h_genes)
            if sim >= jaccard_threshold:
                displaced.append({
                    'reactome_pathway': r_id,
                    'reactome_pathway_name': reactome_id_to_name.get(r_id, r_id),
                    'hallmark_pathway': h_id,
                    'similarity': sim
                })

    displaced_report = pd.DataFrame(displaced)
    displaced_ids = set(displaced_report['reactome_pathway_id']) if len(displaced_report) > 0 else set()

    if displaced_ids:
        logging.info(f"Removing {len(displaced_ids)} Reactome pathways due to Hallmark overlap")
        reactome_df = reactome_df[~reactome_df['pathway_id'].isin(displaced_ids)]

    # Combine
    combined_df = pd.concat([reactome_df, hallmark_df], ignore_index=True)
    logging.info(f"Combined pathway set: {len(reactome_df)} Reactome + {len(hallmark_df)} Hallmarks = {len(combined_df)}")

    return combined_df, displaced_report


def compute_gene_pathway_statistics(pathway_df):
    """
    Compute summary statistics about gene coverage.

    Returns:
        dict with statistics
    """
    all_genes = set()
    gene_counts = Counter()

    for _, row in pathway_df.iterrows():
        genes = row['genes'] if isinstance(row['genes'], list) else eval(row['genes'])
        gene_set = set(genes)
        all_genes.update(gene_set)
        gene_counts.update(gene_set)

    # Gene specificity (how many pathways each gene appears in)
    genes_in_1_pathway = sum(1 for g, c in gene_counts.items() if c == 1)
    genes_in_many = sum(1 for g, c in gene_counts.items() if c > 10)

    stats = {
        'total_pathways': len(pathway_df),
        'total_unique_genes': len(all_genes),
        'mean_genes_per_pathway': pathway_df['gene_count'].mean(),
        'median_genes_per_pathway': pathway_df['gene_count'].median(),
        'genes_in_single_pathway': genes_in_1_pathway,
        'genes_in_10plus_pathways': genes_in_many,
        'mean_pathways_per_gene': np.mean(list(gene_counts.values()))
    }

    return stats


def generate_parameter_string(cfg):
    """
    Generate a parameter string for output filename.

    Format: d{depth}_g{min}-{max}_r{redundancy_threshold}
    """
    depth = cfg['filtering']['target_depth']
    min_g = cfg['filtering']['min_genes']
    max_g = cfg['filtering']['max_genes']
    redundancy = cfg['filtering']['redundancy']['jaccard_threshold']

    return f"d{depth}_g{min_g}-{max_g}_j{int(redundancy*100)}"