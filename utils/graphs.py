"""
Bipartite graph utilities for pathway-gene networks.

Creates the graph structure used by the GATv2-based pathway embedding model.
"""

import logging
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


def build_bipartite_graph(gene_names, pathway_genes):
    """
    Build bipartite graph connecting genes to pathways.

    Graph structure:
    - Nodes 0 to num_genes-1: gene nodes
    - Nodes num_genes to num_genes+num_pathways-1: pathway nodes
    - Edges connect genes to pathways they belong to

    Args:
        gene_names: List of gene names (defines node ordering)
        pathway_genes: dict {pathway_name: list of genes}

    Returns:
        dict containing:
            - edge_index: torch.LongTensor [2, num_edges]
            - num_genes: int
            - num_pathways: int
            - gene_names: list
            - pathway_names: list
            - gene_to_idx: dict {gene_name: node_index}
            - pathway_to_idx: dict {pathway_name: node_index}
            - genes_per_pathway: dict {pathway_name: list of gene indices}
    """
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    num_genes = len(gene_names)

    pathway_names = list(pathway_genes.keys())
    pathway_to_idx = {p: i + num_genes for i, p in enumerate(pathway_names)}
    num_pathways = len(pathway_names)

    # Build edges
    edge_list = []
    genes_per_pathway = {}

    for pathway_name, genes in pathway_genes.items():
        pathway_idx = pathway_to_idx[pathway_name]
        pathway_gene_indices = []

        for gene in genes:
            if gene in gene_to_idx:
                gene_idx = gene_to_idx[gene]
                # Bidirectional edges for message passing
                edge_list.append([gene_idx, pathway_idx])  # gene -> pathway
                edge_list.append([pathway_idx, gene_idx])  # pathway -> gene
                pathway_gene_indices.append(gene_idx)

        genes_per_pathway[pathway_name] = pathway_gene_indices

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    logging.info(f"Built bipartite graph: {num_genes} genes, {num_pathways} pathways, "
                 f"{edge_index.shape[1]} edges")

    return {
        'edge_index': edge_index,
        'num_genes': num_genes,
        'num_pathways': num_pathways,
        'gene_names': gene_names,
        'pathway_names': pathway_names,
        'gene_to_idx': gene_to_idx,
        'pathway_to_idx': pathway_to_idx,
        'genes_per_pathway': genes_per_pathway
    }


def filter_pathways_by_gene_coverage(pathway_df, available_genes, min_genes=3):
    """
    Filter pathways to those with sufficient genes in the expression data.

    Args:
        pathway_df: DataFrame with pathway_id and genes columns
        available_genes: Set of gene names in expression data
        min_genes: Minimum genes from pathway that must be present

    Returns:
        Filtered DataFrame with additional 'genes_present' and 'coverage' columns
    """
    pathway_df = pathway_df.copy()
    available_set = set(available_genes)

    def count_present(genes):
        if isinstance(genes, str):
            genes = eval(genes)
        return len(set(genes) & available_set)

    def get_present_genes(genes):
        if isinstance(genes, str):
            genes = eval(genes)
        return list(set(genes) & available_set)

    pathway_df['genes_present_count'] = pathway_df['genes'].apply(count_present)
    pathway_df['genes_present'] = pathway_df['genes'].apply(get_present_genes)
    pathway_df['coverage'] = pathway_df['genes_present_count'] / pathway_df['gene_count']

    before = len(pathway_df)
    filtered = pathway_df[pathway_df['genes_present_count'] >= min_genes].copy()

    logging.info(f"Gene coverage filter (≥{min_genes} genes present): {before} → {len(filtered)} pathways")
    logging.info(f"  Mean coverage: {filtered['coverage'].mean():.1%}")
    logging.info(f"  Median genes present: {filtered['genes_present_count'].median():.0f}")

    return filtered


def compute_graph_statistics(graph_data):
    """
    Compute statistics about the bipartite graph.

    Returns:
        dict with statistics
    """
    edge_index = graph_data['edge_index']
    num_genes = graph_data['num_genes']
    num_pathways = graph_data['num_pathways']

    # Count edges per gene and per pathway
    gene_degrees = torch.zeros(num_genes, dtype=torch.long)
    pathway_degrees = torch.zeros(num_pathways, dtype=torch.long)

    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        if src < num_genes:
            gene_degrees[src] += 1
        else:
            pathway_degrees[src - num_genes] += 1

    # Each edge is bidirectional, so divide by 2 for actual connections
    gene_degrees = gene_degrees // 2
    pathway_degrees = pathway_degrees // 2

    stats = {
        'num_genes': num_genes,
        'num_pathways': num_pathways,
        'num_edges': edge_index.shape[1] // 2,  # Bidirectional
        'mean_pathways_per_gene': gene_degrees.float().mean().item(),
        'median_pathways_per_gene': gene_degrees.float().median().item(),
        'max_pathways_per_gene': gene_degrees.max().item(),
        'genes_in_single_pathway': (gene_degrees == 1).sum().item(),
        'mean_genes_per_pathway': pathway_degrees.float().mean().item(),
        'median_genes_per_pathway': pathway_degrees.float().median().item(),
        'max_genes_per_pathway': pathway_degrees.max().item(),
    }

    return stats


def truncate_label(label, max_len=40):
    """Truncate long labels for visualization."""
    return label if len(label) <= max_len else label[:max_len] + '...'


def visualize_bipartite_graph(graph_data, pathway_df=None, max_pathways=30,
                              max_genes_per_pathway=10, output_path=None,
                              figsize=(16, 12), seed=42):
    """
    Visualize the bipartite graph with pathways as diamonds and genes as circles.

    Uses a bipartite layout with pathways on one side and genes on the other.

    Args:
        graph_data: Dict from build_bipartite_graph()
        pathway_df: Optional DataFrame with pathway info (for coloring by source)
        max_pathways: Maximum pathways to show
        max_genes_per_pathway: Max genes to show per pathway (for readability)
        output_path: Path to save figure (if None, displays instead)
        figsize: Figure size tuple
        seed: Random seed for layout

    Returns:
        matplotlib Figure object
    """
    logging.info(f"Visualizing bipartite graph (max {max_pathways} pathways)...")

    # Build networkx graph for visualization
    G = nx.Graph()

    gene_names = graph_data['gene_names']
    pathway_names = graph_data['pathway_names']
    genes_per_pathway = graph_data['genes_per_pathway']

    # Select subset of pathways
    selected_pathways = pathway_names[:max_pathways]

    # Collect genes from selected pathways (limit per pathway for readability)
    selected_genes = set()
    pathway_gene_map = {}

    for pathway in selected_pathways:
        gene_indices = genes_per_pathway[pathway]
        # Limit genes per pathway
        limited_indices = gene_indices[:max_genes_per_pathway]
        genes = [gene_names[i] for i in limited_indices]
        pathway_gene_map[pathway] = genes
        selected_genes.update(genes)

    selected_genes = list(selected_genes)

    # Add nodes
    for gene in selected_genes:
        G.add_node(gene, bipartite=0, node_type='gene')

    for pathway in selected_pathways:
        G.add_node(pathway, bipartite=1, node_type='pathway')

    # Add edges
    for pathway, genes in pathway_gene_map.items():
        for gene in genes:
            G.add_edge(pathway, gene)

    # Create bipartite layout
    pos = {}

    # # Pathways on the left
    # for i, pathway in enumerate(selected_pathways):
    #     pos[pathway] = (-1, 1 - 2 * i / max(len(selected_pathways) - 1, 1))

    # Genes on the right - use spring layout for gene positions
    gene_subgraph = G.subgraph(selected_genes)
    gene_pos = nx.spring_layout(gene_subgraph, seed=seed)

    # Scale and shift gene positions to the left side
    for gene in selected_genes:
        x, y = gene_pos[gene]
        pos[gene] = (-1 + x * 0.5, y)

    # Pathways on the right
    for i, pathway in enumerate(selected_pathways):
        pos[pathway] = (1, 1 - 2 * i / max(len(selected_pathways) - 1, 1))

    # # Scale and shift gene positions to the right side
    # for gene in selected_genes:
    #     x, y = gene_pos[gene]
    #     pos[gene] = (1 + x * 0.5, y)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Determine pathway colors (by source if available)
    pathway_colors = {}
    if pathway_df is not None and 'source' in pathway_df.columns:
        source_colors = {'reactome': '#2ecc71', 'hallmark': '#3498db'}
        pathway_sources = dict(zip(pathway_df['pathway_id'], pathway_df['source']))
        for pathway in selected_pathways:
            source = pathway_sources.get(pathway, 'reactome')
            pathway_colors[pathway] = source_colors.get(source, '#95a5a6')
    else:
        # Use colormap
        cmap = plt.get_cmap('tab20')
        for i, pathway in enumerate(selected_pathways):
            pathway_colors[pathway] = cmap(i % 20)

    # Draw edges first (so nodes are on top)
    for pathway, genes in pathway_gene_map.items():
        color = pathway_colors[pathway]
        for gene in genes:
            ax.plot([pos[pathway][0], pos[gene][0]],
                    [pos[pathway][1], pos[gene][1]],
                    color=color, alpha=0.3, linewidth=0.5, zorder=1)

    # Draw gene nodes (circles)
    gene_x = [pos[g][0] for g in selected_genes]
    gene_y = [pos[g][1] for g in selected_genes]
    ax.scatter(gene_x, gene_y, c='lightgray', s=30, marker='o',
               edgecolors='gray', linewidths=0.5, zorder=2, label='Genes')

    # Draw pathway nodes (diamonds)
    for pathway in selected_pathways:
        color = pathway_colors[pathway]
        ax.scatter(pos[pathway][0], pos[pathway][1], c=[color], s=200,
                   marker='D', edgecolors='black', linewidths=1, zorder=3)

    # Add pathway labels
    for pathway in selected_pathways:
        ax.annotate(truncate_label(pathway, 35),
                    xy=(pos[pathway][0] + 0.05, pos[pathway][1]),
                    ha='left', va='center', fontsize=7,
                    fontweight='bold')

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray',
               markersize=8, markeredgecolor='gray', label=f'Genes (n={len(selected_genes)})'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#2ecc71',
               markersize=10, markeredgecolor='black', label='Reactome Pathways'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#3498db',
               markersize=10, markeredgecolor='black', label='Hallmark Gene Sets'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    ax.set_xlim(-2, 2)
    ax.set_title(f'Bipartite Graph: {len(selected_pathways)} Pathways × {len(selected_genes)} Genes',
                 fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        logging.info(f"Saved bipartite graph visualization to {output_path}")

    return fig


def visualize_hypergraph(graph_data, pathway_df=None, max_pathways=20,
                         output_path=None, figsize=(14, 14), seed=42):
    """
    Visualize as a hypergraph with ellipses around pathway gene sets.

    Pathways are shown as diamond nodes, genes as circles, with ellipses
    grouping genes that belong to the same pathway.

    Args:
        graph_data: Dict from build_bipartite_graph()
        pathway_df: Optional DataFrame with pathway info
        max_pathways: Maximum pathways to visualize
        output_path: Path to save figure
        figsize: Figure size tuple
        seed: Random seed for layout

    Returns:
        matplotlib Figure object
    """
    logging.info(f"Visualizing hypergraph (max {max_pathways} pathways)...")

    gene_names = graph_data['gene_names']
    pathway_names = graph_data['pathway_names']
    genes_per_pathway = graph_data['genes_per_pathway']

    # Select subset of pathways
    selected_pathways = pathway_names[:max_pathways]

    # Build graph
    G = nx.Graph()

    # Collect all genes from selected pathways
    all_genes = set()
    pathway_gene_map = {}

    for pathway in selected_pathways:
        gene_indices = genes_per_pathway[pathway]
        genes = [gene_names[i] for i in gene_indices]
        pathway_gene_map[pathway] = genes
        all_genes.update(genes)

        # Add pathway node
        G.add_node(pathway, node_type='pathway')

        # Add gene nodes and edges
        for gene in genes:
            G.add_node(gene, node_type='gene')
            G.add_edge(pathway, gene)

    # Generate layout
    pos = nx.spring_layout(G, seed=seed, k=2 / np.sqrt(len(G.nodes)))

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create colormap for pathways
    cmap = plt.get_cmap('tab20')
    pathway_colors = {p: cmap(i % 20) for i, p in enumerate(selected_pathways)}

    # Draw ellipses around pathway gene sets
    for pathway in selected_pathways:
        genes = pathway_gene_map[pathway]
        genes_in_graph = [g for g in genes if g in pos]

        if len(genes_in_graph) >= 2:
            # Get gene positions
            gene_positions = np.array([pos[g] for g in genes_in_graph])

            # Compute bounding ellipse
            center = gene_positions.mean(axis=0)

            if len(genes_in_graph) > 2:
                # Use covariance for ellipse shape
                cov = np.cov(gene_positions.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)

                # Ellipse dimensions (2 std devs)
                width = 4 * np.sqrt(max(eigenvalues[0], 0.01))
                height = 4 * np.sqrt(max(eigenvalues[1], 0.01))

                # Rotation angle
                angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
            else:
                # For 2 genes, simple ellipse
                width = np.abs(gene_positions[0, 0] - gene_positions[1, 0]) + 0.2
                height = np.abs(gene_positions[0, 1] - gene_positions[1, 1]) + 0.2
                angle = 0

            # Add ellipse
            ellipse = mpatches.Ellipse(
                center, width, height, angle=angle,
                facecolor=pathway_colors[pathway],
                edgecolor=pathway_colors[pathway],
                alpha=0.15, linewidth=2, zorder=1
            )
            ax.add_patch(ellipse)

    # Draw edges
    for u, v in G.edges():
        if u in selected_pathways:
            color = pathway_colors[u]
        elif v in selected_pathways:
            color = pathway_colors[v]
        else:
            color = 'gray'

        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=color, alpha=0.2, linewidth=0.5, zorder=2)

    # Draw gene nodes (small circles)
    gene_nodes = [n for n in G.nodes if G.nodes[n].get('node_type') == 'gene']
    gene_x = [pos[n][0] for n in gene_nodes]
    gene_y = [pos[n][1] for n in gene_nodes]

    # Color genes by their first pathway
    gene_colors = []
    for gene in gene_nodes:
        for pathway in selected_pathways:
            if gene in pathway_gene_map[pathway]:
                gene_colors.append(pathway_colors[pathway])
                break
        else:
            gene_colors.append('lightgray')

    ax.scatter(gene_x, gene_y, c=gene_colors, s=20, marker='o',
               edgecolors='gray', linewidths=0.3, zorder=3)

    # Draw pathway nodes (larger diamonds)
    for pathway in selected_pathways:
        if pathway in pos:
            ax.scatter(pos[pathway][0], pos[pathway][1],
                       c=[pathway_colors[pathway]], s=300, marker='D',
                       edgecolors='black', linewidths=0.5, zorder=4)

    ax.set_title(f'Hypergraph: {len(selected_pathways)} Pathways, {len(gene_nodes)} Genes',
                 fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        logging.info(f"Saved hypergraph visualization to {output_path}")

    # Save legend separately
    if output_path:
        fig_legend, ax_legend = plt.subplots(figsize=(10, max_pathways * 0.4))

        handles = [mpatches.Patch(color=pathway_colors[p],
                                  label=truncate_label(p, 50))
                   for p in selected_pathways]

        ax_legend.legend(handles=handles, loc='center', ncol=1, fontsize=9)
        ax_legend.axis('off')

        legend_path = output_path.replace('.pdf', '_legend.pdf').replace('.png', '_legend.png')
        fig_legend.savefig(legend_path, bbox_inches='tight', dpi=150)
        plt.close(fig_legend)
        logging.info(f"Saved hypergraph legend to {legend_path}")

    plt.close(fig)

    return fig