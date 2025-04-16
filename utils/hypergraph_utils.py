"""
Module for building and visualizing hypergraph representations of pathway data.
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def build_pathway_hypergraph(filtered_genes, pathway_dict):

    print("Building pathway hypergraph...")

    # Initialize the graph
    G = nx.Graph()

    # Add nodes (genes) to the graph
    for gene in filtered_genes:
        G.add_node(gene)

    # Add hyperedges (pathways) to the graph
    for pathway, genes in pathway_dict.items():
        # Add a hyperedge as a set of nodes (genes) in the graph
        G.add_node(pathway)  # Pathway itself as a node
        for gene in genes:
            if gene in filtered_genes:
                G.add_edge(pathway, gene)

    return G


def truncate(label, max_len=40):

    return label if len(label) <= max_len else label[:max_len] + '...'


def visualize_pathway_hypergraph(G, pathway_dict, max_pathways, output_path):

    print(f"Visualizing pathway hypergraph (max {max_pathways} pathways)...")

    sample_pathways = list(pathway_dict.keys())[:max_pathways]

    # Create subgraph with just these sample pathways and genes
    subgraph_nodes = set()
    for pathway in sample_pathways:
        genes_in_pathway = set(pathway_dict[pathway])
        subgraph_nodes.update(genes_in_pathway)
        subgraph_nodes.add(pathway)  # Add pathway node itself

    # Create the subgraph using the filtered nodes
    subgraph = G.subgraph(subgraph_nodes)

    # Generate positions for the nodes
    pos = nx.spring_layout(subgraph, seed=42)

    # Set up the figure
    plt.figure(figsize=(12, 12))

    # Create a colormap for pathways
    pathway_colors = plt.get_cmap("nipy_spectral", len(sample_pathways))

    # Create a dictionary mapping pathways to colors
    pathway_to_color = {pathway: pathway_colors(i)
                        for i, pathway in enumerate(sample_pathways)}

    # Assign node colors based on pathway they belong to
    node_colors = []
    for node in subgraph.nodes:
        if node in pathway_to_color:  # If it's a pathway node, assign its color
            node_colors.append(pathway_to_color[node])
        else:  # If it's a gene, assign the color of the first pathway it belongs to
            pathway = next((p for p in sample_pathways if node in pathway_dict[p]), None)
            if pathway:
                node_colors.append(pathway_to_color[pathway])
            else:
                node_colors.append('skyblue')  # Default color for genes with no pathway

    # Draw the nodes with their respective colors
    nx.draw(subgraph, pos, with_labels=False, node_size=20,
            node_color=node_colors, font_size=12, font_weight='bold',
            edge_color='gray')

    # Loop through each pathway to add a colored shape (ellipse) around the genes
    for pathway in sample_pathways:
        genes = pathway_dict[pathway]
        genes_in_graph = [gene for gene in genes if gene in subgraph_nodes]

        if genes_in_graph:  # Ensure there are genes in the pathway in the subgraph
            # Get the positions of the genes in this pathway
            pathway_positions = [pos[gene] for gene in genes_in_graph
                                 if gene in pos]

            if pathway_positions:  # Ensure there are genes with positions
                # Extract x and y coordinates
                x_values = [p[0] for p in pathway_positions]
                y_values = [p[1] for p in pathway_positions]

                # Define the bounding box (using min/max values)
                min_x, max_x = min(x_values), max(x_values)
                min_y, max_y = min(y_values), max(y_values)

                # Adjust the ellipse to be tighter around the genes
                padding = 0.05  # Padding for visual clarity
                width = max_x - min_x + padding
                height = max_y - min_y + padding

                # Add ellipse around genes in the pathway
                ellipse = mpatches.Ellipse(
                    ((min_x + max_x) / 2, (min_y + max_y) / 2),
                    width, height,
                    edgecolor=pathway_to_color[pathway],
                    facecolor=pathway_to_color[pathway],
                    alpha=0.2, lw=2
                )
                plt.gca().add_patch(ellipse)

    # Add title
    plt.title(f'Hypergraph Representation of a subset of {len(sample_pathways)} Reactome Pathways',)

    plt.savefig(output_path + "hypergraph_subset.pdf", bbox_inches='tight')

    plt.close()  # Close to free memory

    # Create a dummy figure just for the legend
    fig_legend = plt.figure(figsize=(15, 15))

    # Add legend handles only
    handles = [mpatches.Patch(color=pathway_to_color[pathway],
                              label=truncate(pathway, 40))
               for pathway in sample_pathways]

    legend = fig_legend.legend(
        handles=handles,
        loc='center',
        ncol=2,  # Adjust to fit the layout
        fontsize=10
    )

    # Hide axes
    fig_legend.gca().axis('off')

    # Save the legend as a separate file
    fig_legend.savefig(output_path + "hypergraph_legend.pdf", bbox_inches='tight')
    plt.close()