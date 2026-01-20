import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from legacy_code.utils.helpers import ensure_directory


def parse_reactome_gmt(file_path):
    """Parse a Reactome GMT file and return a dictionary and DataFrame."""
    pathway_dict = {}
    pathway_tuples = []

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            pathway_id = parts[0]  # e.g. R-HSA-123456
            pathway_name = parts[1]
            genes = parts[2:]
            pathway_dict[pathway_id] = genes
            pathway_tuples.append((pathway_id, pathway_name, genes))

    return pathway_dict, pd.DataFrame(pathway_tuples, columns=['pathway_name', 'pathway_id', 'genes'])


def load_reactome_pathways(gmt_path, output_path):

    print("Loading Reactome pathways...")
    pathway_dict, pathway_df = parse_reactome_gmt(gmt_path)

    # Ensure output directory exists
    ensure_directory(os.path.dirname(output_path))
    pathway_df.to_csv(output_path, index=False)
    print(f"Reactome pathways saved to {output_path}")

    return pathway_dict, pathway_df


def build_reactome_hierarchy(relations_path, pathways_path):

    print("Building Reactome hierarchy...")

    # Load hierarchy relationships
    relations = pd.read_csv(relations_path, sep="\t", header=None,
                            names=["parent", "child"])

    # Filter for human-only (R-HSA) pathways
    relations = relations[
        relations["parent"].str.startswith("R-HSA") &
        relations["child"].str.startswith("R-HSA")
        ]

    # Load pathway names
    pathway_names = pd.read_csv(pathways_path, sep="\t", header=None,
                                names=["pathway_id", "pathway_name", "species"])

    # Keep only human pathways
    pathway_names = pathway_names[pathway_names['species'] == 'Homo sapiens']

    # Build a directed graph of the hierarchy
    G = nx.DiGraph()
    G.add_edges_from(relations.itertuples(index=False, name=None))

    # Find root nodes (no incoming edges)
    roots = [n for n in G.nodes if G.in_degree(n) == 0]

    # Compute shortest path lengths from all roots
    depths = {}
    for root in roots:
        lengths = nx.single_source_shortest_path_length(G, root)
        for node, depth in lengths.items():
            if node not in depths or depth < depths[node]:
                depths[node] = depth

    # Merge with pathway names
    depth_df = pd.DataFrame.from_dict(depths, orient='index',
                                      columns=['depth']).reset_index()
    depth_df = depth_df.rename(columns={'index': 'pathway_id'})
    depth_df = depth_df.merge(pathway_names, on='pathway_id')

    return G, depth_df


def select_pathways_by_depth(G, depth_df, target_depth=5):

    print(f"Selecting pathways at depth {target_depth}...")

    # Extract all pathways at depth exactly target_depth
    level_pathways = set(depth_df[depth_df['depth'] == target_depth]['pathway_id'].tolist())

    # Find all shallow nodes (depth < target_depth)
    shallow_nodes = set(n for n in G.nodes if n in depth_df['pathway_id'].tolist()
                        and depth_df.loc[depth_df['pathway_id'] == n, 'depth'].iloc[0] < target_depth)

    # For each shallow node, check if any of its descendants reach target_depth
    nodes_with_target_descendants = set()
    for node in shallow_nodes:
        # Get all descendants of the node
        descendants = nx.descendants(G, node)

        # Check if any descendant is at target_depth
        if any(desc in level_pathways for desc in descendants):
            nodes_with_target_descendants.add(node)

    # Shallow nodes without target_depth descendants
    candidates = shallow_nodes - nodes_with_target_descendants

    # Find nodes that are the deepest in their branch
    additional_pathways = set()
    for node in candidates:
        # Check if any of its children are also candidates
        is_deepest = True
        for child in G.successors(node):
            if child in candidates:
                is_deepest = False
                break

        if is_deepest:
            additional_pathways.add(node)

    # Combine target_depth pathways and additional pathways
    final_pathways = level_pathways.union(additional_pathways)

    print(f"Selected {len(level_pathways)} pathways at depth {target_depth}")
    print(
        f"Selected {len(additional_pathways)} additional pathways from branches that don't reach depth {target_depth}")
    print(f"Total selected: {len(final_pathways)} pathways")

    return final_pathways



def filter_pathways_by_size(pathway_df, min_genes, max_genes):

    # Add gene count column if not present
    if 'gene_count' not in pathway_df.columns:
        pathway_df['gene_count'] = pathway_df['genes'].apply(len)

    # Filter by size
    filtered = pathway_df[(pathway_df['gene_count'] >= min_genes) &
                          (pathway_df['gene_count'] <= max_genes)]

    print(f"Filtered from {len(pathway_df)} to {len(filtered)} pathways based on size")
    print(f"  Min genes: {min_genes}, Max genes: {max_genes}")

    return filtered


def plot_pathway_depth_histogram(depth_df, target_depth, output_path):

    plt.figure(figsize=(8, 5))
    plt.hist(depth_df['depth'], bins=range(depth_df['depth'].max() + 1),
             edgecolor='black')
    plt.axvline(x=target_depth, color='red', linestyle='--',
                label=f'Max Pathway depth ({target_depth})')
    plt.xlabel('Pathway Depth', fontsize=12)
    plt.ylabel('Number of Pathways', fontsize=12)
    plt.title('Distribution of Reactome Pathway Depths')
    plt.legend()

    plt.savefig(output_path + "fig_pathway_depth_histogram.pdf", bbox_inches='tight')

    plt.tight_layout()
    plt.close()  # Close figure to conserve memory


def plot_pathway_size_histograms(pathway_df, min_genes, max_genes, base_path):

    # Plot full histogram
    plt.figure(figsize=(10, 6))
    plt.hist(pathway_df['gene_count'], bins=70, color='skyblue', edgecolor='black')
    plt.xlabel('Number of genes per pathway (after filtering)', fontsize=12)
    plt.ylabel('Number of pathways', fontsize=12)
    plt.title('Distribution of selected & filtered pathway sizes')
    plt.axvline(x=min_genes, color='red', linestyle='--', label='Min/Max size threshold')
    plt.axvline(x=max_genes, color='red', linestyle='--')
    plt.savefig(base_path + "fig_pathway_histogram.pdf", bbox_inches='tight')
    plt.close()

    # Plot zoomed min range
    plt.figure(figsize=(10, 6))
    plt.hist(pathway_df['gene_count'], bins=100, range=(1, 100),
             color='darkgray', edgecolor='black')
    plt.xlabel('Number of genes per pathway (after filtering)', fontsize=12)
    plt.ylabel('Number of pathways', fontsize=12)
    plt.title('Distribution of Reactome pathway sizes (filtered, zoomed to 1–100 genes)')
    plt.axvline(x=min_genes, color='red', linestyle='--',
                label=f'Min size threshold ({min_genes})')
    plt.legend()
    plt.savefig(base_path + "fig_pathway_histogram_zoom_min.pdf", bbox_inches='tight')
    plt.close()

    # Plot zoomed max range
    plt.figure(figsize=(10, 6))
    plt.hist(pathway_df['gene_count'], bins=100, range=(50, 700),
             color='darkgray', edgecolor='black')
    plt.xlabel('Number of genes per pathway (after filtering)', fontsize=12)
    plt.ylabel('Number of pathways', fontsize=12)
    plt.title('Distribution of Reactome pathway sizes (filtered, zoomed to 50–700 genes)')
    plt.axvline(x=max_genes, color='red', linestyle='--',
                label=f'Max size threshold ({max_genes})')
    plt.legend()
    plt.savefig(base_path + "fig_pathway_histogram_zoom_max.pdf", bbox_inches='tight')
    plt.close()


def plot_filtered_pathway_histogram(filtered_df, min_genes, max_genes, output_path):

    plt.figure(figsize=(10, 6))
    plt.hist(filtered_df['gene_count'], bins=70, color='skyblue', edgecolor='black')
    plt.xlabel('Number of genes per pathway (after filtering)', fontsize=12)
    plt.ylabel('Number of pathways', fontsize=12)
    plt.title(f'Distribution of selected & filtered pathway sizes - {len(filtered_df)} pathways')
    plt.axvline(x=min_genes, color='red', linestyle='--', label='Min/Max size threshold')
    plt.axvline(x=max_genes, color='red', linestyle='--')

    plt.savefig(output_path + "final_filtered_pathways.pdf" , bbox_inches='tight')

    plt.close()
