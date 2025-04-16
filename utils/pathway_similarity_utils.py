import os
import numpy as np
import pandas as pd
import networkx as nx

from utils.helpers import ensure_directory


def jaccard_similarity(set1, set2):

    intersection = len(set1.intersection(set2))
    union = len(set1) + len(set2) - intersection
    return intersection / union if union > 0 else 0


def select_pathway_to_keep(pathway1, pathway2, pathway_gene_sets):

    # Keep the pathway with more genes
    if len(pathway_gene_sets[pathway1]) != len(pathway_gene_sets[pathway2]):
        return pathway1 if len(pathway_gene_sets[pathway1]) > len(pathway_gene_sets[pathway2]) else pathway2

    # If sizes are equal, take the alphabetically smaller one
    return min(pathway1, pathway2)


def calculate_pathway_similarities(pathway_gene_sets):

    print("Calculating pairwise pathway similarities...")

    n_pathways = len(pathway_gene_sets)
    pathways = list(pathway_gene_sets.keys())
    similarities = []
    redundant_pairs = []
    high_threshold = 1.0  # Threshold for perfect redundancy

    # Process all pairs
    for i in range(n_pathways):
        if i % 100 == 0 and i > 0:
            print(f"Processed {i}/{n_pathways} pathways...")

        for j in range(i + 1, n_pathways):
            sim = jaccard_similarity(
                pathway_gene_sets[pathways[i]],
                pathway_gene_sets[pathways[j]]
            )
            similarities.append(sim)

            # If similarity is high, add to redundant pairs
            if sim >= high_threshold:
                redundant_pairs.append((
                    pathways[i],
                    pathways[j],
                    sim,
                    len(pathway_gene_sets[pathways[i]]),
                    len(pathway_gene_sets[pathways[j]])
                ))

    # Calculate summary statistics
    print("\nSummary of pathway similarity (Jaccard index):")
    print(f"Number of pathway pairs analyzed: {len(similarities)}")
    print(f"Mean similarity: {np.mean(similarities):.4f}")
    print(f"Median similarity: {np.median(similarities):.4f}")
    print(f"Max similarity: {np.max(similarities):.4f}")
    print(f"Min similarity: {np.min(similarities):.4f}")
    print(f"Found {len(redundant_pairs)} pathway pairs with similarity >= {high_threshold}")

    return similarities, redundant_pairs


def identify_redundant_pathways(redundant_pairs, pathway_gene_sets, pathway_id_to_name):

    print("Identifying redundant pathways...")

    # Create dictionaries to store removed pathways and their reasons
    perfect_overlap_removals = {}  # For 100% overlap cases

    # Process perfect overlap pairs first
    # Identify pairs with 100% overlap (Jaccard similarity = 1.0)
    perfect_overlap_pairs = [(p1, p2) for p1, p2, sim, _, _ in redundant_pairs if sim == 1.0]

    # Create a redundancy graph for transitive closure
    redundancy_graph = nx.Graph()
    for p1, p2 in perfect_overlap_pairs:
        redundancy_graph.add_edge(p1, p2)

    # Find all connected components (clusters of redundant pathways)
    redundant_clusters = list(nx.connected_components(redundancy_graph))

    # For each cluster, select one representative pathway
    for cluster in redundant_clusters:
        cluster_list = list(cluster)

        if len(cluster_list) == 1:
            continue  # Skip clusters with only one pathway

        # Multiple redundant pathways, select one using our criteria
        selected = cluster_list[0]
        for path in cluster_list[1:]:
            selected = select_pathway_to_keep(selected, path, pathway_gene_sets)

        # Record the removed pathways and why they were removed
        for path in cluster - {selected}:
            # Format the reason based on our selection criteria
            reason = f"100% overlap with {selected} ({pathway_id_to_name.get(selected, 'Unknown')})"

            # Add size information
            reason += f", kept size: {len(pathway_gene_sets[selected])}, removed size: {len(pathway_gene_sets[path])}"

            perfect_overlap_removals[path] = reason

    return perfect_overlap_removals


def generate_removal_report(removals, pathway_gene_sets, pathway_id_to_name, csv_path, md_path):

    print("Generating pathway removal report...")

    # Create the DataFrame
    removal_report = pd.DataFrame({
        'pathway_id': list(removals.keys()),
        'pathway_name': [pathway_id_to_name.get(p, 'Unknown') for p in removals.keys()],
        'reason_for_removal': list(removals.values()),
        'removal_type': ['Perfect overlap'] * len(removals)
    })

    # Add gene count
    removal_report['gene_count'] = [len(pathway_gene_sets[p]) for p in removal_report['pathway_id']]

    # Sort by removal type and then by pathway ID
    removal_report = removal_report.sort_values(['removal_type', 'pathway_id'])

    # Save the report to CSV
    ensure_directory(os.path.dirname(csv_path))
    removal_report.to_csv(csv_path, index=False)

    # Generate a markdown report for better readability
    ensure_directory(os.path.dirname(md_path))
    with open(md_path, "w") as f:
        f.write("# Removed Pathways Report\n\n")

        f.write("## Summary\n")
        f.write(f"- Total pathways removed: {len(removal_report)}\n")
        f.write(f"- Pathways removed due to perfect overlap: {len(removals)}\n")

        f.write("\n## Perfect Overlap Removals (100% similarity)\n\n")

        perfect_df = removal_report[removal_report['removal_type'] == 'Perfect overlap']
        for _, row in perfect_df.iterrows():
            f.write(f"### {row['pathway_id']} - {row['pathway_name']}\n")
            f.write(f"- Gene count: {row['gene_count']}\n")
            f.write(f"- Reason: {row['reason_for_removal']}\n\n")

    print(f"Removal reports saved to {csv_path} and {md_path}")
    print(f"Total pathways removed: {len(removal_report)}")

    return removal_report