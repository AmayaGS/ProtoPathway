import re
import os
import numpy as np
import pandas as pd

from utils.helpers import ensure_directory


def extract_gene_annotations(gtf_path, output_path):
    print("Extracting gene annotations from GTF file...")
    genes = []

    with open(gtf_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue

            fields = line.strip().split('\t')
            if fields[2] != 'gene':
                continue

            attr_dict = dict(re.findall(r'(\S+)\s+"([^"]+)"', fields[8]))

            gene_id = attr_dict.get('gene_id')
            gene_name = attr_dict.get('gene_name')
            gene_biotype = attr_dict.get('gene_biotype') or attr_dict.get('gene_type')

            genes.append((gene_id, gene_name, gene_biotype))

    # Create DataFrame
    annotations_df = pd.DataFrame(genes, columns=['gene_id', 'gene_name', 'gene_biotype'])

    # Filter for protein-coding genes
    protein_coding_genes = annotations_df.loc[annotations_df['gene_biotype'] == 'protein_coding']
    annotations_df = annotations_df[annotations_df.index.isin(protein_coding_genes.index)]

    # Drop duplicates (some names may be repeated)
    annotations_df = annotations_df.drop_duplicates('gene_name')

    # Ensure output directory exists
    ensure_directory(os.path.dirname(output_path))
    annotations_df.to_csv(output_path, index=False)
    print(f"Annotation file saved to {output_path}")

    return annotations_df, protein_coding_genes['gene_name'].tolist()


def load_expression_data(gene_path):
    gene_df = pd.read_csv(gene_path, index_col=0)
    gene_df = gene_df.transpose()

    return gene_df


def filter_expression_data(gene_df, protein_coding_genes,
                           min_expression, min_proportion,
                           variance_proportion, output_path):

    print("Filtering gene expression data...")

    # Step 1: Filter for protein-coding genes
    gene_df = gene_df[gene_df.index.isin(protein_coding_genes)]
    print(f"After protein-coding filter: {gene_df.shape[0]} genes")

    # Step 2: Filter genes expressed in >min_proportion of patients
    expression_mask = (gene_df > min_expression).sum(axis=1) > (min_proportion * gene_df.shape[1])
    df_filtered = gene_df[expression_mask]
    print(f"After expression filter: {df_filtered.shape[0]} genes")

    # Step 3: Log2 transform
    df_log = np.log2(df_filtered + 1)

    # Step 4: Centering (subtract mean expression per gene)
    df_centered = df_log.sub(df_log.mean(axis=1), axis=0)

    # Step 5: Variance filtering - keep top variance_proportion most variable genes
    N = int(variance_proportion * df_centered.shape[0])
    variances = df_centered.var(axis=1)
    top_genes = variances.nlargest(N).index
    df_final = df_centered.loc[top_genes]
    print(f"After variance filter: {df_final.shape[0]} genes")

    # Save the final DataFrame
    # Ensure output directory exists
    ensure_directory(os.path.dirname(output_path))
    df_final.to_csv(output_path, sep=",", index=True)
    print(f"Filtered expression data saved to {output_path}")

    return df_final

