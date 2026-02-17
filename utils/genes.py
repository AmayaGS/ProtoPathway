"""
Gene expression utilities.

Functions for:
- Loading and parsing expression data
- Extracting protein-coding gene annotations from GTF
- Expression filtering (min expression, variance)
- Computing pathway activity scores
"""

import re
import logging
import numpy as np
import pandas as pd


def load_expression_data(path, transpose_if_needed=True):
    """
    Load gene expression matrix from CSV.

    Automatically detects orientation and returns patients × genes format.

    Args:
        path: Path to CSV file
        transpose_if_needed: If True, transpose to patients × genes if needed

    Returns:
        DataFrame with patients as rows, genes as columns
    """
    df = pd.read_csv(path, index_col=0)

    logging.info(f"Loaded expression data: {df.shape}")

    if transpose_if_needed:
        # Heuristic: if more rows than columns, likely genes × patients
        # TCGA has ~20k genes and ~500 patients typically
        if df.shape[0] > df.shape[1] * 2:
            logging.info("Transposing to patients × genes format")
            df = df.T

    logging.info(f"Expression matrix: {df.shape[0]} patients × {df.shape[1]} genes")

    return df


def extract_protein_coding_genes(gtf_path):
    """
    Extract protein-coding gene names from GTF file.

    Args:
        gtf_path: Path to GTF annotation file

    Returns:
        set: Set of protein-coding gene names
    """
    logging.info(f"Extracting protein-coding genes from {gtf_path}...")

    protein_coding = set()

    with open(gtf_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue

            fields = line.strip().split('\t')
            if len(fields) < 9 or fields[2] != 'gene':
                continue

            # Parse attributes
            attr_str = fields[8]
            attr_dict = dict(re.findall(r'(\S+)\s+"([^"]+)"', attr_str))

            gene_biotype = attr_dict.get('gene_biotype') or attr_dict.get('gene_type')
            gene_name = attr_dict.get('gene_name')

            if gene_biotype == 'protein_coding' and gene_name:
                protein_coding.add(gene_name)

    logging.info(f"Found {len(protein_coding)} protein-coding genes")

    return protein_coding


def filter_expression_data(expr_df, protein_coding_genes=None,
                           min_expression=1, min_proportion=0.1,
                           variance_proportion=0.8,
                           log_transform=True, center=True):
    """
    Filter gene expression data.

    Pipeline:
    1. Filter to protein-coding genes (if provided)
    2. Filter genes expressed above threshold in sufficient patients
    3. Log2 transform
    4. Center (subtract mean per gene)
    5. Keep top variance genes

    Args:
        expr_df: DataFrame with patients × genes
        protein_coding_genes: Set of protein-coding gene names (optional)
        min_expression: Minimum expression value
        min_proportion: Minimum fraction of patients expressing gene
        variance_proportion: Keep top X fraction by variance
        log_transform: Apply log2(x + 1) transform
        center: Subtract mean per gene

    Returns:
        Filtered DataFrame
    """
    df = expr_df.copy()
    logging.info(f"Starting expression filtering: {df.shape}")

    # Step 1: Filter to protein-coding genes
    if protein_coding_genes:
        genes_before = df.shape[1]
        common_genes = list(set(df.columns) & protein_coding_genes)
        df = df[common_genes]
        logging.info(f"Protein-coding filter: {genes_before} → {df.shape[1]} genes")

    # Step 2: Filter by minimum expression
    # Gene must be expressed above threshold in at least min_proportion of patients
    n_patients = df.shape[0]
    min_patients = int(min_proportion * n_patients)

    genes_expressed = (df > min_expression).sum(axis=0) >= min_patients
    df = df.loc[:, genes_expressed]
    logging.info(f"Expression filter (>{min_expression} in >{min_proportion:.0%} patients): {df.shape[1]} genes")

    # Step 3: Log2 transform
    if log_transform:
        df = np.log2(df + 1)
        logging.info("Applied log2(x + 1) transform")

    # Step 4: Center
    if center:
        df = df - df.mean(axis=0)
        logging.info("Centered genes (subtracted mean)")

    # Step 5: Variance filtering
    if variance_proportion < 1.0:
        variances = df.var(axis=0)
        n_keep = int(variance_proportion * len(variances))
        top_genes = variances.nlargest(n_keep).index
        df = df[top_genes]
        logging.info(f"Variance filter (top {variance_proportion:.0%}): {df.shape[1]} genes")

    logging.info(f"Final expression matrix: {df.shape}")

    return df


def compute_pathway_activity(expr_df, pathway_genes):
    """
    Compute pathway activity scores for each patient.

    Simple approach: mean expression of pathway genes per patient.

    Args:
        expr_df: DataFrame with patients × genes
        pathway_genes: dict {pathway_name: list of genes}

    Returns:
        DataFrame with patients × pathways (activity scores)
    """
    activities = {}

    for pathway_name, genes in pathway_genes.items():
        # Find genes present in expression data
        common_genes = list(set(genes) & set(expr_df.columns))

        if len(common_genes) >= 1:
            # Mean expression across pathway genes
            activities[pathway_name] = expr_df[common_genes].mean(axis=1)

    activity_df = pd.DataFrame(activities)
    logging.info(f"Computed activity for {activity_df.shape[1]} pathways")

    return activity_df


def compute_pathway_variance(expr_df, pathway_genes):
    """
    Compute variance of pathway activity across patients.

    Args:
        expr_df: DataFrame with patients × genes
        pathway_genes: dict {pathway_name: list of genes}

    Returns:
        dict: {pathway_name: variance}
    """
    activity_df = compute_pathway_activity(expr_df, pathway_genes)

    variances = activity_df.var(axis=0).to_dict()

    logging.info(f"Pathway variance - min: {min(variances.values()):.4f}, "
                 f"max: {max(variances.values()):.4f}, "
                 f"mean: {np.mean(list(variances.values())):.4f}")

    return variances


def filter_pathways_by_variance(pathway_df, pathway_variances, keep_percentile=75):
    """
    Filter pathways to keep top X percentile by variance.

    Args:
        pathway_df: DataFrame with pathway_id column
        pathway_variances: dict {pathway_id: variance}
        keep_percentile: Keep pathways above this percentile

    Returns:
        Filtered DataFrame
    """
    # Map variances to pathway_df
    pathway_df = pathway_df.copy()
    pathway_df['variance'] = pathway_df['pathway_id'].map(pathway_variances)

    # Handle pathways without variance (no genes in expression data)
    pathway_df = pathway_df.dropna(subset=['variance'])

    # Compute threshold
    threshold = np.percentile(list(pathway_variances.values()), 100 - keep_percentile)

    before = len(pathway_df)
    filtered = pathway_df[pathway_df['variance'] >= threshold].copy()

    logging.info(f"Variance filter (top {keep_percentile}%, threshold={threshold:.4f}): "
                 f"{before} → {len(filtered)} pathways")

    return filtered