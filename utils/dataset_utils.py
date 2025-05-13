import pandas as pd
import numpy as np
import ast

import torch
from torch.utils.data import Dataset

from torch_geometric.data import Data

from utils.survival_utils import discretize_survival_times, get_survival_target


class GeneExpressionDataset(Dataset):

    def __init__(self, config, gene_expr_df, labels_df):

        self.config = config
        self.task = config['execution'].get('task', 'classification')

        # Load gene expression data
        self.gene_expr_df = gene_expr_df

        # Load patient labels
        self.labels_df = labels_df

        # Double check common patient IDs
        self.patient_ids = list(set(self.gene_expr_df.index) &
                                set(self.labels_df[self.config['patient_id']]))

        print(f"Found {len(self.patient_ids)} patients with both expression data and labels")

        if self.task == 'survival':

            self.label_col = self.config['survival']['target_column']
            self.censor_col = self.config['survival']['censorship_column']
            self.n_bins = self.config['survival']['survival_bins']

            # we use the full dataset to determine the bins
            self.patient_df, self.bins = discretize_survival_times(
                self.labels_df,
                label_col=self.label_col,
                censor_col=self.censor_col,
                n_bins=self.n_bins
            )

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        # Get patient ID
        patient_id = self.patient_ids[idx]

        # Get gene expression vector
        gene_expr = self.gene_expr_df.loc[patient_id].values

        # Convert to tensor
        gene_expr_tensor = torch.FloatTensor(gene_expr)

        patient_row = self.patient_df.loc[self.patient_df[self.config['patient_id']] == patient_id].iloc[0]

        if self.task == 'classification':
            # For classification task
            label = patient_row[self.config['label']]
            target = torch.tensor(label, dtype=torch.long)
        else:
            target = get_survival_target(
                patient_row,
                self.label_col,
                self.censor_col,
                patient_row['label']
            )

        return {
            'patient_id': patient_id,
            'data': gene_expr_tensor,
            'id': patient_id,
            **target
        }


def build_incidence_matrix(pathway_genes_path, filtered_genes):

    # Load CSV and parse gene lists
    df = pd.read_csv(pathway_genes_path)
    df['genes'] = df['genes'].apply(ast.literal_eval)

    # Filter genes per pathway using filtered list
    filtered_gene_set = set(filtered_genes)

    # Remove genes not in filtered list from each pathway
    df['filtered_genes'] = df['genes'].apply(lambda gene_list: [g for g in gene_list if g in filtered_gene_set])

    # Remove pathways that now have no genes after filtering
    df = df[df['filtered_genes'].map(len) > 0].reset_index(drop=True)

    # Final list of genes used in the hypergraph (only those in both the expression data and at least one pathway)
    used_genes = sorted(list({g for gene_list in df['filtered_genes'] for g in gene_list}))

    # Indexing
    gene_idx = {g: i for i, g in enumerate(used_genes)}
    pathway_names = df['pathway_name'].tolist()
    pathway_idx = {p: j for j, p in enumerate(pathway_names)}

    # Build incidence matrix
    H = np.zeros((len(used_genes), len(pathway_names)), dtype=np.float32)
    for _, row in df.iterrows():
        j = pathway_idx[row['pathway_name']]
        for gene in row['filtered_genes']:
            i = gene_idx[gene]
            H[i, j] = 1.0

    # Also create edge_index format for PyG compatibility
    # This represents the hypergraph as a bipartite graph
    edge_indices = []
    for i in range(H.shape[0]):  # For each gene
        for j in range(H.shape[1]):  # For each pathway
            if H[i, j] > 0:
                # Gene to pathway connection
                edge_indices.append([i, j + H.shape[0]])  # Offset pathway indices

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t()

    return {
        'edge_index': edge_index,  # Bipartite edge index [2, num_edges]
        'num_genes': len(used_genes),
        'num_pathways': len(pathway_names),
        'gene_names': used_genes,
        'pathway_names': pathway_names,
        'gene_idx': gene_idx,
        'pathway_idx': pathway_idx
    }


class HypergraphDataset(Dataset):
    def __init__(self, config, gene_expr_df, labels_df, hypergraph_data):
        self.config = config

        # Hypergraph structure (shared across all samples)
        self.edge_index = hypergraph_data['edge_index']
        self.gene_names = hypergraph_data['gene_names']
        self.pathway_names = hypergraph_data['pathway_names']
        self.gene_idx = hypergraph_data['gene_idx']
        self.num_genes = hypergraph_data['num_genes']
        self.num_pathways = hypergraph_data['num_pathways']

        # Only keep patients that exist in both dataframes
        self.patient_ids = list(set(gene_expr_df.index) &
                                set(labels_df['Patient_ID']))

        # Filter gene_expr_df to only include genes in the hypergraph
        self.gene_expr_df = gene_expr_df[self.gene_names].loc[self.patient_ids]

        # Get labels
        self.labels_df = labels_df[labels_df['Patient_ID'].isin(self.patient_ids)]

        print(f"Found {len(self.patient_ids)} patients with both expression data and labels")
        print(f"Hypergraph includes {self.num_genes} genes and {self.num_pathways} pathways")

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]

        # Gene expression features [num_genes, 1]
        gene_expr = self.gene_expr_df.loc[patient_id].values
        x_gene = torch.FloatTensor(gene_expr).view(-1, 1)

        # Label
        label = self.labels_df.loc[
            self.labels_df[self.config['patient_id']] == patient_id,
            self.config['label']
        ].iloc[0]
        y = torch.tensor(label, dtype=torch.long)

        # # Create bipartite node features
        # # Gene nodes get expression values, pathway nodes get zeros
        x = torch.zeros((self.num_genes + self.num_pathways, 1), dtype=torch.float)
        x[:self.num_genes] = x_gene

        # Wrap in a PyG Data object
        data = Data(
            x=x,
            edge_index=self.edge_index,
            y=y,
            patient_id=patient_id,
            num_genes=self.num_genes,
            num_pathways=self.num_pathways
        )

        return data
