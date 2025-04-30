import pandas as pd
import numpy as np
import ast

import torch
from torch.utils.data import Dataset

from torch_geometric.data import Data

from PIL import Image


class GeneExpressionDataset(Dataset):

    def __init__(self, config, gene_expr_df, labels_df):

        self.config = config

        # Load gene expression data
        self.gene_expr_df = gene_expr_df

        # Load patient labels
        self.labels_df = labels_df

        # Double check common patient IDs
        self.patient_ids = list(set(self.gene_expr_df.index) &
                                set(self.labels_df['Patient_ID']))

        print(f"Found {len(self.patient_ids)} patients with both expression data and labels")

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        # Get patient ID
        patient_id = self.patient_ids[idx]

        # Get gene expression vector
        gene_expr = self.gene_expr_df.loc[patient_id].values

        # Convert to tensor
        gene_expr_tensor = torch.FloatTensor(gene_expr)

        # Get label for this patient
        label = self.labels_df.loc[self.labels_df[self.config['patient_id']] == patient_id, self.config['label']].iloc[0]
        label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            'data': gene_expr_tensor,
            'target': label_tensor,
            'id': patient_id
        }



def build_incidence_matrix(pathway_genes_path, filtered_genes):

    # Load CSV and parse gene lists
    df = pd.read_csv(pathway_genes_path)
    df['genes'] = df['genes'].apply(ast.literal_eval)

    # Filter genes per pathway using filtered list
    filtered_gene_set = set(filtered_genes)

    # Remove genes not in filtered list from each pathway
    df['filtered_genes'] = df['genes'].apply(lambda gene_list: [g for g in gene_list if g in filtered_gene_set])

    # all_pathway_genes = set()
    # for gene_list in df['genes']:
    #     all_pathway_genes.update(gene_list)
    #
    # # Calculate intersection and unique genes
    # intersection_genes = filtered_gene_set.intersection(all_pathway_genes)
    # only_in_filtered = filtered_gene_set - all_pathway_genes
    # only_in_pathways = all_pathway_genes - filtered_gene_set

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

    # H_tensor = torch.tensor(H, dtype=torch.float)

    # Also create edge_index format for PyG compatibility
    # This represents the hypergraph as a bipartite graph
    edge_indices = []
    for i in range(H.shape[0]):  # For each gene
        for j in range(H.shape[1]):  # For each pathway
            if H[i, j] > 0:
                # Gene to pathway connection
                edge_indices.append([i, j + H.shape[0]])  # Offset pathway indices

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t()

    # # Create hyperedge_index for HypergraphConv
    # # Format: [node_idx, hyperedge_idx]
    # node_indices = []
    # hyperedge_indices = []

    # for j, pathway in enumerate(pathway_names):
    #     for gene in df.loc[df['pathway_name'] == pathway, 'filtered_genes'].iloc[0]:
    #         if gene in gene_idx:  # Ensure the gene is in our index
    #             node_indices.append(gene_idx[gene])
    #             hyperedge_indices.append(j)
    #
    # hyperedge_index = torch.tensor([node_indices, hyperedge_indices], dtype=torch.long)

    return {
        #'H': H_tensor,  # Incidence matrix [num_genes, num_pathways]
        'edge_index': edge_index,  # Bipartite edge index [2, num_edges]
        #'hyperedge_index': hyperedge_index,  # Hyperedge index [2, num_connections]
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
        #self.H = hypergraph_data['H']
        self.edge_index = hypergraph_data['edge_index']
        #self.hyperedge_index = hypergraph_data['hyperedge_index']
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
            #hyperedge_index=self.hyperedge_index,
            #H=self.H,  # Store incidence matrix too if needed
            y=y,
            patient_id=patient_id,
            num_genes=self.num_genes,
            num_pathways=self.num_pathways
        )

        return data

# class HypergraphDataset(Dataset):
#     def __init__(self, config, gene_expr_df, labels_df, incidence_matrix, gene_names, pathway_names):
#
#         self.config = config
#         self.gene_expr_df = gene_expr_df
#         self.labels_df = labels_df
#         self.H = incidence_matrix  # shared across all samples
#         self.gene_names = gene_names
#         self.pathway_names = pathway_names
#
#         # Only keep patients that exist in both dataframes
#         # Double check common patient IDs
#         self.patient_ids = list(set(self.gene_expr_df.index) &
#                                 set(self.labels_df['Patient_ID']))
#
#         # Filter gene_expr_df to only include genes in the incidence matrix
#         self.filtered_genes = gene_names  # this should match columns in gene_expr_df
#         self.gene_expr_df = self.gene_expr_df[self.filtered_genes]
#
#         print(f"Found {len(self.patient_ids)} patients with both expression data and labels")
#         print(f"{self.gene_expr_df.shape[1]} unique genes")
#
#     def __len__(self):
#         return len(self.patient_ids)
#
#     def __getitem__(self, idx):
#         patient_id = self.patient_ids[idx]
#
#         # Expression vector → shape [num_genes, 1]
#         gene_expr = self.gene_expr_df.loc[patient_id].values
#         gene_expr_tensor = torch.FloatTensor(gene_expr).unsqueeze(-1)
#
#         # Label
#         label = self.labels_df.loc[
#             self.labels_df[self.config['patient_id']] == patient_id,
#             self.config['label']
#         ].iloc[0]
#         label_tensor = torch.tensor(label, dtype=torch.long)
#
#         # Wrap in a PyG Data object
#         data = Data(
#             x=gene_expr_tensor,
#             y=label_tensor,
#             incidence_matrix=self.H
#         )
#         data.patient_id = patient_id
#         data.gene_names = self.gene_names
#         data.pathway_names = self.pathway_names
#
#         return data


class ExpressionDataset(Dataset):

    def __init__(self, df, patient_id, label):

        self.labels = df[label].astype(int).tolist()
        self.gene_names = df.columns[3:]
        self.gene_expression = df.iloc[0:, 3:]
        self.patient_ID = df[patient_id].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        patient_ID = self.patient_ID[idx]
        label = torch.tensor(self.labels[idx])
        gene_expression = torch.as_tensor(self.gene_expression.iloc[idx], dtype=torch.float32)
        return [patient_ID, gene_expression, label]


class PathwayDataset(Dataset):

    def __init__(self, df, label):

        self.labels = df[label].astype(int).tolist()
        self.pathway_names = df['pathway'].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx])
        pathway_name = self.pathway_names[idx]
        return [pathway_name, label]


class HistoDataset(Dataset):

    def __init__(self, df, transform, label):

        self.transform = transform
        self.labels = df[label].astype(int).tolist()
        self.filepaths = df['File_location'].tolist()
        self.patient_IDs = df['Patient_ID'].tolist()
        self.filenames = df['Filename'].tolist()
        self.patch_names = df['Patch_name'].tolist()
        self.coordinates = df['Patch_coordinates'].tolist()
        self.stain_types = df['Stain_type'].tolist()

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):

        try:
            image = Image.open(self.filepaths[idx])
            # If the image has an alpha channel, remove it
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            patient_id = self.patient_IDs[idx]
            filename = self.filenames[idx]
            patch_name = self.patch_names[idx]
            coordinate = self.coordinates[idx]
            self.image_tensor = self.transform(image)
            self.image_label = self.labels[idx]
            stain_type = self.stain_types[idx]

            return self.image_tensor, self.image_label, patient_id, filename, patch_name, coordinate, stain_type

        except (FileNotFoundError, IndexError):
            return self.__getitem__(idx)


