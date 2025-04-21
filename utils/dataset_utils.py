import pandas as pd

import torch
from torch.utils.data import Dataset

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


