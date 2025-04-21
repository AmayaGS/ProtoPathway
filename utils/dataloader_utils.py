import pandas as pd
import pickle
import os

import torch
from dataset_utils import HistoDataset



class HistoDataloader:

    def slides_dataloader(self, df, ids, transform, slide_batch, num_workers, shuffle, collate, label, patient_id):
        # TRAIN dict
        patient_subsets = {}

        for i, file in enumerate(ids):
            new_key = f'{file}'
            patient_subset = HistoDataset(df[df[patient_id] == file], transform, label)
            patient_subsets[new_key] = torch.utils.data.DataLoader(patient_subset, batch_size=slide_batch,
                                                                   shuffle=shuffle, num_workers=num_workers,
                                                                   drop_last=False, collate_fn=collate)

        return patient_subsets