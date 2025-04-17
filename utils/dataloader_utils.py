import torch
from dataset_utils import HistoDataset


def load_datasets(train_test_splits, dataset, batch_size):

    datasets = {}

    for fold, split in train_test_splits.items():
        train_ids = split['Train']
        test_ids = split['Test']

        train_dataset = []
        test_dataset = []

        for i in range(len(dataset)):
            data = dataset[i]
            patient_id = data[0]
            if patient_id in train_ids:
                train_dataset.append(data)
            elif patient_id in test_ids:
                test_dataset.append(data)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        datasets[fold] = {
            'Train': train_loader,
            'Test': test_loader
        }

    return datasets








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