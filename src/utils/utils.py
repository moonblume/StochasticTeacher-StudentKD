"""
utils.py
"""

import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Label dicts
pain_labels = {
    'BL1': 0,
    'PA1': 1,
    'PA2': 2,
    'PA3': 3,
    'PA4': 4
}

# Multiclass type
class_types_dict = {
    '0 vs 1 vs 2 vs 3 vs 4': 0,
    '0 vs 4': 1,
    '0 vs 1, 2, 3, 4': 2,
    '0, 1 vs 3, 4': 3,
    '0 vs 3, 4': 4,
    '0, 1 vs 4': 5,
    '0, 1 vs 3': 6,
    '0 vs 1, 2 vs 3, 4': 7
}

def generate_kfolds_index(data_dir, k_folds) -> dict[int, list[str]]:
    """
    Generate k-folds, Leave-One-Subject-Out (LOSO), dataset index and store into a dictionary. The length of the dictionary is equal to the number of
    folds. Each element contains a training set and a testing set.
    :param data_dir: npz files directory
    :param k_folds: the number of folds
    :return: a dict contains k-folds dataset paths, e.g. dict{0: [list[str(train_dir)], list[str(test_dir)]]..., k:[...]}
    """
    if os.path.exists(data_dir):
        print('================= Creating KFolds Index =================')
    else:
        raise FileNotFoundError('================= Data directory does not exist =================')
    npz_files = glob.glob(os.path.join(data_dir, '*.npz'))
    npz_files = np.asarray(npz_files)
    kfolds_names = np.array_split(npz_files, k_folds)

    kfolds_index = {}
    for fold_index in range(0, k_folds):
        test_data = kfolds_names[fold_index].tolist()
        train_data = [files for i, files in enumerate(kfolds_names) if i != fold_index]
        train_data = [files for subfiles in train_data for files in subfiles]
        kfolds_index[fold_index] = [train_data, test_data]
    print('================= {} folds dataset created ================='.format(k_folds))
    return kfolds_index

class BioVidLoader(Dataset):
    """
    Input: a list of npz files' directories from k-folds index
    Output: a tensor of values and labels
    """

    def __init__(self, npz_files, label_converter):
        super(BioVidLoader, self).__init__()

        x_values_list = []
        y_labels_list = []

        for file in npz_files:
            data = np.load(file)
            if 'x' not in data or 'y' not in data:
                raise ValueError(f"File {file} does not contain 'x' or 'y' arrays.")
            
            x_values = data['x']
            y_labels = data['y']

            # print(f"Loaded file: {file}, x_values shape: {x_values.shape}, y_labels shape: {y_labels.shape}")

            if y_labels.ndim == 0:
                y_labels = y_labels.reshape(1)

            if y_labels.size == 0:
                raise ValueError(f"File {file} contains an empty 'y' array.")

            # Verify the consistency of the labels within the file
            unique_labels = np.unique(y_labels)
            if len(unique_labels) != 1:
                raise ValueError(f"Inconsistent labels in file {file}: {unique_labels}")

            # Use the filename to determine the label
            label_name = os.path.basename(file).split('-')[1]
            if label_name not in label_converter:
                raise ValueError(f"Label {label_name} not found in label converter.")

            new_label = label_converter[label_name]

            # Replace all y_labels with the new label
            y_labels = np.full_like(y_labels, new_label)

            x_values_list.append(x_values)
            y_labels_list.append(y_labels)

            # print(f"File: {file}, New Label: {new_label}")

        if not x_values_list or not y_labels_list:
            raise ValueError("No data loaded. Please check the npz files.")

        x_values = np.vstack(x_values_list)
        y_labels = np.concatenate(y_labels_list)

        print(f"Final x_values shape: {x_values.shape}, Final y_labels shape: {y_labels.shape}")

        self.val = torch.from_numpy(x_values).float()
        self.lbl = torch.from_numpy(y_labels).long()

        # Change shape to (Batch size, Channel size, Length)
        self.val = self.val.unsqueeze(1)

    def __len__(self):
        return self.val.shape[0]

    def __getitem__(self, idx):
        return self.val[idx], self.lbl[idx]

def load_data(train_set, valid_set, label_converter, batch_size, num_workers=0) -> tuple[DataLoader, DataLoader, list[int]]:
    """
    Generate dataloader for both training dataset and validation dataset from one of the k-folds.
    :param train_set: training dataset
    :param valid_set: validation dataset
    :param label_converter: convert the original labels to the desired labels
    :param batch_size: batch size
    :param num_workers: 4*GPU
    :return: dataloader for training dataset, validation dataset, the number of samples for each class,
    e.g. two classes -> list[int,int]
    """

    train_dataset = BioVidLoader(train_set, label_converter)
    valid_dataset = BioVidLoader(valid_set, label_converter)

    cat_y = torch.cat((train_dataset.lbl, valid_dataset.lbl))

    unique_counts = cat_y.unique(return_counts=True)
    dist = unique_counts[1].tolist()

    train_loader = DataLoader(train_dataset,
                              num_workers=num_workers,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=False,
                              pin_memory=True)

    valid_loader = DataLoader(valid_dataset,
                              num_workers=num_workers,
                              batch_size=batch_size,
                              shuffle=False,
                              drop_last=False,
                              pin_memory=True)

    return train_loader, valid_loader, dist
