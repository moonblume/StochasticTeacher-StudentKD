"""
metrics_manager.py

This module contains the implementation of the metrics manager, and other metrics related
functions.
"""

import os
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import (f1_score, classification_report, cohen_kappa_score,
                             confusion_matrix, accuracy_score)

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = torch.sum((pred == target).int()).item()
    return correct / len(target)

def f1(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
    return f1_score(pred.cpu().numpy(), target.data.cpu().numpy(), average='macro')

def _calc_metrics(config, checkpoint_dir, fold_id=None):
    n_folds = config["data_loader"]["args"]["num_folds"]
    all_outs, all_trgs = [], []
    outs_list, trgs_list = [], []

    save_dir = os.path.abspath(os.path.join(checkpoint_dir, os.pardir))
    for root, _, files in os.walk(save_dir):
        for file in files:
            if "outs" in file:
                outs_list.append(os.path.join(root, file))
            if "trgs" in file:
                trgs_list.append(os.path.join(root, file))

    if fold_id is not None:
        if fold_id >= len(outs_list):
            raise IndexError(f"Fold ID {fold_id} is out of range for outs_list with length {len(outs_list)}.")
        outs = np.load(outs_list[fold_id])
        trgs = np.load(trgs_list[fold_id])
        all_outs.extend(outs)
        all_trgs.extend(trgs)
        save_dir = os.path.abspath(os.path.join(checkpoint_dir))
    elif len(outs_list) == n_folds:
        for i in range(len(outs_list)):
            outs = np.load(outs_list[i])
            trgs = np.load(trgs_list[i])
            all_outs.extend(outs)
            all_trgs.extend(trgs)

    all_trgs = np.array(all_trgs).astype(int)
    all_outs = np.array(all_outs).astype(int)

    r = classification_report(all_trgs, all_outs, digits=6, output_dict=True)
    cm = confusion_matrix(all_trgs, all_outs)
    df = pd.DataFrame(r)
    df.loc[:, "cohen"] = cohen_kappa_score(all_trgs, all_outs)
    df.loc[:, "accuracy"] = accuracy_score(all_trgs, all_outs)
    df *= 100

    file_name = config["name"] + "_classification_report.xlsx"
    report_save_path = os.path.join(save_dir, file_name)
    df.to_excel(report_save_path)

    cm_file_name = config["name"] + "_confusion_matrix.torch"
    cm_save_path = os.path.join(save_dir, cm_file_name)
    torch.save(cm, cm_save_path)

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        self._data.loc[:, 'total':'average'] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.loc[key, 'total'] += value * n
        self._data.loc[key, 'counts'] += n
        self._data.loc[key, 'average'] = self._data.loc[key, 'total'] / self._data.loc[key, 'counts']

    def avg(self, key):
        return self._data.loc[key, 'average']

    def result(self):
        return self._data['average'].to_dict()
