# -*- coding: utf-8 -*-
# filename: utils.py
# brief: some utility functions
# author: Jia Zhuang
# date: 2020-09-21

from PIL import Image
import os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch import Tensor
from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union, cast

# from torchvision import transforms
# from torch.utils.data import DataLoader
# from math import floor
# import random

# project_dirname = os.path.join(os.path.dirname(__file__), "..")

# def get_data_loaders(data_base, transforms_, two_stream_sampler, seed, image_num, label_num, batch_size=4, labeled_bs=2):
#     def worker_init_fn(worker_id):
#         random.seed(seed+worker_id)
    
#     db_train = data_base(base_dir=project_dirname,
#                     transform=transforms.Compose(transforms_)
#                     )

#     labeled_idxs = list(range(label_num))
#     unlabeled_idxs = list(range(label_num, image_num))
#     # random.shuffle(labeled_idxs)
#     # random.shuffle(unlabeled_idxs)
#     train_labeled_num = floor(label_num * 0.8)
#     train_unlabeled_num = floor(len(unlabeled_idxs) * 0.8)
#     train_labeled_idxs = labeled_idxs[:train_labeled_num]
#     train_unlabeled_idxs = unlabeled_idxs[:train_unlabeled_num]
#     val_labeled_idxs = labeled_idxs[train_labeled_num:]
#     val_unlabeled_idxs = unlabeled_idxs[train_unlabeled_num:]

#     train_batch_sampler = two_stream_sampler(
#         train_labeled_idxs, train_unlabeled_idxs, batch_size, batch_size-labeled_bs)
#     val_batch_sampler = two_stream_sampler(
#         val_labeled_idxs, val_unlabeled_idxs, batch_size, batch_size-labeled_bs)

#     train_loader = DataLoader(db_train, batch_sampler=train_batch_sampler,
#                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
#     val_loader = DataLoader(db_train, batch_sampler=val_batch_sampler,
#                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
#     return train_loader, val_loader


def eval_print_metrics(bat_label, bat_pred, bat_mask):
    assert len(bat_label.size()) == 4 \
        and len(bat_pred.size()) == 4 and len(bat_mask.size()) == 4
    assert bat_label.size()[1] == 1 \
        and bat_pred.size()[1] == 1 and bat_mask.size()[1] == 1
    
    masked_pred = bat_pred * bat_mask
    masked_label = bat_label * bat_mask
    masked_pred_class = (masked_pred > 0.5).float()
    
    precision = float(torch.sum(masked_pred_class * masked_label)) / (float(torch.sum(masked_pred_class)) + 1)
    recall = float(torch.sum(masked_pred_class * masked_label)) / (float(torch.sum(masked_label.float())) + 1)
    f1_score = 2.0 * precision * recall / (precision + recall + 1e-8)
    
    pred_ls = np.array(bat_pred[bat_mask > 0].detach().cpu())
    label_ls = np.array(bat_label[bat_mask > 0].detach().cpu(), dtype=int)
    bat_auc = roc_auc_score(label_ls, pred_ls)
    bat_roc = roc_curve(label_ls, pred_ls)

    print("[*] ...... Evaluation ...... ")
    print(" >>> precision: {:.4f} recall: {:.4f} f1_score: {:.4f} auc: {:.4f}".format(precision, recall, f1_score, bat_auc))

    return precision, recall, f1_score, bat_auc, bat_roc

def paste_and_save(bat_img, bat_label, bat_pred_class, batch_size, cur_bat_num, save_img='pred_imgs'):
    w, h = bat_img.size()[2:4]
    for bat_id in range(bat_img.size()[0]):
        img = Image.fromarray(np.moveaxis(np.array(bat_img.cpu() * 255.0, dtype=np.uint8)[bat_id, :, :, :], 0, 2))
        label = Image.fromarray(np.array(bat_label.cpu() * 255.0, dtype=np.uint8)[bat_id, 0, :, :])
        pred_class = Image.fromarray(np.array(bat_pred_class.cpu() * 255.0, dtype=np.uint8)[bat_id, 0, :, :])
        
        res_id = (cur_bat_num - 1) * batch_size + bat_id
        target = Image.new('RGB', (3 * w, h))
        target.paste(img, box = (0, 0))
        target.paste(label, box = (w, 0))
        target.paste(pred_class, box = (2 * w, 0))
        
        target.save(os.path.join(save_img, "result_{}.png".format(res_id)))
    return

# Assert utils
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())

def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)

def simplex(t: Tensor, axis=1) -> bool:
    _sum = cast(Tensor, t.sum(axis).type(torch.float32))
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])
