# -*- coding: utf-8 -*-
# filename: utils.py
# brief: some utility functions
# author: Jia Zhuang
# date: 2020-09-21

import logging
from PIL import Image
import os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch import Tensor
from parameters import *
from dataloaders.dca1 import DCA1, RandomCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler
from torchvision import transforms
from torch.utils.data import DataLoader
from math import floor
import random

def get_data_loaders(parameters: Parameters):

    def worker_init_fn(worker_id):
        random.seed(parameters.seed+worker_id)

    myTransforms = [ToTensor()]
    if parameters.with_aug: 
        myTransforms = [
            RandomRotFlip(), 
            # RandomCrop(parameters.patch_size)
                        ] + myTransforms

    db_train = DCA1(base_dir=parameters.project_dirname,
                        transform=transforms.Compose(myTransforms)
                        )

    labeled_idxs = list(range(parameters.labelnum))
    unlabeled_idxs = list(range(parameters.labelnum, parameters.imagenum))
    # random.shuffle(labeled_idxs)
    # random.shuffle(unlabeled_idxs)
    train_labeled_num = floor(parameters.labelnum * 0.8)
    train_unlabeled_num = floor(len(unlabeled_idxs) * 0.8)
    train_labeled_idxs = labeled_idxs[:train_labeled_num]
    train_unlabeled_idxs = unlabeled_idxs[:train_unlabeled_num]
    val_labeled_idxs = labeled_idxs[train_labeled_num:]
    val_unlabeled_idxs = unlabeled_idxs[train_unlabeled_num:]

    logging.info(f"train labeled indices = {train_labeled_idxs}")
    logging.info(f"validation labeled indices = {val_labeled_idxs}")
    logging.info(f"train unlabeled indices = {train_unlabeled_idxs}")
    logging.info(f"validation unlabeled indices = {val_unlabeled_idxs}")

    train_batch_sampler = TwoStreamBatchSampler(
        train_labeled_idxs, train_unlabeled_idxs, parameters.batch_size, parameters.batch_size-parameters.labeled_bs)
    val_batch_sampler = TwoStreamBatchSampler(
        val_labeled_idxs, val_unlabeled_idxs, parameters.batch_size, parameters.batch_size-parameters.labeled_bs)


    train_loader = DataLoader(db_train, batch_sampler=train_batch_sampler,
                            num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(db_train, batch_sampler=val_batch_sampler,
                            num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    return train_loader, val_loader


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
