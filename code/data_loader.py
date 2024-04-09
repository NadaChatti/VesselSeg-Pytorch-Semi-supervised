# -*- coding: utf-8 -*-
# filename: data_loader.py
# brief: load DRIVE dataset and form as tensors
# author: Jia Zhuang
# date: 2020-09-18

from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
import os
from glob import glob

def load_dataset(rel_path='.', mode="training", resize=True, resize_shape=(256, 256), labelnum=20):
    datasets_path = os.path.join(rel_path, "datasets", mode)
    src_path = os.path.join(rel_path, "DRIVE", mode)

    image_path = os.path.join(datasets_path, "image.npy")
    label_path = os.path.join(datasets_path, "label.npy")
    mask_path = os.path.join(datasets_path, "mask.npy")

    if os.path.exists(image_path) and \
        os.path.exists(label_path) and \
        os.path.exists(mask_path):
        
        new_input_tensor = np.load(image_path)
        new_label_tensor = np.load(label_path)
        new_mask_tensor = np.load(mask_path)
        return new_input_tensor, new_label_tensor, new_mask_tensor

    train_image_files = sorted(glob(os.path.join(src_path, 'images/*.tif')))
    train_label_files = sorted(glob(os.path.join(src_path, '1st_manual/*.gif')))
    train_mask_files = sorted(glob(os.path.join(src_path, 'mask/*.gif')))
    
    for i, filename in enumerate(train_image_files):
        print('[*] adding {}th {} image : {}'.format(i + 1, mode, filename))
        img = Image.open(filename)
        if resize:
            img = img.resize(resize_shape, Image.LANCZOS)
        imgmat = np.array(img).astype('float')
        imgmat = imgmat / 255.0
        if i == 0:
            input_tensor = np.expand_dims(imgmat, axis=0)
        else:
            tmp = np.expand_dims(imgmat, axis=0)
            input_tensor = np.concatenate((input_tensor, tmp), axis=0)
    new_input_tensor = np.moveaxis(input_tensor, 3, 1)
            
    for i, filename in enumerate(train_label_files):
        print('[*] adding {}th {} label : {}'.format(i + 1, mode, filename))
        Img_label = Image.open(filename)
        if resize:
            Img_label = Img_label.resize(resize_shape, Image.LANCZOS)
            Img_label = Img_label.convert('1')
        label = np.array(Img_label)
        label = label / 1.0
        if i == 0:
            label_tensor = np.expand_dims(label, axis=0)
        else:
            tmp = np.expand_dims(label, axis=0)
            label_tensor = np.concatenate((label_tensor, tmp), axis=0)
    new_label_tensor = np.stack((label_tensor[:,:,:], 1 - label_tensor[:,:,:]), axis=1)
    
    for i, filename in enumerate(train_mask_files):
        print('[*] adding {}th {} mask : {}'.format(i+1, mode, filename))
        Img_mask = Image.open(filename)
        if resize:
            Img_mask = Img_mask.resize(resize_shape, Image.LANCZOS)
            Img_mask = Img_mask.convert('1')
        mask = np.array(Img_mask)
        mask = mask / 1.0
        if i == 0:
            mask_tensor = np.expand_dims(mask, axis=0)
        else:
            tmp = np.expand_dims(mask, axis=0)
            mask_tensor = np.concatenate((mask_tensor, tmp), axis=0)
    new_mask_tensor = np.stack((mask_tensor[:,:,:], mask_tensor[:,:,:]), axis=1)
    if not os.path.exists(datasets_path + "/"):
        os.makedirs(datasets_path + "/")
    np.save(image_path, new_input_tensor)
    np.save(label_path, new_label_tensor)
    np.save(mask_path, new_mask_tensor)
    
    return new_input_tensor, new_label_tensor, new_mask_tensor