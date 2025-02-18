# -*- coding: utf-8 -*-
# filename: test_model.py
# brief: test model on DRIVE dataset
# author: Jia Zhuang
# date: 2020-09-21

from PIL import Image
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os
import sys
import logging
from glob import glob
from dataloaders.drive import load_dataset
from networks.mynet import MyNet
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from utils import paste_and_save, eval_print_metrics
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from utils_dtc.metrics import dice, cal_dice

MODEL = "May31_14-37-10_8labels_beta_0.3_scaling_-1500withAug"
ITERATION = "iter_6000.pth"

project_dirname = os.path.join(os.path.dirname(__file__), "..")


def model_test(model, base_dir, save_imgs, batch_size=2):
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x_tensor, y_tensor = load_dataset(rel_path=project_dirname, mode='test', resize=True)
    num_samples = x_tensor.shape[0]
    writer = SummaryWriter(base_dir)
    dice_score_sum = 0
    cal_dice_score_sum  = 0
    logging.info("[+] ====== Start test... ======")
    num_iters = int(np.ceil(num_samples / batch_size))
    logging.info(f"number of iterations = {num_iters}")

    for ite in range(num_iters):
        torch.cuda.empty_cache()
        logging.info(f"[*] predicting on the {ite + 1}th batch")
        if not ite == num_iters - 1:
            start_id, end_id = ite * batch_size, (ite + 1) * batch_size
            bat_img = torch.Tensor(x_tensor[start_id : end_id, :, :, :]).to(device)
            bat_label = torch.Tensor(y_tensor[start_id : end_id, 0: 1, :, :]).to(device)
        else:
            start_id = ite * batch_size
            bat_img = torch.Tensor(x_tensor[start_id : , :, :, :]).to(device)
            bat_label = torch.Tensor(y_tensor[start_id : , 0: 1, :, :]).to(device)
        _, bat_pred = model(bat_img)
        bat_pred = torch.sigmoid(bat_pred)

        dice_metric = dice(bat_pred, bat_label)
        cl_dice_metric = cal_dice(bat_pred, bat_label)
        dice_score_sum += dice_metric
        cal_dice_score_sum += cl_dice_metric
        writer.add_scalar("test/dice", dice_metric, ite)
        writer.add_scalar("test/cl_dice", cl_dice_metric, ite)

        bat_pred_class = bat_pred.detach() 
        paste_and_save(bat_img, bat_label, bat_pred_class, batch_size, ite + 1, save_imgs)
        
        image = bat_img[0, :, :, :]
        grid_image = make_grid(image, 1, normalize=True)
        writer.add_image('test/Image', grid_image, ite)

        image = bat_pred[0, :, :, :]
        grid_image = make_grid(image, 1, normalize=False)
        writer.add_image('test/Predicted_label', grid_image, ite)

        image = bat_label[0, :, :]
        grid_image = make_grid(image, 1, normalize=False)
        writer.add_image('test/Groundtruth_label',
                            grid_image, ite)
    
    dice_average = dice_score_sum  / num_iters
    cal_dice_average = cal_dice_score_sum  / num_iters
    logging.info('average dice score is {}'.format(dice_average))
    logging.info('average cal dice score is {}'.format(cal_dice_average))

    return


if __name__ == "__main__":

    
    model_path = os.path.join(project_dirname,"model", "DRIVE", MODEL)
    snapshot_path = os.path.join(model_path, f"test_{ITERATION.split('.')[0]}")
    save_mgs_path = snapshot_path + "/pred_imgs"
    datasets_path = project_dirname + "/datasets/test"
    if not os.path.exists(save_mgs_path):
        os.makedirs(save_mgs_path)
    if not os.path.exists(datasets_path):
        os.makedirs(datasets_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # selected_model = sorted(glob(model_path + "/*.pth"))[-1]
    selected_model = model_path + "/" + ITERATION
    logging.info("[*] Selected model for testing: {} ".format(selected_model))
    model = MyNet(n_channels=3, n_classes=2-1,
                   normalization='batchnorm', has_dropout=True)
    model.load_state_dict(torch.load(selected_model, map_location=device))
    model.to(device)
    model_test(model, base_dir=snapshot_path, save_imgs=save_mgs_path, batch_size=2)
    