# -*- coding: utf-8 -*-
# filename: train_model.py
# brief: train U-net model on DRIVE dataset
# author: Jia Zhuang
# date: 2020-09-21

from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from torch import nn
import torch.backends.cudnn as cudnn
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import sys
import random
from dataloaders.dca1 import DCA1, RandomCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler
from networks.mynet import MyNet
import warnings
import logging
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from torch.nn import BCELoss, MSELoss
from datetime import datetime
import shutil

from utils_dtc.losses import dice_loss
from utils_dtc.losses_2 import compute_sdf
from utils_dtc.ramps import sigmoid_rampup

warnings.filterwarnings('ignore')

# set parameters for training
IMAGENUM = 100
LABELNUM = 20
LR = 0.1 # 1e-2
MAX_ITERATIONS = 6000
BATCH_SIZE = 8
LABELED_BS = 4
SAVE_EVERY = 20
EVAL_EVERY = 30
SEED = 1337
NUM_CLASSES = 2
PATCH_SIZE = (112, 112, 3)
BETA = 0.3
SCALING = -200
CONSISTENCY_RAMPUP = 40.0

project_dirname = os.path.join(os.path.dirname(__file__), "..")

db_train = DCA1(base_dir=project_dirname,
                    transform=transforms.Compose([
                    #    RandomRotFlip(),
                    #    RandomCrop(PATCH_SIZE),
                        ToTensor(),
                    ])
                    )

labeled_idxs = list(range(LABELNUM))
unlabeled_idxs = list(range(LABELNUM, IMAGENUM))
batch_sampler = TwoStreamBatchSampler(
    labeled_idxs, unlabeled_idxs, BATCH_SIZE, BATCH_SIZE-LABELED_BS)

def worker_init_fn(worker_id):
    random.seed(SEED+worker_id)
trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                            num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return sigmoid_rampup(epoch, CONSISTENCY_RAMPUP)

def model_train(net, model_path=None):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #optimizer = torch.optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=0.0001)
    #optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    #loss_func = nn.CrossEntropyLoss()
    # x_tensor, y_tensor, m_tensor = load_dataset(mode="training")
    #x_tensor, y_tensor, m_tensor = rim_padding(x_tensor), rim_padding(y_tensor), rim_padding(m_tensor)
    # num_samples = x_tensor.shape[0]
    writer = SummaryWriter(model_path+'/log')
    # ce_loss = BCEWithLogitsLoss()
    ce_loss = BCELoss()
    mse_loss = MSELoss()
    
    max_epoch = MAX_ITERATIONS//len(trainloader)+1
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch in iterator:
        print("[+] ====== Start training... epoch {} ======".format(epoch + 1))

        for _, sampled_batch in enumerate(trainloader):
            bat_img, bat_label = sampled_batch['image'], sampled_batch['label']
            bat_img, bat_label = bat_img.cuda(), bat_label.cuda()

            optimizer.zero_grad()
            bat_pred_tanh, bat_pred = net(bat_img)
            bat_pred_soft = torch.sigmoid(bat_pred)
            # flattened_tensor = bat_label[:labeled_bs].float().view(-1)
            # print("VALUES")
            # for value in flattened_tensor:
            #     if value > 1 or value < 0:
            #         print("========== {}".format(value.item()))
            # calculate the loss
            with torch.no_grad():
                gt_dis = compute_sdf(bat_label[:].cpu(
                ).numpy(), bat_pred[:LABELED_BS, 0, ...].shape)
                gt_dis = torch.from_numpy(gt_dis).float().cuda()
            loss_sdf = mse_loss(bat_pred_tanh[:LABELED_BS, 0, ...], gt_dis)
            loss_seg = ce_loss(
                bat_pred_soft[:LABELED_BS], bat_label[:LABELED_BS].float())
            

            loss_seg_dice = dice_loss(
                bat_pred_soft[:LABELED_BS], bat_label[:LABELED_BS] == 1)
            dis_to_mask = torch.sigmoid(SCALING*bat_pred_tanh)

            consistency_loss = torch.mean((dis_to_mask - bat_pred_soft) ** 2)
            supervised_loss = loss_seg_dice + BETA * loss_sdf
            consistency_weight = get_current_consistency_weight(epoch//150)

            loss = supervised_loss + consistency_weight * consistency_loss

            # loss = criterion(bat_pred_soft, bat_label == 1)
            writer.add_scalar("Loss/train", loss, epoch)
            # if is_eval:
                # if epoch % eval_every == 0:
                #     print("[*] ...... Eval for Epoch: {}, Iter: {} ....... ".format(epoch + 1, ite + 1))
                    # eval_print_metrics(bat_label, bat_pred_soft, bat_mask)

                # print("[+] ====== Epoch {} finished, avg_loss : {:.8f} ======"\
                #       .format(epoch + 1, epoch_avg_loss))
                # writer.add_scalar("Average_loss/train", epoch_avg_loss, epoch)
            writer.add_scalar('loss/loss', loss, epoch)
            # writer.add_scalar('loss/loss_seg', loss_seg, epoch)
            writer.add_scalar('loss/loss_dice', loss_seg_dice, epoch)
            writer.add_scalar('loss/loss_hausdorff', loss_sdf, epoch)
            writer.add_scalar('loss/consistency_weight',
                              consistency_weight, epoch)
            writer.add_scalar('loss/consistency_loss',
                              consistency_loss, epoch)

            print(
                'iteration %d : loss : %f, loss_consis: %f, loss_haus: %f, loss_seg: %f, loss_dice: %f' %
                (epoch, loss.item(), consistency_loss.item(), loss_sdf.item(),
                 loss_seg.item(), loss_seg_dice.item()))
            writer.add_scalar('loss/loss', loss, epoch)
            loss.backward()
            optimizer.step()
            
            image = bat_img[0, :, :, :]
            grid_image = make_grid(image, 1, normalize=True)
            writer.add_image('train/Image', grid_image, epoch)

            image = bat_pred_soft[0, :, :, :]
            grid_image = make_grid(image, 1, normalize=False)
            writer.add_image('train/Predicted_label', grid_image, epoch)

            image = bat_label[0, :, :]
            grid_image = make_grid(image, 1, normalize=False)
            writer.add_image('train/Groundtruth_label',
                                grid_image, epoch)
            image = dis_to_mask[0, 0:1, :, :]
            grid_image = make_grid(image, 1, normalize=False)
            writer.add_image('train/Dis2Mask', grid_image, epoch)

            image = bat_pred_tanh[0, 0:1, :, :]
            grid_image = make_grid(image, 1, normalize=False)
            writer.add_image('train/DistMap', grid_image, epoch)
                
            image = gt_dis[0, :, :]
            grid_image = make_grid(image, 1, normalize=False)
            writer.add_image('train/Groundtruth_DistMap',
                                grid_image, epoch)
                    
            if epoch % SAVE_EVERY == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(epoch) + '.pth')
                torch.save(net.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
            if epoch >= MAX_ITERATIONS:
                    break
        if epoch >= MAX_ITERATIONS:
            iterator.close()
            break
    writer.close()
    return net

if __name__ == "__main__":

    project_dirname = os.path.join(os.path.dirname(__file__), "..")
    
    # if not os.path.exists(project_dirname + "/checkpoint"):
    #     os.mkdir(project_dirname + "/checkpoint")
    if not os.path.exists(project_dirname + "/datasets"):
        os.mkdir(project_dirname + "/datasets")
    if not os.path.exists(project_dirname + "/datasets/training"):
        os.mkdir(project_dirname + "/datasets/training")

    # make logger file
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    snapshot_path = project_dirname + "/model/DRIVE/" + \
    "{}_{}labels_beta_{}_scaling_{}/".format(
        current_time, LABELNUM, BETA, SCALING)
    
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree(project_dirname + '/code', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))
    
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    cudnn.benchmark = False  
    cudnn.deterministic = True  
    
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mynet_ins = MyNet(n_channels=3, n_classes=NUM_CLASSES-1,
                   normalization='batchnorm', has_dropout=True)
    mynet_ins.to(device)
    trained_mynet = model_train(mynet_ins, model_path=snapshot_path)
