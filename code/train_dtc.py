import os
import sys
import cv2
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn import BCELoss, BCEWithLogitsLoss, MSELoss
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from unet_model import Unet
from utils_dtc import ramps, losses, metrics
from data_loader import load_dataset
import datetime

import gc
gc.collect()
torch.cuda.empty_cache()
 
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='data/DRIVE/', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='DRIVE/Unet_DTC_with_consis_weight', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=4,
                    help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.1,
                    help='maximum epoch number to train')
parser.add_argument('--D_lr', type=float,  default=1e-4,
                    help='maximum discriminator learning rate to train')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--labelnum', type=int,  default=16, help='random seed')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--consistency_weight', type=float,  default=0.1,
                    help='balance factor to control supervised loss and consistency loss')
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
parser.add_argument('--beta', type=float,  default=0.3,
                    help='balance factor to control regional and sdm loss')
parser.add_argument('--gamma', type=float,  default=0.5,
                    help='balance factor to control supervised and consistency loss')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="kl", help='consistency_type')
parser.add_argument('--with_cons', type=str,
                    default="without_cons", help='with or without consistency')
parser.add_argument('--consistency', type=float,
                    default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=40.0, help='consistency_rampup')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = ("model/" + args.exp + \
    "_{}labels_beta_{}_{}/".format(
        args.labelnum, args.beta, datetime.datetime.now())).replace(" ", "")

# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs


num_classes = 2


if __name__ == "__main__":
    # make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def create_model(ema=False):
        # Network definition
        net = Unet(img_ch=3, isDeconv=True, isBN=True)
        model = net.cuda()
        return model

    model = create_model()
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr)
    ce_loss = BCELoss()
    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(500))

    iter_num = 0
    max_epoch = 500
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70)
    x_tensor, y_tensor, m_tensor = load_dataset(mode="training")
    num_samples = x_tensor.shape[0]
    
    for epoch_num in iterator:
        epoch_tot_loss = 0
        print("[+] ====== Start training... epoch {} ======".format(epoch_num + 1))
        num_iters = int(np.ceil(num_samples / batch_size))
        shuffle_ids = np.random.permutation(num_samples)
        x_tensor = x_tensor[shuffle_ids, :, :, :]
        y_tensor = y_tensor[shuffle_ids, :, :, :]
        m_tensor = m_tensor[shuffle_ids, :, :, :]
        for iter_num in range(num_iters):
            if not iter_num == num_iters - 1:
                start_id, end_id = iter_num * batch_size, (iter_num + 1) * batch_size
                bat_img = torch.Tensor(x_tensor[start_id : end_id, :, :, :]).cuda()
                bat_label = torch.Tensor(y_tensor[start_id : end_id, 0: 1, :, :]).cuda()
                # bat_mask_2ch = torch.Tensor(m_tensor[start_id : end_id, :, :, :]).cuda()
                bat_mask = torch.Tensor(m_tensor[start_id : end_id, 0: 1, :, :]).cuda()
            else:
                start_id = iter_num * batch_size
                bat_img = torch.Tensor(x_tensor[start_id : , :, :, :]).cuda()
                bat_label = torch.Tensor(y_tensor[start_id : , 0: 1, :, :]).cuda()
                # bat_mask_2ch = torch.Tensor(m_tensor[start_id : , :, :, :]).cuda()
                bat_mask = torch.Tensor(m_tensor[start_id : , 0: 1, :, :]).cuda()

            outputs = model(bat_img)
            loss = ce_loss(outputs, bat_label.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, epoch_num)
            writer.add_scalar('loss/loss', loss, epoch_num)
            logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))
            if epoch_num % 50 == 0:
                image = bat_img[0, :, :, :]
                grid_image = make_grid(image, 1, normalize=True)
                writer.add_image('train/Image', grid_image, epoch_num)

                image = outputs[0, :, :, :]
                grid_image = make_grid(image, 1, normalize=False)
                writer.add_image('train/Predicted_label', grid_image, epoch_num)

                # image = dis_to_mask[0, :, :, :]
                # grid_image = make_grid(image, 1, normalize=False)
                # writer.add_image('train/Dis2Mask', grid_image, epoch_num)

                # cv2 test
                # file = 'ToWriteLabel.jpg'
                # cv2.imwrite(file, bat_label[0, :, :].cpu().numpy())
                
                image = bat_label[0, :, :]
                grid_image = make_grid(image, 1, normalize=False)
                writer.add_image('train/Groundtruth_label',
                                 grid_image, epoch_num)

                # image = gt_dis[0, :, :]
                # grid_image = make_grid(image, 1, normalize=False)
                # writer.add_image('train/Groundtruth_DistMap',
                #                  grid_image, epoch_num)
                
            # change lr
            # if epoch_num % 150 == 0:
            #     lr_ = base_lr * 0.1 ** (epoch_num // 150)
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = lr_
            if epoch_num % 20 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(epoch_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if epoch_num >= max_iterations:
                break
        if epoch_num >= max_iterations:
            iterator.close()
            break
    writer.close()
