import numpy as np
import torch
import torch.backends.cudnn as cudnn

import os
import sys
import random
from parameters import *
from solver import Solver
from networks.mynet import MyNet
import warnings
import logging
from datetime import datetime
import shutil
from utils import get_data_loaders

warnings.filterwarnings('ignore')

# parse arguments
params = Parameters()

if __name__ == "__main__":
  
    if not os.path.exists(params.project_dirname + "/datasets"):
        os.mkdir(params.project_dirname + "/datasets")
    if not os.path.exists(params.project_dirname + "/datasets/training"):
        os.mkdir(params.project_dirname + "/datasets/training")
    
    if not os.path.exists(params.snapshot_path):
        print(params.snapshot_path)
        os.makedirs(params.snapshot_path)
    if os.path.exists(params.snapshot_path + '/code'):
        shutil.rmtree(params.snapshot_path + '/code')
    shutil.copytree(params.project_dirname + '/code', params.snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    # set logging
    logging.basicConfig(filename=params.snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    cudnn.benchmark = False  
    cudnn.deterministic = True  
    
    random.seed(params.seed)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed(params.seed)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cuda:0"):
        torch.cuda.empty_cache()

    train_loader, val_loader = get_data_loaders(params)

    model = MyNet(n_channels=3, n_classes=params.num_classes-1, normalization='batchnorm', has_dropout=True)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=params.lr, momentum=0.9, weight_decay=0.0001)
    solver = Solver(params,model=model, device=device, train_dataloader=train_loader, val_dataloader=val_loader, 
                    optimizer=optimizer) 
                   
    solver.train()