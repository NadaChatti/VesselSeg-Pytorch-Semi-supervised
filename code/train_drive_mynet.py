import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import sys
import random
from solver import Solver
from dataloaders.drive import Drive, RandomCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler
from networks.mynet import MyNet
import warnings
import logging
from datetime import datetime
import shutil
from math import floor
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')

# set parameters for training
IMAGENUM = 20
LABELNUM = 8
LR = 1e-2
MAX_ITERATIONS = 6000
BATCH_SIZE = 4
LABELED_BS = 2
SAVE_EVERY = 100
PRINT_EVERY = 1
SEED = 1337
NUM_CLASSES = 2
PATCH_SIZE = (3, 112, 112)
BETA = 0.3
SCALING = -1500
CONSISTENCY_RAMPUP = 40.0
WITH_AUG = True

project_dirname = os.path.join(os.path.dirname(__file__), "..")
myTransforms = [ToTensor()]
if WITH_AUG: 
    myTransforms = [
        RandomRotFlip(), 
        # RandomCrop(PATCH_SIZE)
                    ] + myTransforms

db_train = Drive(base_dir=project_dirname,
                    transform=transforms.Compose(myTransforms)
                    )

labeled_idxs = list(range(LABELNUM))
unlabeled_idxs = list(range(LABELNUM, IMAGENUM))
# random.shuffle(labeled_idxs)
# random.shuffle(unlabeled_idxs)
train_labeled_num = floor(LABELNUM * 0.8)
train_unlabeled_num = floor(len(unlabeled_idxs) * 0.8)
train_labeled_idxs = labeled_idxs[:train_labeled_num]
train_unlabeled_idxs = unlabeled_idxs[:train_unlabeled_num]
val_labeled_idxs = labeled_idxs[train_labeled_num:]
val_unlabeled_idxs = unlabeled_idxs[train_unlabeled_num:]

train_batch_sampler = TwoStreamBatchSampler(
    train_labeled_idxs, train_unlabeled_idxs, BATCH_SIZE, BATCH_SIZE-LABELED_BS)
val_batch_sampler = TwoStreamBatchSampler(
    val_labeled_idxs, val_unlabeled_idxs, BATCH_SIZE, BATCH_SIZE-LABELED_BS)

def worker_init_fn(worker_id):
    random.seed(SEED+worker_id)

train_loader = DataLoader(db_train, batch_sampler=train_batch_sampler,
                        num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
val_loader = DataLoader(db_train, batch_sampler=val_batch_sampler,
                        num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)


if __name__ == "__main__":

    project_dirname = os.path.join(os.path.dirname(__file__), "..")
    
    if not os.path.exists(project_dirname + "/datasets"):
        os.mkdir(project_dirname + "/datasets")
    if not os.path.exists(project_dirname + "/datasets/training"):
        os.mkdir(project_dirname + "/datasets/training")

    # make logger file
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    snapshot_path = project_dirname + "/model/DRIVE/" + "{}_{}labels_beta_{}_scaling_{}".format(current_time, LABELNUM, BETA, SCALING)
    snapshot_path += "withAug/" if WITH_AUG else "/"
    
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
    if device == torch.device("cuda:0"):
        torch.cuda.empty_cache()

    # Initialize lists to store validation losses for each fold
    val_losses = []
    # Define the number of folds for cross-validation
    # kf = KFold(n_splits=5, shuffle=True)
    # fold = 0

    # for train_index, val_index in kf.split(db_train):
        # fold += 1
        # logging.info(f"Fold {fold}/{kf.get_n_splits()}")

    logging.info(f"train labeled indices = {train_labeled_idxs}")
    logging.info(f"validation labeled indices = {val_labeled_idxs}")
    logging.info(f"train unlabeled indices = {train_unlabeled_idxs}")
    logging.info(f"validation unlabeled indices = {val_unlabeled_idxs}")

    model = MyNet(n_channels=3, n_classes=NUM_CLASSES-1, normalization='batchnorm', has_dropout=True)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=0.0001)
    solver = Solver(model=model, device=device, train_dataloader=train_loader, val_dataloader=val_loader, 
                    optimizer=optimizer, learning_rate=LR, beta=BETA, consistency_rampup=CONSISTENCY_RAMPUP, 
                    labeled_bs=LABELED_BS, max_iterations=MAX_ITERATIONS, model_path=snapshot_path, #+f"/fold{fold}",
                    print_every=PRINT_EVERY, save_every=SAVE_EVERY,scaling=SCALING)
    # trained_mynet = model_train(model, train_loader, val_loader, model_path=snapshot_path+f"/fold{fold}")
    solver.train()

    # Calculate and print average validation loss across folds
    # avg_val_loss = np.mean(val_losses)
    # logging.info(f"Average validation loss across folds: {avg_val_loss:.4f}")