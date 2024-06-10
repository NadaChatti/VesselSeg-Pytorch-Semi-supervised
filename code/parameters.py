# set parameters for training
import argparse
from datetime import datetime
import os

CLDICE = "clDice"
BL = "boundary_loss"


class Parameters(object):
    def __init__(self, args):
        self.imagenum = 100
        self.labelnum = args.labelnum

        self.lr = args.base_lr
        self.max_iterations = args.max_iterations
        self.batch_size = args.batch_size
        self.labeled_bs = args.labeled_bs
        self.seed = args.seed
        self.beta = args.beta
        self.scaling = args.scaling
        self.consistency_rampup = args.consistency_rampup
        self.cldice_alpha = args.alpha
        self.cldice_k = args.k
        self.contribution = args.contribution

        self.save_every = 100
        self.print_every = 1  

        self.patch_size = (3, 128, 128)
        self.num_classes = 2
        self.with_aug = True
        
        self.project_dirname = os.path.join(os.path.dirname(__file__), "..")
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.snapshot_path = self.project_dirname + args.exp 
        if self.contribution == CLDICE or self.contribution == BL:
            self.snapshot_path += self.contribution + "/" + current_time
        else: 
            self.snapshot_path += current_time

        # self.snapshot_path += "withAug/" if self.with_aug else "/"
        print(self.snapshot_path)
        # self.snapshot_path = os.path.join(self.project_dirname, args.exp + current_time)
