# set parameters for training
import argparse
from datetime import datetime
import os

CLDICE = "clDice"
BL = "boundary_loss"

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str,
                    default="/model/DCA1/", help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=4,
                    help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=1e-2,
                    help='base learning rate')
parser.add_argument('--labelnum', type=int,  default=20, help='number of labels')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--scaling', type=int,  default=-1500, help='scaling factor')
parser.add_argument('--consistency_weight', type=float,  default=0.1,
                    help='balance factor to control supervised loss and consistency loss')
parser.add_argument('--beta', type=float,  default=0.3,
                    help='balance factor to control regional and sdm loss')
parser.add_argument('--cldice_alpha', type=float,  default=0.3,
                    help='soft-dice-clDice\'s alpha')
parser.add_argument('--k', type=int,  default=10,
                    help='number of iterations for softSkeletonize')
parser.add_argument('--bl_alpha', type=float,  default=0.1,
                    help='boundary loss\'s alpha')
parser.add_argument('--consistency_rampup', type=float,
                    default=40.0, help='consistency_rampup')
parser.add_argument('--contribution', type=str,
                    default="None", help='the contribution you want to test clDice or boundary_loss')
args = parser.parse_args()

class Parameters(object):
    def __init__(self):
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
        self.cldice_alpha = args.cldice_alpha
        self.cldice_k = args.k
        self.bl_alpha = args.bl_alpha

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
