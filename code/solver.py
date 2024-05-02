import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
import os
import logging
from tensorboardX import SummaryWriter
from torch.nn import BCELoss, MSELoss
from torchvision.utils import make_grid

from utils_dtc.losses import dice_loss
from utils_dtc.losses_2 import compute_sdf
from utils_dtc.ramps import sigmoid_rampup

class Solver(object):
    """
    A Solver encapsulates all the logic necessary for training.

    """

    def __init__(self, model, train_dataloader, val_dataloader, device,
                 optimizer: torch.optim.Optimizer, learning_rate=1e-3, beta=0.3, 
                 labeled_bs=4, scaling=1500, consistency_rampup=40.0,
                 max_iterations=6000,
                 verbose=True, print_every=1, save_every=100, model_path= None,
                 **kwargs):
        """
        Construct a new Solver instance.

        Required arguments:
        - model: A model object conforming to the API described above

        - train_dataloader: A generator object returning training data
        - val_dataloader: A generator object returning validation data

        - loss_func: Loss function object.
        - learning_rate: Float, learning rate used for gradient descent.

        - optimizer: The optimizer specifying the update rule

        Optional arguments:
        - verbose: Boolean; if set to false then no output will be printed during
          training.
        - print_every: Integer; training losses will be printed every print_every
          iterations.
        """
        self.model = model
        self.device = device

        self.learning_rate = learning_rate
        self.beta = beta
        self.scaling = scaling
        self.labeled_bs = labeled_bs
        self.consistency_rampup = consistency_rampup
        self.max_iterations = max_iterations

        self.opt = optimizer
        self.ce_loss = BCELoss()
        self.mse_loss = MSELoss()

        self.model_path = model_path
        self.verbose = verbose
        self.print_every = print_every
        self.writer = SummaryWriter(model_path+'/log')
        self.save_every = save_every
        
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        self.current_patience = 0

        self._reset()

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.best_model_stats = None
        self.best_params = None

        self.train_loss_history = []
        self.val_loss_history = []

        self.train_batch_loss = []
        self.val_batch_loss = []

        self.num_operation = 0
        self.current_patience = 0

    def _step(self, bat_img, bat_label, iter_num, validation=False):
        def get_current_consistency_weight(epoch):
            # Consistency ramp-up from https://arxiv.org/abs/1610.02242
            return sigmoid_rampup(epoch, self.consistency_rampup)
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.

        :param bat_img: batch of training features
        :param bat_label: batch of corresponding training labels
        :param validation: Boolean indicating whether this is a training or
            validation step

        :return loss: Loss between the model prediction for X and the target
            labels y
        """
        loss = None

        # Forward pass
        bat_pred_tanh, bat_pred = self.model(bat_img)
        bat_pred_soft = torch.sigmoid(bat_pred)
        # Compute loss
        with torch.no_grad():
                gt_dis = compute_sdf(bat_label[:].cpu().numpy(), bat_pred[:self.labeled_bs, 0, ...].shape)
                gt_dis = torch.from_numpy(gt_dis).float().to(self.device)
            
        loss_sdf = self.mse_loss(bat_pred_tanh[:self.labeled_bs, 0, ...], gt_dis)    
        loss_seg = self.ce_loss(bat_pred_soft[:self.labeled_bs], bat_label[:self.labeled_bs].float())
        loss_seg_dice = dice_loss(bat_pred_soft[:self.labeled_bs], bat_label[:self.labeled_bs] == 1)
        
        dis_to_mask = torch.sigmoid(self.scaling * bat_pred_tanh)
        consistency_loss = torch.mean((dis_to_mask - bat_pred_soft) ** 2)
        supervised_loss = loss_seg_dice + self.beta * loss_sdf
        
        consistency_weight = get_current_consistency_weight((iter_num - 1) // 150)
        loss = supervised_loss + consistency_weight * consistency_loss
        # Add the regularization
        # loss += sum(self.model.reg.values())

        # Count number of operations
        # self.num_operation += self.model.num_operation

        # Perform gradient update (only in train mode)
        if not validation:
            self.opt.zero_grad()
            # Compute gradients
            loss.backward()
            # Update weights
            self.opt.step()
            # If it was a training step, we need to count operations for
            # backpropagation as well
            # self.num_operation += self.model.num_operation

        # write to TensorBoard
        segment = 'train' if not validation else 'val'
        self.writer.add_scalar(f'{segment}/loss', loss, iter_num)
        self.writer.add_scalar(f'{segment}/loss_dice', loss_seg_dice, iter_num)
        self.writer.add_scalar(f'{segment}/loss_hausdorff', loss_sdf, iter_num)
        self.writer.add_scalar(f'{segment}/consistency_weight', consistency_weight, iter_num)
        self.writer.add_scalar(f'{segment}/consistency_loss', consistency_loss, iter_num)
        self.save_tb_images_training(bat_img[0, :, :, :],
                    bat_pred_soft[0, :, :, :],
                    bat_label[0, :, :],
                    dis_to_mask[0, 0:1, :, :],
                    bat_pred_tanh[0, 0:1, :, :],
                    gt_dis[0, :, :], 
                    iter_num, validation)
        logging.info('iteration %d : loss : %f, loss_consis: %f, loss_haus: %f, loss_seg: %f, loss_dice: %f' %
            (iter_num, loss.item(), consistency_loss.item(), loss_sdf.item(),
            loss_seg.item(), loss_seg_dice.item()))
        return loss

    # def train(self, patience = None):
    def train(self):
        """
        Run optimization to train the model.
        """
        iter_num = 0
        max_epoch = self.max_iterations//(len(self.train_dataloader) + len(self.val_dataloader))+1
        iterator = tqdm(range(max_epoch), ncols=70)
        # Start an epoch
        for epoch in iterator:
            logging.info("\n[+] ====== Start training... epoch {} ======".format(epoch + 1))
            # print(torch.cuda.memory_summary(device=None, abbreviated=False))
            # Iterate over all training samples
            train_epoch_loss = 0.0

            for batch in self.train_dataloader:
                iter_num += 1
                self.model.train()
                # Unpack data
                bat_img, bat_label = batch['image'], batch['label']
                bat_img, bat_label = bat_img.to(self.device), bat_label.to(self.device)

                # Update the model parameters.
                validate = False
                train_loss = self._step(bat_img, bat_label, iter_num, validation=validate)

                self.train_batch_loss.append(train_loss)
                train_epoch_loss += train_loss

                # change lr
                if iter_num % 150 == 0:
                    lr_ = self.learning_rate * 0.1 ** (iter_num // 2500)
                    for param_group in self.opt.param_groups:
                        param_group['lr'] = lr_
                if iter_num % self.save_every == 0:
                    save_model_path = os.path.join(self.model_path, 'iter_' + str(iter_num) + '.pth')
                    torch.save(self.model.state_dict(), save_model_path)
                    logging.info("save model to {}".format(save_model_path))
                if iter_num >= self.max_iterations:
                        break    
                
            
            train_epoch_loss /= len(self.train_dataloader)    
            
            # Iterate over all validation samples
            val_epoch_loss = 0.0

            for batch in self.val_dataloader:
                iter_num += 1
                self.model.eval()
                # Unpack data
                bat_img, bat_label = batch['image'], batch['label']
                bat_img, bat_label = bat_img.to(self.device), bat_label.to(self.device)

                # Compute Loss - no param update at validation time!
                val_loss = self._step(bat_img, bat_label, iter_num, validation=True)
                self.val_batch_loss.append(val_loss)
                val_epoch_loss += val_loss

            val_epoch_loss /= len(self.val_dataloader)

            # Record the losses for later inspection.
            self.train_loss_history.append(train_epoch_loss)
            self.val_loss_history.append(val_epoch_loss)

            if self.verbose and epoch % self.print_every == 0:
                logging.info('(Epoch %d / %d) train loss: %f; val loss: %f' % (
                    epoch + 1, len(iterator), train_epoch_loss.item(), val_epoch_loss.item()))

            # Keep track of the best model
            self.update_best_loss(val_epoch_loss, train_epoch_loss, iter_num)
            # if patience and self.current_patience >= patience:
            #     logging.info("Stopping early at epoch {}!".format(epoch))
            #     break
            if iter_num >= self.max_iterations:
                iterator.close()
                break
        # At the end of training swap the best params into the model
        # self.model.params = self.best_params
        self.writer.close()

    def get_dataset_accuracy(self, loader):
        correct = 0
        total = 0
        for batch in loader:
            X = batch['image']
            y = batch['label']
            y_pred = self.model.forward(X)
            label_pred = np.argmax(y_pred, axis=1)
            correct += sum(label_pred == y)
            if y.shape:
                total += y.shape[0]
            else:
                total += 1
        return correct / total

    def update_best_loss(self, val_loss, train_loss, iteration):
        # Update the model and best loss if we see improvements.
        if not self.best_model_stats or val_loss < self.best_model_stats["val_loss"]:
            self.best_model_stats = {"val_loss":val_loss.item(), "train_loss":train_loss.item()}
            self.best_params = self.model.parameters()
            logging.info(f"Best params at iteration {iteration} with stats {self.best_model_stats}")
        #     self.current_patience = 0
        # else:
        #     self.current_patience += 1

    def save_tb_images_training(self,img, prediction, label, 
                    dis_to_mask, pred_dist_map, gt_distmap, 
                    iteration, validation):
        segment = 'train' if not validation else 'val'
        grid_image = make_grid(img, 1, normalize=True)
        self.writer.add_image(f'{segment}/Image', grid_image, iteration)

        grid_image = make_grid(prediction, 1, normalize=False)
        self.writer.add_image(f'{segment}/Predicted_label', grid_image, iteration)

        grid_image = make_grid(label, 1, normalize=False)
        self.writer.add_image(f'{segment}/Groundtruth_label',
                            grid_image, iteration)

        grid_image = make_grid(dis_to_mask, 1, normalize=False)
        self.writer.add_image(f'{segment}/Dis2Mask', grid_image, iteration)

        grid_image = make_grid(pred_dist_map, 1, normalize=False)
        self.writer.add_image(f'{segment}/DistMap', grid_image, iteration)
            
        grid_image = make_grid(gt_distmap, 1, normalize=False)
        self.writer.add_image(f'{segment}/Groundtruth_DistMap',
                            grid_image, iteration)
