import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
import os
import logging
from tensorboardX import SummaryWriter
from torch.nn import BCELoss, MSELoss
from torchvision.utils import make_grid
from parameters import *
from utils_dtc.losses import BoundaryLoss, dice_loss, cl_dice_loss, soft_dice_cldice
from utils_dtc.losses_2 import compute_sdf
from utils_dtc.ramps import sigmoid_rampup

class Solver(object):
    """
    A Solver encapsulates all the logic necessary for training.

    """

    def __init__(self, parameters: Parameters, model:nn.Module, train_dataloader, val_dataloader, device,
                 optimizer: torch.optim.Optimizer, verbose=True, **kwargs):
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

        self.base_learning_rate = parameters.lr
        self.labeled_bs = parameters.labeled_bs
        self.consistency_rampup = parameters.consistency_rampup
        self.max_iterations = parameters.max_iterations
        self.beta = parameters.beta
        self.scaling = parameters.scaling
        self.cldice_alpha = parameters.cldice_alpha
        self.k = parameters.cldice_k
        self.bl_alpha = parameters.bl_alpha

        self.opt = optimizer
        self.ce_loss = BCELoss()
        self.mse_loss = MSELoss()
        self.boundary_loss = BoundaryLoss(idc=[0]) #originally [1]
        

        self.model_path = parameters.snapshot_path
        self.verbose = verbose
        self.print_every = parameters.print_every
        self.writer = SummaryWriter(self.model_path+'/log')
        self.save_every = parameters.save_every
        
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        self.current_iter = 0
        self.current_lr = self.base_learning_rate

        self.contribution = parameters.contribution
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
        self.current_iter = 0


    def _forward(self, batch, validation):
        """
        Forward pass through the model. Updates metric object with current stats.

        Args:
            batch (dict): Input data batch.
            metrics: Metrics object for tracking.
        """
        def get_current_consistency_weight(epoch):
            # Consistency ramp-up from https://arxiv.org/abs/1610.02242
            return sigmoid_rampup(epoch, self.consistency_rampup)
        loss = None
        # Unpack data
        bat_img, bat_label = batch['image'], batch['label']
        bat_img, bat_label = bat_img.to(self.device), bat_label.to(self.device)
        # Forward pass
        bat_pred_tanh, bat_pred = self.model(bat_img)
        bat_pred_soft = torch.sigmoid(bat_pred)
        # Compute loss
        with torch.no_grad():
                gt_dis = compute_sdf(bat_label[:].cpu().numpy(), bat_pred[:self.labeled_bs, 0, ...].shape)
                gt_dis = torch.from_numpy(gt_dis).float().to(self.device)
                # gt_dis_soft = torch.softmax(gt_dis, dim=1)
            
        loss_sdf = self.mse_loss(bat_pred_tanh[:self.labeled_bs, 0, ...], gt_dis)  
        
        # skel_pred, skel_true, loss_dice, bl_loss = None
        # cldice
        if self.contribution == CLDICE:
            skel_pred, skel_true, loss_dice = soft_dice_cldice(bat_pred_soft[:self.labeled_bs], bat_label[:self.labeled_bs].float(), self.cldice_alpha, self.k)
            bl_loss = 0.0
        # boundary loss
        elif self.contribution == BL:
            bat_pred_soft2 = torch.softmax(bat_pred, dim=1)
            bl_loss = self.boundary_loss(bat_pred_soft2[:self.labeled_bs], gt_dis)
            loss_dice = dice_loss(bat_pred_soft[:self.labeled_bs], bat_label[:self.labeled_bs] == 1)
        else:
            loss_dice = dice_loss(bat_pred_soft[:self.labeled_bs], bat_label[:self.labeled_bs] == 1)
            bl_loss = 0.0
        
        dis_to_mask = torch.sigmoid(self.scaling * bat_pred_tanh)
        consistency_loss = torch.mean((dis_to_mask - bat_pred_soft) ** 2)
        supervised_loss = loss_dice + self.beta * (loss_sdf + self.bl_alpha * bl_loss)
        # supervised_loss = loss_dice + self.beta * loss_sdf
        
        consistency_weight = get_current_consistency_weight((self.current_iter) // 150)
        loss = supervised_loss + consistency_weight * consistency_loss

        # write to TensorBoard
        segment = 'train' if not validation else 'val'
        self.writer.add_scalar('lr', self.current_lr, self.current_iter)
        self.writer.add_scalar(f'consistency_weight', consistency_weight, self.current_iter)
        self.writer.add_scalar(f'{segment}_loss/loss', loss.item(), self.current_iter)
        self.writer.add_scalar(f'{segment}_loss/loss_dice', loss_dice.item(), self.current_iter)
        self.writer.add_scalar(f'{segment}_loss/loss_hausdorff', loss_sdf.item(), self.current_iter)
        if self.contribution == BL:
            self.writer.add_scalar(f'{segment}_loss/loss_boundary', bl_loss.item(), self.current_iter)
        self.writer.add_scalar(f'{segment}_loss/consistency_loss', consistency_loss.item(), self.current_iter)
        # if self.current_iter % self.save_every == 0:
        self.save_tb_images_training(bat_img[0, :, :, :],
                    bat_pred_soft[0, :, :, :],
                    bat_label[0, :, :],
                    dis_to_mask[0, 0:1, :, :],
                    bat_pred_tanh[0, 0:1, :, :],
                    gt_dis[0, :, :], 
                    skel_pred[0, ...] if self.contribution == CLDICE else None,
                    skel_true[0, ...] if self.contribution == CLDICE else None,
                    self.current_iter, segment)
        logging.info('iteration %d : loss : %f, loss_consis: %f, loss_haus: %f, loss_dice: %f' %
            (self.current_iter, loss.item(), consistency_loss.item(), loss_sdf.item(),
             loss_dice.item()))
        # logging.info('iteration %d : loss : %f, loss_consis: %f, loss_haus: %f, loss_dice: %f, loss_boundary: %f' %
        #     (self.current_iter, loss.item(), consistency_loss.item(), loss_sdf.item(),
        #      loss_dice.item(), bl_loss.item()))
         
        # change lr
        if self.current_iter % 1000 == 0:
            self.current_lr = self.base_learning_rate * 0.1 ** (self.current_iter // 2500)
            for param_group in self.opt.param_groups:
                param_group['lr'] = self.current_lr
        if self.current_iter % self.save_every == 0:
            save_model_path = os.path.join(self.model_path, 'iter_' + str(self.current_iter) + '.pth')
            torch.save(self.model.state_dict(), save_model_path)
            logging.info("save model to {}".format(save_model_path))

        return loss

    def _train_loop(self):
        """
        Executes the training loop.

        Handles the iteration over training data batches and performs backpropagation.
        """
        self.model.train()
        self.opt.zero_grad()
        self.empty_cache()
        train_epoch_loss = 0
        # Iterate over all training samples
        for batch in self.train_dataloader:
                self.current_iter += 1
                # Update the model parameters.
                loss = self._forward(batch, validation=False)
                # Compute gradients
                loss.backward()
                # Update weights
                self.opt.step()
                
                self.train_batch_loss.append(loss)
                train_epoch_loss += loss

                if self.current_iter >= self.max_iterations:
                        break    
                
        self.empty_cache()
        return train_epoch_loss

    def _eval_loop(self):
        """
        Executes the evaluation loop.

        Handles the iteration over validation data batches for evaluation purposes.
        """
        self.model.eval()
        self.empty_cache()
        val_epoch_loss = 0

        with torch.no_grad():
            # Iterate over all validation samples
            for batch in self.val_dataloader:
                self.current_iter += 1

                # Update the model parameters.
                loss = self._forward(batch, validation=True)
                
                self.val_batch_loss.append(loss)
                val_epoch_loss += loss
        
        self.empty_cache()
        return val_epoch_loss
                        
    # def train(self, patience = None):
    def train(self):
        """
        Run optimization to train the model.
        """
        max_epoch = self.max_iterations//(len(self.train_dataloader) + len(self.val_dataloader))+1
        iterator = tqdm(range(max_epoch), ncols=70)
        
        # Start an epoch
        for epoch in iterator:
            logging.info("\n[+] ====== Start training... epoch {} ======".format(epoch + 1))
            # print(torch.cuda.memory_summary(device=None, abbreviated=False))
            
            train_epoch_loss = self._train_loop()
            train_epoch_loss /= len(self.train_dataloader)    

            val_epoch_loss = self._eval_loop()
            val_epoch_loss /= len(self.val_dataloader)  

            # Record the losses for later inspection.
            self.train_loss_history.append(train_epoch_loss)
            self.val_loss_history.append(val_epoch_loss)

            if self.verbose and epoch % self.print_every == 0:
                logging.info('(Epoch %d / %d) train loss: %f; val loss: %f' % (
                    epoch + 1, len(iterator), train_epoch_loss.item(), val_epoch_loss.item()))
            
            self.writer.add_scalar(f'train_loss/epoch_loss', train_epoch_loss.item(), self.current_iter)
            self.writer.add_scalar(f'val_loss/epoch_loss', val_epoch_loss.item(), self.current_iter)
            
            # Keep track of the best model
            self.update_best_loss(val_epoch_loss, train_epoch_loss, self.current_iter)
            # if patience and self.current_patience >= patience:
            #     logging.info("Stopping early at epoch {}!".format(epoch))
            #     break
            if self.current_iter >= self.max_iterations:
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
                    skel_pred, skel_tru,
                    iteration, segment):
        grid_image = make_grid(img, 1, normalize=True)
        self.writer.add_image(f'{segment}_vis/Image', grid_image, iteration)

        grid_image = make_grid(prediction, 1, normalize=False)
        self.writer.add_image(f'{segment}_vis/Predicted_label', grid_image, iteration)

        grid_image = make_grid(label, 1, normalize=False)
        self.writer.add_image(f'{segment}_vis/Groundtruth_label',
                            grid_image, iteration)

        grid_image = make_grid(dis_to_mask, 1, normalize=False)
        self.writer.add_image(f'{segment}_vis/Dis2Mask', grid_image, iteration)

        grid_image = make_grid(pred_dist_map, 1, normalize=False)
        self.writer.add_image(f'{segment}_vis/DistMap', grid_image, iteration)
            
        grid_image = make_grid(gt_distmap, 1, normalize=False)
        self.writer.add_image(f'{segment}_vis/Groundtruth_DistMap',
                            grid_image, iteration)
        
        if skel_pred != None:
            grid_image = make_grid(skel_pred, 1, normalize=False)
            self.writer.add_image(f'{segment}_vis/pred_skel',
                                grid_image, iteration)
        if skel_tru != None:
            grid_image = make_grid(skel_tru, 1, normalize=False)
            self.writer.add_image(f'{segment}_vis/GT_skel',
                                grid_image, iteration)

    def empty_cache(self):
        """
        Empties the GPU cache if it is used.
        """

        if self.device == torch.device("cuda:0"):
            torch.cuda.empty_cache()
