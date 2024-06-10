from typing import Iterable, List, Set, cast
import torch
from torch.nn import functional as F
from torch import Tensor, einsum
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


# dice

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

# clDice
class SoftSkeletonize(torch.nn.Module):

    def __init__(self, num_iter=40):

        super(SoftSkeletonize, self).__init__()
        self.num_iter = num_iter

    def soft_erode(self, img):
        if len(img.shape)==4:
            p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
            p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
            return torch.min(p1,p2)
        elif len(img.shape)==5:
            p1 = -F.max_pool3d(-img,(3,1,1),(1,1,1),(1,0,0))
            p2 = -F.max_pool3d(-img,(1,3,1),(1,1,1),(0,1,0))
            p3 = -F.max_pool3d(-img,(1,1,3),(1,1,1),(0,0,1))
            return torch.min(torch.min(p1, p2), p3)

    def soft_dilate(self, img):

        if len(img.shape)==4:
            return F.max_pool2d(img, (3,3), (1,1), (1,1))
        elif len(img.shape)==5:
            return F.max_pool3d(img,(3,3,3),(1,1,1),(1,1,1))

    def soft_open(self, img):
        
        return self.soft_dilate(self.soft_erode(img))

    def soft_skel(self, img):

        img1 = self.soft_open(img)
        skel = F.relu(img-img1)

        for _ in range(self.num_iter):
            img = self.soft_erode(img)
            img1 = self.soft_open(img)
            delta = F.relu(img-img1)
            skel = skel + F.relu(delta - skel * delta)

        return skel

    def forward(self, img):

        return self.soft_skel(img)

def cl_dice_loss(y_pred, y_true):
    # exclude_background = True
    soft_skeletonize = SoftSkeletonize(num_iter=10)
    smooth = 1.
    # if exclude_background:
    #     y_true = y_true[:, 1:, :, :]
    #     y_pred = y_pred[:, 1:, :, :]
    skel_pred = soft_skeletonize(y_pred)
    skel_true = soft_skeletonize(y_true)
    tprec = (torch.sum(torch.multiply(skel_pred, y_true))+smooth)/(torch.sum(skel_pred)+smooth)    
    tsens = (torch.sum(torch.multiply(skel_true, y_pred))+smooth)/(torch.sum(skel_true)+smooth)    
    cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
    return cl_dice

def soft_dice(y_pred, y_true):
    """[function to compute dice loss]

    Args:
        y_true ([float32]): [ground truth image]
        y_pred ([float32]): [predicted image]

    Returns:
        [float32]: [loss value]
    """
    smooth = 1
    intersection = torch.sum((y_true * y_pred))
    coeff = (2. *  intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)
    return (1. - coeff)

def soft_dice_cldice(y_pred, y_true, alpha=0.3, k=10):
    soft_skeletonize = SoftSkeletonize(num_iter=k)
    smooth = 1.
    # if self.exclude_background:
    #     y_true = y_true[:, 1:, :, :]
    #     y_pred = y_pred[:, 1:, :, :]

    dice = soft_dice(y_pred, y_true)
    skel_pred = soft_skeletonize(y_pred)
    skel_true = soft_skeletonize(y_true)
    tprec = (torch.sum(torch.multiply(skel_pred, y_true))+smooth)/(torch.sum(skel_pred)+smooth)    
    tsens = (torch.sum(torch.multiply(skel_true, y_pred))+smooth)/(torch.sum(skel_true)+smooth)    
    cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
    return skel_pred, skel_true, (1.0-alpha)*dice+alpha*cl_dice

# boundary loss

# Assert utils
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())

def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)

def simplex(t: Tensor, axis=1) -> bool:
    _sum = cast(Tensor, t.sum(axis).type(torch.float32))
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)

def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])

class SurfaceLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, dist_maps: Tensor) -> Tensor:
        assert simplex(probs)
        assert not one_hot(dist_maps)

        pc = probs.type(torch.float32)
        dc = dist_maps[None, ...].type(torch.float32)

        # original code
        # pc = probs[:, self.idc, ...].type(torch.float32)
        # dc = dist_maps[:, self.idc, ...].type(torch.float32)

        multipled = einsum("bkwh,bkwh->bkwh", pc, dc)

        loss = multipled.mean()

        return loss

BoundaryLoss = SurfaceLoss