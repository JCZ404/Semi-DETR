
# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weight_reduce_loss
from mmdet.models.utils.transformer import inverse_sigmoid



@LOSSES.register_module()
class BinaryKLDivLoss(nn.Module):      
    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 class_weight=None,
                 use_sigmoid=False,
                 eps=1e-12):
      
        super(BinaryKLDivLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.use_sigmoid = use_sigmoid
        self.eps = eps

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                binary=False):
        """
        pred:   [num_bbox, num_class]
        target: [num_bbox, num_class]
        weight: [num_bbox]
        """
        # import ipdb;ipdb.set_trace()
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        # apply the class_weight
        if self.class_weight is not None:
            class_weight = pred.new_tensor(
                self.class_weight, device=pred.device)
        else:
            class_weight = None

        if binary:
           
            # JS divergence loss as discrite binary distribution
            pred_probs_1 = pred.sigmoid()   # [N, num_class]
            pred_probs_2 = 1 - pred_probs_1
            pred_probs = torch.stack((pred_probs_1, pred_probs_2), dim=-1)          # [N, num_class, 2]

            target_probs_1 = target
            target_probs_2 = 1 - target_probs_1 
            target_probs = torch.stack((target_probs_1, target_probs_2), dim=-1)    # [N, num_class, 2] 

            m = 0.5 * (pred_probs + target_probs)

            loss = 0.0
            loss_1 = F.kl_div((pred_probs + 1e-12).log(), m, reduction='none')      # [N, num_class, 2]
            loss_2 = F.kl_div((target_probs + 1e-12).log(), m, reduction='none')    # [N, num_class, 2]

            loss = (loss_1.sum(-1).sum(-1) + loss_2.sum(-1).sum(-1)) * 0.5          # [N,]
        else:
            # KL divergence loss 
            pred_probs = pred.sigmoid()
            loss = F.kl_div((pred_probs + 1e-12).log(), target, reduction='none').sum(-1)

        if weight is not None:
            loss = weight_reduce_loss(loss, weight.reshape(-1, 1), reduction, avg_factor)
        else:
            loss = weight_reduce_loss(loss, weight, reduction, avg_factor)

        return loss


if __name__ == "__main__":
    BKLDivLoss = BinaryKLDivLoss(use_sigmoid=True)
    data = torch.tensor([[0.2, 0.2, 0.3], [0.7, 0.8, 0.9]])
    pred = inverse_sigmoid(data)
    target = torch.tensor([[0.3, 0.2, 0.2], [0.7, 0.8, 0.9]])
 
    loss = BKLDivLoss(pred, target)
    print(loss)