# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import LOSSES

@LOSSES.register_module()
class SoftmaxFocalLoss(nn.Module):

    def __init__(self,
                 gamma=1.5,
                 reduction='sum',
                 loss_weight=1.0,
                 class_weight=None,
                 use_sigmoid=False):
        """Exand the focal loss to support multi-calssification with softmax as the 
        activation function.
        Focal loss for binary classificaiton, there will have two parameters: gamma and alpha,
        but in the Focal loss for multi-class classification, there will only have the gamma as 
        the parameter.
        Args:
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.use_sigmoid = use_sigmoid

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        # import ipdb;ipdb.set_trace()
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        # apply the class_weight
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(
                self.class_weight, device=cls_score.device)
        else:
            class_weight = None
        # focal loss
        CE = F.cross_entropy(pred, target, weight=class_weight, reduction="none")
        p = torch.exp(-CE)
        loss = (1 - p) ** self.gamma * CE
        if reduction == "none":
            return self.loss_weight * loss
        elif reduction == "sum":
            return self.loss_weight * loss.sum()
