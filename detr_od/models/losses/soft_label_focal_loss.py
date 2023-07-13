import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weight_reduce_loss


ESP = 1e-4
ONE = 0.9999


def _weight_loss(weight, loss, reduction, avg_factor):
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                weight = weight.view(-1, 1)
            else:
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module()
class FocalKLLoss(nn.Module):
    def __init__(self,
                 use_sigmoid=True,
                 reduction='mean',
                 gamma=0.5,
                 loss_weight=1.0):
        """
        The combination of KL Loss: https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
            and focal Loss <https://arxiv.org/abs/1708.02002>`_

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.gamma = gamma

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
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.use_sigmoid:
            # Compute KL loss
            logp = F.logsigmoid(pred)
            Loss_p = F.kl_div(logp, target, reduction='none')
            logp_ = -pred + logp
            Loss_n = F.kl_div(logp_, 1 - target, reduction='none')
            input_sigmoid = pred.sigmoid()

        else:
            # pred is already sigmoid
            pred = torch.clamp(pred, min=ESP, max=ONE)
            target = torch.clamp(target, min=ESP, max=ONE)
            logp_pred_p = torch.log(pred)
            Loss_p = F.kl_div(logp_pred_p, target, reduction='none')
            logp_pred_n = torch.log(1-pred)
            Loss_n = F.kl_div(logp_pred_n, 1 - target, reduction='none')
            input_sigmoid = pred

        loss = Loss_p + Loss_n
        # Focal term
        focal_term = ((target - input_sigmoid) ** 2 + 1e-6) ** (
            0.5 * self.gamma)  # new way to compute **2, smoothing loss
        loss = focal_term * loss
        loss = _weight_loss(weight, loss, reduction, avg_factor)
        loss = loss * self.loss_weight

        return loss
