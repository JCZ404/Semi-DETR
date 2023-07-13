# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F

from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmdet.core.bbox.match_costs.builder import MATCH_COST

@MATCH_COST.register_module()
class SoftmaxFocalLossCost:
    """FocalLossCost.

     Args:
         weight (int | float, optional): loss_weight
         gamma (int | float, optional): focal_loss gamma
         eps (float, optional): default 1e-12

     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import SoftmaxFocalLossCost
         >>> import torch
         >>> self = SoftmaxFocalLossCost()
         >>> cls_pred = torch.rand(4, 3)
         >>> gt_labels = torch.tensor([0, 1, 2])
         >>> factor = torch.tensor([10, 8, 10, 8])
         >>> self(cls_pred, gt_labels)
         tensor([[-0.3236, -0.3364, -0.2699],
                [-0.3439, -0.3209, -0.4807],
                [-0.4099, -0.3795, -0.2929],
                [-0.1950, -0.1207, -0.2626]])
    """

    def __init__(self, weight=1., gamma=1.5, eps=1e-12):
        self.weight = weight
        self.gamma = gamma
        self.eps = eps

    def __call__(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        # import ipdb;ipdb.set_trace()
        # focal loss
        cls_score = cls_pred.softmax(-1)
        cls_cost = -cls_score[:, gt_labels]
        return cls_cost * self.weight



@MATCH_COST.register_module()
class SoftFocalLossCost:
    """FocalLossCost.

     Args:
         weight (int | float, optional): loss_weight
         alpha (int | float, optional): focal_loss alpha
         gamma (int | float, optional): focal_loss gamma
         eps (float, optional): default 1e-12

     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import FocalLossCost
         >>> import torch
         >>> self = FocalLossCost()
         >>> cls_pred = torch.rand(4, 3)
         >>> gt_labels = torch.tensor([0, 1, 2])
         >>> factor = torch.tensor([10, 8, 10, 8])
         >>> self(cls_pred, gt_labels)
         tensor([[-0.3236, -0.3364, -0.2699],
                [-0.3439, -0.3209, -0.4807],
                [-0.4099, -0.3795, -0.2929],
                [-0.1950, -0.1207, -0.2626]])
    """

    def __init__(self, weight=1., alpha=0.25, gamma=2, eps=1e-12, soft_option=0):
        self.soft_option = soft_option
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def __call__(self, cls_pred, gt_labels, gt_scores=None):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        if gt_scores is None:
            cls_pred = cls_pred.sigmoid()
            neg_cost = -(1 - cls_pred + self.eps).log() * (
                1 - self.alpha) * cls_pred.pow(self.gamma)
            pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
                1 - cls_pred).pow(self.gamma)
            cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
            return cls_cost * self.weight
        else:
            prob = cls_pred.sigmoid()   # [N, num_class]
            # 将标量列表表示的label转换为one-hot向量，并去除最后一维的背景类
            one_hot_label = prob.new_zeros(len(gt_labels), len(prob[0])).scatter_(1, gt_labels.unsqueeze(1), 1)    # [num_gt, num_class]
            # 其中soft_label的一行要么只有一个label被激活，要么全为0
            soft_label = gt_scores.unsqueeze(-1) * one_hot_label   # [num_gt, num_class] with soft score as the classification label
            # import ipdb;ipdb.set_trace()
            # 将对应的预测值和target值做成多维的
            num_pred = prob.size(0)
            num_gt = soft_label.size(0)
            prob = prob[:, None, :].repeat(1, num_gt, 1)                    # [N, 1, num_class]
            soft_label = soft_label[None, :, :].repeat(num_pred, 1, 1)      # [1, num_gt, num_class]
            neg_cost = -(1 - prob + self.eps).log() * (1 - soft_label) * torch.pow(soft_label, self.gamma)
            pos_cost = -(prob + self.eps).log() * soft_label * torch.pow(torch.abs(soft_label - prob), self.gamma)
            if self.soft_option == 0:
                neg_cost = neg_cost.sum(dim=-1)
                pos_cost = pos_cost.sum(dim=-1)
                # 参考quality focal loss，计算soft label的focal loss
                cls_cost =  pos_cost - neg_cost
                # 用avg_factor来计算平均值
                return cls_cost * self.weight
            else:
                cls_cost = pos_cost - neg_cost
                return cls_cost[:, torch.arange(num_gt), gt_labels] * self.weight


@MATCH_COST.register_module()
class KLDivCost:
    """KL Divergence based cost calculation.
    """

    def __init__(self, weight=1., eps=1e-12):
        self.weight = weight
       
        self.eps = eps

    def __call__(self, cls_pred, gt_label, gt_scores=None):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_labels (Tensor): [num_gt]
            gt_scores (Tensor): [num_gt, num_class]
        Returns:
            torch.Tensor: cls_cost value with weight
        """
        prob = cls_pred.sigmoid()   # [N, num_class]

             
        
        # 将对应的预测值和target值做成多维的
        num_pred = prob.size(0)
        num_gt = gt_scores.size(0)
        # 取出对应gt_label的score
        tgt_scores = gt_scores[torch.arange(num_gt), gt_label]          # [num_gt]

        prob = prob[:, None, :].repeat(1, num_gt, 1)                    # [N, 1, num_class]
        gt_scores = gt_scores[None, :, :].repeat(num_pred, 1, 1)        # [1, num_gt, num_class]

        # 将每个dim=1的维度的每个元素视为是一个二维的分布，计算对应的KL Divergence  
        pos_cost = (gt_scores / (prob + self.eps) + self.eps).log() * gt_scores                  # [num_pred, num_gt, num_class]
        neg_cost = ((1 - gt_scores) / (1 - prob + self.eps) + self.eps).log() * (1 - gt_scores)  # [num_pred, num_gt, num_class]

        # 注意还要乘以对应的gt label处的score来缩放一下cost，避免出现不同的pseudo bbox由于实际是
        # 同一个位置的框而有相同的score vector
        cls_cost =  pos_cost.sum(dim=-1) + neg_cost.sum(dim=-1)                         # [num_pred, num_gt]
        cls_cost = cls_cost * tgt_scores[None, :].repeat(num_pred, 1)
        # 用avg_factor来计算平均值
        return cls_cost * self.weight