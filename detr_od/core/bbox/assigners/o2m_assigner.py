# Copyright (c) OpenMMLab. All rights reserved.
import torch

import torch.nn.functional as F

from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.core.bbox.match_costs import build_match_cost
from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy
from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner

from detr_ssod.utils import log_every_n, log_image_with_boxes
from .o2m_assign_result import O2MAssignResult


@BBOX_ASSIGNERS.register_module()
class O2MAssigner(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, regression L1 cost and regression iou cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
        bbox_weight (int | float, optional): The scale factor for regression
            L1 cost. Default 1.0.
        iou_weight (int | float, optional): The scale factor for regression
            iou cost. Default 1.0.
        iou_calculator (dict | optional): The config for the iou calculation.
            Default type `BboxOverlaps2D`.
        iou_mode (str | optional): "iou" (intersection over union), "iof"
                (intersection over foreground), or "giou" (generalized
                intersection over union). Default "giou".
    """

    def __init__(self, candidate_topk=13, debug=False):
        self.candidate_topk = candidate_topk
        self.debug = debug
        
    def assign(self,
               bbox_pred,
               cls_pred,
               gt_bboxes,
               gt_labels,
               img_meta,
               gt_bboxes_ignore=None,
               alpha=1,
               beta=6,
               teacher_assign=False,
               multiple_pos=False):
        """Computes one-to-many matching based on the weighted costs.

        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
            img_meta (dict): Meta information for current image.
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        INF = 100000000
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)
        gt_labels = gt_labels.long()
        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ),
                                              -1,
                                              dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)
        assign_metrics = bbox_pred.new_zeros((num_bboxes, ))

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = bbox_pred.new_zeros((num_bboxes, ))
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return O2MAssignResult(
                num_gts, assigned_gt_inds, max_overlaps, assign_metrics, labels=assigned_labels)
        
        img_h, img_w, _ = img_meta['img_shape']
        factor = gt_bboxes.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)

        # 计算predict box与gt box之间的alignment metric
        pred_bboxes = bbox_cxcywh_to_xyxy(bbox_pred) * factor       # [num_bbox,]
        scores = cls_pred                                           # [num_bbox, 80]
        overlaps = bbox_overlaps(pred_bboxes, gt_bboxes).detach()   # [num_bbox, num_gt]
        bbox_scores = scores[:, gt_labels].detach()
        alignment_metrics = bbox_scores ** alpha * overlaps ** beta # [num_bbox, num_gt]

        # the top-k aligned predicted candidate boxes as the potential positive samples
        if teacher_assign and not multiple_pos:
            # option-1: only the top-aligned candidate box is kept as the positive samples
            _, candidate_idxs = alignment_metrics.topk(                 # [top-k, num_gt]
            1, dim=0, largest=True)
        else:  
            _, candidate_idxs = alignment_metrics.topk(                 # [top-k, num_gt]
                self.candidate_topk, dim=0, largest=True)
        candidate_metrics = alignment_metrics[candidate_idxs, torch.arange(num_gts)]

        if teacher_assign and multiple_pos:
            # option-2: dynamic estimate the positive samples for contrastive
            is_pos = torch.zeros_like(candidate_metrics)
            topk_ious, _ = torch.topk(overlaps, self.candidate_topk, dim=0) # [top-k, num_gt]
            dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
            for gt_idx in range(num_gts):
                _, pos_idx = torch.topk(candidate_metrics[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=True)
                is_pos[:, gt_idx][pos_idx] = 1 
            is_pos = is_pos.bool()
        else:
            is_pos = candidate_metrics > 0

        # modify the index
        for gt_idx in range(num_gts):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        candidate_idxs = candidate_idxs.view(-1)

        # deal with a single candidate assigned to multiple gt_bboxes
        overlaps_inf = torch.full_like(overlaps,                    # [num_bbox x num_gt,]
                                       -INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]            # [num_gt x top-k]
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gts, -1).t()            # [num_gt, num_bbox] -> [num_bbox, num_gt]


        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        
        # assign the background class first
        assigned_gt_inds[:] = 0
        assigned_gt_inds[
            max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1
        assign_metrics[
            max_overlaps != -INF] = alignment_metrics[max_overlaps != -INF, argmax_overlaps[max_overlaps != -INF]]

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None
        
        return O2MAssignResult(
            num_gts, assigned_gt_inds, max_overlaps, assign_metrics, labels=assigned_labels)

    
    