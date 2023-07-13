# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet.core.bbox.builder import BBOX_SAMPLERS
from mmdet.core.bbox.samplers.base_sampler import BaseSampler
from .double_sampling_result import DoubleSamplingResult


@BBOX_SAMPLERS.register_module()
class DoubleSampler(BaseSampler):
    """A pseudo sampler that does not do sampling actually."""

    def __init__(self, **kwargs):
        pass

    def _sample_pos(self, **kwargs):
        """Sample positive samples."""
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        """Sample negative samples."""
        raise NotImplementedError

    def sample(self, assign_result, bboxes, gt_bboxes, **kwargs):
        """Directly returns the positive and negative indices  of samples.

        Args:
            assign_result (:obj:`AssignResult`): Assigned results
            bboxes (torch.Tensor): Bounding boxes
            gt_bboxes (torch.Tensor): Ground truth boxes

        Returns:
            :obj:`SamplingResult`: sampler results
        """
        pos_inds_1 = torch.nonzero(
            assign_result.gt_inds_1 > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds_1 = torch.nonzero(
            assign_result.gt_inds_1 == 0, as_tuple=False).squeeze(-1).unique()

        pos_inds_2 = torch.nonzero(
            assign_result.gt_inds_2 > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds_2 = torch.nonzero(
            assign_result.gt_inds_2 == 0, as_tuple=False).squeeze(-1).unique()
        gt_flags = bboxes.new_zeros(bboxes.shape[0], dtype=torch.uint8)
        sampling_result = DoubleSamplingResult(pos_inds_1, neg_inds_1, 
                                               pos_inds_2, neg_inds_2,
                                               bboxes, gt_bboxes,
                                               assign_result, gt_flags)
        return sampling_result
