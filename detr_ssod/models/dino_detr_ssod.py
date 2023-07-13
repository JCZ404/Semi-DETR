import time
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from mmcv.runner.fp16_utils import force_fp32
from mmcv.runner import get_dist_info

from mmdet.core import (bbox2roi, bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models import DETECTORS, build_detector
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models.builder import build_roi_extractor

from detr_ssod.models.multi_stream_detector import MultiSteamDetector
from detr_ssod.models.utils import Transform2D, filter_invalid_class_wise, concat_all_gather
from detr_ssod.utils import log_every_n, log_image_with_boxes
from detr_ssod.utils.structure_utils import dict_split, weighted_loss


try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

try:
    import sklearn.mixture as skm
except ImportError:
    skm = None


class Projector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.ac1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.ac2 = nn.ReLU()


        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=12544, out_features=1024, bias=True)
        self.fc_relu1 = nn.ReLU()

        self.bn = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(in_features=1024, out_features=256, bias=True)
        self.fc_relu2 = nn.ReLU()

    def forward(self, roi_feature_map):
        conv1_out = self.conv1(roi_feature_map)
        bn1_out = self.bn1(conv1_out)
        ac1_out = self.ac1(bn1_out)

        conv2_out = self.conv2(ac1_out)
        bn2_out = self.bn2(conv2_out)
        ac2_out = self.ac2(bn2_out)


        flat = self.flatten(ac2_out)
        fc1_out = self.fc1(flat)
        fc_bn1_out = self.bn(fc1_out)
        fc_relu1_out = self.fc_relu1(fc_bn1_out)

        fc2_out = self.fc2(fc_relu1_out)
        fc_relu2_out = self.fc_relu2(fc2_out)

        return fc_relu2_out


@DETECTORS.register_module()
class DinoDetrSSOD(MultiSteamDetector):
    def __init__(self, model: dict, train_cfg=None, test_cfg=None):
        super(DinoDetrSSOD, self).__init__(
            dict(teacher=build_detector(model), student=build_detector(model)),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        if train_cfg is not None:
            self.freeze("teacher")
            self.unsup_weight = self.train_cfg.unsup_weight

        self.covariance_type = 'diag'
        self.curr_step = 0

        # Build the roi-extractor
        # Note: here, the fine feature map scale is 8x, not 4x in Faster RCNN
        # in DINO, the feature map from backbone is [8x, 16x, 32x], and then 
        # add a extra conv on the last feature map to get a 64x feature similar
        # to the FPN structure
        # import ipdb;ipdb.set_trace()

        bbox_roi_extractor=dict(type='SingleRoIExtractor',
                                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                                out_channels=256,
                                featmap_strides=[8, 16, 32, 64])
        self.roi_extractor = build_roi_extractor(bbox_roi_extractor)

        # Build the projector
        # Note: use the projector to do the feature adaptator, we take the 
        # MLP with the (FC + BN + ReLU) structure, used to convert the feat
        # from backbone feat space to object query space
        feat_dims = self.student.bbox_head.embed_dims
        self.projector = Projector()

        self.eval_count = 0

    def forward_train(self, img, img_metas, **kwargs):
        # import ipdb;ipdb.set_trace()
        super().forward_train(img, img_metas, **kwargs)
        kwargs.update({"img": img})
        kwargs.update({"img_metas": img_metas})
        kwargs.update({"tag": [meta["tag"] for meta in img_metas]})
        data_groups = dict_split(kwargs, "tag")
        for _, v in data_groups.items():
            v.pop("tag")

        loss = {}
        # import ipdb;ipdb.set_trace()
        #! Warnings: By splitting losses for supervised data and unsupervised data with different names,
        #! it means that at least one sample for each group should be provided on each gpu.
        #! In some situation, we can only put one image per gpu, we have to return the sum of loss
        #! and log the loss with logger instead. Or it will try to sync tensors don't exist.
        # import ipdb;ipdb.set_trace();
        if "sup" in data_groups:
            gt_bboxes = data_groups["sup"]["gt_bboxes"]     # unnormalized [x1,y1,x2,y2]
            log_every_n(
                {"sup_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
            )
            # 1. supervised loss
            # pass the curr_step to help the model to know whether to do the warm-up step
            sup_loss = self.student.forward_train(**data_groups["sup"], curr_step=self.curr_step)
            sup_loss = {"sup_" + k: v for k, v in sup_loss.items()}
            loss.update(**sup_loss)
        # import ipdb;ipdb.set_trace()
          
        if "unsup_student" in data_groups:
            # 2. unsupservised loss
            unsup_loss = weighted_loss(
                self.foward_unsup_train(
                    data_groups["unsup_teacher"], data_groups["unsup_student"]
                ),
                weight=self.unsup_weight,
            )
            unsup_loss = {"unsup_" + k: v for k, v in unsup_loss.items()}
            loss.update(**unsup_loss)
       
        return loss

    def foward_unsup_train(self, teacher_data, student_data):
        # sort the teacher and student input to avoid some bugs
        tnames = [meta["filename"] for meta in teacher_data["img_metas"]]
        snames = [meta["filename"] for meta in student_data["img_metas"]]
        tidx = [tnames.index(name) for name in snames]
        with torch.no_grad():
            # 1. get pseudo bbox from the weak augmented images
            teacher_info = self.extract_teacher_info(
                teacher_data["img"][
                    torch.Tensor(tidx).to(teacher_data["img"].device).long()
                ],
                [teacher_data["img_metas"][idx] for idx in tidx],
            )
        # 2. get the prediction of the strong augmented images
        student_info = self.extract_student_info(**student_data)

        student_info['gt_bboxes'] = student_data['gt_bboxes']
        student_info['gt_labels'] = student_data['gt_labels']

        return self.compute_pseudo_label_loss(student_info, teacher_info)

    def compute_pseudo_label_loss(self, student_info, teacher_info):
        # 1. convert the weak augmented pseudo bbox into the strong augmented pseudo bbox
        M = self._get_trans_mat(
            teacher_info["transform_matrix"], student_info["transform_matrix"]
        )

        pseudo_bboxes = self._transform_bbox(
            teacher_info["det_bboxes"],             # 
            M,
            [meta["img_shape"] for meta in student_info["img_metas"]],
        )
        # import ipdb;ipdb.set_trace()
        pseudo_labels = teacher_info["det_labels"]
        pseudo_scores = teacher_info["det_scores"]
        loss = {}

        # 2. loss with pseudo bbox
        unsup_loss = self.unsup_loss(
            student_info,
            teacher_info,
            pseudo_bboxes,
            pseudo_labels,
            pseudo_scores,
        )
        loss.update(unsup_loss)

        return loss


    def unsup_loss(
        self,
        student_info,
        teacher_info,
        pseudo_bboxes,
        pseudo_labels,
        pseudo_scores,
        gt_bboxes_ignore=None,
        **kwargs,
        ):  
        """first take the initial pseudo bboxes and pseudo labels to do label matching with 
        Hungarian Match, and get the cost to fit a GMM model, then do the double filter the
        pseudo label, then combine the fixed threshold filtered labels to do the unsup train
        """

        imgs = student_info['img']
        feats = student_info['backbone_feature']
        img_metas = student_info['img_metas']
        
        # initial pseudo labels with mean + std
        det_labels_list_ = teacher_info['det_labels']
        det_bboxes_list_ = teacher_info['det_bboxes']
        det_scores_list_ = teacher_info['det_scores']


        # do the initial label assignment
        gt_labels_list_ = [label for label in pseudo_labels]
        gt_bboxes_list_ = [bbox for bbox in pseudo_bboxes]
        gt_scores_list_ = [score for score in pseudo_scores]

        all_cls_scores, all_bbox_preds, enc_cls_scores, enc_bbox_preds, dn_cls_scores, dn_bbox_preds = student_info['outs']
        
        # only consider the last decoder layers output
        cls_scores_, bbox_preds_ = all_cls_scores[-1], all_bbox_preds[-1]
        
        num_imgs = cls_scores_.size(0)

        cls_scores_list_ = [cls_scores_[i] for i in range(num_imgs)]
        bbox_preds_list_ = [bbox_preds_[i] for i in range(num_imgs)]

        # collect the batched images instance cost
        cost_ = []
        # import ipdb; ipdb.set_trace()
        match_gt_cost_list, match_gt_inds_list = [], []
        with torch.no_grad():
            for img_id, (cls_scores, bbox_preds, gt_labels, gt_bboxes) in enumerate(zip(cls_scores_list_, bbox_preds_list_, gt_labels_list_, gt_bboxes_list_)):

                # label assignment with Hungarian Match
                num_bboxes = bbox_preds.size(0)
                num_gts = gt_bboxes.size(0)
                if num_bboxes == 0 or num_gts == 0:
                    # put the padding data to avoid the bug
                    match_gt_cost_list.append(cls_scores.new_zeros(0).detach().cpu())
                    match_gt_inds_list.append(cls_scores.new_zeros(0).long().detach().cpu())
                    continue
                # calculate the cost matrix
                img_meta_ = img_metas[img_id]
                img_h, img_w, _ = img_meta_['img_shape']
                factor = gt_bboxes.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)

                # cls cost
                cls_cost = self.student.bbox_head.assigner2.cls_cost(cls_scores, gt_labels)
                # regression L1 cost
                normalize_gt_bboxes = gt_bboxes / factor
                reg_cost = self.student.bbox_head.assigner2.reg_cost(bbox_preds, normalize_gt_bboxes)
                # regression iou cost, defaultly giou is used in official DETR.
                bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factor
                iou_cost = self.student.bbox_head.assigner2.iou_cost(bboxes, gt_bboxes)
                # weighted sum of above three costs
                # import ipdb;ipdb.set_trace()
                cost = cls_cost + reg_cost + iou_cost

                # hungarian match
                cost = cost.detach().cpu()
                # matched_row_inds is the positive samples, matched_col_inds is the corresponding matched gt index
                matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
                matched_row_inds = torch.from_numpy(matched_row_inds).to(
                    bbox_preds.device)
                matched_col_inds = torch.from_numpy(matched_col_inds).to(
                    bbox_preds.device)

                # get the positive samples' cost
                pos_inds_gmm = matched_col_inds                         # pos_inds means the matcheding inital gt bbox index
                pos_cost_gmm = cost[matched_row_inds, matched_col_inds] 

                # save some useful info
                match_gt_cost_list.append(pos_cost_gmm)
                match_gt_inds_list.append(pos_inds_gmm)

                cost_.append(pos_cost_gmm)

       
        if len(cost_) > 0:
            cost_ = torch.cat(cost_).to(bbox_preds_.device)
        else:
            cost_ = bbox_preds_.new_zeros(0)
        rank, world_size = get_dist_info()
      
        # all_gather to get all the prediction
        cost_ = concat_all_gather(cost_).detach().cpu()
     
       
        # the pseudo label used to do the cross-query consistency  
        # pseudo labels for regression and calssification learning        
        gt_labels_list, gt_bboxes_list, gt_scores_list = [], [], []
        # pseudo labels in strong augmentation space for consistency
        unsup_labels_gmm_list, unsup_bboxes_gmm_list, unsup_scores_gmm_list = [], [], []
        # pseudo label in weak augmentation space for consistency
        det_labels_gmm_list, det_bboxes_gmm_list, det_scores_gmm_list = [], [], []

        # fit a gmm model to get the filter threshold
        thr_ = cost_.new_tensor(self._fit_gmm(cost_))
        
        if isinstance(self.train_cfg.pseudo_label_initial_score_thr, float):
            base_thr = self.train_cfg.pseudo_label_initial_score_thr                # default set 0.4
        else:
            # TODO: use dynamic threshold
            raise NotImplementedError("Dynamic Threshold is not implemented yet.")

        # import ipdb;ipdb.set_trace()
        for img_id, (gt_bboxes, gt_labels, gt_scores, det_bboxes, det_labels, det_scores, match_gt_cost, match_gt_inds) in enumerate(zip(gt_bboxes_list_, gt_labels_list_, gt_scores_list_, 
                                                                                                                                        det_bboxes_list_, det_labels_list_, det_scores_list_,
                                                                                                                                        match_gt_cost_list, match_gt_inds_list)):
    
            img_meta = img_metas[img_id]
            if gt_bboxes.dim() == 1:
                gt_bboxes = gt_bboxes.unsqueeze(0)

            valid_inds = torch.nonzero(match_gt_cost <= thr_, as_tuple=False).squeeze().unique()
            valid_gt_inds_1 = match_gt_inds[valid_inds]


            valid_gt_inds_2 = torch.nonzero(gt_scores >= base_thr, as_tuple=False).squeeze().unique()

            
            valid_gt_inds = torch.cat((valid_gt_inds_1.to(imgs.device), valid_gt_inds_2.to(imgs.device))).unique()
  
            gt_bboxes_list.append(gt_bboxes[valid_gt_inds_2, :4])
            gt_labels_list.append(gt_labels[valid_gt_inds_2])
            gt_scores_list.append(gt_scores[valid_gt_inds_2]) 
            
            # ==== High recall pseudo labels for consistency ====
            unsup_bboxes_gmm_list.append(gt_bboxes[valid_gt_inds, :4])
            unsup_labels_gmm_list.append(gt_labels[valid_gt_inds])
            unsup_scores_gmm_list.append(gt_scores[valid_gt_inds]) 

            # the pseudo bbox in the weak aug space
            det_bboxes_gmm_list.append(det_bboxes[valid_gt_inds, :4])
            det_labels_gmm_list.append(det_labels[valid_gt_inds])
            det_scores_gmm_list.append(det_scores[valid_gt_inds])

        
        # change the warm_up state
        if self.curr_step < self.student.bbox_head.warm_up_step:
            self.student.bbox_head.in_warm_up = True
        else:
            self.student.bbox_head.in_warm_up = False

        aug_v1_feat = feats
        with torch.no_grad():
            aug_v2_feat = self.teacher.extract_feat(teacher_info['img'])

        img_metas_v1 = student_info['img_metas']
        img_metas_v2 = teacher_info['img_metas']  
        
    
        # Note: need the labels and boxes as the list of each images
        # NOTE:!!!! This is the target of the dn part, so need the high precision pseudo labels !!!
        targets_v1 = dict()
        targets_v1['labels'] = gt_labels_list
        # becareful when prepare the dn_components, the gt bboxes needs to 
        # be normalized with corresponding w, h, in the cx, cy, w, h format
        normalized_boxes = []
        for img_meta, gt_bbox in zip(img_metas_v1, gt_bboxes_list):
            img_h, img_w, _ = img_meta['img_shape']
            factor = gt_bbox.new_tensor([img_w, img_h, img_w,
                                        img_h]).unsqueeze(0).repeat(
                                            gt_bbox.size(0), 1)
            gt_bbox_ = bbox_xyxy_to_cxcywh(gt_bbox)
            normalized_boxes.append(gt_bbox_ / factor)

        targets_v1['boxes'] = normalized_boxes
        dn_args=(targets_v1, self.student.bbox_head.dn_number, self.student.bbox_head.dn_label_noise_ratio, self.student.bbox_head.dn_box_noise_scale)

        # import ipdb;ipdb.set_trace()
        losses = {}
        # pseudo bboxes in the strong augmented view
        pseudo_bboxes_v1 = unsup_bboxes_gmm_list
        pseudo_labels_v1 = unsup_labels_gmm_list
      
        input_query_label_1, input_query_bbox_1, input_query_label_2, input_query_bbox_2, attn_mask_1, dn_meta_1 = self.prepare_unsup_cdn(teacher_info, student_info, pseudo_bboxes_v1, pseudo_labels_v1, det_bboxes_gmm_list, det_labels_gmm_list, dn_args=dn_args)

        # construct the complete dn querys
        input_query_label_v1 = torch.cat([input_query_label_1, input_query_label_2], dim=1)
        input_query_bbox_v1 = torch.cat([input_query_bbox_1, input_query_bbox_2], dim=1)
        
        # forward with the denoising query
        
        hs_v1, outputs_class_v1, outputs_coord_v1, interm_outputs_class_v1, interm_outputs_coord_v1, \
            consistency_outputs_class_v1, consistency_outputs_coord_v1, dn_outputs_class_v1, dn_outputs_coord_v1 = \
                self.student.bbox_head.forward_dummy(aug_v1_feat, img_metas_v1, input_query_label_v1, input_query_bbox_v1, attn_mask_1, dn_meta_1)
        
            
        # seperate the output and calculate the loss respectively
       
        # Loss Part - 1
    
        unsup_label_loss_v1 = self.student.bbox_head.loss(outputs_class_v1, outputs_coord_v1, interm_outputs_class_v1, interm_outputs_coord_v1, dn_cls_scores=dn_outputs_class_v1, dn_bbox_preds=dn_outputs_coord_v1, \
                                                            gt_bboxes_list=gt_bboxes_list, gt_labels_list=gt_labels_list, gt_scores_list=gt_scores_list, img_metas=img_metas_v1, dn_metas=dn_meta_1, is_pseudo_label=True)
                                                            
        losses.update(unsup_label_loss_v1)

        # Loss Part - 2
        # breakpoint()
        # import ipdb;ipdb.set_trace()
        # weak aug data is from the teacher model so the bbox don't need to do the transform
        prior_info = {}
        prior_info['loss_weights'] = dn_meta_1['loss_weights']
        prior_info['input_query_label_1'] = input_query_label_1

        pseudo_bboxes_v2 = det_bboxes_gmm_list
        pseudo_labels_v2 = det_labels_gmm_list

       
        # NOTE: need the labels and boxes as the list of each images
        # In order the reuse the attn_mask of previou calculated
        targets_v2 = dict()
        targets_v2['labels'] = det_labels_gmm_list
        # becareful when prepare the dn_components, the gt bboxes needs to 
        # be normalized with corresponding w, h, in the cx, cy, w, h format
        normalized_boxes = []
        for img_meta, gt_bbox in zip(img_metas_v2, det_bboxes_gmm_list):
            img_h, img_w, _ = img_meta['img_shape']
            factor = gt_bbox.new_tensor([img_w, img_h, img_w,
                                        img_h]).unsqueeze(0).repeat(
                                            gt_bbox.size(0), 1)
            gt_bbox_ = bbox_xyxy_to_cxcywh(gt_bbox)
            normalized_boxes.append(gt_bbox_ / factor)

        targets_v2['boxes'] = normalized_boxes
        dn_args=(targets_v2, self.student.bbox_head.dn_number, self.student.bbox_head.dn_label_noise_ratio, self.student.bbox_head.dn_box_noise_scale)

        with torch.no_grad():
            input_query_label_1, input_query_bbox_1, input_query_label_2, input_query_bbox_2, attn_mask_2, dn_meta_2  = self.prepare_unsup_cdn(teacher_info, teacher_info, pseudo_bboxes_v2, pseudo_labels_v2, det_bboxes_gmm_list, det_labels_gmm_list, dn_args=dn_args, prior_info=prior_info)
            
            # construct the complete dn querys
            input_query_label_v2 = torch.cat([input_query_label_1, input_query_label_2], dim=1)
            input_query_bbox_v2 = torch.cat([input_query_bbox_1, input_query_bbox_2], dim=1)
            # forward with the denoising query
            # import ipdb;ipdb.set_trace()
            hs_v2, outputs_class_v2, outputs_coord_v2, interm_outputs_class_v2, interm_outputs_coord_v2, \
                consistency_outputs_class_v2, consistency_outputs_coord_v2, dn_outputs_class_v2, dn_outputs_coord_v2  = \
                    self.teacher.bbox_head.forward_dummy(aug_v2_feat, img_metas_v2, input_query_label_v2, input_query_bbox_v2, attn_mask_2, dn_meta_2)
          

        
        # import ipdb;ipdb.set_trace()
        # calculate the consistency loss
        # for each decoder layer
        assert dn_meta_1['pad_size_1'] == dn_meta_2['pad_size_1']
        pad_size = dn_meta_1['pad_size_1']
        known_bid = dn_meta_1['known_bid_1']
        map_known_indice = dn_meta_1['map_known_indice_1']

        loss_weights = dn_meta_1['loss_weights']
        if self.curr_step >= self.student.bbox_head.warm_up_step:
            loss_weights = torch.zeros_like(loss_weights)

        for layer_id in range(len(hs_v1)):

            hs_v1_feats, hs_v2_feats = hs_v1[layer_id][:, :pad_size, ...], hs_v2[layer_id][:, :pad_size, ...]     # hs_v1_feats: [bs, seq_len, embed_dim]  
            
            h1 = hs_v1_feats[known_bid.long(), map_known_indice]
            h2 = hs_v2_feats[known_bid.long(), map_known_indice]

            loss = (F.mse_loss(F.normalize(h1, p=2, dim=-1), F.normalize(h2, p=2, dim=-1).detach(), reduction='none') * loss_weights.unsqueeze(-1)).mean()

            losses.update({"consis_loss.d{}".format(layer_id): 10 * loss})
        return losses

    def prepare_unsup_cdn(self, teacher_info, student_info, pseudo_bboxes, pseudo_labels, det_bboxes, det_laebls, 
                            dn_args=None, hidden_dim=256, num_queries=900, num_classes=80, prior_info=None):
        """prepare the contrastive denoising query and unsupervised consistency denoising query at the 
        same time.
        Args:
            - pseudo_bboxes: the pseudo bboxes on the target augmentation space.
            - pseudo_labels: the pseudo labels on the target augmentation space.
            - det_bboxes: the original detected bboxes on the source augmentation space.
            - det_labels: the original detected labels on the source augmentation space.
        We prepare the cdn and consistency query at the same time. we will have the dn queries in 
        "[consistnecy queries, denoising queries]"
        """
        # import ipdb;ipdb.set_trace()
        # box embed
        # obtain the location info of the cropped patches in corresponding augmentation space
        # convert the pseudo bboxes from unnormalized [x1, y1, x2, y2] to
        # normalized [cx, cy, w, h] format
        img_metas_tgt = student_info['img_metas']
        img_metas_src = teacher_info['img_metas']

        imgs_tgt = student_info['img']
        imgs_src = teacher_info['img']

        norm_pseudo_bboxes_list = []
        

        # ++++++ consistency part ++++++ #
        for img_meta, pseudo_bbox in zip(img_metas_tgt, pseudo_bboxes):
            img_h, img_w, _ = img_meta['img_shape']
        
            # if there exist some empty pseudo bbox
            if pseudo_bbox.size(0) == 0:
                pseudo_bbox = pseudo_bbox.new_tensor([[img_w / 4, img_h / 4, 3 * img_w/ 4, 3 * img_h / 4]])

            factor = pseudo_bbox.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0).repeat(
                                            pseudo_bbox.size(0), 1)
            pseudo_bbox_ = bbox_xyxy_to_cxcywh(pseudo_bbox)
            norm_pseudo_bboxes = pseudo_bbox_ / factor
            norm_pseudo_bboxes = norm_pseudo_bboxes.clamp(min=0.0, max=1.0)
            norm_pseudo_bboxes_list.append(norm_pseudo_bboxes)


        # pseudo label number
        known_num = [bboxes.size(0) for bboxes in norm_pseudo_bboxes_list]

        batch_idx = torch.cat([imgs_tgt.new_full((t.shape[0],), i) for i, t in enumerate(norm_pseudo_bboxes_list)])
        batch_size = len(known_num)

        # dynamic setting the dn_number refer to the prepare the dn_components
        dn_number_1 = 5
        # dn_number_1 = dn_number_1 // (int(max(known_num)))
    
        single_pad_1 = int(max(known_num))
        pad_size_1 = int(single_pad_1 * dn_number_1)     # if there is no pseudo bbox, we still have a single denoising query

        # prepare the consistency query pos
        # don't apply the noise on the pseudo bbox
        batched_tgt_bboxes = torch.cat(norm_pseudo_bboxes_list)

        known_bid_1 = batch_idx.repeat(dn_number_1, 1).view(-1)
        known_bboxes = batched_tgt_bboxes.repeat(dn_number_1, 1)
        consistency_bbox_embed = inverse_sigmoid(known_bboxes)


        padding_label = torch.zeros(pad_size_1, hidden_dim).cuda()
        padding_bbox = torch.zeros(pad_size_1, 4).cuda()

        input_query_label_1 = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox_1 = padding_bbox.repeat(batch_size, 1, 1)
        
        # bbox embed
        map_known_indice_1 = torch.tensor([]).to('cuda')
        if len(known_num):
            map_known_indice_1 = torch.cat([torch.tensor(range(num)) for num in known_num])
            map_known_indice_1 = torch.cat([map_known_indice_1 + single_pad_1 * i for i in range(dn_number_1)]).long()

        input_query_bbox_1[(known_bid_1.long(), map_known_indice_1)] = consistency_bbox_embed

        # query embed
        proposal_bboxes = [bbox[:, :4] for bbox in det_bboxes]
        # store the batched proposal bboxes and loss
        proposal_bboxes_list, loss_weights_list = [], []
        if prior_info is None:
            # when there is no prior info, generate the query embedding
            # denormalize first, to make prepare for the following augmentation
            # from tensor to numpy
            # import ipdb; ipdb.set_trace()
        
            # Note: construct the roi bboxes then repeat dn_number
            # for each image
            for img_id, proposal_bbox in enumerate(proposal_bboxes):
                if proposal_bbox.size(0)  == 0:
                    img_h, img_w, _ = teacher_info['img_metas'][img_id]['img_shape']
                    proposal_bboxes_list.append(proposal_bbox.new_tensor([[img_w / 4, img_h / 4, 3 * img_w/ 4, 3 * img_h / 4]]))
                    loss_weights_list.append(proposal_bbox.new_zeros(1))
                else:
                    proposal_bboxes_list.append(proposal_bbox)
                    loss_weights_list.append(proposal_bbox.new_ones(proposal_bbox.size(0)))

            # concatenate the batched proposal bbox and batched loss_weights
            batched_proposal_bboxes = torch.cat(proposal_bboxes_list)
            batched_loss_weights = torch.cat(loss_weights_list)
            
            # Todo: add the noise like DN on the proposal bboxes?
            # import ipdb;ipdb.set_trace()

            loss_weights = batched_loss_weights.unsqueeze(-1).repeat(dn_number_1, 1)

            # extract the roi feats
            rois_bboxes = torch.cat([known_bid_1.unsqueeze(-1), batched_proposal_bboxes.repeat(dn_number_1, 1)], dim=-1)

            with torch.no_grad():
                # prepare the content feature
                # extract the backbone feature for multi-level feats
                mlvl_feats = self.teacher.extract_feat(imgs_src)   # [8x, 16x, 32x]
                # get the multi-level feature maps
                mlvl_feats, mlvl_masks, mlvl_pos = self.prepare_feats(mlvl_feats, img_metas_src)

                # convert the bbox into rois
                consistency_query_embed =  self.roi_extractor(mlvl_feats, rois_bboxes) # [num_rois, 256, 7, 7]
            # apply the projector
            consistency_query_embed = self.projector(consistency_query_embed)
           
            input_query_label_1[(known_bid_1.long(), map_known_indice_1)] = consistency_query_embed

        else:
            loss_weights = prior_info['loss_weights']
            input_query_label_1 = prior_info['input_query_label_1']


        # ========== query denoising part ========== #
        # import ipdb;ipdb.set_trace()
        # dn query embed
        dn_number_2 = 100

        targets, dn_number_2, label_noise_ratio, box_noise_scale = dn_args

        # preprocess
        # check there is gt bbox for each image
        num_valid_gts = [t.size(0) for t in targets['boxes']]

        gt_labels = []
        gt_bboxes = []

        for i, num in enumerate(num_valid_gts):
            if num == 0:
                # if there is no gt bbox in this image, we randomly generate a 
                # gt bbox and gt label for this image
                # bbox is the normalized cx, cy, w, h format
                tmp_bboxes = torch.tensor([[0.5, 0.5, 0.5, 0.5]]).cuda()
                tmp_labels = torch.randint(0, 80, (1,)).long().cuda()

                gt_bboxes.append(tmp_bboxes)
                gt_labels.append(tmp_labels)

            else:
                gt_bboxes.append(targets['boxes'][i])
                gt_labels.append(targets['labels'][i])

        # positive and negative dn queries
        dn_number_2 = dn_number_2 * 2
        known = [(torch.ones_like(t)).cuda() for t in gt_labels]
        batch_size = len(known)
        known_num = [sum(k) for k in known]
        if int(max(known_num)) == 0:
            dn_number_2 = 1
        else:
            if dn_number_2 >= 100:
                dn_number_2 = dn_number_2 // (int(max(known_num) * 2))
            elif dn_number_2 < 1:
                dn_number_2 = 1
        if dn_number_2 == 0:
            dn_number_2 = 1
        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t for t in gt_labels])
        boxes = torch.cat([t for t in gt_bboxes])
        batch_idx = torch.cat([torch.full_like(t.long(), i) for i, t in enumerate(gt_labels)])

        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)

        known_indice = known_indice.repeat(2 * dn_number_2, 1).view(-1)
        known_labels = labels.repeat(2 * dn_number_2, 1).view(-1)
        known_bid_2 = batch_idx.repeat(2 * dn_number_2, 1).view(-1)
        known_bboxs = boxes.repeat(2 * dn_number_2, 1)
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        if label_noise_ratio > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(-1)   # half of bbox prob
            new_label = torch.randint_like(chosen_indice, 0, num_classes)           # randomly put a new one here
            known_labels_expaned.scatter_(0, chosen_indice, new_label)
        single_pad_2 = int(max(known_num))

        pad_size_2 = int(single_pad_2 * 2 * dn_number_2)  #
        positive_idx = torch.tensor(range(len(boxes))).long().cuda().unsqueeze(0).repeat(dn_number_2, 1)
        positive_idx += (torch.tensor(range(dn_number_2)) * len(boxes) * 2).long().cuda().unsqueeze(1)
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(boxes)
        if box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs)
            known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
            known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2

            diff = torch.zeros_like(known_bboxs)
            diff[:, :2] = known_bboxs[:, 2:] / 2
            diff[:, 2:] = known_bboxs[:, 2:] / 2

            rand_sign = torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
            rand_part = torch.rand_like(known_bboxs)
            rand_part[negative_idx] += 1.0
            rand_part *= rand_sign
            known_bbox_ = known_bbox_ + torch.mul(rand_part,
                                                  diff).cuda() * box_noise_scale
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
            known_bbox_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
            known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]

        m = known_labels_expaned.long().to('cuda')
        input_label_embed = self.student.bbox_head.label_enc(m)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        padding_label = torch.zeros(pad_size_2, hidden_dim).cuda()
        padding_bbox = torch.zeros(pad_size_2, 4).cuda()

        input_query_label_2 = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox_2 = padding_bbox.repeat(batch_size, 1, 1)

        map_known_indice_2 = torch.tensor([]).to('cuda')
        if len(known_num):
            map_known_indice_2 = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
            map_known_indice_2 = torch.cat([map_known_indice_2 + single_pad_2 * i for i in range(2 * dn_number_2)]).long()
        if len(known_bid_2):
            input_query_label_2[(known_bid_2.long(), map_known_indice_2)] = input_label_embed
            input_query_bbox_2[(known_bid_2.long(), map_known_indice_2)] = input_bbox_embed

        # import ipdb;ipdb.set_trace()
        # construct the attention mask
        tgt_size = pad_size_1 + pad_size_2 + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # match query cannot see the any part of the reconstruct
        attn_mask[pad_size_1 + pad_size_2:, :pad_size_1 + pad_size_2] = True   
        # two part attn_mask respectively
        # consistency part
    
        for i in range(dn_number_1):
            if i == 0:
                attn_mask[single_pad_1 * i:single_pad_1  * (i + 1), single_pad_1 * (i + 1):(pad_size_1 + pad_size_2)] = True
            else:
                attn_mask[single_pad_1 * i:single_pad_1 * (i + 1), single_pad_1 * (i + 1):(pad_size_1 + pad_size_2)] = True
                attn_mask[single_pad_1 * i:single_pad_1 * (i + 1), :single_pad_1 * i] = True

        # cdn part
        for j in range(dn_number_2):
            if j == dn_number_2 - 1:
                attn_mask[(pad_size_1 + single_pad_2 * 2 * j): (pad_size_1  + single_pad_2 * 2 * (j + 1)), :(pad_size_1  + single_pad_2 * j * 2)] = True
            else: 
                attn_mask[(pad_size_1 + single_pad_2 * 2 * j): (pad_size_1  + single_pad_2 * 2 * (j + 1)), (pad_size_1 + single_pad_2 * 2 * (j + 1)):(pad_size_1 + pad_size_2)] = True
                attn_mask[(pad_size_1  + single_pad_2 * 2 * j): (pad_size_1  + single_pad_2 * 2 * (j + 1)), :(pad_size_1 + single_pad_2 * 2 * j)] = True

        # import ipdb;ipdb.set_trace()
        
        dn_meta = {
            'pad_size_1': pad_size_1,
            'pad_size_2': pad_size_2,
            'num_dn_group_1': dn_number_1,
            'num_dn_group_2': dn_number_2,
            'known_bid_1': known_bid_1,
            'known_bid_2': known_bid_2,
            'map_known_indice_1': map_known_indice_1,
            'map_known_indice_2': map_known_indice_2,

            'loss_weights': loss_weights if self.curr_step < self.student.bbox_head.warm_up_step else torch.zeros_like(loss_weights)
        }

        return input_query_label_1, input_query_bbox_1, input_query_label_2, input_query_bbox_2, attn_mask, dn_meta

  


    def prepare_feats(self, mlvl_feats, img_metas):
        """Prepare the feature maps used in the DINO encoder, which like the FPN structure. 
        Return:
            - srcs: (list), contain the batched feature maps for multi-level, with the shape (BS, 256, H, W)
            - mlvl_masks: (list), mask of batched feature maps for multi-feature level, (BS, H, W)
            - mlvl_positional_encodings: (list),
        """
        # import ipdb;ipdb.set_trace()

        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        img_masks = mlvl_feats[0].new_ones(
            (batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0
        
        srcs = []
        mlvl_masks = []
        mlvl_positional_encodings = []
        for l, feat in enumerate(mlvl_feats):
            mlvl_masks.append(F.interpolate(img_masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(self.teacher.bbox_head.positional_encoding(mlvl_masks[-1]))
            srcs.append(self.teacher.bbox_head.input_proj[l](feat))
        
        if self.teacher.bbox_head.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.teacher.bbox_head.num_feature_levels):
                if l == _len_srcs:
                    src = self.teacher.bbox_head.input_proj[l](mlvl_feats[-1])
                else:
                    src = self.teacher.bbox_head.input_proj[l](srcs[-1])
                
                srcs.append(src)
                mlvl_masks.append(F.interpolate(img_masks[None], size=src.shape[-2:]).to(torch.bool).squeeze(0))
                mlvl_positional_encodings.append(self.teacher.bbox_head.positional_encoding(mlvl_masks[-1]))
        return srcs, mlvl_masks, mlvl_positional_encodings


    @force_fp32(apply_to=["bboxes", "trans_mat"])
    def _transform_bbox(self, bboxes, trans_mat, max_shape):
        bboxes = Transform2D.transform_bboxes(bboxes, trans_mat, max_shape)
        return bboxes

    @force_fp32(apply_to=["a", "b"])
    def _get_trans_mat(self, a, b):
        return [bt @ at.inverse() for bt, at in zip(b, a)]

    def extract_student_info(self, img, img_metas, **kwargs):
        """Only get some data info of student model
        """
        student_info = {}
        student_info["img"] = img
        feat = self.student.extract_feat(img)
        student_info["backbone_feature"] = feat
     
        # prediction results of the student model
        with torch.no_grad():
            outs = self.student.bbox_head.forward(feat, img_metas)
        student_info['outs'] = outs
        student_info["img_metas"] = img_metas
        student_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in img_metas
        ]
        return student_info

    def _fit_gmm(self, data_points, device=None):
        """fit a GMM model with the data of the memory bank to find relative better 
        cost threshold to filter the pseudo labels.   
        Args:
            data_points: (tensor), the instance-level cost data points. 
        """
      
        pos_cost_gmm = data_points
        if len(pos_cost_gmm) == 0:
            return 0
            
        # initialization of the GMM model
        pos_cost_gmm, sort_inds = pos_cost_gmm.sort()   # from low to high according the cost
        pos_cost_gmm = pos_cost_gmm.view(-1, 1).cpu().numpy()
        min_cost, max_cost = pos_cost_gmm.min(), pos_cost_gmm.max()
        means_init = np.array([min_cost, max_cost]).reshape(2, 1)
        weights_init = np.array([0.5, 0.5])
        precisions_init = np.array([1.0, 1.0]).reshape(2, 1, 1)  # full
        if self.covariance_type == 'spherical':
            precisions_init = precisions_init.reshape(2)
        elif self.covariance_type == 'diag':
            precisions_init = precisions_init.reshape(2, 1)
        elif self.covariance_type == 'tied':
            precisions_init = np.array([[1.0]])
        if skm is None:
            raise ImportError('Please run "pip install sklearn" '
                            'to install sklearn first.')

        # do the GMM model fit
        gmm = skm.GaussianMixture(
            2,
            weights_init=weights_init,
            means_init=means_init,
            precisions_init=precisions_init,
            covariance_type=self.covariance_type,
            reg_covar=1e-5)
        
        if len(pos_cost_gmm) < 2:
            return pos_cost_gmm[0]
        
        gmm.fit(pos_cost_gmm)
       
        gmm_assignment = gmm.predict(pos_cost_gmm)
        scores = gmm.score_samples(pos_cost_gmm)
        gmm_assignment = torch.from_numpy(gmm_assignment).to(device)
        scores = torch.from_numpy(scores).to(device)
        pseudo_mask = gmm_assignment == 0
        cost_thr = None
        if pseudo_mask.nonzero().numel():
            _, pseudo_thr_ind = scores[pseudo_mask].topk(1)
            cost_thr = pos_cost_gmm[pseudo_mask][pseudo_thr_ind]
        if cost_thr is not None:
            return cost_thr
        else:
            # import ipdb;ipdb.set_trace()
            pseudo_mask = gmm_assignment == 1
            _, pseudo_thr_ind = scores[pseudo_mask].topk(1)
            cost_thr = pos_cost_gmm[pseudo_mask][pseudo_thr_ind]
            return cost_thr


    def extract_teacher_info(self, img, img_metas, **kwargs):

        teacher_info = {}
        teacher_info['img'] = img
        feat = self.teacher.extract_feat(img)
        teacher_info["backbone_feature"] = feat
        
        # import ipdb;ipdb.set_trace()
        # TODO: change the output, proposal_list [tensor:[100,5]], proposal_label_list: [tensor:[100]]
        # Note: pass the curr_step to change the warm_up_state of the teacher mode to change the evaluation
        # method, and pass the for_pseudo_label to change the way to generate pseudo label i.e. use NMS or not
        proposal_list = self.teacher.bbox_head.simple_test_bboxes(
            feat, img_metas, rescale=False, curr_step=self.curr_step, for_pseudo_label=True,
        )

        proposal_box_list = [p[0].to(feat[0].device) for p in proposal_list]
        proposal_box_list = [p if p.shape[0] > 0 else p.new_zeros(0, 5) for p in proposal_box_list]
        proposal_label_list = [p[1].to(feat[0].device) for p in proposal_list]
        

        # image level mean + std adaptive threshold
        det_bboxes, det_labels,  det_scores = [], [], []

        # make the change to the specifical dynamic thresholding methods
      
        for proposal_box, proposal_label in zip(proposal_box_list, proposal_label_list):
            # proposal_box: [num_bbox, 5] proposal_label: [num_bbox]
            # obtain the image-level mean + std pseudo label threshold
            avg_score = torch.mean(proposal_box[:, -1])
            std_score = torch.std(proposal_box[:, -1])

            pseudo_thr = avg_score + std_score
        
            # filter the pseudo bbox
            valid_inds = torch.nonzero(proposal_box[:, -1] >= pseudo_thr, as_tuple=False).squeeze().unique()
            
            # filter the empty pseudo labels
            tmp_bboxes = proposal_box[valid_inds, :4]   # unnormalized [x1, y1, x2, y2] format
            bw = tmp_bboxes[:, 2] - tmp_bboxes[:, 0]
            bh = tmp_bboxes[:, 3] - tmp_bboxes[:, 1]

            valid = (bw > 0) & (bh > 0)
            valid_inds = valid_inds[valid]

            det_bboxes.append(proposal_box[valid_inds, :4])
            det_labels.append(proposal_label[valid_inds])
            det_scores.append(proposal_box[valid_inds, 4]) 

       
        teacher_info["det_bboxes"] = det_bboxes
        teacher_info["det_labels"] = det_labels
        teacher_info["det_scores"] = det_scores

        teacher_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in img_metas
        ]
        teacher_info["img_metas"] = img_metas
        return teacher_info

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if not any(["student" in key or "teacher" in key for key in state_dict.keys()]):
            keys = list(state_dict.keys())
            state_dict.update({"teacher." + k: state_dict[k] for k in keys})
            state_dict.update({"student." + k: state_dict[k] for k in keys})
            for k in keys:
                state_dict.pop(k)

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
