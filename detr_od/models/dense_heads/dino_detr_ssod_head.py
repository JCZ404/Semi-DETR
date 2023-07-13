# Copyright (c) OpenMMLab. All rights reserved.
import math
import time
import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Linear, build_activation_layer, bias_init_with_prob
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
from mmcv.runner import force_fp32

from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean, multiclass_nms)
from mmdet.models.utils import build_transformer
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.utils.transformer import inverse_sigmoid

from  mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead


# from .dn_components import prepare_for_cdn_plus, dn_post_process_plus
from .dn_components import *


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@HEADS.register_module()
class DINODETRSSODHead(AnchorFreeHead):
    """Implements the DETR transformer head.

    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_reg_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head. Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.
            Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_bbox (obj:`mmcv.ConfigDict`|dict): Config of the
            regression loss. Default `L1Loss`.
        loss_iou (obj:`mmcv.ConfigDict`|dict): Config of the
            regression iou loss. Default `GIoULoss`.
        tran_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    _version = 2

    def __init__(self,
                 num_classes=80,
                 in_channels=2048,
                 num_query=900,
                 num_reg_fcs=2,
                 transformer=None,
                 num_feature_levels=4,
                 num_backbone_outs=3,
                 backbone_channels=[512, 1024, 2048],
                 sync_cls_avg_factor=False,
                 iter_update=True,
                 dn_number=100,
                 dn_box_noise_scale = 0.4,
                 dn_label_noise_ratio = 0.5,
                 dn_labelbook_size = 81,
                 query_dim=2,
                 dec_pred_class_embed_share=True,
                 dec_pred_bbox_embed_share=True,
                 two_stage_bbox_embed_share=False,
                 two_stage_class_embed_share=False,
                 bbox_embed_diff_each_layer=False,
                 random_refpoints_xy=False,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 loss_cls1=dict(
                    type='TaskAlignedFocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    loss_weight=2.0) ,
                 loss_cls2=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 # training and testing settings
                 train_cfg=dict(
                    assigner1=dict(
                        type='O2MAssigner'),
                    assigner2=dict(
                        type='HungarianAssigner',
                        cls_cost=dict(type='FocalLossCost', weight=2.0),
                        reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                        iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
                        debug=False)),
                 test_cfg=dict(max_per_img=100),
                 init_cfg=None,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        super(AnchorFreeHead, self).__init__(init_cfg)
        # import ipdb; ipdb.set_trace()
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls2.get('class_weight', None)
        if class_weight is not None and (self.__class__ is DABDETRTwoPhaseHead):
            assert isinstance(class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls2.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_classes] = bg_cls_weight
            loss_cls2.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls2:
                loss_cls2.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        if train_cfg:
            assert 'assigner2' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'

            assigner1 = train_cfg['assigner1']
            assigner2 = train_cfg['assigner2']
            assert loss_cls2['loss_weight'] == assigner2['cls_cost']['weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            assert loss_bbox['loss_weight'] == assigner2['reg_cost'][
                'weight'], 'The regression L1 weight for loss and matcher ' \
                'should be exactly the same.'
            assert loss_iou['loss_weight'] == assigner2['iou_cost']['weight'], \
                'The regression iou weight for loss and matcher should be' \
                'exactly the same.'

            # NOTE: here we build to assigner to used in different stage
            self.assigner1 = build_assigner(assigner1)
            self.assigner2 = build_assigner(assigner2)

            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False

        # NOTE: for two difference phase, we have difference classification loss
        self.loss_cls1 = build_loss(loss_cls1)
        self.loss_cls2 = build_loss(loss_cls2)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)

        # setting for prediction head for decoder and encoder
        self.dec_pred_class_embed_share = dec_pred_class_embed_share
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        self.two_stage_bbox_embed_share = two_stage_bbox_embed_share
        self.two_stage_class_embed_share = two_stage_class_embed_share

        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer
        
        # feature levels
        self.num_feature_levels = num_feature_levels
        self.num_backbone_outs = num_backbone_outs
        self.backbone_channels = backbone_channels

        # setting query dim
        self.query_dim = query_dim
        assert query_dim in [2, 4]
        self.iter_update = iter_update
        self.random_refpoints_xy = random_refpoints_xy

        # setting dn training
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size

        if self.loss_cls2.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.activate = build_activation_layer(self.act_cfg)
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims

        assert 'num_feats' in positional_encoding
        num_feats = positional_encoding['num_feats']
        assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
            f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
            f' and {num_feats}.'
        self._init_layers()


        # NOTE: set the warm-up state
        self.warm_up_step = test_cfg['warm_up_step']
        self.in_warm_up = False      # indicator whether in the warm_up stage, default is False 


    def _init_layers(self):
        """Initialize layers of the transformer head."""
        # In DINO, like deformable detr, we use multi-scale feature maps
        if self.num_feature_levels > 1:
            num_backbone_outs = self.num_backbone_outs
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = self.backbone_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.embed_dims, kernel_size=1),
                    nn.GroupNorm(32, self.embed_dims),
                ))
            for _ in range(self.num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.embed_dims, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, self.embed_dims),
                ))
                in_channels = self.embed_dims
            self.input_proj = nn.ModuleList(input_proj_list)
         
        # Initialize the cls and reg head
        # prepare class & box embed
        _class_embed = nn.Linear(self.embed_dims, self.cls_out_channels)
        _bbox_embed = MLP(self.embed_dims, self.embed_dims, 4, 3)
        # init the two embed layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        _class_embed.bias.data = torch.ones(self.cls_out_channels) * bias_value
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        if self.dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(self.transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [copy.deepcopy(_bbox_embed) for i in range(self.transformer.num_decoder_layers)]
        if self.dec_pred_class_embed_share:
            class_embed_layerlist = [_class_embed for i in range(self.transformer.num_decoder_layers)]
        else:
            class_embed_layerlist = [copy.deepcopy(_class_embed) for i in range(self.transformer.num_decoder_layers)]

        self.fc_reg = nn.ModuleList(box_embed_layerlist)
        self.fc_cls = nn.ModuleList(class_embed_layerlist)

        # Initial the two stage encoder prediction head
        if self.two_stage_bbox_embed_share:
            # bbox head share with the decoder
            assert self.dec_pred_class_embed_share and self.dec_pred_bbox_embed_share
            self.fc_enc_reg = _bbox_embed
        else:
            self.fc_enc_reg = copy.deepcopy(_bbox_embed)

        if self.two_stage_class_embed_share:
            assert self.dec_pred_class_embed_share and self.dec_pred_bbox_embed_share
            self.fc_enc_cls = _class_embed
        else:
            self.fc_enc_cls = copy.deepcopy(_class_embed)

        # Initialize the reference embedding
        self.refpoint_embed = None  

        # Initialize the dn label embedding
        self.label_enc = nn.Embedding(self.dn_labelbook_size + 1, self.embed_dims)

    def init_weights(self):
        # The initialization for transformer is important
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
     
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """load checkpoints."""
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since `AnchorFreeHead._load_from_state_dict` should not be
        # called here. Invoking the default `Module._load_from_state_dict`
        # is enough.

        # Names of some parameters in has been changed.
        version = local_metadata.get('version', None)
     
        super(AnchorFreeHead,
              self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)

    def prep_for_dn(self, dn_meta, is_pseudo_label=False):
        """Get some dn part information
        """
        if is_pseudo_label:
            num_dn_groups, pad_size = dn_meta['num_dn_group_2'], dn_meta['pad_size_2']
        else:
            num_dn_groups, pad_size = dn_meta['num_dn_group'], dn_meta['pad_size']
        assert pad_size % num_dn_groups == 0
        single_pad = pad_size // num_dn_groups
        # single_pad: the padding size for each image
        # num_dn_groups: the dn_groups for each batch
        return single_pad, num_dn_groups



    def forward(self, mlvl_feats, 
                      img_metas,  
                      input_query_label=None, 
                      input_query_bbox=None, 
                      attn_mask=None, 
                      dn_meta=None):
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
            mlvl_positional_encodings.append(self.positional_encoding(mlvl_masks[-1]))
            srcs.append(self.input_proj[l](feat))
        
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](mlvl_feats[-1])
                else:
                    src = self.input_proj[l](srcs[-1])
                
                srcs.append(src)
                mlvl_masks.append(F.interpolate(img_masks[None], size=src.shape[-2:]).to(torch.bool).squeeze(0))
                mlvl_positional_encodings.append(self.positional_encoding(mlvl_masks[-1]))
        
     
        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(srcs, \
                                                                             mlvl_masks, \
                                                                             input_query_bbox, \
                                                                             mlvl_positional_encodings, \
                                                                             input_query_label,\
                                                                             attn_mask,\
                                                                             fc_reg=self.fc_reg, fc_cls=self.fc_cls,\
                                                                             fc_enc_reg=self.fc_enc_reg, fc_enc_cls=self.fc_enc_cls)
     
        hs[0] += self.label_enc.weight[0, 0] * 0.0

        # regression
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_fc_reg, layer_hs) in enumerate(zip(reference[:-1], self.fc_reg, hs)):
            layer_delta_unsig = layer_fc_reg(layer_hs)  
            layer_outputs_unsig = layer_delta_unsig  + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord = torch.stack(outputs_coord_list)             # [num_dec_layer, bs, num_query + dn_size, 4]      

        # classification
        outputs_class = torch.stack([layer_fc_cls(layer_hs) for     # [num_dec_layer, bs, num_query + dn_size, cls_out_channels] 
                                     layer_fc_cls, layer_hs in zip(self.fc_cls, hs)])

        # encoder output
        if hs_enc is not None:
            # prepare the intermediate outputs
            interm_outputs_coord = ref_enc[-1]                  # [bs, num_query, 4]
            interm_outputs_class = self.fc_enc_cls(hs_enc[-1])  # [bs, num_query, cls_out_channels]
        

        if self.dn_number > 0 and dn_meta is not None:
          
            outputs_class, outputs_coord, dn_outputs_class, dn_outputs_coord = dn_post_process_plus(outputs_class, outputs_coord, dn_meta)
        else:
            # When test, there is no dn_part
            dn_outputs_class, dn_outputs_coord = None, None

        return outputs_class, outputs_coord, interm_outputs_class, interm_outputs_coord, dn_outputs_class, dn_outputs_coord

    def forward_dummy(self, mlvl_feats,
                      img_metas,
                      input_query_label=None,
                      input_query_bbox=None,
                      attn_mask=None,
                      dn_meta=None):
        # breakpoint()
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
            mlvl_positional_encodings.append(self.positional_encoding(mlvl_masks[-1]))
            srcs.append(self.input_proj[l](feat))
        
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](mlvl_feats[-1])
                else:
                    src = self.input_proj[l](srcs[-1])
                
                srcs.append(src)
                mlvl_masks.append(F.interpolate(img_masks[None], size=src.shape[-2:]).to(torch.bool).squeeze(0))
                mlvl_positional_encodings.append(self.positional_encoding(mlvl_masks[-1]))
        
    
        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(srcs, \
                                                                             mlvl_masks, \
                                                                             input_query_bbox, \
                                                                             mlvl_positional_encodings, \
                                                                             input_query_label,\
                                                                             attn_mask,\
                                                                             fc_reg=self.fc_reg, fc_cls=self.fc_cls,\
                                                                             fc_enc_reg=self.fc_enc_reg, fc_enc_cls=self.fc_enc_cls)
     
        hs[0] += self.label_enc.weight[0, 0] * 0.0

     
        outputs_coord_list = []
        # breakpoint()
        for dec_lid, (layer_ref_sig, layer_fc_reg, layer_hs) in enumerate(zip(reference[:-1], self.fc_reg, hs)):
            layer_delta_unsig = layer_fc_reg(layer_hs)  
            layer_outputs_unsig = layer_delta_unsig  + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        # breakpoint()
        outputs_coord = torch.stack(outputs_coord_list)             # [num_dec_layer, bs, num_query + dn_size, 4]      

        # classification
        outputs_class = torch.stack([layer_fc_cls(layer_hs) for     # [num_dec_layer, bs, num_query + dn_size, cls_out_channels] 
                                     layer_fc_cls, layer_hs in zip(self.fc_cls, hs)])

        # encoder output
        if hs_enc is not None:
            # prepare the intermediate outputs
            interm_outputs_coord = ref_enc[-1]                  # [bs, num_query, 4]
            interm_outputs_class = self.fc_enc_cls(hs_enc[-1])  # [bs, num_query, cls_out_channels]
        

        # dn_components post process
        if self.dn_number > 0 and dn_meta is not None:
            pad_size_1 = dn_meta['pad_size_1']
            pad_size_2 = dn_meta['pad_size_2']
            consistency_outputs_class = outputs_class[:, :, :pad_size_1, :]
            consistency_outputs_coord = outputs_coord[:, :, :pad_size_1, :]

            dn_outputs_class = outputs_class[:, :, pad_size_1:pad_size_1 + pad_size_2, :]
            dn_outputs_coord = outputs_coord[:, :, pad_size_1:pad_size_1 + pad_size_2, :]

            outputs_class = outputs_class[:, :, pad_size_1 + pad_size_2:, :]
            outputs_coord = outputs_coord[:, :, pad_size_1 + pad_size_2:, :]
        else:
            consistency_outputs_class, consistency_outputs_coord, dn_outputs_class, dn_outputs_coord = None, None, None, None

        return hs, outputs_class, outputs_coord, interm_outputs_class, interm_outputs_coord, consistency_outputs_class, consistency_outputs_coord, dn_outputs_class, dn_outputs_coord

    @force_fp32(apply_to=('all_cls_scores', 'all_bbox_preds', 'enc_cls_scores', 'enc_bbox_preds', 'dn_cls_scores', 'dn_bbox_preds' ))
    def loss(self,
             all_cls_scores,        # [num_dec_layer, bs, num_query, cls_out_channels]
             all_bbox_preds,        # [num_dec_layer, bs, num_query, 4]
             enc_cls_scores,    
             enc_bbox_preds,
             dn_cls_scores,         # [num_dec_layer, bs, padding_size, cls_out_channels]
             dn_bbox_preds,         # [num_dec_layer, bs, padding_size, 4]
             gt_bboxes_list,
             gt_labels_list,
             gt_scores_list=None,   # [[num_gt,] x num_img] the soft scores to weight the corresponding classification loss and regression loss
             img_metas=None,
             dn_metas=None,
             gt_bboxes_ignore=None,
             is_pseudo_label=False):
        # import ipdb;ipdb.set_trace()
        
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'
        num_dec_layers = len(all_cls_scores)
        
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        # gt_scores used for the pseudo labels to the GMM pseudo label filtering
        all_gt_scores_list = [gt_scores_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]

        img_metas_list = [img_metas for _ in range(num_dec_layers)]

      
        num_imgs = len(gt_bboxes_list)
        dn_metas_valid = [dn_metas for i in range(num_imgs)]
        dn_metas_list = [dn_metas_valid for _ in range(num_dec_layers)]

        # regression and classification loss of head
        # use the gt_scores_list to make sure whether it's pseudo label or not
        losses_cls, losses_bbox, losses_iou, losses_bbox_xy, losses_bbox_hw  = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list, all_gt_scores_list,
            img_metas_list, all_gt_bboxes_ignore_list)

        # import ipdb; ipdb.set_trace()
        if self.in_warm_up and is_pseudo_label:
            dn_losses_cls = [torch.as_tensor(0.).to('cuda') for i in range(num_dec_layers)]
            dn_losses_bbox =  [torch.as_tensor(0.).to('cuda') for i in range(num_dec_layers)]
            dn_losses_iou =  [torch.as_tensor(0.).to('cuda') for i in range(num_dec_layers)]
            dn_losses_bbox_xy =  [torch.as_tensor(0.).to('cuda') for i in range(num_dec_layers)]
            dn_losses_bbox_hw = [torch.as_tensor(0.).to('cuda') for i in range(num_dec_layers)]
        else:
            dn_losses_cls, dn_losses_bbox, dn_losses_iou, dn_losses_bbox_xy, dn_losses_bbox_hw = multi_apply(
                self.loss_single_dn, dn_cls_scores, dn_bbox_preds,
                all_gt_bboxes_list, all_gt_labels_list, all_gt_scores_list,
                img_metas_list, dn_metas_list, all_gt_bboxes_ignore_list, is_pseudo_label=is_pseudo_label)
     

        loss_dict = dict()
        
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(img_metas))
            ]
            enc_loss_cls, enc_loss_bbox, enc_loss_iou, enc_loss_bbox_xy, enc_loss_bbox_hw = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list, gt_scores_list,
                                 img_metas, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_loss_bbox
            loss_dict['enc_loss_iou'] = enc_loss_iou
            loss_dict['enc_loss_bbox_xy'] = enc_loss_bbox_xy
            loss_dict['enc_loss_bbox_hw'] = enc_loss_bbox_hw


        # normal loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        loss_dict['loss_bbox_xy'] = losses_bbox_xy[-1]
        loss_dict['loss_bbox_hw'] = losses_bbox_hw[-1]


        # denosing loss from the last decoder layer
        loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
        loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
        loss_dict['dn_loss_iou'] = dn_losses_iou[-1]
        loss_dict['dn_loss_bbox_xy'] = dn_losses_bbox_xy[-1]
        loss_dict['dn_loss_bbox_hw'] = dn_losses_bbox_hw[-1]



        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i, loss_bbox_xy_i, loss_bbox_hw_i, \
            dn_loss_cls_i, dn_loss_bbox_i, dn_loss_iou_i, dn_loss_bbox_xy_i, dn_loss_bbox_hw_i  in zip(losses_cls[:-1],
                                                                                                        losses_bbox[:-1],
                                                                                                        losses_iou[:-1],
                                                                                                        losses_bbox_xy[:-1],
                                                                                                        losses_bbox_hw[:-1],
                                                                                                        dn_losses_cls[:-1],
                                                                                                        dn_losses_bbox[:-1],
                                                                                                        dn_losses_iou[:-1],
                                                                                                        dn_losses_bbox_xy[:-1],
                                                                                                        dn_losses_bbox_hw[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            loss_dict[f'd{num_dec_layer}.loss_bbox_xy'] = loss_bbox_xy_i
            loss_dict[f'd{num_dec_layer}.loss_bbox_hw'] = loss_bbox_hw_i

            loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = dn_loss_cls_i
            loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = dn_loss_bbox_i
            loss_dict[f'd{num_dec_layer}.dn_loss_iou'] = dn_loss_iou_i
            loss_dict[f'd{num_dec_layer}.dn_loss_bbox_xy'] = dn_loss_bbox_xy_i
            loss_dict[f'd{num_dec_layer}.dn_loss_bbox_hw'] = dn_loss_bbox_hw_i
            num_dec_layer += 1
        return loss_dict

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_scores_list=None,
                    img_metas=None,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        # breakpoint()
        # import ipdb;ipdb.set_trace()
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]

        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                        gt_bboxes_list, gt_labels_list, gt_scores_list,
                                        img_metas, gt_bboxes_ignore_list)
    
        if self.in_warm_up:
            # Note: in warm_up stage, one-to-many matching is for pseudo bbox and gt bbox
            # and the loss_weight for regression is slightly different
            (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
            norm_alignment_metrics_list, num_total_pos, num_total_neg) = cls_reg_targets
            
            labels = torch.cat(labels_list, 0)
            label_weights = torch.cat(label_weights_list, 0)
            bbox_targets = torch.cat(bbox_targets_list, 0)
            bbox_weights = torch.cat(bbox_weights_list, 0)
            norm_alignment_metrics = torch.cat(norm_alignment_metrics_list, 0)  # [num_query * num_img,]

            # classification loss
            cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
            # construct weighted avg_factor to match with the official DETR repo
            cls_avg_factor = num_total_pos * 1.0 + \
                num_total_neg * self.bg_cls_weight
            if self.sync_cls_avg_factor:
                cls_avg_factor = reduce_mean(
                    cls_scores.new_tensor([cls_avg_factor]))
            cls_avg_factor = max(cls_avg_factor, 1)

            # Compute the average sum of alignment metrics accross all gpus, for
            # normalization purposes
            sum_alignment_metrics = norm_alignment_metrics.new_tensor([norm_alignment_metrics.sum()])
            sum_alignment_metrics = torch.clamp(reduce_mean(sum_alignment_metrics), min=1.).item()

            # task align focal loss take the normalized alignment metric as the classification target
            loss_cls = self.loss_cls1(
                cls_scores.sigmoid(), labels, norm_alignment_metrics, avg_factor=sum_alignment_metrics)

            # Compute the average number of gt boxes across all gpus, for
            # normalization purposes
            num_total_pos = loss_cls.new_tensor([num_total_pos])
            num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

            # construct factors used for rescale bboxes
            factors = []
            for img_meta, bbox_pred in zip(img_metas, bbox_preds):
                img_h, img_w, _ = img_meta['img_shape']
                factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                            img_h]).unsqueeze(0).repeat(
                                                bbox_pred.size(0), 1)
                factors.append(factor)
            factors = torch.cat(factors, 0)

            bg_class_ind = self.num_classes
            pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)

            bbox_preds = bbox_preds.reshape(-1, 4)
            bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
            bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

            if len(pos_inds) > 0:
                pos_bboxes = bboxes[pos_inds]
                pos_bboxes_gt = bboxes_gt[pos_inds]
               
                pos_bbox_weights = bbox_weights[pos_inds]       

                reg_avg_factor = norm_alignment_metrics.new_tensor([pos_bbox_weights[:, 0].sum()])
                reg_avg_factor = torch.clamp(reduce_mean(reg_avg_factor), min=1.).item()

             
                loss_iou = self.loss_iou(
                    pos_bboxes, pos_bboxes_gt, pos_bbox_weights, avg_factor=reg_avg_factor)

                # regression L1 loss
                pos_bbox_preds = bbox_preds[pos_inds]
                pos_bbox_targets = bbox_targets[pos_inds]
                
                loss_bbox = self.loss_bbox(
                    pos_bbox_preds, pos_bbox_targets, pos_bbox_weights, avg_factor=reg_avg_factor)
                loss_bbox_xy = self.loss_bbox(
                    pos_bbox_preds[..., :2], pos_bbox_targets[..., :2], pos_bbox_weights[..., :2], avg_factor=reg_avg_factor)
                loss_bbox_hw = self.loss_bbox(
                    pos_bbox_preds[..., 2:], pos_bbox_targets[..., 2:], pos_bbox_weights[..., 2:], avg_factor=reg_avg_factor)
            else:
                sum_pos_alignment_metrics = norm_alignment_metrics.new_tensor([0])
                sum_pos_alignment_metrics = torch.clamp(reduce_mean(sum_pos_alignment_metrics), min=1.).item()
                loss_bbox = bbox_pred.sum() * 0
                loss_iou = bbox_pred.sum() * 0
                loss_bbox_xy = bbox_pred.sum() * 0
                loss_bbox_hw = bbox_pred.sum() * 0

            return loss_cls, loss_bbox, loss_iou , loss_bbox_xy, loss_bbox_hw

        else:
            # after warm_up, just switch back to the original DETR like assign and loss
            (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
            num_total_pos, num_total_neg) = cls_reg_targets
            labels = torch.cat(labels_list, 0)
            label_weights = torch.cat(label_weights_list, 0)
            bbox_targets = torch.cat(bbox_targets_list, 0)
            bbox_weights = torch.cat(bbox_weights_list, 0)

            # classification loss
            cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
            # construct weighted avg_factor to match with the official DETR repo
            cls_avg_factor = num_total_pos * 1.0 + \
                num_total_neg * self.bg_cls_weight
            if self.sync_cls_avg_factor:
                cls_avg_factor = reduce_mean(
                    cls_scores.new_tensor([cls_avg_factor]))
            cls_avg_factor = max(cls_avg_factor, 1)

            loss_cls = self.loss_cls2(
                cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

            reg_avg_factor = bbox_weights.new_tensor([len(torch.nonzero(bbox_weights.sum(-1) > 0, as_tuple=False).squeeze().unique())])
            reg_avg_factor = torch.clamp(reduce_mean(reg_avg_factor), min=1.).item()


            # construct factors used for rescale bboxes
            factors = []
            for img_meta, bbox_pred in zip(img_metas, bbox_preds):
                img_h, img_w, _ = img_meta['img_shape']
                factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                            img_h]).unsqueeze(0).repeat(
                                                bbox_pred.size(0), 1)
                factors.append(factor)
            factors = torch.cat(factors, 0)

        
            bbox_preds = bbox_preds.reshape(-1, 4)
            bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
            bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

            # regression IoU loss, defaultly GIoU loss
            loss_iou = self.loss_iou(
                bboxes, bboxes_gt, bbox_weights, avg_factor=reg_avg_factor)

            # regression L1 loss
            loss_bbox = self.loss_bbox(
                bbox_preds, bbox_targets, bbox_weights, avg_factor=reg_avg_factor)
            loss_bbox_xy = self.loss_bbox(
                bbox_preds[..., :2], bbox_targets[..., :2], bbox_weights[..., :2], avg_factor=reg_avg_factor)
            loss_bbox_hw = self.loss_bbox(
                bbox_preds[..., 2:], bbox_targets[..., 2:], bbox_weights[..., 2:], avg_factor=reg_avg_factor)

            return loss_cls, loss_bbox, loss_iou , loss_bbox_xy, loss_bbox_hw


    def loss_single_dn(self,
                       cls_scores,
                       bbox_preds,
                       gt_bboxes_list,
                       gt_labels_list,
                       gt_scores_list=None,
                       img_metas=None,
                       dn_metas=None,
                       gt_bboxes_ignore_list=None,
                       is_pseudo_label=False):
    
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]

        # Note: do the pre-process with pseudo labels
       
        cls_reg_targets = self.get_targets_dn(cls_scores_list, bbox_preds_list,
                                    gt_bboxes_list, gt_labels_list,
                                    img_metas, dn_metas, gt_bboxes_ignore_list, is_pseudo_label=is_pseudo_label)
       

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls2(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)


        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(img_metas, bbox_preds):
            img_h, img_w, _ = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        loss_bbox_xy = self.loss_bbox(
            bbox_preds[..., :2], bbox_targets[..., :2], bbox_weights[..., :2], avg_factor=num_total_pos)
        loss_bbox_hw = self.loss_bbox(
            bbox_preds[..., 2:], bbox_targets[..., 2:], bbox_weights[..., 2:], avg_factor=num_total_pos)

        return loss_cls, loss_bbox, loss_iou , loss_bbox_xy, loss_bbox_hw

    def _get_target_single_dn(self,
                              cls_score,
                              bbox_pred,
                              gt_bboxes,
                              gt_labels,
                              img_meta=None,
                              dn_meta=None,
                              gt_bboxes_ignore=None,
                              is_pseudo_label=False):
        """Get the dn_part learning target for a single image
        cls_score: [pad_size, cls_out_channels]
        bbox_pred: [pad_size, 4]
        gt_bboxes: [num_gt, 4]
        gt_labels: [num_gt]

        """
     
        # breakpoint()
        # import ipdb;ipdb.set_trace()
        num_bboxes = bbox_pred.size(0)

      
        single_pad, scalar = self.prep_for_dn(dn_meta, is_pseudo_label=is_pseudo_label) 
        # import ipdb; ipdb.set_trace()
        assert num_bboxes == single_pad * scalar, "The dn_part object query number is incorrect, plz check!"

        if len(gt_labels) > 0:
            # gt_labels: cls labels for a single image [num_gt,]
            t = torch.arange(0, len(gt_labels)).long().cuda()    
            t = t.unsqueeze(0).repeat(scalar, 1)
            tgt_idx = t.flatten()       # tgt_idx: [num_gt x dn_groups] from the padding_size = single_pad x dn_groups
         
            output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
            output_idx = output_idx.flatten()
        else:
            output_idx = tgt_idx = torch.tensor([]).long().cuda()

        
        pos_inds = output_idx
        neg_inds = output_idx + single_pad // 2

        pos_assigned_gt_inds = tgt_idx

        pos_gt_bboxes = gt_bboxes[tgt_idx, :]
        
     
        # label targets
        labels = gt_labels.new_full((single_pad * scalar,), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds].long()

        # Note: becareful there is no gt bbox situation!!!
        if pos_inds.size(0) > 0:
            # There is some gt labels in this image
            label_weights = gt_labels.new_ones(single_pad * scalar)        
        else:
            # There is no gt label in this image, don't calcualte this loss
            label_weights = gt_labels.new_zeros(single_pad * scalar)
        

        # bbox_targets
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w, _ = img_meta['img_shape']

        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        pos_gt_bboxes_normalized = pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)
      
    def get_targets_dn(self,
                       cls_scores_list,
                       bbox_preds_list,
                       gt_bboxes_list,
                       gt_labels_list,
                       img_metas,
                       dn_metas,
                       gt_bboxes_ignore_list=None,
                       is_pseudo_label=False):
        """Get targets for batched images of denosing part.
        Note in the 
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]
        
        (labels_list, label_weights_list, bbox_targets_list,
        bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
        self._get_target_single_dn, cls_scores_list, bbox_preds_list,
        gt_bboxes_list, gt_labels_list, img_metas, dn_metas, gt_bboxes_ignore_list, is_pseudo_label=is_pseudo_label)

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_scores_list=None,
                    img_metas=None,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            tuple: a tuple containing the following targets.

                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        # Note: we use the gt_score to judge whether it's pseudo bboxes
        # or gt bboxes
        if gt_scores_list is None:
            gt_scores_list = [
                gt_scores_list for _ in range(num_imgs)
            ]

        if self.in_warm_up:
            # Note: in warm_up stage, we apply the TOOD stype assigner to do one-to-many assign
            (labels_list, label_weights_list, bbox_targets_list,
            bbox_weights_list, norm_alignment_metrics_list, pos_inds_list, neg_inds_list) = multi_apply(
                self._get_target_single, cls_scores_list, bbox_preds_list,
                gt_bboxes_list, gt_labels_list, gt_scores_list, img_metas, gt_bboxes_ignore_list)
            
            # return the assigned results
            num_total_pos = sum((inds.numel() for inds in pos_inds_list))
            num_total_neg = sum((inds.numel() for inds in neg_inds_list))
            return (labels_list, label_weights_list, bbox_targets_list,
                    bbox_weights_list, norm_alignment_metrics_list, num_total_pos, num_total_neg)

        else:
            # Note: after warm_up, we apply the original DETR stype assigner to do one-to-one assign
            (labels_list, label_weights_list, bbox_targets_list,
            bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
                self._get_target_single, cls_scores_list, bbox_preds_list,
                gt_bboxes_list, gt_labels_list, gt_scores_list, img_metas, gt_bboxes_ignore_list)

            num_total_pos = sum((inds.numel() for inds in pos_inds_list))
            num_total_neg = sum((inds.numel() for inds in neg_inds_list))
            return (labels_list, label_weights_list, bbox_targets_list,
                    bbox_weights_list, num_total_pos, num_total_neg)


    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_bboxes,
                           gt_labels,
                           gt_scores=None,
                           img_meta=None,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            img_meta (dict): Meta information for one image.
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        # breakpoint()
        num_bboxes = bbox_pred.size(0)
        
        if self.in_warm_up:
            # in warm_up, use TOOD stype one-to-many assigner
            assign_result = self.assigner1.assign(bbox_pred, cls_score.sigmoid(), gt_bboxes,
                                             gt_labels, img_meta,
                                             gt_bboxes_ignore)
            assign_ious = assign_result.max_overlaps
            # make the unmatched Iou to 0
            INF = 100000000
            assign_ious[assign_ious == -INF] = 0
            assign_metrics = assign_result.assign_metrics

            sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                                gt_bboxes)
            pos_inds = sampling_result.pos_inds
            neg_inds = sampling_result.neg_inds

            # label targets
            labels = gt_bboxes.new_full((num_bboxes, ),
                                        self.num_classes,
                                        dtype=torch.long)
            label_weights = gt_bboxes.new_ones(num_bboxes)

            # bbox targets
            bbox_targets = torch.zeros_like(bbox_pred)
            bbox_weights = torch.zeros_like(bbox_pred)

            # point-based
            img_h, img_w, _ = img_meta['img_shape']

            # DETR regress the relative position of boxes (cxcywh) in the image.
            # Thus the learning target should be normalized by the image size, also
            # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
            factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
            pos_gt_bboxes_normalized = sampling_result.pos_gt_bboxes / factor
            pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        
            bbox_targets[pos_inds, :] = pos_gt_bboxes_targets

            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds].long()
            
            # take the instance normalization for each gt, make the largest alignment metric
            # score of the assigned proposal equals to the largest IoU
            norm_alignment_metrics = assign_metrics.new_zeros(labels.shape[0])
            pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds
            for gt_inds in torch.unique(pos_assigned_gt_inds):
                gt_class_inds = pos_inds[pos_assigned_gt_inds == gt_inds]
                pos_alignment_metrics = assign_metrics[gt_class_inds]
                pos_ious = assign_ious[gt_class_inds]
                pos_norm_alignment_metrics = pos_alignment_metrics / (pos_alignment_metrics.max() + 10e-8) * pos_ious.max()
                norm_alignment_metrics[gt_class_inds] = pos_norm_alignment_metrics

         
            # make the bbox_weights to the alignment metrics 
            # Note: when in the warm-up stage, we don't decouple the classificaiton and regression
            bbox_weights[pos_inds, :] = norm_alignment_metrics[pos_inds].unsqueeze(-1)

            return (labels, label_weights, bbox_targets, bbox_weights, norm_alignment_metrics, pos_inds,
                    neg_inds) 

        else:
            # after warm_up, we take the original DETR stype assignment 
            # assigner and sampler
            assign_result = self.assigner2.assign(bbox_pred, cls_score, gt_bboxes,
                                                gt_labels, img_meta,
                                                gt_bboxes_ignore)
            sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                                gt_bboxes)
            pos_inds = sampling_result.pos_inds
            neg_inds = sampling_result.neg_inds

            # label targets
            labels = gt_bboxes.new_full((num_bboxes, ),
                                        self.num_classes,
                                        dtype=torch.long)
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds].long()
            
            label_weights = gt_bboxes.new_ones(num_bboxes)

        
            # bbox targets
            bbox_targets = torch.zeros_like(bbox_pred)
            bbox_weights = torch.zeros_like(bbox_pred)

        
            bbox_weights[pos_inds] = 1.0


            img_h, img_w, _ = img_meta['img_shape']

            # DETR regress the relative position of boxes (cxcywh) in the image.
            # Thus the learning target should be normalized by the image size, also
            # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
            factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
            pos_gt_bboxes_normalized = sampling_result.pos_gt_bboxes / factor
            pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
            bbox_targets[pos_inds] = pos_gt_bboxes_targets
            return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                    neg_inds)

    # over-write because img_metas are needed as inputs for bbox_head.
    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_scores=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      curr_step=None,
                      is_pseudo_label=False,
                      **kwargs):
        """Forward function for training mode.

        Args:
            x (list[Tensor]): Features from backbone.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        if curr_step is not None:
            # set the warm_up state
            if curr_step < self.warm_up_step:
                self.in_warm_up = True
            else:
                self.in_warm_up = False

        # prepare the dn_components
        if self.dn_number > 0:
            # prepare the target
                
            # NOTE: need the labels and boxes as the list of each images
            targets = dict()
            targets['labels'] = gt_labels
          
            normalized_boxes = []
            for img_meta, gt_bbox in zip(img_metas, gt_bboxes):
                img_h, img_w, _ = img_meta['img_shape']
                factor = gt_bbox.new_tensor([img_w, img_h, img_w,
                                            img_h]).unsqueeze(0).repeat(
                                                gt_bbox.size(0), 1)
                gt_bbox_ = bbox_xyxy_to_cxcywh(gt_bbox)
                normalized_boxes.append(gt_bbox_ / factor)

            targets['boxes'] = normalized_boxes

            input_query_label, input_query_bbox, attn_mask, dn_meta = prepare_for_cdn_plus(dn_args=(targets, self.dn_number, self.dn_label_noise_ratio, self.dn_box_noise_scale),
                                training=True, num_queries=self.num_query, num_classes=self.num_classes,
                                hidden_dim=self.embed_dims, label_enc=self.label_enc)
        else:
            input_query_bbox = input_query_label = attn_mask = dn_meta = None

        assert proposal_cfg is None, '"proposal_cfg" must be None'
        # change the forward method's parameter
        # import ipdb;ipdb.set_trace()
        outs = self(x, img_metas, input_query_label, input_query_bbox, attn_mask, dn_meta)
        # Note: gt_scores is consider when we use pseudo bbox 
        if gt_scores is None:
            loss_inputs = outs + (gt_bboxes, gt_labels)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, gt_scores)
        losses = self.loss(*loss_inputs, img_metas=img_metas, dn_metas=dn_meta, gt_bboxes_ignore=gt_bboxes_ignore, is_pseudo_label=is_pseudo_label)
        return losses

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def get_bboxes(self,
                   all_cls_scores,
                   all_bbox_preds,
                   enc_cls_scores,
                   enc_bbox_preds,
                   dn_cls_scores,
                   dn_bbox_preds,
                   img_metas,
                   rescale=False,
                   for_pseudo_label=False):
        """Transform network outputs for a batch into bbox predictions.

        Args:
            all_cls_scores_list (list[Tensor]): Classification outputs
                for each feature level. Each is a 4D-tensor with shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds_list (list[Tensor]): Sigmoid regression
                outputs for each feature level. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If True, return boxes in original
                image space. Default False.

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple. \
                The first item is an (n, 5) tensor, where the first 4 columns \
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the \
                5-th column is a score between 0 and 1. The second item is a \
                (n,) tensor where each item is the predicted class label of \
                the corresponding box.
        """
        # NOTE defaultly only using outputs from the last feature level,
        # and only the outputs from the last decoder layer is used.
        cls_scores = all_cls_scores[-1]
        bbox_preds = all_bbox_preds[-1]

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score, bbox_pred,
                                                img_shape, scale_factor,
                                                rescale,
                                                for_pseudo_label=for_pseudo_label)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_score,
                           bbox_pred,
                           img_shape,
                           scale_factor,
                           rescale=False,
                           for_pseudo_label=False):
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (cx, cy, w, h) and
                shape [num_query, 4].
            img_shape (tuple[int]): Shape of input image, (height, width, 3).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool, optional): If True, return boxes in original image
                space. Default False.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels.

                - det_bboxes: Predicted bboxes with shape [num_query, 5], \
                    where the first 4 columns are bounding box positions \
                    (tl_x, tl_y, br_x, br_y) and the 5-th column are scores \
                    between 0 and 1.
                - det_labels: Predicted labels of the corresponding box with \
                    shape [num_query].
        """
        # import ipdb; ipdb.set_trace()
        assert len(cls_score) == len(bbox_pred)
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)
        # exclude background
        if self.loss_cls2.use_sigmoid:
            cls_score = cls_score.sigmoid()
          
            if self.in_warm_up or for_pseudo_label:
                # import pdb; pdb.set_trace()
                nms = dict(type='nms', iou_threshold=0.6, split_thr=-1)
                score_thr = 0.01
                num_class = cls_score.shape[-1]
                # append the background class
                padding = cls_score.new_zeros(cls_score.shape[0], 1)
                cls_score = torch.cat([cls_score, padding], dim=1)
                
                bbox_pred  = bbox_cxcywh_to_xyxy(bbox_pred)
                bbox_pred[:, 0::2] = bbox_pred[:, 0::2] * img_shape[1]
                bbox_pred[:, 1::2] = bbox_pred[:, 1::2] * img_shape[0]
                bbox_pred[:, 0::2].clamp_(min=0, max=img_shape[1])
                bbox_pred[:, 1::2].clamp_(min=0, max=img_shape[0])

                if rescale:
                    bbox_pred /= bbox_pred.new_tensor(scale_factor)
                # apply multi-class nms to filter the duplicate detection boxes
                det_bboxes, det_labels = multiclass_nms(
                    bbox_pred,
                    cls_score,
                    score_thr,
                    nms,
                    max_per_img)
                return det_bboxes, det_labels
            else:
                scores, indexes = cls_score.view(-1).topk(max_per_img)
                det_labels = indexes % self.num_classes
                bbox_index = indexes // self.num_classes
                bbox_pred = bbox_pred[bbox_index]
        else:
            NotImplementedError

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            det_bboxes /= det_bboxes.new_tensor(scale_factor)
        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(1)), -1)

        return det_bboxes, det_labels

    def simple_test_bboxes(self, feats, img_metas, rescale=False, curr_step=None, for_pseudo_label=False):
        """Test det bboxes without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,)
        """
        # import ipdb;ipdb.set_trace()
        if curr_step is not None:
            if curr_step <= self.warm_up_step:
                self.in_warm_up = True
            else:
                self.in_warm_up = False

        # forward of this head requires img_metas
        outs = self.forward(feats, img_metas)
        results_list = self.get_bboxes(*outs, img_metas, rescale=rescale, for_pseudo_label=for_pseudo_label)
        return results_list

    def forward_onnx(self, feats, img_metas):
        """Forward function for exporting to ONNX.

        Over-write `forward` because: `masks` is directly created with
        zero (valid position tag) and has the same spatial size as `x`.
        Thus the construction of `masks` is different from that in `forward`.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[list[Tensor], list[Tensor]]: Outputs for all scale levels.

                - all_cls_scores_list (list[Tensor]): Classification scores \
                    for each scale level. Each is a 4D-tensor with shape \
                    [nb_dec, bs, num_query, cls_out_channels]. Note \
                    `cls_out_channels` should includes background.
                - all_bbox_preds_list (list[Tensor]): Sigmoid regression \
                    outputs for each scale level. Each is a 4D-tensor with \
                    normalized coordinate format (cx, cy, w, h) and shape \
                    [nb_dec, bs, num_query, 4].
        """
        num_levels = len(feats)
        img_metas_list = [img_metas for _ in range(num_levels)]
        return multi_apply(self.forward_single_onnx, feats, img_metas_list)

    def forward_single_onnx(self, x, img_metas):
        """"Forward function for a single feature level with ONNX exportation.

        Args:
            x (Tensor): Input feature from backbone's single stage, shape
                [bs, c, h, w].
            img_metas (list[dict]): List of image information.

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head,
                shape [nb_dec, bs, num_query, cls_out_channels]. Note
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression
                head with normalized coordinate format (cx, cy, w, h).
                Shape [nb_dec, bs, num_query, 4].
        """
        # Note `img_shape` is not dynamically traceable to ONNX,
        # since the related augmentation was done with numpy under
        # CPU. Thus `masks` is directly created with zeros (valid tag)
        # and the same spatial shape as `x`.
        # The difference between torch and exported ONNX model may be
        # ignored, since the same performance is achieved (e.g.
        # 40.1 vs 40.1 for DETR)
        batch_size = x.size(0)
        h, w = x.size()[-2:]
        masks = x.new_zeros((batch_size, h, w))  # [B,h,w]

        x = self.input_proj(x)
        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(
            masks.unsqueeze(1), size=x.shape[-2:]).to(torch.bool).squeeze(1)
        pos_embed = self.positional_encoding(masks)
        outs_dec, _ = self.transformer(x, masks, self.query_embedding.weight,
                                       pos_embed)

        all_cls_scores = self.fc_cls(outs_dec)
        all_bbox_preds = self.fc_reg(self.activate(
            self.reg_ffn(outs_dec))).sigmoid()
        return all_cls_scores, all_bbox_preds

    def onnx_export(self, all_cls_scores_list, all_bbox_preds_list, img_metas):
        """Transform network outputs into bbox predictions, with ONNX
        exportation.

        Args:
            all_cls_scores_list (list[Tensor]): Classification outputs
                for each feature level. Each is a 4D-tensor with shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds_list (list[Tensor]): Sigmoid regression
                outputs for each feature level. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            img_metas (list[dict]): Meta information of each image.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        assert len(img_metas) == 1, \
            'Only support one input image while in exporting to ONNX'

        cls_scores = all_cls_scores_list[-1][-1]
        bbox_preds = all_bbox_preds_list[-1][-1]

        # Note `img_shape` is not dynamically traceable to ONNX,
        # here `img_shape_for_onnx` (padded shape of image tensor)
        # is used.
        img_shape = img_metas[0]['img_shape_for_onnx']
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)
        batch_size = cls_scores.size(0)
        # `batch_index_offset` is used for the gather of concatenated tensor
        batch_index_offset = torch.arange(batch_size).to(
            cls_scores.device) * max_per_img
        batch_index_offset = batch_index_offset.unsqueeze(1).expand(
            batch_size, max_per_img)

        # supports dynamical batch inference
        if self.loss_cls2.use_sigmoid:
            cls_scores = cls_scores.sigmoid()
            scores, indexes = cls_scores.view(batch_size, -1).topk(
                max_per_img, dim=1)
            det_labels = indexes % self.num_classes
            bbox_index = indexes // self.num_classes
            bbox_index = (bbox_index + batch_index_offset).view(-1)
            bbox_preds = bbox_preds.view(-1, 4)[bbox_index]
            bbox_preds = bbox_preds.view(batch_size, -1, 4)
        else:
            scores, det_labels = F.softmax(
                cls_scores, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img, dim=1)
            bbox_index = (bbox_index + batch_index_offset).view(-1)
            bbox_preds = bbox_preds.view(-1, 4)[bbox_index]
            det_labels = det_labels.view(-1)[bbox_index]
            bbox_preds = bbox_preds.view(batch_size, -1, 4)
            det_labels = det_labels.view(batch_size, -1)

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_preds)
        # use `img_shape_tensor` for dynamically exporting to ONNX
        img_shape_tensor = img_shape.flip(0).repeat(2)  # [w,h,w,h]
        img_shape_tensor = img_shape_tensor.unsqueeze(0).unsqueeze(0).expand(
            batch_size, det_bboxes.size(1), 4)
        det_bboxes = det_bboxes * img_shape_tensor
        # dynamically clip bboxes
        x1, y1, x2, y2 = det_bboxes.split((1, 1, 1, 1), dim=-1)
        from mmdet.core.export import dynamic_clip_for_onnx
        x1, y1, x2, y2 = dynamic_clip_for_onnx(x1, y1, x2, y2, img_shape)
        det_bboxes = torch.cat([x1, y1, x2, y2], dim=-1)
        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(-1)), -1)

        return det_bboxes, det_labels
