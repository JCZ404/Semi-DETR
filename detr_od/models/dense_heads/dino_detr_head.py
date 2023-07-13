# Copyright (c) OpenMMLab. All rights reserved.
import math
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Linear, build_activation_layer, bias_init_with_prob
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
from mmcv.runner import force_fp32

from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models.utils import build_transformer
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.utils.transformer import inverse_sigmoid

from  mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead


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
class DINODETRHead(AnchorFreeHead):
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
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 train_cfg=dict(
                     assigner=dict(
                         type='HungarianAssigner',
                         cls_cost=dict(type='ClassificationCost', weight=1.),
                         reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                         iou_cost=dict(
                             type='IoUCost', iou_mode='giou', weight=2.0))),
                 test_cfg=dict(max_per_img=100),
                 init_cfg=None,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is DINODETRHead):
            assert isinstance(class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']
            assert loss_cls['loss_weight'] == assigner['cls_cost']['weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            assert loss_bbox['loss_weight'] == assigner['reg_cost'][
                'weight'], 'The regression L1 weight for loss and matcher ' \
                'should be exactly the same.'
            assert loss_iou['loss_weight'] == assigner['iou_cost']['weight'], \
                'The regression iou weight for loss and matcher should be' \
                'exactly the same.'
            self.assigner = build_assigner(assigner)
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
        self.loss_cls = build_loss(loss_cls)
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

        if self.loss_cls.use_sigmoid:
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
        # In DINO, cause we use two stage method to generate the 
        # image content depended object query, so we actually don't 
        # intitialize the refpoint embedding manually
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

    def prep_for_dn(self, dn_meta):
        """Get some dn part information
        """
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
        # breakpoint()
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
        
        # Transformer forward
        # hs: list with each has the shape of [bs, pad_size + num_query, embed_dim], length 6
        # reference: list with each has the shape of [bs, pad_size + num_query, 4], 
        # hs_enc: list with each has the shape of [bs, num_query, embed_dim], length 1
        # ref_enc: list with each has the shape of [bs, num_query, 4], length 1
        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(srcs, \
                                                                             mlvl_masks, \
                                                                             input_query_bbox, \
                                                                             mlvl_positional_encodings, \
                                                                             input_query_label,\
                                                                             attn_mask,\
                                                                             fc_reg=self.fc_reg, fc_cls=self.fc_cls,\
                                                                             fc_enc_reg=self.fc_enc_reg, fc_enc_cls=self.fc_enc_cls)
        # In case num object=0
        # hs: [num_dec_layer, bs, num_query + dn_size, embed_dim]
        # label_enc: [num_class + 1, embed_dim]
        hs[0] += self.label_enc.weight[0, 0] * 0.0

        # regression
        # Deformable-DETR like iter anchor update
        # reference_before_sigmoid = inverse_sigmoid(reference[:-1]) # n_dec, bs, nq, 4
        # self.fc_reg是多个共享权重的MLP head用来预测box的location
        # 这里在同时处理了原始的object query和dn_part的object query
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
            # ref_enc是初始的从encoder feature map筛选出来送入到decoder中的refpoint_embed, sigmoid normalized
            interm_outputs_coord = ref_enc[-1]                  # [bs, num_query, 4]
            # 这个用来做分类的是用的和fc_class相同结构，但是不同实例的MLP Head
            interm_outputs_class = self.fc_enc_cls(hs_enc[-1])  # [bs, num_query, cls_out_channels]
        

        # dn_components post process
        if self.dn_number > 0 and dn_meta is not None:
            # 从模型的预测中，将dn_part部分的label和bbox的预测拆分出来
            # outputs_class: [num_dec_layer, bs, num_query, cls_out_channels]
            # outputs_coord: [num_dec_layer, bs, num_query, 4]
            # dn_outputs_class: [num_dec_layer, bs, padding_size, cls_out_channels]
            # dn_outputs_coord: [num_dec_layer, bs, padding_size, 4]
            # NOTE: sometimes there is no dn_part
            outputs_class, outputs_coord, dn_outputs_class, dn_outputs_coord = dn_post_process(outputs_class, outputs_coord, dn_meta)
        else:
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
        
        # Transformer forward
        # hs: list with each has the shape of [bs, pad_size + num_query, embed_dim], length 6
        # reference: list with each has the shape of [bs, pad_size + num_query, 4], 
        # hs_enc: list with each has the shape of [bs, num_query, embed_dim], length 1
        # ref_enc: list with each has the shape of [bs, num_query, 4], length 1
        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(srcs, \
                                                                             mlvl_masks, \
                                                                             input_query_bbox, \
                                                                             mlvl_positional_encodings, \
                                                                             input_query_label,\
                                                                             attn_mask,\
                                                                             fc_reg=self.fc_reg, fc_cls=self.fc_cls,\
                                                                             fc_enc_reg=self.fc_enc_reg, fc_enc_cls=self.fc_enc_cls)
        # In case num object=0
        # hs: [num_dec_layer, bs, num_query + dn_size, embed_dim]
        # label_enc: [num_class + 1, embed_dim]
        hs[0] += self.label_enc.weight[0, 0] * 0.0

        # regression
        # Deformable-DETR like iter anchor update
        # reference_before_sigmoid = inverse_sigmoid(reference[:-1]) # n_dec, bs, nq, 4
        # self.fc_reg是多个共享权重的MLP head用来预测box的location
        # 这里在同时处理了原始的object query和dn_part的object query
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
            # ref_enc是初始的从encoder feature map筛选出来送入到decoder中的refpoint_embed, sigmoid normalized
            interm_outputs_coord = ref_enc[-1]                  # [bs, num_query, 4]
            # 这个用来做分类的是用的和fc_class相同结构，但是不同实例的MLP Head
            interm_outputs_class = self.fc_enc_cls(hs_enc[-1])  # [bs, num_query, cls_out_channels]
        

        # dn_components post process
        if self.dn_number > 0 and dn_meta is not None:
            # 从模型的预测中，将dn_part部分的label和bbox的预测拆分出来
            # outputs_class: [num_dec_layer, bs, num_query, cls_out_channels]
            # outputs_coord: [num_dec_layer, bs, num_query, 4]
            # dn_outputs_class: [num_dec_layer, bs, padding_size, cls_out_channels]
            # dn_outputs_coord: [num_dec_layer, bs, padding_size, 4]
            # NOTE: sometimes there is no dn_part
            outputs_class, outputs_coord, dn_outputs_class, dn_outputs_coord = dn_post_process(outputs_class, outputs_coord, dn_meta)
        else:
            dn_outputs_class, dn_outputs_coord = None, None

        return hs, outputs_class, outputs_coord, interm_outputs_class, interm_outputs_coord, dn_outputs_class, dn_outputs_coord


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
             decouple=False):
        
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'
        # breakpoint()
        num_dec_layers = len(all_cls_scores)
        
        # 多个decoder layer都需要进行matcher和loss的计算
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_scores_list = [gt_scores_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]

        img_metas_list = [img_metas for _ in range(num_dec_layers)]

        # 注意计算逻辑，先decoder layer，然后再是每张图片，所以dn_metas要和image metas
        # 一样，进行复制
        num_imgs = len(gt_bboxes_list)
        dn_metas_valid = [dn_metas for i in range(num_imgs)]
        dn_metas_invalid = [None for i in range(num_imgs)]

        dn_metas_list = [dn_metas_valid for _ in range(num_dec_layers)]
        dn_metas_list_ = [dn_metas_invalid for _ in range(num_dec_layers)]

        # 对每个decoder layer进行object query loss的计算
        losses_cls, losses_bbox, losses_iou, losses_bbox_xy, losses_bbox_hw  = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list, all_gt_scores_list, # add the gt_scores_list
            img_metas_list, dn_metas_list_,
            all_gt_bboxes_ignore_list, decouple=decouple)

        # 对每个decoder layer进行denosing query loss的计算
        if dn_cls_scores is None or dn_bbox_preds is None:
            # in case there is no dn_part
            # import ipdb; ipdb.set_trace()
            dn_losses_cls = [torch.as_tensor(0.).to('cuda') for i in range(num_dec_layers)]
            dn_losses_bbox =  [torch.as_tensor(0.).to('cuda') for i in range(num_dec_layers)]
            dn_losses_iou =  [torch.as_tensor(0.).to('cuda') for i in range(num_dec_layers)]
            dn_losses_bbox_xy =  [torch.as_tensor(0.).to('cuda') for i in range(num_dec_layers)]
            dn_losses_bbox_hw = [torch.as_tensor(0.).to('cuda') for i in range(num_dec_layers)]
        else:
            # import ipdb; ipdb.set_trace()
            dn_losses_cls, dn_losses_bbox, dn_losses_iou, dn_losses_bbox_xy, dn_losses_bbox_hw = multi_apply(
                self.loss_single, dn_cls_scores, dn_bbox_preds,
                all_gt_bboxes_list, all_gt_labels_list, all_gt_scores_list,
                img_metas_list, dn_metas_list,
                all_gt_bboxes_ignore_list, decouple=decouple)


        loss_dict = dict()
        
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(img_metas))
            ]
            # be careful that the encoder prediction, we only calculate a single layer loss
            enc_loss_cls, enc_loss_bbox, enc_loss_iou, enc_loss_bbox_xy, enc_loss_bbox_hw = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list, gt_scores_list,    #
                                 img_metas, dn_metas_list_[0], gt_bboxes_ignore, decouple=decouple)
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
                    dn_metas=None,
                    gt_bboxes_ignore_list=None,
                    decouple=False):
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
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]

        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                        gt_bboxes_list, gt_labels_list, gt_scores_list,
                                        img_metas, dn_metas, gt_bboxes_ignore_list, decouple=decouple)
    

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

     
        loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)
        

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
        # change the avg_factor of regression as the count of nonzero's bbox_weight
        reg_avg_factor = len(torch.nonzero(bbox_weights.sum(dim=-1) > 0).squeeze().unique())
        reg_avg_factor = reduce_mean(
                bbox_preds.new_tensor([reg_avg_factor]))
        reg_avg_factor = max(reg_avg_factor, 1)
        
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


    def _get_target_single_dn(self,
                              cls_score,
                              bbox_pred,
                              gt_bboxes,
                              gt_labels,
                              gt_scores=None,
                              img_meta=None,
                              dn_meta=None,
                              gt_bboxes_ignore=None):
        """Get the dn_part learning target for a single image
        cls_score: [pad_size, cls_out_channels]
        bbox_pred: [pad_size, 4]
        gt_bboxes: [num_gt, 4]
        gt_labels: [num_gt]

        """
        assert gt_scores is None, "DN Part don't apply on the unaccurate pseudo bboxes! So don't provide the soft gt_scores!"
        # NOTE: loss_single_dn和loss_single的区别在于这里我们不需要像loss_single一样通过get_target来获得
        # label_target，bbox_target等，因为这里我们直接可以通过dn_meta中的原始的构造dn_part过程的信息
        # 来获得每个object query对应的target
        # breakpoint()
        num_bboxes = bbox_pred.size(0)

        # 从dn_meta中，构造dn_part的target
        # single_pad: 在构造dn_part输入的时候，每张图片的padding的大小
        # scalar: 在构造dn_part输入的时候的dn_groups
        # single_pad: 每张图片的gt_label的padding的大小，这个single_pad包括补齐batch data
        # 和在后面padding一份positive和negative的样本
        # scalar: dn_groups, 就是single_pad的重复的次数
        single_pad, scalar = self.prep_for_dn(dn_meta) 
        # import ipdb; ipdb.set_trace()
        assert num_bboxes == single_pad * scalar, "The dn_part object query number is incorrect, plz check!"

        if len(gt_labels) > 0:
            # gt_labels: cls labels for a single image [num_gt,]
            t = torch.arange(0, len(gt_labels)).long().cuda()    # 注意: torch.range(a, b)会包含b
            t = t.unsqueeze(0).repeat(scalar, 1)
            tgt_idx = t.flatten()       # tgt_idx: [num_gt x dn_groups] from the padding_size = single_pad x dn_groups
            # output_idx相当于是正样本的索引即: pos_inds, 在一张图片里面的索引，因为一张图片padding后的query数量
            # 是padding size，这个相当于在padding size中的索引
            # tgt_idx相当于是正样本对应的gt的索引即: assigned_gt_inds，还不是对应的gt_labels
            output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
            output_idx = output_idx.flatten()
        else:
            output_idx = tgt_idx = torch.tensor([]).long().cuda()

        
        # 每个gt_labels的一个dn_groups中，前single_pad // 2 是正样本，后single_pad // 2是负样本
        # NOTE: 注意这里的single_pad和dn_component中的prepare_for_cdn的single_pad是不一样的
        # 前者是后者的两倍
      
        
        # 按照mmdetection的格式，仿照get_target_single，处理每张图片的target
        pos_inds = output_idx
        neg_inds = output_idx + single_pad // 2

        pos_assigned_gt_inds = tgt_idx

        pos_gt_bboxes = gt_bboxes[tgt_idx, :]

        # label targets
        labels = gt_labels.new_full((single_pad * scalar,), self.num_classes, dtype=torch.long)
        labels[output_idx] = gt_labels[tgt_idx].long()
        label_weights = gt_labels.new_ones(single_pad * scalar)

        # bbox_targets
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w, _ = img_meta['img_shape']

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        pos_gt_bboxes_normalized = pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)
      

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_scores_list=None,
                    img_metas=None,
                    dn_metas=None,
                    gt_bboxes_ignore_list=None,
                    decouple=False):
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
        if gt_scores_list is None:
            gt_scores_list = [
                gt_scores_list for _ in range(num_imgs)
            ]

        if dn_metas[0] is None:
            # only consider decouple the classification label and regression in the object query
            # learning, not the dn_part
            (labels_list, label_weights_list, bbox_targets_list,
            bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
                self._get_target_single, cls_scores_list, bbox_preds_list,
                gt_bboxes_list, gt_labels_list, gt_scores_list, img_metas, gt_bboxes_ignore_list, decouple=decouple)
        else:
            # import ipdb;ipdb.set_trace()
            (labels_list, label_weights_list, bbox_targets_list,
            bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
                self._get_target_single_dn, cls_scores_list, bbox_preds_list,
                gt_bboxes_list, gt_labels_list, gt_scores_list, img_metas, dn_metas, gt_bboxes_ignore_list)
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
                           gt_bboxes_ignore=None,
                           decouple=False):
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
        # import ipdb;ipdb.set_trace()

        # breakpoint()
        num_bboxes = bbox_pred.size(0)
        # assigner and sampler     
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
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

        # Note: if there is the gt_scores, means we will use the gt_scores 
        # as the weight of the classification loss and regression loss 
        if gt_scores is not None:
            # change the positive object regression weight to the soft scores
            # according to the score of the pseudo labels
            valid_inds = torch.nonzero(gt_scores[sampling_result.pos_assigned_gt_inds] > 0.5).squeeze().unique()
            valid_pos_inds = pos_inds[valid_inds]
            bbox_weights[valid_pos_inds] = 1.0
        else:
            bbox_weights[pos_inds] = 1.0

        img_h, img_w, _ = img_meta['img_shape']

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
        pos_gt_bboxes_normalized = sampling_result.pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets              # the regression target is the same no matter decouple or not, cause we can use the weight to adjust the loss
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

    # over-write because img_metas are needed as inputs for bbox_head.
    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
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
        # breakpoint()
        # prepare the dn_components
        # Note: need check whether there exist gt boxes in this batched data
        # if the whole batch has no gt box or gt label, we do't do the dn task
        num_valid_gts = [gt_bbox.size(0) for gt_bbox in gt_bboxes]
        if self.dn_number > 0 and max(num_valid_gts) > 0:
            # prepare the target
            # NOTE: need the labels and boxes as the list of each images
            targets = dict()
            targets['labels'] = gt_labels 
            # becareful when prepare the dn_components, the gt bboxes needs to 
            # be normalized with corresponding w, h, in the cx, cy, w, h format
            normalized_boxes = []
            for img_meta, gt_bbox in zip(img_metas, gt_bboxes):
                img_h, img_w, _ = img_meta['img_shape']
                factor = gt_bbox.new_tensor([img_w, img_h, img_w,
                                            img_h]).unsqueeze(0).repeat(
                                                gt_bbox.size(0), 1)
                gt_bbox_ = bbox_xyxy_to_cxcywh(gt_bbox)
                normalized_boxes.append(gt_bbox_ / factor)

            targets['boxes'] = normalized_boxes

            input_query_label, input_query_bbox, attn_mask, dn_meta = prepare_for_cdn(dn_args=(targets, self.dn_number, self.dn_label_noise_ratio, self.dn_box_noise_scale),
                                training=True, num_queries=self.num_query, num_classes=self.num_classes,
                                hidden_dim=self.embed_dims, label_enc=self.label_enc)
        else:
            input_query_bbox = input_query_label = attn_mask = dn_meta = None

        assert proposal_cfg is None, '"proposal_cfg" must be None'
        # change the forward method's parameter
        outs = self(x, img_metas, input_query_label, input_query_bbox, attn_mask, dn_meta)
        # breakpoint()
        # TODO: when calculate the loss, need consider the dn part loss seperately
      
        loss_inputs = outs + (gt_bboxes, gt_labels)
        losses = self.loss(*loss_inputs, img_metas=img_metas, dn_metas=dn_meta, gt_bboxes_ignore=gt_bboxes_ignore)
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
                   rescale=False):
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
                                                rescale)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_score,
                           bbox_pred,
                           img_shape,
                           scale_factor,
                           rescale=False):
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
        assert len(cls_score) == len(bbox_pred)
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)
        # exclude background
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % self.num_classes
            bbox_index = indexes // self.num_classes
            bbox_pred = bbox_pred[bbox_index]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            bbox_pred = bbox_pred[bbox_index]
            det_labels = det_labels[bbox_index]

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            det_bboxes /= det_bboxes.new_tensor(scale_factor)
        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(1)), -1)

        return det_bboxes, det_labels

    def simple_test_bboxes(self, feats, img_metas, rescale=False):
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
        # forward of this head requires img_metas
        outs = self.forward(feats, img_metas)
        results_list = self.get_bboxes(*outs, img_metas, rescale=rescale)
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
        if self.loss_cls.use_sigmoid:
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
