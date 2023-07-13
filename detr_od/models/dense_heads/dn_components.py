import torch
import torch.nn.functional as F
from mmdet.models.utils.transformer import inverse_sigmoid


def prepare_for_cdn(dn_args, training, num_queries, num_classes, hidden_dim, label_enc):
    """
        A major difference of DINO from DN-DETR is that the author process pattern embedding pattern embedding in its detector
        forward function and use learnable tgt embedding, so we change this function a little bit.
        :param dn_args: targets, dn_number, label_noise_ratio, box_noise_scale
        :param training: if it is training or inference
        :param num_queries: number of queires
        :param num_classes: number of classes
        :param hidden_dim: transformer hidden dim
        :param label_enc: encode labels in dn
        :return:
        """
    if training:
        targets, dn_number, label_noise_ratio, box_noise_scale = dn_args
        # positive and negative dn queries
        dn_number = dn_number * 2
        known = [(torch.ones_like(t)).cuda() for t in targets['labels']]
        batch_size = len(known)
        known_num = [sum(k) for k in known]
        if int(max(known_num)) == 0:
            dn_number = 1
        else:
            if dn_number >= 100:
                # 保证gt_box数量最多的gt_labels或者gt_boxes在进行dn_part的add noise之后
                # 数量就是config里面设置的dn_number x 2
                dn_number = dn_number // (int(max(known_num) * 2))
            elif dn_number < 1:
                dn_number = 1
        if dn_number == 0:
            dn_number = 1
        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t for t in targets['labels']])
        boxes = torch.cat([t for t in targets['boxes']])
        batch_idx = torch.cat([torch.full_like(t.long(), i) for i, t in enumerate(targets['labels'])])

        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)

        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)
        known_labels = labels.repeat(2 * dn_number, 1).view(-1)
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
        known_bboxs = boxes.repeat(2 * dn_number, 1)
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        if label_noise_ratio > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(-1)   # half of bbox prob
            new_label = torch.randint_like(chosen_indice, 0, num_classes)           # randomly put a new one here
            known_labels_expaned.scatter_(0, chosen_indice, new_label)
        single_pad = int(max(known_num))

        pad_size = int(single_pad * 2 * dn_number)  #
        positive_idx = torch.tensor(range(len(boxes))).long().cuda().unsqueeze(0).repeat(dn_number, 1)
        positive_idx += (torch.tensor(range(dn_number)) * len(boxes) * 2).long().cuda().unsqueeze(1)
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
        input_label_embed = label_enc(m)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        padding_label = torch.zeros(pad_size, hidden_dim).cuda()
        padding_bbox = torch.zeros(pad_size, 4).cuda()

        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

        map_known_indice = torch.tensor([]).to('cuda')
        if len(known_num):
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(2 * dn_number)]).long()
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

        tgt_size = pad_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(dn_number):
            if i == 0:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
            if i == dn_number - 1:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * i * 2] = True
            else:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * 2 * i] = True

        dn_meta = {
            'pad_size': pad_size,
            'num_dn_group': dn_number,
        }
    else:

        input_query_label = None
        input_query_bbox = None
        attn_mask = None
        dn_meta = None

    return input_query_label, input_query_bbox, attn_mask, dn_meta


def prepare_for_cdn_plus(dn_args, training, num_queries, num_classes, hidden_dim, label_enc):
    """ A extend version of the original prepare_for_cnd function which can deal with the situation of 
    there is no situation within a image.
    """
    if training:
        targets, dn_number, label_noise_ratio, box_noise_scale = dn_args

        # preprocess
        # check there is gt bbox for each image
        num_valid_gts = [t.size(0) for t in targets['boxes']]

        gt_labels = []
        gt_bboxes = []

        pad_mask = []       # show which image has no gt bbox, 1 for no gt, 0 for valid gt
        for i, num in enumerate(num_valid_gts):
            if num == 0:
                # if there is no gt bbox in this image, we randomly generate a 
                # gt bbox and gt label for this image
                # bbox is the normalized cx, cy, w, h format
                tmp_bboxes = torch.tensor([[0.5, 0.5, 0.5, 0.5]]).cuda()
                tmp_labels = torch.randint(0, 80, (1,)).long().cuda()

                gt_bboxes.append(tmp_bboxes)
                gt_labels.append(tmp_labels)

                pad_mask.append(1)
            else:
                gt_bboxes.append(targets['boxes'][i])
                gt_labels.append(targets['labels'][i])

                pad_mask.append(0)
        pad_mask = torch.tensor(pad_mask).cuda()

        # prepare dn query
        # positive and negative dn queries
        dn_number = dn_number * 2
        known = [(torch.ones_like(t)).cuda() for t in gt_labels]
        batch_size = len(known)
        known_num = [sum(k) for k in known]     # total gt bbox num

        assert int(max(known_num)) != 0, "It's impossible for the gt box num is 0 in a batched images!"
       
        if dn_number >= 100:
            dn_number = dn_number // (int(max(known_num) * 2))
        elif dn_number < 1:
            dn_number = 1
        if dn_number == 0:
            dn_number = 1

        # concate all the gt bounding box together
        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t for t in gt_labels])          # put all the gt_labels of all images together
        boxes = torch.cat([t for t in gt_bboxes])           # put all the gt_bboxes of all images together
        batch_idx = torch.cat([torch.full_like(t.long(), i) for i, t in enumerate(gt_labels)])

        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)

        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)
        known_labels = labels.repeat(2 * dn_number, 1).view(-1)
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
        known_bboxs = boxes.repeat(2 * dn_number, 1)
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        if label_noise_ratio > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(-1)   # half of bbox prob
            new_label = torch.randint_like(chosen_indice, 0, num_classes)           # randomly put a new one here
            known_labels_expaned.scatter_(0, chosen_indice, new_label)
        single_pad = int(max(known_num))

        pad_size = int(single_pad * 2 * dn_number)  #
        positive_idx = torch.tensor(range(len(boxes))).long().cuda().unsqueeze(0).repeat(dn_number, 1)
        positive_idx += (torch.tensor(range(dn_number)) * len(boxes) * 2).long().cuda().unsqueeze(1)
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
        input_label_embed = label_enc(m)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        padding_label = torch.zeros(pad_size, hidden_dim).cuda()
        padding_bbox = torch.zeros(pad_size, 4).cuda()

        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

        map_known_indice = torch.tensor([]).to('cuda')
        if len(known_num):
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(2 * dn_number)]).long()
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

        # import ipdb;ipdb.set_trace()

        tgt_size = pad_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(dn_number):
            if i == 0:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
            if i == dn_number - 1:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * i * 2] = True
            else:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * 2 * i] = True
        # import ipdb;ipdb.set_trace()
        # make the padding mask to calculate the loss
        # pad_mask: [bs,] -> [bs, pad_size]
        pad_mask = pad_mask[:, None].repeat(1, pad_size)
        dn_meta = {
            'pad_size': pad_size,
            'num_dn_group': dn_number,
            'pad_mask': pad_mask
        }
    else:

        input_query_label = None
        input_query_bbox = None
        attn_mask = None
        dn_meta = None

    return input_query_label, input_query_bbox, attn_mask, dn_meta

def prepare_for_cdn_ssod(dn_args, training, num_queries, num_classes, hidden_dim, label_enc, img_metas=None, pseudo_label=False, pseudo_aug_scale=0.06):
    """ A extend version of the original prepare_for_cnd function which can deal with the situation of 
    there is no situation within a image.
    """
    if training:
        targets, dn_number, label_noise_ratio, box_noise_scale = dn_args

        # preprocess
        # check there is gt bbox for each image
        num_valid_gts = [t.size(0) for t in targets['boxes']]

        gt_labels = []
        gt_bboxes = []

        pad_mask = []       # show which image has no gt bbox, 1 for no gt, 0 for valid gt
        for i, num in enumerate(num_valid_gts):
            if num == 0:
                # if there is no gt bbox in this image, we randomly generate a 
                # gt bbox and gt label for this image
                # bbox is the normalized cx, cy, w, h format
                tmp_bboxes = torch.tensor([[0.5, 0.5, 0.5, 0.5]]).cuda()
                tmp_labels = torch.randint(0, 80, (1,)).long().cuda()

                gt_bboxes.append(tmp_bboxes)
                gt_labels.append(tmp_labels)

                pad_mask.append(1)
            else:
                gt_bboxes.append(targets['boxes'][i])
                gt_labels.append(targets['labels'][i])

                pad_mask.append(0)
        pad_mask = torch.tensor(pad_mask).cuda()

        # prepare dn query
        # positive and negative dn queries
        dn_number = dn_number * 2
        known = [(torch.ones_like(t)).cuda() for t in gt_labels]
        batch_size = len(known)
        known_num = [sum(k) for k in known]     # total gt bbox num

        assert int(max(known_num)) != 0, "It's impossible for the gt box num is 0 in a batched images!"
       
        if dn_number >= 100:
            dn_number = dn_number // (int(max(known_num) * 2))
        elif dn_number < 1:
            dn_number = 1
        if dn_number == 0:
            dn_number = 1

        # concate all the gt bounding box together
        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t for t in gt_labels])          # put all the gt_labels of all images together
        boxes = torch.cat([t for t in gt_bboxes])           # put all the gt_bboxes of all images together

        # if pseudo label, we replace the original denoising method
        if pseudo_label:
            factors = []
            for i, (gt_bbox, img_meta) in enumerate(zip(gt_bboxes, img_metas)):
                # gt_bbox: [num_gt, 4]
                img_h, img_w, _ = img_meta['img_shape']
                factor = gt_bbox.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0).repeat(gt_bbox.size(0), 1)
                factors.append(factor)
            factors = torch.cat(factors)

        batch_idx = torch.cat([torch.full_like(t.long(), i) for i, t in enumerate(gt_labels)])

        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)

        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)
        known_labels = labels.repeat(2 * dn_number, 1).view(-1)
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
        known_bboxs = boxes.repeat(2 * dn_number, 1)
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        if pseudo_label:
            known_factors = factors.repeat(2 * dn_number, 1)

        if label_noise_ratio > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(-1)   # half of bbox prob
            new_label = torch.randint_like(chosen_indice, 0, num_classes)           # randomly put a new one here
            known_labels_expaned.scatter_(0, chosen_indice, new_label)
        single_pad = int(max(known_num))

        pad_size = int(single_pad * 2 * dn_number)  #
        positive_idx = torch.tensor(range(len(boxes))).long().cuda().unsqueeze(0).repeat(dn_number, 1)
        positive_idx += (torch.tensor(range(dn_number)) * len(boxes) * 2).long().cuda().unsqueeze(1)
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(boxes)
        if box_noise_scale > 0:
            if not pseudo_label:
                # for real gt bbox, add the bbox noise like original DINO
                known_bbox_ = torch.zeros_like(known_bboxs)         
                # from normalized [cx, cy, w, h] to normalized [x1, y1, x2, y2]
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
            else:
                # for pseudo gt bbox, add the bbox jiterring like soft teacher
                # from normalized [cx, cy, w, h] to unnormalized [x1, y1, x2, y2]
                known_bbox_ = torch.zeros_like(known_bboxs)         
                # from normalized [cx, cy, w, h] to unnormalized [x1, y1, x2, y2]
                known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
                known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2
                known_bbox_ = known_bbox_ * known_factors       # uunnormalized [x1, y1, x2, y2] format
                
                bbox_scale_ = known_bbox_[:, 2:] - known_bbox_[:, :2]
                bbox_scale_ = bbox_scale_.clamp(min=1)[:, None, :].expand(-1, 2, 2).reshape(-1, 4)    

                diff_scale_ = bbox_scale_ * pseudo_aug_scale

                aug_offset_ = torch.randn(known_bbox_.size(0), 4, device=known_bbox_.device) * diff_scale_
                # we don't distinguish the positive or negative query when denosing with pseudo bbox
                known_bbox_ = known_bbox_ + aug_offset_

                # convert from unnormalized [x1, y1, x2, y2] back to normalized [cx, cy, w, h]
                known_bbox_ = known_bbox_ / known_factors
                known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
                known_bbox_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
                known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]

        m = known_labels_expaned.long().to('cuda')
        input_label_embed = label_enc(m)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        padding_label = torch.zeros(pad_size, hidden_dim).cuda()
        padding_bbox = torch.zeros(pad_size, 4).cuda()

        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

        map_known_indice = torch.tensor([]).to('cuda')
        if len(known_num):
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(2 * dn_number)]).long()
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

        tgt_size = pad_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(dn_number):
            if i == 0:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
            if i == dn_number - 1:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * i * 2] = True
            else:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * 2 * i] = True

        # make the padding mask to calculate the loss
        # pad_mask: [bs,] -> [bs, pad_size]
        pad_mask = pad_mask[:, None].repeat(1, pad_size)
        dn_meta = {
            'pad_size': pad_size,
            'num_dn_group': dn_number,
            'pad_mask': pad_mask
        }
    else:

        input_query_label = None
        input_query_bbox = None
        attn_mask = None
        dn_meta = None

    return input_query_label, input_query_bbox, attn_mask, dn_meta

def dn_post_process(outputs_class, outputs_coord, dn_meta):
    """
        post process of dn after output from the transformer
        put the dn part in the dn_meta
    """
    if dn_meta and dn_meta['pad_size'] > 0:
        # dn_meta['pad_size'] is the size of the dn_part in a batched data
        # denosing output: output_known_class [num_dec_layer, bs, num_padding_dn, cls_out_channels]
        dn_output_class = outputs_class[:, :, :dn_meta['pad_size'], :]
        dn_output_coord = outputs_coord[:, :, :dn_meta['pad_size'], :]
        # original output: output_class [num_dec_layer, bs, num_query, cls_out_channels]
        outputs_class = outputs_class[:, :, dn_meta['pad_size']:, :]
        outputs_coord = outputs_coord[:, :, dn_meta['pad_size']:, :]

        return outputs_class, outputs_coord, dn_output_class, dn_output_coord
    else:
        return outputs_class, outputs_coord, None, None


def dn_post_process_plus(outputs_class, outputs_coord, dn_meta):
    """
        post process of dn after output from the transformer
        put the dn part in the dn_meta
    """
    assert dn_meta, 'dn_meta must be not None!'

    # dn_meta['pad_size'] is the size of the dn_part in a batched data
    # denosing output: output_known_class [num_dec_layer, bs, num_padding_dn, cls_out_channels]
    dn_output_class = outputs_class[:, :, :dn_meta['pad_size'], :]
    dn_output_coord = outputs_coord[:, :, :dn_meta['pad_size'], :]
    # original output: output_class [num_dec_layer, bs, num_query, cls_out_channels]
    outputs_class = outputs_class[:, :, dn_meta['pad_size']:, :]
    outputs_coord = outputs_coord[:, :, dn_meta['pad_size']:, :]

    return outputs_class, outputs_coord, dn_output_class, dn_output_coord
   
