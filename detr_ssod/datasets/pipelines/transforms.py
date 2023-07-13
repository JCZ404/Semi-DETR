"""
Modified from https://github.com/google-research/ssl_detection/blob/master/detection/utils/augmentation.py.
"""
import copy
import random
import math
import cv2
import mmcv
import numpy as np
import numbers
import torchvision
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
from mmcv.image.colorspace import bgr2rgb, rgb2bgr
from mmdet.core.mask import BitmapMasks, PolygonMasks
from mmdet.datasets import PIPELINES
from mmdet.datasets.pipelines import Compose as BaseCompose
from mmdet.datasets.pipelines import transforms

from .geo_utils import GeometricTransformationBase as GTrans

PARAMETER_MAX = 10




@PIPELINES.register_module()
class AutoAug:
    
    def __init__(self, policies, record=False):
        assert isinstance(policies, list) and len(policies) > 0, \
            'Policies must be a non-empty list.'
        for policy in policies:
            assert isinstance(policy, list) and len(policy) > 0, \
                'Each policy in policies must be a non-empty list.'
            for augment in policy:
                assert isinstance(augment, dict) and 'type' in augment, \
                    'Each specific augmentation must be a dict with key' \
                    ' "type".'
        # breakpoint()
        self.policies = copy.deepcopy(policies)
        self.transforms = [BaseCompose(policy) for policy in self.policies]
        self.record = record
        self.enable_record(record)

   
    def __call__(self, results):
        # breakpoint()
        # here the each transform in self.transforms list is Compose class instance
        transform = np.random.choice(self.transforms)
        # according the select transform to record the corresponding parameters
        return transform(results)

    def __repr__(self):
        return f'{self.__class__.__name__}(policies={self.policies})'

    def enable_record(self, mode):
        for compose in self.transforms:
            for transform in compose.transforms:
                if getattr(transform, 'enable_record', None):
                    transform.enable_record(mode)



@PIPELINES.register_module()
class RandCrop:
    """Random Crop in DETR-like detectors, which resize image first and then rand crop the images.
    Plz note that, we "merge the Random Resize and Random Crop into one transform", it's different 
    with the orginal implement, but it not affect the final results.
    """
    def __init__(self,
                 resize_scale,
                 resize_mode,
                 keep_ratio,
                 crop_size,
                 crop_type='absolute',
                 allow_negative_crop=False,
                 bbox_clip_border=True,
                 record=False):
        
        # resize part
        self.resize = transforms.Resize(img_scale=resize_scale, 
                                       multiscale_mode=resize_mode,
                                       keep_ratio=keep_ratio)
        

        # crop part
        if crop_type not in [
                'relative_range', 'relative', 'absolute', 'absolute_range'
        ]:
            raise ValueError(f'Invalid crop_type {crop_type}.')
        if crop_type in ['absolute', 'absolute_range']:
            assert crop_size[0] > 0 and crop_size[1] > 0
            assert isinstance(crop_size[0], int) and isinstance(
                crop_size[1], int)
        else:
            assert 0 < crop_size[0] <= 1 and 0 < crop_size[1] <= 1
        self.crop_size = crop_size
        self.crop_type = crop_type
        self.allow_negative_crop = allow_negative_crop
        self.bbox_clip_border = bbox_clip_border
        self.record = record                # record the random crop paramter to do the following gt bbox alignment
        
        # The key correspondence from bboxes to labels and masks.
        self.bbox2label = {
            'gt_bboxes': 'gt_labels',
            'gt_bboxes_ignore': 'gt_labels_ignore'
        }
        self.bbox2mask = {
            'gt_bboxes': 'gt_masks',
            'gt_bboxes_ignore': 'gt_masks_ignore'
        }

    def _crop_data(self, results, crop_size, allow_negative_crop):
        """Function to randomly crop images, bounding boxes, masks, semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
            crop_size (tuple): Expected absolute size after cropping, (h, w).
            allow_negative_crop (bool): Whether to allow a crop that does not
                contain any bbox area. Default to False.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        assert crop_size[0] > 0 and crop_size[1] > 0
        for key in results.get('img_fields', ['img']):
            img = results[key]
            margin_h = max(img.shape[0] - crop_size[0], 0)
            margin_w = max(img.shape[1] - crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

            # crop the image
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape
            results[key] = img

        results['img_shape'] = img_shape
     
        
        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            # e.g. gt_bboxes and gt_bboxes_ignore
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32)
            bboxes = results[key] - bbox_offset
            if self.bbox_clip_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (
                bboxes[:, 3] > bboxes[:, 1])
            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            if (key == 'gt_bboxes' and not valid_inds.any()
                    and not allow_negative_crop):
                return None
            results[key] = bboxes[valid_inds, :]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_key = self.bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_inds]

            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = self.bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][
                    valid_inds.nonzero()[0]].crop(
                        np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = results[key][crop_y1:crop_y2, crop_x1:crop_x2]

        return results, (offset_h, offset_w)

    def _get_crop_size(self, image_size):
        """Randomly generates the absolute crop size based on `crop_type` and
        `image_size`.

        Args:
            image_size (tuple): (h, w).

        Returns:
            crop_size (tuple): (crop_h, crop_w) in absolute pixels.
        """
        h, w = image_size
        if self.crop_type == 'absolute':
            return (min(self.crop_size[0], h), min(self.crop_size[1], w))
        elif self.crop_type == 'absolute_range':
            assert self.crop_size[0] <= self.crop_size[1]
            crop_h = np.random.randint(
                min(h, self.crop_size[0]),
                min(h, self.crop_size[1]) + 1)
            crop_w = np.random.randint(
                min(w, self.crop_size[0]),
                min(w, self.crop_size[1]) + 1)
            return crop_h, crop_w
        elif self.crop_type == 'relative':
            crop_h, crop_w = self.crop_size
            return int(h * crop_h + 0.5), int(w * crop_w + 0.5)
        elif self.crop_type == 'relative_range':
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            crop_h, crop_w = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * crop_h + 0.5), int(w * crop_w + 0.5)


    def __call__(self, results):
        """Call function to randomly crop images, bounding boxes, masks,
        semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        # breakpoint()
        # resize first and record the corresponding scale matrix
        results = self.resize(results)
        if self.record:
            # record the inital scale matrix
            scale_factor = results['scale_factor']
            # resize matrix together the possible random flip matrix as the scale matrix
            GTrans.apply(results, "scale", sx=scale_factor[0], sy=scale_factor[1])
            scale_matrix = results.pop("transform_matrix")
            
        # crop the rescaled images
        image_size = results['img'].shape[:2]
        crop_size = self._get_crop_size(image_size)
        results, (offset_h, offset_w) = self._crop_data(results, crop_size, self.allow_negative_crop)
       
        # record the crop information
        results['crop_info'] = {'scale_matrix':scale_matrix, 'scale_size':image_size, 'crop_offset':(offset_h, offset_w), 'crop_size':crop_size}
        return results


    def enable_record(self, mode: bool = True):
        self.record = mode
    


    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(crop_size={self.crop_size}, '
        repr_str += f'crop_type={self.crop_type}, '
        repr_str += f'allow_negative_crop={self.allow_negative_crop}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str



@PIPELINES.register_module()
class RandGaussianBlur(object):
    """Gaussian blur augmentation """

    def __init__(self, sigma=[.1, 2.], p=0.8):
        self.sigma = sigma
        self.p = p

    def __call__(self, results):
        if np.random.random() > self.p:
            return results
        
        # Note: the image in the results dict is in the numpy format
        # from numpy array to PIL
        pil_img = Image.fromarray(results['img'], mode='RGB')
        sigma = np.random.uniform(self.sigma[0], self.sigma[1])
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=sigma))

        # from PIL to numpy array
        results['img'] = np.array(pil_img)
        return results

    def enable_record(self, mode: bool = True):
        self.record = mode

@PIPELINES.register_module()
class RandGrayscale:
    def __init__(self, p=0.2):
        self.transform = torchvision.transforms.RandomGrayscale(p=p)

    def __call__(self, results):

        pil_img = Image.fromarray(results['img'], mode='RGB')
        pil_img = self.transform(pil_img)
        results['img'] = np.array(pil_img)

        return results

    def enable_record(self, mode: bool = True):
        self.record = mode
        
@PIPELINES.register_module()
class RandColorJitter:
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8):
        self.transform = torchvision.transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        self.p = p
    
    def __call__(self, results):
        if np.random.random() >= self.p:
            return results
        # from numpy array to PIL
        pil_img = Image.fromarray(results['img'], mode='RGB')
        pil_img = self.transform(pil_img)
        results['img'] = np.array(pil_img)

        return results

    def enable_record(self, mode: bool = True):
        self.record = mode


@PIPELINES.register_module()
class RandomErase(object):
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False):
        assert isinstance(value, (numbers.Number, str, tuple, list))
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("range of scale should be between 0 and 1")
        if p < 0 or p > 1:
            raise ValueError("range of random erasing probability should be between 0 and 1")

        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace

    @staticmethod
    def get_params(img, scale, ratio, value=0):
        """Get parameters for ``erase`` for a random erasing.

        Args:
            img (np.array): ndarray image of size (H, W, C) to be erased.
            scale: range of proportion of erased area against input image.
            ratio: range of aspect ratio of erased area.

        Returns:
            tuple: params (i, j, h, w, v) to be passed to ``erase`` for random erasing.
        """
        img_h, img_w, img_c = img.shape
        area = img_h * img_w

        for _ in range(10):
            erase_area = random.uniform(scale[0], scale[1]) * area
            aspect_ratio = random.uniform(ratio[0], ratio[1])

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))

            if h < img_h and w < img_w:
                i = random.randint(0, img_h - h)
                j = random.randint(0, img_w - w)
                if isinstance(value, numbers.Number):
                    v = value
                elif value == 'random':
                    v = np.random.randint(0, 256, size=(h, w, img_c))
                else:
                    raise NotImplementedError('Not implement')
                return i, j, h, w, v

        # Return original image
        return 0, 0, img_h, img_w, img

    def __call__(self, results):
        # import ipdb;ipdb.set_trace()
        if random.uniform(0, 1) >= self.p:
            return results
        img = results['img']
        y, x, h, w, v = self.get_params(img, scale=self.scale, ratio=self.ratio, value=self.value)
        img[y:y + h, x:x + w] = v
        results['img'] = img
        return results

    def enable_record(self, mode: bool = True):
        self.record = mode

@PIPELINES.register_module()
class RandErasing(object):
    """Apply Random Erase multiple times.
    """
    def __init__(self, transforms):
        # import ipdb;ipdb.set_trace()
        self.transform = BaseCompose(transforms)

    def __call__(self, results):
        # import ipdb;ipdb.set_trace()

        results =  self.transform(results)
        return results

    def enable_record(self, mode: bool = True):
        self.record = mode

@PIPELINES.register_module()
class AugmentationUT(object):
    """Unbiased-Teacher style strong augmentations on the unlabeled data.
    """
    def __init__(self, transforms):
        # import ipdb;ipdb.set_trace()
        self.transforms = BaseCompose(transforms)
       

    def __call__(self, results):
        # import ipdb;ipdb.set_trace()
        
        results = self.transforms(results)
        return results

    def enable_record(self, mode: bool = True):
        self.record = mode