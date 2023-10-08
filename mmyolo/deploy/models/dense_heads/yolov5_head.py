# Copyright (c) OpenMMLab. All rights reserved.
import copy
from functools import partial
from typing import List, Optional, Tuple

import torch
from mmdeploy.codebase.mmdet import get_post_processing_params
from mmdeploy.codebase.mmdet.models.layers import multiclass_nms
from mmdeploy.core import FUNCTION_REWRITER
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmyolo.deploy.models.layers import efficient_nms
from mmyolo.models.dense_heads import YOLOv5Head


def yolov5_bbox_decoder(priors: Tensor, bbox_preds: Tensor,
                        stride: int) -> Tensor:
    """Decode YOLOv5 bounding boxes.

    Args:
        priors (Tensor): Prior boxes in center-offset form.
        bbox_preds (Tensor): Predicted bounding boxes.
        stride (int): Stride of the feature map.

    Returns:
        Tensor: Decoded bounding boxes.
    """
    bbox_preds = bbox_preds.sigmoid()

    x_center = (priors[..., 0] + priors[..., 2]) * 0.5
    y_center = (priors[..., 1] + priors[..., 3]) * 0.5
    w = priors[..., 2] - priors[..., 0]
    h = priors[..., 3] - priors[..., 1]

    x_center_pred = (bbox_preds[..., 0] - 0.5) * 2 * stride + x_center
    y_center_pred = (bbox_preds[..., 1] - 0.5) * 2 * stride + y_center
    w_pred = (bbox_preds[..., 2] * 2)**2 * w
    h_pred = (bbox_preds[..., 3] * 2)**2 * h

    decoded_bboxes = torch.stack(
        [x_center_pred, y_center_pred, w_pred, h_pred], dim=-1)

    return decoded_bboxes


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmyolo.models.dense_heads.yolov5_head.'
    'YOLOv5Head.predict_by_feat')
def yolov5_head__predict_by_feat(self,
                                 cls_scores: List[Tensor],
                                 bbox_preds: List[Tensor],
                                 objectnesses: Optional[List[Tensor]] = None,
                                 batch_img_metas: Optional[List[dict]] = None,
                                 cfg: Optional[ConfigDict] = None,
                                 rescale: bool = False,
                                 with_nms: bool = True) -> Tuple[InstanceData]:
    """Transform a batch of output features extracted by the head into
    bbox results.
    Args:
        cls_scores (list[Tensor]): Classification scores for all
            scale levels, each is a 4D-tensor, has shape
            (batch_size, num_priors * num_classes, H, W).
        bbox_preds (list[Tensor]): Box energies / deltas for all
            scale levels, each is a 4D-tensor, has shape
            (batch_size, num_priors * 4, H, W).
        objectnesses (list[Tensor], Optional): Score factor for
            all scale level, each is a 4D-tensor, has shape
            (batch_size, 1, H, W).
        batch_img_metas (list[dict], Optional): Batch image meta info.
            Defaults to None.
        cfg (ConfigDict, optional): Test / postprocessing
            configuration, if None, test_cfg would be used.
            Defaults to None.
        rescale (bool): If True, return boxes in original image space.
            Defaults to False.
        with_nms (bool): If True, do nms before return boxes.
            Defaults to True.
    Returns:
        tuple[Tensor, Tensor]: The first item is an (N, num_box, 5) tensor,
            where 5 represent (tl_x, tl_y, br_x, br_y, score), N is batch
            size and the score between 0 and 1. The shape of the second
            tensor in the tuple is (N, num_box), and each element
            represents the class label of the corresponding box.
    """
    ctx = FUNCTION_REWRITER.get_context()
    detector_type = type(self)
    deploy_cfg = ctx.cfg
    use_efficientnms = deploy_cfg.get('use_efficientnms', False)
    dtype = cls_scores[0].dtype
    device = cls_scores[0].device
    bbox_decoder = self.bbox_coder.decode
    nms_func = multiclass_nms
    if use_efficientnms:
        if detector_type is YOLOv5Head:
            nms_func = partial(efficient_nms, box_coding=0)
            bbox_decoder = yolov5_bbox_decoder
        else:
            nms_func = efficient_nms

    assert len(cls_scores) == len(bbox_preds)
    cfg = self.test_cfg if cfg is None else cfg
    cfg = copy.deepcopy(cfg)

    num_imgs = cls_scores[0].shape[0]
    featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

    mlvl_priors = self.prior_generator.grid_priors(
        featmap_sizes, dtype=dtype, device=device)

    flatten_priors = torch.cat(mlvl_priors)

    mlvl_strides = [
        flatten_priors.new_full(
            (featmap_size[0] * featmap_size[1] * self.num_base_priors, ),
            stride)
        for featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
    ]
    flatten_stride = torch.cat(mlvl_strides)

    # flatten cls_scores, bbox_preds and objectness
    flatten_cls_scores = [
        cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_classes)
        for cls_score in cls_scores
    ]
    cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()

    flatten_bbox_preds = [
        bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
        for bbox_pred in bbox_preds
    ]
    flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)

    if objectnesses is not None:
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]
        flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        cls_scores = cls_scores * (flatten_objectness.unsqueeze(-1))

    scores = cls_scores

    bboxes = bbox_decoder(flatten_priors[None], flatten_bbox_preds,
                          flatten_stride)

    if not with_nms:
        return bboxes, scores

    post_params = get_post_processing_params(deploy_cfg)
    max_output_boxes_per_class = post_params.max_output_boxes_per_class
    iou_threshold = cfg.nms.get('iou_threshold', post_params.iou_threshold)
    score_threshold = cfg.get('score_thr', post_params.score_threshold)
    pre_top_k = post_params.pre_top_k
    keep_top_k = cfg.get('max_per_img', post_params.keep_top_k)

    return nms_func(bboxes, scores, max_output_boxes_per_class, iou_threshold,
                    score_threshold, pre_top_k, keep_top_k)


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmyolo.models.dense_heads.yolov5_head.'
    'YOLOv5Head.predict',
    backend='rknn')
def yolov5_head__predict__rknn(self, x: Tuple[Tensor], *args,
                               **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
    """Perform forward propagation of the detection head and predict detection
    results on the features of the upstream network.

    Args:
        x (tuple[Tensor]): Multi-level features from the
            upstream network, each is a 4D-tensor.
    """
    outs = self(x)
    return outs


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmyolo.models.dense_heads.yolov5_head.'
    'YOLOv5HeadModule.forward',
    backend='rknn')
def yolov5_head_module__forward__rknn(
        self, x: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
    """Forward feature of a single scale level."""
    out = []
    for i, feat in enumerate(x):
        out.append(self.convs_pred[i](feat))
    return out


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmyolo.models.dense_heads.yolov5_head.'
    'YOLOv5Head.predict_by_feat',
    backend='ncnn')
def yolov5_head__predict_by_feat__ncnn(self,
                                 cls_scores: List[Tensor],
                                 bbox_preds: List[Tensor],
                                 objectnesses: Optional[List[Tensor]] = None,
                                 batch_img_metas: Optional[List[dict]] = None,
                                 cfg: Optional[ConfigDict] = None,
                                 rescale: bool = False,
                                 with_nms: bool = True) -> Tuple[InstanceData]:
    ctx = FUNCTION_REWRITER.get_context()
    from mmdeploy.codebase.mmdet.ops import ncnn_detection_output_forward
    from mmdeploy.utils import get_root_logger
    from mmdeploy.utils.config_utils import is_dynamic_shape
    dynamic_flag = is_dynamic_shape(ctx.cfg)
    if dynamic_flag:
        logger = get_root_logger()
        logger.warning('YOLOV5 does not support dynamic shape with ncnn.')
    img_height = int(batch_img_metas[0]['img_shape'][0])
    img_width = int(batch_img_metas[0]['img_shape'][1])

    assert len(cls_scores) == len(bbox_preds) == len(objectnesses)
    device = cls_scores[0].device
    dtype = cls_scores[0].dtype
    cfg = self.test_cfg if cfg is None else cfg
    batch_size = bbox_preds[0].shape[0]
    featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

    # If the shape does not change, use the previous mlvl_priors
    if featmap_sizes != self.featmap_sizes:
        self.mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=dtype,
            device=device)
        self.featmap_sizes = featmap_sizes
    flatten_priors = torch.cat(self.mlvl_priors)

    mlvl_strides = [
        flatten_priors.new_full(
            (featmap_size.numel() * self.num_base_priors, ), stride) for
        featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
    ]
    flatten_stride = torch.cat(mlvl_strides)

    flatten_cls_scores = [
        cls_score.permute(0, 2, 3, 1).reshape(batch_size, -1,
                                              self.num_classes)
        for cls_score in cls_scores
    ]
    flatten_bbox_preds = [
        bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        for bbox_pred in bbox_preds
    ]
    flatten_objectness = [
        objectness.permute(0, 2, 3, 1).reshape(batch_size, -1, 1)
        for objectness in objectnesses
    ]

    cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
    dummy_cls_scores = torch.zeros(
        batch_size, cls_scores.shape[-2], 1, device=cls_scores.device)
    batch_mlvl_scores = torch.cat([dummy_cls_scores, cls_scores], dim=2)
    score_factor = torch.cat(flatten_objectness, dim=1).sigmoid()
    scores = batch_mlvl_scores.permute(0, 2, 1).unsqueeze(3) * \
             score_factor.permute(0, 2, 1).unsqueeze(3)
    scores = scores.squeeze(3).permute(0, 2, 1)

    flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
    assert flatten_bbox_preds.shape[-1] == 4, f'yolov5 needs (B, N, 4) priors, got\
            (B, N, {flatten_bbox_preds.shape[-1]})'

    flatten_decoded_bboxes = self.bbox_coder.decode(
        flatten_priors[None], flatten_bbox_preds, flatten_stride)
    t_x1 = flatten_decoded_bboxes[:, :, 0:1] / img_width
    t_y1 = flatten_decoded_bboxes[:, :, 1:2] / img_height
    t_x2 = flatten_decoded_bboxes[:, :, 2:3] / img_width
    t_y2 = flatten_decoded_bboxes[:, :, 3:4] / img_height
    flatten_decoded_bboxes = torch.cat([t_x1, t_y1, t_x2, t_y2], dim=2)

    batch_mlvl_bboxes = flatten_decoded_bboxes.reshape(batch_size, 1, -1)
    batch_mlvl_scores = scores.reshape(batch_size, 1, -1)
    dummy_mlvl_priors = torch.ones_like(batch_mlvl_bboxes)
    dummy_mlvl_vars = torch.ones_like(batch_mlvl_bboxes)
    batch_mlvl_priors = torch.cat([dummy_mlvl_priors, dummy_mlvl_vars], dim=1)

    deploy_cfg = ctx.cfg
    post_params = get_post_processing_params(deploy_cfg)
    iou_threshold = cfg.nms.get('iou_threshold', post_params.iou_threshold)
    score_threshold = cfg.get('score_thr', post_params.score_threshold)
    pre_top_k = post_params.pre_top_k
    keep_top_k = cfg.get('max_per_img', post_params.keep_top_k)

    vars = torch.tensor([1, 1, 1, 1], dtype=torch.float32)
    output__ncnn = ncnn_detection_output_forward(
        batch_mlvl_bboxes, batch_mlvl_scores, batch_mlvl_priors,
        score_threshold, iou_threshold, pre_top_k, keep_top_k,
        self.num_classes + 1,
        vars.cpu().detach().numpy())
    return output__ncnn
