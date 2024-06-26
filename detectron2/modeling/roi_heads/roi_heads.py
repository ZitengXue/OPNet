# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
import os
from typing import Dict
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd.function import Function
from detectron2.layers import ShapeSpec
from typing import Dict, List, Optional, Tuple, Union
from detectron2.structures import Boxes, Instances, pairwise_iou, ImageList, BitMasks
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from detectron2.layers import cat

from ..backbone.resnet import BottleneckBlock, make_stage
from ..box_regression import Box2BoxTransform
from ..matcher import Matcher
from ..poolers import ROIPooler
from ..proposal_generator.proposal_utils import add_ground_truth_to_proposals
from ..sampling import subsample_labels
from .box_head import build_box_head
from .recls_head import build_recls_head, mask_recls_filter_loss, mask_recls_margin_loss, mask_recls_adaptive_loss
from .fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs,fast_rcnn_inference
from .keypoint_head import build_keypoint_head, keypoint_rcnn_inference, keypoint_rcnn_loss
from .mask_head import build_mask_head, mask_rcnn_inference, amodal_mask_rcnn_inference,\
    mask_rcnn_loss, amodal_mask_rcnn_loss, mask_fm_loss, classes_choose,boundary_loss
from .mask_visible_head import build_visible_mask_head
from .mask_invisible_head import build_invisible_mask_head
from .mask_amodal_head import build_amodal_mask_head
from .recon_net import build_reconstruction_head, mask_recon_loss, mask_recon_inference


ROI_HEADS_REGISTRY = Registry("ROI_HEADS")
ROI_HEADS_REGISTRY.__doc__ = """
Registry for ROI heads in a generalized R-CNN model.
ROIHeads take feature maps and region proposals, and
perform per-region computation.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`ROIHeads`.
"""

logger = logging.getLogger(__name__)


def build_roi_heads(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)


def select_foreground_proposals(proposals, bg_label):
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.

    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.

    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        fg_idxs = torch.nonzero(fg_selection_mask,as_tuple=False).squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks


def select_proposals_with_visible_keypoints(proposals):
    """
    Args:
        proposals (list[Instances]): a list of N Instances, where N is the
            number of images.

    Returns:
        proposals: only contains proposals with at least one visible keypoint.

    Note that this is still slightly different from Detectron.
    In Detectron, proposals for training keypoint head are re-sampled from
    all the proposals with IOU>threshold & >=1 visible keypoint.

    Here, the proposals are first sampled from all proposals with
    IOU>threshold, then proposals with no visible keypoint are filtered out.
    This strategy seems to make no difference on Detectron and is easier to implement.
    """
    ret = []
    all_num_fg = []
    for proposals_per_image in proposals:
        # If empty/unannotated image (hard negatives), skip filtering for train
        if len(proposals_per_image) == 0:
            ret.append(proposals_per_image)
            continue
        gt_keypoints = proposals_per_image.gt_keypoints.tensor
        # #fg x K x 3
        vis_mask = gt_keypoints[:, :, 2] >= 1
        xs, ys = gt_keypoints[:, :, 0], gt_keypoints[:, :, 1]
        proposal_boxes = proposals_per_image.proposal_boxes.tensor.unsqueeze(dim=1)  # #fg x 1 x 4
        kp_in_box = (
            (xs >= proposal_boxes[:, :, 0])
            & (xs <= proposal_boxes[:, :, 2])
            & (ys >= proposal_boxes[:, :, 1])
            & (ys <= proposal_boxes[:, :, 3])
        )
        selection = (kp_in_box & vis_mask).any(dim=1)
        selection_idxs = torch.nonzero(selection).squeeze(1)
        all_num_fg.append(selection_idxs.numel())
        ret.append(proposals_per_image[selection_idxs])

    storage = get_event_storage()
    storage.put_scalar("keypoint_head/num_fg_samples", np.mean(all_num_fg))
    return ret


class ROIHeads(torch.nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.

    It contains logic of cropping the regions, extract per-region features,
    and make per-region predictions.

    It can have many variants, implemented as subclasses of this class.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super(ROIHeads, self).__init__()

        # fmt: off
        self.batch_size_per_image     = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE     #512
        self.positive_sample_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION      #0.25
        self.test_score_thresh        = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST      #0.05
        self.test_nms_thresh          = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST     #0.5
        self.test_detections_per_img  = cfg.TEST.DETECTIONS_PER_IMAGE         #100
        self.in_features              = cfg.MODEL.ROI_HEADS.IN_FEATURES       #p2 p3 p4 p5  
        self.num_classes              = cfg.MODEL.ROI_HEADS.NUM_CLASSES         #d2sa 60
        self.proposal_append_gt       = cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT    #bool
        self.feature_strides          = {k: v.stride for k, v in input_shape.items()}    #p2:4  p3:8  p4:16  p5:32  p6:64
        self.feature_channels         = {k: v.channels for k, v in input_shape.items()}    #256 256 256 256 256
        self.cls_agnostic_bbox_reg    = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG        #bool
        self.smooth_l1_beta           = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA         #0
        # fmt: on

        # Matcher to assign box proposals to gt boxes
        self.proposal_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,    #0.5
            cfg.MODEL.ROI_HEADS.IOU_LABELS,      #][0,1]
            allow_low_quality_matches=False,
        )

        # Box2BoxTransform for bounding box regression
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)

    def _sample_proposals(self, matched_idxs, matched_labels, gt_classes):
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]       #
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes
        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_sample_fraction, self.num_classes
        )
        #positive   negtive  

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        #索引合并
        return sampled_idxs, gt_classes[sampled_idxs]
        #返回索引和索引对应的gt

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals, targets):
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_sample_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)
        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )
            
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())  #neg样本
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    def forward(self, images, features, proposals, targets=None):
        """
        Args:
            images (ImageList):
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (list[Instances]): length `N` list of `Instances`s. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:

                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].
                - gt_masks: PolygonMasks or BitMasks, the ground-truth masks of each instance.
                - gt_keypoints: NxKx3, the groud-truth keypoints for each instance.

        Returns:
            results (list[Instances]): length `N` list of `Instances`s containing the
            detected instances. Returned during inference only; may be [] during training.

            losses (dict[str->Tensor]):
            mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        raise NotImplementedError()


@ROI_HEADS_REGISTRY.register()
class Res5ROIHeads(ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where
    the box and mask head share the cropping and
    the per-region feature computation by a Res5 block.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        assert len(self.in_features) == 1

        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / self.feature_strides[self.in_features[0]], )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        self.mask_on      = cfg.MODEL.MASK_ON
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.res5, out_channels = self._build_res5_block(cfg)
        self.box_predictor = FastRCNNOutputLayers(
            out_channels, self.num_classes, self.cls_agnostic_bbox_reg
        )

        if self.mask_on:
            self.mask_head = build_mask_head(
                cfg,
                ShapeSpec(channels=out_channels, width=pooler_resolution, height=pooler_resolution),
            )

    def _build_res5_block(self, cfg):
        # fmt: off
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels  = num_groups * width_per_group * stage_channel_factor
        out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm                 = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on

        blocks = make_stage(
            BottleneckBlock,
            3,
            first_stride=2,
            in_channels=out_channels // 2,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            norm=norm,
            stride_in_1x1=stride_in_1x1,
        )
        return nn.Sequential(*blocks), out_channels

    def _shared_roi_transform(self, features, boxes):
        x = self.pooler(features, boxes)
        return self.res5(x)

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        del images

        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        pred_class_logits, pred_proposal_deltas = self.box_predictor(feature_pooled)
        del feature_pooled

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )
        if self.training:
            del features
            losses = outputs.losses()
            if self.mask_on:
                proposals, fg_selection_masks = select_foreground_proposals(
                    proposals, self.num_classes
                )
                # Since the ROI feature transform is shared between boxes and masks,
                # we don't need to recompute features. The mask loss is only defined
                # on foreground proposals, so we need to select out the foreground
                # features.
                mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
                del box_features

                mask_logits = self.mask_head(mask_features)
                losses["loss_mask"] = mask_rcnn_loss(mask_logits, proposals)
            return [], losses
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(self, features, instances):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (Instances):
                the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        if self.mask_on:
            features = [features[f] for f in self.in_features]
            x = self._shared_roi_transform(features, [x.pred_boxes for x in instances])
            mask_logits = self.mask_head(x)

            mask_rcnn_inference(mask_logits, instances)
        return instances


@ROI_HEADS_REGISTRY.register()
class StandardROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches (boxes and masks) directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(self, cfg, input_shape):
        super(StandardROIHeads, self).__init__(cfg, input_shape)
        self._cfg = cfg
        self._init_box_head(cfg)
        self._init_mask_head(cfg)
        self._init_keypoint_head(cfg)
        self.iter = 0

    def _init_box_head(self, cfg):
        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        self.box_predictor = FastRCNNOutputLayers(
            self.box_head.output_size, self.num_classes, self.cls_agnostic_bbox_reg
        )

    def _init_mask_head(self, cfg):
        # fmt: off
        self.mask_on           = cfg.MODEL.MASK_ON
        if not self.mask_on:
            return
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        self.inference_embedding = False
        # fmt: on

        in_channels = [self.feature_channels[f] for f in self.in_features][0]
        self.mask_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.mask_head = build_mask_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def _init_keypoint_head(self, cfg):
        # fmt: off
        self.keypoint_on                         = cfg.MODEL.KEYPOINT_ON
        if not self.keypoint_on:
            return
        pooler_resolution                        = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        pooler_scales                            = tuple(1.0 / self.feature_strides[k] for k in self.in_features)  # noqa
        sampling_ratio                           = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
        pooler_type                              = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE
        self.normalize_loss_by_visible_keypoints = cfg.MODEL.ROI_KEYPOINT_HEAD.NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS  # noqa
        self.keypoint_loss_weight                = cfg.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT
        # fmt: on

        in_channels = [self.feature_channels[f] for f in self.in_features][0]

        self.keypoint_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.keypoint_head = build_keypoint_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        self.iter += 1
        del images
        # if self._cfg.MODEL.ROI_HEADS.MASKIOU_AS_SCORES:
        #     self.batch_size_per_image = 1000
        #     self.proposal_append_gt = False
        if self.training or self._cfg.MODEL.ROI_HEADS.MASKIOU_AS_SCORES:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets
        features_list = [features[f] for f in self.in_features]
        if self.training:
            losses = self._forward_box(features_list, proposals)
            # During training the proposals used by the box head are
            # used by the mask, keypoint (and densepose) heads.
            losses.update(self._forward_mask(features_list, proposals))
            losses.update(self._forward_keypoint(features_list, proposals))
            return proposals, losses

        else:

            pred_instances = self._forward_box(features_list, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(self, features, instances):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (Instances):
                the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")
        features = [features[f] for f in self.in_features]

        instances = self._forward_mask(features, instances)
        instances = self._forward_keypoint(features, instances)
        return instances

    def _forward_box(self, features, proposals):
        """
        Forward logic of the box prediction branch.

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features)
        del box_features
        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )
        if self.training:
            return outputs.losses()
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img,
            )
            return pred_instances

    def _forward_mask(self, features, instances):
        """
        Forward logic of the mask prediction branch.

        Args:
            features (list[Tensor]): #level input features for mask prediction
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        if self.training:
            # The loss is only defined on positive proposals.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_features = self.mask_pooler(features, proposal_boxes)  # num_boxes*256*14*14

            mask_logits = self.mask_head(mask_features)                 # num_boxes*1*28*28
            return {"loss_mask": mask_rcnn_loss(mask_logits, proposals)}
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            mask_features = self.mask_pooler(features, pred_boxes)

            mask_logits = self.mask_head(mask_features)
            mask_rcnn_inference(mask_logits, instances)

            return instances

    def _forward_keypoint(self, features, instances):
        """
        Forward logic of the keypoint prediction branch.

        Args:
            features (list[Tensor]): #level input features for keypoint prediction
            instances (list[Instances]): the per-image instances to train/predict keypoints.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_keypoints" and return it.
        """
        if not self.keypoint_on:
            return {} if self.training else instances

        num_images = len(instances)

        if self.training:
            # The loss is defined on positive proposals with at >=1 visible keypoints.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposals = select_proposals_with_visible_keypoints(proposals)
            proposal_boxes = [x.proposal_boxes for x in proposals]

            keypoint_features = self.keypoint_pooler(features, proposal_boxes)
            keypoint_logits = self.keypoint_head(keypoint_features)

            normalizer = (
                num_images
                * self.batch_size_per_image
                * self.positive_sample_fraction
                * keypoint_logits.shape[1]
            )
            loss = keypoint_rcnn_loss(
                keypoint_logits,
                proposals,
                normalizer=None if self.normalize_loss_by_visible_keypoints else normalizer,
            )
            return {"loss_keypoint": loss * self.keypoint_loss_weight}
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            keypoint_features = self.keypoint_pooler(features, pred_boxes)
            keypoint_logits = self.keypoint_head(keypoint_features)
            keypoint_rcnn_inference(keypoint_logits, instances)
            return instances
class _ScaleGradient(Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None

@ROI_HEADS_REGISTRY.register()
class Parallel_Amodal_Visible_ROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches (boxes and masks) directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(self, cfg, input_shape):
        super(Parallel_Amodal_Visible_ROIHeads, self).__init__(cfg, input_shape)
        self._cfg = cfg
        self._init_box_head(cfg)
        self._init_mask_head(cfg,input_shape)
        self.iter = 0
    def _init_box_head2(self, cfg):
        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        self.box_predictor = FastRCNNOutputLayers(
            self.box_head.output_size, self.num_classes, self.cls_agnostic_bbox_reg
        )
    def _init_box_head(self, cfg):
        # fmt: off
        pooler_resolution        = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales            = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio           = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type              = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        cascade_bbox_reg_weights = cfg.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS
        cascade_ious             = cfg.MODEL.ROI_BOX_CASCADE_HEAD.IOUS
        self.num_cascade_stages  = len(cascade_ious)
        assert len(cascade_bbox_reg_weights) == self.num_cascade_stages
        assert cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,  \
            "CascadeROIHeads only support class-agnostic regression now!"
        assert cascade_ious[0] == cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS[0]
        # fmt: on

        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        pooled_shape = ShapeSpec(
            channels=in_channels, width=pooler_resolution, height=pooler_resolution
        )

        self.box_head = nn.ModuleList()
        self.box_predictor = nn.ModuleList()
        self.box2box_transform = []
        self.proposal_matchers = []
        for k in range(self.num_cascade_stages):
            box_head = build_box_head(cfg, pooled_shape)
            self.box_head.append(box_head)
            self.box_predictor.append(
                FastRCNNOutputLayers(
                    box_head.output_size, self.num_classes, cls_agnostic_bbox_reg=True
                )
            )
            self.box2box_transform.append(Box2BoxTransform(weights=cascade_bbox_reg_weights[k]))

            if k == 0:
                # The first matching is done by the matcher of ROIHeads (self.proposal_matcher).
                self.proposal_matchers.append(None)
            else:
                self.proposal_matchers.append(
                    Matcher([cascade_ious[k]], [0, 1], allow_low_quality_matches=False)
                )

    def _init_mask_head(self, cfg,input_shape):
        # fmt: off
        self.mask_on           = cfg.MODEL.MASK_ON
        self.inference_embedding = False
        if not self.mask_on:
            return
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on
        #加
        in_features=cfg.MODEL.ROI_HEADS.IN_FEATURES
        boundary_resolution     = 28
        boundary_in_features    = ["p2"]
        boundary_scales         = tuple(1.0 / input_shape[k].stride for k in boundary_in_features)


        in_channels = [self.feature_channels[f] for f in self.in_features][0]
        self.mask_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        #加
        self.boundary_pooler = ROIPooler(
            output_size=boundary_resolution,
            scales=boundary_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type
        )
        self.mask_head = build_mask_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

        if cfg.MODEL.ROI_MASK_HEAD.RECON_NET.NAME != "":
            self.recon_mask_ths = cfg.MODEL.ROI_MASK_HEAD.RECON_NET.MASK_THS
            self.recon_alpha = cfg.MODEL.ROI_MASK_HEAD.RECON_NET.ALPHA
            self.recon_net = build_reconstruction_head(cfg, ShapeSpec(channels=in_channels, width=pooler_resolution,
                                                                      height=pooler_resolution))
            self.SPRef = cfg.MODEL.ROI_MASK_HEAD.RECON_NET.MEMORY_REFINE
            self.SPk = cfg.MODEL.ROI_MASK_HEAD.RECON_NET.MEMORY_REFINE_K
            if self.SPRef:
                self.mask_head.recon_net = self.recon_net
        else:
            self.recon_net = None
            self.recon_alpha = None

        if cfg.MODEL.ROI_MASK_HEAD.RECLS_NET.NAME != "":
            self.recls_mode = cfg.MODEL.ROI_MASK_HEAD.RECLS_NET.MODE
            self.recls_box_ths = cfg.MODEL.ROI_MASK_HEAD.RECLS_NET.BOX_THS
            self.recls_mask_ths = cfg.MODEL.ROI_MASK_HEAD.RECLS_NET.MASK_THS
            self.recls = build_recls_head(cfg, ShapeSpec(channels=in_channels, height=pooler_resolution,
                                                         width=pooler_resolution))
            self.gt_weight = cfg.MODEL.ROI_MASK_HEAD.RECLS_NET.GT_WEIGHT
        else:
            self.recls = None
        self.use_boundary = cfg.MODEL.ROI_HEADS.USE_BOUNDARY
    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        self.targets = targets
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)

        del targets
        features_list = [features[f] for f in self.in_features]
        boundary_features = [features["p2"]]
        #print("boundary_type:",boundary_features[0].type)
        if self.training:
            # losses = self._forward_box2(features_list, proposals)
            losses = self._forward_box(features_list, proposals,self.targets)

            # During training the proposals used by the box head are
            # used by the mask, keypoint (and densepose) heads.

            losses.update(self._forward_mask(features_list,boundary_features,proposals))

            return proposals, losses
        else:
            if (self._cfg.MODEL.ROI_MASK_HEAD.RECON_NET.NAME != "" or self._cfg.MODEL.ROI_MASK_HEAD.RECLS_NET.NAME != "")\
                    and not self.inference_embedding:
                pred_instances = self._forward_box(features_list, proposals)
                pred_instances = self.forward_with_given_boxes(features,boundary_features, pred_instances)
                #pred_instances = self._forward_box_and_mask_inference(features,boundary_features, pred_instances)
            else:
                pred_instances = self._forward_box(features_list, proposals)
                # During inference cascaded prediction is used: the mask and keypoints heads are only
                # applied to the top scoring box detections.
                #pred_instances = self.forward_with_given_boxes(features, boundary_features,pred_instances)
                pred_instances = self.forward_with_given_boxes(features,boundary_features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(self, features,boundary_features, instances):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (Instances):
                the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        # if len(instances[0]) == 0 or len(self.targets[0]) == 0:
        #     return instances
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")
        features = [features[f] for f in self.in_features]
        if self.inference_embedding:
            instances = self.add_ground_truth_for_inference_embedding(instances)

        instances = self._forward_mask(features,boundary_features, instances)
        return instances
    def _forward_box2(self, features, proposals):
        """
        Forward logic of the box prediction branch.

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        self.pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features)
        del box_features
        outputs = FastRCNNOutputs(
            self.box2box_transform,
            self.pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )
        if self.training:
            return outputs.losses()
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img,
                targets=self.targets if self.inference_embedding else None,
            )

        return pred_instances
    def _forward_box(self, features, proposals, targets=None):
            head_outputs = []
            image_sizes = [x.image_size for x in proposals]
            for k in range(self.num_cascade_stages):
                if k > 0:
                    # The output boxes of the previous stage are the input proposals of the next stage
                    proposals = self._create_proposals_from_boxes(
                        head_outputs[-1].predict_boxes(), image_sizes
                    )
                    if self.training:
                        proposals = self._match_and_label_boxes(proposals, k, targets)
                head_outputs.append(self._run_stage(features, proposals, k))

            if self.training:
                losses = {}
                storage = get_event_storage()
                for stage, output in enumerate(head_outputs):
                    with storage.name_scope("stage{}".format(stage)):
                        stage_losses = output.losses()
                    losses.update({k + "_stage{}".format(stage): v for k, v in stage_losses.items()})
                return losses
            else:
                # Each is a list[Tensor] of length #image. Each tensor is Ri x (K+1)
                scores_per_stage = [h.predict_probs() for h in head_outputs]

                # Average the scores across heads
                scores = [
                    sum(list(scores_per_image)) * (1.0 / self.num_cascade_stages)
                    for scores_per_image in zip(*scores_per_stage)
                ]
                # Use the boxes of the last head
                boxes = head_outputs[-1].predict_boxes()
                pred_instances, _ = fast_rcnn_inference(
                    boxes,
                    scores,
                    image_sizes,
                    self.test_score_thresh,
                    self.test_nms_thresh,
                    self.test_detections_per_img,
                )
                return pred_instances

    def _forward_mask(self, features,boundary_features, instances):
        """
        Forward logic of the mask prediction branch.

        Args:
            features (list[Tensor]): #level input features for mask prediction
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        storage = get_event_storage()
        self.iter += 1
        if not self.mask_on:
            return {} if self.training else instances
        
        if self.training:
            # The loss is only defined on positive proposals.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_features = self.mask_pooler(features, proposal_boxes)  # num_boxes*256*14*14

            boundary_features = self.boundary_pooler(boundary_features, proposal_boxes)
            losses = {}

            if self.use_boundary:
                mask_logits, boundary_logits ,feature_matching= self.mask_head(mask_features, boundary_features,proposals,train=True)  # num_boxes*1*28*28
            else:
                mask_logits = self.mask_head(mask_features,boundary_features,proposals,train=True)
            # weights = self.attention_weights(mask_logits[0], proposals)
            losses.update({"loss_amask": 3*amodal_mask_rcnn_loss(
                mask_logits[0][0], proposals, mode="amodal", version="n")})
            losses.update({"loss_amask_refine":amodal_mask_rcnn_loss(
                mask_logits[0][2], proposals, mode="amodal", version="n")})
            losses.update({"loss_amask_gt": amodal_mask_rcnn_loss(
                mask_logits[0][3], proposals, mode="amodal", version="n")})
            if self.use_boundary:
                losses.update({"loss_boundary1":boundary_loss(mask_logits[0][0],boundary_logits[0],proposals)})
                losses.update({"loss_boundary2":boundary_loss(mask_logits[0][1],boundary_logits[1],proposals)})
                # losses.update({"loss_boundary3":boundary_loss(mask_logits[0][2],boundary_logits[2],proposals)})
            losses.update({"loss_vmask": amodal_mask_rcnn_loss(
                mask_logits[0][1], proposals, mode="visible", version="n")})
            losses.update({"loss_fm": mask_fm_loss(feature_matching,(0.01, 0.05))})

            return losses

        else:
            pred_boxes = [x.pred_boxes for x in instances]
            mask_features = self.mask_pooler(features, pred_boxes)
            boundary_features = self.boundary_pooler(boundary_features, pred_boxes)
            mask_logits = self.mask_head(mask_features, boundary_features,instances,train=False)

            amodal_mask_rcnn_inference(mask_logits, instances)
            
            if self._cfg.MODEL.ROI_MASK_HEAD.RECON_NET.NAME == "General_Recon_Net" and self.inference_embedding \
                    and len(self.targets[0]) > 0:
                mask_recon_inference(instances, self.targets, self.recon_net, iou_ths=self.recon_mask_ths)

            return instances

    def _forward_box_and_mask_inference(self, features,boundary_features, pred_instances):
        """
        Forward logic of the box and mask prediction branch together in inference with reconstruction net
                "gt_classes", "gt_boxes".

        Returns:
            In training, do not use this.
            In inference, a list of `Instances`, the predicted instances.
        """

        features_list = [features[f] for f in self.in_features]
        pred_instances = self._forward_mask(features_list,boundary_features, pred_instances)
        return pred_instances

    def add_ground_truth_for_inference_embedding(self, instances):
        for instance in instances:
            instance.proposal_boxes = instance.pred_boxes
        self.proposal_append_gt = False
        instances = self.label_and_sample_proposals(instances, self.targets)

        for instance in instances:
            instance.pred_boxes = instance.proposal_boxes
            instance.gt_classes_inference = instance.gt_classes
            instance.gt_masks_inference = instance.gt_masks

            instance.remove("gt_classes")
            instance.remove("gt_boxes")
            instance.remove("gt_masks")
            instance.remove("proposal_boxes")

        return instances

    def attention_weights(self, mask_logits, instances):
        amodal_mask_logits = mask_logits[0]
        visible_mask_logits = mask_logits[1]
        # refine_mask_logits = mask_logits[2]
        mask_side_len = amodal_mask_logits.size(2)
        index = 0

        amodal_weights = []
        visible_weights = []
        # refine_mask_logits = []
        for instances_per_image in instances:
            gt_amodal_per_image = instances_per_image.gt_masks.crop_and_resize(
                instances_per_image.proposal_boxes.tensor, mask_side_len
            ).unsqueeze(1).to(device=amodal_mask_logits.device)

            gt_visible_per_image = instances_per_image.gt_visible_masks.crop_and_resize(
                instances_per_image.proposal_boxes.tensor, mask_side_len
            ).unsqueeze(1).to(device=visible_mask_logits.device)

            amodal_pred_per_image = amodal_mask_logits[index: index + len(instances_per_image)] > 0
            visible_pred_per_image = visible_mask_logits[index: index + len(instances_per_image)] > 0

            iou_amodal_mask = torch.sum((gt_amodal_per_image * amodal_pred_per_image) > 0, dim=(1, 2, 3)).float() / \
                              torch.sum((gt_amodal_per_image + amodal_pred_per_image) > 0, dim=(1, 2, 3)).float()
            iou_visible_mask = torch.sum((gt_visible_per_image * visible_pred_per_image) > 0, dim=(1, 2, 3)).float() / \
                               torch.sum((gt_visible_per_image + visible_pred_per_image) > 0, dim=(1, 2, 3)).float()

            amodal_weights.append(iou_amodal_mask.unsqueeze(1).unsqueeze(2))
            visible_weights.append(iou_visible_mask.unsqueeze(1).unsqueeze(2))
        amodal_weights = cat(amodal_weights, dim=0)
        visible_weights = cat(visible_weights, dim=0)

        indices = torch.nonzero((~torch.isfinite(amodal_weights)),as_tuple=False)
        amodal_weights[indices[:, 0]] = 0
        indices = torch.nonzero((~torch.isfinite(visible_weights)),as_tuple=False)
        visible_weights[indices[:, 0]] = 0

        amodal_weights = amodal_weights * torch.ones(amodal_mask_logits.size(0), mask_side_len, mask_side_len).to(device=amodal_mask_logits.device)
        visible_weights = visible_weights * torch.ones(visible_mask_logits.size(0), mask_side_len, mask_side_len).to(device=visible_mask_logits.device)
        return [amodal_weights, visible_weights]
    #加
    def _create_proposals_from_boxes(self, boxes, image_sizes):
        """
        Args:
            boxes (list[Tensor]): per-image predicted boxes, each of shape Ri x 4
            image_sizes (list[tuple]): list of image shapes in (h, w)
        Returns:
            list[Instances]: per-image proposals with the given boxes.
        """
        # Just like RPN, the proposals should not have gradients
        boxes = [Boxes(b.detach()) for b in boxes]
        proposals = []
        for boxes_per_image, image_size in zip(boxes, image_sizes):
            boxes_per_image.clip(image_size)
            if self.training:
                # do not filter empty boxes at inference time,
                # because the scores from each stage need to be aligned and added later
                boxes_per_image = boxes_per_image[boxes_per_image.nonempty()]
            prop = Instances(image_size)
            prop.proposal_boxes = boxes_per_image
            proposals.append(prop)
        return proposals
    def _match_and_label_boxes(self, proposals, stage, targets):
            """
            Match proposals with groundtruth using the matcher at the given stage.
            Label the proposals as foreground or background based on the match.
            Args:
                proposals (list[Instances]): One Instances for each image, with
                    the field "proposal_boxes".
                stage (int): the current stage
                targets (list[Instances]): the ground truth instances
            Returns:
                list[Instances]: the same proposals, but with fields "gt_classes" and "gt_boxes"
            """
            num_fg_samples, num_bg_samples = [], []
            for proposals_per_image, targets_per_image in zip(proposals, targets):
                match_quality_matrix = pairwise_iou(
                    targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
                )
                # proposal_labels are 0 or 1
                matched_idxs, proposal_labels = self.proposal_matchers[stage](match_quality_matrix)
                if len(targets_per_image) > 0:
                    gt_classes = targets_per_image.gt_classes[matched_idxs]
                    # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
                    gt_classes[proposal_labels == 0] = self.num_classes
                    gt_boxes = targets_per_image.gt_boxes[matched_idxs]
                else:
                    gt_classes = torch.zeros_like(matched_idxs) + self.num_classes
                    gt_boxes = Boxes(
                        targets_per_image.gt_boxes.tensor.new_zeros((len(proposals_per_image), 4))
                    )
                proposals_per_image.gt_classes = gt_classes
                proposals_per_image.gt_boxes = gt_boxes

                num_fg_samples.append((proposal_labels == 1).sum().item())
                num_bg_samples.append(proposal_labels.numel() - num_fg_samples[-1])

            # Log the number of fg/bg samples in each stage
            storage = get_event_storage()
            storage.put_scalar(
                "stage{}/roi_head/num_fg_samples".format(stage),
                sum(num_fg_samples) / len(num_fg_samples),
            )
            storage.put_scalar(
                "stage{}/roi_head/num_bg_samples".format(stage),
                sum(num_bg_samples) / len(num_bg_samples),
            )
            return proposals
    #加
    def _run_stage(self, features, proposals, stage):
            """
            Args:
                features (list[Tensor]): #lvl input features to ROIHeads
                proposals (list[Instances]): #image Instances, with the field "proposal_boxes"
                stage (int): the current stage
            Returns:
                FastRCNNOutputs: the output of this stage
            """
            box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
            # The original implementation averages the losses among heads,
            # but scale up the parameter gradients of the heads.
            # This is equivalent to adding the losses among heads,
            # but scale down the gradients on features.
            box_features = _ScaleGradient.apply(box_features, 1.0 / self.num_cascade_stages)
            box_features = self.box_head[stage](box_features)
            pred_class_logits, pred_proposal_deltas = self.box_predictor[stage](box_features)
            del box_features

            outputs = FastRCNNOutputs(
                self.box2box_transform[stage],
                pred_class_logits,
                pred_proposal_deltas,
                proposals,
                self.smooth_l1_beta,
            )
            return outputs
def get_pred_masks_logits_by_cls(pred_mask_logits, instances):
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)

    if isinstance(instances, list):
        mask_side_len = pred_mask_logits.size(2)
        assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"

        gt_classes = []
        gt_masks = []
        for instances_per_image in instances:
            if len(instances_per_image) == 0:
                continue
            if not cls_agnostic_mask:
                gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
                gt_classes.append(gt_classes_per_image)
            gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
                instances_per_image.proposal_boxes.tensor, mask_side_len
            ).to(device=pred_mask_logits.device)
            # num_regions*28*28
            # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
            gt_masks.append(gt_masks_per_image)
        if len(gt_masks) == 0:
            return pred_mask_logits.sum() * 0
        gt_masks = cat(gt_masks, dim=0)

        if cls_agnostic_mask:
            pred_mask_logits = pred_mask_logits[:, 0]
        else:
            indices = torch.arange(total_num_masks)
            gt_classes = cat(gt_classes, dim=0)
            pred_mask_logits = pred_mask_logits[indices, gt_classes]

        gt_masks = gt_masks.float()

        return pred_mask_logits.unsqueeze(1), gt_masks.unsqueeze(1)
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = instances

        pred_mask_logits = pred_mask_logits[indices, gt_classes]

        return pred_mask_logits.unsqueeze(1)


@ROI_HEADS_REGISTRY.register()
class AmodalROIHeads(ROIHeads):
    """
    A Standard ROIHeads which contains additional heads for the prediction of amodal masks (amodal mask head)
and the occlusion mask (occlusion mask head).
    """

    def __init__(self, cfg, input_shape):
        super(AmodalROIHeads, self).__init__(cfg, input_shape)
        self._init_box_head(cfg)
        self._init_amodal_mask_head(cfg)
        self._init_visible_mask_head(cfg)
        self._init_invisible_mask_head(cfg)

    def _init_box_head(self, cfg):
        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        self.train_on_pred_boxes = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        self.box_predictor = FastRCNNOutputLayers(
            self.box_head.output_size, self.num_classes, self.cls_agnostic_bbox_reg
        )

    def _init_amodal_mask_head(self, cfg):
        # fmt: off
        self.mask_on = cfg.MODEL.MASK_ON
        if not self.mask_on:
            return

        pooler_resolution = cfg.MODEL.ROI_AMODAL_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio = cfg.MODEL.ROI_AMODAL_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_AMODAL_MASK_HEAD.POOLER_TYPE
        # fmt: on
        in_channels = [self.feature_channels[f] for f in self.in_features][0]
        self.mask_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.amodal_mask_head = build_amodal_mask_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def _init_visible_mask_head(self, cfg):
        # fmt: off
        self.mask_on = cfg.MODEL.MASK_ON
        if not self.mask_on:
            return
        pooler_resolution = cfg.MODEL.ROI_VISIBLE_MASK_HEAD.POOLER_RESOLUTION
        # fmt: on
        in_channels = [self.feature_channels[f] for f in self.in_features][0]
        self.visible_mask_head = build_visible_mask_head(                               # this mask head means visible mask head
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def _init_invisible_mask_head(self, cfg):
        # fmt: off
        self.mask_on = cfg.MODEL.MASK_ON
        if not self.mask_on:
            return
        self.invisible_mask_head = build_invisible_mask_head(cfg)

    def _forward_amodal_mask(self, features: List[torch.Tensor], instances: List[Instances]):
        """
        Forward logic of the mask prediction branch.

        Args:
            features (list[Tensor]): #level input features for mask prediction
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses and pred_mask_logits
            In inference, update `instances` with new fields "pred_masks" and return it and pred_mask_logits
        """
        if not self.mask_on:
            return {} if self.training else instances
        if self.training:
            # The loss is only defined on positive proposals.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_features = self.mask_pooler(features, proposal_boxes)
            return self.amodal_mask_head(mask_features, proposals)
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            mask_features = self.mask_pooler(features, pred_boxes)
            return self.amodal_mask_head(mask_features, instances)

    def _forward_visible_mask(self, features: List[torch.Tensor], instances: List[Instances]):
        """
        Forward logic of the mask prediction branch.

        Args:
            features (list[Tensor]): #level input features for mask prediction
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses and pred_mask_logits
            In inference, update `instances` with new fields "pred_masks" and return it and pred_mask_logits
        """
        if not self.mask_on:
            return {} if self.training else instances
        if self.training:
            # The loss is only defined on positive proposals.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_features = self.mask_pooler(features, proposal_boxes)
            return self.visible_mask_head(mask_features, proposals)     #    This mask head = visible mask head
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            mask_features = self.mask_pooler(features, pred_boxes)
            return self.visible_mask_head(mask_features, instances)     # This mask head = visible mask head

    def _forward_invisible_mask(self, pred_amodal_mask_logits, pred_visible_mask_logits, instances):
        if not self.mask_on:
            return {} if self.training else instances

        if self.training:
            # The loss is only defined on positive proposals.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            pred_invisible_mask_logtis = pred_amodal_mask_logits - F.relu(pred_visible_mask_logits)
            return self.invisible_mask_head(pred_invisible_mask_logtis, proposals)
        else:
            pred_invisible_mask_logtis = pred_amodal_mask_logits - F.relu(pred_visible_mask_logits)
            return self.invisible_mask_head(pred_invisible_mask_logtis, instances)

    def forward(
            self,
            images: ImageList,
            features: Dict[str, torch.Tensor],
            proposals: List[Instances],
            targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        features_list = [features[f] for f in self.in_features]

        if self.training:
            losses = self._forward_box(features_list, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            amodal_mask_loss, amodal_mask_logits = self._forward_amodal_mask(features_list, proposals)
            losses.update(amodal_mask_loss)
            visible_mask_loss, visible_mask_logits = self._forward_visible_mask(features_list, proposals)
            losses.update(visible_mask_loss)
            losses.update(self._forward_invisible_mask(amodal_mask_logits, visible_mask_logits, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features_list, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def _forward_box(
            self, features: List[torch.Tensor], proposals: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features)
        del box_features

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )
        if self.training:
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = outputs.predict_boxes_for_gt_classes()
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return outputs.losses()
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            return pred_instances

    def forward_with_given_boxes(
            self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (list[Instances]):
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`. and 'pred_visible_masks'
        """

        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")
        features_list = [features[f] for f in self.in_features]
        instances, amodal_mask_logits = self._forward_amodal_mask(features_list, instances)
        instances, visible_mask_logits = self._forward_visible_mask(features_list, instances)
        instances = self._forward_invisible_mask(amodal_mask_logits, visible_mask_logits, instances)
        return instances