# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import Image
import copy
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
import pdb
from .memo_functions import vq, vq_st
from typing import List
from detectron2.structures import Instances
from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from detectron2.modeling.backbone.ODConv import ODConv2d
import numpy as np
import matplotlib.pyplot as plt
import datetime, time
# from detectron2.layers import conv_with_kaiming_uniform
ROI_MASK_HEAD_REGISTRY = Registry("ROI_MASK_HEAD")
ROI_MASK_HEAD_REGISTRY.__doc__ = """
Registry for mask heads, which predicts instance masks given
per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""

def dice_loss_func(input, target):
    smooth = 1.
    n = input.size(0)
    iflat = input.view(n, -1)
    tflat = target.view(n, -1)
    intersection = (iflat * tflat).sum(1)
    loss = 1 - ((2. * intersection + smooth) /
                (iflat.sum(1) + tflat.sum(1) + smooth))
    return loss.mean()

def boundary_loss_func(boundary_logits, gtmasks):
    """
    Args:
        boundary_logits (Tensor): A tensor of shape (B, H, W) or (B, H, W)
        gtmasks (Tensor): A tensor of shape (B, H, W) or (B, H, W)
    """
    laplacian_kernel = torch.tensor(
        [-1, -1, -1, -1, 8, -1, -1, -1, -1],
        dtype=torch.float32, device=boundary_logits.device).reshape(1, 1, 3, 3).requires_grad_(False)
    boundary_logits = boundary_logits.unsqueeze(1)
    boundary_targets = F.conv2d(gtmasks.unsqueeze(1), laplacian_kernel, padding=1)
    boundary_targets = boundary_targets.clamp(min=0)
    boundary_targets[boundary_targets > 0.1] = 1
    boundary_targets[boundary_targets <= 0.1] = 0

    if boundary_logits.shape[-1] != boundary_targets.shape[-1]:
        boundary_targets = F.interpolate(
            boundary_targets, boundary_logits.shape[2:], mode='nearest')

    bce_loss = F.binary_cross_entropy_with_logits(boundary_logits, boundary_targets)
    dice_loss = dice_loss_func(torch.sigmoid(boundary_logits), boundary_targets)
    return bce_loss + dice_loss

def mask_rcnn_loss(pred_mask_logits, instances):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.

    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
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

    if gt_masks.dtype == torch.bool:
        gt_masks_bool = gt_masks
    else:
        # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
        gt_masks_bool = gt_masks > 0.5
    # Log the training accuracy (using gt classes and 0.5 threshold)
    mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool
    mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
    num_positive = gt_masks_bool.sum().item()
    false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
        gt_masks_bool.numel() - num_positive, 1.0
    )
    false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(num_positive, 1.0)

    storage = get_event_storage()
    storage.put_scalar("mask_rcnn/accuracy", mask_accuracy)
    storage.put_scalar("mask_rcnn/false_positive", false_positive)
    storage.put_scalar("mask_rcnn/false_negative", false_negative)
    mask_loss = F.binary_cross_entropy_with_logits(
        pred_mask_logits, gt_masks.to(dtype=torch.float32), reduction="mean"
    )
    return mask_loss


def amodal_mask_rcnn_loss(pred_mask_logits,instances, weights=None, mode="amodal", version="n"):


    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
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
        if mode == "amodal":
            gt_per_image = instances_per_image.gt_masks.crop_and_resize(
                instances_per_image.proposal_boxes.tensor, mask_side_len
            ).to(device=pred_mask_logits.device)
        elif mode == "visible":
            gt_per_image = instances_per_image.gt_visible_masks.crop_and_resize(
                instances_per_image.proposal_boxes.tensor, mask_side_len
            ).to(device=pred_mask_logits.device)
        elif mode =="invisible":
            gt_per_image = instances_per_image.gt_occluded_masks.crop_and_resize(
                instances_per_image.proposal_boxes.tensor, mask_side_len
            ).to(device=pred_mask_logits.device)
        # num_regions*28*28
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len for amodal or visible
        gt_masks.append(gt_per_image)
    if len(gt_masks) == 0:
        return pred_mask_logits.sum() * 0
    gt_masks = cat(gt_masks, dim=0)     #gt_mask ROI Align一下


    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]

    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        pred_mask_logits = pred_mask_logits[indices, gt_classes]


    if gt_masks.dtype == torch.bool:
        gt_masks_bool = gt_masks
    else:
        # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
        gt_masks_bool = gt_masks > 0.5

    # Log the training accuracy for amodal mask(using gt classes and 0.5 threshold)
    mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool
    mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
    num_positive = gt_masks_bool.sum().item()
    false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
        gt_masks_bool.numel() - num_positive, 1.0
    )
    false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(num_positive, 1.0)

    storage = get_event_storage()
    storage.put_scalar("mask_rcnn/{}_{}_accuracy".format(mode, version), mask_accuracy)
    storage.put_scalar("mask_rcnn/{}_{}_false_positive".format(mode, version), false_positive)
    storage.put_scalar("mask_rcnn/{}_{}_false_negative".format(mode, version), false_negative)

    if isinstance(weights, float):
        mask_loss = weights * F.binary_cross_entropy_with_logits(
            pred_mask_logits, gt_masks.to(dtype=torch.float32), reduction="mean"
        )
    else:
        mask_loss = F.binary_cross_entropy_with_logits(
            pred_mask_logits, gt_masks.to(dtype=torch.float32), weight=weights, reduction="mean"
        )


    # return mask_loss
    return mask_loss


def boundary_loss(pred_mask_logits,pred_boundary_logits,instances, weights=None, mode="amodal", version="n"):

    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
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
        if mode == "amodal":
            gt_per_image = instances_per_image.gt_masks.crop_and_resize(
                instances_per_image.proposal_boxes.tensor, mask_side_len
            ).to(device=pred_mask_logits.device)
        elif mode == "visible":
            gt_per_image = instances_per_image.gt_visible_masks.crop_and_resize(
                instances_per_image.proposal_boxes.tensor, mask_side_len
            ).to(device=pred_mask_logits.device)
        elif mode == "occluder":
            gt_per_image = instances_per_image.gt_occluded_masks.crop_and_resize(
                instances_per_image.proposal_boxes.tensor, mask_side_len
            ).to(device=pred_mask_logits.device)
        # num_regions*28*28
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len for amodal or visible
        gt_masks.append(gt_per_image)
    if len(gt_masks) == 0:
        return pred_mask_logits.sum() * 0
    gt_masks = cat(gt_masks, dim=0)     #gt_mask ROI Align一下


    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
        pred_boundary_logits = pred_boundary_logits[:,0]
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        pred_mask_logits = pred_mask_logits[indices, gt_classes]
        pred_boundary_logits = pred_boundary_logits[indices,gt_classes]
    if gt_masks.dtype == torch.bool:
        gt_masks_bool = gt_masks
    else:
        # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
        gt_masks_bool = gt_masks > 0.5
    gt_masks = gt_masks.to(dtype=torch.float32)
    boundary_loss = boundary_loss_func(pred_boundary_logits,gt_masks)

    # return mask_loss
    return boundary_loss
def mask_fm_loss(features, betas):
    assert len(features) != 0
    loss = 0
    betas = betas * 2 if len(features[0]) == 2 * len(betas) else betas
    if len(features) == 2:
        for f1, f2, beta in zip(features[0], features[1], betas):
            n = f1.size(0)
            loss += beta * torch.mean(1 - F.cosine_similarity(f1.view(n, -1), f2.view(n, -1).detach()))
            # loss += F.mse_loss(f1, f2.detach()) * beta / len(betas)
    if len(features) == 3:
        for f1, f2, f3, beta in zip(features[0], features[1], features[2], betas):
            n = f1.size(0)
            loss += (3 - F.cosine_similarity(f1.view(n, -1), f2.view(n, -1)) * beta +
                     F.cosine_similarity(f2.view(n, -1), f3.view(n, -1)) * beta +
                     F.cosine_similarity(f1.view(n, -1), f3.view(n, -1)) / 3)

    return loss


def mask_rcnn_inference(pred_mask_logits, pred_instances):
    """
    Convert pred_mask_logits to estimated foreground probability masks while also
    extracting only the masks for the predicted classes in pred_instances. For each
    predicted box, the mask of the same class is attached to the instance by adding a
    new "pred_masks" field to pred_instances.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. Each Instances must have field "pred_classes".

    Returns:
        None. pred_instances will contain an extra "pred_masks" field storing a mask of size (Hmask,
            Wmask) for predicted class. Note that the masks are returned as a soft (non-quantized)
            masks the resolution predicted by the network; post-processing steps, such as resizing
            the predicted masks to the original image resolution and/or binarizing them, is left
            to the caller.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1

    if cls_agnostic_mask:
        mask_probs_pred = pred_mask_logits.sigmoid()
    else:
        # Select masks corresponding to the predicted classes
        num_masks = pred_mask_logits.shape[0]
        class_pred = cat([i.pred_classes for i in pred_instances])
        indices = torch.arange(num_masks, device=class_pred.device)
        mask_probs_pred = pred_mask_logits[indices, class_pred][:, None].sigmoid()
    # mask_probs_pred.shape: (B, 1, Hmask, Wmask)

    num_boxes_per_image = [len(i) for i in pred_instances]
    mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)

    for prob, instances in zip(mask_probs_pred, pred_instances):
        instances.pred_masks = prob  # (1, Hmask, Wmask)


def amodal_mask_rcnn_inference(multi_pred_mask_logits, pred_instances):
    # amodal processing
    # tmp_pred=multi_pred_mask_logits[0].unsqueeze(0)
    # print(tmp_pred.shape)
    # multi_pred_mask_logits2=[]
    # multi_pred_mask_logits2.append(tmp_pred)
    pred_mask_logits_lst = []
    for i in multi_pred_mask_logits:
        pred_mask_logits_lst += [x for x in i]

    for i in range(len(pred_mask_logits_lst)):
        pred_mask_logits = pred_mask_logits_lst[i]
        cls_agnostic_mask = pred_mask_logits.size(1) == 1

        if cls_agnostic_mask:
            mask_probs_pred = pred_mask_logits.sigmoid()
        else:
            # Select masks corresponding to the predicted classes
            num_masks = pred_mask_logits.shape[0]
            class_pred = cat([i.pred_classes for i in pred_instances])
            indices = torch.arange(num_masks, device=class_pred.device)

            #print(pred_mask_logits.shape)
            #print(pred_instances)
            #print(class_pred)

            mask_probs_pred = pred_mask_logits[indices,class_pred][:, None].sigmoid()
        # mask_probs_pred.shape: (B, 1, Hmask, Wmask)
        #print(mask_probs_pred.shape)
        num_boxes_per_image = [len(i) for i in pred_instances]
        mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)

        for prob, instances in zip(mask_probs_pred, pred_instances):
            if i == 0:
                instances.pred_amodal_masks = prob  # (1, Hmask, Wmask)
            elif i == 1:
                instances.pred_visible_masks = prob  # (1, Hmask, Wmask)
            elif i == 2:
                instances.pred_amodal2_masks = prob  # (1, Hmask, Wmask)
            # elif i == 3:
            #     instances.pred_final_amodal = prob  # (1, Hmask, Wmask)



class BaseMaskRCNNHead(nn.Module):
    """
    Implement the basic Mask R-CNN losses and inference logic.
    """

    def __init__(self, cfg, input_shape):
        super().__init__()
        self.vis_period = cfg.VIS_PERIOD

    def forward(self, x, instances: List[Instances]):
        """
        Args:
            x: input region feature(s) provided by :class:`ROIHeads`.
            instances (list[Instances]): contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.
        Returns:
            A dict of losses in training. The predicted "instances" in inference.
        """
        x = self.layers(x)
        if self.training:
            return {"loss_mask": mask_rcnn_loss(x, instances, self.vis_period)}
        else:
            mask_rcnn_inference(x, instances)
            return instances

    def layers(self, x):
        """
        Neural network layers that makes predictions from input features.
        """
        raise NotImplementedError

# from ops_dcnv3.modules import DCNv3
# from torch import nn
# class dcv_block(nn.Module):
#     def __init__(self,channel,kernel=1,stride=1,group=1,dilation=1,activation=True) -> None:
#         super(dcv_block,self).__init__()
#         self.dcnv3=DCNv3(channel,kernel_size=kernel,stride=stride,group=group,dilation=dilation,pad=0)
#         self.bn=nn.BatchNorm2d(channel)
#         default_act=nn.SiLU()
#         self.activate_func=default_act if activation is True else activation if isinstance(activation, nn.Module) else nn.Identity()
#     def forward(self,x):
#         x=x.permute(0,2,3,1)
#         x=self.dcnv3(x)
#         x=x.permute(0,3,1,2)
#         x=self.activate_func(self.bn(x))
#         return x
# class dcv_bottle_neck(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
@ROI_MASK_HEAD_REGISTRY.register()
class Parallel_Amodal_Visible_Head(nn.Module):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv: the number of conv layers
            conv_dim: the dimension of the conv layers
            norm: normalization for the conv layers
        """
        super(Parallel_Amodal_Visible_Head, self).__init__()

        # fmt: off
        self.cfg = cfg
        num_classes       = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        conv_dims         = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        self.norm         = cfg.MODEL.ROI_MASK_HEAD.NORM
        num_conv          = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        # num_vis_conv      = cfg.MODEL.ROI_MASK_HEAD.NUM_VIS_CONV
        self.fm           = cfg.MODEL.ROI_MASK_HEAD.AMODAL_FEATURE_MATCHING
        self.fm_beta      = cfg.MODEL.ROI_MASK_HEAD.AMODAL_FM_BETA
        input_channels    = input_shape.channels
        cls_agnostic_mask = cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK
        self.SPRef        = cfg.MODEL.ROI_MASK_HEAD.RECON_NET.MEMORY_REFINE
        self.SPk          = cfg.MODEL.ROI_MASK_HEAD.RECON_NET.MEMORY_REFINE_K
        self.version      = cfg.MODEL.ROI_MASK_HEAD.VERSION
        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)
        self.attention_mode = cfg.MODEL.ROI_MASK_HEAD.ATTENTION_MODE
        # fmt: on
        num_boundary_conv = 2
        self.use_boundary = cfg.MODEL.ROI_HEADS.USE_BOUNDARY
        self.amodal_conv_norm_relus = []
        self.visible_conv_norm_relus = []
        self.refine_conv_norm_relus = []
        self.RAB = []
        self.idx=0
        for k in range(num_conv):
            a_conv = Conv2d(
                input_channels if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, conv_dims),
                activation=F.relu,
            )
            self.add_module("amodal_mask_fcn{}".format(k + 1), a_conv)
            self.amodal_conv_norm_relus.append(a_conv)

            v_conv = Conv2d(
                input_channels if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, conv_dims),
                activation=F.relu,
            )
            self.add_module("visible_mask_fcn{}".format(k + 1), v_conv)
            self.visible_conv_norm_relus.append(v_conv)

            r_conv = Conv2d(
                input_channels if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, conv_dims),
                activation=F.relu,
            )
            self.add_module("refine_mask_fcn{}".format(k+1),r_conv)
            self.refine_conv_norm_relus.append(r_conv)
        # self.refine_conv_norm_relus = nn.Sequential(
        #     ODConv2d(256, 256, kernel_size=3, stride=1, padding=1,
        #             reduction=0.0625, kernel_num=4),
        #     # nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     ODConv2d(256, 256, kernel_size=3, stride=1, padding=1,
        #             reduction=0.0625, kernel_num=4),
        #     # nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     ODConv2d(256, 256, kernel_size=3, stride=1, padding=1,
        #             reduction=0.0625, kernel_num=4),
        #     # nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     ODConv2d(256, 256, kernel_size=3, stride=1, padding=1,
        #             reduction=0.0625, kernel_num=4),
        #     # nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),

        # )
        
        self.amodal_pool = nn.AvgPool2d(kernel_size=2)
        self.amodal_deconv = ConvTranspose2d(
            conv_dims if num_conv > 0 else input_channels,
            conv_dims,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        self.visible_deconv = ConvTranspose2d(
                conv_dims if num_conv > 0 else input_channels,
                conv_dims,
                kernel_size=2,
                stride=2,
                padding=0,
            )
        self.r_deconv = ConvTranspose2d(
                conv_dims if num_conv > 0 else input_channels,
                conv_dims,
                kernel_size=2,
                stride=2,
                padding=0,
            )
        self.dcn_group_v =[]
        self.dcn_group_a =[]
        self.query_v = Conv2d(input_channels,input_channels,1,1,0)
        self.key_v = Conv2d(input_channels,input_channels,1,1,0)
        self.value_v = Conv2d(input_channels,input_channels,1,1,0)
        self.query_a = Conv2d(input_channels,input_channels,1,1,0)
        
        self.key_a = Conv2d(input_channels,input_channels,1,1,0)
        self.value_a = Conv2d(input_channels,input_channels,1,1,0)
        self.out_v = Conv2d(input_channels,input_channels,1,1,0)
        self.out_a = Conv2d(input_channels,input_channels,1,1,0)
        self.blocker_a = nn.BatchNorm2d(input_channels, eps=1e-04) # should be zero initialized
        self.blocker_v = nn.BatchNorm2d(input_channels, eps=1e-04) # should be zero initialized


        self.dcn_group_v.append(self.query_v)
        self.dcn_group_v.append(self.key_v)
        self.dcn_group_v.append(self.value_v)
        self.dcn_group_v.append(self.out_v)
        # self.dcn_group_v.append(self.blocker_v)
        self.dcn_group_a.append(self.query_a)
        self.dcn_group_a.append(self.key_a)
        self.dcn_group_a.append(self.value_a)
        self.dcn_group_a.append(self.out_a)
        # self.dcn_group_a.append(self.blocker_a)
        # for k in range(len(self.dcn_group_v)):
        #         self.add_module("dcn_v{}".format(k+1),self.dcn_group_v[k])
        for k in range(len(self.dcn_group_a)):
                self.add_module("dcn_a{}".format(k+1),self.dcn_group_a[k])
        # for layer in self.dcn_group_v:
        #     weight_init.c2_msra_fill(layer)
        for layer in self.dcn_group_a:
            weight_init.c2_msra_fill(layer)       
        # # self.RAB1 = Conv2d(
        #         conv_dims,
        #         conv_dims,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1,
        #         bias=not self.norm,
        #         norm=get_norm(self.norm, conv_dims),
        #         activation=F.relu,
        #     )
        # self.RAB2 = Conv2d(
        #         conv_dims,
        #         conv_dims,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1,
        #         bias=not self.norm,
        #         norm=get_norm(self.norm, conv_dims),
        #         activation=F.relu,
        #     )
        # self.RAB3 =Conv2d(
        #         conv_dims,
        #         conv_dims,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1,
        #         bias=not self.norm,
        #         norm=get_norm(self.norm, conv_dims),
        #         activation=F.relu,
        #     )
        # self.RAB4=Conv2d(
        #         conv_dims,
        #         conv_dims,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1,
        #         bias=not self.norm,
        #         norm=get_norm(self.norm, conv_dims),
        #         activation=F.relu,
        #     )

        self.mask_final_fusion = Conv2d(
            conv_dims, conv_dims,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=not self.norm,
            norm=get_norm(self.norm, conv_dims),
            activation=F.relu)

        self.downsample = Conv2d(
            conv_dims, conv_dims,
            kernel_size=3,
            padding=1,
            stride=2,
            bias=not self.norm,
            norm=get_norm(self.norm, conv_dims),
            activation=F.relu
        )

        cur_channels = input_shape.channels


        self.mask_to_boundary = Conv2d(
            conv_dims, conv_dims,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=not self.norm,
            norm=get_norm(self.norm, conv_dims),
            activation=F.relu
        )

        self.boundary_to_mask = Conv2d(
            conv_dims, conv_dims,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=not self.norm,
            norm=get_norm(self.norm, conv_dims),
            activation=F.relu
        )
        self.boundary_deconv = ConvTranspose2d(
            conv_dims, conv_dims, kernel_size=2, stride=2, padding=0
        )

        self.boundary_predictor = Conv2d(cur_channels, num_classes, kernel_size=1, stride=1, padding=0)

 

        self.mask_final_fusion_v = Conv2d(
            conv_dims, conv_dims,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=not self.norm,
            norm=get_norm(self.norm, conv_dims),
            activation=F.relu)


        self.mask_to_boundary_v = Conv2d(
            conv_dims, conv_dims,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=not self.norm,
            norm=get_norm(self.norm, conv_dims),
            activation=F.relu
        )

        self.boundary_to_mask_v = Conv2d(
            conv_dims, conv_dims,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=not self.norm,
            norm=get_norm(self.norm, conv_dims),
            activation=F.relu
        )
        self.boundary_deconv_v = ConvTranspose2d(
            conv_dims, conv_dims, kernel_size=2, stride=2, padding=0
        )

        self.boundary_predictor_v = Conv2d(cur_channels, num_classes, kernel_size=1, stride=1, padding=0)

        # self.mask_final_fusion_o = Conv2d(
        #     conv_dims, conv_dims,
        #     kernel_size=3,
        #     padding=1,
        #     stride=1,
        #     bias=not self.norm,
        #     norm=get_norm(self.norm, conv_dims),
        #     activation=F.relu)


        # self.mask_to_boundary_o = Conv2d(
        #     conv_dims, conv_dims,
        #     kernel_size=1,
        #     padding=0,
        #     stride=1,
        #     bias=not self.norm,
        #     norm=get_norm(self.norm, conv_dims),
        #     activation=F.relu
        # )

        # self.boundary_to_mask_o = Conv2d(
        #     conv_dims, conv_dims,
        #     kernel_size=1,
        #     padding=0,
        #     stride=1,
        #     bias=not self.norm,
        #     norm=get_norm(self.norm, conv_dims),
        #     activation=F.relu
        # )
        # self.boundary_deconv_o = ConvTranspose2d(
        #     conv_dims, conv_dims, kernel_size=2, stride=2, padding=0
        # )

        self.boundary_fcns = []
        cur_channels = input_shape.channels
        for k in range(num_boundary_conv):
            conv = Conv2d(
                cur_channels,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, conv_dims),
                activation=F.relu,
            )
            self.add_module("boundary_fcn{}".format(k + 1), conv)
            self.boundary_fcns.append(conv)
            cur_channels = conv_dims
        self.boundary_predictor_occluder = Conv2d(cur_channels, num_classes, kernel_size=1, stride=1, padding=0)
        self.boundary_fcns_visible = []
        cur_channels = input_shape.channels
        for k in range(num_boundary_conv):
            conv = Conv2d(
                cur_channels,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, conv_dims),
                activation=F.relu,
            )
            self.add_module("boundary_fcn_v{}".format(k + 1), conv)
            self.boundary_fcns_visible.append(conv)
            cur_channels = conv_dims
        # self.boundary_fcns_occluder = []
        # for k in range(num_boundary_conv):
        #     conv = Conv2d(
        #         cur_channels,
        #         conv_dims,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1,
        #         bias=not self.norm,
        #         norm=get_norm(self.norm, conv_dims),
        #         activation=F.relu,
        #     )
        #     self.add_module("boundary_fcn_occluder{}".format(k + 1), conv)
        #     self.boundary_fcns_occluder.append(conv)
        #     cur_channels = conv_dims
        num_mask_classes = 1 if cls_agnostic_mask else num_classes
        self.amodal_predictor = Conv2d(conv_dims, num_mask_classes, kernel_size=1, stride=1, padding=0)
        self.visible_predictor = Conv2d(conv_dims, num_mask_classes, kernel_size=1, stride=1, padding=0)
        self.r_predictor = Conv2d(conv_dims,1,kernel_size=1,stride=1,padding=0)

        nn.init.normal_(self.boundary_predictor.weight, std=0.001)
        nn.init.normal_(self.boundary_predictor_v.weight, std=0.001)
        # nn.init.normal_(self.boundary_predictor_occluder.weight, std=0.001)
        nn.init.normal_(self.amodal_predictor.weight, std=0.001)
        nn.init.normal_(self.r_predictor.weight,std=0.001)
        nn.init.normal_(self.visible_predictor.weight, std=0.001)

        if self.amodal_predictor.bias is not None:
            nn.init.constant_(self.amodal_predictor.bias, 0)
        if self.visible_predictor.bias is not None:
            nn.init.constant_(self.visible_predictor.bias, 0)
        if self.r_predictor.bias is not None:
            nn.init.constant_(self.r_predictor.bias, 0)
        if self.boundary_predictor.bias is not None:
            nn.init.constant_(self.boundary_predictor.bias, 0.001)
        if self.boundary_predictor_v.bias is not None:
            nn.init.constant_(self.boundary_predictor_v.bias, 0.001)
        # if self.boundary_predictor_v.bias is not None:
        #     nn.init.constant_(self.boundary_predictor_occluder.bias, 0.001)

        for layer in self.amodal_conv_norm_relus + [self.amodal_deconv] + self.visible_conv_norm_relus + [self.visible_deconv] +\
            [self.boundary_deconv,self.boundary_to_mask,self.mask_to_boundary,self.mask_final_fusion,self.downsample,self.boundary_deconv_v,self.boundary_to_mask_v,self.mask_to_boundary_v,self.mask_final_fusion_v
             ]:
            weight_init.c2_msra_fill(layer)
        for layer in self.boundary_fcns:
            weight_init.c2_msra_fill(layer)
        for layer in self.boundary_fcns_visible:
            weight_init.c2_msra_fill(layer)
        # for layer in self.boundary_fcns_occluder:
        #     weight_init.c2_msra_fill(layer)

    def forward(self, x, boundary_features,instances=None,train=None):
        output_mask_logits = []
        boundary_features=self.downsample(boundary_features)
        if train:
            if self.use_boundary:
                masks_logits, boundary_logits,feature_matching = self.forward_through(x, x,boundary_features,instances)
                output_mask_logits.append(masks_logits)
                return output_mask_logits, boundary_logits,feature_matching
            else:
                 masks_logits = self.forward_through_without_boundary(x,x)
                 output_mask_logits.append(masks_logits)
                 return output_mask_logits
        else:
            if self.use_boundary:
                masks_logits = self.forward_through_evaluate(x, x,boundary_features,instances)
                #print(masks_logits.shape)
                output_mask_logits.append(masks_logits)
                return output_mask_logits
            else:
                masks_logits = self.forward_through_without_boundary(x,x) 
                output_mask_logits.append(masks_logits)
                return output_mask_logits
        


    def forward_through(self, x1, x2, boundary_features,instances):       #Parallel_Amodal_Visible_Head
        features=[]
        B, C, H, W = x1.size()
        for layer in self.visible_conv_norm_relus:
            visible_features = layer(x2)
                # x: B,C,H,W
        # x_query: B,C,HW
        #x_input = AddCoords()(x)
        # if visible_features.shape[0]!= 0:
        #     query_v = self.query_v(visible_features).view(B, C, -1)
        #     # x_query: B,HW,C
        #     query_v = torch.transpose(query_v, 1, 2)
        #     # x_key: B,C,HW
        #     key_v = self.key_v(x2).view(B, C, -1)
        #     # x_value: B,C,HW
        #     value_v = self.value_v(x2).view(B, C, -1)
        #     # x_value: B,HW,C
        #     value_v = torch.transpose(value_v, 1, 2)
        #     # W = Q^T K: B,HW,HW
        #     w_v = torch.matmul(query_v, key_v) * (1.0/128)
        #     w_v = F.softmax(w_v, dim=-1)
        #     # x_relation = WV: B,HW,C
        #     relation_v = torch.matmul(w_v, value_v)
        #     # x_relation = B,C,HW
        #     relation_v = torch.transpose(relation_v, 1, 2)
        #     # x_relation = B,C,H,W
        #     relation_v = relation_v.view(B,C,H,W)

        #     relation_v = self.out_v(relation_v)
        #     relation_v = self.blocker_v(relation_v)
            
        #     visible_features = visible_features + relation_v
        features.append(visible_features)
        boundary_features_visible=boundary_features+self.mask_to_boundary_v(visible_features)
        if boundary_features_visible.shape[0]!= 0:
            query_v = self.query_v(boundary_features_visible).view(B, C, -1)
            # x_query: B,HW,C
            query_v = torch.transpose(query_v, 1, 2)
            # x_key: B,C,HW
            key_v = self.key_v(boundary_features_visible).view(B, C, -1)
            # x_value: B,C,HW
            value_v = self.value_v(boundary_features_visible).view(B, C, -1)
            # x_value: B,HW,C
            value_v = torch.transpose(value_v, 1, 2)
            # W = Q^T K: B,HW,HW
            w_v = torch.matmul(query_v, key_v) * (1.0/128)
            w_v = F.softmax(w_v, dim=-1)
            # x_relation = WV: B,HW,C
            relation_v = torch.matmul(w_v, value_v)
            # x_relation = B,C,HW
            relation_v = torch.transpose(relation_v, 1, 2)
            # x_relation = B,C,H,W
            relation_v = relation_v.view(B,C,H,W)

            relation_v = self.out_v(relation_v)
            relation_v = self.blocker_v(relation_v)
            
            boundary_features_visible = boundary_features_visible + relation_v



        for layer in self.boundary_fcns_visible:
            boundary_features_visible=layer(boundary_features_visible)
        visible_features = self.boundary_to_mask_v(boundary_features_visible)+visible_features
        visible_features = self.mask_final_fusion_v(visible_features)
        features.append(visible_features)
        visible_features = F.relu(self.visible_deconv(visible_features), inplace=True)

        visible_mask_logits = self.visible_predictor(visible_features)
        boundary_features_visible=F.relu(self.boundary_deconv_v(boundary_features_visible))
        boundary_logits_visible = self.boundary_predictor_v(boundary_features_visible)



        for layer in self.amodal_conv_norm_relus:
            amodal_features = layer(x1+features[0])
                # x_query: B,C,HW
        #x_input = AddCoords()(x)
        # if amodal_features.shape[0]!=0:
        #     query_a = self.query_a(amodal_features).view(B, C, -1)
        #     # x_query: B,HW,C
        #     query_a = torch.transpose(query_a, 1, 2)
        #     # x_key: B,C,HW
        #     key_a = self.key_a(x1).view(B, C, -1)
        #     # x_value: B,C,HW
        #     value_a = self.value_a(x1).view(B, C, -1)
        #     # x_value: B,HW,C
        #     value_a = torch.transpose(value_a, 1, 2)
        #     # W = Q^T K: B,HW,HW
        #     w_a = torch.matmul(query_a, key_a) * (1.0/128)
        #     w_a = F.softmax(w_a, dim=-1)
        #     # x_relation = WV: B,HW,C
        #     relation_a = torch.matmul(w_a, value_a)
        #     # x_relation = B,C,HW
        #     relation_a = torch.transpose(relation_a, 1, 2)
        #     # x_relation = B,C,H,W
        #     relation_a = relation_a.view(B,C,H,W)

        #     relation_a = self.out_a(relation_a)
        #     relation_a = self.blocker_a(relation_a)
        #     amodal_features = amodal_features + relation_a
        features.append(amodal_features)
        boundary_features1 = boundary_features + self.mask_to_boundary(amodal_features)
        if boundary_features1.shape[0]!= 0:
            query_v = self.query_v(boundary_features1).view(B, C, -1)
            # x_query: B,HW,C
            query_v = torch.transpose(query_v, 1, 2)
            # x_key: B,C,HW
            key_v = self.key_v(boundary_features1).view(B, C, -1)
            # x_value: B,C,HW
            value_v = self.value_v(boundary_features1).view(B, C, -1)
            # x_value: B,HW,C
            value_v = torch.transpose(value_v, 1, 2)
            # W = Q^T K: B,HW,HW
            w_v = torch.matmul(query_v, key_v) * (1.0/128)
            w_v = F.softmax(w_v, dim=-1)
            # x_relation = WV: B,HW,C
            relation_v = torch.matmul(w_v, value_v)
            # x_relation = B,C,HW
            relation_v = torch.transpose(relation_v, 1, 2)
            # x_relation = B,C,H,W
            relation_v = relation_v.view(B,C,H,W)

            relation_v = self.out_v(relation_v)
            relation_v = self.blocker_v(relation_v)
            
            boundary_features1 = boundary_features1 + relation_v
        for layer in self.boundary_fcns:
            boundary_features1 = layer(boundary_features1)
        amodal_features = self.boundary_to_mask(boundary_features1) + amodal_features 
        amodal_features = self.mask_final_fusion(amodal_features)
        amodal = F.relu(self.amodal_deconv(amodal_features), inplace=True)
        amodal_mask_logits = self.amodal_predictor(amodal)

        boundary_features1 = F.relu(self.boundary_deconv(boundary_features1))
        boundary_logits = self.boundary_predictor(boundary_features1)
        # RAB_A= self.RAB1(features[0])
        # RAB_V =self.RAB2(features[1])
        # amodal_refine_features = torch.max(RAB_A,RAB_V)


        # # amodal_fusion_features = self.RAB3(features[1])
        # # ov_fusion_features = self.RAB4(amodal_refine_features)
        # # amodal_final_features = torch.max(ov_fusion_features,amodal_fusion_features)
        # # for layer in self.amodal_conv_norm_relus:
        # #     amodal_refine_features = layer(amodal_refine_features)
        # for layer in self.amodal_conv_norm_relus:
        #     amodal_refine_features =layer(amodal_refine_features)
        
        # amodal_final = self.amodal_deconv(amodal_refine_features)
        # amodal_final = self.amodal_predictor(amodal_final)
        # amodal_final = self.amodal_deconv(amodal_final_features)
        # # amodal_final = self.amodal_predictor(amodal_final)
        feature_matching=[]
        amodal_attention = self.amodal_pool(classes_choose(amodal_mask_logits, instances)).unsqueeze(1).sigmoid()
        amodal_features_refine = amodal_attention * x1 +x1
        for layer in self.refine_conv_norm_relus:
            amodal_features_refine = layer(amodal_features_refine)
            feature_matching.append(amodal_features_refine)
        amodal_refine = self.r_deconv(amodal_features_refine)
        feature_matching.append(amodal_refine)
        amodal_refine_logits = self.r_predictor(amodal_refine)
        if instances[0].has("gt_masks"):
            mask_side_len = x1.size(2)
            gt_attention, _ = get_gt_masks(instances, mask_side_len, amodal_refine_logits)
        amodal_features_gt = gt_attention * x1 +x1
        for layer in self.refine_conv_norm_relus:
            amodal_features_gt=layer(amodal_features_gt)
            feature_matching.append(amodal_features_gt)
        amodal_gt = self.r_deconv(amodal_features_gt)
        feature_matching.append(amodal_gt)
        amodal_gt_logits = self.r_predictor(amodal_gt)
        features_amodal=[]
        features_amodal.append(feature_matching[3])
        features_amodal.append(feature_matching[4])
        features_gt=[]
        features_gt.append(feature_matching[8])
        features_gt.append(feature_matching[9])
        # attention = visible_attention ^ occluder_attention
        del feature_matching
        feature_fm=[]
        feature_fm.append(features_amodal)
        feature_fm.append(features_gt)
        # fusion = union_attention * x1
        # for layer in self.amodal_conv_norm_relus:
        #     feature = layer(fusion)
        # feature = F.relu(self.amodal_deconv(feature), inplace=True)
        # amodal_final =  self.amodal_predictor(feature)
        return [amodal_mask_logits, visible_mask_logits,amodal_refine_logits,amodal_gt_logits], [boundary_logits,boundary_logits_visible],feature_fm
    def forward_through_without_boundary(self, x1, x2):       #Parallel_Amodal_Visible_Head
        # for layer in self.occluder_conv_norm_relus:
        #     occluder = layer(x1)
        # occluder = F.relu(self.occluder_deconv(occluder))
        # occluder_logits = self.occluder_predictor(occluder)
        for layer in self.amodal_conv_norm_relus:
            x1 = layer(x1)
        x1 = F.relu(self.amodal_deconv(x1), inplace=True)
        amodal_mask_logits = self.amodal_predictor(x1)

        for layer in self.visible_conv_norm_relus:
            x2 = layer(x2)
        x2 = F.relu(self.visible_deconv(x2), inplace=True)
        visible_mask_logits = self.visible_predictor(x2)

        return [amodal_mask_logits, visible_mask_logits]
    def save_logits_heatmap(self,logits,instances,perfix="",index=0):
        save_dir='/media/xzt/T7 Shield/heatmap/'
        #save_dir='/home/xzt/Desktop/Amodal-Segmentation-Based-on-Visible-Region-Segmentation-and-Shape-Prior-main/heatmap/'
        for instance_per_image in instances:
            pred_classes=instance_per_image.pred_classes
            for idx in range(len(pred_classes)):
                if pred_classes[idx]!=instance_per_image.gt_classes_inference[idx]:
                    continue
                img_name="{}_{}_{}".format(perfix,index,idx)
                file_dir=save_dir+img_name
                choose_logits=logits[idx,pred_classes[idx]]
                t=choose_logits.cpu()
                t=t.data.numpy()
                t=1/(1+np.exp(-t))
                camp=plt.get_cmap('jet')
                heat_map=camp(t)
                # X = Image.fromarray(heat_map)
                #new_name = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')
                #file_dir+=new_name
                plt.imsave(file_dir+".png",heat_map)

    def save_feature_map(self,features,perfix=""):
        save_dir="/media/xzt/'T7 Shiled'/heatmap/"
        file_name=perfix+"_feature_map"
        new_name = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')
        file_dir=save_dir+file_name+new_name
        #B 256 14 14
        f_num=features.size(1)
        feats=features.cpu().data.numpy()
        feats=feats.squeeze(0)
        row_num=16
        plt.figure()
        for index in range(1,f_num):
            plt.subplot(row_num,row_num,index)
            plt.imshow(feats[index-1],cmap="gray")
        plt.savefig(file_dir)

    def forward_through_evaluate(self,x1,x2,boundary_features,instances):
        features=[]
        B, C, H, W = x1.size()
        for layer in self.visible_conv_norm_relus:
            visible_features = layer(x2)
                # x: B,C,H,W
        # x_query: B,C,HW
        #x_input = AddCoords()(x)
        # if visible_features.shape[0]!= 0:
        #     query_v = self.query_v(visible_features).view(B, C, -1)
        #     # x_query: B,HW,C
        #     query_v = torch.transpose(query_v, 1, 2)
        #     # x_key: B,C,HW
        #     key_v = self.key_v(x2).view(B, C, -1)
        #     # x_value: B,C,HW
        #     value_v = self.value_v(x2).view(B, C, -1)
        #     # x_value: B,HW,C
        #     value_v = torch.transpose(value_v, 1, 2)
        #     # W = Q^T K: B,HW,HW
        #     w_v = torch.matmul(query_v, key_v) * (1.0/128)
        #     w_v = F.softmax(w_v, dim=-1)
        #     # x_relation = WV: B,HW,C
        #     relation_v = torch.matmul(w_v, value_v)
        #     # x_relation = B,C,HW
        #     relation_v = torch.transpose(relation_v, 1, 2)
        #     # x_relation = B,C,H,W
        #     relation_v = relation_v.view(B,C,H,W)

        #     relation_v = self.out_v(relation_v)
        #     relation_v = self.blocker_v(relation_v)
            
        #     visible_features = visible_features + relation_v
        features.append(visible_features)
        boundary_features_visible=boundary_features+self.mask_to_boundary_v(visible_features)
        if boundary_features_visible.shape[0]!= 0:
            query_v = self.query_v(boundary_features_visible).view(B, C, -1)
            # x_query: B,HW,C
            query_v = torch.transpose(query_v, 1, 2)
            # x_key: B,C,HW
            key_v = self.key_v(boundary_features_visible).view(B, C, -1)
            # x_value: B,C,HW
            value_v = self.value_v(boundary_features_visible).view(B, C, -1)
            # x_value: B,HW,C
            value_v = torch.transpose(value_v, 1, 2)
            # W = Q^T K: B,HW,HW
            w_v = torch.matmul(query_v, key_v) * (1.0/128)
            w_v = F.softmax(w_v, dim=-1)
            # x_relation = WV: B,HW,C
            relation_v = torch.matmul(w_v, value_v)
            # x_relation = B,C,HW
            relation_v = torch.transpose(relation_v, 1, 2)
            # x_relation = B,C,H,W
            relation_v = relation_v.view(B,C,H,W)

            relation_v = self.out_v(relation_v)
            relation_v = self.blocker_v(relation_v)
            
            boundary_features_visible = boundary_features_visible + relation_v



        for layer in self.boundary_fcns_visible:
            boundary_features_visible=layer(boundary_features_visible)
        visible_features = self.boundary_to_mask_v(boundary_features_visible)+visible_features
        visible_features = self.mask_final_fusion_v(visible_features)
        features.append(visible_features)
        visible_features = F.relu(self.visible_deconv(visible_features), inplace=True)

        visible_mask_logits = self.visible_predictor(visible_features)
        boundary_features_visible=F.relu(self.boundary_deconv_v(boundary_features_visible))
        boundary_logits_visible = self.boundary_predictor_v(boundary_features_visible)



        for layer in self.amodal_conv_norm_relus:
            amodal_features = layer(x1+features[0])
                # x_query: B,C,HW
        #x_input = AddCoords()(x)
        # if amodal_features.shape[0]!=0:
        #     query_a = self.query_a(amodal_features).view(B, C, -1)
        #     # x_query: B,HW,C
        #     query_a = torch.transpose(query_a, 1, 2)
        #     # x_key: B,C,HW
        #     key_a = self.key_a(x1).view(B, C, -1)
        #     # x_value: B,C,HW
        #     value_a = self.value_a(x1).view(B, C, -1)
        #     # x_value: B,HW,C
        #     value_a = torch.transpose(value_a, 1, 2)
        #     # W = Q^T K: B,HW,HW
        #     w_a = torch.matmul(query_a, key_a) * (1.0/128)
        #     w_a = F.softmax(w_a, dim=-1)
        #     # x_relation = WV: B,HW,C
        #     relation_a = torch.matmul(w_a, value_a)
        #     # x_relation = B,C,HW
        #     relation_a = torch.transpose(relation_a, 1, 2)
        #     # x_relation = B,C,H,W
        #     relation_a = relation_a.view(B,C,H,W)

        #     relation_a = self.out_a(relation_a)
        #     relation_a = self.blocker_a(relation_a)
        #     amodal_features = amodal_features + relation_a
        features.append(amodal_features)
        boundary_features1 = boundary_features + self.mask_to_boundary(amodal_features)
        if boundary_features1.shape[0]!= 0:
            query_v = self.query_v(boundary_features1).view(B, C, -1)
            # x_query: B,HW,C
            query_v = torch.transpose(query_v, 1, 2)
            # x_key: B,C,HW
            key_v = self.key_v(boundary_features1).view(B, C, -1)
            # x_value: B,C,HW
            value_v = self.value_v(boundary_features1).view(B, C, -1)
            # x_value: B,HW,C
            value_v = torch.transpose(value_v, 1, 2)
            # W = Q^T K: B,HW,HW
            w_v = torch.matmul(query_v, key_v) * (1.0/128)
            w_v = F.softmax(w_v, dim=-1)
            # x_relation = WV: B,HW,C
            relation_v = torch.matmul(w_v, value_v)
            # x_relation = B,C,HW
            relation_v = torch.transpose(relation_v, 1, 2)
            # x_relation = B,C,H,W
            relation_v = relation_v.view(B,C,H,W)

            relation_v = self.out_v(relation_v)
            relation_v = self.blocker_v(relation_v)
            
            boundary_features1 = boundary_features1 + relation_v
        for layer in self.boundary_fcns:
            boundary_features1 = layer(boundary_features1)
        boundary_features_amodal=F.relu(self.boundary_deconv(boundary_features1))
        boundary_logits_amodal = self.boundary_predictor(boundary_features_amodal)

        amodal_features = self.boundary_to_mask(boundary_features1) + amodal_features 
        amodal_features = self.mask_final_fusion(amodal_features)
        amodal = F.relu(self.amodal_deconv(amodal_features), inplace=True)
        amodal_mask_logits = self.amodal_predictor(amodal)
        
        if amodal_mask_logits.size(0)!=0:
        # RAB_A= self.RAB1(features[0])
        # RAB_V =self.RAB2(features[1])
        # amodal_refine_features = torch.max(RAB_A,RAB_V)


        # # amodal_fusion_features = self.RAB3(features[1])
        # # ov_fusion_features = self.RAB4(amodal_refine_features)
        # # amodal_final_features = torch.max(ov_fusion_features,amodal_fusion_features)
        # # for layer in self.amodal_conv_norm_relus:
        # #     amodal_refine_features = layer(amodal_refine_features)
        # for layer in self.amodal_conv_norm_relus:
        #     amodal_refine_features =layer(amodal_refine_features)
        
        # amodal_final = self.amodal_deconv(amodal_refine_features)
        # amodal_final = self.amodal_predictor(amodal_final)
        # amodal_final = self.amodal_deconv(amodal_final_features)
        # # amodal_final = self.amodal_predictor(amodal_final)
            # feature_matching=[]
            amodal_attention = self.amodal_pool(classes_choose(amodal_mask_logits, instances)).unsqueeze(1).sigmoid()
            if amodal_attention is not None:
                x1 = amodal_attention * x1+x1
        for layer in self.refine_conv_norm_relus:
            x1 = layer(x1)
            
        amodal_refine = self.r_deconv(x1)
        amodal_refine_logits = self.r_predictor(amodal_refine)

        # self.save_logits_heatmap(boundary_logits_visible,instances,perfix="boundary_visible",index=self.idx)
        # self.save_logits_heatmap(amodal_refine_logits,instances,perfix="refine_amodal",index=self.idx)
        # self.save_logits_heatmap(boundary_logits_amodal,instances,perfix="boudnary_amodal",index=self.idx)
        # self.save_logits_heatmap(amodal_mask_logits,instances,perfix="coarse_amodal",index=self.idx)
        # self.save_logits_heatmap(visible_mask_logits,instances,perfix="visible",index=self.idx)
        # self.idx+=1
        # boundary_features1 = F.relu(self.boundary_deconv(boundary_features1))
        # boundary_logits = self.boundary_predictor(boundary_features1)

        # RAB_A= self.RAB1(features[0])
        # RAB_V =self.RAB2(features[1])
        # amodal_refine_features = torch.max(RAB_A,RAB_V)
        # for layer in self.amodal_conv_norm_relus:
        #     amodal_refine_features =layer(amodal_refine_features)
        # amodal_final = self.amodal_deconv(amodal_refine_features)
        # amodal_final = self.amodal_predictor(amodal_final)
        # amodal_fusion_features = self.RAB3(features[1])
        # ov_fusion_features = self.RAB4(amodal_refine_features)
        # amodal_final_features = torch.max(ov_fusion_features,amodal_fusion_features)
        # for layer in self.amodal_conv_norm_relus:
        #     amodal_refine_features = layer(amodal_refine_features)
        # for layer in self.amodal_conv_norm_relus:
        #     amodal_final_features =layer(amodal_final_features)
        
        # amodal_refine = self.amodal_deconv(amodal_refine_features)
        # amodal_refine = self.amodal_predictor(amodal_refine)
        # amodal_final = self.amodal_deconv(amodal_final_features)
        # amodal_final = self.amodal_predictor(amodal_final)
        # visible_attention = self.amodal_pool(classes_choose(visible_mask_logits, instances)).unsqueeze(1).sigmoid()
        # occluder_attention = self.amodal_pool(classes_choose(occluder_logits, instances)).unsqueeze(1).sigmoid()
        # union_attention, _ = torch.max(torch.cat((visible_attention, occluder_attention), dim=1), dim=1, keepdim=True)
        # fusion = union_attention * x1
        # for layer in self.amodal_conv_norm_relus:
        #     feature = layer(fusion)
        # feature = F.relu(self.amodal_deconv(feature), inplace=True)
        # amodal_final =  self.amodal_predictor(feature)
        return [amodal_mask_logits, visible_mask_logits,amodal_refine_logits]
    def single_head_forward(self, x, head="amodal"):
        features = []
        i = 0
        if head == "amodal":
            for layer in self.amodal_conv_norm_relus:
                x = layer(x)
                if i in self.fm:
                    features.append(x)
                i += 1
            x = F.relu(self.amodal_deconv(x), inplace=True)
            if i in self.fm:
                features.append(x)
            mask_logits = self.amodal_predictor(x)

        elif head == "visible":
            for layer in self.visible_conv_norm_relus:
                x = layer(x)
                if i in self.fm:
                    features.append(x)
                i += 1
            x = F.relu(self.visible_deconv(x), inplace=True)
            if i in self.fm:
                features.append(x)
            mask_logits = self.visible_predictor(x)
        else:
            raise ValueError("Do not have this head")

        return mask_logits, features

    def shape_prior_ref_forward(self, x, refined_visible_logits, shape_prior, instances=None):
        shape_prior = F.avg_pool2d(shape_prior, 2)
        visible_attention = self.amodal_pool(classes_choose(refined_visible_logits, instances)).unsqueeze(1).sigmoid()
        x = self.fuse_layer(cat([x * visible_attention, shape_prior], dim=1))
        amodal_masks_logits, _ = self.single_head_forward(x, "amodal")

        return amodal_masks_logits
def get_gt_masks(instances, mask_side_len, pred_mask_logits):
    amodal_gt_masks = []
    visible_gt_masks = []

    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue

        amodal_gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, mask_side_len
        ).to(device=pred_mask_logits[0].device)
        # num_regions*28*28
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        amodal_gt_masks.append(amodal_gt_masks_per_image)

        visible_gt_masks_per_image = instances_per_image.gt_visible_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, mask_side_len
        ).to(device=pred_mask_logits[0].device)
        # num_regions*28*28
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        visible_gt_masks.append(visible_gt_masks_per_image)
    if len(amodal_gt_masks) == 0:
        return pred_mask_logits[0].sum() * 0
    amodal_gt_masks = cat(amodal_gt_masks, dim=0).unsqueeze(1)
    visible_gt_masks = cat(visible_gt_masks, dim=0).unsqueeze(1)
    #
    # vis.images(amodal_gt_masks, win_name="amodal_gt_masks", nrow=16)
    # vis.images(visible_gt_masks, win_name="visible_gt_masks", nrow=16)
    return amodal_gt_masks, visible_gt_masks

def classes_choose(logits, instances_cls):
    cls_agnostic_mask = logits.size(1) == 1
    total_num_masks = logits.size(0)

    if isinstance(instances_cls, list):
        assert logits.size(0) == sum(len(x) for x in instances_cls)

        classes_label = []
        for instances_per_image in instances_cls:
            if len(instances_per_image) == 0:
                continue
            if not cls_agnostic_mask:
                if instances_per_image.has("gt_classes"):
                    classes_label_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
                elif instances_per_image.has("pred_classes"):
                    classes_label_per_image = instances_per_image.pred_classes.to(dtype=torch.int64)
                else:
                    raise ValueError("classes label missing")
                classes_label.append(classes_label_per_image)
        try:
            classes_label = cat(classes_label, dim=0)
        except:
            pass
    else:
        assert logits.size(0) == instances_cls.size(0)
        classes_label = instances_cls

    if cls_agnostic_mask:
        pred_logits = logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        pred_logits = logits[indices, classes_label]

    return pred_logits



class ResBlock(nn.Module):
    def __init__(self, dim, norm="BN", **kwargs):
        super().__init__()
        num_groups = kwargs.pop("num_group", None)
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            get_norm(norm, dim) if not num_groups else get_norm(norm, dim, num_groups=num_groups),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            get_norm(norm, dim) if not num_groups else get_norm(norm, dim, num_groups=num_groups),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return x + self.block(x)


def build_mask_head(cfg, input_shape):
    """
    Build a mask head defined by `cfg.MODEL.ROI_MASK_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_MASK_HEAD.NAME
    return ROI_MASK_HEAD_REGISTRY.get(name)(cfg, input_shape)

# def build_pre_mask_head(cfg, input_shape):
#     """
#     Build a mask head defined by `cfg.MODEL.ROI_MASK_HEAD.NAME`.
#     """
#     name = cfg.MODEL.ROI_PRE_MASK_HEAD.NAME
#     return ROI_MASK_HEAD_REGISTRY.get(name)(cfg, input_shape)