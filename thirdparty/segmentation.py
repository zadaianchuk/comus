# torchvision (https://github.com/pytorch/vision)
# Copyright (c) Soumith Chintala 2016.  All rights reserved.
# BSD 3-Clause License.

# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified from
# https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/deeplabv3.py
# to allow different urls for pretrating resnet-50 backbone


import logging

from torch.hub import load_state_dict_from_url
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.resnet import Bottleneck, ResNet
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3
from torchvision.models.segmentation.fcn import FCN, FCNHead

log = logging.getLogger(__name__)


def get_resnet(
    pretrained,
    url="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth",
):
    """Resnet with DINO ResNet-50 pretrained weights"""
    model = ResNet(Bottleneck, [3, 4, 6, 3], replace_stride_with_dilation=[False, True, True])
    if pretrained:
        state_dict = load_state_dict_from_url(url)
        model.load_state_dict(state_dict, strict=False)
    return model


def segmentation_model(name, backbone_name, num_classes, pretrained_backbone):
    """DeepLabv3 segmentation model"""
    assert backbone_name == "resnet50", f"backbone {backbone_name} is not supported as of now"

    backbone = get_resnet(pretrained=pretrained_backbone)
    out_layer = "layer4"
    out_inplanes = 2048
    return_layers = {out_layer: "out"}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model_map = {
        "deeplabv3": (DeepLabHead, DeepLabV3),
        "fcn": (FCNHead, FCN),
    }
    classifier = model_map[name][0](out_inplanes, num_classes)
    base_model = model_map[name][1]

    return base_model(backbone, classifier, None)
