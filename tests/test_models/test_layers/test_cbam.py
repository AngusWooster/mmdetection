# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
import torch.nn.functional as F
from mmengine.model import constant_init

from mmdet.models.layers import CBAM


def test_cbam():

    cbam = CBAM(channels=32)
    x = torch.randn((2, 32, 10, 10))
    x_out = cbam(x)

    print(f"out: {x_out.shape}")
    assert x_out.shape == torch.Size((2, 32, 10, 10))

test_cbam()