import torch
import torch.nn.functional as F

import torch
import torch.nn as nn

from changedetection_c.models.Mamba_backbone import Backbone_VSSM
#from classification.models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute

#from classification.models.local_vmamba import VSSM, VSSBlock, Permute

from classification.models.vmamba import LayerNorm2d
from classification.models.spatialmamba import SpatialMamba,Backbone_SpatialMamba

"""
导入swin transformer块
"""
from changedetection_c.models.Mamba_backbone import swin_base_patch4_window7_224


import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

from changedetection_c.models.ChangeDecoder import ChangeDecoder
#from changedetection_c.models.ChangeDecoder_spatial import ChangeDecoder

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count


class STMambaBCD(nn.Module):
    def __init__(self, pretrained, **kwargs):
        super(STMambaBCD, self).__init__()

        self.encoder = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs)
#导入swin transformer块
        #self.encoder = swin_base_patch4_window7_224()

        #self.encoder = Backbone_SpatialMamba(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs)
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        
        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )
 


#kwargs['norm_layer'].lower()：从 kwargs 字典中获取键为 'norm_layer' 的值，并将其转换为小写。这是为了确保在查找字典时不区分大小写。
#NORMLAYERS.get(..., None)：使用 .get() 方法从 _NORMLAYERS 字典中查找对应的规范化层。如果找到了匹配的层，则返回该层；如果没有找到，则返回 None。这样可以避免因键不存在而引发的 KeyError。
        norm_layer: nn.Module = _NORMLAYERS.get(kwargs['norm_layer'].lower(), None)        
        ssm_act_layer: nn.Module = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(kwargs['mlp_act_layer'].lower(), None)

        # Remove the explicitly passed args from kwargs to avoid "got multiple values" error
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['norm_layer', 'ssm_act_layer', 'mlp_act_layer']}

        self.decoder = ChangeDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        self.main_clf = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y# 用二次样条插值上采样

    def forward(self, pre_data, post_data):
        # Encoder processing
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)
        # Decoder processing - passing encoder outputs to the decoder
        output = self.decoder(pre_features, post_features)

        # if output is None:
        #     raise ValueError("Decoder 输出为 None")
        # print("Decoder output shape:", output.shape)  # 应为 [4, 128, H, W]

        output = self.main_clf(output)# 分类操作

        ## 将输出调整为与输入前期数据相同的空间尺寸，使用双线性插值
        output = F.interpolate(output, size=pre_data.size()[-2:], mode='bilinear')
        return output


