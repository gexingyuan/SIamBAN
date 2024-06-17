# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn.functional as F

# 计算输入张量（x）和相应核张量之间的交叉相关
def xcorr_slow(x, kernel):
    """for loop to calculate cross correlation, slow version
    """
    batch = x.size()[0]     # 获取输入张量 x 的批次大小
    out = []
    for i in range(batch):      # 遍历批次中的每个元素
        px = x[i]           # 获取当前批次元素的输入张量
        pk = kernel[i]      # 获取当前批次元素的核张量
        px = px.view(1, -1, px.size()[1], px.size()[2])
        pk = pk.view(1, -1, pk.size()[1], pk.size()[2])
        po = F.conv2d(px, pk)       # 执行2D卷积操作
        out.append(po)      # 将当前批次的输出结果添加到输出列表中
    out = torch.cat(out, 0)
    return out

# 使用分组卷积来加速计算（是最快且最常用的实现，尤其在大规模数据处理时）
def xcorr_fast(x, kernel):
    """group conv2d to calculate cross correlation, fast version
    """
    batch = kernel.size()[0]
    pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])
    px = x.view(1, -1, x.size()[2], x.size()[3])
    po = F.conv2d(px, pk, groups=batch)     # 输入张量 px 和核张量 pk 将被分成 batch 组，每个组将进行独立的卷积操作
    po = po.view(batch, -1, po.size()[2], po.size()[3])
    return po

# 深度交叉相关操作，分组数量为批次大小乘以通道数，这样每个通道都有自己的核进行卷积操作。（更适用于特定的深度学习模型结构或要求更少参数量的情况）
def xcorr_depthwise(x, kernel):
    """depthwise cross correlation
    """
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch*channel, x.size(2), x.size(3))
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch*channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out
