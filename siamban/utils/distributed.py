# Copyright (c) SenseTime. All Rights Reserved.
# 处理分布式训练相关的功能
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import socket
import logging

import torch
import torch.nn as nn
import torch.distributed as dist

from pysot.utils.log_helper import log_once

logger = logging.getLogger('global')


def average_reduce(v):
    if get_world_size() == 1:
        return v
    tensor = torch.cuda.FloatTensor(1)
    tensor[0] = v
    dist.all_reduce(tensor)
    v = tensor[0] / get_world_size()
    return v


class DistModule(nn.Module):
    def __init__(self, module, bn_method=0):
        super(DistModule, self).__init__()
        self.module = module
        self.bn_method = bn_method
        if get_world_size() > 1:
            broadcast_params(self.module)
        else:
            self.bn_method = 0  # single proccess

    def forward(self, *args, **kwargs):
        broadcast_buffers(self.module, self.bn_method)
        return self.module(*args, **kwargs)

    def train(self, mode=True):
        super(DistModule, self).train(mode)
        self.module.train(mode)
        return self


def broadcast_params(model):
    """ broadcast model parameters """
    for p in model.state_dict().values():
        dist.broadcast(p, 0)


def broadcast_buffers(model, method=0):
    """ broadcast model buffers """
    if method == 0:
        return

    world_size = get_world_size()

    for b in model._all_buffers():
        if method == 1:  # broadcast from main proccess
            dist.broadcast(b, 0)
        elif method == 2:  # average
            dist.all_reduce(b)
            b /= world_size
        else:
            raise Exception('Invalid buffer broadcast code {}'.format(method))


inited = False

# 初始化分布式训练，确保每个进程关联到正确的 GPU 设备，并且初始化了分布式训练的进程组
def _dist_init():
    '''
    if guess right:
        ntasks: world_size (process num)
        proc_id: rank
    '''
    rank = int(os.environ['RANK'])          # 表示当前进程在整个分布式训练中的排名
    num_gpus = torch.cuda.device_count()    # 总GPU数
    torch.cuda.set_device(rank % num_gpus)  # 将当前进程关联到对应的 GPU 设备
    dist.init_process_group(backend='nccl') #  初始化分布式训练的进程组，使用的后端是 'nccl'，即 NVIDIA NCCL 库。
    world_size = dist.get_world_size()      # 总进程数
    return rank, world_size


def _get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip


# 分布式训练
def dist_init():
    global rank, world_size, inited
    if torch.cuda.device_count() == 1:      # 如果只有一个 GPU，则将 rank 设置为 0，world_size 设置为 1。
        rank, world_size = 0, 1
    else:         # 如果有多个GPU，则尝试调用_dist_init() 函数来进行真正的分布式初始化，获取当前进程的 rank 和总进程数 world_size
        try:
            rank, world_size = _dist_init()
        except RuntimeError as e:
            if 'public' in e.args[0]:
                logger.info(e)
                logger.info('Warning: use single process')
                rank, world_size = 0, 1
            else:
                raise RuntimeError(*e.args)
    inited = True
    return rank, world_size


def get_rank():
    if not inited:
        raise(Exception('dist not inited'))
    return rank


def get_world_size():
    if not inited:
        raise(Exception('dist not inited'))
    return world_size


def reduce_gradients(model, _type='sum'):
    types = ['sum', 'avg']
    assert _type in types, 'gradients method must be in "{}"'.format(types)
    log_once("gradients method is {}".format(_type))
    if get_world_size() > 1:
        for param in model.parameters():
            if param.requires_grad:
                dist.all_reduce(param.grad.data)
                if _type == 'avg':
                    param.grad.data /= get_world_size()
    else:
        return None
