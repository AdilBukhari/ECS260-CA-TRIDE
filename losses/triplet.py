'''
Copyright (C) 2019-2021, Mo Zhou <cdluminate@gmail.com>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import os
import torch as th
import numpy as np
import math as m
import sys
import datasets
import models
import tensorflow as tf
from models import template_rank
import configs
from utility import utils
import torch.nn as nn
import torch.nn.functional as F
from .miner import miner
import functools as ft
import pytest
import rich

c = rich.get_console()
last_modify_epoch = 0.0

def get_margin(metric):
    """Helper to get margin based on metric type"""
    return configs.triplet.margin_cosine if metric in ('C', 'N') else configs.triplet.margin_euclidean

def fn_ptriplet_kernel(repA: th.Tensor, repP: th.Tensor, repN: th.Tensor, *, metric: str, margin: float):
    '''Core computation for spc-2 triplet loss'''
    global last_modify_epoch
    
    if metric == 'C':
        dap = 1 - F.cosine_similarity(repA, repP, dim=-1)
        dan = 1 - F.cosine_similarity(repA, repN, dim=-1)
        loss = (dap - dan + margin).relu().mean()
    elif metric in ('E', 'N'):
        d_ap = F.pairwise_distance(repA, repP, p=2)
        d_an = F.pairwise_distance(repA, repN, p=2)
        _, ap_top_half_idx = th.topk(d_ap, len(d_ap)//2, largest=False) 
        _, an_top_half_idx = th.topk(d_an, len(d_an)//2, largest=False) 
        ap_top_half = d_ap[ap_top_half_idx].mean()
        an_top_half = d_an[an_top_half_idx].mean()

        loss = F.triplet_margin_loss(repA, repP, repN, margin=0.2)
        if last_modify_epoch == 1.0:
            loss += 1/2 * (ap_top_half - an_top_half + 0.04).relu() 
            last_modify_epoch = 0.0
        else:
            last_modify_epoch = 1.0
    else:
        raise ValueError(f'Illegal metric type {metric}!')
    return loss

def fn_ptriplet(repres: th.Tensor, labels: th.Tensor, *, metric: str, minermethod: str, p_switch: float = -1.0, xa: bool = False):
    '''Variant of triplet loss that accepts [cls=1,cls=1,cls=2,cls=2] batch'''
    margin = get_margin(metric)
    if metric in ('C', 'N'):
        repres = F.normalize(repres, p=2, dim=-1)
        
    anc, pos, neg = miner(repres, labels, template_rank.epoch_n, template_rank.maxepoch, 
                         pnp.Perturbing_method, method=minermethod, metric=metric, margin=margin, p_switch=p_switch)
    
    if xa:
        return fn_ptriplet_kernel(repres[anc, :].detach(), repres[pos, :], repres[neg, :], metric=metric, margin=margin)
    return fn_ptriplet_kernel(repres[anc, :], repres[pos, :], repres[neg, :], metric=metric, margin=margin)

class TripletLoss(th.nn.Module):
    def __init__(self, metric='N', minermethod='spc2-random', xa=False):
        super().__init__()
        self._metric = metric
        self._minermethod = minermethod
        self._xa = xa
        self._datasetspec = 'SPC-2'

    def __call__(self, *args, **kwargs):
        return fn_ptriplet(*args, metric=self._metric, minermethod=self._minermethod, xa=self._xa, **kwargs)

    def determine_metric(self):
        return self._metric

    def datasetspec(self):
        return self._datasetspec

    def raw(self, repA, repP, repN, epoch, max_epoch, *, override_margin: float = None):
        template_rank.epoch_n = epoch
        template_rank.maxepoch = max_epoch
        margin = override_margin if override_margin is not None else get_margin(self._metric)
        return fn_ptriplet_kernel(repA, repP, repN, metric=self._metric, margin=margin)

# Create specific loss instances through factory functions
def create_triplet_loss(metric, minermethod='spc2-random', xa=False):
    return TripletLoss(metric=metric, minermethod=minermethod, xa=xa)

# Test functions
@pytest.mark.parametrize('metric, minermethod', [
    (m, mm) for m in ('C', 'E', 'N') 
    for mm in ('spc2-random', 'spc2-distance', 'spc2-hard', 'spc2-softhard', 'spc2-semihard', 'spc2-ghard')
])
def test_fn_ptriplet(metric: str, minermethod: str):
    output, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    loss = fn_ptriplet(output, labels, metric=metric, minermethod=minermethod)
    loss.backward()

def test_triplet_loss():
    output, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    loss_fn = create_triplet_loss('N')
    loss = loss_fn(output, labels)
    loss.backward()
