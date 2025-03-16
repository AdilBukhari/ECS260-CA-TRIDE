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
import sys
sys.path.append('/home/tianqiwei/jupyter/rob_IR/')
import datasets
import configs
from utility import utils
import torch
import torch.nn as nn
from torch.autograd import Variable
from .miner import miner
import functools as ft
import itertools as it
import pytest


def _get_metric_config(metric):
    """Helper function to get metric-specific configuration"""
    if metric in ('C', 'N'):
        return configs.glift.margin_cosine, True
    return configs.glift.margin_euclidean, False

def _calculate_distance(p, n, normalize=False):
    """Helper function to calculate distance based on normalization"""
    if normalize:
        return 1 - th.nn.functional.cosine_similarity(p, n, dim=-1)
    return th.nn.functional.pairwise_distance(p, n, p=2)

def fn__pglift(repres: th.Tensor, labels: th.Tensor, *, metric: str):
    margin, should_normalize = _get_metric_config(metric)
    
    if should_normalize:
        repres = th.nn.functional.normalize(repres, p=2, dim=-1)
    
    anc, pos, neg = miner(repres, labels, method='spc2-lifted', metric=metric)
    
    losses = []
    for idx, p_idx, n_idx in zip(anc, pos, neg):
        repA = repres[idx].view(-1)
        repP = repres[p_idx]
        repN = repres[n_idx]
        
        pos_dist = _calculate_distance(repA, repP, should_normalize)
        neg_dist = _calculate_distance(repA, repN, should_normalize)
        
        loss = th.logsumexp(pos_dist, dim=-1) + th.logsumexp(margin - neg_dist, dim=-1)
        losses.append(loss.relu())
    
    return th.mean(th.stack(losses)) + configs.glift.l2_weight * th.mean(repres.norm(p=2, dim=-1))

class pglift(th.nn.Module):
    _datasetspec = 'SPC-2'
    _metric = None
    
    def __call__(self, *args, **kwargs):
        return fn__pglift(*args, metric=self._metric, **kwargs)
    
    def determine_metric(self):
        return self._metric
    
    def datasetspec(self):
        return self._datasetspec

class pgliftC(pglift):
    _metric = 'C'

class pgliftE(pglift):
    _metric = 'E'

class pgliftN(pglift):
    _metric = 'N'

def test_fn_glift():
    output = th.rand(10, 32, requires_grad=True)
    labels = th.randint(3, (10,))
    for metric in ('C', 'E', 'N'):
        loss = fn__pglift(output, labels, metric=metric)
        loss.backward()

def test_glift():
    output = th.rand(10, 32, requires_grad=True)
    labels = th.randint(3, (10,))
    for func in (pgliftC, pgliftE, pgliftN):
        loss = func()(output, labels)
        loss.backward()
