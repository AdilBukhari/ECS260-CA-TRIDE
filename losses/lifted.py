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


def fn__pglift(repres: th.Tensor, labels: th.Tensor, *, metric: str):
    '''
    Generalized lifted-structure loss function
    '''
    # Determine the margin for the specific metric
    if metric in ('C', 'N'):
        margin = configs.glift.margin_cosine
        repres = th.nn.functional.normalize(repres, p=2, dim=-1)
    elif metric == 'E':
        margin = configs.glift.margin_euclidean

    # Sampling
    anc, pos, neg = miner(repres, labels, method='spc2-lifted', metric=metric)

    # Calculate Loss
    def __pdist(p, n):
        if metric in ('E', 'N'):
            return th.nn.functional.pairwise_distance(p, n, p=2)
        return 1 - th.nn.functional.cosine_similarity(p, n, dim=-1)

    losses = [
        (th.logsumexp(__pdist(repres[idx, :].view(-1), repres[pos[i], :]), dim=-1) +
         th.logsumexp(margin - __pdist(repres[idx, :].view(-1), repres[neg[i], :]), dim=-1)).relu()
        for i, idx in enumerate(anc)
    ]

    loss = th.mean(th.stack(losses)) + configs.glift.l2_weight * th.mean(repres.norm(p=2, dim=-1))
    return loss

class pglift(th.nn.Module):
    _datasetspec = 'SPC-2'

    def __init__(self, metric):
        super().__init__()
        self._metric = metric

    def forward(self, *args, **kwargs):
        return fn__pglift(*args, metric=self._metric, **kwargs)

    def determine_metric(self):
        return self._metric

    def datasetspec(self):
        return self._datasetspec

class pgliftC(pglift):
    def __init__(self):
        super().__init__('C')

class pgliftE(pglift):
    def __init__(self):
        super().__init__('E')

class pgliftN(pglift):
    def __init__(self):
        super().__init__('N')

@pytest.mark.parametrize('metric', ('C', 'E', 'N'))
def test_fn_glift(metric):
    output, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    loss = fn__pglift(output, labels, metric=metric)
    loss.backward()

@pytest.mark.parametrize('func', (pgliftC, pgliftE, pgliftN))
def test_glift(func):
    output, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    loss = func()(output, labels)
    loss.backward()
