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

import torch as th
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('/home/tianqiwei/jupyter/rob_IR/')
import datasets
import configs
from utility import utils
import functools as ft
from .miner import miner
import pytest
import itertools as it

def fn_pcontrast_kernel(repA: th.Tensor, repP: th.Tensor, repN: th.Tensor,
                        *, metric: str, margin: float):
    '''
    <functional> the core computation for spc-2 contrastive loss.
    '''
    targets = th.ones(repA.size(0)).to(repA.device)
    if metric == 'C':
        lap = F.cosine_embedding_loss(repA, repP, targets, margin=margin)
        lan = F.cosine_embedding_loss(repA, repN, -targets, margin=margin)
    else:
        __pd = ft.partial(th.nn.functional.pairwise_distance, p=2)
        lap = th.nan_to_num(__pd(repA, repP).mean(), nan=0.0)
        lan = th.nan_to_num(th.masked_select(margin - __pd(repA, repN), lambda x: x > 0).mean(), nan=0.0)
    return lap + lan

def fn_pcontrast(repres: th.Tensor, labels: th.Tensor, *,
                 metric: str, minermethod: str = 'spc2-random', p_switch: float = -1.0):
    '''
    Functional version of contrastive loss function with cosine distance
    as the distance metric. Metric is either 'C' (for cosine) or 'E' for
    euclidean.
    Dataset type should be SPC-2 (according to ICML20 reference)
    '''
    margin = configs.contrastive.margin_cosine if metric in ('C', 'N') else configs.contrastive.margin_euclidean
    if metric in ('C', 'N'):
        repres = th.nn.functional.normalize(repres, p=2, dim=-1)
    ancs, poss, negs = miner(repres, labels, method=minermethod, metric=metric, margin=margin, p_switch=p_switch)
    return fn_pcontrast_kernel(repres[ancs, :], repres[poss, :], repres[negs, :], metric=metric, margin=margin)

class pcontrast(th.nn.Module):
    def __init__(self, metric, minermethod='spc2-random', p_switch=-1.0):
        super().__init__()
        self.metric = metric
        self.minermethod = minermethod
        self.p_switch = p_switch

    def forward(self, *args, **kwargs):
        return fn_pcontrast(*args, metric=self.metric, minermethod=self.minermethod, p_switch=self.p_switch, **kwargs)

    def raw(self, repA, repP, repN):
        margin = configs.contrastive.margin_cosine if self.metric in ('C', 'N') else configs.contrastive.margin_euclidean
        return fn_pcontrast_kernel(repA, repP, repN, metric=self.metric, margin=margin)

class pcontrastC(pcontrast):
    def __init__(self):
        super().__init__('C')

class pcontrastE(pcontrast):
    def __init__(self):
        super().__init__('E')

class pcontrastN(pcontrast):
    def __init__(self):
        super().__init__('N')

class pdcontrastN(pcontrast):
    def __init__(self):
        super().__init__('N', minermethod='spc2-distance')

class pDcontrastN(pcontrast):
    def __init__(self):
        super().__init__('N', minermethod='spc2-distance', p_switch=0.15)

@pytest.mark.parametrize('metric, minermethod',
                         it.product(('C', 'E', 'N'), ('spc2-random', 'spc2-distance')))
def test_pcontrast(metric, minermethod):
    output, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    loss = fn_pcontrast(output, labels, metric=metric, minermethod=minermethod)
    loss.backward()

@pytest.mark.parametrize('metric', 'CEN')
def test_pcontrast_raw(metric: str):
    rA, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    rP = th.rand(10, 32, requires_grad=True)
    rN = th.rand(10, 32, requires_grad=True)
    lossfunc = {'C': pcontrastC, 'N': pcontrastN, 'E': pcontrastE}[metric]()
    if metric in ('C', 'N'):
        _N = ft.partial(F.normalize, dim=-1)
        rA, rP, rN = _N(rA), _N(rP), _N(rN)
    loss = lossfunc.raw(rA, rP, rN)
    loss.backward()
