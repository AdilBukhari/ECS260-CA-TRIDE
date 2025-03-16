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

def get_margin(metric):
    """Helper function to determine margin based on metric type"""
    return configs.contrastive.margin_cosine if metric in ('C', 'N') else configs.contrastive.margin_euclidean

def fn_pcontrast_kernel(repA: th.Tensor, repP: th.Tensor, repN: th.Tensor, *, metric: str, margin: float):
    '''<functional> the core computation for spc-2 contrastive loss.'''
    if metric == 'C':
        targets = th.ones(repA.size(0)).to(repA.device)
        loss = F.cosine_embedding_loss(repA, repP, targets, margin=margin) + \
               F.cosine_embedding_loss(repA, repN, -targets, margin=margin)
    else:  # metric in ('E', 'N')
        pd = ft.partial(F.pairwise_distance, p=2)
        lap = pd(repA, repP).mean()
        lap = th.tensor(0.).to(repA.device) if th.isnan(lap) else lap
        
        lan = margin - pd(repA, repN)
        lan = th.masked_select(lan, lan > 0.).mean()
        lan = th.tensor(0.).to(repA.device) if th.isnan(lan) else lan
        
        loss = lap + lan
    return loss

def fn_pcontrast(repres: th.Tensor, labels: th.Tensor, *, metric: str, minermethod: str = 'spc2-random', p_switch: float = -1.0):
    '''Functional version of contrastive loss function'''
    margin = get_margin(metric)
    if metric in ('C', 'N'):
        repres = F.normalize(repres, p=2, dim=-1)
    
    ancs, poss, negs = miner(repres, labels, method=minermethod, metric=metric, margin=margin, p_switch=p_switch)
    return fn_pcontrast_kernel(repres[ancs, :], repres[poss, :], repres[negs, :], metric=metric, margin=margin)

class BaseContrast(th.nn.Module):
    """Base class for all contrast losses"""
    _datasetspec = 'SPC-2'
    
    def __init__(self, metric, minermethod='spc2-random', p_switch=-1.0):
        super().__init__()
        self._metric = metric
        self._minermethod = minermethod
        self._p_switch = p_switch

    def __call__(self, *args, **kwargs):
        return fn_pcontrast(*args, metric=self._metric, minermethod=self._minermethod, p_switch=self._p_switch, **kwargs)

    def determine_metric(self):
        return self._metric

    def datasetspec(self):
        return self._datasetspec

    def raw(self, repA, repP, repN):
        if self._metric in ('C', 'N'):
            repA, repP, repN = map(lambda x: F.normalize(x, dim=-1), (repA, repP, repN))
        return fn_pcontrast_kernel(repA, repP, repN, metric=self._metric, margin=get_margin(self._metric))

# Simplified class definitions using inheritance
class pcontrastC(BaseContrast):
    def __init__(self):
        super().__init__('C')

class pcontrastE(BaseContrast):
    def __init__(self):
        super().__init__('E')

class pcontrastN(BaseContrast):
    def __init__(self):
        super().__init__('N')

class pdcontrastN(BaseContrast):
    def __init__(self):
        super().__init__('N', 'spc2-distance')

class pDcontrastN(BaseContrast):
    def __init__(self):
        super().__init__('N', 'spc2-distance', 0.15)

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
