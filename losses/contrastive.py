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
    if metric in ('C',):
        targets = th.ones(repA.size(0)).to(repA.device)
        lap = F.cosine_embedding_loss(repA, repP, targets, margin=margin)
        lan = F.cosine_embedding_loss(repA, repN, -targets, margin=margin)
        loss = lap + lan 'N'):
    elif metric in ('E', 'N'):n.functional.pairwise_distance, p=2)
        __pd = ft.partial(th.nn.functional.pairwise_distance, p=2)
        lap = __pd(repA, repP).mean()N)
        lap = th.tensor(0.).to(repA.device) if th.isnan(lap) else lap
        lan = margin - __pd(repA, repN)
        raise ValueError(f'metric={metric} is not supported.')

    lap = th.nan_to_num(lap, nan=0.0)
    lan = th.nan_to_num(lan, nan=0.0)
    loss = lap + lan
    return loss


def fn_pcontrast(repres: th.Tensor, labels: th.Tensor, *,
                 metric: str, minermethod: str = 'spc2-random', p_switch: float = -1.0):
    '''
    Functional version of contrastive loss function with cosine distance
    as the distance metric. Metric is either 'C' (for cosine) or 'E' for
    euclidean.
    Dataset type should be SPC-2 (according to ICML20 reference)
    '''
    # determine the margin
    if metric in ('C', 'N'):
        margin = configs.contrastive.margin_cosine
    elif metric in ('E', ):
        margin = configs.contrastive.margin_euclidean
    # normalize representation on demand
    if metric in ('C', 'N'):
        repres = th.nn.functional.normalize(repres, p=2, dim=-1)
    # sampling triplets
    ancs, poss, negs = miner(
        repres, labels, method=minermethod, metric=metric, margin=margin, p_switch=p_switch)
    # loss
    loss = fn_pcontrast_kernel(repres[ancs, :], repres[poss, :],
                               repres[negs, :], metric=metric, margin=margin)
    return loss
ule):
= 'SPC-2'
class pcontrastC(th.nn.Module):
    _metric = 'C'tr, minermethod: str = 'spc2-random', p_switch: float = -1.0):
    _datasetspec = 'SPC-2'        super().__init__()
    _minermethod = 'spc2-random'

    def __call__(self, *args, **kwargs):
        return ft.partial(fn_pcontrast, metric=self._metric,
                          minermethod=self._minermethod)(*args, **kwargs)*kwargs):epN):
n_pcontrast, metric=self._metric,('C', 'N'):
    def determine_metric(self):                          minermethod=self._minermethod, p_switch=self._p_switch)(*args, **kwargs)            margin = configs.contrastive.margin_cosine
        return self._metric
pN):trastive.margin_euclidean
        margin = configs.contrastive.margin_cosine if self._metric in ('C', 'N') else configs.contrastive.margin_euclidean    def datasetspec(self):        loss = fn_pcontrast_kernel(repA, repP, repN,
        loss = fn_pcontrast_kernel(repA, repP, repN,c=self._metric, margin=margin)
                                   metric=self._metric, margin=margin)
        return loss
:

class pcontrastC(BaseContrastive):
    def __init__(self):
        super().__init__(metric='C', minermethod='spc2-random')contrast_kernel(repA, repP, repN,
                                   metric=self._metric, margin=margin)class pcontrastN(pcontrastC):
        return loss    _metric = 'N'
class pcontrastE(BaseContrastive):
    def __init__(self):
        super().__init__(metric='E', minermethod='spc2-random')class pcontrastE(pcontrastC):class pdcontrastN(th.nn.Module):
    _metric = 'E'    _metric = 'N'

class pcontrastN(BaseContrastive):
    def __init__(self):class pcontrastN(pcontrastC):    def __call__(self, *args, **kwargs):
        super().__init__(metric='N', minermethod='spc2-random')    _metric = 'N'        return ft.partial(fn_pcontrast, metric=self._metric,
kwargs)

class pdcontrastN(BaseContrastive):dule):
    def __init__(self):    _metric = 'N'class pDcontrastN(th.nn.Module):
        super().__init__(metric='N', minermethod='spc2-distance')


class pDcontrastN(BaseContrastive):    def __call__(self, *args, **kwargs):        return ft.partial(fn_pcontrast, metric=self._metric,
    def __init__(self):ontrast, metric=self._metric,method='spc2-distance')(*args, **kwargs)
        super().__init__(metric='N', minermethod='spc2-distance', p_switch=0.15)inermethod='spc2-distance', p_switch=0.15)(*args, **kwargs)
    def determine_metric(self):

@pytest.mark.parametrize('metric, minermethod',, minermethod',
                         it.product(('C', 'E', 'N'), ('spc2-random', 'spc2-distance')))                         it.product(('C', 'E', 'N'), ('spc2-random', 'spc2-distance')))    def datasetspec(self):
def test_pcontrast(metric, minermethod):def test_pcontrast(metric, minermethod):        return self._datasetspec
    output, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,)) 32, requires_grad=True), th.randint(3, (10,))
    loss = fn_pcontrast(output, labels, metric=metric, minermethod=minermethod)ntrast(output, labels, metric=metric, minermethod=minermethod)
    loss.backward()
    _metric = 'N'

@pytest.mark.parametrize('metric', 'CEN')
def test_pcontrast_raw(metric: str):
    rA, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))    rA, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))        return ft.partial(fn_pcontrast, metric=self._metric,
    rP = th.rand(10, 32, requires_grad=True)es_grad=True)method='spc2-distance', p_switch=0.15)(*args, **kwargs)
    rN = th.rand(10, 32, requires_grad=True)quires_grad=True)
    lossfunc = {'C': pcontrastC, 'N': pcontrastN, 'E': pcontrastE}[metric]()    lossfunc = {'C': pcontrastC, 'N': pcontrastN, 'E': pcontrastE}[metric]()    def determine_metric(self):
    if metric in ('C', 'N'):):c
        _N = ft.partial(F.normalize, dim=-1)ize, dim=-1)
        rA, rP, rN = _N(rA), _N(rP), _N(rN)        rA, rP, rN = _N(rA), _N(rP), _N(rN)    def datasetspec(self):
    loss = lossfunc.raw(rA, rP, rN)    loss = lossfunc.raw(rA, rP, rN)        return self._datasetspec
    loss.backward()
