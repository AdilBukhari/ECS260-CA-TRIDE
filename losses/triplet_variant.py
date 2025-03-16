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
import torch.nn.functional as F
import rich
c = rich.get_console()

# Helper function to determine margin and normalize if needed
def get_margin_and_normalize(metric, repres):
    if metric in ('C', 'N'):
        margin = configs.triplet.margin_cosine
        margin2 = configs.quadruplet.margin2_cosine if 'quadruplet' in configs else None
        repres = F.normalize(repres, p=2, dim=-1)
    elif metric == 'E':
        margin = configs.triplet.margin_euclidean
        margin2 = configs.quadruplet.margin2_euclidean if 'quadruplet' in configs else None
    else:
        raise ValueError(f'Illegal metric type {metric}!')
    return margin, margin2, repres

def fn_pquad(repres: th.Tensor, labels: th.Tensor, *, metric: str,
             minermethod: str, p_switch: float = -1.0):
    '''
    Quadruplet Loss Function
    '''
    margin, margin2, repres = get_margin_and_normalize(metric, repres)
    anc, pos, neg = miner(repres, labels, method=minermethod,
                          metric=metric, margin=margin, p_switch=p_switch)
    mask2 = th.logical_and(neg != neg.view(-1, 1),
                           labels.view(-1)[neg] != labels.view(-1)[neg].view(-1, 1))
    neg2 = [np.random.choice(th.where(mask)[0].cpu()) if any(th.where(mask)[0])
            else np.random.choice(repres.size(0)) for mask in mask2]
    neg2 = th.tensor(neg2).to(repres.device)
    __cos = ft.partial(F.cosine_similarity, dim=-1)
    __euc = ft.partial(F.pairwise_distance, p=2)
    if metric == 'C':
        dap = 1 - __cos(repres[anc, :], repres[pos, :])
        dan = 1 - __cos(repres[anc, :], repres[neg, :])
        tloss = (dap - dan + margin).clamp(min=0.).mean()
        dnn = 1 - __cos(repres[neg, :], repres[neg2, :])
    else:
        dap = __euc(repres[anc, :], repres[pos, :])
        dan = __euc(repres[anc, :], repres[neg, :])
        tloss = (dap - dan + margin).relu().mean()
        dnn = __euc(repres[neg, :], repres[neg2, :])
    qloss = (dap - dnn + margin2).relu().mean()
    return tloss + qloss

@pytest.mark.parametrize('metric, minermethod', it.product(('C', 'E', 'N'),
                                                           ('spc2-random', 'spc2-distance', 'spc2-hard', 'spc2-softhard', 'spc2-semihard')))
def test_fn_pquad(metric, minermethod):
    output, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    loss = fn_pquad(output, labels, metric=metric, minermethod=minermethod)
    loss.backward()

class pquad(th.nn.Module):
    _datasetspec = 'SPC-2'
    _minermethod = 'spc2-random'

    def __call__(self, *args, **kwargs):
        return ft.partial(fn_pquad, metric=self._metric,
                          minermethod=self._minermethod)(*args, **kwargs)

    def determine_metric(self):
        return self._metric

    def datasetspec(self):
        return self._datasetspec

class pquadC(pquad):
    _metric = 'C'

class pquadE(pquad):
    _metric = 'E'

class pquadN(pquad):
    _metric = 'N'

class pdquadN(pquad):
    _metric = 'N'
    _minermethod = 'spc2-distance'

@pytest.mark.parametrize('func', (pquadC, pquadE, pquadN, pdquadN))
def test_pquad(func):
    output, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    loss = func()(output, labels)
    loss.backward()

def fn_rhomboid(repres: th.Tensor, labels: th.Tensor, *,
                metric: str, minermethod: str, p_switch: float = -1.0):
    '''
    my private rhomboid loss implementation (for SPC-2 batch)
    '''
    margin, _, repres = get_margin_and_normalize(metric, repres)
    anc, pos, neg = miner(repres, labels, method=minermethod,
                          metric=metric, margin=margin, p_switch=p_switch)
    ne2 = (neg - th.sign((neg % 2) - 0.5)).long()
    if metric == 'C':
        __dist = lambda x, y: 1 - F.cosine_similarity(x, y)
    else:
        __dist = F.pairwise_distance
    dap = __dist(repres[anc, :], repres[pos, :])
    dan = __dist(repres[anc, :], repres[neg, :])
    loss = (dap - dan + margin).relu().mean()
    xdap = __dist(repres[neg, :], repres[ne2, :])
    xdan = __dist(repres[neg, :], repres[anc, :])
    xloss = (xdap - xdan + margin).relu().mean()
    return loss + xloss

@pytest.mark.parametrize('metric, minermethod', it.product(('C', 'E', 'N'),
                                                           ('spc2-random', 'spc2-distance', 'spc2-hard', 'spc2-softhard', 'spc2-semihard')))
def test_fn_rhomboid(metric, minermethod):
    output, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    loss = fn_rhomboid(output, labels, metric=metric, minermethod=minermethod)
    loss.backward()

class prhom(th.nn.Module):
    _datasetspec = 'SPC-2'
    _minermethod = 'spc2-random'

    def __call__(self, *args, **kwargs):
        return ft.partial(fn_rhomboid, metric=self._metric,
                          minermethod=self._minermethod)(*args, **kwargs)

    def determine_metric(self):
        return self._metric

    def datasetspec(self):
        return self._datasetspec

class prhomC(prhom):
    _metric = 'C'

class prhomE(prhom):
    _metric = 'E'

class prhomN(prhom):
    _metric = 'N'

class pdrhomN(prhom):
    _metric = 'N'
    _minermethod = 'spc2-distance'

@pytest.mark.parametrize('func', (prhomC, prhomE, prhomN, pdrhomN))
def test_prhom(func):
    output, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    loss = func()(output, labels)
    loss.backward()

def fn_pgil(repres: th.Tensor, labels: th.Tensor,
            *, metric: str, minermethod: str):
    '''
    GIL for Deep Metric Learning
    '''
    anc, pos, neg = miner(repres, labels, method=minermethod,
                          metric=metric)
    if metric in ('C', 'N'):
        repres = F.normalize(repres, p=2)
    rA, rP, rN = repres[anc, :], repres[pos, :], repres[neg, :]
    if metric == 'C':
        margin = configs.triplet.margin_cosine
        dap = 1 - F.cosine_similarity(rA, rP, dim=-1)
        dan = 1 - F.cosine_similarity(rA, rN, dim=-1)
        dpn = 1 - F.cosine_similarity(rP, rN, dim=-1)
    else:
        margin = configs.triplet.margin_euclidean
        dap = F.pairwise_distance(rA, rP, p=2)
        dan = F.pairwise_distance(rA, rN, p=2)
        dpn = F.pairwise_distance(rP, rN, p=2)
    l1 = (th.mul(rP, rP / 2 - rA).sum(-1) + 1.0) * dap.detach()
    l2 = (th.mul(rN, rA - rN / 2).sum(-1) + 1.0) / (dan**3).detach()
    loss = th.cat([l1, l2]).mean()
    return loss

class pgil(th.nn.Module):
    _datasetspec = 'SPC-2'
    _minermethod = 'spc2-random'

    def __call__(self, *args, **kwargs):
        if hasattr(self, '_minermethod'):
            return ft.partial(fn_pgil, metric=self._metric,
                              minermethod=self._minermethod)(*args, **kwargs)
        else:
            return ft.partial(fn_pgil, metric=self._metric)(
                *args, **kwargs)

    def determine_metric(self):
        return self._metric

    def datasetspec(self):
        return self._datasetspec

class pgilC(pgil):
    _metric = 'C'

class pgilE(pgil):
    _metric = 'E'

class pgilN(pgil):
    _metric = 'N'

@pytest.mark.parametrize('func', (pgilC, pgilE, pgilN))
def test_pgil(func):
    output, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    loss = func()(output, labels)
    loss.backward()
