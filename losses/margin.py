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
import os
import numpy as np
import functools as ft
from .miner import miner
import sys
sys.path.append('/home/tianqiwei/jupyter/rob_IR/')
import datasets
import configs
from utility import utils
import torch.nn.functional as F
import itertools as it
import pytest

def calculate_distances(repA: th.Tensor, repP: th.Tensor, repN: th.Tensor, metric: str):
    if metric in ('E', 'N'):
        dap = F.pairwise_distance(repA, repP)
        dan = F.pairwise_distance(repA, repN)
    elif metric == 'C':
        dap = 1 - F.cosine_similarity(repA, repP)
        dan = 1 - F.cosine_similarity(repA, repN)
    else:
        raise ValueError("Unsupported metric type")
    return dap, dan

def fn_pmargin_kernel(repA: th.Tensor, repP: th.Tensor, repN: th.Tensor,
                      *, metric: str, margin: float, beta: th.Tensor):
    dap, dan = calculate_distances(repA, repP, repN, metric)
    lap = th.nan_to_num(((dap - beta + margin).relu()).mean(), nan=0.0)
    lan = th.nan_to_num(((beta - dan + margin).relu()).mean(), nan=0.0)
    return lap + lan

def fn_pmargin(repres: th.Tensor, labels: th.Tensor, *,
               beta: float = configs.margin.beta,
               margin: float = configs.margin.margin,
               metric: str, minermethod: str = 'spc2-random'):
    if metric in ('C', 'N'):
        repres = F.normalize(repres, dim=-1)
    ancs, poss, negs = miner(repres, labels, method=minermethod, metric=metric)
    return fn_pmargin_kernel(repres[ancs], repres[poss], repres[negs],
                            metric=metric, margin=margin, beta=beta)

class MarginBase(th.nn.Module):
    _margin: float = configs.margin.margin
    _minermethod = 'spc2-random'
    _optim_lr = configs.margin.lr_beta

    def __init__(self, metric):
        super().__init__()
        self._metric = metric
        self.beta = th.nn.Parameter(th.tensor(configs.margin.beta))

    def raw(self, repA, repP, repN):
        return fn_pmargin_kernel(repA, repP, repN, metric=self._metric,
                               margin=self._margin, beta=self.beta)

    def forward(self, *args, **kwargs):
        if int(os.getenv('DEBUG', -1)) > 0:
            print('* margin: current beta = ', self.beta.data)
        return fn_pmargin(*args, metric=self._metric,
                         minermethod=self._minermethod,
                         beta=self.beta, margin=self._margin, **kwargs)

    def determine_metric(self): return self._metric
    def datasetspec(self): return 'SPC-2'
    def getOptim(self): return th.optim.SGD(self.parameters(), lr=self._optim_lr)

class pmarginC(MarginBase):
    def __init__(self): super().__init__('C')
class pmarginE(MarginBase):
    def __init__(self): super().__init__('E')
class pmarginN(MarginBase):
    def __init__(self): super().__init__('N')
class pdmarginN(pmarginN): _minermethod = 'spc2-distance'

# Metric classes dictionary for dynamic instantiation
MARGIN_CLASSES = {
    'C': pmarginC,
    'E': pmarginE,
    'N': pmarginN,
}

@pytest.mark.parametrize('metric, minermethod', it.product(('C', 'E', 'N'),
                                                           ('spc2-random', 'spc2-distance')))
def test_fn_pmargin(metric, minermethod):
    output, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    loss = fn_pmargin(output, labels, metric=metric, minermethod=minermethod)
    loss.backward()


@pytest.mark.parametrize('func', (pmarginC, pmarginE, pmarginN, pdmarginN))
def test_pmargin(func):
    output, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    loss = func()(output, labels)
    loss.backward()


@pytest.mark.parametrize('func', (pmarginC, pmarginE, pmarginN, pdmarginN))
def test_pmargin_raw(func: object):
    rA = th.rand(10, 32, requires_grad=True)
    rP = th.rand(10, 32, requires_grad=True)
    rN = th.rand(10, 32, requires_grad=True)
    if func._metric in ('C', 'N'):
        _N = ft.partial(F.normalize, dim=-1)
        rA, rP, rN = _N(rA), _N(rP), _N(rN)
    loss = func().raw(rA, rP, rN)
    loss.backward()
