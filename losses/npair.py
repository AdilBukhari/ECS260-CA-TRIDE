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
import torch.nn as nn
import functools as ft
import pytest
import configs
from .miner import miner

def fn__pnpair(repres: th.Tensor, labels: th.Tensor, *, metric: str):
    if metric in ('C', 'N'):
        repres = th.nn.functional.normalize(repres, p=2, dim=-1)
    
    anc, pos, neg = miner(repres, labels, method='spc2-npair', metric=metric)
    losses = []
    
    for idx, p, n in zip(anc, pos, neg):
        repA = th.nn.functional.normalize(repres[idx, :].view(-1), dim=-1)
        repP = repres[p, :]
        repN = repres[n, :]
        inner = th.mv(repN - repP, repA)
        losses.append(th.log(1 + th.sum(th.exp(inner))))

    return th.mean(th.stack(losses)) + configs.npair.l2_weight * th.mean(repres.norm(p=2, dim=-1))

class NPairLoss(th.nn.Module):
    _datasetspec = 'SPC-2'
    
    def __init__(self, metric):
        super().__init__()
        self._metric = metric
    
    def __call__(self, *args, **kwargs):
        return ft.partial(fn__pnpair, metric=self._metric)(*args, **kwargs)
    
    def determine_metric(self):
        return self._metric
    
    def datasetspec(self):
        return self._datasetspec

# Metric-specific instances
pnpairC = lambda: NPairLoss('C')
pnpairE = lambda: NPairLoss('E')
pnpairN = lambda: NPairLoss('N')

@pytest.mark.parametrize('metric', ('C', 'E', 'N'))
def test_fn_pnpair(metric):
    output, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    loss = fn__pnpair(output, labels, metric=metric)
    loss.backward()


@pytest.mark.parametrize('func', (pnpairC, pnpairE, pnpairN))
def test_pnpair(func):
    output, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    loss = func()(output, labels)
    loss.backward()
