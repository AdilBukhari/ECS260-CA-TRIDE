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
from .miner import miner
import pytest


def fn__pglift(repres: th.Tensor, labels: th.Tensor, *, metric: str, margin_cosine: float = 0.2, margin_euclidean: float = 1.0, l2_weight: float = 0.0):
    '''
    Generalized lifted-structure loss function
    '''
    # Determine the margin for the specific metric
    if metric in ('C', 'N'):
        margin = margin_cosine
        repres = th.nn.functional.normalize(repres, p=2, dim=-1)
    elif metric in ('E',):
        margin = margin_euclidean
    # Sampling
    anc, pos, neg = miner(repres, labels, method='spc2-lifted', metric=metric)
    # Calculate Loss
    losses = []
    for (i, idx) in enumerate(anc):
        repA = repres[idx, :].view(-1)
        repP = repres[pos[i], :]
        repN = repres[neg[i], :]

        if metric in ('E', 'N'):
            pdist = th.nn.functional.pairwise_distance(repA, repP, p=2)
            ndist = th.nn.functional.pairwise_distance(repA, repN, p=2)
        else:
            pdist = 1 - th.nn.functional.cosine_similarity(repA, repP, dim=-1)
            ndist = 1 - th.nn.functional.cosine_similarity(repA, repN, dim=-1)

        pos_term = pdist.unsqueeze(0).logsumexp(dim=-1)
        neg_term = (margin - ndist).unsqueeze(0).logsumexp(dim=-1)
        losses.append((pos_term + neg_term).relu())
    loss = th.mean(th.stack(losses)) + l2_weight * th.mean(repres.norm(p=2, dim=-1))
    return loss


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
