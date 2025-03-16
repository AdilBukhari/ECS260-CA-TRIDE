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


def get_metric_params(metric):
    """Helper function to get metric-specific parameters"""
    if metric in ('C', 'N'):
        margin = configs.triplet.margin_cosine
        margin2 = configs.quadruplet.margin2_cosine
        use_cosine = True
    else:  # metric == 'E'
        margin = configs.triplet.margin_euclidean
        margin2 = configs.quadruplet.margin2_euclidean
        use_cosine = False
    return margin, margin2, use_cosine

def calculate_distance(x, y, use_cosine=True):
    """Helper function to calculate distance between tensors"""
    if use_cosine:
        return 1 - F.cosine_similarity(x, y, dim=-1)
    return F.pairwise_distance(x, y, p=2)

def fn_pquad(repres: th.Tensor, labels: th.Tensor, *, metric: str,
             minermethod: str, p_switch: float = -1.0):
    margin, margin2, use_cosine = get_metric_params(metric)
    if use_cosine:
        repres = F.normalize(repres, p=2, dim=-1)
    
    anc, pos, neg = miner(repres, labels, method=minermethod,
                         metric=metric, margin=margin, p_switch=p_switch)
    
    # Sample second negative
    mask2 = th.logical_and(neg != neg.view(-1, 1),
                          labels.view(-1)[neg] != labels.view(-1)[neg].view(-1, 1))
    neg2 = th.tensor([np.random.choice(th.where(mask)[0].cpu() if any(th.where(mask)[0])
                      else range(repres.size(0))) for mask in mask2]).to(repres.device)
    
    # Calculate distances
    dap = calculate_distance(repres[anc], repres[pos], use_cosine)
    dan = calculate_distance(repres[anc], repres[neg], use_cosine)
    dnn = calculate_distance(repres[neg], repres[neg2], use_cosine)
    
    tloss = (dap - dan + margin).clamp(min=0.).mean()
    qloss = (dap - dnn + margin2).relu().mean()
    
    return tloss + qloss


@pytest.mark.parametrize('metric, minermethod', it.product(('C', 'E', 'N'),
                                                           ('spc2-random', 'spc2-distance', 'spc2-hard', 'spc2-softhard', 'spc2-semihard')))
def test_fn_pquad(metric, minermethod):
    output, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    loss = fn_pquad(output, labels, metric=metric, minermethod=minermethod)
    loss.backward()


class MetricLoss(th.nn.Module):
    """Base class for metric losses"""
    _datasetspec = 'SPC-2'
    _minermethod = 'spc2-random'
    
    def __call__(self, *args, **kwargs):
        return ft.partial(self._loss_fn, metric=self._metric,
                         minermethod=self._minermethod)(*args, **kwargs)
    
    def determine_metric(self): return self._metric
    def datasetspec(self): return self._datasetspec

class pquad(MetricLoss):
    _loss_fn = staticmethod(fn_pquad)

class pquadC(pquad): _metric = 'C'
class pquadE(pquad): _metric = 'E'
class pquadN(pquad): _metric = 'N'
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
    margin, _, use_cosine = get_metric_params(metric)
    if use_cosine:
        repres = F.normalize(repres, p=2, dim=-1)
    
    anc, pos, neg = miner(repres, labels, method=minermethod,
                         metric=metric, margin=margin, p_switch=p_switch)
    ne2 = (neg - th.sign((neg % 2) - 0.5)).long()
    
    dap = calculate_distance(repres[anc], repres[pos], use_cosine)
    dan = calculate_distance(repres[anc], repres[neg], use_cosine)
    xdap = calculate_distance(repres[neg], repres[ne2], use_cosine)
    xdan = calculate_distance(repres[neg], repres[anc], use_cosine)
    
    loss = (dap - dan + margin).relu().mean()
    xloss = (xdap - xdan + margin).relu().mean()
    
    return loss + xloss


@pytest.mark.parametrize('metric, minermethod', it.product(('C', 'E', 'N'),
                                                           ('spc2-random', 'spc2-distance', 'spc2-hard', 'spc2-softhard', 'spc2-semihard')))
def test_fn_rhomboid(metric, minermethod):
    output, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    loss = fn_rhomboid(output, labels, metric=metric, minermethod=minermethod)
    loss.backward()


class prhom(MetricLoss):
    _loss_fn = staticmethod(fn_rhomboid)

class prhomC(prhom): _metric = 'C'
class prhomE(prhom): _metric = 'E'
class prhomN(prhom): _metric = 'N'
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
    # sample the triplets
    anc, pos, neg = miner(repres, labels, method=minermethod,
                          metric=metric)
    # normalize
    if metric in ('C', 'N'):
        repres = F.normalize(repres, p=2)
    # loss function
    rA, rP, rN = repres[anc, :], repres[pos, :], repres[neg, :]
    if metric == 'C':
        margin = configs.triplet.margin_cosine
        dap = 1 - F.cosine_similarity(rA, rP, dim=-1)
        dan = 1 - F.cosine_similarity(rA, rN, dim=-1)
        dpn = 1 - F.cosine_similarity(rP, rN, dim=-1)
    elif metric in ('E', 'N'):
        margin = configs.triplet.margin_euclidean
        dap = F.pairwise_distance(rA, rP, p=2)
        dan = F.pairwise_distance(rA, rN, p=2)
        dpn = F.pairwise_distance(rP, rN, p=2)
    else:
        raise NotImplementedError
    if metric == 'N':
        margin = configs.triplet.margin_cosine
    # [method 1: move anchor]
    #mask_repulse = (dap > dan).view(-1)
    ##loss_repulse = ((repres[anc, :] * (repres[anc, :] - repres[neg, :])).sum(-1) + 1.0) / (dan ** 2)
    # loss_repulse = ((repres[anc, :] * (repres[neg, :] - repres[anc, :])).sum(-1) + 1.0) #/ (dan ** 2)
    #mask_attract = (dap <= dan).view(-1)
    ##loss_attract = ((repres[anc, :] * (repres[pos, :] - repres[anc, :])).sum(-1) + 1.0) / (dap ** 2)
    # loss_attract = ((repres[anc, :] * (repres[anc, :] - repres[pos, :])).sum(-1) + 1.0) #/ (dap ** 2)
    #lrep = th.masked_select(loss_repulse, mask_repulse)
    #latt = th.masked_select(loss_attract, mask_attract)
    #loss = th.cat([lrep, latt]).mean()
    # [method 2: move pos and neg] : working
    #loss_attract = (th.mul(repres[pos, :], repres[pos, :] - repres[anc, :]).sum(-1) + 1.0) * (dap ** 2)
    #loss_repulse = (th.mul(repres[neg, :], repres[anc, :] - repres[neg, :]).sum(-1) + 1.0) / (dan ** 2)
    #loss = th.cat([loss_attract, loss_repulse]).mean()
    # [method 3: ring all]
    # loss = th.cat([
    #    # anchor: attract and repulse
    #    (th.mul(repres[anc, :], repres[pos, :] - repres[anc, :]).sum(-1) + 1.0) * (dap ** 2),
    #    #(th.mul(repres[anc, :], repres[anc, :] - repres[neg, :]).sum(-1) + 1.0) / (dan ** 2),
    #    # positive: attract and repulse
    #    (th.mul(repres[pos, :], repres[anc, :] - repres[pos, :]).sum(-1) + 1.0) * (dap ** 2),
    #    #(th.mul(repres[pos, :], repres[pos, :] - repres[neg, :]).sum(-1) + 1.0) / (dpn ** 2),
    #    # negative: repulse
    #    (th.mul(repres[neg, :], repres[neg, :] - repres[anc, :]).sum(-1) + 1.0) / (dan ** 2),
    #    (th.mul(repres[neg, :], repres[neg, :] - repres[pos, :]).sum(-1) + 1.0) / (dpn ** 2),
    #    ]).mean()
    # [method 4: all / no weight]
    # loss = th.cat([
    #    (th.mul(rA, rN - rP).sum(-1) + 1.0),
    #    (th.mul(rP, rN - rA).sum(-1) + 1.0),
    #    (th.mul(rN, rA + rP - rN).sum(-1) + 1.0),
    #    ]).mean()
    # [method 5: all / has weight]
    # loss = th.cat([
    #    (th.mul(rA, rA - rP).sum(-1) + 1.0) * (dap ** 2),
    #    (th.mul(rA, rN - rA).sum(-1) + 1.0) / (dan ** 2),
    #    (th.mul(rP, rP - rA).sum(-1) + 1.0) * (dap ** 2),
    #    (th.mul(rP, rN - rP).sum(-1) + 1.0) / (dpn ** 2),
    #    (th.mul(rN, rA - rN/2.).sum(-1) + 1.0) / (dan ** 2),
    #    (th.mul(rN, rP - rN/2.).sum(-1) + 1.0) / (dpn ** 2),
    #    ]).mean()
    # [static weight + triplet mask]
    # loss = th.stack([
    #    (th.mul(rA, rA/2. - rP).sum(-1) + 1.0) * (dap/dan),
    #    (th.mul(rA, rN - rA/2.).sum(-1) + 1.0) / (dap/dan),
    #    (th.mul(rP, rP/2. - rA).sum(-1) + 1.0) * (dap/dan),
    #    (th.mul(rP, rN - rP/2.).sum(-1) + 1.0) / (dap/dpn),
    #    (th.mul(rN, rA - rN/2.).sum(-1) + 1.0) / (dap/dan),
    #    (th.mul(rN, rP - rN/2.).sum(-1) + 1.0) / (dap/dpn),
    #    ]).mean(0)
    #mask = (dap - dan + margin >= 0.).view(-1)
    #loss = th.masked_select(loss, mask).mean()
    # [static weight + pair mask]
    # loss = th.cat([
    #    th.masked_select((th.mul(rA, rA - rP).sum(-1) + 1.0) * (dap ** 2).detach(), dap > margin),
    #    th.masked_select((th.mul(rA, rN - rA).sum(-1) + 1.0) / (dan ** 2).detach(), dan < margin),
    #    th.masked_select((th.mul(rP, rP - rA).sum(-1) + 1.0) * (dap ** 2).detach(), dap > margin),
    #    th.masked_select((th.mul(rP, rN - rP).sum(-1) + 1.0) / (dpn ** 2).detach(), dpn < margin),
    #    th.masked_select((th.mul(rN, rA - rN/2.).sum(-1) + 1.0) / (dan ** 2).detach(), dan < margin),
    #    th.masked_select((th.mul(rN, rP - rN/2.).sum(-1) + 1.0) / (dpn ** 2).detach(), dpn < margin),
    #    ]).mean()
    # [no weight + triplet mask]
    # loss = th.stack([
    #    th.mul(rA, rA/2. - rP).sum(-1) + 1.0,
    #    th.mul(rA, rN - rA/2.).sum(-1) + 1.0,
    #    th.mul(rP, rP/2. - rA).sum(-1) + 1.0,
    #    th.mul(rP, rN - rP/2.).sum(-1) + 1.0,
    #    th.mul(rN, rA - rN/2.).sum(-1) + 1.0,
    #    th.mul(rN, rP - rN/2.).sum(-1) + 1.0,
    #    ]).mean(0)
    #mask = (dap - dan + margin >= 0.).view(-1)
    #loss = th.masked_select(loss, mask).mean()
    # [normalized]
    # loss = th.stack([
    #    (th.mul(rA, F.normalize(rA/2. - rP)).sum(-1) + 1.0) * (dap ** 2),
    #    (th.mul(rA, F.normalize(rN - rA/2.)).sum(-1) + 1.0) / (dan ** 2),
    #    (th.mul(rP, F.normalize(rP/2. - rA)).sum(-1) + 1.0) * (dap ** 2),
    #    (th.mul(rP, F.normalize(rN - rP/2.)).sum(-1) + 1.0) / (dpn ** 2),
    #    (th.mul(rN, F.normalize(rA - rN/2.)).sum(-1) + 1.0) / (dan ** 2),
    #    (th.mul(rN, F.normalize(rP - rN/2.)).sum(-1) + 1.0) / (dpn ** 2),
    #    ]).mean(0)
    #mask = (dap - dan + margin >= 0.).view(-1)
    #loss = th.masked_select(loss, mask).mean()
    # [ simple ]
    # 1. should not use mask
    #l1 = (th.mul(rP, rP/2 - rA).sum(-1) + 1.0) * (dap ** 2)
    #l2 = (th.mul(rN, rA - rN/2).sum(-1) + 1.0) / (dan ** 2)
    # [ simple: direction + norm . pow(1)
    #l1 = (th.mul(rP, rP/2 - rA).sum(-1) + 1.0)
    #l2 = (th.mul(rN, rA - rN/2).sum(-1) + 1.0) / (dan**2).detach()
    # [simple: direction + norm . pow(2)
    l1 = (th.mul(rP, rP / 2 - rA).sum(-1) + 1.0) * dap.detach()
    l2 = (th.mul(rN, rA - rN / 2).sum(-1) + 1.0) / (dan**3).detach()
    loss = th.cat([l1, l2]).mean()
    return loss


class pgil(MetricLoss):
    _loss_fn = staticmethod(fn_pgil)

class pgilC(pgil): _metric = 'C'
class pgilE(pgil): _metric = 'E'
class pgilN(pgil): _metric = 'N'


@pytest.mark.parametrize('func', (pgilC, pgilE, pgilN))
def test_pgil(func):
    output, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    loss = func()(output, labels)
    loss.backward()
