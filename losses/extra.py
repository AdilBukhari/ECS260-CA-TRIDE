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

Loss functions borrowed from Pytorch-Metric-Learning
  https://kevinmusgrave.github.io/pytorch-metric-learning
'''
import numpy as np
import torch as th
import sys
sys.path.append('/home/tianqiwei/jupyter/rob_IR/')
import datasets
import configs
from utility import utils
import pytorch_metric_learning as dml
import pytorch_metric_learning.losses
import pytorch_metric_learning.miners
import pytorch_metric_learning.reducers
import pytorch_metric_learning.distances


def _index_filter(indices: tuple, most: int):
    '''
    Pytorch-metric-learning's miners outputs too many usable tuples
    so that OOM is very easy to trigger.
    '''
    num_indices = len(indices[0])
    if num_indices > most:
        sel = th.randperm(num_indices)[:most].to(indices[0].device)
        return tuple(ind[sel] for ind in indices)
    return indices


class ExtraLossN(th.nn.Module):
    def __init__(self, loss_func, miner, metric, datasetspec='SPC-2'):
        super().__init__()
        self._lossfunc = loss_func
        self._miner = miner
        self._metric = metric
        self._datasetspec = datasetspec

    def forward(self, *args, **kwargs):
        repres, labels = args[0], args[1].view(-1)
        repres = th.nn.functional.normalize(repres, p=2)
        indices = self._miner(repres, labels)
        indices = _index_filter(indices, repres.size(0))
        return self._lossfunc(repres, labels, indices)

    def determine_metric(self):
        return self._metric

    def datasetspec(self):
        return self._datasetspec


class pstripN(ExtraLossN):
    def __init__(self):
        loss_func = dml.losses.TripletMarginLoss(
            margin=configs.triplet.margin_euclidean,
            reducer=dml.reducers.ThresholdReducer(low=0.),
            distance=dml.distances.LpDistance(
                p=2, power=1, normalize_embeddings=True)
        )
        miner = dml.miners.TripletMarginMiner(
            margin=configs.triplet.margin_euclidean,
            type_of_triplets='semihard')
        super().__init__(loss_func, miner, 'N')


class pangularN(ExtraLossN):
    def __init__(self):
        super().__init__(dml.losses.AngularLoss(), dml.miners.AngularMiner(), 'N')


class pncaN(ExtraLossN):
    def __init__(self):
        loss_func = dml.losses.NCALoss()
        miner = dml.miners.TripletMarginMiner(
            margin=configs.triplet.margin_euclidean,
            type_of_triplets='semihard')
        super().__init__(loss_func, miner, 'N')


def test_pstripN():
    output, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    loss = pstripN()(output, labels)
    loss.backward()


def test_pangularN():
    output, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    loss = pangularN()(output, labels)
    loss.backward()


def test_pncaN():
    output, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    loss = pncaN()(output, labels)
    loss.backward()
