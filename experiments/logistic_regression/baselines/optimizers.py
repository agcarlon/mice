"""
Baseline optimizer implementations for logistic-regression comparisons.
"""

from __future__ import annotations

from functools import partial
from itertools import cycle

import numpy as np


def sampler_finite(n: int, sample_iterator):
    return [sample_idx for _, sample_idx in zip(range(n), sample_iterator)]


class SAG:
    def __init__(self, func, sample, batchsize=100, max_cost=1000, verbose=False):
        self.func = func
        self.sample = sample
        self.batchsize = batchsize
        self.max_cost = max_cost
        self.verbose = verbose
        self.grad_mem = []
        self.sampler = self.create_sampler()
        self.datasize = len(sample)
        self.counter = 0
        self.log = []
        self.average = 0
        self.force_exit = False
        self.initialized = False

    def evaluate(self, x):
        if self.verbose:
            print("Evaluating SAG")
        if not self.initialized:
            self.grad_mem = self.func(x, self.sample)
            self.average = np.nanmean(self.grad_mem, axis=0)
            self.counter = len(self.sample)
            self.log.append(self.counter)
            self.initialized = True
            return self.average
        sample_idxs = self.sampler(self.batchsize)
        sample = self.sample[sample_idxs] if isinstance(self.sample, np.ndarray) else [self.sample[i] for i in sample_idxs]
        deltas = self.func(x, sample) - self.grad_mem[sample_idxs]
        self.average += np.sum(deltas, axis=0) / self.datasize
        self.grad_mem[sample_idxs] += deltas
        self.counter += self.batchsize
        self.log.append(self.counter)
        self.check_max_cost()
        return self.average

    def check_max_cost(self):
        if self.counter > self.max_cost:
            self.force_exit = True
            return True
        return False

    def create_sampler(self):
        sample_idx = np.arange(len(self.sample))
        sample_iterator = cycle(sample_idx)
        return partial(sampler_finite, sample_iterator=sample_iterator)


class SAGA:
    def __init__(self, func, sample, batchsize=100, max_cost=1000, verbose=False):
        self.func = func
        self.sample = sample
        self.batchsize = batchsize
        self.max_cost = max_cost
        self.verbose = verbose
        self.grad_mem = []
        self.sampler = self.create_sampler()
        self.datasize = len(sample)
        self.counter = 0
        self.log = []
        self.average = 0
        self.force_exit = False
        self.initialized = False

    def evaluate(self, x):
        if self.verbose:
            print("Evaluating SAGA")
        if not self.initialized:
            self.grad_mem = self.func(x, self.sample)
            self.average = np.nanmean(self.grad_mem, axis=0)
            self.counter = len(self.sample)
            self.log.append(self.counter)
            self.initialized = True
            return self.average
        sample_idxs = self.sampler(self.batchsize)
        sample = self.sample[sample_idxs] if isinstance(self.sample, np.ndarray) else [self.sample[i] for i in sample_idxs]
        deltas = self.func(x, sample) - self.grad_mem[sample_idxs]
        estim = self.average + np.sum(deltas, axis=0) / self.batchsize
        self.average += np.sum(deltas, axis=0) / self.datasize
        self.grad_mem[sample_idxs] += deltas
        self.counter += self.batchsize
        self.log.append(self.counter)
        self.check_max_cost()
        return estim

    def check_max_cost(self):
        if self.counter > self.max_cost:
            self.force_exit = True
            return True
        return False

    def create_sampler(self):
        sample_idx = np.arange(len(self.sample))
        sample_iterator = cycle(sample_idx)
        return partial(sampler_finite, sample_iterator=sample_iterator)


class SARAH:
    def __init__(self, func, sample, batchsize=100, max_cost=1000, m=10000, verbose=False):
        self.func = func
        self.sample = sample
        self.batchsize = batchsize
        self.max_cost = max_cost
        self.verbose = verbose
        self.sampler = self.create_sampler()
        self.counter = 0
        self.m = m
        self.estim = []
        self.datasize = len(sample)
        self.log = []
        self.force_exit = False
        self.initialized = False

    def evaluate(self, x, k):
        if self.verbose:
            print("Evaluating SARAH")
        if not self.initialized or not (k % self.m):
            self.estim = np.mean(self.func(x, self.sample), axis=0)
            self.x_ = x
            self.counter += len(self.sample)
            self.log.append(self.counter)
            self.initialized = True
            return self.estim
        sample_idxs = self.sampler(self.batchsize)
        sample = self.sample[sample_idxs] if isinstance(self.sample, np.ndarray) else [self.sample[i] for i in sample_idxs]
        deltas = self.func(x, sample) - self.func(self.x_, sample)
        self.estim = self.estim + np.mean(deltas, axis=0)
        self.x_ = x
        self.counter += 2 * self.batchsize
        self.log.append(self.counter)
        self.check_max_cost()
        return self.estim

    def check_max_cost(self):
        if self.counter > self.max_cost:
            self.force_exit = True
            return True
        return False

    def create_sampler(self):
        sample_idx = np.arange(len(self.sample))
        sample_iterator = cycle(sample_idx)
        return partial(sampler_finite, sample_iterator=sample_iterator)


class SVRG:
    def __init__(self, func, sample, batchsize=100, max_cost=1000, m=10000, verbose=False):
        self.func = func
        self.sample = sample
        self.batchsize = batchsize
        self.max_cost = max_cost
        self.verbose = verbose
        self.sampler = self.create_sampler()
        self.counter = 0
        self.m = m
        self.average = []
        self.datasize = len(sample)
        self.log = []
        self.force_exit = False
        self.initialized = False

    def evaluate(self, x, k):
        if self.verbose:
            print("Evaluating SVRG")
        if not self.initialized or not (k % self.m):
            self.grad_mem = self.func(x, self.sample)
            self.average = np.nanmean(self.grad_mem, axis=0)
            self.x_ = x
            self.counter += len(self.sample)
            self.log.append(self.counter)
            self.initialized = True
            return self.average
        sample_idxs = self.sampler(self.batchsize)
        sample = self.sample[sample_idxs] if isinstance(self.sample, np.ndarray) else [self.sample[i] for i in sample_idxs]
        deltas = self.func(x, sample) - self.grad_mem[sample_idxs]
        estim = self.average + np.mean(deltas, axis=0)
        self.counter += self.batchsize
        self.log.append(self.counter)
        self.check_max_cost()
        return estim

    def check_max_cost(self):
        if self.counter > self.max_cost:
            self.force_exit = True
            return True
        return False

    def create_sampler(self):
        sample_idx = np.arange(len(self.sample))
        sample_iterator = cycle(sample_idx)
        return partial(sampler_finite, sample_iterator=sample_iterator)
