import numpy as np


class Delta:
    """Class for deltas in MICE

    :param x: can be of any shape
    :param c: cost of evaluating Delta (integer)
    :param x_l1: a x to estimate the delta with respect with (optional)
    """

    def __init__(self, x, sampler, c=2, x_l1=None, m_min=5):
        self.x_l = x
        self.x_l1 = x_l1
        self.f_delta = np.array([]).reshape(0, len(x))
        self.f_l = np.array([]).reshape(0, len(x))
        self.f_delta_av = np.array(0.)
        self.v_l = None
        self.v_batch = None
        self.m = 0
        self.c = c
        self.m_min = m_min
        self.sampler = sampler

    def update_delta(self, mice, m):
        """Updates Delta object to the new value for sample size.

        :param mice: object from class MICE
        :param m : new sample size for delta (int)
        """
        if self.m < m:
            m_to_samp = np.ceil(m - self.m).astype('int')
            samples = self.sampler(m_to_samp)
            if len(self.x_l1) == 0:
                self.f_delta = np.vstack(
                    [self.f_delta, mice.grad(self.x_l, samples)])
                mice.counter += self.c * m_to_samp
                self.v_batch = mice.var(self.f_delta)
            else:
                new_f_l = mice.grad(self.x_l, samples)
                new_f_delta = new_f_l - mice.grad(self.x_l1, samples)
                self.f_l = np.vstack([self.f_l, new_f_l])
                self.f_delta = np.vstack([self.f_delta, new_f_delta])
                mice.counter += self.c * m_to_samp
                self.v_batch = mice.var(self.f_l)
            self.v_l = mice.var(self.f_delta)
            self.f_delta_av = mice.aggr(self.f_delta)
            self.m = len(self.f_delta)
        return

    def restart(self, mice):
        new_delta = mice.create_delta(self.x_l, c=1)
        # new_delta = deepcopy(self)
        new_delta.x_l1 = None
        new_delta.f_delta = new_delta.f_l
        new_delta.f_delta_av = mice.aggr(new_delta.f_delta)
        new_delta.v_l = self.v_batch
        new_delta.v_batch = self.v_batch
        # new_delta.v_batch = self.v_batch
        # new_delta.f_l = self.f_l
        new_delta.m = self.m
        new_delta.m_min = mice.m_restart_min
        new_delta.c = 1
        new_delta.sampler = self.sampler
        return new_delta

    def __call__(self):
        for key in self.__dict__.keys():
            print(f'{key}: {self.__dict__[key]}')


class LightDelta:
    """Class for low memory deltas in MICE

    :param x: can be of any shape
    :param c: cost of evaluating Delta (integer)
    :param x_l1: a x to estimate the delta with respect with (optional)
    """

    def __init__(self, x, sampler, c=2, x_l1=None):
        self.x_l = x
        self.x_l1 = x_l1
        self.f_l = np.array(0.)
        self.f_delta_av = np.array(0.)
        self.v_l = None
        self.v_batch = None
        self.m2_del = np.array(0.)
        self.m2_l = np.array(0.)
        self.m = 0
        self.c = c
        self.sampler = sampler

    def update_delta(self, mice, m):
        """Updates Delta object to the new value for sample size.

        :param mice: object from class MICE
        :param m : new sample size for delta (int)
        """
        if self.m < m:
            m_to_samp = np.ceil(m - self.m).astype('int')
            samples = self.sampler(m_to_samp)
            if self.c == 1:
                new_values = mice.grad(self.x_l, samples)
                self.f_delta_av, self.m2_del = new_update(
                    self.m, self.f_delta_av, self.m2_del, new_values)
                self.m += m_to_samp
                mice.counter += self.c * m_to_samp
                self.v_batch = self.m2_del / (self.m - 1)
            else:
                new_f_ls = mice.grad(self.x_l, samples)
                new_f_l1s = mice.grad(self.x_l1, samples)
                self.f_l, self.m2_l = new_update(
                    self.m, self.f_l, self.m2_l, new_f_ls)
                self.f_delta_av, self.m2_del = new_update(
                    self.m, self.f_delta_av, self.m2_del, new_f_ls - new_f_l1s)
                self.m += m_to_samp
                mice.counter += self.c * m_to_samp
                self.v_batch = self.m2_l / (self.m - 1)
            self.v_l = self.m2_del / (self.m - 1)
        return

    def restart(self, mice):
        new_delta = mice.create_delta(self.x_l, c=1)
        # new_delta = deepcopy(self)
        new_delta.x_l1 = None
        new_delta.f_l = self.f_l
        new_delta.f_delta_av = self.f_l
        new_delta.v_l = self.v_batch
        new_delta.v_batch = self.v_batch
        new_delta.m2_del = self.m2_l
        new_delta.m2_l = self.m2_l
        new_delta.m = self.m
        new_delta.c = 1
        new_delta.sampler = self.sampler
        return new_delta

    def __call__(self):
        for key in self.__dict__.keys():
            print(f'{key}: {self.__dict__[key]}')


class DeltaResampling:
    """Class for low memory deltas in MICE that estimates norm using Jackknife

    :param x: can be of any shape
    :param c: cost of evaluating Delta (integer)
    :param x_l1: a x to estimate the delta with respect with (optional)
    """

    def __init__(self, x, sampler, c=2, x_l1=None, re_part=2, m_min=5):
        self.x_l = x
        self.x_l1 = x_l1
        self.f_l = np.array(0.)
        # self.f_ls = [0.0 for i in range(re_part)]
        self.f_ls = np.zeros((re_part, len(x)))
        self.f_delta_av = np.zeros((len(x)))
        # self.f_deltas = [0.0 for i in range(re_part)]
        self.f_deltas = np.zeros((re_part, len(x)))
        self.f_ms = [0 for _ in range(re_part)]
        self.v_l = None
        self.v_batch = None
        self.m2_del = np.array(0.)
        self.m2_l = np.array(0.)
        self.m = 0
        self.m_prev = 0
        self.c = c
        self.m_min = m_min
        self.sampler = sampler
        self.re_part = re_part

    def update_delta(self, mice, m):
        """Updates Delta object to the new value for sample size.

        :param mice: object from class MICE
        :param m : new sample size for delta (int)
        """
        if self.m < m:
            m_to_samp = np.ceil(m - self.m).astype('int')
            samples = self.sampler(m_to_samp)
            if self.c == 1:
                new_values = mice.grad(self.x_l, samples)
                self.f_delta_av, self.m2_del = new_update(
                    self.m, self.f_delta_av, self.m2_del, new_values)
                ms = np.arange(self.m, self.m + m_to_samp)
                idxs = ms % self.re_part
                for idx in range(self.re_part):
                    mask = ~(idxs == idx)
                    m_ = self.f_ms[idx]
                    m_new = mask.sum()
                    self.f_deltas[idx] = (
                        self.f_deltas[idx] * m_
                        + np.sum(new_values[mask], axis=0)) / (m_ + m_new)
                    self.f_ms[idx] += m_new
                self.m += m_to_samp
                mice.counter += self.c * m_to_samp
                self.v_batch = self.m2_del / (self.m - 1)
                self.f_l = self.f_delta_av
                self.f_ls = self.f_deltas
            else:
                new_f_ls = mice.grad(self.x_l, samples)
                new_f_l1s = mice.grad(self.x_l1, samples)
                self.f_l, self.m2_l = new_update(
                    self.m, self.f_l, self.m2_l, new_f_ls)
                new_f_deltas = new_f_ls - new_f_l1s
                self.f_delta_av, self.m2_del = new_update(
                    self.m, self.f_delta_av, self.m2_del, new_f_deltas)
                ms = np.arange(self.m, self.m + m_to_samp)
                idxs = ms % self.re_part
                for idx in range(self.re_part):
                    mask = ~(idxs == idx)
                    m_ = self.f_ms[idx]
                    m_new = mask.sum()
                    self.f_deltas[idx] = (
                        self.f_deltas[idx] * m_
                        + np.sum(new_f_deltas[mask], axis=0)) / (m_ + m_new)
                    self.f_ls[idx] = (
                        self.f_ls[idx] * m_
                        + np.sum(new_f_ls[mask], axis=0)) / (m_ + m_new)
                    self.f_ms[idx] += m_new
                self.m += m_to_samp
                mice.counter += self.c * m_to_samp
                self.v_batch = self.m2_l / (self.m - 1)
            self.v_l = self.m2_del / (self.m - 1)
        return

    def restart(self, mice):
        new_delta = mice.create_delta(self.x_l, c=1)
        # new_delta = deepcopy(self)
        new_delta.x_l1 = None
        new_delta.f_delta_av = self.f_l
        new_delta.f_deltas = self.f_ls
        new_delta.f_l = new_delta.f_delta_av
        new_delta.f_ls = new_delta.f_deltas
        new_delta.f_ms = self.f_ms
        new_delta.v_l = self.v_batch
        new_delta.v_batch = self.v_batch
        new_delta.m2_l = self.m2_l
        new_delta.m2_del = self.m2_l
        new_delta.m = self.m
        new_delta.m_min = mice.m_restart_min
        new_delta.c = 1
        new_delta.sampler = self.sampler
        return new_delta

    def __call__(self):
        for key in self.__dict__.keys():
            print(f'{key}: {self.__dict__[key]}')


class DeltaBayHess:
    """Class for low memory deltas in MICE for Bayesian Hessian inverse

    :param x: can be of any shape
    :param c: cost of evaluating Delta (integer)
    :param x_l1: a x to estimate the delta with respect with (optional)
    """

    def __init__(self, x, sampler, c=2, x_l1=None, re_part=2, m_min=5):
        self.x_l = x
        self.x_l1 = x_l1
        self.f_l = np.array(0.)
        # self.f_ls = [0.0 for i in range(re_part)]
        self.f_ls = np.zeros((re_part, len(x)))
        self.f_delta_av = np.zeros((len(x)))
        # self.f_deltas = [0.0 for i in range(re_part)]
        self.f_deltas = np.zeros((re_part, len(x)))
        self.f_ms = [0 for _ in range(re_part)]
        self.v_l = None
        self.v_batch = None
        self.m2_del = np.zeros(len(x))
        self.m2_l = np.zeros(len(x))
        self.m = 0
        self.c = c
        self.m_min = m_min
        self.sampler = sampler
        self.re_part = re_part

    def update_delta(self, mice, m):
        """Updates Delta object to the new value for sample size.

        :param mice: object from class MICE
        :param m : new sample size for delta (int)
        """
        if self.m < m:
            m_to_samp = np.ceil(m - self.m).astype('int')
            samples = self.sampler(m_to_samp)

            # # test
            #
            # av = np.zeros(2)
            # # m2 = np.zeros(2)
            # m2 = 0
            # k = 0
            # num_batch = 1
            # for k in range(num_batch):
            #     # av, m2 = new_update(k*1000/num_batch, av, m2, samples[k::num_batch])
            #     av, m2 = update_bay(k*1000/num_batch, av, m2, samples[k::num_batch])
            #     # av, m2 = new_update(k, av, m2, [samples[k]])
            #     # av, m2 = update_stats(k, av, m2, samples[k])
            #     # av, m2 = update_bay(k, av, m2, samples[k::100])
            #     k += 1
            #
            # print(m2/999)
            # print(np.cov(samples.T, ddof=1))
            # print(np.diag(np.cov(samples.T, ddof=1)).sum())

            if self.c == 1:
                new_values = mice.grad(self.x_l, samples)
                self.f_delta_av, self.m2_del = update_bay(
                        self.m, self.f_delta_av, self.m2_del, new_values)
                ms = np.arange(self.m, self.m + m_to_samp)
                idxs = ms % self.re_part
                for idx in range(self.re_part):
                    mask = ~(idxs == idx)
                    m_ = self.f_ms[idx]
                    m_new = mask.sum()
                    self.f_deltas[idx] = \
                        (self.f_deltas[idx]*m_ + np.sum(new_values[mask],
                                                        axis=0)) / (m_ + m_new)
                    self.f_ms[idx] += m_new
                self.m += m_to_samp
                mice.counter += self.c*m_to_samp
                self.v_batch = np.sum(self.m2_del)/(self.m - 1)
                self.f_l = self.f_delta_av
                self.f_ls = self.f_deltas
            else:
                new_f_ls = mice.grad(self.x_l, samples)
                new_f_l1s = mice.grad(self.x_l1, samples)
                self.f_l, self.m2_l = update_bay(
                        self.m, self.f_l, self.m2_l, new_f_ls)
                new_f_deltas = new_f_ls - new_f_l1s
                self.f_delta_av, self.m2_del = update_bay(
                    self.m, self.f_delta_av, self.m2_del, new_f_deltas)
                ms = np.arange(self.m, self.m + m_to_samp)
                idxs = ms % self.re_part
                for idx in range(self.re_part):
                    mask = ~(idxs == idx)
                    m_ = self.f_ms[idx]
                    m_new = mask.sum()
                    self.f_deltas[idx] = \
                        (self.f_deltas[idx]*m_ + np.sum(new_f_deltas[mask],
                                                        axis=0))/(m_ + m_new)
                    self.f_ls[idx] = \
                        (self.f_ls[idx]*m_ + np.sum(new_f_ls[mask],
                                                    axis=0))/(m_ + m_new)
                    self.f_ms[idx] += m_new
                self.m += m_to_samp
                mice.counter += self.c*m_to_samp
                self.v_batch = np.sum(self.m2_l)/(self.m - 1)
            self.v_l = np.sum(self.m2_del)/(self.m - 1)
        return

    def restart(self, mice):
        new_delta = mice.create_delta(self.x_l, c=1)
        # new_delta = deepcopy(self)
        new_delta.x_l1 = None
        new_delta.f_delta_av = self.f_l
        new_delta.f_deltas = self.f_ls
        new_delta.f_l = new_delta.f_delta_av
        new_delta.f_ls = new_delta.f_deltas
        new_delta.f_ms = self.f_ms
        new_delta.v_l = self.v_batch
        new_delta.v_batch = self.v_batch
        new_delta.m2_l = self.m2_l
        new_delta.m2_del = self.m2_l
        new_delta.m = self.m
        new_delta.m_min = mice.m_restart_min
        new_delta.c = 1
        new_delta.sampler = self.sampler
        return new_delta

    def __call__(self):
        for key in self.__dict__.keys():
            print(f'{key}: {self.__dict__[key]}')


def new_update(count, mean, m2, new_values):
    counts = np.arange(count + 1, count + len(new_values) + 1)
    size = (len(new_values[0]), 1)
    d_mean = (mean * count + np.cumsum(new_values, axis=0)) / \
        np.tile(counts, size).T
    # D_m = np.vstack(mean, d_mean)
    factors = np.divide(counts, (counts - 1), where=(counts > 1))
    m2 += ((np.linalg.norm(new_values - d_mean, axis=1))**2 * factors).sum()
    return d_mean[-1], m2


def update_bay(count, mean, m2, new_values):
    counts = np.arange(count+1, count+len(new_values)+1)
    size = (len(new_values[0]), 1)
    d_mean = (mean*count + np.cumsum(new_values, axis=0))\
        / np.tile(counts, size).T
    # D_m = np.vstack(mean, d_mean)
    factors = np.divide(counts, (counts-1), where=(counts > 1))
    # m2 += ((np.linalg.norm(new_values-d_mean, axis=1))**2*factors).sum()
    m2 += np.sum((new_values-d_mean)**2*np.tile(factors, size).T, axis=0)
    # set_trace()
    return d_mean[-1], m2
