from mice import MICE
import numpy as np
from numpy.linalg import norm
from time import time
import os
from load_data import load_data
from copy import deepcopy
from IPython.core.debugger import set_trace
import matplotlib.pyplot as plt

# from pycallgraph import PyCallGraph, GlobbingFilter, Config
# from pycallgraph.output import GraphvizOutput


def get_bias(dF, true_grad):
    Mls_ = [lvl.m for lvl in dF.deltas]
    Mls = [lvl.m_prev for lvl in dF.deltas]
    expected_bias = [Mls[0]/Mls_[0] *
                     (dF.deltas[0].f_delta_av - true_grad(dF.deltas[0].x_l))]
    for m_p, m_k, delta in zip(Mls[1:], Mls_[1:], dF.deltas[1:-1]):
        err_l = delta.f_delta_av - \
            (true_grad(delta.x_l) - true_grad(delta.x_l1))
        expected_bias.append(m_p/m_k * err_l)

    # exp_bias = np.sum(expected_bias, axis=0)
    return expected_bias


def bias_test(dF, x, true_grad, iter, num_reps=1000000):
    grads = []
    Mls = [lvl.m for lvl in dF.deltas]
    for i in range(num_reps):
        dF_ = deepcopy(dF)
        # grads.append(dF_.evaluate(x))
        dF_.deltas.append(dF_.create_delta(
            x, c=2, x_l1=dF_.deltas[-1].x_l))
        dF_.deltas[-1].update_delta(dF_, 10)
        for delta in dF_.deltas:
            delta.update_delta(dF_, delta.m*2)
        grads.append(dF_.aggr_deltas())
    bias = np.mean(grads, axis=0) - true_grad(x)
    Mls_ = [lvl.m for lvl in dF_.deltas[:-1]]
    expected_bias = [Mls[0]/Mls_[0] *
                     (dF.deltas[0].f_delta_av - true_grad(dF.deltas[0].x_l))]
    for m_p, m_k, delta in zip(Mls[1:], Mls_[1:], dF.deltas[1:]):
        err_l = delta.f_delta_av - \
            (true_grad(delta.x_l) - true_grad(delta.x_l1))
        expected_bias.append(m_p/m_k * err_l)

    bias_rep = np.cumsum(grads, axis=0)/np.arange(1, num_reps+1)[:, np.newaxis] - true_grad(x)
    print('eee')

def sgd_mice(dataset,
            Lambda=1e-5,
            st=.1,
            seed=0):
    print(f'Solving logistic regression of {dataset} dataset using SGD-MICE')
    clip = None
    directory = f'{dataset}/'
    name = f'sgd_mice'

    t0 = time()
    np.random.seed(seed)

    if not os.path.exists(directory):
        os.makedirs(directory)

    name = directory + '/' + name

    t0 = time()
    # thetas_ = np.load(f'{directory}{dataset}.npy', allow_pickle=True)
    thetas = load_data(f'{directory}{dataset}', 8124, 112)

    # X = np.stack(thetas[:, 0])
    # Y = np.stack(thetas[:, 1])
    X = thetas[0]
    Y = thetas[1]
    Y = Y*2 - 3

    thetas = [*zip(X, Y)]
    np.random.shuffle(thetas)
    datasize, n_features = np.shape(X)
    n_dim = n_features

    print(f'Load time: {time() - t0}')

    W0 = np.zeros(shape=(n_features))
    # np.save(f'{directory}Optimum.npy', W0)
    opt_W = np.load(f'{directory}Optimum.npy')
    # W0 = np.array(opt_W)

    t0 = time()
    # hess = .25*X.T @ X / datasize + Lambda*np.eye(n_features)
    # L = largest_eigsh(hess, 1, which='LM')[0][0]
    L = 0.25 * np.mean((X**2).sum(axis=1)) + Lambda
    print(f'L: {L}')
    print(f'L estim. time: {time() - t0}')

    def sigmoid(Z):
        return 1/(1+np.exp(-Z))

    def logloss_full(W):
        ls = (np.log(1 + np.exp(Y * (X @ W)))) + .5*Lambda*(W @ W)
        return np.mean(ls)

    def lossgrad_full(W):
        gr3 = (sigmoid(Y * (X @ W))*Y) @ X / datasize + Lambda * W
        return gr3

    def logloss(W, thetas):
        X_, Y_ = zip(*thetas)
        ls = np.log(1 + np.exp(Y_ * (X_ @ W))) + .5*Lambda*(W @ W)
        return np.array(ls)

    def lossgrad(W, theta):
        X_, Y_ = zip(*theta)
        gr2 = np.tile(sigmoid(Y_ * (X_ @ W))*Y_,
                      [n_features, 1]).T * X_ + Lambda * np.tile(W, [len(X_), 1])
        return gr2

    def hessian(W):
        hess = np.zeros((n_features, n_features))
        for x, y in zip(X, Y):
            z = y * (x @ W)
            hess += sigmoid(z)*(1 - sigmoid(z)) * np.outer(x, x)
        hess /= datasize
        hess += Lambda*np.eye(n_features)
        return hess

    def accuracy(W):
        p_true = sigmoid(-X @ W)
        p_false = sigmoid(X @ W)
        P = p_true > p_false
        acc = np.mean((P*2-1) == Y)
        return acc

    def cos_angle(x, y):
        return (x @ y)/(norm(x)*norm(y))

    opt_loss = logloss_full(opt_W)
    # W0 = -opt_W
    start_loss = logloss_full(W0)
    epochs = 10
    relax_const = 1.1

    print(f'start loss: {start_loss:.6f}, opt loss:{opt_loss:.6f}')
    df = MICE(lossgrad,
                   sampler=thetas,
                   eps=.7,
                   dropping=True,
                   drop_param=0.0,
                   min_batch=10,
                   restart_factor=10,
                   restart=True,
                   restart_param=0.0,
                   max_cost=epochs*datasize,
                   mice_type='resampling',
                   max_hierarchy_size=100,
                   verbose=False)
    grad = lossgrad_full(W0)

    W = [W0]
    iters = epochs
    losses = [logloss_full(W[-1])]
    costs = [1e-3]
    total_cost = epochs*datasize
    k = 0
    step_size = 2/(L+Lambda)*(1/(1+df.eps**2))
    coss_bias = []
    while not df.terminate:
        k += 1
        grad = df.evaluate(W[-1])
        bias = get_bias(df, lossgrad_full)
        coss_bias.append(cos_angle(np.sum(bias, axis=0), W[-1] - opt_W))
        if df.terminate:
            break
        W.append(W[-1] - step_size*grad)
        # W.append(W[-1] - 1/L*grad)
        print(logloss_full(W[-1]))
        losses.append(logloss_full(W[-1]))
        costs.append(df.counter)

    losses = np.array(losses) - opt_loss
    return losses, costs


if __name__ == '__main__':
    loss, cost = sgd_mice('mushrooms')
