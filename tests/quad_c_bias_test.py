import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from IPython.core.debugger import set_trace
from copy import deepcopy

sys.path.append("../mice")
from mice import MICE, plot_mice, plot_config

def bias_L2_convergence(dF, x, true_grad, iter, num_reps=1000):
    grads = []
    m_increase_factor = 2
    Mls = [lvl.m for lvl in dF.deltas]
    true_grads = [true_grad(dF.deltas[0].x_l)]
    expected_bias = [1/m_increase_factor * (dF.deltas[0].f_delta_av - true_grad(dF.deltas[0].x_l))]
    for delta in dF.deltas[1:]:
        true_grads.append((true_grad(delta.x_l) - true_grad(delta.x_l1)))
        err_l = delta.f_delta_av - \
            (true_grad(delta.x_l) - true_grad(delta.x_l1))
        expected_bias.append(1/m_increase_factor * err_l)

    inner = []
    for i in range(num_reps):
        dF_ = deepcopy(dF)
        # grads.append(dF_.evaluate(x))
        dF_.deltas.append(dF_.create_delta(
            x, c=2, x_l1=dF_.deltas[-1].x_l))
        dF_.deltas[-1].update_delta(dF_, 10)
        for delta in dF_.deltas:
            delta.update_delta(dF_, delta.m*m_increase_factor)
        grads.append(dF_.aggr_deltas())
        inner.append([])
        n_lvls = len(dF.deltas)
        for l in range(n_lvls):
            for j in range(l+1, n_lvls):
                inner_lj = (dF_.deltas[l].f_delta_av - true_grads[l]) \
                            @ (dF_.deltas[j].f_delta_av - true_grads[j])
                inner[-1].append(inner_lj)

    bias = np.mean(grads, axis=0) - true_grad(x)



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


    exp_bias = np.sum(expected_bias, axis=0)

    error = np.abs(bias_rep - exp_bias)


    fig, axs = plt.subplots(2, 1, figsize=(5, 5), sharex=True)
    axs[0].loglog(error[:, 0], label='Absolute error convergence')
    axs[0].loglog(np.abs(bias_rep[:, 0]), label='Empirical bias')
    axs[0].axhline(np.abs(exp_bias[0]), ls='--', c='k', label='Expected bias')
    axs[0].legend()
    axs[0].set_ylabel(r'$\nabla F_0$')
    axs[0].grid()
    axs[0].set_title(
        f'Iteration: {iter}, length of hierarchy: {len(dF_.deltas)}')

    axs[1].loglog(error[:, 1], label='Absolute error convergence')
    axs[1].loglog(np.abs(bias_rep[:, 1]), label='Empirical bias')
    axs[1].axhline(np.abs(exp_bias[1]), ls='--', c='k', label='Expected bias')
    axs[1].set_ylabel(r'$\nabla F_1$')
    axs[1].set_xlabel('Repetitions')
    axs[1].grid()
    fig.tight_layout()
    fig.savefig(f'bias_conv_{iter}.pdf')

    error = np.linalg.norm(bias_rep - exp_bias, axis=1)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.loglog(error, label=r'Error norm')
    ax.set_ylabel('Error')
    ax.set_xlabel('Repetitions')
    ax.grid()
    ax.set_title(f'Iteration: {iter}, length of hierarchy: {len(dF_.deltas)}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'bias_norm_conv_{iter}.pdf')


def sgd_mice(eps_rel=1., kappa=100):
    directory = 'tests/QuadC'
    name = f'SGD_MICE'
    if not os.path.exists(directory):
        os.makedirs(directory)

    subdir = directory + '/' + name
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    name = subdir + '/' + name

    np.random.seed(0)

    H0 = np.eye(2)
    H1 = np.array(
        [[kappa * 2, .5],
         [.5,    1]]
    )
    b = np.ones(2)

    EH = .5 * (H0 + H1)

    def objf(x, theta):
        H = H0 * (1 - theta) + H1 * theta
        return .5 * (x @ H @ x) - b @ x

    def dobjf(x, theta):
        grad = np.outer(x, (1 - theta)).T + np.outer((x @ H1), theta).T - b
        return grad

    def Eobjf(x):
        return .5 * (x @ EH @ x) - b @ x

    def Edobjf(x):
        return x @ EH - b

    optimum = np.linalg.solve(EH, b)
    f_opt = Eobjf(optimum)

    L = np.linalg.eig(EH)[0].max()
    mu = np.linalg.eig(EH)[0].min()
    print(f'True kappa: {L/mu}')

    def sampler(n):
        return np.random.uniform(0, 1, int(n))

    dF = MICE(dobjf,
              sampler=sampler,
              eps=eps_rel,
              max_cost=1e7,
              m_min=5,
              restart=False,
              dropping=False,
              verbose=False)

    chain_size = []
    grad = []

    n_iter = 21
    X = [np.array([2., 5.])]
    test_iters = [2, 3, 4, 5, 10, 20, 49, 100, 500]
    k = 0
    while (not dF.force_exit) and k < n_iter:
        k += 1
        if k in test_iters:
            # bias_L2_convergence(dF, X[-1], Edobjf, k)
            bias_test(dF, X[-1], Edobjf, k)
        grad.append(dF.evaluate(X[-1]))
        if dF.force_exit:
            break
        stepsize = 2.0 / (mu + L) / (1 + dF.eps**2)
        X.append(X[-1] - stepsize * grad[-1])
        print(f'k: {k}, {dF.log[-1][0]}, Vl: {dF.log[-1][0]}, X: {X[-1]}, '
              f'grad: {grad[-1]}, '
              f'#grad: {dF.counter}, '
              f'eps.: {dF.eps}')
        chain_size.append(len(dF.deltas))
    print(dF.aggr_cost)
    Fs = [Eobjf(x) - f_opt for x in X]
    chain_size.append(len(dF.deltas))

    log = dF.log
    log['x'] = X
    log['estimate'] = grad
    dFs = np.vstack([Edobjf(x) for x in X])
    log['opt_gap'] = Fs
    log['grad_norm'] = np.linalg.norm(np.vstack(log['estimate']), axis=1)
    log['dist_to_opt'] = np.linalg.norm(np.vstack(log['x']) - optimum, axis=1)
    log['chain_size'] = chain_size
    log['rel_error'] = np.linalg.norm(
        np.vstack(grad) - dFs, axis=1) / np.linalg.norm(dFs, axis=1)
    log['iteration'] = log.index + 1

    log.to_pickle(name)

    mc_conv_grads = [1e5, 1e6]
    mc_conv_loss = [1e-2, 1e-2 * 10**-.5]
    mc_conv_loss2 = [1e-2, 1e-2 * 10**-1]

    mc_conv_dist = [1e-1, 1e-1 * 10**-.5]
    mc_conv_dist2 = [1e-1, 1e-1 * 10**-1]

    mc_conv_norm = [1e-1, 1e-1 * 10**-.5]
    mc_conv_norm2 = [1e-1, 1e-1 * 10**-1]

    assympt = log['num_grads'] > 1e5
    assympt[len(assympt) - 1] = False
    P = np.polyfit(np.log(log['num_grads'][assympt]),
                   np.log(log['opt_gap'][assympt]), 1)
    print(P)

    P = np.polyfit(np.log(log['num_grads'][assympt]),
                   np.log(log['dist_to_opt'][assympt]), 1)
    print(P)

    P = np.polyfit(np.log(log['num_grads'][assympt]),
                   np.log(log['grad_norm'][assympt]), 1)
    print(P)
    print(Fs[-1])

    plot_config()
    fig, axs = plt.subplots(3, 1, figsize=(8, 8))
    axs[0] = plot_mice(log, axs[0], x='num_grads', y='opt_gap', legend=False)
    axs[0].set_ylabel(r'$F(\boldsymbol{\xi}_k) - F(\boldsymbol{\xi}^*)$')
    axs[0].plot([], [], 'k--', label=r'$\mathcal{O}(M^{-1/2})$')
    axs[0].plot(mc_conv_grads, mc_conv_loss2, 'k-.',
                label=r'$\mathcal{O}(M^{-1})$')
    axs[0].legend()

    axs[1] = plot_mice(log, axs[1], x='num_grads',
                       y='dist_to_opt', legend=False)
    axs[1].set_ylabel(
        r'$\left\rVert \boldsymbol{\xi}_k - \boldsymbol{\xi}^*\right\rVert$')
    axs[1].plot(mc_conv_grads, mc_conv_dist, 'k--')
    axs[2] = plot_mice(log, axs[2], x='num_grads',
                       y='grad_norm', legend=False)
    axs[2].set_ylabel(
        r'$\left\rVert\nabla_{\boldsymbol{\xi}} \mathcal{F}_{k}\right\rVert$')
    axs[2].set_xlabel(r'\# Grad')
    axs[2].plot(mc_conv_grads, mc_conv_norm, 'k--')
    plt.tight_layout()
    plt.savefig(name + '_all_per_grads.pdf')

    fig, axs = plt.subplots(3, 1, figsize=(8, 8))
    axs[0] = plot_mice(log, axs[0], x='iteration', y='opt_gap',
                       style='semilogy')
    axs[0].set_ylabel(r'$F(\boldsymbol{\xi}_k) - F(\boldsymbol{\xi}^*)$')
    axs[1] = plot_mice(log, axs[1], x='iteration', y='dist_to_opt',
                       legend=False, style='semilogy')
    axs[1].set_ylabel(
        r'$\left\rVert \boldsymbol{\xi}_k - \boldsymbol{\xi}^*\right\rVert$')
    axs[2] = plot_mice(log, axs[2], x='iteration', y='grad_norm',
                       legend=False, style='semilogy')
    axs[2].set_ylabel(
        r'$\left\rVert\nabla_{\boldsymbol{\xi}} \mathcal{F}_{k}\right\rVert$')
    axs[2].set_xlabel(r'iteration')
    plt.tight_layout()
    plt.savefig(name + '_all_per_iter.pdf')

    fig, axs = plt.subplots(1, 1, figsize=(6, 4))
    axs = plot_mice(log, axs, x='iteration', y='chain_size',
                    style='plot', legend=False)
    chain_diff = log['chain_size'].diff()
    clip_iter = log[(chain_diff < 0) & (log['event'] != 'restart')].index
    for it in clip_iter:
        axs.plot(log[it - 1:it + 1]['iteration'],
                 log[it - 1:it + 1]['chain_size'], 'r')
    axs.plot([], [], 'r', label='Clipping')
    axs.legend()
    axs.set_xlabel('iteration')
    axs.set_ylabel(r'$| \mathcal{L}_{k} |$')
    fig.tight_layout()
    plt.savefig(name + '_chain_size.pdf')

    fig, axs = plt.subplots(3, 1, figsize=(8, 8))
    axs[0] = plot_mice(log, axs[0], x='iteration', y='vl', style='semilogy')
    axs[0].set_ylabel(r'$V_{\ell}$')
    axs[1] = plot_mice(log, axs[1], x='iteration', y='num_grads',
                       style='semilogy', legend=False)
    axs[1].set_ylabel(r'\# Grad')
    axs[2] = plot_mice(log, axs[2], x='iteration', y='rel_error',
                       style='semilogy', legend=False)
    axs[2].set_ylabel(
        r'$\|\nabla_{\boldsymbol{\xi}} \mathcal{F}_k - \nabla_{\boldsymbol{\xi}} F_k\| / \|\nabla_{\boldsymbol{\xi}} F_k\|$')
    axs[2].set_xlabel(r'iteration')
    axs[2].plot(np.full(len(log), eps_rel), 'k--')
    axs[2].text(30, eps_rel * 1.06, r'$\epsilon$',
                verticalalignment='baseline')
    plt.tight_layout()
    plt.savefig(name + '_vl_num_grads_err.pdf')


if __name__ == '__main__':
    sgd_mice()
