import os
import matplotlib.pyplot as plt
import numpy as np
from mice import MICE, plot_mice, plot_config
from IPython.core.debugger import set_trace


def sgd_mice(eps_rel=1., kappa=100):
    directory = 'QuadC'
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
              m_min=5)

    chain_size = []
    grad = []

    n_iter = 1e7
    X = [np.array([20., 50.])]

    k = 0
    stepsize = 1 / L
    # stepsize = 2.0 / (mu + L) / (1 + dF.eps**2)
    while (not dF.force_exit) and k < n_iter:
        k += 1
        grad.append(dF.evaluate(X[-1]))
        if dF.force_exit:
            break
        X.append(X[-1] - stepsize * grad[-1])
        print(f'k: {k}, {dF.log[-1][0]}, Vl: {dF.log[-1][2]}, X: {X[-1]}, '
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

    rate = (1 - stepsize*mu*(1 - 2*eps_rel**2))
    iter_conv = [int(len(log)*.9), len(log)]
    iter_conv_loss = [1e-2, 1e-2 * rate**(iter_conv[1] - iter_conv[0])]

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

    P = np.polyfit(log.iteration, np.log(log['opt_gap']), 1)

    P2 = np.polyfit(log.iteration, 2*np.log(log['dist_to_opt']), 1)
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
    # axs[0].plot(iter_conv, iter_conv_loss, 'k--', label=r'$B^k$')
    axs[0].legend()
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
        r'$\|\nabla_{\boldsymbol{\xi}} \mathcal{F}_k - \nabla_{\boldsymbol{'
        r'\xi}} F_k\| / \|\nabla_{\boldsymbol{\xi}} F_k\|$')
    axs[2].set_xlabel(r'iteration')
    axs[2].plot(np.full(len(log), eps_rel), 'k--')
    axs[2].text(30, eps_rel * 1.06, r'$\epsilon$',
                verticalalignment='baseline')
    plt.tight_layout()
    plt.savefig(name + '_vl_num_grads_err.pdf')

    fig, axs = plt.subplots(2, 1, figsize=(8, 8))
    axs[0] = plot_mice(log, axs[0], x='iteration', y='bias_rel_err',
                       style='semilogy', legend=True)
    axs[0].set_ylabel(r'$\|\mathcal{B}_k\| / \|\mathcal{F}_k\|$')
    axs[0].plot(np.full(len(log), eps_rel), 'k--')
    axs[0].text(30, eps_rel * 1.06, r'$\epsilon$',
                verticalalignment='baseline')
    axs[1] = plot_mice(log, axs[1], x='iteration', y='rel_error',
                       style='semilogy', legend=False)
    axs[1].set_ylabel(
        r'$\|\nabla_{\boldsymbol{\xi}} \mathcal{F}_k - \nabla_{\boldsymbol{'
        r'\xi}} F_k\| / \|\nabla_{\boldsymbol{\xi}} F_k\|$')
    axs[1].set_xlabel(r'iteration')
    axs[1].plot(np.full(len(log), eps_rel), 'k--')
    axs[1].text(30, eps_rel * 1.06, r'$\epsilon$',
                verticalalignment='baseline')
    plt.tight_layout()
    plt.savefig(name + '_errors.pdf')


if __name__ == '__main__':
    sgd_mice(eps_rel=.5)
