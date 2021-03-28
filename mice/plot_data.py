import sys
sys.path.append("../../..")

from IPython.core.debugger import set_trace
import numpy as np
import PlotConfig
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from MICE import plot_mice


def plot_data(Log, output, eps=None, text=None):
    with PdfPages(output) as pdf:
        if text is not None:
            fig, axs = plt.subplots(1, 1, figsize=(5, 5))
            axs.text(.5, .7, text[0], horizontalalignment='center',
                     verticalalignment='center', wrap=True)
            axs.text(.5, .3, text[1], horizontalalignment='center',
                     verticalalignment='center', wrap=True)
            plt.setp(axs, frame_on=False, xticks=(), yticks=())
            pdf.savefig(fig)

        fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        axs[0] = plot_mice(Log, axs[0], x='num_grads', y='opt_gap')
        axs[0].set_ylabel(r'$F(\boldsymbol{\xi}_k) - F(\boldsymbol{\xi}^*)$')
        axs[1] = plot_mice(Log, axs[1], x='num_grads',
                           y='dist_to_opt', legend=False)
        axs[1].set_ylabel(
            r'$\left\rVert \boldsymbol{\xi}_k - \boldsymbol{\xi}^*\right\rVert$')
        axs[2] = plot_mice(Log, axs[2], x='num_grads',
                           y='grad_norm', legend=False)
        axs[2].set_ylabel(
            r'$\left\rVert\nabla_{\boldsymbol{\xi}} \mathcal{F}_{k}\right\rVert$')
        axs[2].set_xlabel(r'\# Grad')
        plt.tight_layout()
        pdf.savefig(fig)

        fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        axs[0] = plot_mice(Log, axs[0], x='runtime', y='opt_gap')
        axs[0].set_ylabel(r'$F(\boldsymbol{\xi}_k) - F(\boldsymbol{\xi}^*)$')
        axs[1] = plot_mice(Log, axs[1], x='runtime',
                           y='dist_to_opt', legend=False)
        axs[1].set_ylabel(
            r'$\left\rVert \boldsymbol{\xi}_k - \boldsymbol{\xi}^*\right\rVert$')
        axs[2] = plot_mice(Log, axs[2], x='runtime',
                           y='grad_norm', legend=False)
        axs[2].set_ylabel(
            r'$\left\rVert\nabla_{\boldsymbol{\xi}} \mathcal{F}_{k} \right\rVert$')
        axs[2].set_xlabel(r'Runtime (s)')
        fig.tight_layout()
        pdf.savefig(fig)

        fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        axs[0] = plot_mice(Log, axs[0], x='iteration', y='opt_gap',
                           style='semilogy')
        axs[0].set_ylabel(r'$F(\boldsymbol{\xi}_k) - F(\boldsymbol{\xi}^*)$')
        axs[1] = plot_mice(Log, axs[1], x='iteration', y='dist_to_opt',
                           legend=False, style='semilogy')
        axs[1].set_ylabel(
            r'$\left\rVert \boldsymbol{\xi}_k - \boldsymbol{\xi}^*\right\rVert$')
        axs[2] = plot_mice(Log, axs[2], x='iteration', y='grad_norm',
                           legend=False, style='semilogy')
        axs[2].set_ylabel(
            r'$\left\rVert\nabla_{\boldsymbol{\xi}} \mathcal{F}_{k}\right\rVert$')
        axs[2].set_xlabel(r'iteration')
        plt.tight_layout()
        pdf.savefig(fig)

        fig, axs = plt.subplots(1, 1, figsize=(6, 4))
        axs = plot_mice(Log, axs, x='iteration', y='runtime',
                        style='semilogy', markers=False, color='C0')
        axs = plot_mice(Log, axs, x='iteration', y='grads_time',
                        style='semilogy', markers=False, color='C1')
        axs = plot_mice(Log, axs, x='iteration', y='aggr_time',
                        style='semilogy', markers=False, color='C2')
        axs.set_xlabel('iteration')
        axs.set_ylabel('Runtime (s)')
        axs.legend(('Total time', 'Gradient evaluation time', 'Aggregation time'))
        fig.tight_layout()
        pdf.savefig(fig)

        fig, axs = plt.subplots(1, 1, figsize=(6, 4))
        axs = plot_mice(Log, axs, x='iteration', y='grads_time_rel',
                        style='plot', markers=False, color='C1')
        axs = plot_mice(Log, axs, x='iteration', y='aggr_time_rel',
                        style='plot', markers=False, color='C2')
        axs.set_xlabel('iteration')
        axs.set_ylabel('Runtime (relative)')
        axs.legend(('Relative gradient evaluation time',
                    'Relative aggregation time'))
        fig.tight_layout()
        pdf.savefig(fig)

        fig, axs = plt.subplots(1, 1, figsize=(6, 4))
        axs = plot_mice(Log, axs, x='iteration', y='chain_size', style='plot', legend=False)
        # chain_diff = Log['chain_size'].diff()
        # clip_iter = Log[(chain_diff < 0) & (Log['event'] != 'restart')].index
        # for it in clip_iter:
        #     axs.plot(Log[it-1:it+1]['iteration'], Log[it-1:it+1]['chain_size'], 'r')
        # axs.plot([], [], 'r', label='Clipping')
        axs.legend()
        axs.set_xlabel('iteration')
        axs.set_ylabel(r'$| \mathcal{L}_{k} |$')
        fig.tight_layout()
        pdf.savefig(fig)

        fig, axs = plt.subplots(1, 1, figsize=(6, 4))
        axs = plot_mice(Log, axs, x='iteration', y='times_iter', style='plot')
        axs.set_xlabel('iteration')
        axs.set_ylabel('Time per iteration (s)')
        fig.tight_layout()
        pdf.savefig(fig)

        fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        axs[0] = plot_mice(Log, axs[0], x='iteration',
                           y='vl', style='semilogy')
        axs[0].set_ylabel(r'$V_{\ell}$')
        axs[1] = plot_mice(Log, axs[1], x='iteration', y='num_grads',
                           style='semilogy', legend=False)
        axs[1].set_ylabel(r'\# Grad')
        axs[2] = plot_mice(Log, axs[2], x='iteration', y='rel_error',
                           style='semilogy', legend=False)
        axs[2].set_ylabel(
            r'$\|\nabla_{\boldsymbol{\xi}} \mathcal{F}_k - \nabla_{\boldsymbol{\xi}} F_k\| / \|\nabla_{\boldsymbol{\xi}} F_k\|$')
        axs[2].set_xlabel(r'iteration')
        if eps is not None:
            axs[2].plot(np.full(len(Log), eps), 'k--')
            axs[2].text(1, eps*1.06,
                        r'$\epsilon$', verticalalignment='baseline')
        # PlotObjs += axs.semilogy(np.full(np.size(StatErrGrad),self.epsd),'k--')
        plt.tight_layout()
        pdf.savefig(fig)


def plot_data_logreg(Log, output, eps=None, text=None):
    with PdfPages(output) as pdf:
        if text is not None:
            fig, axs = plt.subplots(1, 1, figsize=(5, 5))
            axs.text(.5, .7, text[0], horizontalalignment='center',
                     verticalalignment='center', wrap=True)
            axs.text(.5, .3, text[1], horizontalalignment='center',
                     verticalalignment='center', wrap=True)
            plt.setp(axs, frame_on=False, xticks=(), yticks=())
            pdf.savefig(fig)

        fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        axs[0] = plot_mice(Log, axs[0], x='num_grads', y='rel_loss')
        axs[0].set_ylabel(r'$(F_k - F^*)/(F_0 - F^*)$')
        axs[1] = plot_mice(Log, axs[1], x='num_grads',
                           y='dist_to_opt', legend=False)
        axs[1].set_ylabel(
            r'$\left\rVert \boldsymbol{\xi}_k - \boldsymbol{\xi}^*\right\rVert$')
        axs[2] = plot_mice(Log, axs[2], x='num_grads',
                           y='grad_norm', legend=False)
        axs[2].set_ylabel(
            r'$\left\rVert\nabla_{\boldsymbol{\xi}} \mathcal{F}_{k}\right\rVert$')
        axs[2].set_xlabel(r'\# Grad')
        plt.tight_layout()
        pdf.savefig(fig)

        fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        axs[0] = plot_mice(Log, axs[0], x='runtime', y='rel_loss')
        axs[0].set_ylabel(r'$(F_k - F^*)/(F_0 - F^*)$')
        axs[1] = plot_mice(Log, axs[1], x='runtime',
                           y='dist_to_opt', legend=False)
        axs[1].set_ylabel(
            r'$\left\rVert \boldsymbol{\xi}_k - \boldsymbol{\xi}^*\right\rVert$')
        axs[2] = plot_mice(Log, axs[2], x='runtime',
                           y='grad_norm', legend=False)
        axs[2].set_ylabel(
            r'$\left\rVert\nabla_{\boldsymbol{\xi}} \mathcal{F}_{k} \right\rVert$')
        axs[2].set_xlabel(r'Runtime (s)')
        fig.tight_layout()
        pdf.savefig(fig)

        fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        axs[0] = plot_mice(Log, axs[0], x='iteration', y='rel_loss',
                           style='semilogy')
        axs[0].set_ylabel(r'$(F_k - F^*)/(F_0 - F^*)$')
        axs[1] = plot_mice(Log, axs[1], x='iteration', y='dist_to_opt',
                           legend=False, style='semilogy')
        axs[1].set_ylabel(
            r'$\left\rVert \boldsymbol{\xi}_k - \boldsymbol{\xi}^*\right\rVert$')
        axs[2] = plot_mice(Log, axs[2], x='iteration', y='grad_norm',
                           legend=False, style='semilogy')
        axs[2].set_ylabel(
            r'$\left\rVert\nabla_{\boldsymbol{\xi}} \mathcal{F}_{k}\right\rVert$')
        axs[2].set_xlabel(r'iteration')
        plt.tight_layout()
        pdf.savefig(fig)

        fig, axs = plt.subplots(1, 1, figsize=(6, 4))
        axs = plot_mice(Log, axs, x='iteration', y='times_iter',
                        style='semilogy', markers=False, color='C0')
        axs = plot_mice(Log, axs, x='iteration', y='grads_time',
                        style='semilogy', markers=False, color='C1')
        axs = plot_mice(Log, axs, x='iteration', y='aggr_time',
                        style='semilogy', markers=False, color='C2')
        axs = plot_mice(Log, axs, x='iteration', y='resampling_time',
                        style='semilogy', markers=False, color='C3')
        axs = plot_mice(Log, axs, x='iteration', y='clipping_time',
                        style='semilogy', markers=False, color='C4')
        axs.set_xlabel('iteration')
        axs.set_ylabel('Runtime (s)')
        axs.legend(('Total time', 'Gradient evaluation', 'Aggregation',
                    'Resampling', 'Clipping'))
        fig.tight_layout()
        pdf.savefig(fig)

        fig, axs = plt.subplots(1, 1, figsize=(6, 4))
        axs = plot_mice(Log, axs, x='iteration', y='grads_time_rel',
                        style='plot', markers=False, color='C1')
        axs = plot_mice(Log, axs, x='iteration', y='aggr_time_rel',
                        style='plot', markers=False, color='C2')
        axs = plot_mice(Log, axs, x='iteration', y='resampling_time_rel',
                        style='plot', markers=False, color='C3')
        axs = plot_mice(Log, axs, x='iteration', y='clipping_time_rel',
                        style='plot', markers=False, color='C4')
        axs.set_xlabel('iteration')
        axs.set_ylabel('Runtime (relative)')
        axs.legend(('Gradient evaluation', 'Aggregation',
                    'Resampling', 'Clipping'))
        fig.tight_layout()
        pdf.savefig(fig)

        fig, axs = plt.subplots(1, 1, figsize=(6, 4))
        axs = plot_mice(Log, axs, x='iteration', y='chain_size', style='plot', legend=False)
        chain_diff = Log['chain_size'].diff()
        clip_iter = Log[(chain_diff < 0) & (Log['event'] != 'restart')].index
        for it in clip_iter:
            axs.plot(Log[it-1:it+1]['iteration'], Log[it-1:it+1]['chain_size'], 'r')
        axs.plot([], [], 'r', label='Clipping')
        axs.legend()
        axs.set_xlabel('iteration')
        axs.set_ylabel(r'$| \mathcal{L}_{k} |$')
        fig.tight_layout()
        pdf.savefig(fig)

        fig, axs = plt.subplots(1, 1, figsize=(6, 4))
        axs = plot_mice(Log, axs, x='iteration', y='times_iter', style='plot')
        axs.set_xlabel('iteration')
        axs.set_ylabel('Time per iteration (s)')
        fig.tight_layout()
        pdf.savefig(fig)

        fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        axs[0] = plot_mice(Log, axs[0], x='iteration',
                           y='vl', style='semilogy')
        axs[0].set_ylabel(r'$V_{\ell}$')
        axs[1] = plot_mice(Log, axs[1], x='iteration', y='num_grads',
                           style='semilogy', legend=False)
        axs[1].set_ylabel(r'\# Grad')
        axs[2] = plot_mice(Log, axs[2], x='iteration', y='rel_error',
                           style='semilogy', legend=False)
        axs[2].set_ylabel(
            r'$\|\nabla_{\boldsymbol{\xi}} \mathcal{F}_k - \nabla_{\boldsymbol{\xi}} F_k\| / \|\nabla_{\boldsymbol{\xi}} F_k\|$')
        axs[2].set_xlabel(r'iteration')
        if eps is not None:
            axs[2].plot(np.full(len(Log), eps), 'k--')
            axs[2].text(1, eps*1.06,
                        r'$\epsilon_{\text{stat}}^{\%}$', verticalalignment='baseline')
        # PlotObjs += axs.semilogy(np.full(np.size(StatErrGrad),self.epsd),'k--')
        plt.tight_layout()
        pdf.savefig(fig)


def plot_data_logreg_lbfgs(Log, output, eps=None, text=None):
    with PdfPages(output) as pdf:
        if text is not None:
            fig, axs = plt.subplots(1, 1, figsize=(5, 5))
            axs.text(.5, .7, text[0], horizontalalignment='center',
                     verticalalignment='center', wrap=True)
            axs.text(.5, .3, text[1], horizontalalignment='center',
                     verticalalignment='center', wrap=True)
            plt.setp(axs, frame_on=False, xticks=(), yticks=())
            pdf.savefig(fig)

        fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        axs[0] = plot_mice(Log, axs[0], x='num_grads', y='rel_loss')
        axs[0].set_ylabel(r'$(F_k - F^*)/(F_0 - F^*)$')
        axs[1] = plot_mice(Log, axs[1], x='num_grads',
                           y='dist_to_opt', legend=False)
        axs[1].set_ylabel(
            r'$\left\rVert \boldsymbol{\xi}_k - \boldsymbol{\xi}^*\right\rVert$')
        axs[2] = plot_mice(Log, axs[2], x='num_grads',
                           y='grad_norm', legend=False)
        axs[2].set_ylabel(
            r'$\left\rVert\nabla_{\boldsymbol{\xi}} \mathcal{F}_{k}\right\rVert$')
        axs[2].set_xlabel(r'\# Grad')
        plt.tight_layout()
        pdf.savefig(fig)

        fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        axs[0] = plot_mice(Log, axs[0], x='runtime', y='rel_loss')
        axs[0].set_ylabel(r'$(F_k - F^*)/(F_0 - F^*)$')
        axs[1] = plot_mice(Log, axs[1], x='runtime',
                           y='dist_to_opt', legend=False)
        axs[1].set_ylabel(
            r'$\left\rVert \boldsymbol{\xi}_k - \boldsymbol{\xi}^*\right\rVert$')
        axs[2] = plot_mice(Log, axs[2], x='runtime',
                           y='grad_norm', legend=False)
        axs[2].set_ylabel(
            r'$\left\rVert\nabla_{\boldsymbol{\xi}} \mathcal{F}_{k} \right\rVert$')
        axs[2].set_xlabel(r'Runtime (s)')
        fig.tight_layout()
        pdf.savefig(fig)

        fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        axs[0] = plot_mice(Log, axs[0], x='iteration', y='rel_loss',
                           style='semilogy')
        axs[0].set_ylabel(r'$(F_k - F^*)/(F_0 - F^*)$')
        axs[1] = plot_mice(Log, axs[1], x='iteration', y='dist_to_opt',
                           legend=False, style='semilogy')
        axs[1].set_ylabel(
            r'$\left\rVert \boldsymbol{\xi}_k - \boldsymbol{\xi}^*\right\rVert$')
        axs[2] = plot_mice(Log, axs[2], x='iteration', y='grad_norm',
                           legend=False, style='semilogy')
        axs[2].set_ylabel(
            r'$\left\rVert\nabla_{\boldsymbol{\xi}} \mathcal{F}_{k}\right\rVert$')
        axs[2].set_xlabel(r'iteration')
        plt.tight_layout()
        pdf.savefig(fig)

        fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        axs[0] = plot_mice(Log, axs[0], x='iteration',
                           y='vl', style='semilogy')
        axs[0].set_ylabel(r'$V_{\ell}$')
        axs[1] = plot_mice(Log, axs[1], x='iteration', y='num_grads',
                           style='semilogy', legend=False)
        axs[1].set_ylabel(r'\# Grad')
        axs[2] = plot_mice(Log, axs[2], x='iteration', y='rel_error',
                           style='semilogy', legend=False)
        axs[2].set_ylabel(
            r'$\|\nabla_{\boldsymbol{\xi}} \mathcal{F}_k - \nabla_{\boldsymbol{\xi}} F_k\| / \|\nabla_{\boldsymbol{\xi}} F_k\|$')
        axs[2].set_xlabel(r'iteration')
        if eps is not None:
            axs[2].plot(np.full(len(Log), eps), 'k--')
            axs[2].text(1, eps*1.06,
                        r'$\epsilon_{\text{stat}}^{\%}$', verticalalignment='baseline')
        # PlotObjs += axs.semilogy(np.full(np.size(StatErrGrad),self.epsd),'k--')
        plt.tight_layout()
        pdf.savefig(fig)

        fig, axs = plt.subplots(1, 1, figsize=(6, 4))
        axs = plot_mice(Log, axs, x='iteration', y='chain_size', style='plot', legend=False)
        # chain_diff = Log['chain_size'].diff()
        # clip_iter = Log[(chain_diff < 0) & (Log['event'] != 'restart')].index
        # for it in clip_iter:
        #     axs.plot(Log[it-1:it+1]['iteration'], Log[it-1:it+1]['chain_size'], 'r')
        # axs.plot([], [], 'r', label='Clipping')
        axs.legend()
        axs.set_xlabel('iteration')
        axs.set_ylabel(r'$| \mathcal{L}_{k} |$')
        fig.tight_layout()
        pdf.savefig(fig)

        fig, axs = plt.subplots(1, 1, figsize=(6, 4))
        axs = plot_mice(Log, axs, x='iteration', y='grads_ls', style='plot',
                        color='C0', markers=False, legend=False)
        axs = plot_mice(Log, axs, x='iteration', y='grads_curv', style='plot',
                        color='C1', markers=False, legend=False)
        axs = plot_mice(Log, axs, x='iteration', y='grads_mice', style='plot',
                        color='C2', markers=False, legend=False)
        axs.set_xlabel('iteration')
        axs.set_ylabel(r'\# Grad (\%)')
        axs.legend(('Line search', 'Curvature check', 'MICE'))
        fig.tight_layout()
        pdf.savefig(fig)

        fig, axs = plt.subplots(1, 1, figsize=(6, 4))
        axs = plot_mice(Log, axs, x='iteration', y='times_iter', style='plot')
        axs.set_xlabel('iteration')
        axs.set_ylabel('Time per iteration (s)')
        fig.tight_layout()
        pdf.savefig(fig)

        fig, axs = plt.subplots(1, 1, figsize=(6, 4))
        axs = plot_mice(Log, axs, x='iteration', y='times_iter',
                        style='semilogy', markers=False, color='C0')
        axs = plot_mice(Log, axs, x='iteration', y='grads_time',
                        style='semilogy', markers=False, color='C1')
        axs = plot_mice(Log, axs, x='iteration', y='aggr_time',
                        style='semilogy', markers=False, color='C2')
        axs = plot_mice(Log, axs, x='iteration', y='resampling_time',
                        style='semilogy', markers=False, color='C3')
        axs = plot_mice(Log, axs, x='iteration', y='clipping_time',
                        style='semilogy', markers=False, color='C4')
        axs.set_xlabel('iteration')
        axs.set_ylabel('Runtime (s)')
        axs.legend(('Total time', 'Gradient evaluation', 'Aggregation',
                    'Resampling', 'Clipping'))
        fig.tight_layout()
        pdf.savefig(fig)

        fig, axs = plt.subplots(1, 1, figsize=(6, 4))
        axs = plot_mice(Log, axs, x='iteration', y='grads_time_rel',
                        style='plot', markers=False, color='C1')
        axs = plot_mice(Log, axs, x='iteration', y='aggr_time_rel',
                        style='plot', markers=False, color='C2')
        axs = plot_mice(Log, axs, x='iteration', y='resampling_time_rel',
                        style='plot', markers=False, color='C3')
        axs = plot_mice(Log, axs, x='iteration', y='clipping_time_rel',
                        style='plot', markers=False, color='C4')
        axs.set_xlabel('iteration')
        axs.set_ylabel('Runtime (relative)')
        axs.legend(('Gradient evaluation', 'Aggregation',
                    'Resampling', 'Clipping'))
        fig.tight_layout()
        pdf.savefig(fig)
