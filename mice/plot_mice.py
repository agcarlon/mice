try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    raise Exception("'matplotlib' is not installed. Install matplotlib to use "
                    "plot_mice")
import matplotlib
from shutil import which as find_executable




def plot_mice(data,
              ax,
              x,
              y,
              style='loglog',
              markers=True,
              legend=True,
              color='C0'):
    """
    Plot MICE logs on the given Matplotlib axes.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing optimization log data, including event markers.
    ax : matplotlib.axes.Axes
        Axes object where the data will be plotted.
    x : str
        Column name used for the x-axis.
    y : str
        Column name used for the y-axis.
    style : {'loglog', 'semilogy', 'semilogx', 'plot'}
        Plot style controlling linear/log scaling of axes.
    markers : bool
        If True, adds event markers (start, add, dropped, restart, end).
    legend : bool
        If True, adds a legend.
    color : str
        Line color passed to Matplotlib.

    Returns
    -------
    matplotlib.axes.Axes
        The updated axes.
    """
    start = data[data['event'] == 'start']
    adds = data[data['event'] == 'add']
    dropped = data[data['event'] == 'dropped']
    restarts = data[data['event'] == 'restart']
    end = data[data['event'] == 'end']

    drop = dropped.size > 0
    plot_style = {
        'loglog': ax.loglog,
        'semilogy': ax.semilogy,
        'semilogx': ax.semilogx,
        'plot': ax.plot
    }
    plot_func = plot_style[style]
    plots = []
    plots += [plot_func(data[x], data[y], color=color)]
    if markers:
        plots += [plot_func(start[x], start[y], 'bs', ms=5, label='Start')]
        if drop:
            plots += [plot_func(dropped[x], dropped[y],
                                'kx', ms=3, label='Dropped')]
        plots += [plot_func(adds[x], adds[y], 'co', ms=3, label='Add')]
        plots += [plot_func(restarts[x], restarts[y],
                            'rs', ms=5, label='Restart')]
        plots += [plot_func(end[x], end[y], 's',
                            color='purple', ms=5, label='End')]
    if legend and markers:
        # ax.legend(plots[1:])
        ax.legend()
    return ax


def _plot_config():
    if find_executable('latex'):
        matplotlib.rcParams['text.usetex'] = True
        matplotlib.rcParams['text.latex.preamble'] = [
            r'\usepackage{amsfonts}'
            r'\usepackage{amsthm}'
            r'\usepackage{amsmath}'
            r'\usepackage{amssymb}'][0]
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = 'Computer Modern'
    matplotlib.rcParams['xtick.labelsize'] = 12
    matplotlib.rcParams['ytick.labelsize'] = 12
    matplotlib.rcParams.update({'font.size': 12})
    matplotlib.rcParams['axes.linewidth'] = 1
    matplotlib.rcParams['axes.facecolor'] = 'white'
    matplotlib.rcParams['axes.grid'] = True
    matplotlib.rcParams['grid.linestyle'] = '--'
    matplotlib.rcParams['grid.color'] = 'grey'
    matplotlib.rcParams['grid.linewidth'] = 0.5
    matplotlib.rc('axes', edgecolor='black')

    # plt.rc('font', family='sans-serif')


_plot_config()
