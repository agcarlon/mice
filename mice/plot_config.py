import matplotlib.pyplot as plt
import matplotlib


def plot_config():
    # matplotlib.rcParams['text.usetex'] = True
    # matplotlib.rcParams['text.latex.preamble'] = [
    #     r'\usepackage{amsfonts}'
    #     r'\usepackage{amsthm}'
    #     r'\usepackage{amsmath}'
    #     r'\usepackage{amssymb}'][0]
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

    plt.rc('font', family='sans-serif')
