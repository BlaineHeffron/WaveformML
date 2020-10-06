import itertools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import OrderedDict

mpl.use('Agg')

# initialize globals
cmaps = OrderedDict()
tab_colors = ['tab:blue', 'tab:red', 'tab:brown', 'tab:purple', 'tab:orange', 'tab:green', 'tab:grey', 'tab:olive',
              'tab:cyan', 'tab:pink']
# see markers list here https://matplotlib.org/3.2.1/api/markers_api.html
category_markers = ['.', '^', 'o', 'v', 's', 'P', 'x', '*', 'd', 'h', '8', 'D', '|', '1', 'p', '<', 'H', '4']
category_styles = ['-', '--', '--', '-', ':']

# ================================================================================== #
# color maps taken from https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
# ================================================================================== #

cmaps['Sequential'] = [
    'viridis', 'plasma', 'inferno', 'magma', 'cividis']
cmaps['Sequential2'] = [
    'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
    'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
cmaps['SequentialBanding'] = [
    'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
    'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
    'hot', 'afmhot', 'gist_heat', 'copper']
cmaps['Diverging'] = [
    'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
    'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']
cmaps['Cyclic'] = ['twilight', 'twilight_shifted', 'hsv']
cmaps['Qualitative'] = ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                        'Dark2', 'Set1', 'Set2', 'Set3',
                        'tab10', 'tab20', 'tab20b', 'tab20c']
cmaps['Miscellaneous'] = [
    'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
    'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
    'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']


def plot_confusion_matrix(cm, classes, normalize=False, title='', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    if title != '':
        plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig


def plot_countour(X, Y, Z, xlabel, ylabel, zlabel):
    Z = np.transpose(Z)
    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z)
    ax.clabel(CS, inline=True, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("{} distribution".format(zlabel))
    return fig


def plot_bar(X, Y, xlabel, ylabel):
    fig, ax = plt.subplots()
    plt.bar(X, Y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig


def plot_roc(data, class_names):
    # Plot all ROC curves
    lw = 4
    fig, ax = plt.subplots()
    for i, classd in enumerate(data):
        plt.plot(classd[0], classd[1],
                 label=class_names[i],
                 color=tab_colors[i % 10],
                 marker=category_markers[i % len(category_markers)],
                 ls=category_styles[i % len(category_styles)],
                 linewidth=lw)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    return fig


def plot_pr(data, class_names):
    # Plot all ROC curves
    lw = 4
    fig, ax = plt.subplots()
    for i, classd in enumerate(data):
        plt.plot(classd[1], classd[0],
                 label=class_names[i],
                 color=tab_colors[i % 10],
                 marker=category_markers[i % len(category_markers)],
                 ls=category_styles[i % len(category_styles)],
                 linewidth=lw)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    plt.legend(loc="lower right")
    return fig


def plot_wfs(data, n, labels, plot_errors=False):
    lw = 3
    data *= (2 ** 14 - 1)
    if plot_errors:
        errors = np.sqrt(data)
    fig, ax = plt.subplots()
    x = np.arange(2, 600, 4)
    for i, classd in enumerate(data):
        if classd.shape[0] == 2*x.shape[0]:
            classd = classd[:150] + classd[150:]
        tot = n[i]
        if plot_errors:
            plt.bar(x, classd / tot,
                    label=labels[i],
                    color=tab_colors[i % 10],
                    ls=category_styles[i % len(category_styles)],
                    linewidth=lw,
                    yerr=errors[i] / tot)
        else:
            plt.bar(x, classd / tot,
                    label=labels[i],
                    color=tab_colors[i % 10],
                    ls=category_styles[i % len(category_styles)],
                    linewidth=lw)
    ax.set_xlabel('t [ns]')
    ax.set_ylabel('rate [counts/ns]')
    plt.legend(loc="upper right")
    return fig
