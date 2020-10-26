import itertools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from math import ceil, floor
from collections import OrderedDict

from src.utils.util import safe_divide

mpl.use('Agg')
plt.rcParams['font.size'] = '12'
TITLE_SIZE = 16

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
    fig.subplots_adjust(bottom=0.18)
    return fig


def plot_n_contour(X, Y, Z, xlabel, ylabel, title, suptitle=None):
    n_categories = len(title)
    nrows = ceil((n_categories - 1) / 3)
    fig_height = 4.0
    if (n_categories < 3):
        fig, axes = plt.subplots(ceil(n_categories / 3), n_categories, figsize=(fig_height * 3.9, fig_height * nrows))
    else:
        fig, axes = plt.subplots(ceil(n_categories / 3), 3, figsize=(fig_height * 3.9, fig_height * nrows))
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=TITLE_SIZE)
    for z, t, i in zip(Z, title, range(n_categories)):
        z = np.transpose(z)
        # axes[int(floor(i / 3)), i % 3].clabel(CS, inline=True)
        if n_categories < 4:
            CS = axes[i].contourf(X, Y, z, cmap=plt.cm.BrBG)
            axes[i].set_title(t, fontsize=TITLE_SIZE)
            if i == 0:
                axes[i].set_ylabel(ylabel)
            else:
                axes[i].tick_params(axis='y', labelcolor='w')
            axes[i].set_xlabel(xlabel)
            plt.colorbar(CS, ax=axes[i])
        else:
            CS = axes[int(floor(i / 3)), i % 3].contourf(X, Y, z, cmap=plt.cm.BrBG)
            axes[int(floor(i / 3)), i % 3].set_title(t, fontsize=TITLE_SIZE)
            if i % 3 == 0:
                axes[int(floor(i / 3)), i % 3].set_ylabel(ylabel)
            else:
                axes[int(floor(i / 3)), i % 3].tick_params(axis='y', labelcolor='w')
            if floor(i / 3) == floor((n_categories - 1) / 3):
                axes[int(floor(i / 3)), i % 3].set_xlabel(xlabel)
            else:
                axes[int(floor(i / 3)), i % 3].tick_params(axis='x', labelcolor='w')
            plt.colorbar(CS, ax=axes[floor(i / 3), i % 3])
    i = 0
    for ax in fig.get_axes():
        if i == n_categories:
            break
        ax.label_outer()
        i += 1
    if n_categories < 4:
        fig.subplots_adjust(bottom=0.18)
    return fig


def plot_contour(X, Y, Z, xlabel, ylabel, title):
    Z = np.transpose(Z)
    fig, ax = plt.subplots()
    CS = ax.contourf(X, Y, Z, cmap=plt.cm.BrBG)
    ax.clabel(CS, inline=True, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=TITLE_SIZE)
    return fig


def plot_bar(X, Y, xlabel, ylabel):
    fig, ax = plt.subplots()
    plt.bar(X, Y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig


def plot_n_hist1d(xedges, vals, title, xlabel, ylabel, suptitle=None, norm_to_bin_width=True):
    n_categories = len(title)
    nrows = ceil((n_categories - 1) / 3)
    fig_height = 4.0
    if (n_categories < 3):
        fig, axes = plt.subplots(ceil(n_categories / 3), n_categories, figsize=(fig_height * 3.9, fig_height * nrows))
    else:
        fig, axes = plt.subplots(ceil(n_categories / 3), 3, figsize=(fig_height * 3.9, fig_height * nrows))
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=TITLE_SIZE)
    xwidth = xedges[1] - xedges[0]
    for m in range(n_categories):
        if norm_to_bin_width:
            vals[m] = vals[m].astype(np.float32)
            vals[m] *= (1. / xwidth)
        tot = vals[m].shape[0]
        w = np.zeros((tot,))
        xs = np.zeros((tot,))
        n = 0
        for i in range(len(xedges) - 1):
            x = xwidth * i + xwidth / 2.
            w[n] = vals[m][i]
            xs[n] = x
            n += 1
        if (n_categories < 4):
            axes[m].hist(xs, bins=xedges, weights=w)
            axes[m].set_xlabel(xlabel)
            if m == 0:
                axes[m].set_ylabel(ylabel)
            axes[m].set_title(title[m], fontsize=TITLE_SIZE)
        else:
            axes[floor(m / 3), m % 3].hist(xs, bins=xedges, weights=w)
            if floor(m / 3) == floor((n_categories - 1) / 3):
                axes[floor(m / 3), m % 3].set_xlabel(xlabel)
            if m % 3 == 0:
                axes[floor(m / 3), m % 3].set_ylabel(ylabel)
            axes[floor(m / 3), m % 3].set_title(title[m], fontsize=TITLE_SIZE)
    i = 0
    for ax in fig.get_axes():
        if i == n_categories:
            break
        ax.label_outer()
        i += 1
    if n_categories < 4:
        fig.subplots_adjust(bottom=0.18)
    # cb.set_label(zlabel, rotation=270)
    return fig


def plot_n_hist2d(xedges, yedges, vals, title, xlabel, ylabel, suptitle=None, norm_to_bin_width=True, logz = True):
    n_categories = len(title)
    nrows = ceil((n_categories - 1) / 3)
    fig_height = 4.0
    if n_categories < 3:
        fig, axes = plt.subplots(ceil(n_categories / 3), n_categories, figsize=(fig_height * 3.9, fig_height * nrows))
    else:
        fig, axes = plt.subplots(ceil(n_categories / 3), 3, figsize=(fig_height * 3.9, fig_height * nrows))
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=TITLE_SIZE)
    xwidth = xedges[1] - xedges[0]
    ywidth = yedges[1] - yedges[0]
    for m in range(n_categories):
        if norm_to_bin_width:
            vals[m] = vals[m].astype(np.float32)
            vals[m] *= 1. / (xwidth * ywidth)
        tot = vals[m].shape[0] * vals[m].shape[1]
        w = np.zeros((tot,))
        xs = np.zeros((tot,))
        ys = np.zeros((tot,))
        n = 0
        for i in range(len(xedges) - 1):
            x = xwidth * i + xwidth / 2.
            for j in range(len(yedges) - 1):
                y = ywidth * j + ywidth / 2.
                w[n] = vals[m][i, j]
                xs[n] = x
                ys[n] = y
                n += 1
        if n_categories < 4:
            if logz:
                h = axes[m].hist2d(xs, ys, bins=[xedges, yedges], weights=w, cmap=plt.cm.BrBG, norm=LogNorm())
            else:
                h = axes[m].hist2d(xs, ys, bins=[xedges, yedges], weights=w, cmap=plt.cm.BrBG)
            if floor(m / 3) == floor((n_categories - 1) / 3):
                axes[m].set_xlabel(xlabel)
            else:
                axes[m].tick_params(axis='x', labelcolor='w')
            if m == 0:
                axes[m].set_ylabel(ylabel)
            else:
                axes[m].tick_params(axis='y', labelcolor='w')
            axes[m].set_title(title[m], fontsize=TITLE_SIZE)
            plt.colorbar(h[3], ax=axes[m])
        else:
            if logz:
                h = axes[floor(m / 3), m % 3].hist2d(xs, ys, bins=[xedges, yedges], weights=w, cmap=plt.cm.BrBG, norm=LogNorm())
            else:
                h = axes[floor(m / 3), m % 3].hist2d(xs, ys, bins=[xedges, yedges], weights=w, cmap=plt.cm.BrBG)
            if floor(m / 3) == floor((n_categories - 1) / 3):
                axes[floor(m / 3), m % 3].set_xlabel(xlabel)
            else:
                axes[floor(m / 3), m % 3].tick_params(axis='x', labelcolor='w')
            if m % 3 == 0:
                axes[floor(m / 3), m % 3].set_ylabel(ylabel)
            else:
                axes[floor(m / 3), m % 3].tick_params(axis='y', labelcolor='w')
            axes[floor(m / 3), m % 3].set_title(title[m], fontsize=TITLE_SIZE)
            plt.colorbar(h[3], ax=axes[floor(m / 3), m % 3])
    i = 0
    for ax in fig.get_axes():
        if i == n_categories:
            break
        ax.label_outer()
        i += 1
    # cb.set_label(zlabel, rotation=270)
    if n_categories < 4:
        fig.subplots_adjust(bottom=0.18)
    return fig


def plot_hist2d(xedges, yedges, vals, title, xlabel, ylabel, zlabel, norm_to_bin_width=True, logz=True):
    fig, ax = plt.subplots()
    xwidth = xedges[1] - xedges[0]
    ywidth = yedges[1] - yedges[0]
    if norm_to_bin_width:
        vals = vals.astype(np.float32)
        vals *= 1. / (xwidth * ywidth)
    tot = vals.shape[0] * vals.shape[1]
    w = np.zeros((tot,))
    xs = np.zeros((tot,))
    ys = np.zeros((tot,))
    n = 0
    for i in range(len(xedges) - 1):
        x = xwidth * i + xwidth / 2.
        for j in range(len(yedges) - 1):
            y = ywidth * j + ywidth / 2.
            w[n] = vals[i, j]
            xs[n] = x
            ys[n] = y
            n += 1
    if logz:
        h = plt.hist2d(xs, ys, bins=[xedges, yedges], weights=w, cmap=plt.cm.BrBG, norm=LogNorm())
    else:
        h = plt.hist2d(xs, ys, bins=[xedges, yedges], weights=w, cmap=plt.cm.BrBG)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=TITLE_SIZE)
    plt.colorbar(h[3])
    # cb.set_label(zlabel, rotation=270)
    return fig


def plot_hist1d(xedges, vals, title, xlabel, ylabel, norm_to_bin_width=True):
    fig, ax = plt.subplots()
    xwidth = xedges[1] - xedges[0]
    if norm_to_bin_width:
        vals = vals.astype(np.float32)
        vals /= xwidth
    tot = vals.shape[0] * vals.shape[1]
    xs = np.zeros((tot,))
    n = 0
    for i in range(len(xedges) - 1):
        x = xwidth * i + xwidth / 2.
        xs[n] = x
        n += 1
    h = plt.hist(xs, bins=xedges, weights=vals, cmap=plt.cm.BrBG)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=TITLE_SIZE)
    plt.colorbar(h[3])
    return fig


def plot_roc(data, class_names):
    # Plot all ROC curves
    lw = 4
    fig, ax = plt.subplots()
    for i, classd in enumerate(data):
        plt.plot(classd[0], classd[1],
                 label=class_names[i],
                 color=tab_colors[i % 10],
                 # marker=category_markers[i % len(category_markers)],
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
                 # marker=category_markers[i % len(category_markers)],
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
    lw = 2
    data *= (2 ** 14 - 1)
    fig, ax = plt.subplots()
    x = np.arange(2, 600, 4)
    for i in range(len(labels)):
        if data.shape[1] == 2 * x.shape[0]:
            y = data[i, :150] + data[i, 150:]
        else:
            y = data[i]
        tot = n[i]
        if plot_errors:
            errors = np.sqrt(y)
            plt.errorbar(x, safe_divide(y, tot),
                         label=labels[i],
                         color=tab_colors[i % 10],
                         ls=category_styles[i % len(category_styles)],
                         linewidth=lw,
                         yerr=safe_divide(errors[i], tot))
        else:
            plt.plot(x, safe_divide(y, tot),
                     label=labels[i],
                     color=tab_colors[i % 10],
                     ls=category_styles[i % len(category_styles)],
                     linewidth=lw)
    ax.set_xlabel('t [ns]')
    ax.set_ylabel('rate [counts/ns]')
    plt.legend(loc="upper right")
    return fig
