import itertools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
from matplotlib import rcParams
import matplotlib.dates as mdate
from pytz import timezone
from math import ceil, floor
from collections import OrderedDict

from src.utils.util import safe_divide, write_x_y_csv

mpl.use('Agg')
plt.rcParams['font.size'] = '12'
TITLE_SIZE = 16

# initialize globals
cmaps = OrderedDict()
tab_colors = ['tab:blue', 'tab:red', 'tab:brown', 'tab:purple', 'black', 'tab:green', 'tab:grey', 'tab:olive',
              'tab:cyan', 'tab:pink', 'tab:orange']
tab_colors = ['#377eb8', '#ff7f00', '#4daf4a','#a65628', '#984ea3',
                  '#f781bf',
                  '#999999', '#e41a1c', '#dede00']
# see markers list here https://matplotlib.org/3.2.1/api/markers_api.html
category_markers = ['.', '^', 'o', 'v', 's', 'P', 'x', '*', 'd', 'h', '8', 'D', '|', '1', 'p', '<', 'H', '4']
category_styles = ['-', '--', '-.', ':']

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
cmaps['Qualitative'] = ['Pastel1', 'Pasel2', 'Paired', 'Accent',
                        'Dark2', 'Set1', 'Set2', 'Set3',
                        'tab10', 'tab20', 'tab20b', 'tab20c']
cmaps['Miscellaneous'] = [
    'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
    'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
    'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']


def plot_z_acc_matrix(cm, nx, ny, title, zlabel="mean average error [mm]", cmap=plt.cm.viridis):
    fontsize = 12
    fig = plt.figure(figsize=(9, 6.5))
    min = 10000000
    for i in range(nx):
        for j in range(ny):
            if min > cm[i, j] > 0:
                min = cm[i,j]
    cm = cm.transpose()
    cm = np.ma.masked_where(cm == 0, cm)
    cmap.set_bad(color='black')
    plt.imshow(cm, interpolation='nearest', cmap=cmap, origin="lower")
    if title != '':
        plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label(zlabel, labelpad=18)
    tick_x = np.arange(nx)
    tick_y = np.arange(ny)
    tick_labelx = np.arange(1, nx + 1)
    tick_labely = np.arange(1, ny + 1)
    plt.xticks(tick_x, tick_labelx)
    plt.yticks(tick_y, tick_labely)
    fmt = '.0f'
    thresh = (cm.max() + min) / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j] != 0:
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                     color="black" if cm[i, j] > thresh else "white", fontsize=fontsize)

    plt.tight_layout()
    plt.ylabel('y segment')
    plt.xlabel('x segment')
    fig.subplots_adjust(left=0.1)
    return fig


def plot_confusion_matrix(cm, classes, normalize=False, title='', cmap=plt.cm.Blues):
    fontsize = 12
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fontsize = 16
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    if title != '':
        plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=fontsize)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig.subplots_adjust(bottom=0.24)
    return fig


def plot_n_contour(X, Y, Z, xlabel, ylabel, title, suptitle=None, cm=plt.cm.viridis):
    n_categories = len(title)
    nrows = ceil(n_categories / 3)
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
            CS = axes[i].contourf(X, Y, z, cmap=cm)
            axes[i].set_title(t, fontsize=TITLE_SIZE)
            if i == 0:
                axes[i].set_ylabel(ylabel)
            else:
                axes[i].tick_params(axis='y', labelcolor='w')
            axes[i].set_xlabel(xlabel)
            plt.colorbar(CS, ax=axes[i])
        else:
            CS = axes[int(floor(i / 3)), i % 3].contourf(X, Y, z, cmap=cm)
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


def plot_contour(X, Y, Z, xlabel, ylabel, title, filled=True, cm=plt.cm.viridis):
    Z = np.transpose(Z)
    fig, ax = plt.subplots()
    if filled:
        CS = ax.contourf(X, Y, Z, cmap=cm)
        plt.colorbar(CS, ax=ax)
    else:
        CS = ax.contour(X, Y, Z, cmap=cm)
        ax.clabel(CS, inline=True)
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


def plot_n_hist1d(xedges, vals, title, xlabel, ylabel, suptitle=None, norm_to_bin_width=True, logy=True):
    n_categories = len(title)
    nrows = ceil(n_categories / 3)
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
            x = xedges[0] + xwidth * i + xwidth / 2.
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
            else:
                axes[floor(m / 3), m % 3].tick_params(axis='x', labelcolor='w')
            if m % 3 == 0:
                axes[floor(m / 3), m % 3].set_ylabel(ylabel)
            axes[floor(m / 3), m % 3].set_title(title[m], fontsize=TITLE_SIZE)
    i = 0
    if logy:
        for ax in fig.get_axes():
            if i == n_categories:
                break
            # ax.label_outer()
            ax.set_yscale('log')
            i += 1
    if n_categories < 4:
        fig.subplots_adjust(bottom=0.18)
    # cb.set_label(zlabel, rotation=270)
    return fig


def plot_n_hist2d(xedges, yedges, vals, title, xlabel, ylabel, suptitle=None, norm_to_bin_width=True, logz=True,
                  cm=plt.cm.viridis):
    n_categories = len(title)
    nrows = ceil(n_categories / 3)
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
            x = xedges[0] + xwidth * i + xwidth / 2.
            for j in range(len(yedges) - 1):
                y = yedges[0] + ywidth * j + ywidth / 2.
                if vals[m][i, j] <= 0 and logz:
                    w[n] = 1. / (xwidth * ywidth) if norm_to_bin_width else 1.
                else:
                    w[n] = vals[m][i, j]
                xs[n] = x
                ys[n] = y
                n += 1
        if n_categories < 4:
            if logz:
                h = axes[m].hist2d(xs, ys, bins=[xedges, yedges], weights=w, cmap=cm, norm=LogNorm())
            else:
                h = axes[m].hist2d(xs, ys, bins=[xedges, yedges], weights=w, cmap=cm)
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
                h = axes[floor(m / 3), m % 3].hist2d(xs, ys, bins=[xedges, yedges], weights=w, cmap=cm, norm=LogNorm())
            else:
                h = axes[floor(m / 3), m % 3].hist2d(xs, ys, bins=[xedges, yedges], weights=w, cmap=cm)
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


def plot_hist2d(xedges, yedges, vals, title, xlabel, ylabel, zlabel, norm_to_bin_width=True, logz=True,
                cm=plt.cm.viridis):
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
        x = xedges[0] + xwidth * i + xwidth / 2.
        for j in range(len(yedges) - 1):
            y = yedges[0] + ywidth * j + ywidth / 2.
            w[n] = vals[i, j]
            xs[n] = x
            ys[n] = y
            n += 1
    if logz:
        h = plt.hist2d(xs, ys, bins=[xedges, yedges], weights=w, cmap=cm, norm=LogNorm())
    else:
        h = plt.hist2d(xs, ys, bins=[xedges, yedges], weights=w, cmap=cm)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=TITLE_SIZE)
    cb = plt.colorbar(h[3])
    if(zlabel):
        cb.set_label(zlabel, rotation=270, labelpad=20)
    return fig


def plot_hist1d(xedges, vals, title, xlabel, ylabel, norm_to_bin_width=True, logy=True):
    fig, ax = plt.subplots()
    xwidth = xedges[1] - xedges[0]
    if norm_to_bin_width:
        vals = vals.astype(np.float32)
        vals /= xwidth
    tot = vals.shape[0]
    xs = np.zeros((tot,))
    n = 0
    for i in range(len(xedges) - 1):
        x = xedges[0] + xwidth * i + xwidth / 2.
        xs[n] = x
        n += 1
    h = plt.hist(xs, bins=xedges, weights=vals)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale('log')
    ax.set_title(title, fontsize=TITLE_SIZE)
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


def plot_wfs(data, n, labels, plot_errors=False, normalize=False, write_pulses=False):
    lw = 2
    data *= (2 ** 14 - 1)
    dlen = data.shape[1]
    fig, ax = plt.subplots()
    x = np.arange(2, dlen*2, 4)
    for i in range(len(labels)):
        if data.shape[1] == 2 * x.shape[0]:
            y = data[i, :int(dlen/2)] + data[i, int(dlen/2):]
        else:
            y = data[i]
        tot = n[i]
        y = safe_divide(y, tot)
        if normalize:
            y = safe_divide(y, sum(y))
        if plot_errors:
            errors = np.sqrt(y)
            plt.errorbar(x, y,
                         label=labels[i],
                         color=tab_colors[i % 10],
                         ls=category_styles[i % len(category_styles)],
                         linewidth=lw,
                         yerr=safe_divide(errors[i], tot))
        else:
            plt.plot(x, y,
                     label=labels[i],
                     color=tab_colors[i % 10],
                     ls=category_styles[i % len(category_styles)],
                     linewidth=lw)
        if write_pulses:
            write_x_y_csv("pulse_{}.csv".format(labels[i]), "time [ns]", "counts", x, y)

    ax.set_xlabel('t [ns]')
    ax.set_ylabel('rate [counts/ns]')
    plt.legend(loc="upper right")
    return fig


def GetMPLStyles():
    style_list = []
    for i in range(20):
        style_list.append(tab_colors[i%10] + category_styles[i%len(category_styles)])
    return style_list


def ScatterPlt(xaxis,yvals,xlabel,ylabel,outname=None,title=None,errbar=None, marker='o', ylog=False, ignore_zeros=False):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    if ignore_zeros:
        yvals[yvals == 0] = np.nan
    if(title):
        ax1.set_title(title)
    if(errbar):
        ax1.errorbar(xaxis,yvals,yerr=errbar,fmt='')
    else:
        ax1.scatter(xaxis,yvals,marker=marker)
    if(ylog):
        ax1.set_yscale('log')
    if outname is not None:
        plt.savefig(outname)
        plt.close()
    return fig

def MultiScatterPlot(xaxis, yvals, errors, line_labels, xlabel, ylabel,
                     colors=None, styles=None, ignore_zeros=False,
                     xmax=-1, ymax=-1, ymin=None, xmin=None, ylog=True, xdates=False,
                     vertlines=None, vlinelabel=None, xlog=False, title=None,  figsize=(12, 9)):
    if colors is None:
        colors = []
    if styles is None:
        styles = []
    if ignore_zeros:
        for i in range(len(yvals)):
            yvals[i][yvals[i] == 0] = np.nan
            yvals[i][errors[i] == 0] = np.nan
            errors[i][errors[i] == 0] = np.nan
    if(xdates):
        xaxis = mdate.epoch2num(xaxis)
        if(vertlines):
            vertlines = mdate.epoch2num(vertlines)
    rcParams.update({'font.size': 18})
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    if(ymin is None):
        if(ylog):
            ymin = min([min(y)  for y in yvals])*0.5
            if(ymin <= 0):
                print("error: ymin is ", ymin, " on a log-y axis. Defaulting to 1e-5")
                ymin = 1e-5
        else:
            ymin = min([min(y)  for y in yvals])
            if ymin < 0: ymin *= 1.05
            else: ymin *= .95
    if(xmin is None):
        xmin = min(xaxis)
    if(xlog):
        ax1.set_xscale('log')
    else:
        ax1.set_xscale('linear')
    if(ylog):
        ax1.set_yscale('log')
    else:
        ax1.set_yscale('linear')
    if(ymax == -1):
        if(ylog):
            ax1.set_ylim(ymin,max([max(y) for y in yvals])*1.5)
        else:
            ax1.set_ylim(ymin,max([max(y) for y in yvals])*1.05)
    else:
        ax1.set_ylim(ymin,ymax)
    if(xmax == -1):
        ax1.set_xlim(xmin,max(xaxis))
    else:
        ax1.set_xlim(xmin,xmax)
    #for i, y in enumerate(yvals):
    if not colors:
        colors = tab_colors
    if not styles:
        styles = category_styles
    for i in range(len(yvals)):
        ax1.errorbar(xaxis,yvals[i],yerr=errors[i],color=colors[i%10],fmt=category_markers[i%len(category_markers)])
    if(vertlines is not None):
        for v in vertlines:
            ax1.axvline(v,color='k',linestyle='-')#,label=vlinelabel)
    if(xdates):
        #date_fmt = '%y-%m-%d %H:%M:%S'
        date_fmt = '%y-%m-%d'
        date_formatter = mdate.DateFormatter(date_fmt,tz=timezone('US/Eastern'))
        ax1.xaxis.set_major_formatter(date_formatter)
        fig.autofmt_xdate()
    else:
        pass
        #start,end = ax1.get_xlim()
        #diff = (end - start) / 8.
        #ax1.xaxis.set_ticks(np.arange(start,end,diff))
        #ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
        #plt.setp(ax1.get_xticklabels(), rotation=30,\
        #horizontalalignment='right')
    box = ax1.get_position()
    if(len(yvals) == 1):
        if(not title):
            ax1.set_title(line_labels[0])
        else:
            ax1.set_title(title)
    else:
        if(title):
            ax1.set_title(title)
        ax1.set_position([box.x0,box.y0,box.width,box.height])
        ax1.legend(line_labels,loc='center left', \
                   bbox_to_anchor=(0.5,0.85),ncol=1)
        rcParams.update({'font.size':14})
    #plt.gcf().subplots_adjust(left=0.16)
    #plt.gcf().subplots_adjust(bottom=0.22)
    #plt.gcf().subplots_adjust(right=0.05)
    #plt.savefig(outname)
    #plt.close()
    return fig


def MultiLinePlot(xaxis, yvals, line_labels, xlabel, ylabel,
                  colors=None, styles=None,
                  xmax=-1, ymax=-1, ymin=None, xmin=None, ylog=True, xdates=False,
                  vertlines=None, vlinelabel=None, xlog=False, title=None,
                  width_factor=0.9, legend_xoff=0.4, legend_yoff=0.75, ignore_zeros=False):
    if colors is None:
        colors = []
    if styles is None:
        styles = []
    if(xdates):
        xaxis = mdate.epoch2num(xaxis)
        if(vertlines):
            vertlines = mdate.epoch2num(vertlines)
    if ignore_zeros:
        for i in range(len(yvals)):
            yvals[i][yvals[i] == 0] = np.nan
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    if(ymin is None):
        if(ylog):
            ymin = min([min(y)  for y in yvals])*0.5
            if(ymin <= 0):
                print("error: ymin is ", ymin, " on a log-y axis. Defaulting to 1e-5")
                ymin = 1e-5
        else:
            ymin = min([min(y)  for y in yvals])*0.95
    if(xmin is None):
        xmin = min(xaxis)
    if(xlog):
        ax1.set_xscale('log')
    else:
        ax1.set_xscale('linear')
    if(ylog):
        ax1.set_yscale('log')
    else:
        ax1.set_yscale('linear')
    if(ymax == -1):
        if(ylog):
            ax1.set_ylim(ymin,max([max(y) for y in yvals])*1.5)
        else:
            ax1.set_ylim(ymin,max([max(y) for y in yvals])*1.05)
    else:
        ax1.set_ylim(ymin,ymax)
    if(xmax == -1):
        ax1.set_xlim(xmin,max(xaxis))
    else:
        ax1.set_xlim(xmin,xmax)
    #for i, y in enumerate(yvals):
    if not colors:
        colors = tab_colors
    if not styles:
        styles = category_styles
    for i in range(len(yvals)):
        ax1.plot(xaxis,yvals[i],color=colors[i%10],linestyle=styles[i%len(styles)])
    if(vertlines is not None):
        for v in vertlines:
            ax1.axvline(v,color='k',linestyle='-')#,label=vlinelabel)
    if(xdates):
        #date_fmt = '%y-%m-%d %H:%M:%S'
        date_fmt = '%y-%m-%d'
        date_formatter = mdate.DateFormatter(date_fmt,tz=timezone('US/Eastern'))
        ax1.xaxis.set_major_formatter(date_formatter)
        fig.autofmt_xdate()
    else:
        pass
        #start,end = ax1.get_xlim()
        #diff = (end - start) / 8.
        #ax1.xaxis.set_ticks(np.arange(start,end,diff))
        #ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
        #plt.setp(ax1.get_xticklabels(), rotation=30,\
        #horizontalalignment='right')
    box = ax1.get_position()
    if(len(yvals) == 1):
        if(not title):
            ax1.set_title(line_labels[0])
        else:
            ax1.set_title(title)
    else:
        if(title):
            ax1.set_title(title)
        ax1.set_position([box.x0,box.y0,box.width*width_factor,box.height])
        ax1.legend(line_labels,loc='center left',\
                bbox_to_anchor=(legend_xoff,legend_yoff),ncol=1)
        rcParams.update({'font.size':18})
    #plt.gcf().subplots_adjust(left=0.16)
    #plt.gcf().subplots_adjust(bottom=0.22)
    #plt.gcf().subplots_adjust(right=0.05)
    #plt.savefig(outname)
    #plt.close()
    return fig

def gen_animation(figures, outfile):
    frames = []
    fig = plt.figure()
    for i in range(len(figures)):
        frames.append([figures[i]])
    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
                                    repeat_delay=1000)
    ani.save(outfile)
    plt.clf()
    plt.close('all')