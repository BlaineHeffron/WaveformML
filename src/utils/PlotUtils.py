import itertools
import numpy as np
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
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
    lw = 2
    fig, ax = plt.subplots()
    colors = ['navy', 'red', 'black', 'brown', 'purple', 'aqua', 'darkorange', 'cornflowerblue']
    for i, classd in enumerate(data):
        plt.plot(classd[0], classd[1],
                 label=class_names[i],
                 color=colors[i % 8], linewidth=4)
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    return fig

def plot_pr(data, class_names):
    # Plot all ROC curves
    lw = 2
    fig, ax = plt.subplots()
    colors = ['navy', 'red', 'black', 'brown', 'purple', 'aqua', 'darkorange', 'cornflowerblue']
    for i, classd in enumerate(data):
        plt.plot(classd[1], classd[0],
                 label=class_names[i],
                 color=colors[i % 8], linewidth=4)
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    plt.legend(loc="lower right")
    return fig
