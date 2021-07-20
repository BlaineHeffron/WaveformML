from typing import List, Union, Tuple

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from src.utils.PlotUtils import plot_hist2d, plot_hist1d, plot_z_acc_matrix
from src.utils.SparseUtils import safe_divide_2d
from src.utils.util import get_bins, safe_divide


def moment_prod(x, counts):
    return np.sum(counts * x[None, :], axis=1) / np.sum(counts, axis=1)


def calc_photon_moments(dist_vec, n):
    output = np.zeros((dist_vec.shape[0], n))
    n_samples = dist_vec.shape[1] / 2
    pulses = dist_vec[:, :n_samples] + dist_vec[:, n_samples:]
    for i in range(n):
        output[:, i] = stats.moment(pulses, moment=i + 2, axis=1)
    return output


def calc_time_moments(dist_vec, n):
    """dist_vec is the vector of distributions, batch index is given by the first index, 1-d distribution along second dimension"""
    output = np.zeros((dist_vec.shape[0], n))
    n_samples = dist_vec.shape[1] / 2
    pulses = dist_vec[:, :n_samples] + dist_vec[:, n_samples:]
    for i in range(n):
        output[:, i] = moment_prod(np.arange(2, n_samples * 4 + 2, 4) ** (i + 2), pulses)
    return output


class StatsAggregator:
    def __init__(self, logger):
        self.metric_metadata = {}
        self.results = {}
        self.logger = logger
        self.colormap = plt.cm.viridis

    def get_metadata(self, name, prop_name, base_name="results"):
        if base_name not in self.metric_metadata:
            raise ValueError("{} not registered to metadata".format(base_name))
        if name not in self.metric_metadata[base_name]:
            raise ValueError("{0} not registered to metadata in base {1}".format(name, base_name))
        if prop_name not in self.metric_metadata[base_name][name]:
            raise ValueError("{0} not registered in metadata in {1}.{2}".format(prop_name, base_name, name))
        return self.metric_metadata[base_name][name][prop_name]

    def set_tuple_metadata(self, name, metadata, meta_name, base_name="results"):
        dim = self.get_metadata(name, "dim", base_name)
        if isinstance(metadata, tuple):
            self.metric_metadata[base_name][name][meta_name] = metadata
        elif isinstance(metadata, bool):
            if metadata:
                self.metric_metadata[base_name][name][meta_name] = tuple(1 for _ in range(dim))
            else:
                self.metric_metadata[base_name][name][meta_name] = tuple(0 for _ in range(dim))
        else:
            raise ValueError("invalid parameter underflow, must be a boolean or tuple")

    def get_tuple_metadata(self, name, meta_name, base_name="results"):
        return self.metric_metadata[base_name][name][meta_name]

    def set_bin_edges(self, name, low_bounds, up_bounds, n, base_name="results"):
        dim = self.metric_metadata[base_name][name]["dim"]
        self.metric_metadata[base_name][name]["bin_edges"] = tuple(
            get_bins(low_bounds[i], up_bounds[i], n[i]) for i in range(dim))

    def register_duplicates(self, names, n_bins: Union[Tuple, List], lower_bounds: Union[Tuple, List],
                            upper_bounds: Union[Tuple, List], dim, dim_names: Union[Tuple, List],
                            dim_units: Union[Tuple, List], metric_name, metric_units, base_name="results",
                            underflow=True, overflow=True, scale=1.0):
        for name in names:
            self.register_aggregator(name, n_bins, lower_bounds, upper_bounds, dim, dim_names, dim_units, metric_name,
                                     metric_units, base_name, underflow, overflow, scale)

    def register_aggregator(self, name, n_bins: Union[Tuple, List], lower_bounds: Union[Tuple, List],
                            upper_bounds: Union[Tuple, List], dim, dim_names: Union[Tuple, List],
                            dim_units: Union[Tuple, List], metric_name, metric_units, base_name="results",
                            underflow=True, overflow=True, scale=1.0):
        if not hasattr(self, base_name):
            setattr(self, base_name, {})
        if base_name not in self.metric_metadata:
            self.metric_metadata[base_name] = {}
        if name not in self.metric_metadata[base_name]:
            self.metric_metadata[base_name][name] = {"dim": dim, "n_bins": n_bins, "dim_names": dim_names,
                                                     "dim_units": dim_units, "metric_units": metric_units,
                                                     "metric_name": metric_name, "scale": scale}
        else:
            raise ValueError("property {0} has already been registered to {1}".format(name, base_name))
        self.set_tuple_metadata(name, underflow, "underflow", base_name)
        self.set_tuple_metadata(name, overflow, "overflow", base_name)
        underflow = self.get_tuple_metadata(name, "underflow", base_name)
        overflow = self.get_tuple_metadata(name, "overflow", base_name)
        self.set_bin_edges(name, lower_bounds, upper_bounds, n_bins, base_name)
        getattr(self, base_name)[name] = (np.zeros(tuple(n_bins[i] + underflow[i] + overflow[i] for i in range(dim)),
                                                   dtype=np.float32),
                                          np.zeros(tuple(n_bins[i] + underflow[i] + overflow[i] for i in range(dim)),
                                                   dtype=np.int32))

    def get_plot_metadata(self, name, base_name="results"):
        labels = []
        for n, unit in zip(
                self.metric_metadata[base_name][name]["dim_names"],
                self.metric_metadata[base_name][name]["dim_units"]):
            if unit == "":
                labels.append("{0}".format(n))
            else:
                labels.append("{0} [{1}]".format(n, unit))
        return self.metric_metadata[base_name][name]["dim"], \
               self.metric_metadata[base_name][name]["scale"], \
               self.metric_metadata[base_name][name]["bin_edges"], \
               labels, self.metric_metadata[base_name][name]["dim_units"], \
               self.metric_metadata[base_name][name]["metric_name"], \
               self.metric_metadata[base_name][name]["metric_units"], \
               self.metric_metadata[base_name][name]["n_bins"], \
               self.metric_metadata[base_name][name]["dim_names"]

    def get_plot_ranges(self, name, base_name="results"):
        under = self.metric_metadata[base_name][name]["underflow"]
        dim = self.metric_metadata[base_name][name]["dim"]
        n_bins = self.metric_metadata[base_name][name]["n_bins"]
        lower = []
        upper = []
        for i in range(dim):
            if under[i]:
                lower.append(1)
                upper.append(n_bins[i] + 1)
            else:
                lower.append(0)
                upper.append(n_bins[i])
        return lower, upper

    def increment_metric(self, name, results, bin_indices, base_name="results"):
        if len(results.shape) != 1:
            raise ValueError("results must be a 1 dimensional array")
        if self.metric_metadata[base_name][name]["dim"] == 1:
            getattr(self, base_name)[name][1][bin_indices[0]] += results.shape[0]
            getattr(self, base_name)[name][0][bin_indices[0]] += np.sum(results)
        if self.metric_metadata[base_name][name]["dim"] == 2:
            getattr(self, base_name)[name][1][bin_indices[0], bin_indices[1]] += results.shape[0]
            getattr(self, base_name)[name][0][bin_indices[0], bin_indices[1]] += np.sum(results)
        elif self.metric_metadata[base_name][name]["dim"] == 3:
            getattr(self, base_name)[name][1][bin_indices[0], bin_indices[1], bin_indices[2]] += results.shape[0]
            getattr(self, base_name)[name][0][bin_indices[0], bin_indices[1], bin_indices[2]] += np.sum(results)
        elif self.metric_metadata[base_name][name]["dim"] == 4:
            getattr(self, base_name)[name][1][bin_indices[0], bin_indices[1], bin_indices[2], bin_indices[3]] += \
                results.shape[0]
            getattr(self, base_name)[name][0][bin_indices[0], bin_indices[1], bin_indices[2], bin_indices[3]] += np.sum(
                results)
        else:
            raise ValueError("dim > 4 not supported ")

    def log_total(self, name, log_name, plot_title, base_name="results"):
        if np.max(getattr(self, base_name)[name][1]) <= 0:
            return
        dim, _, bin_edges, labels, units, _, _, n_bins, dim_names = self.get_plot_metadata(name, base_name)
        low, up = self.get_plot_ranges(name, base_name)
        if dim == 1:
            if units[0]:
                ylabel = "total [" + units[0] + r"$^{-1}$]"
            else:
                ylabel = "total"
            fig = plot_hist1d(bin_edges, getattr(self, base_name)[name][1][low[0]:up[0]], plot_title, labels[0], ylabel,
                              norm_to_bin_width=True)
        elif dim == 2:
            if units[0] and units[1]:
                zlabel = "total [" + units[0] + r"$^{-1}$" + units[1] + r"$^{-1}$]"
            elif units[0]:
                zlabel = "total [" + units[0] + r"$^{-1}$]"
            elif units[1]:
                zlabel = "total [" + units[1] + r"$^{-1}$]"
            else:
                zlabel = "total"
            fig = plot_hist2d(bin_edges[0], bin_edges[1],
                              getattr(self, base_name)[name][1][low[0]:up[0], low[1]:up[1]],
                              plot_title, labels[0], labels[1], zlabel, cm=self.colormap)
        elif dim == 3:
            bm = self.get_bin_midpoints(name, 2, base_name)
            if units[0] and units[1]:
                zlabel = "total [" + units[0] + r"$^{-1}$" + units[1] + r"$^{-1}$]"
            elif units[0]:
                zlabel = "total [" + units[0] + r"$^{-1}$]"
            elif units[1]:
                zlabel = "total [" + units[1] + r"$^{-1}$]"
            else:
                zlabel = "total"
            for i in range(n_bins[2]):
                fig = plot_hist2d(bin_edges[0], bin_edges[1],
                                  getattr(self, base_name)[name][1][low[0]:up[0], low[1]:up[1], i],
                                  plot_title, labels[0], labels[1], zlabel, cm=self.colormap)
                bm = self.get_bin_midpoints(name, 2, base_name)
                log_name = "{0} = {1}".format(dim_names[2], bm[i])
                self.logger.experiment.add_figure(log_name, fig)
        else:
            raise ValueError("no method to plot dim > 3")
        if dim < 3:
            self.logger.experiment.add_figure(log_name, fig)

    def log_metric(self, name, log_name, plot_title, base_name="results"):
        if np.max(getattr(self, base_name)[name][1]) <= 0:
            return
        dim, scale, bin_edges, labels, units, \
        metric_name, metric_units, n_bins, dim_names = self.get_plot_metadata(name, base_name)
        low, up = self.get_plot_ranges(name, base_name)
        if metric_units:
            label = "{0} [{1}]".format(metric_name, metric_units)
        else:
            label = metric_name
        if dim == 1:
            fig = plot_hist1d(bin_edges, scale * safe_divide(getattr(self, base_name)[name][0][low[0]:up[0]],
                                                             getattr(self, base_name)[name][1][low[0]:up[0]]),
                              plot_title, labels[0], label,
                              norm_to_bin_width=True)
        elif dim == 2:
            fig = plot_hist2d(bin_edges[0], bin_edges[1],
                              scale * safe_divide_2d(getattr(self, base_name)[name][0][low[0]:up[0], low[1]:up[1]],
                                                     getattr(self, base_name)[name][1][low[0]:up[0], low[1]:up[1]]),
                              plot_title, labels[0], labels[1], label, cm=self.colormap, norm_to_bin_width=False,
                              logz=False)

        elif dim == 3:
            bm = self.get_bin_midpoints(name, 2, base_name)
            for i in range(n_bins[2]):
                self.logger.experiment.add_figure(log_name + "_{}".format(i),
                                                  plot_z_acc_matrix(
                                                      scale * safe_divide_2d(
                                                          getattr(self, base_name)[name][0][:, :, i],
                                                          getattr(self, base_name)[name][1][:, :, i]),
                                                      n_bins[0], n_bins[1], "{0} = {1}".format(dim_names[2], bm[i]),
                                                      zlabel=label))
        else:
            raise ValueError("no method to plot dim > 3")
        if dim < 3:
            self.logger.experiment.add_figure(log_name, fig)

    def get_bin_midpoints(self, name, dim, base_name="results"):
        bin_edges = self.metric_metadata[base_name][name]["bin_edges"][dim]
        half = (bin_edges[1] - bin_edges[0]) / 2.
        return [be + half for be in bin_edges[0:-1]]

    def log_segment_metric(self, name, log_name, base_name="results"):
        if np.max(getattr(self, base_name)[name][1]) <= 0:
            return
        dim, scale, bin_edges, labels, units, \
        metric_name, metric_units, n_bins, dim_names = self.get_plot_metadata(name, base_name)
        if metric_units:
            label = "{0} [{1}]".format(metric_name, metric_units)
        else:
            label = metric_name
        if dim == 2:
            self.logger.experiment.add_figure(log_name,
                                              plot_z_acc_matrix(
                                                  scale * safe_divide_2d(
                                                      getattr(self, base_name)[name][0],
                                                      getattr(self, base_name)[name][1]),
                                                  n_bins[0], n_bins[1], dim_names[2], zlabel=label))
        elif dim == 3:
            bm = self.get_bin_midpoints(name, 2, base_name)
            for i in range(n_bins[2]):
                self.logger.experiment.add_figure(log_name + "_{}".format(i),
                                                  plot_z_acc_matrix(
                                                      scale * safe_divide_2d(
                                                          getattr(self, base_name)[name][0][:, :, i],
                                                          getattr(self, base_name)[name][1][:, :, i]),
                                                      n_bins[0], n_bins[1], "{0} = {1}".format(dim_names[2], bm[i]),
                                                      zlabel=label))
