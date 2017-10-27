from collections import OrderedDict

import numpy as np
from matplotlib import pyplot

from tvb_epilepsy.base.configurations import FOLDER_FIGURES
from tvb_epilepsy.base.constants import SHOW_FLAG, SAVE_FLAG, FIG_FORMAT, VERY_LARGE_SIZE
from tvb_epilepsy.base.plot_utils import plot_regions2regions, save_figure, check_show, plot_vector
from tvb_epilepsy.base.utils.data_structures_utils import reg_dict, formal_repr, sort_dict
from tvb_epilepsy.base.utils.math_utils import normalize_weights, compute_in_degree


class Connectivity(object):
    file_path = None
    weights = None
    normalized_weights = None
    tract_lengths = None
    region_labels = None
    centers = None
    hemispheres = None
    orientations = None
    areas = None

    def __init__(self, file_path, weights, tract_lengths, labels=np.array([]), centers=np.array([]),
                 hemispheres=np.array([]), orientation=np.array([]), areas=np.array([]),
                 normalized_weights=np.array([])):
        self.file_path = file_path
        self.weights = weights
        if len(normalized_weights) == 0:
            normalized_weights = normalize_weights(weights)
        self.normalized_weights = normalized_weights
        self.tract_lengths = tract_lengths
        self.region_labels = labels
        self.centers = centers
        self.hemispheres = hemispheres
        self.orientations = orientation
        self.areas = areas

    def regions_labels2inds(self, labels):
        inds = []
        for lbl in labels:
            inds.append(np.where(self.region_labels == lbl)[0][0])
        if len(inds)==1:
            return inds[0]
        else:
            return inds

    def summary(self):
        d = {"a. centers": reg_dict(self.centers, self.region_labels),
#             "c. normalized weights": self.normalized_weights,
#             "d. tract_lengths": reg_dict(self.tract_lengths, self.region_labels),
             "b. areas": reg_dict(self.areas, self.region_labels)}
        return formal_repr(self, OrderedDict(sorted(d.items(), key=lambda t: t[0]) ) )

    @property
    def number_of_regions(self):
        return self.centers.shape[0]

    def __repr__(self):
        d = {"f. normalized weights": reg_dict(self.normalized_weights, self.region_labels),
             "g. weights": reg_dict(self.weights, self.region_labels),
             "h. tract_lengths": reg_dict(self.tract_lengths, self.region_labels),
             "a. region_labels": reg_dict(self.region_labels),
             "b. centers": reg_dict(self.centers, self.region_labels),
             "c. hemispheres": reg_dict(self.hemispheres, self.region_labels),
             "d. orientations": reg_dict(self.orientations, self.region_labels),
             "e. areas": reg_dict(self.areas, self.region_labels)}
        return formal_repr(self, sort_dict(d))

    def __str__(self):
        return self.__repr__()

    def plot(self, show_flag=SHOW_FLAG, save_flag=SAVE_FLAG, figure_dir=FOLDER_FIGURES,
                      figure_format=FIG_FORMAT, figure_name='Connectivity ', figsize=VERY_LARGE_SIZE):

        # plot connectivity
        pyplot.figure(figure_name + str(self.number_of_regions), figsize)
        # plot_regions2regions(conn.weights, conn.region_labels, 121, "weights")
        plot_regions2regions(self.normalized_weights, self.region_labels, 121, "normalised weights")
        plot_regions2regions(self.tract_lengths, self.region_labels, 122, "tract lengths")

        if save_flag:
            save_figure(figure_dir=figure_dir, figure_format=figure_format,
                        figure_name=figure_name.replace(" ", "_").replace("\t", "_"))
        check_show(show_flag=show_flag)

    def plot_stats(self, show_flag=SHOW_FLAG, save_flag=SAVE_FLAG, figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT,
                   figsize=VERY_LARGE_SIZE, figure_name='HeadStats '):
        pyplot.figure("Head stats " + str(self.number_of_regions), figsize=figsize)
        areas_flag = len(self.areas) == len(self.region_labels)
        ax = plot_vector(compute_in_degree(self.normalized_weights), self.region_labels, 111 + 10 * areas_flag,
                         "w in-degree")
        ax.invert_yaxis()
        if len(self.areas) == len(self.region_labels):
            ax = plot_vector(self.areas, self.region_labels, 122, "region areas")
            ax.invert_yaxis()
        if save_flag:
            save_figure(figure_dir=figure_dir, figure_format=figure_format,
                        figure_name=figure_name.replace(" ", "").replace("\t", ""))
        check_show(show_flag=show_flag)