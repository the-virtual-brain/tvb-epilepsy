# coding=utf-8

from tvb_fit.tvb_epilepsy.base.constants.config import FiguresConfig
import matplotlib
matplotlib.use(FiguresConfig().MATPLOTLIB_BACKEND)

import numpy

from tvb_fit.tvb_epilepsy.service.lsa_service import LSAService

from tvb_utils.data_structures_utils import dicts_of_lists_to_lists_of_dicts
from tvb_plot.base_plotter import BasePlotter


class LSAPlotter(BasePlotter):

    def __init__(self, config=None):
        super(LSAPlotter, self).__init__(config)

    def plot_lsa_eigen_vals_vectors(self, lsa_service, lsa_hypothesis, region_labels=[]):
        fig_name = lsa_hypothesis.name + " " + "Eigen-values-vectors"
        n_subplots = lsa_service.eigen_vectors_number + 1
        plot_types = ["vector"] * n_subplots
        names = ["LSA eigenvalues"]
        data = [lsa_service.eigen_values]
        n_regions = lsa_hypothesis.number_of_regions
        if len(lsa_service.eigen_values) == 2*n_regions:
            region_labels = numpy.array(["-".join(lbl) for lbl in
                                         (list(zip(n_regions*["x1"], region_labels)) +
                                          list(zip(n_regions*["z"], region_labels)))])
            index_doubling = lambda index: \
                numpy.concatenate([numpy.array(index), numpy.array(index) + lsa_hypothesis.number_of_regions]).tolist()
            eig_vec = lambda v: numpy.log10(numpy.abs(v))
            name_fun = lambda ii: ["log10(abs(LSA eigenvectror " + str(ii + 1) + "))"]
        else:
            index_doubling = lambda index: numpy.array(index).tolist()
            eig_vec = lambda v: v
            name_fun = lambda ii: ["LSA eigenvectror " + str(ii + 1)]
        indices = [[]]
        for ii in range(lsa_service.eigen_vectors_number):
            names += name_fun(ii)
            data += [eig_vec(lsa_service.eigen_vectors[:, ii])]
            indices += [index_doubling(lsa_hypothesis.lsa_propagation_indices)]

        plot_dict_list = dicts_of_lists_to_lists_of_dicts({"name": names, "data": data, "focus_indices": indices,
                                                           "plot_type": plot_types})
        description = "LSA eigenvalues and first " + str(lsa_service.eigen_vectors_number) + " eigenvectors"

        return self.plot_in_columns(plot_dict_list, region_labels, width_ratios=[],
                                    left_ax_focus_indices=index_doubling(lsa_hypothesis.lsa_propagation_indices),
                                    right_ax_focus_indices=index_doubling(lsa_hypothesis.lsa_propagation_indices),
                                    description=description, title=fig_name, figure_name=fig_name)

    def plot_lsa(self, lsa_hypothesis, model_configuration, weighted_eigenvector_sum=None, eigen_vectors_number=None,
                 region_labels=[], pse_results=None, title="Hypothesis Overview", lsa_service=None):
        if isinstance(lsa_service, LSAService):
            f2 = self.plot_lsa_eigen_vals_vectors(lsa_service, lsa_hypothesis, region_labels)
            weighted_eigenvector_sum = lsa_service.weighted_eigenvector_sum
            eigen_vectors_number = lsa_service.eigen_vectors_number
        else:
            f2=None
            if weighted_eigenvector_sum is None:
                weighted_eigenvector_sum = self.config.calcul.WEIGHTED_EIGENVECTOR_SUM

        hyp_dict_list = lsa_hypothesis.prepare_for_plot(model_configuration.connectivity)
        model_config_dict_list = model_configuration.prepare_for_plot()[:2]

        model_config_dict_list += hyp_dict_list
        plot_dict_list = model_config_dict_list

        if pse_results is not None and isinstance(pse_results, dict):
            fig_name = lsa_hypothesis.name + " PSE " + title
            ind_ps = len(plot_dict_list) - 2
            for ii, value in enumerate(["lsa_propagation_strengths", "e_values", "x0_values"]):
                ind = ind_ps - ii
                if ind >= 0:
                    if pse_results.get(value, False).any():
                        plot_dict_list[ind]["data_samples"] = pse_results.get(value)
                        plot_dict_list[ind]["plot_type"] = "vector_violin"

        else:
            fig_name = lsa_hypothesis.name + " " + title

        description = ""
        if weighted_eigenvector_sum:
            description = "LSA PS: absolute eigenvalue-weighted sum of "
            if eigen_vectors_number is not None:
                description += "first " + str(eigen_vectors_number) + " "
            description += "eigenvectors has been used"

        f1 = self.plot_in_columns(plot_dict_list, region_labels, width_ratios=[],
                                  left_ax_focus_indices=lsa_hypothesis.all_disease_indices,
                                  right_ax_focus_indices=lsa_hypothesis.lsa_propagation_indices,
                                  description=description, title=title, figure_name=fig_name)

        if f2:
            return f1, f2
        else:
            return f1
