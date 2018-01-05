from collections import OrderedDict
import numpy as np
from tvb_epilepsy.base.utils.data_structures_utils import reg_dict, formal_repr, sort_dict, labels_to_inds, \
    construct_import_path
from tvb_epilepsy.base.utils.math_utils import normalize_weights
from tvb_epilepsy.base.h5_model import convert_to_h5_model


class Connectivity(object):
    file_path = None
    weights = None
    normalized_weights = None
    tract_lengths = None
    region_labels = None
    centres = None
    hemispheres = None
    orientations = None
    areas = None

    def __init__(self, file_path, weights, tract_lengths, labels=np.array([]), centres=np.array([]),
                 hemispheres=np.array([]), orientation=np.array([]), areas=np.array([]),
                 normalized_weights=np.array([])):
        self.file_path = file_path
        self.weights = weights
        if len(normalized_weights) == 0:
            normalized_weights = normalize_weights(weights, remove_diagonal=True, ceil=1.0)
        self.normalized_weights = normalized_weights
        self.tract_lengths = tract_lengths
        self.region_labels = labels
        self.centres = centres
        self.hemispheres = hemispheres
        self.orientations = orientation
        self.areas = areas
        self.context_str = "from " + construct_import_path(__file__) + " import Connectivity"
        self.create_str = "Connectivity('" + self.file_path + "', np.array([]), np.array([]))"

    def regions_labels2inds(self, labels):
        inds = []
        for lbl in labels:
            inds.append(np.where(self.region_labels == lbl)[0][0])
        if len(inds) == 1:
            return inds[0]
        else:
            return inds

    def summary(self):
        d = {"a. centres": reg_dict(self.centres, self.region_labels),
             #             "c. normalized weights": self.normalized_weights,
             #             "d. tract_lengths": reg_dict(self.tract_lengths, self.region_labels),
             "b. areas": reg_dict(self.areas, self.region_labels)}
        return formal_repr(self, OrderedDict(sorted(d.items(), key=lambda t: t[0])))

    @property
    def number_of_regions(self):
        return self.centres.shape[0]

    def __repr__(self):
        d = {"f. normalized weights": reg_dict(self.normalized_weights, self.region_labels),
             "g. weights": reg_dict(self.weights, self.region_labels),
             "h. tract_lengths": reg_dict(self.tract_lengths, self.region_labels),
             "a. region_labels": reg_dict(self.region_labels),
             "b. centres": reg_dict(self.centres, self.region_labels),
             "c. hemispheres": reg_dict(self.hemispheres, self.region_labels),
             "d. orientations": reg_dict(self.orientations, self.region_labels),
             "e. areas": reg_dict(self.areas, self.region_labels)}
        return formal_repr(self, sort_dict(d))

    def __str__(self):
        return self.__repr__()

    def _prepare_for_h5(self, connectivity_variants=False):
        h5_model = convert_to_h5_model(self)
        h5_model.add_or_update_metadata_attribute("EPI_Type", "Connectivity")
        h5_model.add_or_update_metadata_attribute("EPI_Version", "1")
        h5_model.add_or_update_metadata_attribute("Number_of_regions", str(self.weights.shape[0]))
        if connectivity_variants:
            del h5_model.datasets_dict["/normalized_weights"]
            h5_model.add_or_update_datasets_attribute("/normalized_weights/weights",
                                                      self.normalized_weights)
            h5_model.add_or_update_metadata_attribute("/normalized_weights/Operations",
                                                      "[Removing diagonal, normalizing with 95th percentile, "
                                                      "and ceiling to it]")
        return h5_model

    def write_to_h5(self, folder, filename="", connectivity_variants=False):
        if filename == "":
            filename = self.name + ".h5"
        h5_model = self._prepare_for_h5(connectivity_variants)
        h5_model.write_to_h5(folder, filename)

    def get_regions_inds_by_labels(self, lbls):
        return labels_to_inds(self.region_labels, lbls)
