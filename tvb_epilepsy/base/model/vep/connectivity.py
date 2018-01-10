import numpy as np
from tvb_epilepsy.base.utils.data_structures_utils import reg_dict, formal_repr, sort_dict, labels_to_inds
from tvb_epilepsy.base.utils.math_utils import normalize_weights


class ConnectivityH5Field():
    WEIGHTS = "weights"
    TRACTS = "tract_lengths"
    CENTERS = "centres"
    REGION_LABELS = "region_labels"
    ORIENTATIONS = "orientations"
    HEMISPHERES = "hemispheres"


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

    def regions_labels2inds(self, labels):
        inds = []
        for lbl in labels:
            inds.append(np.where(self.region_labels == lbl)[0][0])
        if len(inds) == 1:
            return inds[0]
        else:
            return inds

    def get_regions_inds_by_labels(self, lbls):
        return labels_to_inds(self.region_labels, lbls)
