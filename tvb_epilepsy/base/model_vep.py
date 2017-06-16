"""
A module for Virtual Epileptic Patient model classes

class Head
class Connectivity
class Surface
class Sensors
"""
from tvb_epilepsy.base.utils import reg_dict, formal_repr, normalize_weights
from collections import OrderedDict


class Head(object):
    """
    One patient virtualization. Fully configured for defining hypothesis on it.
    """

    def __init__(self, connectivity, cortical_surface, rm, vm, t1, name='',
                 eeg_sensors_dict=None, meg_sensors_dict=None, seeg_sensors_dict=None):

        self.connectivity = connectivity
        self.cortical_surface = cortical_surface
        self.region_mapping = rm
        self.volume_mapping = vm
        self.t1_background = t1

        self.sensorsEEG = eeg_sensors_dict
        self.sensorsMEG = meg_sensors_dict
        self.sensorsSEEG = seeg_sensors_dict

        if len(name) == 0:
            self.name = 'Head' + str(self.number_of_regions)
        else:
            self.name = name

    @property
    def number_of_regions(self):
        return self.connectivity.number_of_regions

    def filter_regions(self, filter_arr):
        return self.connectivity.region_labels[filter_arr]

    def __repr__(self):
        d = {"1. name": self.name,
             "2. connectivity": self.connectivity,
             "5. surface": self.cortical_surface,
             "3. RM": reg_dict(self.region_mapping, self.connectivity.region_labels),
             "4. VM": reg_dict(self.volume_mapping, self.connectivity.region_labels),
             "6. T1": self.t1_background,
             "7. SEEG": self.sensorsSEEG,
             "8. EEG": self.sensorsEEG,
             "9. MEG": self.sensorsMEG }
        return formal_repr(self, OrderedDict(sorted(d.items(), key=lambda t: t[0]) ) )

    def __str__(self):
        return self.__repr__()


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

    def __init__(self, file_path, weights, tract_lengths, labels=None, centers=None, hemispheres=None,
                 orientation=None, areas=None, normalized_weights=None, ):
        self.file_path = file_path
        self.weights = weights
        if normalized_weights is None:
            normalized_weights = normalize_weights(weights)
        self.normalized_weights = normalized_weights
        self.tract_lengths = tract_lengths
        self.region_labels = labels
        self.centers = centers
        self.hemispheres = hemispheres
        self.orientations = orientation
        self.areas = areas

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
        return formal_repr(self, OrderedDict(sorted(d.items(), key=lambda t: t[0]) ) )

    def __str__(self):
        return self.__repr__()

class Surface(object):
    vertices = None
    triangles = None
    vertex_normals = None
    triangle_normals = None

    def __init__(self, vertices, triangles, vertex_normals=None, triangle_normals=None):
        self.vertices = vertices
        self.triangles = triangles
        self.vertex_normals = vertex_normals
        self.triangle_normals = triangle_normals

    def __repr__(self):
        d = {"a. vertices": self.vertices,
             "b. triangles": self.triangles,
             "c. vertex_normals": self.vertex_normals,
             "d. triangle_normals": self.triangle_normals}
        return formal_repr(self, OrderedDict(sorted(d.items(), key=lambda t: t[0]) ) )

    def __str__(self):
        return self.__repr__()


class Sensors(object):
    TYPE_EEG = 'EEG'
    TYPE_MEG = "MEG"
    TYPE_SEEG = "SEEG"

    labels = None
    locations = None
    orientations = None
    s_type = None

    def __init__(self, labels, locations, orientations=None, s_type=TYPE_SEEG):
        self.labels = labels
        self.locations = locations
        self.orientations = orientations
        self.s_type = s_type

    def summary(self):
        d = {"a. sensors type": self.s_type,
             "b. locations": reg_dict(self.locations, self.labels)}
        return formal_repr(self, OrderedDict(sorted(d.items(), key=lambda t: t[0]) ) )


    @property
    def number_of_sensors(self):
        return self.locations.shape[0]

    def __repr__(self):
        d = {"a. sensors type": self.s_type,
             "b. labels": reg_dict(self.labels),
             "c. locations": reg_dict(self.locations, self.labels),
             "d. orientations": reg_dict(self.orientations, self.labels) }
        return formal_repr(self, OrderedDict(sorted(d.items(), key=lambda t: t[0]) ) )

    def __str__(self):
        return self.__repr__()

