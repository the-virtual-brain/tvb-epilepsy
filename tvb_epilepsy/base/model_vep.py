"""
A module for Virtual Epileptic Patient model classes

class Head
class Connectivity
class Surface
class Sensors
"""
from collections import OrderedDict
import numpy as np

from tvb_epilepsy.base.utils import reg_dict, formal_repr, normalize_weights, calculate_in_degree

from matplotlib import pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tvb_epilepsy.base.constants import FOLDER_FIGURES, LARGE_SIZE, FIG_FORMAT, SAVE_FLAG, SHOW_FLAG
from tvb_epilepsy.base.plot_tools import _plot_vector, _plot_regions2regions, _save_figure, _check_show


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

    def plot(self, show_flag=SHOW_FLAG, save_flag=SAVE_FLAG, figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT):

        # plot connectivity
        self.connectivity.plot(show_flag, save_flag, figure_dir, figure_format)

        self.connectivity.plot_stats(show_flag, save_flag, figure_dir,figure_format)

        # plot sensor projections
        count = plot_sensor_dict(self.sensorsSEEG, self.connectivity.region_labels, 1, show_flag, save_flag,
                                 figure_dir, figure_format)
        count = plot_sensor_dict(self.sensorsEEG, self.connectivity.region_labels, count, show_flag, save_flag,
                                 figure_dir, figure_format)
        count = plot_sensor_dict(self.sensorsMEG, self.connectivity.region_labels, count, show_flag, save_flag,
                                 figure_dir, figure_format)

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

    def plot(self, show_flag=SHOW_FLAG, save_flag=SAVE_FLAG, figure_dir=FOLDER_FIGURES,
                      figure_format=FIG_FORMAT, figure_name='Connectivity ', figsize=LARGE_SIZE):

        # plot connectivity
        pyplot.figure(figure_name + str(self.number_of_regions), figsize)
        # _plot_regions2regions(conn.weights, conn.region_labels, 121, "weights")
        _plot_regions2regions(self.normalized_weights, self.region_labels, 121, "normalised weights")
        _plot_regions2regions(self.tract_lengths, self.region_labels, 122, "tract lengths")

        if save_flag:
            _save_figure(figure_dir=figure_dir, figure_format=figure_format,
                         figure_name=figure_name.replace(" ", "_").replace("\t", "_"))
        _check_show(show_flag=show_flag)

    def plot_stats(self, show_flag=SHOW_FLAG, save_flag=SAVE_FLAG, figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT,
                    figure_name='HeadStats '):
        pyplot.figure("Head stats " + str(self.number_of_regions), figsize=LARGE_SIZE)
        ax = _plot_vector(calculate_in_degree(self.normalized_weights), self.region_labels, 121, "w in-degree")
        ax.invert_yaxis()
        if self.areas is not None:
            ax = _plot_vector(self.areas, self.region_labels, 122, "region areas")
            ax.invert_yaxis()
        if save_flag:
            _save_figure(figure_dir=figure_dir, figure_format=figure_format,
                         figure_name=figure_name.replace(" ", "").replace("\t", ""))
        _check_show(show_flag=show_flag)


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

    def plot(self, projection, region_labels, figure=None, title="Projection", y_labels=1, x_labels=1, x_ticks=None,
             y_ticks=None, show_flag=SHOW_FLAG, save_flag=SAVE_FLAG, figure_dir=FOLDER_FIGURES,
             figure_format=FIG_FORMAT, figure_name=''):

        if not (isinstance(figure, pyplot.Figure)):
            figure = pyplot.figure(title, figsize=LARGE_SIZE)

        n_sensors = self.number_of_sensors
        n_regions = len(region_labels)

        if x_ticks is None:
            x_ticks = np.array(range(n_sensors), dtype=np.int32)
        if y_ticks is None:
            y_ticks = np.array(range(n_regions), dtype=np.int32)

        cmap = pyplot.set_cmap('autumn_r')
        img = pyplot.imshow(projection[x_ticks][:, y_ticks].T, cmap=cmap, interpolation='none')
        pyplot.grid(True, color='black')
        if y_labels > 0:
            region_labels = np.array(["%d. %s" % l for l in zip(range(n_regions), region_labels)])
            pyplot.yticks(y_ticks, region_labels[y_ticks])
        else:
            pyplot.yticks(y_ticks)
        if x_labels > 0:
            sensor_labels = np.array(["%d. %s" % l for l in zip(range(n_sensors), self.labels)])
            pyplot.xticks(x_ticks, sensor_labels[x_ticks], rotation=90)
        else:
            pyplot.xticks(x_ticks)

        ax = figure.get_axes()[0]
        ax.autoscale(tight=True)
        pyplot.title(title)

        divider = make_axes_locatable(ax)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        pyplot.colorbar(img, cax=cax1)  # fraction=0.046, pad=0.04) #fraction=0.15, shrink=1.0

        if figure_name == "":
            figure_name = title

        _save_figure(save_flag, figure_dir=figure_dir, figure_format=figure_format, figure_name=title)
        _check_show(show_flag)

        return figure


def plot_sensor_dict(sensor_dict, region_labels, count=1, show_flag=SHOW_FLAG, save_flag=SAVE_FLAG,
                     figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT):
    # plot sensors:
    for sensors, projection in sensor_dict.sensorsSEEG.iteritems():
        if projection is None:
            continue
        sensors.plot(projection, region_labels, title=str(count) + " - " + sensors.s_type + " - Projection",
                                show_flag=show_flag, save_flag=save_flag, figure_dir=figure_dir,
                                figure_format=figure_format)
        count += 1
        
    return count
            