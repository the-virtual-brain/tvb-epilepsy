from collections import OrderedDict

import numpy as np
from matplotlib import pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable

from tvb_epilepsy.base.configurations import FOLDER_FIGURES
from tvb_epilepsy.base.constants import SHOW_FLAG, SAVE_FLAG, FIG_FORMAT, VERY_LARGE_SIZE
from tvb_epilepsy.base.plot_utils import save_figure, check_show
from tvb_epilepsy.base.utils.data_structures_utils import reg_dict, formal_repr, sort_dict
from tvb_epilepsy.base.utils.math_utils import compute_projection


class Sensors(object):
    TYPE_EEG = 'EEG'
    TYPE_MEG = "MEG"
    TYPE_SEEG = "SEEG"

    labels = np.array([])
    locations = np.array([])
    orientations = np.array([])
    s_type = ''

    def __init__(self, labels, locations, orientations=np.array([]), s_type=TYPE_SEEG):
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
        return formal_repr(self, sort_dict(d))

    def __str__(self):
        return self.__repr__()

    def contact_label_to_index(self, labels):
        indexes = []
        for label in labels:
            try:
                indexes.append(np.where([np.array(lbl) == np.array(label) for lbl in self.labels])[0][0])
            except:
                print("WTF")
        if len(indexes) == 1:
            return indexes[0]
        else:
            return indexes

    def calculate_projection(self, connectivity):
        return compute_projection(self.locations, connectivity.centers, normalize=95, ceil=False)

    def plot(self, projection, region_labels, figure=None, title="Projection", y_labels=1, x_labels=1,
             x_ticks=np.array([]), y_ticks=np.array([]), show_flag=SHOW_FLAG, save_flag=SAVE_FLAG,
             figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT, figsize=VERY_LARGE_SIZE, figure_name=''):

        if not (isinstance(figure, pyplot.Figure)):
            figure = pyplot.figure(title, figsize=figsize)

        n_sensors = self.number_of_sensors
        n_regions = len(region_labels)

        if len(x_ticks) == 0:
            x_ticks = np.array(range(n_sensors), dtype=np.int32)
        if len(y_ticks) == 0:
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

        save_figure(save_flag, figure_dir=figure_dir, figure_format=figure_format, figure_name=title)
        check_show(show_flag)

        return figure


def plot_sensor_dict(sensor_dict, region_labels, count=1, show_flag=SHOW_FLAG, save_flag=SAVE_FLAG,
                     figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT):
    # plot sensors:
    for sensors, projection in sensor_dict.iteritems():
        if len(projection) == 0:
            continue
        sensors.plot(projection, region_labels, title=str(count) + " - " + sensors.s_type + " - Projection",
                                show_flag=show_flag, save_flag=save_flag, figure_dir=figure_dir,
                                figure_format=figure_format)
        count += 1

    return count