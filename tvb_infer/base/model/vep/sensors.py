# coding=utf-8
from enum import Enum
import re
import numpy as np
from tvb_infer.base.utils.data_structures_utils import reg_dict, formal_repr, sort_dict, ensure_list, \
                                                                                    labels_to_inds, monopolar_to_bipolar
from tvb_infer.base.utils.data_structures_utils import split_string_text_numbers
from tvb_infer.base.computations.math_utils import compute_gain_matrix


# SDE model inversion constants
class SensorTypes(Enum):
    TYPE_EEG = 'EEG'
    TYPE_MEG = "MEG"
    TYPE_SEEG = "SEEG"


class SensorsH5Field(object):
    GAIN_MATRIX = "gain_matrix"
    LABELS = "labels"
    LOCATIONS = "locations"
    NEEDLES = "needles"


class Sensors(object):
    TYPE_EEG = SensorTypes.TYPE_EEG.value
    TYPE_MEG = SensorTypes.TYPE_MEG.value
    TYPE_SEEG = SensorTypes.TYPE_SEEG.value

    labels = np.array([])
    locations = np.array([])
    needles = np.array([])
    orientations = np.array([])
    gain_matrix = np.array([])
    s_type = TYPE_SEEG

    def __init__(self, labels, locations, needles=np.array([]), orientations=np.array([]), gain_matrix=np.array([]),
                 s_type=TYPE_SEEG):
        self.labels = labels
        self.locations = locations
        self.needles = needles
        self.channel_labels = np.array([])
        self.orientations = orientations
        self.gain_matrix = gain_matrix
        self.s_type = s_type
        self.elec_labels = np.array([])
        self.elec_inds = np.array([])
        if len(self.labels) > 1:
            self.elec_labels, self.elec_inds = self.group_sensors_to_electrodes()
            if self.needles.size == self.number_of_sensors:
                self.channel_labels, self.channel_inds = self.get_inds_labels_from_needles()
            else:
                self.channel_labels, self.channel_inds = self.group_sensors_to_electrodes()
                self.get_needles_from_inds_labels()

    @property
    def number_of_sensors(self):
        return self.locations.shape[0]

    def __repr__(self):
        d = {"1. sensors' type": self.s_type,
             "2. number of sensors": self.number_of_sensors,
             "3. labels": reg_dict(self.labels),
             "4. locations": reg_dict(self.locations, self.labels),
             "5. orientations": reg_dict(self.orientations, self.labels),
             "6. gain_matrix": self.gain_matrix}
        return formal_repr(self, sort_dict(d))

    def __str__(self):
        return self.__repr__()

    # TODO: verify this try and change message
    def sensor_label_to_index(self, labels):
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

    def get_sensors_inds_by_sensors_labels(self, lbls):
        # Make sure that the labels are not bipolar:
        lbls = [label.split("-")[0] for label in ensure_list(lbls)]
        return labels_to_inds(self.labels, lbls)

    def get_elecs_inds_by_elecs_labels(self, lbls):
        return labels_to_inds(self.elec_labels, lbls)

    def get_sensors_inds_by_elec_labels(self, lbls):
        elec_inds = self.get_elecs_inds_by_elecs_labels(lbls)
        sensors_inds = []
        for ind in elec_inds:
            sensors_inds += self.elec_inds[ind]
        return np.unique(sensors_inds)

    def compute_gain_matrix(self, connectivity, normalize=95, ceil=False):
        return compute_gain_matrix(self.locations, connectivity.centres, normalize=normalize, ceil=ceil)

    def get_inds_labels_from_needles(self):
        channel_inds = []
        channel_labels = []
        for id in np.unique(self.needles):
            inds = np.where(self.needles == id)[0]
            channel_inds.append(inds)
            label = split_string_text_numbers(self.labels[inds[0]])[0][0]
            channel_labels.append(label)
        return channel_labels, channel_inds

    def get_needles_from_inds_labels(self):
        self.needles = np.zeros((self.number_of_sensors,), dtype="i")
        for id, inds in enumerate(self.channel_inds):
            self.needles[inds] = id

    def group_sensors_to_electrodes(self, labels=None):
        if labels is None:
            labels = self.labels
        sensor_names = np.array(split_string_text_numbers(labels))
        elec_labels = np.unique(sensor_names[:, 0])
        elec_inds = []
        for chlbl in elec_labels:
            elec_inds.append(np.where(sensor_names[:, 0] == chlbl)[0])
        return elec_labels, elec_inds

    def get_bipolar_sensors(self, sensors_inds=None):
        if sensors_inds is None:
            sensors_inds = range(self.number_of_sensors)
        # bipolar_sensors_lbls = []
        # bipolar_sensors_inds = []
        # for ind in range(len(sensors_inds) - 1):
        #     iS1 = sensors_inds[ind]
        #     iS2 = sensors_inds[ind + 1]
        #     if (self.labels[iS1][0] == self.labels[iS2][0]) and \
        #             int(re.findall(r'\d+', self.labels[iS1])[0]) == \
        #             int(re.findall(r'\d+', self.labels[iS2])[0]) - 1:
        #         bipolar_sensors_lbls.append(self.labels[iS1] + "-" + self.labels[iS2])
        #         bipolar_sensors_inds.append(iS1)
        # return bipolar_sensors_inds, bipolar_sensors_lbls
        return monopolar_to_bipolar(self.labels, sensors_inds)

    def get_bipolar_elecs(self, elecs):
        try:
            bipolar_sensors_lbls = []
            bipolar_sensors_inds = []
            for elec_ind in elecs:
                curr_inds, curr_lbls = self.get_bipolar_sensors(sensors_inds=self.elec_inds[elec_ind])
                bipolar_sensors_inds.append(curr_inds)
                bipolar_sensors_lbls.append(curr_lbls)
        except:
            elecs_inds = self.get_elecs_inds_by_elecs_labels(elecs)
            bipolar_sensors_inds, bipolar_sensors_lbls = self.get_bipolar_elecs(elecs_inds)
        return bipolar_sensors_inds, bipolar_sensors_lbls
