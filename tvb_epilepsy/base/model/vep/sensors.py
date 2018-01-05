from collections import OrderedDict
import re
import numpy as np
from tvb_epilepsy.base.utils.data_structures_utils import reg_dict, formal_repr, sort_dict, labels_to_inds, \
    split_string_text_numbers, construct_import_path
from tvb_epilepsy.base.utils.math_utils import compute_gain_matrix
from tvb_epilepsy.base.h5_model import convert_to_h5_model


class Sensors(object):
    TYPE_EEG = 'EEG'
    TYPE_MEG = "MEG"
    TYPE_SEEG = "SEEG"
    SENSORS_TYPES = [TYPE_SEEG, TYPE_MEG, TYPE_EEG]

    labels = np.array([])
    locations = np.array([])
    needles = np.array([])
    orientations = np.array([])
    gain_matrix = np.array([])
    s_type = ''

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
        self.context_str = "from " + construct_import_path(__file__) + " import Sensors"
        self.create_str = "Sensors(np.array([]), np.array([]), s_type='" + self.s_type + "')"

    def summary(self):
        d = {"1. sensors type": self.s_type,
             "2. locations": reg_dict(self.locations, self.labels)}
        return formal_repr(self, OrderedDict(sorted(d.items(), key=lambda t: t[0])))

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

    def _prepare_for_h5(self):
        h5_model = convert_to_h5_model(self)
        h5_model.add_or_update_metadata_attribute("EPI_Type", "Sensors")
        h5_model.add_or_update_metadata_attribute("EPI_Version", "1")
        h5_model.add_or_update_metadata_attribute("Number_of_sensors", str(self.number_of_sensors))
        h5_model.add_or_update_metadata_attribute("Sensors_subtype", self.s_type)
        h5_model.add_or_update_metadata_attribute("/gain_matrix/Min", str(self.gain_matrix.min()))
        h5_model.add_or_update_metadata_attribute("/gain_matrix/Max", str(self.gain_matrix.max()))
        return h5_model

    def write_to_h5(self, folder, filename=""):
        if filename == "":
            filename = self.name + ".h5"
        h5_model = self._prepare_for_h5()
        h5_model.write_to_h5(folder, filename)

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
        return labels_to_inds(self.labels, lbls)

    def get_elecs_inds_by_elecs_labels(self, lbls):
        return labels_to_inds(self.elec_labels, lbls)

    def get_sensors_inds_by_elec_labels(self, lbls):
        elec_inds = self.get_elecs_inds_by_elecs_labels(lbls)
        sensors_inds = []
        for ind in elec_inds:
            sensors_inds += self.elec_inds[ind]
        return np.unique(sensors_inds)

    # TODO: duplicated+not used
    def calculate_projection(self, connectivity):
        return compute_gain_matrix(self.locations, connectivity.centers, normalize=95, ceil=1.0)

    def compute_gain_matrix(self, connectivity):
        return compute_gain_matrix(self.locations, connectivity.centres, normalize=95, ceil=1.0)

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
        bipolar_sensors_lbls = []
        bipolar_sensors_inds = []
        for ind in range(len(sensors_inds) - 1):
            iS1 = sensors_inds[ind]
            iS2 = sensors_inds[ind + 1]
            if (self.labels[iS1][0] == self.labels[iS2][0]) and \
                    int(re.findall(r'\d+', self.labels[iS1])[0]) == \
                    int(re.findall(r'\d+', self.labels[iS2])[0]) - 1:
                bipolar_sensors_lbls.append(self.labels[iS1] + "-" + self.labels[iS2])
                bipolar_sensors_inds.append(iS1)
        return bipolar_sensors_inds, bipolar_sensors_lbls

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
