import numpy as np
from sklearn.cluster import AgglomerativeClustering
from tvb_epilepsy.base.model.vep.sensors import Sensors
from tvb_epilepsy.base.utils.data_structures_utils import ensure_list
from tvb_epilepsy.base.utils.log_error_utils import raise_value_error, warning
from tvb_epilepsy.base.utils.math_utils import select_greater_values_array_inds


class HeadService(object):

    def compute_nearest_regions_to_sensors(self, head, sensors=None, target_contacts=None, s_type=Sensors.TYPE_SEEG,
                                           sensors_id=0, n_regions=None, gain_matrix_th=None):
        if not (isinstance(sensors, Sensors)):
            sensors = head.get_sensors_id(s_type=s_type, sensor_ids=sensors_id)
        n_contacts = sensors.labels.shape[0]
        if isinstance(target_contacts, (list, tuple, np.ndarray)):
            target_contacts = ensure_list(target_contacts)
            for itc, tc in enumerate(target_contacts):
                if isinstance(tc, int):
                    continue
                elif isinstance(tc, basestring):
                    target_contacts[itc] = sensors.contact_label_to_index([tc])
                else:
                    raise_value_error("target_contacts[" + str(itc) + "] = " + str(tc) +
                                      "is neither an integer nor a string!")
        else:
            target_contacts = range(n_contacts)
        auto_flag = False
        if n_regions is "all":
            n_regions = head.connectivity.number_of_regions
        elif not (isinstance(n_regions, int)):
            auto_flag = True
        nearest_regions = []
        for tc in target_contacts:
            projs = sensors.gain_matrix[tc]
            inds = np.argsort(projs)[::-1]
            if auto_flag:
                n_regions = select_greater_values_array_inds(projs[inds], threshold=gain_matrix_th)
            inds = inds[:n_regions]
            nearest_regions.append((inds, head.connectivity.region_labels[inds], projs[inds]))
        return nearest_regions

    def select_sensors_power(self, sensors, power, selection=[], power_th=0.5):
        if len(selection) == 0:
            selection = range(sensors.number_of_sensors)
        return (np.array(selection)[select_greater_values_array_inds(power, power_th)]).tolist()

    def select_sensors_rois(self, sensors, rois=None, initial_selection=[], gain_matrix_th=0.5):
        if len(initial_selection) == 0:
            initial_selection = range(sensors.number_of_sensors)
        selection = []
        if sensors.gain_matrix is None:
            raise_value_error("Projection matrix is not set!")
        else:
            for proj in sensors.gain_matrix[initial_selection].T[rois]:
                selection += (
                    np.array(initial_selection)[select_greater_values_array_inds(proj, gain_matrix_th)]).tolist()
        return np.unique(selection).tolist()

    def select_sensors_corr(self, sensors, distance, initial_selection=[], n_electrodes=10, sensors_per_electrode=1,
                            power=None, group_electrodes=False):
        if len(initial_selection) == 0:
            initial_selection = range(sensors.number_of_sensors)
        n_sensors = len(initial_selection)
        if n_sensors > 2:
            initial_selection = np.array(initial_selection)
            distance = 1.0 - distance
            if group_electrodes:
                elec_labels, elec_inds = sensors.group_sensors_to_electrodes(sensors.labels[initial_selection])
                if len(elec_labels) >= 2:
                    noconnectivity = np.ones((n_sensors, n_sensors))
                    for ch in elec_inds:
                        noconnectivity[np.meshgrid(ch, ch)] = 0.0
                    distance = distance * noconnectivity
            n_electrodes = np.minimum(np.maximum(n_electrodes, 3), n_sensors // sensors_per_electrode)
            clustering = AgglomerativeClustering(n_electrodes, affinity="precomputed", linkage="average")
            clusters_labels = clustering.fit_predict(distance)
            selection = []
            for cluster_id in range(len(np.unique(clusters_labels))):
                cluster_inds = np.where(clusters_labels == cluster_id)[0]
                n_select = np.minimum(sensors_per_electrode, len(cluster_inds))
                if power is not None and len(ensure_list(power)) == n_sensors:
                    inds_select = np.argsort(power[cluster_inds])[-n_select:]
                else:
                    inds_select = range(n_select)
                selection.append(initial_selection[cluster_inds[inds_select]])
            return np.unique(np.hstack(selection)).tolist()
        else:
            warning("Number of sensors' left < 6!\n" + "Skipping clustering and returning all of them!")
            return initial_selection
