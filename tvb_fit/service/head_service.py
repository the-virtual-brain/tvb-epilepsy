import numpy as np

from tvb.datatypes.connectivity import Connectivity as TVB_Connectivity
from tvb_fit.base.model.virtual_patient.connectivity import Connectivity
from tvb_fit.base.model.virtual_patient.sensors import Sensors
from tvb_fit.base.utils.data_structures_utils import ensure_list
from tvb_fit.base.utils.log_error_utils import raise_value_error, initialize_logger
from tvb_fit.base.computations.math_utils import select_greater_values_array_inds, compute_gain_matrix, \
                                                      select_by_hierarchical_group_metric_clustering


class HeadService(object):
    logger = initialize_logger(__name__)

    def compute_nearest_regions_to_sensors(self, head, sensors=None, target_contacts=None, s_type=Sensors.TYPE_SEEG,
                                           sensors_id=0, n_regions=None, gain_matrix_th=None,
                                           gain_matrix_percentile=None):
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
        if n_regions is "all":
            n_regions = head.connectivity.number_of_regions
        nearest_regions = []
        for tc in target_contacts:
            projs = sensors.gain_matrix[tc]
            inds = np.argsort(projs)[::-1]
            n_regions = select_greater_values_array_inds(projs[inds], threshold=gain_matrix_th,
                                                         percentile=gain_matrix_percentile, n_regions=n_regions)
            inds = inds[:n_regions]
            nearest_regions.append((inds, head.connectivity.region_labels[inds], projs[inds]))
        return nearest_regions

    def select_sensors_power(self, sensors, power, selection=[], power_th=None, power_percentile=None, n_sensors=None):
        if len(selection) == 0:
            selection = range(sensors.number_of_sensors)
        return (np.array(selection)[select_greater_values_array_inds(power, power_th, power_percentile,
                                                                     n_sensors)]).tolist()

    def select_sensors_rois(self, sensors, rois=None, initial_selection=[], gain_matrix_th=0.5,
                            gain_matrix_percentile=None, n_rois=None):
        if len(initial_selection) == 0:
            initial_selection = range(sensors.number_of_sensors)
        selection = []
        if sensors.gain_matrix is None:
            raise_value_error("Projection matrix is not set!")
        else:
            for proj in sensors.gain_matrix[initial_selection].T[rois]:
                selection += (np.array(initial_selection)
                [select_greater_values_array_inds(proj, gain_matrix_th, gain_matrix_percentile, n_rois)]).tolist()
        return np.unique(selection).tolist()

    def sensors_in_electrodes_disconnectivity(self, sensors, sensors_labels=[]):
        if len(sensors_labels) < 2:
            sensors_labels = sensors.labels
        n_sensors = len(sensors_labels)
        elec_labels, elec_inds = sensors.group_sensors_to_electrodes(sensors_labels)
        if len(elec_labels) >= 2:
            disconnectivity = np.ones((n_sensors, n_sensors))
            for ch in elec_inds:
                disconnectivity[np.meshgrid(ch, ch)] = 0.0
        return disconnectivity

    def select_sensors_corr(self, sensors, distance, initial_selection=[], n_electrodes=10, sensors_per_electrode=1,
                            power=None, group_electrodes=False):
        if len(initial_selection) == 0:
            initial_selection = range(sensors.number_of_sensors)
        n_sensors = len(initial_selection)
        if n_sensors > 2:
            initial_selection = np.array(initial_selection)
            distance = 1.0 - distance
            if group_electrodes:
                disconnectivity = self.sensors_in_electrodes_disconnectivity(sensors, sensors.labels[initial_selection])
            selection = \
                select_by_hierarchical_group_metric_clustering(distance, disconnectivity, metric=power,
                                                               n_groups=n_electrodes,
                                                               members_per_group=sensors_per_electrode)
            return np.unique(np.hstack(initial_selection[selection])).tolist()
        else:
            self.logger.warning("Number of sensors' left < 2!\n" + "Skipping clustering and returning all of them!")
            return initial_selection

    def compute_gain_matrix(self, head, sensors, normalize=100.0, ceil=False):
        return compute_gain_matrix(sensors.locations, head.connectivity.centres, normalize=normalize, ceil=ceil)

    def vp2tvb_connectivity(self, vp_conn, model_connectivity=None, time_delay_flag=1):
        if model_connectivity is None:
            model_connectivity = vp_conn.normalized_weights
        return TVB_Connectivity(use_storage=False, weights=model_connectivity,
                                tract_lengths=time_delay_flag * vp_conn.tract_lengths,
                                region_labels=vp_conn.region_labels, centres=vp_conn.centres,
                                hemispheres=vp_conn.hemispheres, orientations=vp_conn.orientations, areas=vp_conn.areas)

    def tvb2vp_connectivity(self, tvb_conn, model_connectivity=None, time_delay_flag=1):
        if model_connectivity is None:
            model_connectivity = tvb_conn.weights
        return Connectivity("", model_connectivity, time_delay_flag * tvb_conn.tract_lengths,
                            labels=tvb_conn.region_labels, centres=tvb_conn.centres, hemispheres=tvb_conn.hemispheres,
                            orientations=tvb_conn.orientations, areas=tvb_conn.areas, normalized_weights=np.array([]))


