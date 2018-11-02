import numpy as np
from tvb.datatypes.connectivity import Connectivity as TVB_Connectivity
from tvb_fit.base.constants import SEEG_DIPOLE_SIGMA
from tvb_fit.base.utils.log_error_utils import initialize_logger

SIGMA = SEEG_DIPOLE_SIGMA


class HeadService(object):
    logger = initialize_logger(__name__)

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

    def vp2tvb_connectivity(self, vp_conn, model_connectivity=None, time_delay_flag=1):
        if model_connectivity is None:
            model_connectivity = vp_conn.normalized_weights
        return TVB_Connectivity(use_storage=False, weights=model_connectivity,
                                tract_lengths=time_delay_flag * vp_conn.tract_lengths,
                                region_labels=vp_conn.region_labels, centres=vp_conn.centres,
                                hemispheres=vp_conn.hemispheres, orientations=vp_conn.orientations, areas=vp_conn.areas)
