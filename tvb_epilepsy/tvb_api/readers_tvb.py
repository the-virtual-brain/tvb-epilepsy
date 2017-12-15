"""
Read VEP related entities from TVB format and data-structures
"""
import os

import numpy as np

from tvb.basic.profile import TvbProfile

TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_epilepsy.base.utils.log_error_utils import warning
from tvb_epilepsy.base.utils.data_structures_utils import ensure_list
from tvb_epilepsy.base.model.vep.surface import Surface
from tvb_epilepsy.base.model.vep.sensors import Sensors
from tvb_epilepsy.base.model.vep.connectivity import Connectivity
from tvb_epilepsy.base.model.vep.head import Head
from tvb_epilepsy.base.readers import ABCReader

from tvb.datatypes import connectivity, surfaces, region_mapping, sensors, structural, projections


class TVBReader(ABCReader):
    def read_connectivity(self, path):
        tvb_conn = connectivity.Connectivity.from_file(path)
        return Connectivity(path, tvb_conn.weights, tvb_conn.tract_lengths,
                            tvb_conn.region_labels, tvb_conn.centres,
                            tvb_conn.hemispheres, tvb_conn.orientations, tvb_conn.areas)

    def read_cortical_surface(self, path):
        if os.path.isfile(path):
            tvb_srf = surfaces.CorticalSurface.from_file(path)
            return Surface(tvb_srf.vertices, tvb_srf.triangles,
                           tvb_srf.vertex_normals, tvb_srf.triangle_normals)
        else:
            warning("\nNo Cortical Surface file found at path " + path + "!")
            return []

    def read_region_mapping(self, path):
        if os.path.isfile(path):
            tvb_rm = region_mapping.RegionMapping.from_file(path)
            return tvb_rm.array_data
        else:
            warning("\nNo Region Mapping file found at path " + path + "!")
            return []

    def read_volume_mapping(self, path):
        if os.path.isfile(path):
            tvb_vm = region_mapping.RegionVolumeMapping.from_file(path)
            return tvb_vm.array_data
        else:
            warning("\nNo Volume Mapping file found at path " + path + "!")
            return []

    def read_t1(self, path):
        if os.path.isfile(path):
            tvb_t1 = structural.StructuralMRI.from_file(path)
            return tvb_t1.array_data
        else:
            warning("\nNo Structural MRI file found at path " + path + "!")
            return []

    def read_sensors(self, filename, root_folder, s_type):
        filename = ensure_list(filename)
        path = os.path.join(root_folder, filename[0])
        if os.path.isfile(path):
            if s_type == Sensors.TYPE_EEG:
                tvb_sensors = sensors.SensorsEEG.from_file(path)
            elif s_type == Sensors.TYPE_MEG:
                tvb_sensors = sensors.SensorsMEG.from_file(path)
            else:
                tvb_sensors = sensors.SensorsInternal.from_file(path)
            if len(filename) > 1:
                gain_matrix = self.read_gain_matrix(os.path.join(root_folder, filename[1]), s_type)
            else:
                gain_matrix = np.array([])
            return Sensors(tvb_sensors.labels, tvb_sensors.locations,
                           orientations=tvb_sensors.orientations, gain_matrix=gain_matrix, s_type=s_type)
        else:
            warning("\nNo Sensor file found at path " + path + "!")
            return None

    def read_gain_matrix(self, path, s_type):
        if os.path.isfile(path):
            if s_type == Sensors.TYPE_EEG:
                tvb_prj = projections.ProjectionSurfaceEEG.from_file(path)
            elif s_type == Sensors.TYPE_MEG:
                tvb_prj = projections.ProjectionSurfaceMEG.from_file(path)
            else:
                tvb_prj = projections.ProjectionSurfaceSEEG.from_file(path)
            return tvb_prj.gain_matrix_data
        else:
            warning("\nNo Projection Matrix file found at path " + path + "!")
            return None

    def read_head(self, root_folder, name='',
                  connectivity_file="connectivity.zip",
                  surface_file="surface.zip",
                  region_mapping_file="region_mapping.txt",
                  eeg_sensors_files=[("eeg_brainstorm_65.txt", "gain_matrix_eeg_65_surface_16k.npy")],
                  meg_sensors_files=[("meg_brainstorm_276.txt", "gain_matrix_meg_276_surface_16k.npy")],
                  seeg_sensors_files=[("seeg_588.txt", "gain_matrix_seeg_588_surface_16k.npy")],
                  ):
        conn = self.read_connectivity(os.path.join(root_folder, connectivity_file))
        srf = self.read_cortical_surface(os.path.join(root_folder, surface_file))
        rm = self.read_region_mapping(os.path.join(root_folder, region_mapping_file))
        vm = None
        t1 = None
        sensorsSEEG = []
        for s_files in ensure_list(seeg_sensors_files):
            sensorsSEEG.append(self.read_sensors(s_files, root_folder, Sensors.TYPE_SEEG))
        sensorsEEG = []
        for s_files in ensure_list(eeg_sensors_files):
            sensorsEEG.append(self.read_sensors(s_files, root_folder, Sensors.TYPE_EEG))
        sensorsMEG = []
        for s_files in ensure_list(meg_sensors_files):
            sensorsMEG.append(self.read_sensors(s_files, root_folder, Sensors.TYPE_MEG))
        return Head(conn, srf, rm, vm, t1, name, sensorsSEEG=sensorsSEEG, sensorsEEG=sensorsEEG, sensorsMEG=sensorsMEG)
