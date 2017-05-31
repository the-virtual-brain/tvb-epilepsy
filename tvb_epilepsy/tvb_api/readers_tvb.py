"""
Read VEP related entities from TVB format and data-structures
"""

from tvb.basic.profile import TvbProfile

TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

import os
from tvb.datatypes import connectivity, surfaces, region_mapping, sensors, structural, projections
from tvb_epilepsy.base.model_vep import Connectivity, Surface, Sensors, Head
from tvb_epilepsy.base.readers import ABCReader


class TVBReader(ABCReader):
    def read_connectivity(self, path):
        tvb_conn = connectivity.Connectivity.from_file(path)
        return Connectivity(path, tvb_conn.weights, tvb_conn.tract_lengths,
                            tvb_conn.region_labels, tvb_conn.centres,
                            tvb_conn.hemispheres, tvb_conn.orientations, tvb_conn.areas)

    def read_cortical_surface(self, path):
        tvb_srf = surfaces.CorticalSurface.from_file(path)
        return Surface(tvb_srf.vertices, tvb_srf.triangles,
                       tvb_srf.vertex_normals, tvb_srf.triangle_normals)

    def read_region_mapping(self, path):
        tvb_rm = region_mapping.RegionMapping.from_file(path)
        return tvb_rm.array_data

    def read_volume_mapping(self, path):
        tvb_vm = region_mapping.RegionVolumeMapping.from_file(path)
        return tvb_vm.array_data

    def read_t1(self, path):
        tvb_t1 = structural.StructuralMRI.from_file(path)
        return tvb_t1.array_data

    def read_sensors(self, path, s_type):
        if s_type == Sensors.TYPE_EEG:
            tvb_sensors = sensors.SensorsEEG.from_file(path)
        elif s_type == Sensors.TYPE_MEG:
            tvb_sensors = sensors.SensorsMEG.from_file(path)
        else:
            tvb_sensors = sensors.SensorsInternal.from_file(path)
        return Sensors(tvb_sensors.labels, tvb_sensors.locations, tvb_sensors.orientations, s_type)

    def read_projection(self, path, s_type):
        if s_type == Sensors.TYPE_EEG:
            tvb_prj = projections.ProjectionSurfaceEEG.from_file(path)
        elif s_type == Sensors.TYPE_MEG:
            tvb_prj = projections.ProjectionSurfaceMEG.from_file(path)
        else:
            tvb_prj = projections.ProjectionSurfaceSEEG.from_file(path)
        return tvb_prj.projection_data

    def read_head(self, root_folder, name=''):
        conn = self.read_connectivity(os.path.join(root_folder, "connectivity", "connectivity_76.zip"))
        srf = self.read_cortical_surface(os.path.join(root_folder, "surfaceData", "cortex_16384.zip"))
        rm = self.read_region_mapping(os.path.join(root_folder, "regionMapping", "regionMapping_16k_76.txt"))
        vm = None
        t1 = None

        s = self.read_sensors(os.path.join(root_folder, "sensors", "eeg_brainstorm_65.txt"), Sensors.TYPE_EEG)
        pm = self.read_projection(os.path.join(root_folder, "projectionMatrix", "projection_eeg_65_surface_16k.npy"),
                                  Sensors.TYPE_EEG)
        eeg_sensors_dict = {s: pm}

        s = self.read_sensors(os.path.join(root_folder, "sensors", "meg_brainstorm_276.txt"), Sensors.TYPE_MEG)
        pm = self.read_projection(os.path.join(root_folder, "projectionMatrix", "projection_meg_276_surface_16k.npy"),
                                  Sensors.TYPE_MEG)
        meg_sensors_dict = {s: pm}

        s = self.read_sensors(os.path.join(root_folder, "sensors", "seeg_588.txt"), Sensors.TYPE_SEEG)
        pm = self.read_projection(os.path.join(root_folder, "projectionMatrix", "projection_seeg_588_surface_16k.npy"),
                                  Sensors.TYPE_SEEG)
        seeg_sensors_dict = {s: pm}

        return Head(conn, srf, rm, vm, t1, name, eeg_sensors_dict, meg_sensors_dict, seeg_sensors_dict)
