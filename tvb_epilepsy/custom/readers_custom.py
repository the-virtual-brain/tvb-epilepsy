"""
Read VEP related entities from custom format and data-structures
"""

import os
import h5py
import numpy as np

from tvb_epilepsy.base.utils.data_structures_utils import ensure_list
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger, warning
from tvb_epilepsy.base.model.vep.surface import Surface
from tvb_epilepsy.base.model.vep.sensors import Sensors, TYPE_SEEG, TYPE_EEG, TYPE_MEG
from tvb_epilepsy.base.model.vep.connectivity import Connectivity
from tvb_epilepsy.base.model.vep.head import Head
from tvb_epilepsy.base.readers import ABCReader


class CustomReader(ABCReader):

    logger = initialize_logger(__name__)

    def read_connectivity(self, h5_path):
        """
        :param h5_path: Path towards a custom Connectivity H5 file
        :return: Weights, Tracts, Region centers
        """
        self.logger.info("Reading a Connectivity from: " + h5_path)
        h5_file = h5py.File(h5_path, 'r', libver='latest')
        self.logger.debug("Structures: " + str(h5_file["/"].keys()))
        self.logger.debug("Weights shape:" + str(h5_file['/weights'].shape))
        weights = h5_file['/weights'][()]
        tract_lengths = h5_file['/tract_lengths'][()]
        # TODO: should change to English centers than French centres!
        region_centers = h5_file['/centres'][()]
        region_labels = h5_file['/region_labels'][()]
        orientations = h5_file['/orientations'][()]
        hemispheres = h5_file['/hemispheres'][()]
        h5_file.close()
        return Connectivity(h5_path, weights, tract_lengths, region_labels, region_centers, hemispheres, orientations)

    def read_cortical_surface(self, h5_path):
        if os.path.isfile(h5_path):
            self.logger.info("Reading Surface from " + h5_path)
            h5_file = h5py.File(h5_path, 'r', libver='latest')
            vertices = h5_file['/vertices'][()]
            triangles = h5_file['/triangles'][()]
            vertex_normals = h5_file['/vertex_normals'][()]
            h5_file.close()
            return Surface(vertices, triangles, vertex_normals)
        else:
            warning("\nNo Cortical Surface file found at path " + h5_path + "!")
            return []

    def _read_data_field(self, h5_path):
        self.logger.info("Reading 'data' from H5 " + h5_path)
        h5_file = h5py.File(h5_path, 'r', libver='latest')
        data = h5_file['/data'][()]
        h5_file.close()
        return data

    def read_region_mapping(self, h5_path):
        if os.path.isfile(h5_path):
            return self._read_data_field(h5_path)
        else:
            warning("\nNo Region Mapping file found at path " + h5_path + "!")
            return []

    def read_volume_mapping(self, h5_path):
        if os.path.isfile(h5_path):
            return self._read_data_field(h5_path)
        else:
            warning("\nNo Volume Mapping file found at path " + h5_path + "!")
            return []

    def read_t1(self, h5_path):
        if os.path.isfile(h5_path):
            return self._read_data_field(h5_path)
        else:
            warning("\nNo Structural MRI file found at path " + h5_path + "!")
            return []

    def read_sensors(self, filename, root_folder, s_type):
        filename = ensure_list(filename)
        path = os.path.join(root_folder, filename[0])
        projection = None
        if os.path.isfile(path):
            self.logger.info("Reading Sensors from: " + path)
            h5_file = h5py.File(path, 'r', libver='latest')
            labels = h5_file['/labels'][()]
            locations = h5_file['/locations'][()]
            # TODO: check if h5py returns None for non existing datasets
            if '/orientations' in h5_file:
                orientations = h5_file['/orientations'][()]
            else:
                orientations = None
            if '/projection' in h5_file:
                projection = h5_file['/projection'][()]
            elif len(filename) > 1:
                path = os.path.join(root_folder, filename[1])
                if os.path.isfile(path):
                    projection = self.read_projection(path, s_type)
            h5_file.close()
            return Sensors(labels, locations, orientations, projection, s_type=s_type)
        else:
            warning("\nNo Sensor file found at path " + path + "!")
            return None

    def read_projection(self, path, s_type):
        if os.path.isfile(path):
            return np.load(path)
        else:
            warning("\nNo Projection Matrix file found at path " + path + "!")
            return []

    def read_head(self, root_folder, name='',
                  connectivity_file="Connectivity.h5",
                  surface_file="CorticalSurface.h5",
                  region_mapping_file="RegionMapping.h5",
                  volume_mapping_file="VolumeMapping.h5",
                  structural_mri_file="StructuralMRI.h5",
                  seeg_sensors_files=[("SensorsSEEG_114.h5", ), ("SensorsSEEG_125.h5", )],
                  eeg_sensors_files=[],
                  meg_sensors_files=[],
                  ):
        conn = self.read_connectivity(os.path.join(root_folder, "Connectivity.h5"))
        srf = self.read_cortical_surface(os.path.join(root_folder, "CorticalSurface.h5"))
        rm = self.read_region_mapping(os.path.join(root_folder, "RegionMapping.h5"))
        vm = self.read_volume_mapping(os.path.join(root_folder, "VolumeMapping.h5"))
        t1 = self.read_volume_mapping(os.path.join(root_folder, "StructuralMRI.h5"))
        sensorsSEEG = []
        for s_files in ensure_list(seeg_sensors_files):
            sensorsSEEG.append(self.read_sensors(s_files, root_folder, TYPE_SEEG))
        sensorsEEG = []
        for s_files in ensure_list(eeg_sensors_files):
            sensorsEEG.append(self.read_sensors(s_files, root_folder, TYPE_EEG))
        sensorsMEG = []
        for s_files in ensure_list(meg_sensors_files):
            sensorsMEG.append(self.read_sensors(s_files, root_folder, TYPE_MEG))
        return Head(conn, srf, rm, vm, t1, name, sensorsSEEG=sensorsSEEG, sensorsEEG=sensorsEEG, sensorsMEG=sensorsMEG)

    def read_epileptogenicity(self, root_folder, name="ep"):
        """
        :param
            root_folder: Path towards a valid custom Epileptogenicity H5 file
            name: the name of the hypothesis
        :return: Timeseries in a numpy array
        """
        path = os.path.join(root_folder, name, name + ".h5")
        self.logger.info("Reading Epileptogenicity from:\n" + str(path))
        h5_file = h5py.File(path, 'r', libver='latest')
        self.logger.info("Structures:\n" + str(h5_file["/"].keys()))
        self.logger.info("Values expected shape: " + str(h5_file['/values'].shape))
        values = h5_file['/values'][()]
        self.logger.info("Actual values shape\: " + str(values.shape))
        h5_file.close()
        return values
