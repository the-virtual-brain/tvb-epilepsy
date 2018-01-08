"""
Read VEP related entities from custom format and data-structures
"""

import os
import h5py
import numpy as np
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger, warning
from tvb_epilepsy.base.model.vep.surface import Surface
from tvb_epilepsy.base.model.vep.sensors import Sensors
from tvb_epilepsy.base.model.vep.connectivity import Connectivity
from tvb_epilepsy.base.model.vep.head import Head
from tvb_epilepsy.base.readers import ABCReader


class CustomReader(ABCReader):
    logger = initialize_logger(__name__)

    def read_connectivity(self, h5_path):
        """
        :param h5_path: Path towards a custom Connectivity H5 file
        :return: Weights, Tracts, Region centres
        """
        self.logger.info("Reading a Connectivity from: " + h5_path)
        h5_file = h5py.File(h5_path, 'r', libver='latest')
        self.logger.debug("Structures: " + str(h5_file["/"].keys()))
        self.logger.debug("Weights shape:" + str(h5_file['/weights'].shape))
        weights = h5_file['/weights'][()]
        tract_lengths = h5_file['/tract_lengths'][()]
        # TODO: should change to English centres than French centres!
        region_centres = h5_file['/centres'][()]
        region_labels = h5_file['/region_labels'][()]
        orientations = h5_file['/orientations'][()]
        hemispheres = h5_file['/hemispheres'][()]
        h5_file.close()
        return Connectivity(h5_path, weights, tract_lengths, region_labels, region_centres, hemispheres, orientations)

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

    def read_sensors_of_type(self, sensors_file, type):
        if not os.path.exists(sensors_file):
            self.logger.warning("Senors file %s does not exist!" % sensors_file)
            return None

        self.logger.info("Starting to read sensors of type %s from: %s" % (type, sensors_file))
        h5_file = h5py.File(sensors_file, 'r', libver='latest')

        labels = h5_file['/labels'][()]
        locations = h5_file['/locations'][()]

        if '/orientations' in h5_file:
            orientations = h5_file['/orientations'][()]
        else:
            orientations = None
        if '/gain_matrix' in h5_file:
            gain_matrix = h5_file['/gain_matrix'][()]
        else:
            gain_matrix = None

        h5_file.close()
        return Sensors(labels, locations, orientations=orientations, gain_matrix=gain_matrix, s_type=type)

    def read_sensors(self, head_folder):
        sensors_prefix = "Sensors"
        sensors_seeg = []
        sensors_eeg = []
        sensors_meg = []

        all_head_files = os.listdir(head_folder)
        for head_file in all_head_files:
            str_head_file = str(head_file)
            if not str_head_file.startswith(sensors_prefix):
                continue

            type = str_head_file[7:str_head_file.index("_")]
            if type == Sensors.TYPE_SEEG:
                sensors_seeg.append(self.read_sensors_of_type(os.path.join(head_folder, head_file), Sensors.TYPE_SEEG))
            if type == Sensors.TYPE_EEG:
                sensors_eeg.append(self.read_sensors_of_type(os.path.join(head_folder, head_file), Sensors.TYPE_EEG))
            if type == Sensors.TYPE_MEG:
                sensors_meg.append(self.read_sensors_of_type(os.path.join(head_folder, head_file), Sensors.TYPE_MEG))

        return sensors_seeg, sensors_eeg, sensors_meg

    def read_gain_matrix(self, path, s_type):
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
                  seeg_sensors_files=[],
                  eeg_sensors_files=[],
                  meg_sensors_files=[],
                  ):
        conn = self.read_connectivity(os.path.join(root_folder, connectivity_file))
        srf = self.read_cortical_surface(os.path.join(root_folder, surface_file))
        rm = self.read_region_mapping(os.path.join(root_folder, region_mapping_file))
        vm = self.read_volume_mapping(os.path.join(root_folder, volume_mapping_file))
        t1 = self.read_t1(os.path.join(root_folder, structural_mri_file))

        if seeg_sensors_files == [] and eeg_sensors_files == [] and meg_sensors_files == []:
            sensorsSEEG, sensorsEEG, sensorsMEG = self.read_sensors(root_folder)

        else:
            sensorsSEEG = []
            if len(seeg_sensors_files) > 0:
                for sensors_file in seeg_sensors_files:
                    sensorsSEEG.append(
                        self.read_sensors_of_type(os.path.join(root_folder, sensors_file), Sensors.TYPE_SEEG))

            sensorsEEG = []
            if len(eeg_sensors_files) > 0:
                for sensors_file in eeg_sensors_files:
                    sensorsEEG.append(
                        self.read_sensors_of_type(os.path.join(root_folder, sensors_file), Sensors.TYPE_EEG))

            sensorsMEG = []
            if len(meg_sensors_files) > 0:
                for sensors_file in meg_sensors_files:
                    sensorsMEG.append(
                        self.read_sensors_of_type(os.path.join(root_folder, sensors_file), Sensors.TYPE_MEG))

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
