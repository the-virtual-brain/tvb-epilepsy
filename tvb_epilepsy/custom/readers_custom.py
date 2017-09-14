"""
Read VEP related entities from custom format and data-structures
"""

import os 

import h5py

from tvb_epilepsy.base.utils import warning, raise_not_implemented_error, ensure_list, initialize_logger, \
                                    calculate_projection
from tvb_epilepsy.base.model.model_vep import Connectivity, Surface, Sensors, Head
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

    def read_sensors(self, h5_path, s_type):
        if os.path.isfile(h5_path):
            self.logger.info("Reading Sensors from: " + h5_path)
            h5_file = h5py.File(h5_path, 'r', libver='latest')

            labels = h5_file['/labels'][()]
            locations = h5_file['/locations'][()]

            h5_file.close()
            return Sensors(labels, locations, s_type=s_type)
        else:
            warning("\nNo Sensor file found at path " + h5_path + "!")
            return None

    def read_projection(self, path, s_type):
        warning("Custom projection matrix reading not implemented yet!")
        return []
        # raise_not_implemented_error()

    def read_sensors_projections(self, root_folder, conn, sensor_files, s_type):
        sensors_dict = {}
        for sensor_file in ensure_list(sensor_files):
            sensor = self.read_sensors(os.path.join(root_folder, sensor_file[0]), s_type)
            if isinstance(sensor, Sensors):
                projection = self.read_projection(os.path.join(root_folder, sensor_file[1]), s_type)
                if projection==[]:
                    warning("Calculating projection matrix based solely on euclidean distance!")
                    projection = calculate_projection(sensor, conn)
                sensors_dict[sensor] = projection
        return sensors_dict

    def read_head(self, root_folder, name='',
                  connectivity_file="Connectivity.h5",
                  surface_file="CorticalSurface.h5",
                  region_mapping_file="RegionMapping.h5",
                  volume_mapping_file="VolumeMapping.h5",
                  structural_mri_file="StructuralMRI.h5",
                  seeg_sensors_files=[("SensorsSEEG_114.h5", ""), ("SensorsSEEG_125.h5", "")],
                  eeg_sensors_files=[("eeg_brainstorm_65.txt", "projection_eeg_65_surface_16k.npy")],
                  meg_sensors_files=[("meg_brainstorm_276.txt", "projection_meg_276_surface_16k.npy")],
                  ):

        conn = self.read_connectivity(os.path.join(root_folder, "Connectivity.h5"))
        srf = self.read_cortical_surface(os.path.join(root_folder, "CorticalSurface.h5"))
        rm = self.read_region_mapping(os.path.join(root_folder, "RegionMapping.h5"))
        vm = self.read_volume_mapping(os.path.join(root_folder, "VolumeMapping.h5"))
        t1 = self.read_volume_mapping(os.path.join(root_folder, "StructuralMRI.h5"))

        seeg_sensors_dict = self.read_sensors_projections(root_folder, conn, seeg_sensors_files, Sensors.TYPE_SEEG)

        eeg_sensors_dict = self.read_sensors_projections(root_folder, conn, eeg_sensors_files, Sensors.TYPE_EEG)

        meg_sensors_dict = self.read_sensors_projections(root_folder, conn, meg_sensors_files, Sensors.TYPE_MEG)

        return Head(conn, srf, rm, vm, t1, name, eeg_sensors_dict, meg_sensors_dict, seeg_sensors_dict)

    def read_epileptogenicity(self, root_folder, name="ep"):
        """
        :param
            root_folder: Path towards a valid custom Epileptogenicity H5 file
            name: the name of the hypothesis
        :return: Timeseries in a numpy array
        """
        path = os.path.join(root_folder, name, name + ".h5")

        print "Reading Epileptogenicity from:", path
        h5_file = h5py.File(path, 'r', libver='latest')

        print "Structures:", h5_file["/"].keys()
        print "Values expected shape:", h5_file['/values'].shape

        values = h5_file['/values'][()]
        print "Actual values shape", values.shape

        h5_file.close()
        return values
