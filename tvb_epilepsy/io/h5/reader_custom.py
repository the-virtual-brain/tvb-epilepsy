import os
import numpy
import h5py
from tvb_epilepsy.base.model.vep.connectivity import Connectivity, ConnectivityH5Field
from tvb_epilepsy.base.model.vep.head import Head
from tvb_epilepsy.base.model.vep.sensors import Sensors, SensorsH5Field
from tvb_epilepsy.base.model.vep.surface import Surface, SurfaceH5Field
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger
from tvb_epilepsy.io.h5.reader import ABCH5Reader


class CustomH5Reader(ABCH5Reader):
    logger = initialize_logger(__name__)

    connectivity_filename = "Connectivity.h5"
    cortical_surface_filename = "CorticalSurface.h5"
    region_mapping_filename = "RegionMapping.h5"
    volume_mapping_filename = "VolumeMapping.h5"
    structural_mri_filename = "StructuralMRI.h5"
    sensors_filename_prefix = "Sensors"
    sensors_filename_separator = "_"

    def read_connectivity(self, path):
        """
        :param path: Path towards a custom Connectivity H5 file
        :return: Connectivity object
        """
        self.logger.info("Starting to read a Connectivity from: %s" % path)
        h5_file = h5py.File(path, 'r', libver='latest')

        weights = h5_file['/' + ConnectivityH5Field.WEIGHTS][()]
        tract_lengths = h5_file['/' + ConnectivityH5Field.TRACTS][()]
        region_centres = h5_file['/' + ConnectivityH5Field.CENTERS][()]
        region_labels = h5_file['/' + ConnectivityH5Field.REGION_LABELS][()]
        orientations = h5_file['/' + ConnectivityH5Field.ORIENTATIONS][()]
        hemispheres = h5_file['/' + ConnectivityH5Field.HEMISPHERES][()]

        h5_file.close()

        conn = Connectivity(path, weights, tract_lengths, region_labels, region_centres, hemispheres, orientations)
        self.logger.info("Successfully read connectvity: %s" % conn)

        return conn

    def read_surface(self, path):
        """
        :param path: Path towards a custom Surface H5 file
        :return: Surface object
        """
        if not os.path.isfile(path):
            self.logger.warning("Surface file %s does not exist" % path)
            return None

        self.logger.info("Starting to read Surface from: %s" % path)
        h5_file = h5py.File(path, 'r', libver='latest')

        vertices = h5_file['/' + SurfaceH5Field.VERTICES][()]
        triangles = h5_file['/' + SurfaceH5Field.TRIANGLES][()]
        vertex_normals = h5_file['/' + SurfaceH5Field.VERTEX_NORMALS][()]

        h5_file.close()

        surface = Surface(vertices, triangles, vertex_normals)
        self.logger.info("Successfully read surface: %s" % surface)

        return surface

    def read_sensors(self, path):
        """
        :param path: Path towards a custom head folder
        :return: 3 lists with all sensors from Path by type
        """
        sensors_seeg = []
        sensors_eeg = []
        sensors_meg = []

        self.logger.info("Starting to read all Sensors from: %s" % path)

        all_head_files = os.listdir(path)
        for head_file in all_head_files:
            str_head_file = str(head_file)
            if not str_head_file.startswith(self.sensors_filename_prefix):
                continue

            type = str_head_file[len(self.sensors_filename_prefix):str_head_file.index(self.sensors_filename_separator)]
            if type == Sensors.TYPE_SEEG:
                sensors_seeg.append(self.read_sensors_of_type(os.path.join(path, head_file), Sensors.TYPE_SEEG))
            if type == Sensors.TYPE_EEG:
                sensors_eeg.append(self.read_sensors_of_type(os.path.join(path, head_file), Sensors.TYPE_EEG))
            if type == Sensors.TYPE_MEG:
                sensors_meg.append(self.read_sensors_of_type(os.path.join(path, head_file), Sensors.TYPE_MEG))

        self.logger.info("Successfuly read all sensors from: %s" % path)

        return sensors_seeg, sensors_eeg, sensors_meg

    def read_sensors_of_type(self, sensors_file, type):
        """
        :param
            sensors_file: Path towards a custom Sensors H5 file
            type: Senors type
        :return: Sensors object
        """
        if not os.path.exists(sensors_file):
            self.logger.warning("Senors file %s does not exist!" % sensors_file)
            return None

        self.logger.info("Starting to read sensors of type %s from: %s" % (type, sensors_file))
        h5_file = h5py.File(sensors_file, 'r', libver='latest')

        labels = h5_file['/' + SensorsH5Field.LABELS][()]
        locations = h5_file['/' + SensorsH5Field.LOCATIONS][()]

        if '/orientations' in h5_file:
            orientations = h5_file['/orientations'][()]
        else:
            orientations = None
        if '/' + SensorsH5Field.GAIN_MATRIX in h5_file:
            gain_matrix = h5_file['/' + SensorsH5Field.GAIN_MATRIX][()]
        else:
            gain_matrix = None

        h5_file.close()

        sensors = Sensors(labels, locations, orientations=orientations, gain_matrix=gain_matrix, s_type=type)
        self.logger.info("Successfully read sensors: %s" % sensors)

        return sensors

    def read_volume_mapping(self, path):
        """
        :param path: Path towards a custom VolumeMapping H5 file
        :return: volume mapping in a numpy array
        """
        if not os.path.isfile(path):
            self.logger.warning("VolumeMapping file %s does not exist" % path)
            return numpy.array([])

        self.logger.info("Starting to read VolumeMapping from: %s" % path)
        h5_file = h5py.File(path, 'r', libver='latest')

        data = h5_file['/data'][()]

        h5_file.close()
        self.logger.info("Successfully read volume mapping: %s" % data)

        return data

    def read_region_mapping(self, path):
        """
        :param path: Path towards a custom RegionMapping H5 file
        :return: region mapping in a numpy array
        """
        if not os.path.isfile(path):
            self.logger.warning("RegionMapping file %s does not exist" % path)
            return numpy.array([])

        self.logger.info("Starting to read RegionMapping from: %s" % path)
        h5_file = h5py.File(path, 'r', libver='latest')

        data = h5_file['/data'][()]

        h5_file.close()
        self.logger.info("Successfully read region mapping: %s" % data)

        return data

    def read_t1(self, path):
        """
        :param path: Path towards a custom StructuralMRI H5 file
        :return: structural MRI in a numpy array
        """
        if not os.path.isfile(path):
            self.logger.warning("StructuralMRI file %s does not exist" % path)
            return numpy.array([])

        self.logger.info("Starting to read StructuralMRI from: %s" % path)
        h5_file = h5py.File(path, 'r', libver='latest')

        data = h5_file['/data'][()]

        h5_file.close()
        self.logger.info("Successfully read structural MRI: %s" % data)

        return data

    def read_head(self, path):
        """
        :param path: Path towards a custom head folder
        :return: Head object
        """
        self.logger.info("Starting to read Head from: %s" % path)
        conn = self.read_connectivity(os.path.join(path, self.connectivity_filename))
        srf = self.read_surface(os.path.join(path, self.cortical_surface_filename))
        rm = self.read_region_mapping(os.path.join(path, self.region_mapping_filename))
        vm = self.read_volume_mapping(os.path.join(path, self.volume_mapping_filename))
        t1 = self.read_t1(os.path.join(path, self.structural_mri_filename))
        sensorsSEEG, sensorsEEG, sensorsMEG = self.read_sensors(path)

        head = Head(conn, srf, rm, vm, t1, path, sensorsSEEG=sensorsSEEG, sensorsEEG=sensorsEEG, sensorsMEG=sensorsMEG)
        self.logger.info("Successfully read Head: %s" % head)

        return head

    def read_epileptogenicity(self, root_folder, name="ep"):
        """
        :param
            root_folder: Path towards a valid custom Epileptogenicity H5 file
            name: the name of the hypothesis
        :return: Timeseries in a numpy array
        """
        path = os.path.join(root_folder, name, name + ".h5")
        self.logger.info("Starting to read Epileptogenicity from: %s" % path)
        h5_file = h5py.File(path, 'r', libver='latest')

        values = h5_file['/values'][()]

        h5_file.close()
        self.logger.info("Successfully read epileptogenicity values: %s" % values)

        return values

    def read_timeseries(self, path):
        """
        :param path: Path towards a valid TimeSeries H5 file
        :return: Timeseries data and time in 2 numpy arrays
        """
        self.logger.info("Starting to read TimeSeries from: %s" % path)
        h5_file = h5py.File(path, 'r', libver='latest')

        data = h5_file['/data'][()]
        total_time = int(h5_file["/"].attrs["Simulated_period"][0])
        nr_of_steps = int(h5_file["/data"].attrs["Number_of_steps"][0])
        start_time = float(h5_file["/data"].attrs["Start_time"][0])
        time = numpy.linspace(start_time, total_time, nr_of_steps)

        self.logger.info("First Channel sv sum: " + str(numpy.sum(data[:, 0])))
        self.logger.info("Successfully read Timeseries: %s" % data)
        h5_file.close()

        return time, data
