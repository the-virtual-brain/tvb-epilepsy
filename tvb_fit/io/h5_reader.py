# -*- coding: utf-8 -*-

import os
import h5py
from collections import OrderedDict
import numpy
from tvb_fit.base.datatypes.dot_dicts import DictDot, OrderedDictDot
from tvb_fit.base.model.probabilistic_models.probabilistic_model_base import ProbabilisticModelBase
from tvb_fit.base.utils.log_error_utils import initialize_logger, raise_value_error
from tvb_fit.base.utils.data_structures_utils import isequal_string, ensure_list
from tvb_fit.base.model.virtual_patient.connectivity import Connectivity, ConnectivityH5Field
from tvb_fit.base.model.virtual_patient.head import Head
from tvb_fit.base.model.virtual_patient.sensors import Sensors, SensorsH5Field
from tvb_fit.base.model.virtual_patient.surface import Surface, SurfaceH5Field
from tvb_fit.base.model.model_configuration import ModelConfiguration
from tvb_fit.base.model.timeseries import TimeseriesDimensions, Timeseries
from tvb_fit.base.model.simulation_settings import SimulationSettings
from tvb_fit.base.model.parameter import Parameter
from tvb_fit.service.model_configuration_builder import ModelConfigurationBuilder
from tvb_fit.service.probabilistic_parameter_builder import generate_probabilistic_parameter, \
    generate_negative_lognormal_parameter
from tvb_fit.io.h5_writer import H5Writer


H5_TYPE_ATTRIBUTE = H5Writer().H5_TYPE_ATTRIBUTE
H5_SUBTYPE_ATTRIBUTE = H5Writer().H5_SUBTYPE_ATTRIBUTE
H5_TYPES_ATTRUBUTES = [H5_TYPE_ATTRIBUTE, H5_SUBTYPE_ATTRIBUTE]


class H5Reader(object):
    logger = initialize_logger(__name__)

    connectivity_filename = "Connectivity.h5"
    cortical_surface_filename = "CorticalSurface.h5"
    region_mapping_filename = "RegionMapping.h5"
    volume_mapping_filename = "VolumeMapping.h5"
    structural_mri_filename = "StructuralMRI.h5"
    sensors_filename_prefix = "Sensors"
    sensors_filename_separator = "_"

    def read_simulator_model(self, path, model_builder_fun):
        """
        :param path: Path towards a TVB model H5 file
        :return: TVB model object
        """
        self.logger.info("Starting to read epileptor model from: %s" % path)
        h5_file = h5py.File(path, 'r', libver='latest')
        try:
            model_name = h5_file["/"].attrs[H5_SUBTYPE_ATTRIBUTE]
            model = model_builder_fun(model_name)
        except:
            raise_value_error("No model read from model configuration file!: %s" % str(path))

        return H5GroupHandlers().read_simulator_model_group(h5_file, model, "/")

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
        self.logger.info("Successfully read connectvity from: %s" % path)

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
        self.logger.info("Successfully read surface from: %s" % path)

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
            if type.upper() == Sensors.TYPE_SEEG:
                sensors_seeg.append(self.read_sensors_of_type(os.path.join(path, head_file), Sensors.TYPE_SEEG))
            if type.upper() == Sensors.TYPE_EEG:
                sensors_eeg.append(self.read_sensors_of_type(os.path.join(path, head_file), Sensors.TYPE_EEG))
            if type.upper() == Sensors.TYPE_MEG:
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
        self.logger.info("Successfully read sensors from: %s" % sensors_file)

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
        self.logger.info("Successfully read volume mapping!")  #: %s" % data)

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
        self.logger.info("Successfully read region mapping!")  #: %s" % data)

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
        self.logger.info("Successfully read structural MRI from: %s" % path)

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
        self.logger.info("Successfully read Head from: %s" % path)

        return head

    def read_model_configuration_builder(self, path, default_model="Epileptor",
                                         model_configuration_builder=ModelConfigurationBuilder):
        """
        :param path: Path towards a ModelConfigurationService H5 file
        :return: ModelConfigurationService object
        """
        self.logger.info("Starting to read ModelConfigurationService from: %s" % path)
        h5_file = h5py.File(path, 'r', libver='latest')

        try:
            model_name = h5_file.attrs["model_name"]
        except:
            self.logger.warning("No model_name read from model configuration builder file!: %s" % str(path))
            self.logger.warning("Setting default model!: %s" % default_model)
            model_name = default_model

        mc_service = model_configuration_builder(model_name)

        for dataset in h5_file.keys():
            if dataset != "model":
                mc_service.set_attribute(dataset, h5_file["/" + dataset][()])

        for attr in h5_file.attrs.keys():
            mc_service.set_attribute(attr, h5_file.attrs[attr])

        h5_file.close()
        return mc_service

    def read_model_configuration(self, path, default_model="Epileptor", model_configuration=ModelConfiguration):
        """
        :param path: Path towards a EpileptorModelConfiguration H5 file
        :return: EpileptorModelConfiguration object
        """
        self.logger.info("Starting to read ModelConfiguration from: %s" % path)
        h5_file = h5py.File(path, 'r', libver='latest')

        try:
            model_name = h5_file.attrs["model_name"]
        except:
            self.logger.warning("No model_name read from model configuration file!: %s" % str(path))
            self.logger.warning("Setting default model!: %s" % default_model)
            model_name =default_model

        model_configuration = model_configuration(model_name)
        for dataset in h5_file.keys():
            if dataset != "model":
                model_configuration.set_attribute(dataset, h5_file["/" + dataset][()])

        for attr in h5_file.attrs.keys():
            model_configuration.set_attribute(attr, h5_file.attrs[attr])

        h5_file.close()
        return model_configuration

    def read_simulation_settings(self, path):
        """
        :param path: Path towards a SimulationSettings H5 file
        :return: SimulationSettings
        """
        self.logger.info("Starting to read SimulationSettings from: %s" % path)
        h5_file = h5py.File(path, 'r', libver='latest')

        sim_settings = SimulationSettings()
        for dataset in h5_file.keys():
            sim_settings.set_attribute(dataset, h5_file["/" + dataset][()])

        for attr in h5_file.attrs.keys():
            sim_settings.set_attribute(attr, h5_file.attrs[attr])

        h5_file.close()
        return sim_settings

    def read_ts(self, path):
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
        self.logger.info("Successfully read timeseries!")  #: %s" % data)
        h5_file.close()

        return time, data

    def read_timeseries(self, path, timeseries=Timeseries):
        """
        :param path: Path towards a valid TimeSeries H5 file
        :return: Timeseries data and time in 2 numpy arrays
        """
        self.logger.info("Starting to read TimeSeries from: %s" % path)
        h5_file = h5py.File(path, 'r', libver='latest')

        data = h5_file['/data'][()]
        time = h5_file['/time'][()]
        labels = h5_file['/labels'][()]
        variables = h5_file['/variables'][()]
        time_unit = h5_file.attrs["time_unit"]
        self.logger.info("First Channel sv sum: " + str(numpy.sum(data[:, 0])))
        self.logger.info("Successfully read Timeseries!")  #: %s" % data)
        h5_file.close()

        return timeseries(data, {TimeseriesDimensions.SPACE.value: labels,
                                 TimeseriesDimensions.VARIABLES.value: variables},
                          time[0], numpy.mean(numpy.diff(time)), time_unit)

    def read_dictionary(self, path, type=None):
        """
        :param path: Path towards a dictionary H5 file
        :return: dict
        """
        self.logger.info("Starting to read a dictionary from: %s" % path)
        h5_file = h5py.File(path, 'r', libver='latest')
        dictionary = H5GroupHandlers().read_dictionary_from_group(h5_file, type)
        h5_file.close()
        return dictionary

    def read_list_of_dicts(self, path, type=None):
        self.logger.info("Starting to read a list of dictionaries from: %s" % path)
        h5_file = h5py.File(path, 'r', libver='latest')
        list_of_dicts = []
        id = 0
        h5_group_handlers = H5GroupHandlers()
        while 1:
            try:
                dict_group = h5_file[str(id)]
            except:
                break
            list_of_dicts.append(h5_group_handlers.read_dictionary_from_group(dict_group, type))
            id += 1
        h5_file.close()
        return list_of_dicts

    def read_probabilistic_model(self, path):
        h5_file = h5py.File(path, 'r', libver='latest')
        epi_subtype = h5_file.attrs[H5_SUBTYPE_ATTRIBUTE]

        probabilistic_model = None
        if ProbabilisticModelBase.__class__.find(epi_subtype) >= 0:
            probabilistic_model = ProbabilisticModelBase()
        else:
            raise_value_error(epi_subtype +
                              "does not correspond to the available probabilistic model!:\n" +
                              ProbabilisticModelBase.__class__)

        for attr in h5_file.attrs.keys():
            if attr not in H5_TYPES_ATTRUBUTES:
                probabilistic_model.__setattr__(attr, h5_file.attrs[attr])

        for key, value in h5_file.items():
            if isinstance(value, h5py.Dataset):
                probabilistic_model.__setattr__(key, value[()])
            if isinstance(value, h5py.Group):
                h5_group_handlers = H5GroupHandlers()
                if key == "parameters":  # and value.attrs[epi_subtype_key] == OrderedDict.__name__:
                    parameters = h5_group_handlers.handle_group_parameters(value)

                    probabilistic_model.__setattr__(key, parameters)

                if key == "ground_truth":
                    h5_group_handlers.handle_group_ground_truth(value, probabilistic_model)

        h5_file.close()
        return probabilistic_model


class H5GroupHandlers(object):

    def read_dictionary_from_group(self, group, type=None):
        dictionary = dict()
        for dataset in group.keys():
            dictionary.update({dataset: group[dataset][()]})
        for attr in group.attrs.keys():
            dictionary.update({attr: group.attrs[attr]})
        if type is None:
            type = group.attrs[H5_SUBTYPE_ATTRIBUTE]
        if isequal_string(type, "DictDot"):
            return DictDot(dictionary)
        elif isequal_string(type, "OrderedDictDot"):
            return OrderedDictDot(dictionary)
        else:
            return dictionary

    def read_simulator_model_group(self, h5_file, model, group):
        for dataset in h5_file[group].keys():
            if dataset in ["variables_of_interest", "state_variables"] :
                setattr(model, dataset, ensure_list(h5_file[group][dataset][()]))
            else:
                setattr(model, dataset, h5_file[group][dataset][()])

        for attr in h5_file[group].attrs.keys():
            setattr(model, attr, h5_file[group].attrs[attr])

        return model

    def read_model_configuration_from_group(self, h5_file, group_name, default_model_name="Epileptor"):
        model_name = h5_file[group_name].attrs.get("model_name", h5_file.attrs.get("model_name", None))
        if model_name is None:
            self.logger.warning("No model_name read from model configuration file!: %s" % str(h5_file))
            self.logger.warning("Setting default model!: %s" + default_model_name)
            model_name = default_model_name

        model_configuration = ModelConfiguration(model_name)
        for dataset in h5_file[group_name].keys():
            if dataset != "model":
                model_configuration.set_attribute(dataset, h5_file[group_name + "/" + dataset][()])

        for attr in h5_file[group_name].attrs.keys():
            model_configuration.set_attribute(attr, h5_file[group_name].attrs[attr])

        return model_configuration

    def handle_group_parameters(self, h5_group_value):
        def strip_key_name(key):
            if key != "star":
                if key.find("_ProbabilityDistribution_") >= 0:
                    key_name = key.split("_ProbabilityDistribution_")[-1]
                elif key.find("_Parameter_") >= 0:
                    key_name = key.split("_Parameter_")[-1]
                else:
                    key_name = key
            return key_name

        def setattr_param(param, key, key_name, value):
            param.__setattr__(key_name, value)
            if key != key_name:
                try:
                    param.__setattr__(key, value)
                except:
                    pass

        def set_parameter_datasets(param, h5location):
            for key in h5location.keys():
                if key != "star":
                    key_name = strip_key_name(key)
                    if key.find("p_shape") >= 0:
                        setattr_param(param, key, key_name, tuple(h5location[key][()]))
                    else:
                        setattr_param(param, key, key_name, h5location[key][()])

        def set_parameter_attributes(param, h5location):
            for key in h5location.attrs.keys():
                if key not in H5_TYPES_ATTRUBUTES:
                    setattr_param(param, key, strip_key_name(key), h5location.attrs[key])

        parameters = OrderedDict()
        for group_key, group_value in h5_group_value.iteritems():
            param_epi_subtype = group_value.attrs[H5_SUBTYPE_ATTRIBUTE]
            if param_epi_subtype == "ProbabilisticParameter":
                parameter = generate_probabilistic_parameter(
                    probability_distribution=group_value.attrs["type"])
            elif param_epi_subtype == "NegativeLognormal":
                parameter = generate_negative_lognormal_parameter("", 1.0, 0.0, 2.0)
                set_parameter_datasets(parameter.star, group_value["star"])
                set_parameter_attributes(parameter.star, group_value["star"])
            else:
                parameter = Parameter()

            set_parameter_datasets(parameter, group_value)
            set_parameter_attributes(parameter, group_value)

            parameters.update({group_key: parameter})

        return parameters

    def handle_group_ground_truth(self, h5_group_value, probabilistic_model):
        for dataset in h5_group_value.keys():
            probabilistic_model.ground_truth[dataset] = h5_group_value[dataset][()]
        for attr in h5_group_value.attrs.keys():
            if attr not in H5_TYPES_ATTRUBUTES:
                probabilistic_model.ground_truth[attr] = h5_group_value.attrs[attr]
