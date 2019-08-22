# -*- coding: utf-8 -*-

import h5py
from collections import OrderedDict

from tvb_fit.base.model.probabilistic_models.probabilistic_model_base import ProbabilisticModelBase
from tvb_fit.base.model.model_configuration import ModelConfiguration
from tvb_fit.base.model.simulation_settings import SimulationSettings
from tvb_fit.base.model.parameter import Parameter
from tvb_fit.service.model_configuration_builder import ModelConfigurationBuilder
from tvb_fit.service.probabilistic_parameter_builder import generate_probabilistic_parameter, \
    generate_negative_lognormal_parameter
from tvb_fit.io.h5_writer import H5Writer

from tvb_scripts.utils.log_error_utils import initialize_logger, raise_value_error
from tvb_scripts.utils.data_structures_utils import ensure_list
from tvb_scripts.io.h5_reader import H5Reader as H5ReaderBase
from tvb_scripts.io.h5_reader import H5GroupHandlers as H5GroupHandlersBase


H5_TYPE_ATTRIBUTE = H5Writer().H5_TYPE_ATTRIBUTE
H5_SUBTYPE_ATTRIBUTE = H5Writer().H5_SUBTYPE_ATTRIBUTE
H5_TYPES_ATTRUBUTES = [H5_TYPE_ATTRIBUTE, H5_SUBTYPE_ATTRIBUTE]


class H5Reader(H5ReaderBase):
    logger = initialize_logger(__name__)

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


class H5GroupHandlers(H5GroupHandlersBase):

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
        for group_key, group_value in h5_group_value.items():
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
