# -*- coding: utf-8 -*-

import h5py
import numpy

from tvb_scripts.utils.file_utils import change_filename_or_overwrite
from tvb_scripts.io.h5_writer import H5Writer as H5WriterBase


KEY_TYPE = "EPI_Type"
KEY_VERSION = "EPI_Version"
KEY_DATE = "EPI_Last_update"
KEY_NODES = "Number_of_nodes"
KEY_SENSORS = "Number_of_sensors"
KEY_MAX = "Max_value"
KEY_MIN = "Min_value"
KEY_CHANNELS = "Number_of_channels"
KEY_SV = "Number_of_state_variables"
KEY_STEPS = "Number_of_steps"
KEY_SAMPLING = "Sampling_period"
KEY_START = "Start_time"


class H5Writer(H5WriterBase):

    def write_model_configuration_builder(self, model_configuration_builder, path, nr_regions=None):
        """
        :param model_configuration_builder: ModelConfigurationService object to write in H5
        :param path: H5 path to be written
        """
        self.write_object_to_file(path, model_configuration_builder, "HypothesisModel", nr_regions)

    def write_model_configuration(self, model_configuration, path, nr_regions=None):
        """
        :param model_configuration: EpileptorModelConfiguration object to write in H5
        :param path: H5 path to be written
        """
        self.write_object_to_file(path, model_configuration, "HypothesisModel", nr_regions)

    # TODO: can this be visualized? should we keep groups?
    def write_simulation_settings(self, simulation_settings, path, nr_regions=None):
        """
        :param simulation_settings: SimulationSettings object to write in H5
        :param path: H5 path to be written
        """
        self.write_object_to_file(path, simulation_settings, "HypothesisModel", nr_regions)

    def write_simulator_model(self, simulator_model, path, nr_regions=None):
        # simulator_model.variables_of_interest = numpy.array(simulator_model.variables_of_interest)
        # simulator_model.state_variables = numpy.array(simulator_model.state_variables)
        self.write_object_to_file(path, simulator_model, "HypothesisModel", nr_regions)

    def write_pse_service(self, pse_service, path):
        """
        :param pse_service: PSEService object to write in H5
        :param path: H5 path to be written
        """
        if "params_vals" not in dir(pse_service):
            params_samples = pse_service.pse_params.T
        else:
            params_samples = pse_service.params_vals

        pse_dict = {"task": pse_service.task,
                    "params_names": pse_service.params_names,
                    "params_paths": pse_service.params_paths,
                    "params_indices": numpy.array([str(inds) for inds in pse_service.params_indices], dtype="S"),
                    "params_samples": params_samples}

        self.write_dictionary(pse_dict, path)

    def write_sensitivity_analysis_service(self, sensitivity_service, path):
        """
        :param sensitivity_service: SensitivityAnalysisService object to write in H5
        :param path: H5 path to be written
        """
        sensitivity_service_dict = {"method": sensitivity_service.method,
                                    "calc_second_order": sensitivity_service.calc_second_order,
                                    "conf_level": sensitivity_service.conf_level,
                                    "n_inputs": sensitivity_service.n_inputs,
                                    "n_outputs": sensitivity_service.n_outputs,
                                    "input_names": sensitivity_service.input_names,
                                    "output_names": sensitivity_service.output_names,
                                    "input_bounds": sensitivity_service.input_bounds,
                                    }

        self.write_dictionary(sensitivity_service_dict, path)

    def write_probabilistic_model(self, probabilistic_model, nr_regions, path):
        """
        :param object:
        :param path:H5 path to be written
        """

        def _set_parameter_to_group(parent_group, parameter, nr_regions, param_name=None):
            if param_name is None:
                this_param_group = parent_group.create_group(parameter.name)
            else:
                this_param_group = parent_group.create_group(param_name)
            this_param_group, parameter_subgroups = \
                self._prepare_object_for_group(this_param_group, parameter, nr_regions=nr_regions)
            for param_subgroup_key in parameter_subgroups:
                if param_subgroup_key.find("p_shape") >= 0:
                    this_param_group[param_subgroup_key] = numpy.array(getattr(param_value, param_subgroup_key))
                elif param_subgroup_key == "star":
                    this_param_group, parameter_subgroup = \
                        _set_parameter_to_group(this_param_group, parameter.star, nr_regions, "star")
                else:
                    parameter_subgroup = param_group.create_group(param_subgroup_key)
                    parameter_subgroup, _ = self._prepare_object_for_group(parameter_subgroup,
                                                                           getattr(param_value, param_subgroup_key),
                                                                           nr_regions)
            return parent_group, this_param_group

        h5_file = h5py.File(change_filename_or_overwrite(path), 'a', libver='latest')

        datasets_dict, metadata_dict, groups_keys = self._determine_datasets_and_attributes(probabilistic_model,
                                                                                            nr_regions)
        h5_file.attrs.create(self.H5_TYPE_ATTRIBUTE, "HypothesisModel")
        h5_file.attrs.create(self.H5_SUBTYPE_ATTRIBUTE, probabilistic_model.__class__.__name__)

        self._write_dicts_at_location(datasets_dict, metadata_dict, h5_file)

        for group_key in groups_keys:

            if group_key == "parameters":
                group = h5_file.create_group(group_key)
                group.attrs.create(self.H5_SUBTYPE_ATTRIBUTE, probabilistic_model.parameters.__class__.__name__)
                for param_key, param_value in probabilistic_model.parameters.items():
                    group, param_group = _set_parameter_to_group(group, param_value, nr_regions, param_key)

            else:
                group = h5_file.create_group(group_key)
                group.attrs.create(self.H5_SUBTYPE_ATTRIBUTE,
                                   getattr(probabilistic_model, group_key).__class__.__name__)
                group, _ = self._prepare_object_for_group(group, getattr(probabilistic_model, group_key), nr_regions)

        h5_file.close()
