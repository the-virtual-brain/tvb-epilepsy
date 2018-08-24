import os
import h5py

import numpy

from tvb_fit.tvb_epilepsy.base.model.timeseries import Timeseries
from tvb_fit.base.utils.log_error_utils import raise_value_error, raise_error
from tvb_fit.base.utils.file_utils import change_filename_or_overwrite, write_metadata
from tvb_fit.io.h5_writer import KEY_TYPE, KEY_DATE, KEY_VERSION, KEY_MAX, KEY_MIN, KEY_STEPS, \
    KEY_CHANNELS, KEY_SV, KEY_SAMPLING, KEY_START
from tvb_fit.io.h5_writer import H5Writer as H5WriterBase


class H5Writer(H5WriterBase):

    def write_hypothesis(self, hypothesis, path):
        """
        :param hypothesis: DiseaseHypothesis object to write in H5
        :param path: H5 path to be written
        """
        h5_file = h5py.File(change_filename_or_overwrite(path), 'a', libver='latest')

        h5_hypo = hypothesis.prepare_hypothesis_for_h5()

        h5_file.create_dataset("x0_values", data=h5_hypo.x0_values)
        h5_file.create_dataset("e_values", data=h5_hypo.e_values)
        h5_file.create_dataset("w_values", data=h5_hypo.w_values)
        h5_file.create_dataset("lsa_propagation_strengths", data=h5_hypo.lsa_propagation_strengths)

        # TODO: change HypothesisModel to GenericModel here and inside Epi
        h5_file.attrs.create(self.H5_TYPE_ATTRIBUTE, "HypothesisModel")
        h5_file.attrs.create(self.H5_SUBTYPE_ATTRIBUTE, h5_hypo.__class__.__name__)
        h5_file.attrs.create("name", h5_hypo.name)
        h5_file.attrs.create("number_of_regions", h5_hypo.number_of_regions)
        h5_file.attrs.create("type", h5_hypo.type)
        h5_file.attrs.create("x0_indices", h5_hypo.x0_indices)
        h5_file.attrs.create("e_indices", h5_hypo.e_indices)
        h5_file.attrs.create("w_indices", h5_hypo.w_indices)
        h5_file.attrs.create("lsa_propagation_indices", h5_hypo.lsa_propagation_indices)

        h5_file.close()

    def write_lsa_service(self, lsa_service, path, nr_regions=None):
        """
        :param lsa_service: LSAService object to write in H5
        :param path: H5 path to be written
        """
        self.write_object_to_file(path, lsa_service, "HypothesisModel", nr_regions)

    def write_ts_epi(self, raw_ts, sampling_period, path, source_ts=None):
        path = change_filename_or_overwrite(os.path.join(path))

        if raw_ts is None or len(raw_ts.squeezed.shape) != 3:
            raise_value_error("Invalid TS data 3D (time, regions, sv) expected", self.logger)
        self.logger.info("Writing a TS at:\n" + path)
        if source_ts is None:
            source_ts = raw_ts.source
        h5_file = h5py.File(path, 'a', libver='latest')
        h5_file.create_dataset("/data", data=raw_ts.squeezed)
        h5_file.create_dataset("/lfpdata", data=source_ts.squeezed)
        write_metadata({KEY_TYPE: "TimeSeries"}, h5_file, KEY_DATE, KEY_VERSION)
        write_metadata({KEY_MAX: raw_ts.squeezed.max(), KEY_MIN: raw_ts.squeezed.min(),
                        KEY_STEPS: raw_ts.squeezed.shape[0], KEY_CHANNELS: raw_ts.squeezed.shape[1],
                        KEY_SV: raw_ts.squeezed.shape[2], KEY_SAMPLING: sampling_period,
                        KEY_START: raw_ts.time_start}, h5_file, KEY_DATE, KEY_VERSION, "/data")
        write_metadata({KEY_MAX: source_ts.squeezed.max(), KEY_MIN: source_ts.squeezed.min(),
                        KEY_STEPS: source_ts.squeezed.shape[0], KEY_CHANNELS: source_ts.squeezed.shape[1],
                        KEY_SV: 1, KEY_SAMPLING: sampling_period, KEY_START: source_ts.time_start}, h5_file, KEY_DATE,
                       KEY_VERSION, "/lfpdata")
        h5_file.close()

    def write_ts_seeg_epi(self, seeg_data, sampling_period, path):
        if not os.path.exists(path):
            raise_error("TS file %s does not exist. First define the raw data!" + path, self.logger)
            return
        sensors_name = "SeegSensors-" + str(seeg_data.shape[1])

        self.logger.info("Writing a TS at:\n" + path + ", " + sensors_name)
        try:
            if isinstance(seeg_data, Timeseries):
                seeg_ts = seeg_data.squeezed
            h5_file = h5py.File(path, 'a', libver='latest')
            h5_file.create_dataset("/" + sensors_name, data=seeg_ts)
            write_metadata({KEY_MAX: seeg_ts.max(), KEY_MIN: seeg_ts.min(), KEY_STEPS: seeg_ts.shape[0],
                            KEY_CHANNELS: seeg_ts.shape[1], KEY_SV: 1, KEY_SAMPLING: sampling_period, KEY_START: 0.0},
                           h5_file, KEY_DATE, KEY_VERSION, "/" + sensors_name)
            h5_file.close()
        except Exception as e:
            raise_error(str(e) + "\nSeeg dataset already written as " + sensors_name, self.logger)

    def write_model_inversion_service(self, model_inversion_service, path, nr_regions=None):
        """
        :param model_inversion_service: ModelInversionService object to write in H5
        :param path: H5 path to be written
        """
        if getattr(model_inversion_service, "signals_inds", None) is not None:
            model_inversion_service.signals_inds = numpy.array(model_inversion_service.signals_inds)

        self.write_object_to_file(path, model_inversion_service, "HypothesisModel", nr_regions)

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
            this_param_group, parameter_subgroups = self._prepare_object_for_group(this_param_group, parameter,
                                                                                   nr_regions=nr_regions,
                                                                                   regress_subgroups=False)
            for param_subgroup_key in parameter_subgroups:
                if param_subgroup_key.find("p_shape") >= 0:
                    this_param_group[param_subgroup_key] = numpy.array(getattr(param_value, param_subgroup_key))
                elif param_subgroup_key == "star":
                    this_param_group, parameter_subgroup = \
                        _set_parameter_to_group(this_param_group, parameter.star, nr_regions, "star")
                else:
                    this_param_group.create_group(param_subgroup_key)
                    this_param_group[param_subgroup_key] =\
                        self._prepare_object_for_group(this_param_group[param_subgroup_key],
                                                      getattr(param_value, param_subgroup_key), nr_regions)
            return parent_group, this_param_group

        h5_file = h5py.File(change_filename_or_overwrite(path), 'a', libver='latest')

        datasets_dict, metadata_dict, groups_keys = self._determine_datasets_and_attributes(probabilistic_model,
                                                                                            nr_regions)
        h5_file.attrs.create(self.H5_TYPE_ATTRIBUTE, "HypothesisModel")
        h5_file.attrs.create(self.H5_SUBTYPE_ATTRIBUTE, probabilistic_model.__class__.__name__)

        self._write_dicts_at_location(datasets_dict, metadata_dict, h5_file)

        for group_key in groups_keys:
            if group_key == "active_regions":
                h5_file.create_dataset(group_key, data=numpy.array(probabilistic_model.active_regions))

            elif group_key == "parameters":
                group = h5_file.create_group(group_key)
                group.attrs.create(self.H5_SUBTYPE_ATTRIBUTE, probabilistic_model.parameters.__class__.__name__)
                for param_key, param_value in probabilistic_model.parameters.items():
                    group, param_group =_set_parameter_to_group(group, param_value, nr_regions, param_key)

            else:
                group = h5_file.create_group(group_key)
                group.attrs.create(self.H5_SUBTYPE_ATTRIBUTE,
                                   getattr(probabilistic_model, group_key).__class__.__name__)
                group = self._prepare_object_for_group(group, getattr(probabilistic_model, group_key), nr_regions)

        h5_file.close()
