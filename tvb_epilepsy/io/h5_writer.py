import os
import h5py
import numpy
from tvb_epilepsy.base.utils.log_error_utils import raise_error, raise_value_error, initialize_logger
from tvb_epilepsy.base.utils.file_utils import change_filename_or_overwrite, write_metadata
from tvb_epilepsy.base.model.vep.connectivity import ConnectivityH5Field
from tvb_epilepsy.base.model.vep.sensors import SensorsH5Field
from tvb_epilepsy.base.model.vep.surface import SurfaceH5Field
from tvb_epilepsy.base.model.timeseries import Timeseries
from tvb_epilepsy.io.h5_model import convert_to_h5_model

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


class H5Writer(object):
    logger = initialize_logger(__name__)

    H5_TYPE_ATTRIBUTE = "EPI_Type"
    H5_SUBTYPE_ATTRIBUTE = "EPI_Subtype"

    # TODO: write variants.
    def write_connectivity(self, connectivity, path):
        """
        :param connectivity: Connectivity object to be written in H5
        :param path: H5 path to be written
        """
        h5_file = h5py.File(change_filename_or_overwrite(path), 'a', libver='latest')

        h5_file.create_dataset(ConnectivityH5Field.WEIGHTS, data=connectivity.weights)
        h5_file.create_dataset(ConnectivityH5Field.TRACTS, data=connectivity.tract_lengths)
        h5_file.create_dataset(ConnectivityH5Field.CENTERS, data=connectivity.centres)
        h5_file.create_dataset(ConnectivityH5Field.REGION_LABELS, data=connectivity.region_labels)
        h5_file.create_dataset(ConnectivityH5Field.ORIENTATIONS, data=connectivity.orientations)
        h5_file.create_dataset(ConnectivityH5Field.HEMISPHERES, data=connectivity.hemispheres)

        h5_file.attrs.create(self.H5_TYPE_ATTRIBUTE, "Connectivity")
        h5_file.attrs.create("Number_of_regions", str(connectivity.number_of_regions))

        if connectivity.normalized_weights.size > 0:
            dataset = h5_file.create_dataset("normalized_weights/" + ConnectivityH5Field.WEIGHTS,
                                             data=connectivity.normalized_weights)
            dataset.attrs.create("Operations", "Removing diagonal, normalizing with 95th percentile, and ceiling to it")

        self.logger.info("Connectivity has been written to file: %s" % path)
        h5_file.close()

    def write_sensors(self, sensors, path):
        """
        :param sensors: Sensors object to write in H5
        :param path: H5 path to be written
        """
        h5_file = h5py.File(change_filename_or_overwrite(path), 'a', libver='latest')

        h5_file.create_dataset(SensorsH5Field.LABELS, data=sensors.labels)
        h5_file.create_dataset(SensorsH5Field.LOCATIONS, data=sensors.locations)
        h5_file.create_dataset(SensorsH5Field.NEEDLES, data=sensors.needles)

        gain_dataset = h5_file.create_dataset(SensorsH5Field.GAIN_MATRIX, data=sensors.gain_matrix)
        gain_dataset.attrs.create("Max", str(sensors.gain_matrix.max()))
        gain_dataset.attrs.create("Min", str(sensors.gain_matrix.min()))

        h5_file.attrs.create(self.H5_TYPE_ATTRIBUTE, "Sensors")
        h5_file.attrs.create("Number_of_sensors", str(sensors.number_of_sensors))
        h5_file.attrs.create("Sensors_subtype", sensors.s_type)

        self.logger.info("Sensors have been written to file: %s" % path)
        h5_file.close()

    def write_surface(self, surface, path):
        """
        :param surface: Surface object to write in H5
        :param path: H5 path to be written
        """
        h5_file = h5py.File(change_filename_or_overwrite(path), 'a', libver='latest')

        h5_file.create_dataset(SurfaceH5Field.VERTICES, data=surface.vertices)
        h5_file.create_dataset(SurfaceH5Field.TRIANGLES, data=surface.triangles)
        h5_file.create_dataset(SurfaceH5Field.VERTEX_NORMALS, data=surface.vertex_normals)

        h5_file.attrs.create(self.H5_TYPE_ATTRIBUTE, "Surface")
        h5_file.attrs.create("Surface_subtype", surface.surface_subtype)
        h5_file.attrs.create("Number_of_triangles", surface.triangles.shape[0])
        h5_file.attrs.create("Number_of_vertices", surface.vertices.shape[0])
        h5_file.attrs.create("Voxel_to_ras_matrix", str(surface.vox2ras.flatten().tolist())[1:-1].replace(",", ""))

        self.logger.info("Surface has been written to file: %s" % path)
        h5_file.close()

    def write_head(self, head, path):
        """
        :param head: Head object to be written
        :param path: path to head folder
        """
        self.logger.info("Starting to write Head folder: %s" % head)

        if not (os.path.isdir(path)):
            os.mkdir(path)
        self.write_connectivity(head.connectivity, os.path.join(path, "Connectivity.h5"))
        self.write_surface(head.cortical_surface, os.path.join(path, "CorticalSurface.h5"))
        for sensor_list in (head.sensorsSEEG, head.sensorsEEG, head.sensorsMEG):
            for sensors in sensor_list:
                self.write_sensors(sensors,
                                   os.path.join(path, "Sensors%s_%s.h5" % (sensors.s_type, sensors.number_of_sensors)))

        self.logger.info("Successfully wrote Head folder at: %s" % path)

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
        h5_file.attrs.create("number_of_regions", h5_hypo.number_of_regions)
        h5_file.attrs.create("type", h5_hypo.type)
        h5_file.attrs.create("x0_indices", h5_hypo.x0_indices)
        h5_file.attrs.create("e_indices", h5_hypo.e_indices)
        h5_file.attrs.create("w_indices", h5_hypo.w_indices)
        h5_file.attrs.create("lsa_propagation_indices", h5_hypo.lsa_propagation_indices)

        h5_file.close()

    def write_model_configuration(self, model_configuration, path):
        """
        :param model_configuration: ModelConfiguration object to write in H5
        :param path: H5 path to be written
        """
        h5_file = h5py.File(change_filename_or_overwrite(path), 'a', libver='latest')

        datasets_dict, metadata_dict, _ = self._determine_datasets_and_attributes(model_configuration)

        for key, value in datasets_dict.iteritems():
            h5_file.create_dataset(key, data=value)

        h5_file.attrs.create(self.H5_TYPE_ATTRIBUTE, "HypothesisModel")
        h5_file.attrs.create(self.H5_SUBTYPE_ATTRIBUTE, model_configuration.__class__.__name__)

        for key, value in metadata_dict.iteritems():
            h5_file.attrs.create(key, value)

        h5_file.close()

    def write_model_configuration_builder(self, model_configuration_builder, path):
        """
        :param model_configuration_builder: ModelConfigurationService object to write in H5
        :param path: H5 path to be written
        """
        h5_file = h5py.File(change_filename_or_overwrite(path), 'a', libver='latest')

        datasets_dict, metadata_dict, _ = self._determine_datasets_and_attributes(model_configuration_builder)

        for key, value in datasets_dict.iteritems():
            h5_file.create_dataset(key, data=value)

        h5_file.attrs.create(self.H5_TYPE_ATTRIBUTE, "HypothesisModel")
        h5_file.attrs.create(self.H5_SUBTYPE_ATTRIBUTE, model_configuration_builder.__class__.__name__)

        for key, value in metadata_dict.iteritems():
            h5_file.attrs.create(key, value)

        h5_file.close()

    def write_lsa_service(self, lsa_service, path):
        """
        :param lsa_service: LSAService object to write in H5
        :param path: H5 path to be written
        """
        h5_file = h5py.File(change_filename_or_overwrite(path), 'a', libver='latest')

        datasets_dict, metadata_dict, _ = self._determine_datasets_and_attributes(lsa_service)

        for key, value in datasets_dict.iteritems():
            h5_file.create_dataset(key, data=value)

        h5_file.attrs.create(self.H5_TYPE_ATTRIBUTE, "HypothesisModel")
        h5_file.attrs.create(self.H5_SUBTYPE_ATTRIBUTE, lsa_service.__class__.__name__)

        for key, value in metadata_dict.iteritems():
            h5_file.attrs.create(key, value)

        h5_file.close()

    def write_model_inversion_service(self, model_inversion_service, path):
        """
        :param model_inversion_service: ModelInversionService object to write in H5
        :param path: H5 path to be written
        """
        if getattr(model_inversion_service, "signals_inds", None) is not None:
            model_inversion_service.signals_inds = numpy.array(model_inversion_service.signals_inds)

        h5_file = h5py.File(change_filename_or_overwrite(path), 'a', libver='latest')

        datasets_dict, metadata_dict, _ = self._determine_datasets_and_attributes(model_inversion_service)

        for key, value in datasets_dict.iteritems():
            h5_file.create_dataset(key, data=value)

        h5_file.attrs.create(self.H5_TYPE_ATTRIBUTE, "HypothesisModel")
        h5_file.attrs.create(self.H5_SUBTYPE_ATTRIBUTE, model_inversion_service.__class__.__name__)

        for key, value in metadata_dict.iteritems():
            h5_file.attrs.create(key, value)

        h5_file.close()

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

    def write_dictionary(self, dictionary, path):
        """
        :param dictionary: dictionary to write in H5
        :param path: H5 path to be written
        """
        h5_file = h5py.File(change_filename_or_overwrite(path), 'a', libver='latest')

        for key, value in dictionary.iteritems():
            try:
                if isinstance(value, numpy.ndarray) and value.size > 0:
                    h5_file.create_dataset(key, data=value)
                else:
                    if isinstance(value, list) and len(value) > 0:
                        h5_file.create_dataset(key, data=value)
                    else:
                        h5_file.attrs.create(key, value)
            except:
                self.logger.warning("Did not manage to write " + key + " to h5 file " + path + " !")

        h5_file.attrs.create(self.H5_TYPE_ATTRIBUTE, "HypothesisModel")
        h5_file.attrs.create(self.H5_SUBTYPE_ATTRIBUTE, dictionary.__class__.__name__)

        h5_file.close()

    # TODO: can this be visualized? should we keep groups?
    def write_simulation_settings(self, simulation_settings, path):
        """
        :param simulation_settings: SimulationSettings object to write in H5
        :param path: H5 path to be written
        """
        h5_file = h5py.File(change_filename_or_overwrite(path), 'a', libver='latest')

        datasets_dict, metadata_dict, _ = self._determine_datasets_and_attributes(simulation_settings)

        for key, value in datasets_dict.iteritems():
            h5_file.create_dataset(key, data=value)

        h5_file.attrs.create(self.H5_TYPE_ATTRIBUTE, "HypothesisModel")
        h5_file.attrs.create(self.H5_SUBTYPE_ATTRIBUTE, simulation_settings.__class__.__name__)

        for key, value in metadata_dict.iteritems():
            h5_file.attrs.create(key, value)

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
        except Exception, e:
            raise_error(str(e) + "\nSeeg dataset already written as " + sensors_name, self.logger)

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

    def write_ts(self, raw_data, sampling_period, path):
        path = change_filename_or_overwrite(path)

        self.logger.info("Writing a TS at:\n" + path)
        h5_file = h5py.File(path, 'a', libver='latest')
        write_metadata({KEY_TYPE: "TimeSeries"}, h5_file, KEY_DATE, KEY_VERSION)
        if isinstance(raw_data, dict):
            for data in raw_data:
                if len(raw_data[data].shape) == 2 and str(raw_data[data].dtype)[0] == "f":
                    h5_file.create_dataset("/" + data, data=raw_data[data])
                    write_metadata({KEY_MAX: raw_data[data].max(), KEY_MIN: raw_data[data].min(),
                                    KEY_STEPS: raw_data[data].shape[0], KEY_CHANNELS: raw_data[data].shape[1],
                                    KEY_SV: 1, KEY_SAMPLING: sampling_period, KEY_START: 0.0}, h5_file, KEY_DATE,
                                   KEY_VERSION, "/" + data)
                else:
                    raise_value_error("Invalid TS data. 2D (time, nodes) numpy.ndarray of floats expected")
        elif isinstance(raw_data, numpy.ndarray):
            if len(raw_data.shape) != 2 and str(raw_data.dtype)[0] != "f":
                h5_file.create_dataset("/data", data=raw_data)
                write_metadata({KEY_MAX: raw_data.max(), KEY_MIN: raw_data.min(), KEY_STEPS: raw_data.shape[0],
                                KEY_CHANNELS: raw_data.shape[1], KEY_SV: 1, KEY_SAMPLING: sampling_period,
                                KEY_START: 0.0}, h5_file, KEY_DATE, KEY_VERSION, "/data")
            else:
                raise_value_error("Invalid TS data. 2D (time, nodes) numpy.ndarray of floats expected")
        elif isinstance(raw_data, Timeseries):
            if len(raw_data.shape) != 4 and str(raw_data.dtype)[0] != "f":
                h5_file.create_dataset("/data", data=raw_data.data)
                h5_file.create_dataset("/time", data=raw_data.time_line)
                h5_file.create_dataset("/labels", data=raw_data.space_labels)
                h5_file.create_dataset("/variables", data=raw_data.dimension_labels.get("state_variables", []))
                h5_file.attrs.create("time_unit", raw_data.time_unit)
                write_metadata({KEY_MAX: raw_data.data.max(), KEY_MIN: raw_data.data.min(),
                                KEY_STEPS: raw_data.data.shape[0],KEY_CHANNELS: raw_data.data.shape[1],
                                KEY_SV: 1, KEY_SAMPLING: raw_data.time_step,
                                KEY_START: raw_data.time_start}, h5_file, KEY_DATE, KEY_VERSION, "/data")
            else:
                raise_value_error("Invalid TS data. 4D (time, nodes) numpy.ndarray of floats expected")
        else:
            raise_value_error("Invalid TS data. Dictionary or 2D (time, nodes) numpy.ndarray of floats expected")
        h5_file.close()

    def write_timeseries(self, timeseries, path):
        self.write_ts(timeseries, timeseries.time_step, path)

    def write_simulator_model(self, simulator_model, nr_regions, path):
        h5_file = h5py.File(change_filename_or_overwrite(path), 'a', libver='latest')

        datasets_dict, metadata_dict, _ = self._determine_datasets_and_attributes(simulator_model, nr_regions)

        for key, value in datasets_dict.iteritems():
            h5_file.create_dataset(key, data=value)

        h5_file.attrs.create(self.H5_TYPE_ATTRIBUTE, "HypothesisModel")
        h5_file.attrs.create(self.H5_SUBTYPE_ATTRIBUTE, simulator_model.__class__.__name__)

        for key, value in metadata_dict.iteritems():
            h5_file.attrs.create(key, value)

        h5_file.close()

    def _write_dicts_at_location(self, datasets_dict, metadata_dict, location):
        for key, value in datasets_dict.iteritems():
            location.create_dataset(key, data=value)

        for key, value in metadata_dict.iteritems():
            location.attrs.create(key, value)

    def write_statistical_model(self, statistical_model, nr_regions, path):
        """
        :param object:
        :param path:H5 path to be written
        """
        h5_file = h5py.File(change_filename_or_overwrite(path), 'a', libver='latest')

        datasets_dict, metadata_dict, groups_keys = self._determine_datasets_and_attributes(statistical_model,
                                                                                            nr_regions)
        h5_file.attrs.create(self.H5_TYPE_ATTRIBUTE, "HypothesisModel")
        h5_file.attrs.create(self.H5_SUBTYPE_ATTRIBUTE, statistical_model.__class__.__name__)

        self._write_dicts_at_location(datasets_dict, metadata_dict, h5_file)

        for group_key in groups_keys:
            if group_key == "model_config":
                group = h5_file.create_group(group_key)
                group_datasets, group_metadata, _ = self._determine_datasets_and_attributes(
                    statistical_model.model_config, nr_regions)

                group.attrs.create(self.H5_SUBTYPE_ATTRIBUTE, statistical_model.model_config.__class__.__name__)

                self._write_dicts_at_location(group_datasets, group_metadata, group)

            if group_key == "parameters":
                group = h5_file.create_group(group_key)
                group.attrs.create(self.H5_SUBTYPE_ATTRIBUTE, statistical_model.parameters.__class__.__name__)
                for param_key, param_value in statistical_model.parameters.iteritems():
                    param_group = group.create_group(param_key)
                    param_group_datasets, param_group_metadata, _ = self._determine_datasets_and_attributes(param_value,
                                                                                                            nr_regions)

                    param_group.attrs.create(self.H5_SUBTYPE_ATTRIBUTE, param_value.__class__.__name__)

                    self._write_dicts_at_location(param_group_datasets, param_group_metadata, param_group)

        h5_file.close()

    def write_generic(self, object, folder, path):
        """
        :param object:
        :param path:H5 path to be written
        """
        h5_model = convert_to_h5_model(object)

        h5_model.add_or_update_metadata_attribute(self.H5_TYPE_ATTRIBUTE, "HypothesisModel")
        h5_model.add_or_update_metadata_attribute(self.H5_SUBTYPE_ATTRIBUTE, object.__class__.__name__)

        h5_model.write_to_h5(folder, path)

    def _determine_datasets_and_attributes(self, current_object, datasets_size=None):
        datasets_dict = {}
        metadata_dict = {}
        groups_keys = []

        for key, value in vars(current_object).iteritems():
            if isinstance(value, numpy.ndarray):
                if datasets_size is not None and value.size == datasets_size:
                    datasets_dict.update({key: value})
                else:
                    if datasets_size is None and value.size > 0:
                        datasets_dict.update({key: value})
                    else:
                        metadata_dict.update({key: value})
            else:
                if isinstance(value, (float, int, long, complex, str)):
                    metadata_dict.update({key: value})
                else:
                    groups_keys.append(key)

        return datasets_dict, metadata_dict, groups_keys
