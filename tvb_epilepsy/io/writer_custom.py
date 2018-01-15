import os
import h5py
import numpy
from tvb_epilepsy.base.h5_model import convert_to_h5_model
from tvb_epilepsy.base.model.vep.connectivity import ConnectivityH5Field
from tvb_epilepsy.base.model.vep.sensors import SensorsH5Field
from tvb_epilepsy.base.model.vep.surface import SurfaceH5Field
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger


class CustomH5Writer(object):
    logger = initialize_logger(__name__)

    CUSTOM_TYPE_ATTRIBUTE = "EPI_Type"
    CUSTOM_SUBTYPE_ATTRIBUTE = "EPI_Subtype"

    # TODO: write variants.
    def write_connectivity(self, connectivity, path):
        """
        :param connectivity: Connectivity object to be written in H5
        :param path: H5 path to be written
        """
        h5_file = h5py.File(path, 'a', libver='latest')

        h5_file.create_dataset(ConnectivityH5Field.WEIGHTS, data=connectivity.weights)
        h5_file.create_dataset(ConnectivityH5Field.TRACTS, data=connectivity.tract_lengths)
        h5_file.create_dataset(ConnectivityH5Field.CENTERS, data=connectivity.centres)
        h5_file.create_dataset(ConnectivityH5Field.REGION_LABELS, data=connectivity.region_labels)
        h5_file.create_dataset(ConnectivityH5Field.ORIENTATIONS, data=connectivity.orientations)
        h5_file.create_dataset(ConnectivityH5Field.HEMISPHERES, data=connectivity.hemispheres)

        h5_file.attrs.create(self.CUSTOM_TYPE_ATTRIBUTE, "Connectivity")
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
        h5_file = h5py.File(path, 'a', libver='latest')

        h5_file.create_dataset(SensorsH5Field.LABELS, data=sensors.labels)
        h5_file.create_dataset(SensorsH5Field.LOCATIONS, data=sensors.locations)
        h5_file.create_dataset(SensorsH5Field.NEEDLES, data=sensors.needles)

        gain_dataset = h5_file.create_dataset(SensorsH5Field.GAIN_MATRIX, data=sensors.gain_matrix)
        gain_dataset.attrs.create("Max", str(sensors.gain_matrix.max()))
        gain_dataset.attrs.create("Min", str(sensors.gain_matrix.min()))

        h5_file.attrs.create(self.CUSTOM_TYPE_ATTRIBUTE, "Sensors")
        h5_file.attrs.create("Number_of_sensors", str(sensors.number_of_sensors))
        h5_file.attrs.create("Sensors_subtype", sensors.s_type)

        self.logger.info("Sensors have been written to file: %s" % path)
        h5_file.close()

    def write_surface(self, surface, path):
        """
        :param surface: Surface object to write in H5
        :param path: H5 path to be written
        """
        h5_file = h5py.File(path, 'a', libver='latest')

        h5_file.create_dataset(SurfaceH5Field.VERTICES, data=surface.vertices)
        h5_file.create_dataset(SurfaceH5Field.TRIANGLES, data=surface.triangles)
        h5_file.create_dataset(SurfaceH5Field.VERTEX_NORMALS, data=surface.vertex_normals)

        h5_file.attrs.create(self.CUSTOM_TYPE_ATTRIBUTE, "Surface")
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
        h5_file = h5py.File(path, 'a', libver='latest')

        h5_file.create_dataset("x0_values", data=hypothesis.x0_values)
        h5_file.create_dataset("e_values", data=hypothesis.e_values)
        h5_file.create_dataset("w_values", data=hypothesis.w_values)
        h5_file.create_dataset("lsa_propagation_strengths", data=hypothesis.lsa_propagation_strengths)

        # TODO: change HypothesisModel to GenericModel here and inside Epi
        h5_file.attrs.create(self.CUSTOM_TYPE_ATTRIBUTE, "HypothesisModel")
        h5_file.attrs.create(self.CUSTOM_SUBTYPE_ATTRIBUTE, hypothesis.__class__.__name__)
        h5_file.attrs.create("number_of_regions", hypothesis.number_of_regions)
        h5_file.attrs.create("type", hypothesis.type)
        h5_file.attrs.create("x0_indices", hypothesis.x0_indices)
        h5_file.attrs.create("e_indices", hypothesis.e_indices)
        h5_file.attrs.create("w_indices", hypothesis.w_indices)
        h5_file.attrs.create("lsa_propagation_indices", hypothesis.lsa_propagation_indices)

        h5_file.close()

    def write_model_configuration(self, model_configuration, path):
        """
        :param model_configuration: ModelConfiguration object to write in H5
        :param path: H5 path to be written
        """
        h5_file = h5py.File(path, 'a', libver='latest')

        datasets_dict, metadata_dict = self._determine_datasets_and_attributes(model_configuration)

        for key, value in datasets_dict.iteritems():
            h5_file.create_dataset(key, data=value)

        h5_file.attrs.create(self.CUSTOM_TYPE_ATTRIBUTE, "HypothesisModel")
        h5_file.attrs.create(self.CUSTOM_SUBTYPE_ATTRIBUTE, model_configuration.__class__.__name__)

        for key, value in metadata_dict.iteritems():
            h5_file.attrs.create(key, value)

        h5_file.close()

    def write_model_configuration_service(self, model_configuration_service, path):
        """
        :param model_configuration_service: ModelConfigurationService object to write in H5
        :param path: H5 path to be written
        """
        h5_file = h5py.File(path, 'a', libver='latest')

        datasets_dict, metadata_dict = self._determine_datasets_and_attributes(model_configuration_service)

        for key, value in datasets_dict.iteritems():
            h5_file.create_dataset(key, data=value)

        h5_file.attrs.create(self.CUSTOM_TYPE_ATTRIBUTE, "HypothesisModel")
        h5_file.attrs.create(self.CUSTOM_SUBTYPE_ATTRIBUTE, model_configuration_service.__class__.__name__)

        for key, value in metadata_dict.iteritems():
            h5_file.attrs.create(key, value)

        h5_file.close()

    def write_lsa_service(self, lsa_service, path):
        """
        :param lsa_service: LSAService object to write in H5
        :param path: H5 path to be written
        """
        h5_file = h5py.File(path, 'a', libver='latest')

        datasets_dict, metadata_dict = self._determine_datasets_and_attributes(lsa_service)

        for key, value in datasets_dict.iteritems():
            h5_file.create_dataset(key, data=value)

        h5_file.attrs.create(self.CUSTOM_TYPE_ATTRIBUTE, "HypothesisModel")
        h5_file.attrs.create(self.CUSTOM_SUBTYPE_ATTRIBUTE, lsa_service.__class__.__name__)

        for key, value in metadata_dict.iteritems():
            h5_file.attrs.create(key, value)

        h5_file.close()

    def write_model_inversion_service(self, model_inversion_service, path):
        """
        :param model_inversion_service: ModelInversionService object to write in H5
        :param path:  H5 path to be written
        """
        h5_file = h5py.File(path, 'a', libver='latest')

        datasets_dict, metadata_dict = self._determine_datasets_and_attributes(model_inversion_service)

        for key, value in datasets_dict.iteritems():
            h5_file.create_dataset(key, data=value)

        h5_file.attrs.create(self.CUSTOM_TYPE_ATTRIBUTE, "HypothesisModel")
        h5_file.attrs.create(self.CUSTOM_SUBTYPE_ATTRIBUTE, model_inversion_service.__class__.__name__)

        for key, value in metadata_dict.iteritems():
            h5_file.attrs.create(key, value)

        h5_file.close()

    def write_dictionary(self, dictionary, path):
        """
        :param dictionary: dictionary to write in H5
        :param path: H5 path to be written
        """
        h5_file = h5py.File(path, 'a', libver='latest')

        for key, value in dictionary.iteritems():
            if isinstance(value, numpy.ndarray) and value.size > 0:
                h5_file.create_dataset(key, data=value)
            else:
                h5_file.attrs.create(key, value)

        h5_file.attrs.create(self.CUSTOM_TYPE_ATTRIBUTE, "HypothesisModel")
        h5_file.attrs.create(self.CUSTOM_SUBTYPE_ATTRIBUTE, dictionary.__class__.__name__)

        h5_file.close()

    def write_generic(self, object, folder, path):
        """
        :param object:
        :param path:H5 path to be written
        """
        convert_to_h5_model(object).write_to_h5(folder, path)

    def _determine_datasets_and_attributes(self, object):
        datasets_dict = {}
        metadata_dict = {}

        for key, value in vars(object).iteritems():
            if (isinstance(value, numpy.ndarray)) and value.size > 0:
                datasets_dict.update({key: value})
            else:
                if isinstance(value, (float, int, long, complex, str)):
                    metadata_dict.update({key: value})

        return datasets_dict, metadata_dict
