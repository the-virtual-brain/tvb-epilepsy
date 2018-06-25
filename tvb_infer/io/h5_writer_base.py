import h5py
import numpy

from tvb_infer.base.utils.log_error_utils import initialize_logger, warning
from tvb_infer.base.utils.file_utils import change_filename_or_overwrite
from tvb_infer.io.h5_model import convert_to_h5_model


class H5WriterBase(object):
    logger = initialize_logger(__name__)

    H5_TYPE_ATTRIBUTE = "EPI_Type"
    H5_SUBTYPE_ATTRIBUTE = "EPI_Subtype"

    def _determine_datasets_and_attributes(self, object, datasets_size=None):
        datasets_dict = {}
        metadata_dict = {}
        groups_keys = []

        try:
            if isinstance(object, dict):
                dict_object = object
            else:
                dict_object = vars(object)
            for key, value in dict_object.items():
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
        except:
            msg = "Failed to decompose group object: " + str(object) + "!"
            try:
                self.logger.info(str(object.__dict__))
            except:
                msg += "\n It has no __dict__ attribute!"
            warning(msg, self.logger)

        return datasets_dict, metadata_dict, groups_keys

    def _write_dicts_at_location(self, datasets_dict, metadata_dict, location):
        for key, value in datasets_dict.items():
            location.create_dataset(key, data=value)

        for key, value in metadata_dict.items():
            location.attrs.create(key, value)
        return location

    def _prepare_object_for_group(self, group, object, h5_type_attribute="HypothesisModel", nr_regions=None):
        group.attrs.create(self.H5_TYPE_ATTRIBUTE, h5_type_attribute)
        group.attrs.create(self.H5_SUBTYPE_ATTRIBUTE, object.__class__.__name__)
        datasets_dict, metadata_dict, subgroups = self._determine_datasets_and_attributes(object, nr_regions)
        group = self._write_dicts_at_location(datasets_dict, metadata_dict, group)
        return group, subgroups

    def write_object_to_file(self, path, object, h5_type_attribute="HypothesisModel", nr_regions=None):
        h5_file = h5py.File(change_filename_or_overwrite(path), 'a', libver='latest')
        h5_file, _ = self._prepare_object_for_group(h5_file, object, h5_type_attribute, nr_regions)
        h5_file.close()

    # TODO: this should be deprecated when/if _determine_datasets_and_attributes becomes recursive into groups
    def write_generic(self, object, path):
        """
        :param object:
        :param path:H5 path to be written
        """
        h5_model = convert_to_h5_model(object)

        h5_model.add_or_update_metadata_attribute(self.H5_TYPE_ATTRIBUTE, "HypothesisModel")
        h5_model.add_or_update_metadata_attribute(self.H5_SUBTYPE_ATTRIBUTE, object.__class__.__name__)

        h5_model.write_to_h5(path)