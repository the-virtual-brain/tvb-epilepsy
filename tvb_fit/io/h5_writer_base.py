# -*- coding: utf-8 -*-

import h5py
import numpy
from tvb_fit.base.utils.log_error_utils import initialize_logger, warning
from tvb_fit.base.utils.data_structures_utils import is_numeric
from tvb_fit.base.utils.file_utils import change_filename_or_overwrite


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
                            # TODO: check how this works! Be carefull not to include lists and tuples if possible in tvb_fit classes!
                elif isinstance(object, (list, tuple)):
                    warning("Writing %s %s to h5 file as a numpy array dataset !" % (value.__class__, key), self.logger)
                    datasets_dict.update({key: numpy.array(value)})
                else:
                    if is_numeric(value) or isinstance(value, str):
                        metadata_dict.update({key: value})
                    elif not(callable(value)):
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
            try:
                location.create_dataset(key, data=value)
            except:
                warning("Failed to write to %s dataset %s %s:\n%s !" %
                        (str(location), value.__class__, key, str(value)), self.logger)

        for key, value in metadata_dict.items():
            try:
                location.attrs.create(key, value)
            except:
                warning("Failed to write to %s attribute %s %s:\n%s !" %
                        (str(location), value.__class__, key, str(value)), self.logger)
        return location

    def _prepare_object_for_group(self, group, object, h5_type_attribute="HypothesisModel", nr_regions=None,
                                  regress_subgroups=True):
        group.attrs.create(self.H5_TYPE_ATTRIBUTE, h5_type_attribute)
        group.attrs.create(self.H5_SUBTYPE_ATTRIBUTE, object.__class__.__name__)
        datasets_dict, metadata_dict, subgroups = self._determine_datasets_and_attributes(object, nr_regions)
        # If empty return None
        if len(datasets_dict) == len(metadata_dict) == len(subgroups) == 0:
            if isinstance(group, h5py._hl.files.File):
                if regress_subgroups:
                    return group
                else:
                    return group, subgroups
            else:
                return None
        else:
            if len(datasets_dict) > 0 or len(metadata_dict) > 0:
                if isinstance(group, h5py._hl.files.File):
                    group = self._write_dicts_at_location(datasets_dict, metadata_dict, group)
                else:
                    self._write_dicts_at_location(datasets_dict, metadata_dict, group)
            # Continue recursively going deeper in the object
            if regress_subgroups:
                for subgroup in subgroups:
                    child_object = getattr(object, subgroup, None)
                    if child_object is not None:
                        group.create_group(subgroup)
                        temp = self._prepare_object_for_group(group[subgroup], child_object,
                                                              h5_type_attribute, nr_regions)
                        # If empty delete it
                        if temp is None or len(temp.keys()) == 0:
                            del group[subgroup]

                return group
            else:
                return group, subgroups

    def write_object_to_file(self, path, object, h5_type_attribute="HypothesisModel", nr_regions=None):
        h5_file = h5py.File(change_filename_or_overwrite(path), 'a', libver='latest')
        h5_file = self._prepare_object_for_group(h5_file, object, h5_type_attribute, nr_regions)
        h5_file.close()
