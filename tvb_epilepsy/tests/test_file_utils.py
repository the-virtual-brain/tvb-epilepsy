import os
from tvb_epilepsy.base.utils.file_utils import change_filename_or_overwrite
from tvb_epilepsy.io.h5_writer import H5Writer
from tvb_epilepsy.tests.base import get_temporary_folder


class TestFileUtils(object):
    writer = H5Writer()

    def test_change_filename_or_overwrite_always_overwrite(self):
        filename = "Test.h5"
        test_file = os.path.join(get_temporary_folder(), filename)
        self.writer.write_dictionary({"a": [1, 2, 3]}, test_file)

        assert os.path.exists(test_file)

        change_filename_or_overwrite(test_file, True)

        assert not os.path.exists(test_file)
