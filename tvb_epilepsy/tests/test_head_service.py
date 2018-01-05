import os
from tvb_epilepsy.base.constants.configurations import FOLDER_LOGS, FOLDER_RES, FOLDER_FIGURES
from tvb_epilepsy.base.model.vep.head import Head
from tvb_epilepsy.base.model.vep.surface import Surface
from tvb_epilepsy.custom.readers_custom import CustomReader
from tvb_epilepsy.service.head_service import HeadService
from tvb_epilepsy.tests.base import get_temporary_folder, remove_temporary_test_files

data_dir = "data"


class TestHeadService():
    head_service = HeadService()

    @classmethod
    def setup_class(cls):
        for direc in (FOLDER_LOGS, FOLDER_RES, FOLDER_FIGURES):
            if not os.path.exists(direc):
                os.makedirs(direc)

    def _prepare_dummy_head(self):
        reader = CustomReader()
        # TODO: add sensors to head to increase coverage
        connectivity = reader.read_connectivity(os.path.join(data_dir, "Connectivity.h5"))
        cort_surface = Surface([], [])
        head = Head(connectivity, cort_surface)

        return head

    # def test_compute_nearest_regions_to_sensors(self):

    def test_write_head_folder(self):
        head = self._prepare_dummy_head()

        head_folder = os.path.join(get_temporary_folder(), "Head_dummy")

        assert not os.path.exists(head_folder)

        self.head_service.write_head_folder(head, head_folder)

        assert os.path.exists(head_folder)
        assert os.listdir(head_folder) is not []

    def test_plot_head(self):
        head = self._prepare_dummy_head()
        filename1 = "Connectivity_.png"
        filename2 = "HeadStats.png"

        assert not os.path.exists(os.path.join(FOLDER_FIGURES, filename1))
        assert not os.path.exists(os.path.join(FOLDER_FIGURES, filename2))

        self.head_service.plot_head(head, save_flag=True, show_flag=False, figure_dir=FOLDER_FIGURES)

        assert os.path.exists(os.path.join(FOLDER_FIGURES, filename1))
        assert os.path.exists(os.path.join(FOLDER_FIGURES, filename2))

    # def test_select_sensors_power(self):

    # def test_select_sensors_rois(self):

    # def test_select_sensors_corr(self):

    @classmethod
    def teardown_class(cls):
        remove_temporary_test_files()
        for direc in (FOLDER_LOGS, FOLDER_RES, FOLDER_FIGURES):
            for dir_file in os.listdir(direc):
                os.remove(os.path.join(os.path.abspath(direc), dir_file))
            os.removedirs(direc)
