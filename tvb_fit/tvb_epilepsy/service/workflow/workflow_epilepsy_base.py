from abc import ABCMeta  # abstractmethod,

from tvb_fit.service.workflow_service_base import WorkflowServiceBase

from tvb_fit.tvb_epilepsy.base.constants.config import Config
from tvb_fit.tvb_epilepsy.io.h5_reader import H5Reader
from tvb_fit.tvb_epilepsy.io.h5_writer import H5Writer
from tvb_fit.tvb_epilepsy.plot.plotter import Plotter

from tvb_scripts.io.tvb_data_reader import TVBReader


class WorkflowEpilepsyBase(WorkflowServiceBase):
    __metaclass__ = ABCMeta

    def __init__(self, config=Config(), reader=None, writer=None, plotter=None):
        if not isinstance(config, Config):
            config = Config()
        if not isinstance(reader, (H5Reader, TVBReader)):
            if config.input.IS_TVB_MODE:
                reader = TVBReader()
            else:
                reader = H5Reader()
        if not isinstance(writer, H5Writer):
            writer = H5Writer()
        if not isinstance(plotter, Plotter):
            plotter = Plotter(config)
        super(WorkflowEpilepsyBase, self).__init__(config, reader, writer, plotter)
