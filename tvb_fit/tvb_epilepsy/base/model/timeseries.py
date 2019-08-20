from copy import deepcopy
from enum import Enum

import numpy as np

from tvb_utils.log_error_utils import initialize_logger
from tvb_timeseries.model.timeseries import Timeseries as TimeseriesBase, TimeseriesDimensions


class PossibleVariables(Enum):
    X1 = "x1"
    X2 = "x2"
    LFP = "lfp"
    SOURCE = "source"
    Z = "z"
    Y1 = "y1"
    Y2 = "y2"
    G = "g"
    SLOPE_T = "slope_t"
    IEXT2_T = "Iext2_t"
    IEXT1_T = "Iext1_t"
    K_T = "K_t"
    X0_T = "x0_t"
    SEEG = "seeg"


class Timeseries(TimeseriesBase):

    logger = initialize_logger(__name__)

    def get_source(self):
        if TimeseriesDimensions.VARIABLES.value not in self.labels_dimensions.keys():
            self.logger.error("No state variables are defined for this instance!")
            raise ValueError

        if PossibleVariables.SOURCE.value in self.labels_dimensions[TimeseriesDimensions.VARIABLES.value]:
            return self.get_variables(PossibleVariables.SOURCE.value)
        if PossibleVariables.X1.value in self.labels_dimensions[TimeseriesDimensions.VARIABLES.value]:
            y0_ts = self.get_variables(PossibleVariables.X1.value)
            if PossibleVariables.X2.value in self.labels_dimensions[TimeseriesDimensions.VARIABLES.value]:
                self.logger.info("%s is computed using %s and %s state variables!" % (
                    PossibleVariables.SOURCE.value, PossibleVariables.X1.value, PossibleVariables.X2.value))
                y2_ts = self.get_variables(PossibleVariables.X2.value)
                source_data = y2_ts.data - y0_ts.data
            else:
                self.logger.warn("%s is computed using %s state variable!" % (
                    PossibleVariables.SOURCE.value, PossibleVariables.X1.value))
                source_data = -y0_ts.data
            source_dim_labels = deepcopy(self.labels_dimensions)
            source_dim_labels[self.labels_ordering[1]] = np.array([PossibleVariables.SOURCE.value])
            return self.duplicate(data=source_data, labels_dimensions=source_dim_labels)

        self.logger.error(
            "%s is not computed and cannot be computed now because state variables %s and %s are not defined!" % (
                PossibleVariables.SOURCE.value, PossibleVariables.X1.value, PossibleVariables.X2.value))
        raise ValueError


