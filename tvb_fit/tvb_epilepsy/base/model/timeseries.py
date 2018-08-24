
from enum import Enum
from collections import OrderedDict
from tvb_fit.base.utils.log_error_utils import initialize_logger
from tvb_fit.base.model.timeseries import Timeseries as TimeseriesBase, TimeseriesDimensions


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
        if TimeseriesDimensions.VARIABLES.value not in self.dimension_labels.keys():
            self.logger.error("No state variables are defined for this instance!")
            raise ValueError

        if PossibleVariables.SOURCE.value in self.dimension_labels[TimeseriesDimensions.VARIABLES.value]:
            return self.get_state_variable(PossibleVariables.SOURCE.value)
        if PossibleVariables.X1.value in self.dimension_labels[TimeseriesDimensions.VARIABLES.value]:
            y0_ts = self.get_state_variable(PossibleVariables.X1.value)
            if PossibleVariables.X2.value in self.dimension_labels[TimeseriesDimensions.VARIABLES.value]:
                self.logger.info("%s is computed using %s and %s state variables!" % (
                    PossibleVariables.SOURCE.value, PossibleVariables.X1.value, PossibleVariables.X2.value))
                y2_ts = self.get_state_variable(PossibleVariables.X2.value)
                source_data = y2_ts.data - y0_ts.data
            else:
                self.logger.warn("%s is computed using %s state variable!" % (
                    PossibleVariables.SOURCE.value, PossibleVariables.X1.value))
                source_data = -y0_ts.data
            source_dim_labels = OrderedDict(
                {TimeseriesDimensions.SPACE.value: self.dimension_labels[TimeseriesDimensions.SPACE.value],
                 TimeseriesDimensions.VARIABLES.value: [PossibleVariables.SOURCE.value]})
            return Timeseries(source_data, source_dim_labels, self.time_start, self.time_step, self.time_unit)
        self.logger.error(
            "%s is not computed and cannot be computed now because state variables %s and %s are not defined!" % (
                PossibleVariables.SOURCE.value, PossibleVariables.X1.value, PossibleVariables.X2.value))
        raise ValueError


