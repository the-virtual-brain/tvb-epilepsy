
import numpy as np

from tvb_epilepsy.base.utils.log_error_utils import initialize_logger
from tvb_epilepsy.plot.plotter import Plotter
from tvb_epilepsy.service.stochastic_parameter_factory import set_parameter


logger = initialize_logger(__name__)

if __name__ == "__main__":
    plotter = Plotter()
    x = set_parameter("x", optimize_pdf=True, use="manual", x_lo=0.0, x_hi=2.0, x_pdf="lognormal",
                      x_pdf_params={"skew": 0.0, "mean": 0.1 / 0.025}, x_mean=0.1, x_std=0.025)

    axes, fig = plotter.plot_stochastic_parameter(x, np.arange(-0.01, 0.01, 0.001))

    logger.info("Done")
