# coding=utf-8

import numpy as np

from tvb_epilepsy.base.utils.log_error_utils import initialize_logger
from tvb_epilepsy.plot.plotter import Plotter
from tvb_epilepsy.service.stochastic_parameter_factory import set_parameter

logger = initialize_logger(__name__)

if __name__ == "__main__":
    plotter = Plotter()
    x0 = set_parameter("x0", optimize_pdf=True, use="manual", x0_lo=0.0, x0_hi=2.0, x0_pdf="lognormal",
                       x0_pdf_params={"skew": 0.0, "mean": 0.5 / 0.05}, x0_mean=0.5, x0_std=0.05)

    axes, fig = plotter.plot_stochastic_parameter(x0, np.arange(-0.01, 2.0, 0.01))
    logger.info("Done")
