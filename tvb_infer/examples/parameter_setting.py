# coding=utf-8

import numpy as np

from tvb_infer.base.utils.log_error_utils import initialize_logger
from tvb_infer.plot.plotter import Plotter
from tvb_infer.service.probabilistic_parameter_builder import set_parameter

logger = initialize_logger(__name__)

if __name__ == "__main__":
    plotter = Plotter()
    x0 = set_parameter("x0", optimize_pdf=True, use="manual", x0_lo=0.0, x0_hi=2.0, x0_pdf="lognormal",
                       x0_pdf_params={"skew": 0.0, "mean": 0.5 / 0.05}, x0_mean=0.5, x0_std=0.05)

    axes, fig = plotter.plot_probabilistic_parameter(x0, np.arange(-0.01, 2.0, 0.01))

    # Testing for converting from symmetric matrix to two flattened columns and backwards:
    # a = np.array([[11, 12, 13, 14],
    #               [21, 22, 23, 24],
    #               [31, 32, 33, 34],
    #               [41, 42, 43, 44]])
    # b = np.stack([a[np.triu_indices(4, 1)], a.T[np.triu_indices(4, 1)]]).T
    # c = np.ones((4,4))
    # icon = -1
    # for ii in range(4):
    #     for jj in range(ii, 4):
    #         if (ii == jj):
    #             c[ii, jj] = 0
    #         else:
    #             icon += 1
    #             c[ii, jj] = b[icon, 0]
    #             c[jj, ii] = b[icon, 1]
    # plotter = Plotter()
    # x0 = set_parameter("x0", optimize_pdf=True, use="manual", x0_lo=0.0, x0_hi=2.0, x0_pdf="lognormal",
    #                   x0_pdf_params={"skew": 0.0, "mean": 0.5 / 0.05}, x0_mean=0.5, x0_std=0.05)
    #
    # axes, fig = plotter.plot_probabilistic_parameter(x0, np.arange(-0.01, 2.0, 0.01))

    logger.info("Done")
