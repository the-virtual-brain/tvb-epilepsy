
import os
import numpy
from tvb_epilepsy.base.utils import get_logger, assert_equal_objects, write_object_to_h5_file, read_object_from_h5_file
from tvb_epilepsy.base.equations import *
from tvb_epilepsy.base.symbolic_equations import *
from tvb_epilepsy.base.equilibrium_computation import *

if __name__ == "__main__":

    logger = get_logger(__name__)

    x1 = numpy.expand_dims(numpy.array([-4.0/3, -1.5, -5.0/3], dtype="float32"), 1).T
    w = numpy.array([[0,0.45,0.5], [0.45,0,0.55], [0.5,0.55, 0]])
    n = x1.size
    z = 3.0 * numpy.ones(x1.shape, dtype=x1.dtype)

    y = 10.25 * numpy.ones(x1.shape, dtype=x1.dtype)
    x2 = -0.33 * numpy.ones(x1.shape, dtype=x1.dtype)
    y2 = 0.0 * numpy.ones(x1.shape, dtype=x1.dtype)
    g = 0.1*x1 # * numpy.ones(x1.shape, dtype=x1.dtype)

    x0 = 0.5 * numpy.ones(x1.shape, dtype=x1.dtype)

    x0cr = 2.46 * numpy.ones(x1.shape, dtype=x1.dtype)
    r = 0.39815 * numpy.ones(x1.shape, dtype=x1.dtype)

    K = 0.1 * numpy.ones(x1.shape, dtype=x1.dtype)
    yc = 1.0 * numpy.ones(x1.shape, dtype=x1.dtype)
    Iext1 = 3.1 * numpy.ones(x1.shape, dtype=x1.dtype)
    slope = 0.0 * numpy.ones(x1.shape, dtype=x1.dtype)
    Iext2 = 0.45 * numpy.ones(x1.shape, dtype=x1.dtype)

    zmode = numpy.array("lin")
    pmode  = numpy.array("const")

    model =  "EpileptorDP2D"

    x1eq  = x1
    if model == "EpileptorDP2D":
        if zmode == 'sig':
            # 2D approximation, Proix et al 2014
            zeq = calc_eq_z_2d(x1eq, yc, Iext1)
        else:
            zeq = z
        equilibrium_point = numpy.r_[x1eq, zeq].astype('float32')
        model_vars = 2
        dfun = calc_dfun(x1eq, zeq, yc, Iext1, x0, x0cr, r, K, w, model_vars, zmode, pmode, x1_neg=True,
                         x2_neg=True, x0_var=x0, slope_var=slope, Iext1_var=Iext1, Iext2_var=Iext2, K_var=K,
                         slope=slope, a=1.0, b=-2.0, d=5.0, s=6.0, Iext2=Iext2, gamma=0.01, tau1=1.0, tau0=2857.0,
                         tau2=10.0)
    else:
        # all >=6D models
        y1eq = calc_eq_y1(x1eq, yc)
        zeq = calc_eq_z_6d(x1eq, y1eq, Iext1)

        geq = calc_eq_g(x1eq)

        if model == "EpileptorDPrealistic":
            # the 11D "realistic" simulations model
            from tvb_epilepsy.tvb_api.epileptor_models import EpileptorDPrealistic
            slope_eq, Iext2_eq = EpileptorDPrealistic.fun_slope_Iext2(zeq, geq, pmode, slope, Iext2)
            (x2eq, y2eq) = calc_eq_pop2(x1eq, zeq, Iext2_eq.T)
            equilibrium_point = numpy.r_[x1eq, y1eq, zeq, x2eq, y2eq, geq, x0, slope_eq.T, Iext1, Iext2_eq.T, K].astype('float32')
            model_vars = 11
            dfun = calc_dfun(x1eq, zeq, yc, Iext1, x0, x0cr, r, K, w, model_vars, zmode, pmode, x1_neg=True,
                             y1=y1eq, x2=x2eq, y2=y2eq, g=geq, x2_neg=True,
                             x0_var=x0, slope_var=slope_eq, Iext1_var=Iext1, Iext2_var=Iext2_eq, K_var=K,
                             slope=slope, a=1.0, b=-2.0, d=5.0, s=6.0, Iext2=Iext2, gamma=0.01, tau1=1.0, tau0=2857.0,
                             tau2=10.0)
        else:
            # all >=6D models
            (x2eq, y2eq) = calc_eq_pop2(x1eq, zeq, Iext2)
            equilibrium_point = numpy.r_[x1eq, y1eq, zeq, x2eq, y2eq, geq].astype('float32')
            model_vars = 6
            dfun=calc_dfun(x1eq, zeq, yc, Iext1, x0, x0cr, r, K, w, model_vars, zmode, pmode, x1_neg=True,
                      y1=y1eq, x2=x2eq, y2=y2eq, g=geq, x2_neg=True,
                      x0_var=x0, slope_var=slope, Iext1_var=Iext1, Iext2_var=Iext2, K_var=K,
                      slope=slope, a=1.0, b=-2.0, d=5.0, s=6.0, Iext2=Iext2, gamma=0.01, tau1=1.0, tau0=2857.0,
                      tau2=10.0)

    print equilibrium_point

    print dfun