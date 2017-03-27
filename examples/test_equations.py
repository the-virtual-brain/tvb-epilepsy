
import os
import numpy
from tvb_epilepsy.base.utils import assert_equal_objects, write_object_to_h5_file, read_object_from_h5_file
from tvb_epilepsy.base.constants import *
from tvb_epilepsy.base.calculations import *
from tvb_epilepsy.base.symbolic import * 
from tvb_epilepsy.base.equilibrium_computation import *

if __name__ == "__main__":

    x1 = numpy.expand_dims(numpy.array([-4.0/3, -1.5, -5.0/3], dtype="float32"), 1).T
    x1eq = x1
    w = numpy.array([[0,0.45,0.5], [0.45,0,0.55], [0.5,0.55, 0]])
    n = x1.size

    # y1 = 10.25 * numpy.ones(x1.shape, dtype=x1.dtype)
    # x2 = -0.33 * numpy.ones(x1.shape, dtype=x1.dtype)
    # y2 = 0.0 * numpy.ones(x1.shape, dtype=x1.dtype)
    # g = 0.1*x1 # * numpy.ones(x1.shape, dtype=x1.dtype)

    K = 0.1 * numpy.ones(x1.shape, dtype=x1.dtype)
    yc = 1.0 * numpy.ones(x1.shape, dtype=x1.dtype)
    Iext1 = 3.1 * numpy.ones(x1.shape, dtype=x1.dtype)
    slope = 0.0 * numpy.ones(x1.shape, dtype=x1.dtype)
    Iext2 = 0.45 * numpy.ones(x1.shape, dtype=x1.dtype)

    x1, K = assert_arrays([x1, K])
    w = assert_arrays([w]) #, (x1.size, x1.size)

    # Update zeq given the specific model, and assuming the hypothesis x1eq for the moment in the context of a 2d model:
    # It is assumed that the model.x0 has been adjusted already at the phase of model creation
    zeq = calc_eq_z_2d(x1eq, yc, Iext1)
    print "zeq="
    print zeq
    z = zeq

    x0cr, r = calc_x0cr_r(yc, Iext1, zmode=numpy.array("lin"), x1_rest=X1_DEF, x1_cr=X1_EQ_CR_DEF, x0def=X0_DEF,
                           x0cr_def=X0_CR_DEF)
    print "x0cr, r="
    print x0cr, r

    x0 = calc_x0(x1eq, zeq, K, w, x0cr, r, model="2d", zmode=numpy.array("lin"), z_pos=True)
    print "x0="
    print x0

    zmode = numpy.array("lin")
    pmode = numpy.array("const")

    model = "EpileptorDP"

    if model == "EpileptorDP2D":

        # 2D approximation, Proix et al 2014
        eq = numpy.r_[x1eq, zeq].astype('float32')
        model_vars = 2
        dfun = calc_dfun(x1eq, zeq, yc, Iext1, x0, K, w, model_vars, x0cr=x0cr, r=r,
                         zmode=zmode, pmode=pmode,
                         x0_var=x0, slope_var=slope, Iext1_var=Iext1, Iext2_var=Iext2, K_var=K, slope=slope, 
                         a=1.0, b=-2.0, d=5.0, s=6.0, Iext2=Iext2, gamma=0.1, tau1=1.0, tau0=2857.0, tau2=10.0, 
                         output_mode="arrays")

        jac = calc_jac(x1eq, zeq, yc, Iext1, x0, K, w, model_vars, x0cr=x0cr, r=r, 
                       zmode=zmode, pmode=pmode, x1_neg=True, z_pos=True, x2_neg=True,
                       x0_var=x0, slope_var=slope, Iext1_var=Iext1, Iext2_var=Iext2, K_var=K,
                       slope=slope, a=1.0, b=-2.0, d=5.0, s=6.0, Iext2=Iext2, gamma=0.1, tau1=1.0, tau0=2857.0,
                       tau2=10.0)

    else:

        x0_6d = rescale_x0(x0, yc, Iext1, zmode=numpy.array("lin"))
        print "x0_6d="
        print x0_6d

        if model == "EpileptorDPrealistic":

            # the 11D "realistic" simulations model
            from tvb_epilepsy.tvb_api.epileptor_models import EpileptorDPrealistic

            eq, slope_eq, Iext2_eq = calc_eq_11d(zeq, yc, Iext1, Iext2, slope, x0_6d, K,
                                                                EpileptorDPrealistic.fun_slope_Iext2, a=1.0, b=-2.0,
                                                                d=0.5, gamma=0.1, pmode=pmode)
            model_vars = 11
            dfun = calc_dfun(x1eq, zeq, yc, Iext1, r, K, w, model_vars, zmode, pmode,
                             y1=eq[1], x2=eq[3], y2=eq[4], g=eq[5],
                             x0_var=eq[6], slope_var=eq[7], Iext1_var=eq[8], Iext2_var=eq[9], K_var=eq[10],
                             slope=slope, a=1.0, b=3.0, d=5.0, s=6.0, Iext2=Iext2, gamma=0.1, tau1=1.0, tau0=2857.0,
                             tau2=10.0, output_mode="array")

            jac = calc_jac(x1eq, zeq, yc, Iext1, x0, K, w, model_vars,
                           zmode, pmode, x1_neg=True, z_pos=True, x2_neg=True,
                           y1=eq[1], x2=eq[3], y2=eq[4], g=eq[5],
                           x0_var=eq[6], slope_var=eq[7], Iext1_var=eq[8], Iext2_var=eq[9], K_var=eq[10],
                           slope=slope, a=1.0, b=3.0, d=5.0, s=6.0, Iext2=Iext2, gamma=0.1, tau1=1.0, tau0=2857.0,
                           tau2=10.0)

        else:

            # all >=6D models
            eq = calc_eq_6d(zeq, yc.T, Iext1.T, Iext2.T, a=1.0, b=3.0, d=5.0, gamma=0.1)
            model_vars = 6
            dfun = calc_dfun(x1eq, zeq, yc, Iext1, r, K, w, model_vars, zmode,
                             y1=eq[1], x2=eq[3], y2=eq[4], g=eq[5],
                             slope=slope, a=1.0, b=3.0, d=5.0, s=6.0, Iext2=Iext2, gamma=0.1, tau1=1.0, tau0=2857.0,
                             tau2=10.0, output_mode="array")

            jac = calc_jac(x1eq, zeq, yc, Iext1, r, K, w, model_vars,
                           zmode, x1_neg=True, z_pos=True, x2_neg=True,
                           y1=eq[1], x2=eq[3], y2=eq[4], g=eq[5],
                           slope=slope, a=1.0, b=3.0, d=5.0, s=6.0, Iext2=Iext2, gamma=0.1, tau1=1.0, tau0=2857.0,
                           tau2=10.0)

    write_object_to_h5_file({"eq": eq, "dfun": numpy.array(dfun), "jac": numpy.array(jac)},
                            os.path.join(FOLDER_RES, model+"Symbolic"+str(SYMBOLIC_CALCULATIONS_FLAG)+".h5"))

    # Test coupling:
    x = numpy.reshape(symbol_vars(3, ["x1"])[0], (1, 3))
    K = numpy.reshape(symbol_vars(3, ["K"])[0], (1, 3))
    w = symbol_vars(3, ["w"], dims=2)[0]
    coupling = calc_coupling(x.T, K, w)
    print coupling.flatten().tolist(), coupling.shape
    print symbol_eqtn_coupling(3)[1]

    # Test calc_fx1_2d_taylor
    x = symbol_vars(3, ["x1"])[0]
    x_taylor = symbol_vars(3, ["x1lin"])[0]  # x_taylor = -4.5/3 (=x1lin)
    fx1lin = calc_fx1_2d_taylor(x, x_taylor, z, yc, Iext1, slope, a=1.0, b=-2, tau1=1.0, x1_neg=True, order=2).flatten()
    for ii in range(3):
        print fx1lin[ii].expand(x[ii]).collect(x[ii])

    # Test calc_fx1y1_6d_diff_x1
    x = symbol_vars(3, ["x1"])[0]
    fx1y1_6d_diff_x1 = calc_fx1y1_6d_diff_x1(x, yc, Iext1, a=1.0, b=3.0, d=5.0, tau1=1.0).flatten()
    for ii in range(3):
        print fx1y1_6d_diff_x1[ii].expand(x[ii]).collect(x[ii])

    # test eq_x1_hypo_x0_optimize_jac
    ix0 = numpy.array([1, 2])
    iE = numpy.array([0])
    # x = numpy.empty_like(x1).flatten()
    # x[ix0] = x1eq[0, ix0]
    # x[iE] = x0[0, iE]
    # jac = eq_x1_hypo_x0_optimize_jac(x, ix0, iE, x1, z, x0[0, ix0], x0cr, r, yc, Iext1, K, w)
    vx = symbol_eqtn_fx1(3, model="2d", x1_neg=True, slope="slope", Iext1="Iext1")[2]
    x = numpy.empty_like(vx["x1"]).flatten()
    x[ix0] = vx["x1"][ix0]
    vz = symbol_eqtn_fz(3, zmode=numpy.array("lin"), z_pos=True, model="2d", x0="x0", K="K")[2]
    x[iE] = vz["x0"][iE]
    p = x1.shape
    numpy.fill_diagonal(vz["w"], 0.0)
    vx["x1"], vx["z"], vz["x0cr"], vz["r"], vx["y1"], vx["Iext1"], vz["K"] = \
        assert_arrays([vx["x1"], vx["z"], vz["x0cr"], vz["r"], vx["y1"], vx["Iext1"], vz["K"]],
                      (1, vx["x1"].size))
    jac = eq_x1_hypo_x0_optimize_jac(x, ix0, iE, vx["x1"], vx["z"], vz["x0"][ix0], vz["x0cr"], vz["r"], vx["y1"],
                                     vx["Iext1"], vz["K"], vz["w"])

    print jac