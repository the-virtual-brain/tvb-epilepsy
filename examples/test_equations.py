
import os
import numpy
from tvb_epilepsy.base.utils import assert_equal_objects, write_object_to_h5_file, read_object_from_h5_file
from tvb_epilepsy.base.constants import *
from tvb_epilepsy.base.calculations import *
from tvb_epilepsy.base.symbolic import * 
from tvb_epilepsy.base.equilibrium_computation import *

if __name__ == "__main__":

    n = 3
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

    zmode = numpy.array("lin")
    pmode = numpy.array("const")

    z = calc_eq_z_2d(x1, yc, Iext1)

    x0cr, r = calc_x0cr_r(yc, Iext1, zmode=zmode, x1_rest=X1_DEF, x1_cr=X1_EQ_CR_DEF, x0def=X0_DEF,
                           x0cr_def=X0_CR_DEF)  #test=True
    print "x0cr, r="
    print x0cr, r

    x0 = calc_x0(x1, z, K, w, x0cr, r, model="2d", zmode=zmode, z_pos=True)
    print "x0="
    print x0



    model = "EpileptorDP"
    print model

    b = -2.0

    # 2D approximation, Proix et al 2014
    if numpy.all(b == -2.0):
        x1eq = x1
        zeq = z
    else:
        x1eq = calc_eq_x1(yc, Iext1, x0, K, w, a=1.0, b=-2.0, d=5.0, x0cr=x0cr, r=r, zmode=zmode, model="2d")
        zeq = calc_eq_z_2d(x1eq, yc, Iext1)

    if model == "EpileptorDP2D":

        eq = numpy.r_[x1eq, zeq].astype('float32')

        model_vars = 2
        dfun = calc_dfun(eq[0], eq[1], yc, Iext1, x0, K, w, model_vars, x0cr=x0cr, r=r,
                         zmode=zmode, pmode=pmode,
                         x0_var=x0, slope_var=slope, Iext1_var=Iext1, Iext2_var=Iext2, K_var=K, slope=slope, 
                         a=1.0, b=-2.0, d=5.0, s=6.0, Iext2=Iext2, gamma=0.1, tau1=1.0, tau0=2857.0, tau2=10.0, 
                         output_mode="array")

        jac = calc_jac(eq[0], eq[1], yc, Iext1, x0, K, w, model_vars, x0cr=x0cr, r=r,
                       zmode=zmode, pmode=pmode, x1_neg=True, z_pos=True, x2_neg=False,
                       x0_var=x0, slope_var=slope, Iext1_var=Iext1, Iext2_var=Iext2, K_var=K,
                       slope=slope, a=1.0, b=-2.0, d=5.0, s=6.0, Iext2=Iext2, gamma=0.1, tau1=1.0, tau0=2857.0,
                       tau2=10.0)

    else:

        x0_6d = calc_rescaled_x0(x0, yc, Iext1, zmode=zmode)
        print "x0_6d="
        print x0_6d

        if model == "EpileptorDPrealistic":

            # the 11D "realistic" simulations model
            from tvb_epilepsy.tvb_api.epileptor_models import EpileptorDPrealistic

            eq, slope_eq, Iext2_eq = calc_eq_11d(x0_6d, K, w, yc, Iext1, Iext2, slope,
                                                 EpileptorDPrealistic.fun_slope_Iext2, x1, -2.0, a=1.0, b=3.0, d=5.0,
                                                 zmode=zmode, pmode=pmode)
            model_vars = 11
            dfun = calc_dfun(eq[0], eq[2], yc, Iext1, x0_6d, K, w, model_vars, x0cr, r, zmode, pmode,
                             y1=eq[1], x2=eq[3], y2=eq[4], g=eq[5],
                             x0_var=eq[6], slope_var=eq[7], Iext1_var=eq[8], Iext2_var=eq[9], K_var=eq[10],
                             slope=slope, a=1.0, b=3.0, d=5.0, s=6.0, Iext2=Iext2, gamma=0.1, tau1=1.0, tau0=2857.0,
                             tau2=10.0, output_mode="array")

            jac = calc_jac(eq[0], eq[2], yc, Iext1, x0, K, w, model_vars,
                           zmode, pmode, x1_neg=True, z_pos=True, x2_neg=False,
                           y1=eq[1], x2=eq[3], y2=eq[4], g=eq[5],
                           x0_var=eq[6], slope_var=eq[7], Iext1_var=eq[8], Iext2_var=eq[9], K_var=eq[10],
                           slope=slope, a=1.0, b=3.0, d=5.0, s=6.0, Iext2=Iext2, gamma=0.1, tau1=1.0, tau0=2857.0,
                           tau2=10.0)

        else:

            # all >=6D models
            eq = calc_eq_6d(x0_6d, K, w,  yc, Iext1, Iext2, x1, -2.0, a=1.0, b=3.0, d=5.0, zmode=zmode)
            model_vars = 6
            dfun = calc_dfun(eq[0], eq[2], yc, Iext1, x0_6d, K, w, model_vars, x0cr, r, zmode,
                             y1=eq[1], x2=eq[3], y2=eq[4], g=eq[5],
                             slope=slope, a=1.0, b=3.0, d=5.0, s=6.0, Iext2=Iext2, gamma=0.1, tau1=1.0, tau0=2857.0,
                             tau2=10.0, output_mode="array")

            jac = calc_jac(eq[0], eq[2], yc, Iext1, r, K, w, model_vars,
                           zmode, x1_neg=True, z_pos=True, x2_neg=False,
                           y1=eq[1], x2=eq[3], y2=eq[4], g=eq[5],
                           slope=slope, a=1.0, b=3.0, d=5.0, s=6.0, Iext2=Iext2, gamma=0.1, tau1=1.0, tau0=2857.0,
                           tau2=10.0)

    write_object_to_h5_file({"eq": eq, "dfun": numpy.array(dfun), "jac": numpy.array(jac)},
                            os.path.join(FOLDER_RES, model+"Symbolic"+str(SYMBOLIC_CALCULATIONS_FLAG)+".h5"))

    model = str(model_vars)+"d"

    sx1, sy1, sz, sx2, sy2, sg, sx0, sx0cr, sr, sK, syc, sIext1, sIext2, sslope, sa, sb, sd, stau1, stau0, stau2, v = \
    symbol_vars(n, ["x1", "y1", "z", "x2", "y2", "g", "x0", "x0cr", "r", "K", "yc", "Iext1", "Iext2",
                    "slope", "a", "b", "d", "tau1", "tau0", "tau2"], shape=(1, 3))
    sw, vw = symbol_vars(n, ["w"], dims=2, output_flag="numpy_array")
    v.update(vw)
    del vw
    numpy.fill_diagonal(sw, 0.0)
    sw = Array(sw)

    a = numpy.ones((1,n))
    b2 = -2.0 * a
    b6 = 3.0 * a
    d = 5.0 * a
    s = 6.0 * a
    tau1 = a
    tau0 = a
    tau2 = a
    x1sq = -4.0/3 * a
    if model == "2d":
        y1 = yc
    else:
        y1 = eq[1]
        x2 = eq[3]
        y2 = eq[4]
        g = eq[5]
        if model == "11d":
            x0_var = eq[6]
            slope_var = eq[7]
            Iext1_var = eq[8]
            Iext2_var = eq[9]
            K_var = eq[10]

    print "\nTest symbolic x0cr, r calculation:"
    x0cr2, r2 = calc_x0cr_r(syc, sIext1, zmode=zmode, x1_rest=X1_DEF, x1_cr=X1_EQ_CR_DEF, x0def=X0_DEF,
                              x0cr_def=X0_CR_DEF) #test=True
    print x0cr2, x0cr2.shape
    print r2, r2.shape
    print calc_x0cr_r(yc, Iext1, zmode=zmode, x1_rest=X1_DEF, x1_cr=X1_EQ_CR_DEF, x0def=X0_DEF, x0cr_def=X0_CR_DEF)
    lx0cr_r, sx0cr_r, v = symbol_eqtn_x0cr_r(n, zmode=zmode, shape=(1, 3))  #symbol_calc_x0cr_r(n, zmode=zmode, shape=(1, 3))
    sx0cr_r = list(sx0cr_r)
    for ii in range(2):
        sx0cr_r[ii] = Matrix(sx0cr_r[ii])
        for iv in range(n):
            sx0cr_r[ii][iv] = sx0cr_r[ii][iv].subs([(v["a"][0, iv], a[0, iv]), (v["b"][0, iv], b2[0, iv]),
                                                    (v["x1_rest"][0, iv], X1_DEF), (v["x0_rest"][0, iv], X0_DEF),
                                                    (v["x1_cr"][0, iv], X1_EQ_CR_DEF), (v["x0_cr"][0, iv], X0_CR_DEF)])
            print sx0cr_r[ii][iv]
        print sx0cr_r[ii].shape
        print lx0cr_r[ii](yc, Iext1, a, b2, X1_DEF*a, X1_EQ_CR_DEF*a, X0_DEF*a, X0_CR_DEF*a)

    print "\nTest coupling:"
    coupling = calc_coupling(sx1, sK, sw)
    print coupling, coupling.shape, calc_coupling(x1, K, w)
    scoupling = symbol_eqtn_coupling(n, shape=(1, n))[:2]
    print scoupling[1], scoupling[1].shape, scoupling[0](x1, K, w)

    print "\nTest coupling derivative to x1"
    coupling_diff = calc_coupling_diff(sK, sw)
    for ii in range(n):
        print coupling_diff[ii]
    print coupling_diff.shape, calc_coupling_diff(K, w)
    scoupling_diff = symbol_calc_coupling_diff(n, ix=None, jx=None, K="K")[:2]
    for ii in range(n):
        print scoupling_diff[1][ii]
    print scoupling_diff[1].shape, scoupling_diff[0](K, w)

    print "\nTest the fz with substitution of z via fx1"
    fx1z = calc_fx1z(sx1, sx0, sK, sw, syc, sIext1, sx0cr, sr, sa, sb, sd, stau1, stau0, model=model,
                     zmode=zmode)
    for ii in range(n):
        print fx1z[0, ii]
    if model == "2d":
        print fx1z.shape, calc_fx1z(x1, x0, K, w, yc, Iext1, x0cr, r, a=1.0, b=-2.0, d=5.0, tau1=1.0, tau0=1.0,
                                      model=model, zmode=zmode)
    else:
        print fx1z.shape, calc_fx1z(x1, x0_6d, K, w, yc, Iext1, x0cr, r, a=1.0, b=3.0, d=5.0, tau1=1.0, tau0=1.0,
                                    model=model, zmode=zmode)
    sfx1z = symbol_eqtn_fx1z(n, model, zmode, shape=(1, n))[:2]
    for ii in range(n):
        print sfx1z[1][0, ii]
    if model == "2d":
        print sfx1z[1].shape, sfx1z[0](x1, x0, K, w, x0cr, r, yc, Iext1, a, b2, tau1, tau0)
    else:
        print sfx1z[1].shape, sfx1z[0](x1, x0_6d, K, w, yc, Iext1, a, b6, d, tau1, tau0)

    print "\nTest the derivative to x1 of fz with substitution of z via fx1"
    fx1z_diff = calc_fx1z_diff(sx1, sK, sw, sa, sb, sd, stau1, stau0, model=model, zmode=zmode)
    for ii in range(n):
        print fx1z_diff[ii]
    if model == "2d":
        print fx1z_diff.shape, calc_fx1z_diff(x1, K, w, a=1.0, b=-2.0, d=5.0, tau1=1.0, tau0=1.0, model=model,
                                              zmode=zmode)
    else:
        print fx1z_diff.shape, calc_fx1z_diff(x1, K, w, a=1.0, b=3.0, d=5.0, tau1=1.0, tau0=1.0, model=model,
                                              zmode=zmode)
    sfx1z_diff = symbol_eqtn_fx1z_diff(n, model, zmode)[:2]
    for ii in range(n):
        print sfx1z_diff[1][ii, :]
    if model == "2d":
        print sfx1z_diff[1].shape, sfx1z_diff[0](x1, K, w, a, b2, tau1, tau0)
    else:
        print sfx1z_diff[1].shape, sfx1z_diff[0](x1, K, w, a, b6, d, tau1, tau0)

    if model != "2d":
        print "\nTest symbolic fx2 with substitution of y2 via fy2"
        sfx2y2 = symbol_eqtn_fx2y2(n, x2_neg=False, shape=(1, n))[:2]
        for ii in range(n):
            print sfx2y2[1][0, ii]
        print sfx2y2[1].shape, sfx2y2[0](x2, z, g, Iext2, s, tau1)

    print "\nTest calc_fx1_2d_taylor"
    x_taylor = symbol_vars(n, ["x1lin"], shape=(1, n))[0]  # x_taylor = -4.5/3 (=x1lin)
    fx1lin = calc_fx1_2d_taylor(sx1, x_taylor, sz, syc, sIext1, sslope, sa, sb, stau1, x1_neg=True, order=2,
                                shape=(1, n))
    sfx1lin = symbol_calc_2d_taylor(n, "x1lin", order=2, x1_neg=True, slope="slope", Iext1="Iext1", shape=(1, n))[:2]
    print fx1lin.shape, calc_fx1_2d_taylor(x1, -1.5, z, yc, Iext1, slope, a=1.0, b=-2, tau1=1.0, x1_neg=True, order=2,
                                           shape=(1, n))
    for ii in range(3):
        print fx1lin[0, ii].expand(sx1[0, ii]).collect(sx1[0, ii])
        print sfx1lin[1][0, ii].expand(sx1[0, ii]).collect(sx1[0, ii])
    print sfx1lin[1].shape, sfx1lin[0](x1, -1.5 * numpy.ones(x1.shape), z, yc, Iext1, slope, a, b2, tau1)

    print "\nTest calc_fx1y1_6d_diff_x1"
    fx1y1_6d_diff_x1 = calc_fx1y1_6d_diff_x1(sx1, syc, sIext1, sa, sb, sd, stau1, stau0)
    sfx1y1_6d_diff_x1 = symbol_calc_fx1y1_6d_diff_x1(n, shape=(1, n))[:2]
    print fx1y1_6d_diff_x1.shape, calc_fx1y1_6d_diff_x1(x1, yc, Iext1, a, b2, d, tau1, tau0)
    for ii in range(n):
        print fx1y1_6d_diff_x1[0, ii].expand(sx1[0, ii]).collect(sx1[0, ii])
        print sfx1y1_6d_diff_x1[1][0, ii].expand(sx1[0, ii]).collect(sx1[0, ii])
    print sfx1y1_6d_diff_x1[1].shape, sfx1y1_6d_diff_x1[0](x1, yc, Iext1, a, b2, d, tau1)

    print "\nTest eq_x1_hypo_x0_optimize_fun & eq_x1_hypo_x0_optimize_jac"
    ix0 = numpy.array([1, 2])
    iE = numpy.array([0])
    # x = numpy.empty_like(x1).flatten()
    # x[ix0] = x1eq[0, ix0]
    # x[iE] = x0[0, iE]
    # jac = eq_x1_hypo_x0_optimize_jac(x, ix0, iE, x1, z, x0[0, ix0], x0cr, r, yc, Iext1, K, w)
    x = numpy.empty_like(sx1).flatten()
    x[ix0] = sx1[0, ix0]
    vz = symbol_eqtn_fz(n, zmode=numpy.array("lin"), z_pos=True, model="2d", x0="x0", K="K")[2]
    x[iE] = sx0[0, iE]
    p = x1.shape
    opt_fun = eq_x1_hypo_x0_optimize_fun(x, ix0, iE, sx1, numpy.array(sz), sx0[0, ix0], sx0cr, sr, syc, sIext1, sK, sw)
    print "opt_fun: ", opt_fun
    opt_jac = eq_x1_hypo_x0_optimize_jac(x, ix0, iE, sx1, numpy.array(sz), sx0[0, ix0], sx0cr, sr, sy1, sIext1, sK, sw)
    print "opt_jac: ", opt_jac

    x1EQopt, x0solopt = eq_x1_hypo_x0_optimize(ix0, iE, x1eq, zeq, x0[:, ix0], x0cr, r, yc, Iext1, K, w)
    print "Solution with optimization:"
    print "x1EQ: ", x1EQopt
    print "x0sol: ", x0solopt

    x1EQlinTaylor, x0sollinTaylor = eq_x1_hypo_x0_linTaylor(ix0, iE, x1eq, zeq, x0[:, ix0], x0cr, r, yc, Iext1, K, w)
    print "Solution with linear Taylor approximation:"
    print "x1EQ: ", x1EQlinTaylor
    print "x0sol: ", x0sollinTaylor

    print "\nTest calc_fz_jac_square_taylor"
    fz_jac_square_taylor = calc_fz_jac_square_taylor(numpy.array(sz), syc, sIext1, sK, sw, tau1=1.0, tau0=1.0)
    print "fz_jac_square_taylor:"
    print fz_jac_square_taylor
    print calc_fz_jac_square_taylor(z, yc, Iext1, K, w, tau1=1.0, tau0=1.0)

    lfz_jac_square_taylor, sfz_jac_square_taylor, v = symbol_calc_fz_jac_square_taylor(n)
    sfz_jac_square_taylor = Matrix(sfz_jac_square_taylor).reshape(n, n)
    print "fz_jac_square_taylor:"
    for iv in range(n):
        for jv in range(n):
            sfz_jac_square_taylor[iv, jv] = sfz_jac_square_taylor[iv, jv].subs([(v["x_taylor"][jv], x1sq[0, jv]),
                                                                                (v["a"][jv], 1.0), (v["b"][jv], -2.0),
                                                                                (v["tau1"][iv], 1.0),
                                                                                (v["tau0"][iv], 1.0)])
        print sfz_jac_square_taylor[iv, :]
    print lfz_jac_square_taylor(zeq, yc, Iext1, K, w, a, b2, tau1, tau0, x1sq)

    print "This is the end..."