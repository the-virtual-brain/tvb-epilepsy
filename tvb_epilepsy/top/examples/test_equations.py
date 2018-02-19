# coding=utf-8

from tvb_epilepsy.base.computations.equilibrium_computation import *


if __name__ == "__main__":
    logger = initialize_logger(__name__)
    n = 3
    x1 = numpy.array([-4.1/3, -4.9/3, -5.0/3], dtype="float32")
    x1eq = x1
    w = numpy.array([[0, 0.1, 0.9], [0.1, 0, 0.0], [0.9, 0.0, 0]])
    n = x1.size
    # y1 = 10.25 * numpy.ones(x1.shape, dtype=x1.dtype)
    # x2 = -0.33 * numpy.ones(x1.shape, dtype=x1.dtype)
    # y2 = 0.0 * numpy.ones(x1.shape, dtype=x1.dtype)
    # g = 0.1*x1 # * numpy.ones(x1.shape, dtype=x1.dtype)
    K = 0.0 * K_DEF * numpy.ones(x1.shape, dtype=x1.dtype)
    yc = YC_DEF * numpy.ones(x1.shape, dtype=x1.dtype)
    Iext1 = I_EXT1_DEF * numpy.ones(x1.shape, dtype=x1.dtype)
    slope = SLOPE_DEF * numpy.ones(x1.shape, dtype=x1.dtype)
    Iext2 = I_EXT2_DEF * numpy.ones(x1.shape, dtype=x1.dtype)
    a = A_DEF
    b = B_DEF
    d = D_DEF
    s = S_DEF
    gamma = GAMMA_DEF
    tau1 = TAU1_DEF
    tau2 = TAU2_DEF
    tau0 = TAU0_DEF
    x1, K = assert_arrays([x1, K])
    w = assert_arrays([w])  # , (x1.size, x1.size)
    zmode = numpy.array("lin")
    pmode = numpy.array("const")
    z = calc_eq_z(x1, yc, Iext1, "2d", x2=0.0, slope=slope, a=a, b=b, d=d, x1_neg=True)
    logger.info("\nzeq = " + str(z))
    x0cr, r = calc_x0cr_r(yc, Iext1, zmode=zmode, x1_rest=X1_DEF, x1_cr=X1_EQ_CR_DEF, x0def=X0_DEF, x0cr_def=X0_CR_DEF)
    logger.info("\nx0cr = " + str(x0cr))
    logger.info("\nr = " + str(r))
    x0 = calc_x0(x1, z, K, w, zmode=zmode, z_pos=True)
    logger.info("\nx0 = " + str(x0))
    x0_values = calc_model_x0_to_x0_val(x0, yc, Iext1, a, b, d, zmode=np.array("lin"))
    logger.info("\nx0_values = " + str(x0_values))
    model = "EpileptorDPrealistic"
    logger.info("\nmodel = " + model)
    x1eq = x1
    zeq = z
    if model == "EpileptorDP2D":
        eq = numpy.c_[x1eq, zeq].T.astype('float32')
        model_vars = 2
        dfun = calc_dfun(eq[0].T, eq[1].T, yc, Iext1, x0, K, w, model_vars, zmode=zmode, pmode=pmode,
                         x0_var=x0, slope_var=slope, Iext1_var=Iext1, Iext2_var=Iext2, K_var=K, slope=slope, 
                         a=a, b=b, d=d, s=s, Iext2=Iext2, gamma=gamma, tau1=tau1, tau0=tau0, tau2=tau2,
                         output_mode="array")

        jac = calc_jac(eq[0].T, eq[1].T, yc, Iext1, x0, K, w, model_vars, zmode=zmode, pmode=pmode, x1_neg=True,
                       z_pos=True, x2_neg=False, x0_var=x0, slope_var=slope, Iext1_var=Iext1, Iext2_var=Iext2, K_var=K,
                       slope=slope, a=a, b=b, d=d, s=s, Iext2=Iext2, gamma=gamma, tau1=tau1, tau0=tau0, tau2=tau2)
    else:
        if model == "EpileptorDPrealistic":
            # the 11D "realistic" simulations model
            from tvb_epilepsy.base.epileptor_models import EpileptorDPrealistic
            eq, slope_eq, Iext2_eq = calc_eq_11d(x0, K, w, yc, Iext1, Iext2, slope,
                                                 EpileptorDPrealistic.fun_slope_Iext2, x1, a=a, b=b, d=d,
                                                 zmode=zmode, pmode=pmode)
            model_vars = 11
            dfun = calc_dfun(eq[0].T, eq[2].T, yc, Iext1, x0, K, w, model_vars, zmode, pmode,
                             y1=eq[1].T, x2=eq[3].T, y2=eq[4].T, g=eq[5].T,
                             x0_var=eq[6].T, slope_var=eq[7].T, Iext1_var=eq[8].T, Iext2_var=eq[9].T, K_var=eq[10].T,
                             slope=slope, a=a, b=b, d=d, s=s, Iext2=Iext2, gamma=gamma, tau1=tau1, tau0=tau0, tau2=tau2,
                             output_mode="array")
            jac = calc_jac(eq[0].T, eq[2].T, yc, Iext1, x0, K, w, model_vars, zmode, pmode,
                           x1_neg=True, z_pos=True, x2_neg=False, y1=eq[1].T, x2=eq[3].T, y2=eq[4].T, g=eq[5].T,
                           x0_var=eq[6].T, slope_var=eq[7].T, Iext1_var=eq[8].T, Iext2_var=eq[9].T, K_var=eq[10].T,
                           slope=slope, a=a, b=b, d=d, s=s, Iext2=Iext2, gamma=gamma, tau1=tau1, tau0=tau0, tau2=tau2)
        else:
            # all >=6D models
            eq = calc_eq_6d(x0, K, w,  yc, Iext1, Iext2, x1, a=a, b=b, d=d, zmode=zmode)
            model_vars = 6
            dfun = calc_dfun(eq[0].T, eq[2].T, yc, Iext1, x0, K, w, model_vars, zmode,
                             y1=eq[1].T, x2=eq[3].T, y2=eq[4].T, g=eq[5].T,
                             slope=slope, a=a, b=b, d=d, s=s, Iext2=Iext2, gamma=gamma, tau1=tau1, tau0=tau0, tau2=tau2,
                             output_mode="array")
            jac = calc_jac(eq[0].T, eq[2].T, yc, Iext1, r, K, w, model_vars, zmode,
                           x1_neg=True, z_pos=True, x2_neg=False, y1=eq[1].T, x2=eq[3].T, y2=eq[4].T, g=eq[5].T,
                           slope=slope, a=a, b=b, d=d, s=s, Iext2=Iext2, gamma=gamma, tau1=tau1, tau0=tau0, tau2=tau2)
    logger.info("\neq = " + str(eq))
    logger.info("\ndfun(eq) = " + str(dfun))
    logger.info("\njac(eq) = " + str(jac))
    model = str(model_vars)+"d"
    sx1, sy1, sz, sx2, sy2, sg, sx0, sx0_val, sK, syc, sIext1, sIext2, sslope, sa, sb, sd, stau1, stau0, stau2, v = \
        symbol_vars(n, ["x1", "y1", "z", "x2", "y2", "g", "x0", "x0_val", "K", "yc", "Iext1", "Iext2",
                        "slope", "a", "b", "d", "tau1", "tau0", "tau2"], shape=(3,))
    sw, vw = symbol_vars(n, ["w"], dims=2, output_flag="numpy_array")
    v.update(vw)
    del vw
    numpy.fill_diagonal(sw, 0.0)
    sw = Array(sw)
    a = numpy.ones((n, ))
    b = 3.0 * a
    d = 5.0 * a
    s = 6.0 * a
    tau1 = a
    tau0 = a
    tau2 = a
    x1sq = -4.0/3 * a
    if model == "2d":
        y1 = yc
    else:
        y1 = eq[1].T
        x2 = eq[3].T
        y2 = eq[4].T
        g = eq[5].T
        if model == "11d":
            x0_var = eq[6].T
            slope_var = eq[7].T
            Iext1_var = eq[8].T
            Iext2_var = eq[9].T
            K_var = eq[10].T

    # -----------------------------------------------------------------------------------------------------------------

    logger.info("\n\nTest symbolic x0cr, r calculation...")
    x0cr2, r2 = calc_x0cr_r(syc, sIext1, zmode=zmode, x1_rest=X1_DEF, x1_cr=X1_EQ_CR_DEF, x0def=X0_DEF,
                            x0cr_def=X0_CR_DEF)  # test=True
    logger.info("\nx0cr2 = " + str(x0cr2))
    logger.info("\nx0cr2 shape = " + str(x0cr2.shape))
    logger.info("\nr2 = " + str(r2))
    logger.info("\nr2 shape = " + str(r2.shape))
    logger.info("\ncalc_x0cr_r = " + str(calc_x0cr_r(yc, Iext1, zmode=zmode, x1_rest=X1_DEF, x1_cr=X1_EQ_CR_DEF,
                                                     x0def=X0_DEF, x0cr_def=X0_CR_DEF)))
    lx0cr_r, sx0cr_r, v = symbol_eqtn_x0cr_r(n, zmode=zmode,
                                             shape=(n,))  # symbol_calc_x0cr_r(n, zmode=zmode, shape=(3, ))
    sx0cr_r = list(sx0cr_r)
    for ii in range(2):
        sx0cr_r[ii] = Matrix(sx0cr_r[ii])
        for iv in range(n):
            sx0cr_r[ii][iv] = sx0cr_r[ii][iv].subs([(v["a"][iv], a[iv]), (v["b"][iv], b[iv]), (v["d"][iv], d[iv]),
                                                    (v["x1_rest"][iv], X1_DEF), (v["x0_rest"][iv], X0_DEF),
                                                    (v["x1_cr"][iv], X1_EQ_CR_DEF), (v["x0_cr"][iv], X0_CR_DEF)])
            logger.info("\nsymbolic x0cr_r[" + str(ii) + "," + str(iv) + "] = " + str(sx0cr_r[ii][iv]))
        logger.info("\nx0cr_r[" + str(ii) + "] shape = " + str(sx0cr_r[ii].shape))
        logger.info("\nlambda x0cr_r[" + str(ii) + "] = " + str(lx0cr_r[ii](yc, Iext1, a, b, d, X1_DEF * a,
                                                                            X1_EQ_CR_DEF * a,
                                                                            X0_DEF * a, X0_CR_DEF * a)))
    # ------------------------------------------------------------------------------------------------------------------
    logger.info("\n\nTest coupling...")
    coupling = calc_coupling(sx1, sK, sw)
    scoupling = symbol_eqtn_coupling(n, shape=(n,))[:2]
    logger.info("\ncoupling equation = " + str(coupling))
    logger.info("\nsymbolic coupling equation = " + str(scoupling[1]))
    logger.info("\ncoupling = " + str(calc_coupling(x1, K, w)))
    logger.info("\nsymbolic coupling = " + str(scoupling[0](x1, K, w)))
    logger.info("\ncoupling shape = " + str(coupling.shape))
    logger.info("\nsymbolic coupling shape = " + str(scoupling[1].shape))
    # ------------------------------------------------------------------------------------------------------------------
    logger.info("\n\nTest coupling derivative to x1...")
    coupling_diff = calc_coupling_diff(sK, sw)
    scoupling_diff = symbol_calc_coupling_diff(n, ix=None, jx=None, K="K")[:2]
    for ii in range(n):
        logger.info("\ncoupling_diff[" + str(ii) + "] equation= " + str(coupling_diff[ii]))
    for ii in range(n):
        logger.info("\nsymbolic coupling_diff[" + str(ii) + "] equation= " + str(scoupling_diff[1][ii]))
    logger.info("\ncoupling_diff = " + str(calc_coupling_diff(K, w)))
    logger.info("\nsymbolic coupling_diff = " + str(scoupling_diff[0](K, w)))
    logger.info("\ncoupling_diff shape = " + str(coupling_diff.shape))
    logger.info("\nsymbolic coupling_diff shape = " + str(scoupling_diff[1].shape))
    # ------------------------------------------------------------------------------------------------------------------
    logger.info("\n\nTest the fz with substitution of z via fx1...")
    fx1z = calc_fx1z(sx1, sx0, sK, sw, syc, sIext1, sa, sb, sd, stau1, stau0, zmode=zmode)
    sfx1z = symbol_eqtn_fx1z(n, model, zmode, shape=(n,))[:2]
    for ii in range(n):
        logger.info("\nfx1z[" + str(ii) + "] equation= " + str(fx1z[ii]))
    for ii in range(n):
        logger.info("\nsymbolic fx1z[" + str(ii) + "] equation= " + str(fx1z[ii]))
    if model == "2d":
        logger.info("\nfx1z = " + str(calc_fx1z(x1, x0, K, w, yc, Iext1, a=a, b=b, d=d, tau1=tau1,
                                                tau0=tau0, model=model, zmode=zmode)))
    else:
        logger.info("\nfx1z = " + str(calc_fx1z(x1, x0, K, w, yc, Iext1, a=a, b=b, d=d, tau1=tau1,
                                                tau0=tau0, model=model, zmode=zmode)))
    if model == "2d":
        logger.info("\nsymbolic fx1z = " + str(sfx1z[0](x1, x0, K, w, yc, Iext1, a, b, d, tau1, tau0)))
    else:
        logger.info("\nsymbolic fx1z = " + str(sfx1z[0](x1, x0, K, w, yc, Iext1, a, b, d, tau1, tau0)))
    logger.info("\nfx1z shape = " + str(fx1z.shape))
    logger.info("\nsymbolic fx1z shape = " + str(sfx1z[1].shape))
    # ------------------------------------------------------------------------------------------------------------------
    logger.info("\n\nTest the derivative to x1 of fz with substitution of z via fx1...")
    fx1z_diff = calc_fx1z_diff(sx1, sK, sw, sa, sb, sd, stau1, stau0, model=model, zmode=zmode)
    sfx1z_diff = symbol_eqtn_fx1z_diff(n, model, zmode)[:2]
    for ii in range(n):
        logger.info("\nfx1z_diff[" + str(ii) + "] equation : " + str(fx1z_diff[ii]))
    for ii in range(n):
        logger.info("\nsymbolic fx1z_diff[" + str(ii) + "] equation : " + str(sfx1z_diff[1][ii, :]))
    if model == "2d":
        logger.info("\nfx1z_diff = " + str(calc_fx1z_diff(x1, K, w, a=a, b=b, d=d, tau1=tau1, tau0=tau0,
                                                          model=model, zmode=zmode)))
    else:
        logger.info("\nfx1z_diff = " + str(calc_fx1z_diff(x1, K, w, a=a, b=b, d=d, tau1=tau1, tau0=tau0,
                                                          model=model, zmode=zmode)))
    if model == "2d":
        logger.info("\nsymbolic fx1z_diff = " + str(sfx1z_diff[0](x1, K, w, a, b, d, tau1, tau0)))
    else:
        logger.info("\nsymbolic fx1z_diff = " + str(sfx1z_diff[0](x1, K, w, a, b, d, tau1, tau0)))
    logger.info("\nfx1z_diff shape = " + str(fx1z_diff.shape))
    logger.info("\nsymbolic fx1z_diff shape = " + str(sfx1z_diff[1].shape))
    # ------------------------------------------------------------------------------------------------------------------
    if model != "2d":
        logger.info("\n\nTest symbolic fx2 with substitution of y2 via fy2...")
        sfx2y2 = symbol_eqtn_fx2y2(n, x2_neg=False, shape=(n, ))[:2]
        for ii in range(n):
            logger.info("\nsymbolic fx2y2[" + str(ii) + "] equation : " + str(sfx2y2[1][ii]))
        logger.info("\nsymbolic fx2y2 = " + str(sfx2y2[0](x2, z, g, Iext2, s, tau1)))
        logger.info("\nsymbolic fx2y2 shape = " + str(sfx2y2[1].shape))
    # ------------------------------------------------------------------------------------------------------------------
    logger.info("\n\nTest calc_fx1_2d_taylor...")
    x_taylor = symbol_vars(n, ["x1lin"], shape=(n, ))[0]  # x_taylor = -4.5/3 (=x1lin)
    fx1lin = calc_fx1_2d_taylor(sx1, x_taylor, sz, syc, sIext1, sslope, sa, sb, stau1, x1_neg=True, order=2,
                                shape=(n, ))
    sfx1lin = symbol_calc_2d_taylor(n, "x1lin", order=2, x1_neg=True, slope="slope", Iext1="Iext1", shape=(n, ))[:2]
    for ii in range(3):
        logger.info("\nfx1lin[" + str(ii) + "] equation : " + str(fx1lin[ii].expand(sx1[ii]).collect(sx1[ii])))
    for ii in range(3):
        logger.info("\nsymbolic fx1lin[" + str(ii) + "] equation : " +
                    str(sfx1lin[1][ii].expand(sx1[ii]).collect(sx1[ii])))
    logger.info("\nfx1lin = " + str(calc_fx1_2d_taylor(x1, -1.5, z, yc, Iext1, slope, a=a, b=b, d=d, tau1=tau1,
                                                       x1_neg=True, order=2, shape=(n, ))))
    logger.info("\nsymbolic fx1lin = " + str(sfx1lin[0](x1, -1.5 * numpy.ones(x1.shape), z, yc, Iext1, slope, a, b, d,
                                                        tau1)))
    logger.info("\nfx1lin shape = " + str(fx1lin.shape))
    logger.info("\nsymbolic fx1lin shape = " + str(sfx1lin[1].shape))
    # ------------------------------------------------------------------------------------------------------------------
    logger.info("\n\nTest calc_fx1y1_6d_diff_x1...")
    fx1y1_6d_diff_x1 = calc_fx1y1_6d_diff_x1(sx1, syc, sIext1, sa, sb, sd, stau1, stau0)
    sfx1y1_6d_diff_x1 = symbol_calc_fx1y1_6d_diff_x1(n, shape=(n, ))[:2]
    for ii in range(n):
        logger.info("\nfx1y1_6d_diff_x1[" + str(ii) + "] equation : " +
                    str(fx1y1_6d_diff_x1[ii].expand(sx1[ii]).collect(sx1[ii])))
    for ii in range(n):
        logger.info("\nsymbolic fx1y1_6d_diff_x1[" + str(ii) + "] equation : " +
                    str(sfx1y1_6d_diff_x1[1][ii].expand(sx1[ii]).collect(sx1[ii])))
    logger.info("\nfx1y1_6d_diff_x1 = " + str(calc_fx1y1_6d_diff_x1(x1, yc, Iext1, a, b, d, tau1, tau0)))
    logger.info("\nsymbolic fx1y1_6d_diff_x1 = " + str(sfx1y1_6d_diff_x1[0](x1, yc, Iext1, a, b, d, tau1)))
    logger.info("\nfx1y1_6d_diff_x1 shape = " + str(fx1y1_6d_diff_x1.shape))
    logger.info("\nsymbolic fx1y1_6d_diff_x1 shape = " + str(sfx1y1_6d_diff_x1[1].shape))
    # ------------------------------------------------------------------------------------------------------------------
    logger.info("\n\nTest eq_x1_hypo_x0_optimize_fun & eq_x1_hypo_x0_optimize_jac...")
    ix0 = numpy.array([1, 2])
    iE = numpy.array([0])
    x = numpy.empty_like(sx1).flatten()
    x[ix0] = sx1[ix0]
    vz = symbol_eqtn_fz(n, zmode=numpy.array("lin"), z_pos=True, x0="x0", K="K")[2]
    x[iE] = sx0[iE]
    p = x1.shape
    eq_x1_hypo_x0_optimize(ix0, iE, x1eq, zeq, x0[ix0], K, w, yc, Iext1, a=A_DEF, b=B_DEF, d=D_DEF, slope=SLOPE_DEF)
    opt_fun = eq_x1_hypo_x0_optimize_fun(x, ix0, iE, sx1, numpy.array(sz), sx0[ix0], sK, sw, syc, sIext1)
    logger.info("\nopt_fun = " + str(opt_fun))
    opt_jac = eq_x1_hypo_x0_optimize_jac(x, ix0, iE, sx1, numpy.array(sz), sx0[ix0], sK, sw, sy1, sIext1)
    logger.info("\nopt_jac = " + str(opt_jac))
    x1EQopt, x0solopt = eq_x1_hypo_x0_optimize(ix0, iE, x1eq, zeq, x0[ix0], K, w, yc, Iext1)
    logger.info("\nSolution with optimization:")
    logger.info("\nx1EQ = " + str(x1EQopt))
    logger.info("\nx0sol = " + str(x0solopt))
    x1EQlinTaylor, x0sollinTaylor = eq_x1_hypo_x0_linTaylor(ix0, iE, x1eq, zeq, x0[ix0], K, w, yc, Iext1)
    logger.info("\nSolution with linear Taylor approximation:")
    logger.info("\nx1EQ = " + str(x1EQlinTaylor))
    logger.info("\nx0sol = " + str(x0sollinTaylor))
    # ------------------------------------------------------------------------------------------------------------------
    logger.info("\n\nTest calc_fz_jac_square_taylor...")
    fz_jac_square_taylor = calc_fz_jac_square_taylor(numpy.array(sz), syc, sIext1, sK, sw, tau1=tau1, tau0=tau0)
    lfz_jac_square_taylor, sfz_jac_square_taylor, v = symbol_calc_fz_jac_square_taylor(n)
    sfz_jac_square_taylor = Matrix(sfz_jac_square_taylor).reshape(n, n)
    logger.info("\nfz_jac_square_taylor equation:" + str(fz_jac_square_taylor))
    logger.info("\nsymbolic fz_jac_square_taylor equation:")
    for iv in range(n):
        for jv in range(n):
            sfz_jac_square_taylor[iv, jv] = sfz_jac_square_taylor[iv, jv].subs([(v["x_taylor"][jv], x1sq[jv]),
                                                                                (v["a"][jv], a[jv]),
                                                                                (v["b"][jv], b[jv]),
                                                                                (v["d"][jv], d[jv]),
                                                                                (v["tau1"][iv], tau1[iv]),
                                                                                (v["tau0"][iv], tau2[iv])])
        logger.info("\n" + str(sfz_jac_square_taylor[iv, :]))
    logger.info("\nfz_jac_square_taylor = " + str(calc_fz_jac_square_taylor(z, yc, Iext1, K, w, tau1=tau1, tau0=tau0)))
    logger.info("\nsymbolic fz_jac_square_taylor = " +
                str(lfz_jac_square_taylor(zeq, yc, Iext1, K, w, a, b, d, tau1, tau0, x1sq)))
    logger.info("\nThis is the end...")
