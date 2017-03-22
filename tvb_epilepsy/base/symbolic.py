import numpy
from numpy import array, empty, empty_like, ones, zeros, multiply, dot, power, divide, sum, exp, reshape, diag, expand_dims, where
from sympy import Symbol, symbols, exp, solve, lambdify, Matrix, series # diff, MatrixSymbol
from tvb_epilepsy.base.equations import *

def symbol_vars(n_regions, vars_str, dims=1, ind_str="_", shape=None, numpy_flag=None):

    vars_out = list()
    vars_dict = {}

    if dims == 1:

        if numpy_flag is None:
            numpy_flag = True

        if shape is None:
            shape = (n_regions,)

        for vs in vars_str:
            temp = [Symbol(vs+ind_str+'%d' % i_n) for i_n in range(n_regions)]
            if numpy_flag:
                temp = reshape(temp, shape)
            vars_out.append(temp)
            vars_dict[vs] = vars_out[-1:][0]

    elif dims == 0:

        if numpy_flag is None:
            numpy_flag = False

        if shape is None:
            shape = (1,)

        for vs in vars_str:
            temp = Symbol(vs)
            if numpy_flag:
                temp = reshape(temp, shape)
            vars_out.append(temp)
            vars_dict[vs] = vars_out[-1:][0]

    elif dims == 2:

        if numpy_flag is None:
            numpy_flag = True

        if shape is None:
            shape = (n_regions, n_regions)

        for vs in vars_str:
            temp = []
            for i_n in range(n_regions):
                temp.append([Symbol(vs + ind_str + '%d' % i_n + ind_str + '%d' % j_n) for j_n in range(n_regions)])
            if numpy_flag:
                temp = reshape(temp, shape)
            vars_out.append(temp)
            vars_dict[vs] = vars_out[-1:][0]
    else:
        raise ValueError("The dimensionality of the variables is neither 1 nor 2: " + str(dims))

    vars_out.append(vars_dict)

    return tuple(vars_out)


def symbol_eqtn_coupling(n, ix=None, jx=None, K="K", shape=None):

    # Only difference coupling for the moment.
    # TODO: Extend for different coupling forms

    x1, K, vars_dict = symbol_vars(n, ["x1", K], shape=shape)
    w, vars_dict_w = symbol_vars(n, ["w"], dims=2)
    vars_dict.update(vars_dict_w)

    if ix is None:
        ix = range(n)

    if jx is None:
        jx = range(n)

    coupling = eqtn_coupling(x1, K, w, ix, jx)

    return lambdify([x1, K, w], coupling.tolist(), "numpy"), coupling, vars_dict


def symbol_eqtn_x0(n, zmode=numpy.array("lin"), z_pos=True, model="2d", K="K", shape=None):

    x1, z, K, vars_dict = symbol_vars(n, ["x1", "z", K], shape=shape)

    w, temp = symbol_vars(n, ["w"], dims=2)
    vars_dict.update(temp)

    if model == "2d":

        x0cr, r, temp = symbol_vars(n, ["x0cr", "r"], shape=shape)

        vars_dict.update(temp)

        x0 = eqtn_x0(x1, z, model, zmode, z_pos, K, w, x0cr, r)

        return lambdify([x1, z, x0cr, r, K, w], x0.tolist(), "numpy"), x0, vars_dict

    else:

        x0 = eqtn_x0(x1, z, model, zmode, z_pos, K, w)

        return lambdify([x1, z, K, w], x0.tolist(), "numpy"), x0, vars_dict


def symbol_eqtn_fx1(n, model="2d", x1_neg=True, slope="slope", Iext1="Iext1", shape=None):

    x1, z, y1, slope, Iext1, a, b, tau1, vars_dict = symbol_vars(n, ["x1", "z", "y1", slope, Iext1, "a", "b", "tau1"],
                                                                 shape=shape)

    if model == "2d":

        fx1 = eqtn_fx1(x1, z, y1, Iext1, slope, a, b, tau1, x1_neg, model, x2=None)

        return lambdify([x1, z, y1, Iext1, slope, a, b, tau1], fx1.tolist(), "numpy"), fx1, vars_dict

    else:

        x2, vx2 = symbol_vars(n, ["x2"], shape=shape)
        vars_dict.update(vx2)

        fx1 = eqtn_fx1(x1, z, y1, Iext1, slope, a, b, tau1, x1_neg, model, x2=x2)

        return lambdify([x1, z, y1, x2, Iext1, slope, a, b, tau1], fx1.tolist(), "numpy"), fx1, vars_dict



def symbol_eqtn_fy1(n, shape=None):

    x1, y1, yc, d, tau1, vars_dict = symbol_vars(n, ["x1", "y1", "yc", "d", "tau1"], shape=shape)

    fy1 = eqtn_fy1(x1, yc, y1, d, tau1)

    return lambdify([x1, y1, yc, d, tau1], fy1.tolist(), "numpy"), fy1, vars_dict


def symbol_eqtn_fz(n, zmode=numpy.array("lin"), z_pos=True, model="2d", x0="x0", K="K", shape=None):

    x1, z, x0, K, tau1, tau0, vars_dict = symbol_vars(n, ["x1", "z", x0, K, "tau1", "tau0"], shape=shape)

    w, temp = symbol_vars(n, ["w"], dims=2)
    vars_dict.update(temp)

    if model == "2d":

        x0cr, r, temp = symbol_vars(n, ["x0cr", "r"], shape=shape)

        vars_dict.update(temp)

        fz = eqtn_fz(x1, z, x0, tau1, tau0, model, zmode, z_pos, K=K, w=w, x0cr=x0cr, r=r)

        return lambdify([x1, z, x0, x0cr, r, K, w], x0.tolist(), "numpy"), fz, vars_dict

    else:

        fz = eqtn_fz(x1, z, x0, tau1, tau0, model, zmode, z_pos, K=K, w=w, x0cr=None, r=None)

        return lambdify([x1, z, x0, K, w], x0.tolist(), "numpy"), fz, vars_dict


def symbol_eqtn_fx2(n, Iext2="Iext2", shape=None):

    x2, y2, z, g, Iext2, tau1, vars_dict = symbol_vars(n, ["x2", "y2", "z", "g", Iext2, "tau1"], shape=shape)

    fx2 = eqtn_fx2(x2, y2, z, g, Iext2, tau1)

    return lambdify([x2, y2, z, g, Iext2, tau1], fx2.tolist(), "numpy"), fx2, vars_dict


def symbol_eqtn_fy2(n, x2_neg=True, shape=None):

    x2, y2, s, tau1, tau2, vars_dict = symbol_vars(n, ["x2", "y2", "s", "tau1", "tau2"], shape=shape)

    fy2 = eqtn_fy2(x2, y2, s, tau1, tau2, x2_neg)

    return lambdify([x2, y2, s, tau1, tau2], fy2.tolist(), "numpy"), fy2, vars_dict


def symbol_eqtn_fg(n, shape=None):

    x1, g, gamma, tau1, vars_dict = symbol_vars(n, ["x1", "g", "gamma", "tau1"], shape=shape)

    fg = eqtn_fg(x1, g, gamma, tau1)

    return lambdify([x1, g, gamma, tau1], fg.tolist(), "numpy"), fg, vars_dict


def symbol_eqtn_fx0(n, shape=None):

    x0_var, x0, tau1, vars_dict = symbol_vars(n, ["x0_var", "x0", "tau1"], shape=shape)

    fx0 = eqtn_fx0(x0_var, x0, tau1)

    return lambdify([x0_var, x0, tau1], fx0.tolist(), "numpy"), fx0, vars_dict


def symbol_eqtn_fslope(n, pmode=array("const"), shape=None):

    slope_var, z, g, slope, tau1, vars_dict = symbol_vars(n, ["slope_var", "z", "g", "slope", "tau1"], shape=shape)

    from tvb_epilepsy.tvb_api.epileptor_models import EpileptorDPrealistic
    slope_eq = EpileptorDPrealistic.fun_slope_Iext2(z, g, pmode, slope, 0.0)[0]

    fslope = eqtn_fslope(slope_var, slope_eq, tau1)

    vars_dict["pmode"] = pmode

    if pmode=="z":
        return lambdify([slope_var, z, tau1], fslope.tolist(), "numpy"), fslope, vars_dict
    elif pmode == "g":
        return lambdify([slope_var, g, tau1], fslope.tolist(), "numpy"), fslope, vars_dict
    elif pmode == "z*g":
        return lambdify([slope_var, z, g, tau1], fslope.tolist(), "numpy"), fslope, vars_dict
    else:
        return lambdify([slope_var, slope, tau1], fslope.tolist(), "numpy"), fslope, vars_dict


def symbol_eqtn_fIext1(n, shape=None):

    Iext1_var, Iext1, tau1, tau0, vars_dict = symbol_vars(n, ["Iext1_var", "Iext1", "tau1", "tau0"], shape=shape)

    fIext1 = eqtn_fIext1(Iext1_var, Iext1, tau1, tau0)

    return lambdify([Iext1_var, Iext1, tau1, tau0], fIext1.tolist(), "numpy"), fIext1, vars_dict


def symbol_eqtn_fIext2(n, pmode=array("const"), shape=None):

    Iext2_var, z, g, Iext2, tau1, vars_dict = symbol_vars(n, ["Iext2_var", "z", "g", "Iext2", "tau1"], shape=shape)

    from tvb_epilepsy.tvb_api.epileptor_models import EpileptorDPrealistic
    Iext2_eq = EpileptorDPrealistic.fun_Iext2_Iext2(z, g, pmode, 0.0, Iext2)[1]

    fIext2 = eqtn_fIext2(Iext2_var, Iext2_eq, tau1)

    vars_dict["pmode"] = pmode

    if pmode=="z":
        return lambdify([Iext2_var, z, tau1], fIext2.tolist(), "numpy"), fIext2, vars_dict
    elif pmode == "g":
        return lambdify([Iext2_var, g, tau1], fIext2.tolist(), "numpy"), fIext2, vars_dict
    elif pmode == "z*g":
        return lambdify([Iext2_var, z, g, tau1], fIext2.tolist(), "numpy"), fIext2, vars_dict
    else:
        return lambdify([Iext2_var, Iext2, tau1], fIext2.tolist(), "numpy"), fIext2, vars_dict


def symbol_eqtn_fK(n, shape=None):

    K_var, K, tau1, tau0, vars_dict = symbol_vars(n, ["K_var", "K", "tau1", "tau0"], shape=shape)

    fK = eqtn_fK(K_var, K, tau1, tau0)

    return lambdify([K_var, K, tau1, tau0], fK.tolist(), "numpy"), fK, vars_dict


def symbol_eqtn_fparam_vars(n, pmode=array("const"), shape=None):

    fx0_lambda, fx0, vars_dict = symbol_eqtn_fx0(n, shape)

    slope_lambda, fslope, temp = symbol_eqtn_fslope(n, pmode, shape)
    vars_dict.update(temp)

    Iext1_lambda, fIext1, temp = symbol_eqtn_fIext1(n, shape=shape)
    vars_dict.update(temp)

    Iext2_lambda, fIext2, temp = symbol_eqtn_fIext2(n, pmode, shape)
    vars_dict.update(temp)

    K_lambda, fK, temp = symbol_eqtn_fK(n, shape)
    vars_dict.update(temp)

    return [fx0_lambda, slope_lambda, Iext1_lambda, Iext2_lambda, K_lambda], \
           [fx0, fslope, fIext1, fIext2, fK], vars_dict


def symbol_eqnt_dfun(n, model_vars, zmode=array("lin"), x1_neg=True, x2_neg=True, z_pos=True,
                     pmode=array("const"), shape=None):

    f_lambda = []
    f_sym = []

    if model_vars == 2:

        fl, fs, symvars = symbol_eqtn_fx1(n, model="2d", x1_neg=x1_neg, slope="slope", Iext1="Iext1", shape=shape)
        f_lambda.append(fl)
        f_sym.append(fs)

        fl, fs, temp = symbol_eqtn_fz(n, zmode=zmode, z_pos=z_pos, model="2d", x0="x0", K="K", shape=shape)
        f_lambda.append(fl)
        f_sym.append(fs)

        #         [x1, z, yc, slope, Iext1, x0, x0cr, r, K, w, a, b, tau1, tau0]
        symvars.update(temp)

    elif model_vars == 6:

        fl, fs, symvars = symbol_eqtn_fx1(n, model="6d", x1_neg=x1_neg, slope="slope", Iext1="Iext1", shape=shape)
        f_lambda.append(fl)
        f_sym.append(fs)

        fl, fs, temp = symbol_eqtn_fy1(n, shape)
        f_lambda.append(fl)
        f_sym.append(fs)
        symvars.update(temp)

        fl, fs, temp = symbol_eqtn_fz(n, zmode=zmode, z_pos=z_pos, model="6d", x0="x0", K="K", shape=shape)
        f_lambda.append(fl)
        f_sym.append(fs)
        symvars.update(temp)

        fl, fs, temp = symbol_eqtn_fx2(n, Iext2="Iext2", shape=shape)
        f_lambda += fl
        f_sym += fs
        symvars.update(temp)

        fl, fs, temp = symbol_eqtn_fy2(n, x2_neg, shape)
        f_lambda += fl
        f_sym += fs
        symvars.update(temp)

        fl, fs, temp = symbol_eqtn_fg(n, shape)
        f_lambda.append(fl)
        f_sym.append(fs)
        symvars.update(temp)

    elif model_vars == 11:

        fl, fs, symvars = symbol_eqtn_fx1(n, model="11d", x1_neg=x1_neg, slope="slope_var", Iext1="Iext1", shape=shape)
        f_lambda.append(fl)
        f_sym.append(fs)

        fl, fs, temp = eqtn_fy1(n, shape)
        f_lambda.append(fl)
        f_sym.append(fs)
        symvars.update(temp)

        fl, fs, temp = symbol_eqtn_fz(n, zmode=zmode, z_pos=z_pos, model="11d", x0="x0_var", K="K_var", shape=shape)
        f_lambda.append(fl)
        f_sym.append(fs)
        symvars.update(temp)

        fl, fs, temp = symbol_eqtn_fx2(n, Iext2="Iext2_var", shape=shape)
        f_lambda += fl
        f_sym += fs
        symvars.update(temp)

        fl, fs, temp = symbol_eqtn_fy2(n, x2_neg, shape)
        f_lambda += fl
        f_sym += fs
        symvars.update(temp)

        fl, fs, temp = symbol_eqtn_fg(n, shape)
        f_lambda.append(fl)
        f_sym.append(fs)
        symvars.update(temp)

        fl, fs, temp = symbol_eqtn_fparam_vars(n, pmode, shape)
        f_lambda += fl
        f_sym += fs
        symvars.update(temp)

    return f_lambda, f_sym, symvars


def symbol_calc_2d_taylor(n, x_taylor="x1lin", order=2, x1_neg=True, slope="slope", Iext1="Iext1", shape=None):

    x, x_taylor = symbols("x " + x_taylor)

    fx1lin, v = symbol_eqtn_fx1(n, model="2d", x1_neg=x1_neg, slope=slope, Iext1=Iext1, shape=shape)[1:]

    v.update({"x_taylor": x_taylor})

    for ix in range(v["x1"].size):
        fx1lin[ix] = series(fx1lin[ix], x=x, x0=x_taylor, n=order).removeO().simplify().subs(x, v["x1"]).flatten()[ix]

    return lambdify([v["x1"], v["z"], v["y1"], v[Iext1], v[slope], v["a"], v["b"], v["tau1"]], fx1lin.tolist(), "numpy"), \
           fx1lin, v


def symbol_calc_fx1z_2d_x1neg_zpos_jac(n, ix0, iE):

    fx1, v = symbol_eqtn_fx1(n, model="2d", x1_neg=True, slope="slope", Iext1="Iext1", shape=None)[1:]

    fz, vz = symbol_eqtn_fz(n, zmode=array("lin"), zpos=True, model="2d", x0="x0", K="K", shape=None)[1:]

    v.update(vz)
    del vz

    x = empty_like(v["x1"])
    x[iE] = v["x0"][iE]
    x[ix0] = v["x1"][ix0]
    x = Matrix(x.tolist()).T

    jac = []
    for ix in range(n):
        fx1[ix] = fx1[ix].subs(v["tau1"][ix], 1.0).subs(v["z"][ix], 0.0).expand(v["x1"][ix]).collect(v["x1"][ix])
        fz[ix] = fz[ix].subs(v["z"][ix], fx1[ix]).expand(v["x1"][ix]).collect(v["x1"][ix])
        jac.append(Matrix([fz[ix]]).jacobian(x)[:])

    #jac = array(jac)

    return lambdify([v["x1"], v["z"], v["x0"], v["x0cr"], v["r"], v["yc"], v["Iext1"], v["K"], v["w"], v["a"], v["b"],
                     v["tau1"], v["tau0"]], jac, "numpy"), jac, v


def symbol_calc_fx1y1_6d_diff_x1(n):

    fx1, v = symbol_eqtn_fx1(n, model="6d", x1_neg=True, slope="slope", Iext1="Iext1", shape=None)[1:]

    fy1, vy = symbol_eqtn_fy1(n, shape=None)[1:]

    v.update(vy)
    del vy

    x = Matrix(v["x1"].tolist()).T

    dfx1 = []
    for ix in range(n):
        fy1[ix] = fy1[ix].subs(v["y1"], 0.0).subs(v["tau1"], 1.0)
        fx1[ix] = fx1[ix].subs(v["y1"], fy1).expand(v["x1"][ix]).collect(v["x1"][ix])
        dfx1.append(Matrix([fx1[ix]]).jacobian(x)[:])

    dfx1 = reshape(dfx1, v["x1"].shape)

    return lambdify([v["x1"], v["yc"], v["Iext1"], v["a"], v["b"], v["d"], v["tau1"]], dfx1, "numpy"), dfx1, v


def symbol_calc_jac(n_regions, model_vars, zmode=array("lin"), x1_neg=True, x2_neg=True, z_pos=True,
                    pmode=array("const")):

    dfun_sym, v = symbol_eqnt_dfun(n_regions, model_vars, zmode, x1_neg, x2_neg, z_pos, pmode)[1:]

    dfun_sym = Matrix(Matrix(dfun_sym)[:])

    jac_lambda = []

    ind = lambda x: x*n_regions + array(range(n_regions))

    if model_vars == 2:

        jac_sym = dfun_sym.jacobian(Matrix(Matrix([v["x1"], v["z"]])[:]))

        jac_lambda.append(lambdify([v["x1"], v["z"], v["yc"], v["Iext1"], v["slope"], v["a"], v["b"], v["tau1"]],
                          jac_sym[ind(0), :], "numpy"))
        jac_lambda.append(lambdify([v["x1"], v["z"], v["x0"], v["x0cr"], v["r"], v["K"], v["w"], v["tau1"], v["tau0"]],
                          jac_sym[ind(1), :], "numpy"))

    elif model_vars == 6:

        jac_sym = dfun_sym.jacobian(Matrix(Matrix([v["x1"], v["y1"], v["z"], v["x2"], v["y2"], v["g"]])[:]))

        jac_lambda.append(lambdify([v["x1"], v["y1"], v["z"], v["x2"], v["y2"], v["g"],
                                    v["Iext1"], v["slope"], v["a"], v["b"], v["tau1"]], jac_sym[ind(0), :], "numpy"))
        jac_lambda.append(lambdify([v["x1"], v["y1"], v["z"], v["x2"], v["y2"], v["g"],
                                    v["yc"], v["d"], v["tau1"]], jac_sym[ind(1), :], "numpy"))
        jac_lambda.append(lambdify([v["x1"], v["y1"], v["z"], v["x2"], v["y2"], v["g"],
                                    v["x0"], v["K"], v["w"], v["tau1"], v["tau0"]],
                                   jac_sym[ind(2), :], "numpy"))
        jac_lambda.append(lambdify([v["x1"], v["y1"], v["z"], v["x2"], v["y2"], v["g"],
                                    v["Iext2"], v["tau1"]], jac_sym[ind(3), :], "numpy"))
        jac_lambda.append(lambdify([v["x1"], v["y1"], v["z"], v["x2"], v["y2"], v["g"],
                                    v["s"], v["tau1"], v["tau0"]], jac_sym[ind(4), :], "numpy"))
        jac_lambda.append(lambdify([v["x1"], v["y1"], v["z"], v["x2"], v["y2"], v["g"],
                                    v["gamma"], v["tau1"]], jac_sym[ind(5), :], "numpy"))

    elif model_vars == 11:

        jac_sym = dfun_sym.jacobian(Matrix(Matrix([v["x1"], v["y1"], v["z"], v["x2"], v["y2"], v["g"],
                                        v["x0_var"], v["slope_var"], v["Iext1_var"], v["Iext2_var"], v["K_var"]])[:]))

        jac_lambda.append(lambdify([v["x1"], v["y1"], v["z"], v["x2"], v["y2"], v["g"],
                                    v["x0_var"], v["Iext1_var"], v["Iext2_var"], v["slope_var"], v["K_var"],
                               v["a"], v["b"], v["tau1"]], jac_sym[ind(0), :], "numpy"))
        jac_lambda.append(lambdify([v["x1"], v["y1"], v["z"], v["x2"], v["y2"], v["g"],
                               v["x0_var"], v["Iext1_var"], v["Iext2_var"], v["slope_var"], v["K_var"],
                               v["yc"], v["d"], v["tau1"]], jac_sym[ind(1), :], "numpy"))
        jac_lambda.append(lambdify([v["x1"], v["y1"], v["z"], v["x2"], v["y2"], v["g"],
                               v["x0_var"], v["Iext1_var"], v["Iext2_var"], v["slope_var"], v["K_var"],
                               v["w"], v["tau1"], v["tau0"]], jac_sym[ind(2), :], "numpy"))
        jac_lambda.append(lambdify([v["x1"], v["y1"], v["z"], v["x2"], v["y2"], v["g"],
                               v["x0_var"], v["Iext1_var"], v["Iext2_var"], v["slope_var"], v["K_var"],
                               v["tau1"]], jac_sym[ind(3), :], "numpy"))
        jac_lambda.append(lambdify([v["x1"], v["y1"], v["z"], v["x2"], v["y2"], v["g"],
                               v["x0_var"], v["Iext1_var"], v["Iext2_var"], v["slope_var"], v["K_var"],
                               v["s"], v["tau1"], v["tau0"]], jac_sym[ind(4), :], "numpy"))
        jac_lambda.append(lambdify([v["x1"], v["y1"], v["z"], v["x2"], v["y2"], v["g"],
                               v["x0_var"], v["Iext1_var"], v["Iext2_var"], v["slope_var"], v["K_var"],
                               v["gamma"], v["tau1"]], jac_sym[ind(5), :], "numpy"))
        jac_lambda.append(lambdify([v["x1"], v["y1"], v["z"], v["x2"], v["y2"], v["g"],
                                    v["x0_var"], v["Iext1_var"], v["Iext2_var"], v["slope_var"], v["K_var"],
                                    v["x0"], v["tau1"]], jac_sym[ind(6), :], "numpy"))
        jac_lambda.append(lambdify([v["x1"], v["y1"], v["z"], v["x2"], v["y2"], v["g"],
                                    v["x0_var"], v["Iext1_var"], v["Iext2_var"], v["slope_var"], v["K_var"],
                                    v["slope"], v["tau1"]], jac_sym[ind(7), :], "numpy"))
        jac_lambda.append(lambdify([v["x1"], v["y1"], v["z"], v["x2"], v["y2"], v["g"],
                                    v["x0_var"], v["Iext1_var"], v["Iext2_var"], v["slope_var"], v["K_var"],
                                    v["Iext1"], v["tau1"], v["tau0"]], jac_sym[ind(8), :], "numpy"))
        jac_lambda.append(lambdify([v["x1"], v["y1"], v["z"], v["x2"], v["y2"], v["g"],
                                    v["x0_var"], v["Iext1_var"], v["Iext2_var"], v["slope_var"], v["K_var"],
                                    v["Iext2"], v["tau1"]], jac_sym[ind(9), :], "numpy"))
        jac_lambda.append(lambdify([v["x1"], v["y1"], v["z"], v["x2"], v["y2"], v["g"],
                                    v["x0_var"], v["Iext1_var"], v["Iext2_var"], v["slope_var"], v["K_var"],
                                    v["K"], v["tau1"], v["tau0"]], jac_sym[ind(10), :], "numpy"))

    return jac_lambda, jac_sym, v


