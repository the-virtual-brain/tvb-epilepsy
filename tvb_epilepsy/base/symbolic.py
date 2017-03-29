import numpy
from numpy import array, empty, empty_like, ones, zeros, multiply, dot, power, divide, sum, exp, reshape, diag, expand_dims, where
from sympy import Symbol, symbols, exp, solve, solveset, Interval, S, lambdify, Matrix, series, oo # diff, MatrixSymbol
from tvb_epilepsy.base.constants import X0_DEF, X0_CR_DEF, X1_DEF, X1_EQ_CR_DEF
from tvb_epilepsy.base.utils import assert_arrays
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
            temp = [Symbol(vs+ind_str+'%d' % i_n, real=True) for i_n in range(n_regions)]
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
            temp = Symbol(vs, real=True)
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
                temp.append([Symbol(vs + ind_str + '%d' % i_n + ind_str + '%d' % j_n, real=True)
                             for j_n in range(n_regions)])
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

    coupling = Matrix(eqtn_coupling(x1, K, w, ix, jx))

    return lambdify([x1, K, w], coupling, "numpy"), coupling, vars_dict


def symbol_eqtn_x0(n, zmode=numpy.array("lin"), z_pos=True, model="2d", K="K", shape=None):

    x1, z, K, vars_dict = symbol_vars(n, ["x1", "z", K], shape=shape)

    w, temp = symbol_vars(n, ["w"], dims=2)
    vars_dict.update(temp)

    if model == "2d":

        x0cr, r, temp = symbol_vars(n, ["x0cr", "r"], shape=shape)

        vars_dict.update(temp)

        x0 = Matrix(eqtn_x0(x1, z, model, zmode, z_pos, K, w, coupl=None, x0cr=x0cr, r=r))

        x0_lambda = lambdify([x1, z, x0cr, r, K, w], x0, "numpy")

    else:

        x0 = Matrix(eqtn_x0(x1, z, model, zmode, z_pos, K, w))

        x0_lambda = lambdify([x1, z, K, w], x0, "numpy"), x0, vars_dict

    return x0_lambda, x0, vars_dict


def symbol_eqtn_fx1(n, model="2d", x1_neg=True, slope="slope", Iext1="Iext1", shape=None):

    x1, z, y1, slope, Iext1, a, b, tau1, vars_dict = symbol_vars(n, ["x1", "z", "y1", slope, Iext1, "a", "b", "tau1"],
                                                                 shape=shape)

    if model == "2d":

        fx1 = Matrix(eqtn_fx1(x1, z, y1, Iext1, slope, a, b, tau1, x1_neg, model, x2=None))

        return lambdify([x1, z, y1, Iext1, slope, a, b, tau1], fx1, "numpy"), fx1, vars_dict

    else:

        x2, vx2 = symbol_vars(n, ["x2"], shape=shape)
        vars_dict.update(vx2)

        fx1 = Matrix(eqtn_fx1(x1, z, y1, Iext1, slope, a, b, tau1, x1_neg, model, x2=x2))

        return lambdify([x1, z, y1, x2, Iext1, slope, a, b, tau1], fx1, "numpy"), fx1, vars_dict



def symbol_eqtn_fy1(n, shape=None):

    x1, y1, yc, d, tau1, vars_dict = symbol_vars(n, ["x1", "y1", "yc", "d", "tau1"], shape=shape)

    fy1 = Matrix(eqtn_fy1(x1, yc, y1, d, tau1))

    return lambdify([x1, y1, yc, d, tau1], fy1, "numpy"), fy1, vars_dict


def symbol_eqtn_fz(n, zmode=numpy.array("lin"), z_pos=True, model="2d", x0="x0", K="K", shape=None):

    x1, z, x0, K, tau1, tau0, vars_dict = symbol_vars(n, ["x1", "z", x0, K, "tau1", "tau0"], shape=shape)

    w, temp = symbol_vars(n, ["w"], dims=2)
    vars_dict.update(temp)

    if model == "2d":

        x0cr, r, temp = symbol_vars(n, ["x0cr", "r"], shape=shape)

        vars_dict.update(temp)

        fz = Matrix(eqtn_fz(x1, z, x0, tau1, tau0, model, zmode, z_pos, K=K, w=w, x0cr=x0cr, r=r)[:])

        fz_lambda = lambdify([x1, z, x0, x0cr, r, K, w, tau1, tau0], fz, "numpy")

    else:

        fz = Matrix(eqtn_fz(x1, z, x0, tau1, tau0, model, zmode, z_pos, K=K, w=w, x0cr=None, r=None))

        fz_lambda = lambdify([x1, z, x0, K, w, tau1, tau0], fz, "numpy")

    return fz_lambda, fz, vars_dict


def symbol_eqtn_fx2(n, Iext2="Iext2", shape=None):

    x2, y2, z, g, Iext2, tau1, vars_dict = symbol_vars(n, ["x2", "y2", "z", "g", Iext2, "tau1"], shape=shape)

    fx2 = Matrix(eqtn_fx2(x2, y2, z, g, Iext2, tau1))

    return lambdify([x2, y2, z, g, Iext2, tau1], fx2, "numpy"), fx2, vars_dict


def symbol_eqtn_fy2(n, x2_neg=True, shape=None):

    x2, y2, s, tau1, tau2, vars_dict = symbol_vars(n, ["x2", "y2", "s", "tau1", "tau2"], shape=shape)

    fy2 = Matrix(eqtn_fy2(x2, y2, s, tau1, tau2, x2_neg))

    return lambdify([x2, y2, s, tau1, tau2], fy2, "numpy"), fy2, vars_dict


def symbol_eqtn_fg(n, shape=None):

    x1, g, gamma, tau1, vars_dict = symbol_vars(n, ["x1", "g", "gamma", "tau1"], shape=shape)

    fg = Matrix(eqtn_fg(x1, g, gamma, tau1))

    return lambdify([x1, g, gamma, tau1], fg, "numpy"), fg, vars_dict


def symbol_eqtn_fx0(n, shape=None):

    x0_var, x0, tau1, vars_dict = symbol_vars(n, ["x0_var", "x0", "tau1"], shape=shape)

    fx0 = Matrix(eqtn_fx0(x0_var, x0, tau1))

    return lambdify([x0_var, x0, tau1], fx0, "numpy"), fx0, vars_dict


def symbol_eqtn_fslope(n, pmode=array("const"), shape=None):

    slope_var, z, g, slope, tau1, vars_dict = symbol_vars(n, ["slope_var", "z", "g", "slope", "tau1"], shape=shape)

    from tvb_epilepsy.tvb_api.epileptor_models import EpileptorDPrealistic
    slope_eq = EpileptorDPrealistic.fun_slope_Iext2(z, g, pmode, slope, 0.0)[0]

    fslope = Matrix(eqtn_fslope(slope_var, slope_eq, tau1))

    vars_dict["pmode"] = pmode

    if pmode=="z":
        fslope_lambda = lambdify([slope_var, z, tau1], fslope, "numpy")
    elif pmode == "g":
        fslope_lambda = lambdify([slope_var, g, tau1], fslope, "numpy"),
    elif pmode == "z*g":
        fslope_lambda = lambdify([slope_var, z, g, tau1], fslope, "numpy")
    else:
        fslope_lambda = lambdify([slope_var, slope, tau1], fslope, "numpy")

    return fslope_lambda, fslope, vars_dict


def symbol_eqtn_fIext1(n, shape=None):

    Iext1_var, Iext1, tau1, tau0, vars_dict = symbol_vars(n, ["Iext1_var", "Iext1", "tau1", "tau0"], shape=shape)

    fIext1 = Matrix(eqtn_fIext1(Iext1_var, Iext1, tau1, tau0))

    return lambdify([Iext1_var, Iext1, tau1, tau0], fIext1, "numpy"), fIext1, vars_dict


def symbol_eqtn_fIext2(n, pmode=array("const"), shape=None):

    Iext2_var, z, g, Iext2, tau1, vars_dict = symbol_vars(n, ["Iext2_var", "z", "g", "Iext2", "tau1"], shape=shape)

    from tvb_epilepsy.tvb_api.epileptor_models import EpileptorDPrealistic
    Iext2_eq = EpileptorDPrealistic.fun_slope_Iext2(z, g, pmode, 0.0, Iext2)[1]

    fIext2 = Matrix(eqtn_fIext2(Iext2_var, Iext2_eq, tau1))

    vars_dict["pmode"] = pmode

    if pmode=="z":
        fIext2_lambda = lambdify([Iext2_var, z, tau1], fIext2, "numpy")
    elif pmode == "g":
        fIext2_lambda = lambdify([Iext2_var, g, tau1], fIext2, "numpy")
    elif pmode == "z*g":
        fIext2_lambda = lambdify([Iext2_var, z, g, tau1], fIext2, "numpy")
    else:
        fIext2_lambda = lambdify([Iext2_var, Iext2, tau1], fIext2, "numpy")

    return fIext2_lambda, fIext2, vars_dict

def symbol_eqtn_fK(n, shape=None):

    K_var, K, tau1, tau0, vars_dict = symbol_vars(n, ["K_var", "K", "tau1", "tau0"], shape=shape)

    fK = Matrix(eqtn_fK(K_var, K, tau1, tau0))

    return lambdify([K_var, K, tau1, tau0], fK, "numpy"), fK, vars_dict


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

    return (fx0_lambda, slope_lambda, Iext1_lambda, Iext2_lambda, K_lambda), \
           (fx0, fslope, fIext1, fIext2, fK), vars_dict


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
        f_lambda.append(fl)
        f_sym.append(fs)
        symvars.update(temp)

        fl, fs, temp = symbol_eqtn_fy2(n, x2_neg, shape)
        f_lambda.append(fl)
        f_sym.append(fs)
        symvars.update(temp)

        fl, fs, temp = symbol_eqtn_fg(n, shape)
        f_lambda.append(fl)
        f_sym.append(fs)
        symvars.update(temp)

    elif model_vars == 11:

        fl, fs, symvars = symbol_eqtn_fx1(n, model="11d", x1_neg=x1_neg, slope="slope_var", Iext1="Iext1", shape=shape)
        f_lambda.append(fl)
        f_sym.append(fs)

        fl, fs, temp = symbol_eqtn_fy1(n, shape)
        f_lambda.append(fl)
        f_sym.append(fs)
        symvars.update(temp)

        fl, fs, temp = symbol_eqtn_fz(n, zmode=zmode, z_pos=z_pos, model="11d", x0="x0_var", K="K_var", shape=shape)
        f_lambda.append(fl)
        f_sym.append(fs)
        symvars.update(temp)

        fl, fs, temp = symbol_eqtn_fx2(n, Iext2="Iext2_var", shape=shape)
        f_lambda.append(fl)
        f_sym.append(fs)
        symvars.update(temp)

        fl, fs, temp = symbol_eqtn_fy2(n, x2_neg, shape)
        f_lambda.append(fl)
        f_sym.append(fs)
        symvars.update(temp)

        fl, fs, temp = symbol_eqtn_fg(n, shape)
        f_lambda.append(fl)
        f_sym.append(fs)
        symvars.update(temp)

        fl, fs, temp = symbol_eqtn_fparam_vars(n, pmode, shape)
        f_lambda += list(fl)
        f_sym += list(fs)
        symvars.update(temp)

    return f_lambda, f_sym, symvars


def symbol_calc_jac(n_regions, model_vars, zmode=array("lin"), x1_neg=True, x2_neg=True, z_pos=True,
                    pmode=array("const")):

    dfun_sym, v = symbol_eqnt_dfun(n_regions, model_vars, zmode, x1_neg, x2_neg, z_pos, pmode)[1:]

    dfun_sym = Matrix(Matrix(dfun_sym)[:])

    jac_lambda = []

    ind = lambda x: x*n_regions + array(range(n_regions))

    if model_vars == 2:

        jac_sym = dfun_sym.jacobian(Matrix(Matrix([v["x1"], v["z"]])[:]))

        jac_lambda.append(lambdify([v["x1"], v["z"], v["y1"], v["Iext1"], v["slope"], v["a"], v["b"], v["tau1"]],
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
                                    v["s"], v["tau1"], v["tau2"]], jac_sym[ind(4), :], "numpy"))
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
                               v["s"], v["tau1"], v["tau2"]], jac_sym[ind(4), :], "numpy"))
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


def symbol_calc_coupling_diff(n, ix=None, jx=None, K="K"):

    if ix is None:
        ix = range(n)

    if jx is None:
        jx = range(n)

    coupl, v = symbol_eqtn_coupling(n, ix, jx, K)[1:]

    x = Matrix(v["x1"][jx].tolist())
    dcoupl_dx = coupl.jacobian(x)

    return lambdify([v["K"], v["w"]], dcoupl_dx, "numpy"), dcoupl_dx, v


def symbol_calc_2d_taylor(n, x_taylor="x1lin", order=2, x1_neg=True, slope="slope", Iext1="Iext1", shape=None):

    fx1lin, v = symbol_eqtn_fx1(n, model="2d", x1_neg=x1_neg, slope=slope, Iext1=Iext1)[1:]

    x_taylor= symbol_vars(n, [x_taylor])[0]

    v.update({"x_taylor": x_taylor})

    for ix in range(v["x1"].size):
        fx1lin[ix] = series(fx1lin[ix], x=v["x1"][ix], x0=x_taylor[ix], n=order).removeO()  #

    if shape is not None:
        fx1lin = fx1lin.reshape(shape[0],shape[1])

    return lambdify([v["x1"], x_taylor, v["z"], v["y1"], v[Iext1], v[slope], v["a"], v["b"], v["tau1"]], fx1lin, "numpy"), \
           fx1lin, v


def symbol_calc_fx1z_2d_x1neg_zpos_jac(n, ix0, iE):

    fx1, v = symbol_eqtn_fx1(n, model="2d", x1_neg=True, slope="slope", Iext1="Iext1", shape=None)[1:]

    fz, vz = symbol_eqtn_fz(n, zmode=array("lin"), z_pos=True, model="2d", x0="x0", K="K", shape=None)[1:]

    v.update(vz)
    del vz

    x = empty_like(v["x1"])
    x[iE] = v["x0"][iE]
    x[ix0] = v["x1"][ix0]
    x = Matrix(x.tolist()).T

    jac = []
    for ix in range(n):
        fx1[ix] = fx1[ix].subs(v["tau1"][ix], 1.0).subs(v["z"][ix], 0.0)
        fz[ix] = fz[ix].subs(v["z"][ix], fx1[ix])
        jac.append(Matrix([fz[ix]]).jacobian(x)[:])

    jac = Matrix(jac[:])

    return lambdify([v["x1"], v["z"], v["x0"], v["x0cr"], v["r"], v["y1"], v["Iext1"], v["K"], v["w"], v["a"], v["b"],
                     v["tau1"], v["tau0"]], jac, "numpy"), jac, v


def symbol_calc_fx1y1_6d_diff_x1(n, shape=None):

    fx1, v = symbol_eqtn_fx1(n, model="6d", x1_neg=True, slope="slope", Iext1="Iext1", shape=None)[1:]

    fy1, vy = symbol_eqtn_fy1(n, shape=None)[1:]

    v.update(vy)
    del vy

    dfx1 = []
    for ix in range(n):
        fy1[ix] = fy1[ix].subs(v["y1"][ix], 0.0).subs(v["tau1"][ix], 1.0)
        fx1[ix] = fx1[ix].subs(v["y1"][ix], fy1[ix])
        dfx1.append(fx1[ix].diff(v["x1"][ix]))

    dfx1 = Matrix(dfx1)
    if shape is not None:
        dfx1 = dfx1.reshape(shape[0],shape[1])

    return lambdify([v["x1"], v["yc"], v["Iext1"], v["a"], v["b"], v["d"], v["tau1"]], dfx1, "numpy"), dfx1, v


def symbol_calc_x0cr_r(n, zmode=array("lin"), x1_rest=X1_DEF, x1_cr=X1_EQ_CR_DEF, x0def=X0_DEF, x0cr_def=X0_CR_DEF,
                       shape=None):

    # Define the z equilibrium expression...
    # if epileptor_model == "2d":
    zeq, vx = symbol_eqtn_fx1(n, model="2d", x1_neg=True, slope="slope", Iext1="Iext1")[1:]

    for iv in range(n):
        zeq[iv] = zeq[iv].subs([(vx["z"][iv], 0.0), (vx["tau1"][iv], 1.0)])

    # else:
    # zeq = calc_fx1(x1eq, z=0.0, y1=y1=calc_fy1(x1eq, y11), Iext1=I1, model="6d", x1_neg=True,
    # shape=Iext1.shape).tolist()

    # Define the fz expression...
    # fz = calc_fz(x1eq, z=zeq, x0=x0, x0cr=x0cr, r=r, zmode=zmode, z_pos=True, model="2d", shape=Iext1.shape).tolist()
    fz, v = symbol_eqtn_fz(n, zmode, z_pos=True, model="2d", x0="x0", K="K")[1:]
    for iv in range(n):
        fz[iv] = fz[iv].subs([(v['K'][iv], 0.0), (v["tau1"][iv], 1.0), (v["tau0"][iv], 1.0), (v["z"][iv], zeq[iv])])

    v.update(vx)

    # solve the fz expression for rx0 and x0cr, assuming the following two points (x1eq,x0) = [(-5/3,0.0),(-4/3,1.0)]...
    # ...and WITHOUT COUPLING
    x0cr = []
    r = []
    for iv in range(n):
        fz_sol = solve([fz[iv].subs([(v["x1"][iv], x1_rest), (v["x0"][iv], x0def),
                                (zeq[iv], zeq[iv].subs(v["x1"][iv], x1_rest))]),
                        fz[iv].subs([(v["x1"][iv], x1_cr), (v["x0"][iv], x0cr_def),
                                (zeq[iv], zeq[iv].subs(v["x1"][iv], x1_cr))])],
                        v["x0cr"][iv], v["r"][iv])
        x0cr.append(fz_sol[v["x0cr"][iv]])
        r.append(fz_sol[v["r"][iv]])

    # Convert the solution of x0cr from expression to function that accepts numpy arrays as inputs:
    x0cr = Matrix(x0cr)
    r = Matrix(r)
    if shape is not None:
        x0cr = x0cr.reshape(shape[0], shape[1])
        r = r.reshape(shape[0], shape[1])

    return (lambdify([v["y1"], v["Iext1"], v["a"], v["b"]], x0cr, 'numpy'),
           lambdify([v["y1"], v["Iext1"], v["a"], v["b"]], r, 'numpy')), (x0cr, r), v


def symbol_eqtn_fx1z(n, model="6d", zmode=array("lin"), shape=None):  #x1_neg=True, z_pos=True,

    # TODO: for the extreme z_pos = False case where we have terms like 0.1 * z ** 7
    # TODO: for the extreme x1_neg = False case where we have to solve for x2 as well

    fx1, v = symbol_eqtn_fx1(n, model, x1_neg=True, slope="slope", Iext1="Iext1")[1:]

    fz, vz = symbol_eqtn_fz(n, zmode, True, model, x0="x0", K="K")[1:]

    v.update(vz)
    del vz

    if model != "2d":

        y1eq, vy = symbol_eqtn_fy1(n)[1:]

        for iv in range(n):
            y1eq[iv] = y1eq[iv].subs([(vy["tau1"][iv], 1.0), (vy["y1"][iv], 0.0)])

        v.update(vy)
        del vy

        z = []
        for iv in range(n):
            z.append(fx1[iv].subs([(v["tau1"][iv], 1.0), (v["y1"][iv], y1eq[iv]), (v["z"][iv], 0.0)]))

    else:

        z = []
        for iv in range(n):
            z.append(fx1[iv].subs([(v["tau1"][iv], 1.0), (v["z"][iv], 0.0)]))

    fx1z = []
    for iv in range(n):
        fx1z.append(fz[iv].subs([(v["z"][iv], z[iv])]))

    # Convert the solution of x0cr from expression to function that accepts numpy arrays as inputs:
    fx1z = Matrix(fx1z)
    if shape is not None:
        fx1z = fx1z.reshape(shape[0], shape[1])

    if model == "2d":
        fx1z_lambda = lambdify([v["x1"], v["x0"], v["K"], v["w"], v["x0cr"], v["r"], v["y1"], v["Iext1"], v["a"],
                                v["b"], v["tau1"], v["tau0"]], fx1z, 'numpy')
    else:
        fx1z_lambda = lambdify([v["x1"], v["x0"], v["K"], v["w"], v["yc"], v["Iext1"], v["a"], v["b"], v["d"],
                                v["tau1"], v["tau0"]], fx1z, 'numpy')

    return fx1z_lambda, fx1z, v


def symbol_eqtn_fx1z_diff(n, model="6d", zmode=array("lin")): #x1_neg=True, , z_pos=True

    # TODO: for the extreme z_pos = False case where we have terms like 0.1 * z ** 7
    # TODO: for the extreme x1_neg = False case where we have to solve for x2 as well

    fx1z, v = symbol_eqtn_fx1z(n, model, zmode)[1:]

    #fx1z = Matrix(Matrix(fx1z)[:])

    dfx1z_dx1 = fx1z.jacobian(Matrix(Matrix([v["x1"]])[:]))

    if model == "2d":
        dfx1z_dx1_lambda = lambdify([v["x1"], v["K"], v["w"], v["a"], v["b"], v["tau1"], v["tau0"]],
                                     dfx1z_dx1, 'numpy')
    else:
        dfx1z_dx1_lambda = lambdify([v["x1"], v["K"], v["w"], v["a"], v["b"], v["d"], v["tau1"], v["tau0"]],
                                     dfx1z_dx1, 'numpy')

    return dfx1z_dx1_lambda, dfx1z_dx1, v


def symbol_eqtn_fx2y2(n, x2_neg=True):

    y2eq, vy = symbol_eqtn_fy2(n, x2_neg=x2_neg)[1:]

    for iv in range(n):
        y2eq[iv] = y2eq[iv].subs([(vy["tau1"][iv], 1.0), (vy["tau2"][iv], 1.0), (vy["y2"][iv], 0.0)])

    fx2, v = symbol_eqtn_fx2(n, Iext2="Iext2")[1:]

    v.update(vy)
    del vy

    for iv in range(n):
        fx2[iv] = fx2[iv].subs(v["y2"][iv], y2eq[iv])

    return lambdify([v["x2"], v["z"], v["g"], v["Iext2"], v["s"], v["tau1"]], fx2, 'numpy'), fx2, v


def symbol_calc_eq_x1(n, model="6d", zmode=array("lin")):

    fx1z, v = symbol_eqtn_fx1z(n, model, zmode, x1_neg=True, z_pos=True)[1:]
    fsolve = []
    for iv in range(n):
        fsolve.append(fx1z[iv].subs([(v["tau1"][iv], 1.0), (v["tau0"][iv], 1.0)]))

    x1 = [Symbol("x_" + '%d' % i_n, real=True, negative=True) for i_n in range(v["x1"].size)]

    x1eq = solve(fsolve, x1[:])

    # Convert the solution of x0cr from expression to function that accepts numpy arrays as inputs:
    x1eq = Matrix(x1eq)

    if model == "2d":
        x1eq_lambda = lambdify([v["x0"], v["K"], v["w"], v["x0cr"], v["r"], v["yc"], v["Iext1"], v["a"], v["b"]], x1eq,
                               'numpy')
    else:
        x1eq_lambda = lambdify([v["x0"], v["K"], v["w"], v["yc"], v["Iext1"], v["a"], v["b"], v["d"]], x1eq, 'numpy')

    return x1eq_lambda, x1eq, v


def symbol_calc_eq_x2(n, x2_neg=True):

    fx2, v = symbol_eqtn_fx2y2(n, x2_neg=x2_neg)[1:]

    for iv in range(n):
        fx2[iv] = fx2[iv].subs(v["tau1"][iv], 1.0)

    x2eq = []
    for iv in range(n):
        x2eq.append(solve(fx2[iv], v["x2"][iv]))

    x2eq = Matrix(array(x2eq))

    return lambdify([v["z"], v["g"], v["Iext2"], v["s"]], x2eq, 'numpy'), x2eq, v

