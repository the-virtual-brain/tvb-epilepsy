import numpy
from numpy import array, empty_like, reshape
from sympy import Symbol, solve, solveset, lambdify, series, Matrix  # diff, ArraySymbol
from sympy.tensor.array import Array

from tvb_epilepsy.base.computations.equations_utils import *
from tvb_epilepsy.base.utils.data_structures_utils import shape_to_size


def symbol_vars(n_regions, vars_str, dims=1, ind_str="_", shape=None, output_flag="numpy_array"):
    vars_out = list()
    vars_dict = {}
    if dims == 1:
        if shape is None:
            if output_flag == "sympy_array":
                shape = (n_regions, 1)
            else:
                shape = (n_regions, )
        for vs in vars_str:
            temp = [Symbol(vs+ind_str+'%d' % i_n, real=True) for i_n in range(n_regions)]
            if output_flag == "numpy_array":
                temp = reshape(temp, shape)
            elif output_flag == "sympy_array":
                temp = Array(temp).reshape(shape[0], shape[1])
            vars_out.append(temp)
            vars_dict[vs] = vars_out[-1:][0]
    elif dims == 0:
        if shape is None:
            if output_flag == "sympy_array":
                shape = (1, 1)
            else:
                shape = (1, )
        for vs in vars_str:
            temp = Symbol(vs, real=True)
            if output_flag == "numpy_array":
                temp = reshape(temp, shape)
            elif output_flag == "sympy_array":
                temp = Array(temp).reshape(shape[0], shape[1])
            vars_out.append(temp)
            vars_dict[vs] = vars_out[-1:][0]
    elif dims == 2:
        if shape is None:
            shape = (n_regions, n_regions)
        for vs in vars_str:
            temp = []
            for i_n in range(n_regions):
                temp.append([Symbol(vs + ind_str + '%d' % i_n + ind_str + '%d' % j_n, real=True)
                             for j_n in range(n_regions)])
            if output_flag == "numpy_array":
                temp = reshape(temp, shape)
            elif output_flag == "sympy_array":
                temp = Array(temp).reshape(shape[0], shape[1])
            vars_out.append(temp)
            vars_dict[vs] = vars_out[-1:][0]
    else:
        raise_value_error("The dimensionality of the variables is neither 1 nor 2: " + str(dims))
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
    coupling = Array(eqtn_coupling(x1, K, w, ix, jx))
    return lambdify([x1, K, w], coupling, "numpy"), coupling, vars_dict


def symbol_eqtn_x0cr_r(n, zmode=numpy.array("lin"), shape=None):
    Iext1, yc, a, b, d, x1_rest, x1_cr, x0_rest, x0_cr, vars_dict = \
        symbol_vars(n, ["Iext1", "yc", "a", "b", "d", "x1_rest", "x1_cr", "x0_rest", "x0_cr"], shape=shape)
    x0cr, r = eqtn_x0cr_r(yc, Iext1, a, b, d, x1_rest, x1_cr, x0_rest, x0_cr, zmode=zmode)
    x0cr = Array(x0cr)
    r = Array(r)
    x0cr_lambda = lambdify([yc, Iext1, a, b, d, x1_rest, x1_cr, x0_rest, x0_cr,], x0cr, "numpy")
    r_lambda = lambdify([yc, Iext1, a, b, d, x1_rest, x1_cr, x0_rest, x0_cr,], r, "numpy")
    return (x0cr_lambda, r_lambda), (x0cr, r), vars_dict


def symbol_eqtn_x0(n, zmode=numpy.array("lin"), z_pos=True, K="K", shape=None):
    x1, z, K, vars_dict = symbol_vars(n, ["x1", "z", K], shape=shape)
    w, temp = symbol_vars(n, ["w"], dims=2)
    vars_dict.update(temp)
    x0 = Array(eqtn_x0(x1, z, zmode, z_pos, K, w))
    x0_lambda = lambdify([x1, z, K, w], x0, "numpy"), x0, vars_dict
    return x0_lambda, x0, vars_dict


def symbol_eqtn_fx1(n, model="2d", x1_neg=True, slope="slope", Iext1="Iext1", shape=None):
    x1, z, y1, slope, Iext1, a, b, d, tau1, vars_dict = symbol_vars(n, ["x1", "z", "y1", slope, Iext1, "a", "b", "d",
                                                                        "tau1"], shape=shape)
    if model == "2d":
        fx1 = Array(eqtn_fx1(x1, z, y1, Iext1, slope, a, b, d, tau1, x1_neg, model, x2=None))
        return lambdify([x1, z, y1, Iext1, slope, a, b, d, tau1], fx1, "numpy"), fx1, vars_dict
    else:
        x2, vx2 = symbol_vars(n, ["x2"], shape=shape)
        vars_dict.update(vx2)
        fx1 = Array(eqtn_fx1(x1, z, y1, Iext1, slope, a, b, d, tau1, x1_neg, model, x2=x2))
        return lambdify([x1, z, y1, x2, Iext1, slope, a, b, d, tau1], fx1, "numpy"), fx1, vars_dict



def symbol_eqtn_fy1(n, shape=None):
    x1, y1, yc, d, tau1, vars_dict = symbol_vars(n, ["x1", "y1", "yc", "d", "tau1"], shape=shape)
    fy1 = Array(eqtn_fy1(x1, yc, y1, d, tau1))
    return lambdify([x1, y1, yc, d, tau1], fy1, "numpy"), fy1, vars_dict


def symbol_eqtn_fz(n, zmode=numpy.array("lin"), z_pos=True, x0="x0", K="K", shape=None):
    x1, z, x0, K, tau1, tau0, vars_dict = symbol_vars(n, ["x1", "z", x0, K, "tau1", "tau0"], shape=shape)
    w, temp = symbol_vars(n, ["w"], dims=2)
    vars_dict.update(temp)
    fz = Array(eqtn_fz(x1, z, x0, tau1, tau0, zmode, z_pos, K=K, w=w))
    fz_lambda = lambdify([x1, z, x0, K, w, tau1, tau0], fz, "numpy")
    return fz_lambda, fz, vars_dict


def symbol_eqtn_fx2(n, Iext2="Iext2", shape=None):
    x2, y2, z, g, Iext2, tau1, vars_dict = symbol_vars(n, ["x2", "y2", "z", "g", Iext2, "tau1"], shape=shape)
    fx2 = Array(eqtn_fx2(x2, y2, z, g, Iext2, tau1))
    return lambdify([x2, y2, z, g, Iext2, tau1], fx2, "numpy"), fx2, vars_dict


def symbol_eqtn_fy2(n, x2_neg=False, shape=None):
    x2, y2, s, tau1, tau2, vars_dict = symbol_vars(n, ["x2", "y2", "s", "tau1", "tau2"], shape=shape)
    fy2 = Array(eqtn_fy2(x2, y2, s, tau1, tau2, x2_neg))
    return lambdify([x2, y2, s, tau1, tau2], fy2, "numpy"), fy2, vars_dict


def symbol_eqtn_fg(n, shape=None):
    x1, g, gamma, tau1, vars_dict = symbol_vars(n, ["x1", "g", "gamma", "tau1"], shape=shape)
    fg = Array(eqtn_fg(x1, g, gamma, tau1))
    return lambdify([x1, g, gamma, tau1], fg, "numpy"), fg, vars_dict


def symbol_eqtn_fx0(n, shape=None):
    x0_var, x0, tau1, vars_dict = symbol_vars(n, ["x0_var", "x0", "tau1"], shape=shape)
    fx0 = Array(eqtn_fx0(x0_var, x0, tau1))
    return lambdify([x0_var, x0, tau1], fx0, "numpy"), fx0, vars_dict


def symbol_eqtn_fslope(n, pmode=array("const"), shape=None):
    slope_var, z, g, slope, tau1, vars_dict = symbol_vars(n, ["slope_var", "z", "g", "slope", "tau1"], shape=shape)
    from tvb_epilepsy.base.epileptor_models import EpileptorDPrealistic
    slope_eq = EpileptorDPrealistic.fun_slope_Iext2(z, g, pmode, slope, 0.0)[0]
    fslope = Array(eqtn_fslope(slope_var, slope_eq, tau1))
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
    fIext1 = Array(eqtn_fIext1(Iext1_var, Iext1, tau1, tau0))
    return lambdify([Iext1_var, Iext1, tau1, tau0], fIext1, "numpy"), fIext1, vars_dict


def symbol_eqtn_fIext2(n, pmode=array("const"), shape=None):
    Iext2_var, z, g, Iext2, tau1, vars_dict = symbol_vars(n, ["Iext2_var", "z", "g", "Iext2", "tau1"], shape=shape)
    from tvb_epilepsy.base.epileptor_models import EpileptorDPrealistic
    Iext2_eq = EpileptorDPrealistic.fun_slope_Iext2(z, g, pmode, 0.0, Iext2)[1]
    fIext2 = Array(eqtn_fIext2(Iext2_var, Iext2_eq, tau1))
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
    fK = Array(eqtn_fK(K_var, K, tau1, tau0))
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


def symbol_eqnt_dfun(n, model_vars, zmode=array("lin"), x1_neg=True, x2_neg=False, z_pos=True,
                     pmode=array("const"), output_mode="array", shape=None):
    f_sym = []
    if output_mode != "array":
        f_lambda = []
    if model_vars == 2:
        flx1, fs, v = symbol_eqtn_fx1(n, model="2d", x1_neg=x1_neg, slope="slope", Iext1="Iext1", shape=shape)
        f_sym.append(fs)
        flz, fs, temp = symbol_eqtn_fz(n, zmode=zmode, z_pos=z_pos, x0="x0", K="K", shape=shape)
        f_sym.append(fs)
        v.update(temp)
        if output_mode == "array":
            symvars = [v["x1"], v["z"], v["y1"], v["Iext1"], v["x0"], v["K"], v["w"], v["slope"],
                       v["a"], v["b"], v["d"], v["tau1"], v["tau0"]]
        else:
            f_lambda.append(flx1).append(flz)
    elif model_vars == 6:
        flx1, fs, v = symbol_eqtn_fx1(n, model="6d", x1_neg=x1_neg, slope="slope", Iext1="Iext1", shape=shape)
        f_sym.append(fs)
        fly1, fs, temp = symbol_eqtn_fy1(n, shape)
        f_sym.append(fs)
        v.update(temp)
        flz, fs, temp = symbol_eqtn_fz(n, zmode=zmode, z_pos=z_pos, x0="x0", K="K", shape=shape)
        f_sym.append(fs)
        v.update(temp)
        flx2, fs, temp = symbol_eqtn_fx2(n, Iext2="Iext2", shape=shape)
        f_sym.append(fs)
        v.update(temp)
        fly2, fs, temp = symbol_eqtn_fy2(n, x2_neg, shape)
        f_sym.append(fs)
        v.update(temp)
        flg, fs, temp = symbol_eqtn_fg(n, shape)
        f_sym.append(fs)
        v.update(temp)
        if output_mode == "array":
            symvars = [v["x1"], v["y1"], v["z"], v["x2"], v["y2"], v["g"],
                       v["yc"], v["Iext1"], v["Iext2"], v["x0"], v["K"], v["w"], v["slope"],
                       v["a"], v["b"], v["d"], v["s"], v["gamma"], v["tau1"], v["tau0"], v["tau2"]]
        else:
            f_lambda.append(flx1).append(fly1).append(flz).append(flx2).append(fly2).append(flg)
    elif model_vars == 11:
        flx1, fs, v = symbol_eqtn_fx1(n, model="11d", x1_neg=x1_neg, slope="slope_var", Iext1="Iext1", shape=shape)
        f_sym.append(fs)
        fly1, fs, temp = symbol_eqtn_fy1(n, shape)
        f_sym.append(fs)
        v.update(temp)
        flz, fs, temp = symbol_eqtn_fz(n, zmode=zmode, z_pos=z_pos, x0="x0_var", K="K_var", shape=shape)
        f_sym.append(fs)
        v.update(temp)
        flx2, fs, temp = symbol_eqtn_fx2(n, Iext2="Iext2_var", shape=shape)
        f_sym.append(fs)
        v.update(temp)
        fly2, fs, temp = symbol_eqtn_fy2(n, x2_neg, shape)
        f_sym.append(fs)
        v.update(temp)
        flg, fs, temp = symbol_eqtn_fg(n, shape)
        f_sym.append(fs)
        v.update(temp)
        flparams, fs, temp = symbol_eqtn_fparam_vars(n, pmode, shape)
        f_sym += list(fs)
        v.update(temp)
        if output_mode == "array":
            symvars = [v["x1"], v["y1"], v["z"], v["x2"], v["y2"], v["g"],
                       v["x0_var"], v["slope_var"], v["Iext1_var"], v["Iext2_var"], v["K_var"],
                       v["yc"], v["Iext1"], v["Iext2"], v["x0"], v["K"], v["w"], v["slope"],
                       v["a"], v["b"], v["d"], v["s"], v["gamma"], v["tau1"], v["tau0"], v["tau2"]]
        else:
            f_lambda.append(flx1).append(fly1).append(flz).append(flx2).append(fly2).append(flg)
            f_lambda += flparams
    if output_mode == "array":
        f_sym = Array(f_sym)
        # shape = f_sym.shape
        # if shape[0] > shape[1]:
        #     f_sym =
        f_lambda = lambdify(symvars, f_sym, "numpy")
    return f_lambda, f_sym, v


def symbol_calc_jac(n_regions, model_vars, zmode=array("lin"), x1_neg=True, x2_neg=False, z_pos=True,
                    pmode=array("const")):
    dfun_sym, v = symbol_eqnt_dfun(n_regions, model_vars, zmode, x1_neg, x2_neg, z_pos, pmode)[1:]
    dfun_sym = Matrix(dfun_sym)
    jac_lambda = []
    ind = lambda x: x*n_regions + array(range(n_regions))
    if model_vars == 2:
        jac_sym = dfun_sym.jacobian((Matrix([v["x1"], v["z"]]).reshape(2 * n_regions, 1)))
        jac_lambda.append(lambdify([v["x1"], v["z"], v["y1"], v["Iext1"], v["slope"], v["a"], v["b"], v["d"], v["tau1"]],
                          jac_sym[ind(0), :], "numpy"))
        jac_lambda.append(lambdify([v["x1"], v["z"], v["x0"], v["K"], v["w"], v["tau1"], v["tau0"]],
                          jac_sym[ind(1), :], "numpy"))
    elif model_vars == 6:
        jac_sym = dfun_sym.jacobian(Matrix([v["x1"], v["y1"], v["z"], v["x2"], v["y2"], v["g"]]).
                                    reshape(6 * n_regions, 1))
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
        jac_sym = dfun_sym.jacobian(Matrix([v["x1"], v["y1"], v["z"], v["x2"], v["y2"], v["g"],
                                            v["x0_var"], v["slope_var"], v["Iext1_var"], v["Iext2_var"], v["K_var"]])
                                    .reshape(11 * n_regions, 1))
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
    x = Matrix(v["x1"][jx])
    dcoupl_dx = Array(Matrix(coupl).jacobian(x))
    return lambdify([v["K"], v["w"]], dcoupl_dx, "numpy"), dcoupl_dx, v


def symbol_calc_2d_taylor(n, x_taylor="x1lin", order=2, x1_neg=True, slope="slope", Iext1="Iext1", shape=None):
    fx1ser, v = symbol_eqtn_fx1(n, model="2d", x1_neg=x1_neg, slope=slope, Iext1=Iext1)[1:]
    fx1ser = fx1ser.tolist()
    x_taylor= symbol_vars(n, [x_taylor])[0]
    v.update({"x_taylor": x_taylor})
    for ix in range(shape_to_size(v["x1"].shape)):
        fx1ser[ix] = series(fx1ser[ix], x=v["x1"][ix], x0=x_taylor[ix], n=order).removeO()  #
    fx1ser = Array(fx1ser)
    if shape is not None:
        if len(shape) > 1:
            fx1ser = fx1ser.reshape(shape[0],shape[1])
        else:
            fx1ser = fx1ser.reshape(shape[0], )
    return lambdify([v["x1"], x_taylor, v["z"], v["y1"], v[Iext1], v[slope], v["a"], v["b"], v["d"], v["tau1"]],
                    fx1ser, "numpy"), fx1ser, v


def symbol_calc_fx1z_2d_x1neg_zpos_jac(n, ix0, iE):
    fx1, v = symbol_eqtn_fx1(n, model="2d", x1_neg=True, slope="slope", Iext1="Iext1", shape=None)[1:]
    fx1 = fx1.tolist()
    fz, vz = symbol_eqtn_fz(n, zmode=array("lin"), z_pos=True, x0="x0", K="K", shape=None)[1:]
    fz = fz.tolist()
    v.update(vz)
    del vz
    x = empty_like(v["x1"])
    x[iE] = v["x0"][iE]
    x[ix0] = v["x1"][ix0]
    x = Matrix(x)
    jac = []
    for ix in range(n):
        fx1[ix] = fx1[ix].subs(v["tau1"][ix], 1.0).subs(v["z"][ix], 0.0)
        fz[ix] = fz[ix].subs(v["z"][ix], fx1[ix])
        jac.append(Matrix([fz[ix]]).jacobian(x)[:])
    jac = Array(jac)
    return lambdify([v["x1"], v["z"], v["x0"], v["y1"], v["Iext1"], v["K"], v["w"], v["a"], v["b"], v["d"], v["tau1"],
                     v["tau0"]], jac, "numpy"), jac, v


def symbol_calc_fx1y1_6d_diff_x1(n, shape=None):
    fx1, v = symbol_eqtn_fx1(n, model="6d", x1_neg=True, slope="slope", Iext1="Iext1", shape=None)[1:]
    fx1 = fx1.tolist()
    fy1, vy = symbol_eqtn_fy1(n, shape=None)[1:]
    fy1 = fy1.tolist()
    v.update(vy)
    del vy
    dfx1 = []
    for ix in range(n):
        fy1[ix] = fy1[ix].subs(v["y1"][ix], 0.0).subs(v["tau1"][ix], 1.0)
        fx1[ix] = fx1[ix].subs(v["y1"][ix], fy1[ix])
        dfx1.append(fx1[ix].diff(v["x1"][ix]))
    dfx1 = Array(dfx1)
    if shape is not None:
        if len(shape) > 1:
            dfx1 = dfx1.reshape(shape[0], shape[1])
        else:
            dfx1 = dfx1.reshape(shape[0], )
    return lambdify([v["x1"], v["yc"], v["Iext1"], v["a"], v["b"], v["d"], v["tau1"]], dfx1, "numpy"), dfx1, v


def symbol_calc_x0cr_r(n, zmode=array("lin"), shape=None):
    # Define the z equilibrium expression...
    zeq, vx = symbol_eqtn_fx1(n, model="2d", x1_neg=True, slope="slope", Iext1="Iext1")[1:]
    zeq = zeq.tolist()
    for iv in range(n):
        zeq[iv] = zeq[iv].subs([(vx["z"][iv], 0.0), (vx["tau1"][iv], 1.0)])
    # Define the fz expression...
    # fz = calc_fz(x1eq, z=zeq, x0_values=x0_values, r=r, zmode=zmode, z_pos=True, shape=Iext1.shape).tolist()
    fz, v = symbol_eqtn_fz(n, zmode, z_pos=True, model="2d", x0="x0", K="K")[1:]
    fz = fz.tolist()
    for iv in range(n):
        fz[iv] = fz[iv].subs([(v['K'][iv], 0.0), (v["tau1"][iv], 1.0), (v["tau0"][iv], 1.0), (v["z"][iv], zeq[iv])])
    v.update(vx)
    del vx
    x1_rest, x1_cr, x0_rest, x0_cr, x0cr, r, vv = \
        symbol_vars(len(zeq), ["x1_rest", "x1_cr", "x0_rest", "x0_cr", "x0cr", "r"], shape=v["x1"].shape)
    v.update(vv)
    del vv
    # solve the fz expression for rx0 and x0cr, assuming the following two points
    # (x1eq,x0_values) = [(-5/3,0.0),(-4/3,1.0)]...
    # ...and WITHOUT COUPLING
    x0cr = []
    r = []
    for iv in range(n):
        fz_sol = solve([fz[iv].subs([(v["x1"][iv], x1_rest[iv]),
                                     (v["x0"][iv], x0_rest[iv]*v["r"][iv] - v["x0cr"][iv]),
                                     (zeq[iv], zeq[iv].subs(v["x1"][iv], x1_rest[iv]))]),
                        fz[iv].subs([(v["x1"][iv], x1_cr[iv]),
                                     (v["x0"][iv], x0_cr[iv]*v["r"][iv] - v["x0cr"][iv]),
                                    (zeq[iv], zeq[iv].subs(v["x1"][iv], x1_cr[iv]))])],
                        v["x0cr"][iv], v["r"][iv])
        x0cr.append(fz_sol[v["x0cr"][iv]])
        r.append(fz_sol[v["r"][iv]])
    # Convert the solution of x0cr from expression to function that accepts numpy arrays as inputs:
    x0cr = Array(x0cr)
    r = Array(r)
    if shape is not None:
        if len(shape) > 1:
            x0cr = x0cr.reshape(shape[0], shape[1])
            r = r.reshape(shape[0], shape[1])
        else:
            x0cr = x0cr.reshape(shape[0], )
            r = r.reshape(shape[0], )
    return (lambdify([v["y1"], v["Iext1"], v["a"], v["b"], v["d"], v["x1_rest"], v["x1_cr"], v["x0_rest"], v["x0_cr"]],
                     x0cr, 'numpy'),
           lambdify([v["y1"], v["Iext1"], v["a"], v["b"], v["d"], v["x1_rest"], v["x1_cr"], v["x0_rest"], v["x0_cr"]],
                    r, 'numpy')), \
           (x0cr, r), v


def symbol_eqtn_fx1z(n, model="6d", zmode=array("lin"), shape=None):  #x1_neg=True, z_pos=True,
    # TODO: for the extreme z_pos = False case where we have terms like 0.1 * z ** 7
    # TODO: for the extreme x1_neg = False case where we have to solve for x2 as well
    fx1, v = symbol_eqtn_fx1(n, model, x1_neg=True, slope="slope", Iext1="Iext1")[1:]
    fx1 = fx1.tolist()
    fz, vz = symbol_eqtn_fz(n, zmode, True, x0="x0", K="K")[1:]
    fz = fz.tolist()
    v.update(vz)
    del vz
    if model != "2d":
        y1eq, vy = symbol_eqtn_fy1(n)[1:]
        y1eq = y1eq.tolist()
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
    fx1z = Array(fx1z)
    if shape is not None:
        if len(shape) > 1:
            fx1z = fx1z.reshape(shape[0], shape[1])
        else:
            fx1z = fx1z.reshape(shape[0], )
    fx1z_lambda = lambdify([v["x1"], v["x0"], v["K"], v["w"], v["y1"], v["Iext1"], v["a"], v["b"], v["d"], v["tau1"],
                            v["tau0"]], fx1z, 'numpy')
    return fx1z_lambda, fx1z, v


def symbol_eqtn_fx1z_diff(n, model, zmode=array("lin")): #x1_neg=True, , z_pos=True
    # TODO: for the extreme z_pos = False case where we have terms like 0.1 * z ** 7
    # TODO: for the extreme x1_neg = False case where we have to solve for x2 as well
    fx1z, v = symbol_eqtn_fx1z(n, model, zmode)[1:]
    #fx1z = Array(Array(fx1z)[:])
    dfx1z_dx1 = Array(Matrix(fx1z).jacobian(Matrix([v["x1"]])))
    dfx1z_dx1_lambda = lambdify([v["x1"], v["K"], v["w"], v["a"], v["b"], v["d"], v["tau1"], v["tau0"]],
                                dfx1z_dx1, 'numpy')
    return dfx1z_dx1_lambda, dfx1z_dx1, v


def symbol_eqtn_fx2y2(n, x2_neg=False, shape=None):
    y2eq, vy = symbol_eqtn_fy2(n, x2_neg=x2_neg)[1:]
    y2eq = y2eq.tolist()
    for iv in range(n):
        y2eq[iv] = y2eq[iv].subs([(vy["tau1"][iv], 1.0), (vy["tau2"][iv], 1.0), (vy["y2"][iv], 0.0)])
    fx2, v = symbol_eqtn_fx2(n, Iext2="Iext2")[1:]
    fx2 = fx2.tolist()
    v.update(vy)
    del vy
    for iv in range(n):
        fx2[iv] = fx2[iv].subs(v["y2"][iv], y2eq[iv])
    fx2 = Array(fx2)
    if shape is not None:
        if len(shape) > 1:
            fx2 = fx2.reshape(shape[0], shape[1])
        else:
            fx2 = fx2.reshape(shape[0], )
    return lambdify([v["x2"], v["z"], v["g"], v["Iext2"], v["s"], v["tau1"]], fx2, 'numpy'), fx2, v


def symbol_calc_fz_jac_square_taylor(n):
    fx1sq, v = symbol_calc_2d_taylor(n, x_taylor="x1sq", order=3, x1_neg=True, slope="slope", Iext1="Iext1")[1:]
    fx1sq = fx1sq.tolist()
    fz, vz = symbol_eqtn_fz(n, zmode=array("lin"), z_pos=True)[1:]
    fz = fz.tolist()
    v.update(vz)
    del vz
    x1 = []
    #dfx1z = []
    for iv in range(n):
        x1.append(list(solveset(fx1sq[iv], v["x1"][iv]))[0])
        #dfx1z.append(x1[iv].diff(v["z"][iv]))
    for iv in range(n):
        for jv in range(n):
            fz[iv] = fz[iv].subs(v["x1"][jv], x1[jv])
    fz_jac = Matrix(fz).jacobian(Matrix([v["z"]]))
    # for iv in range(n):
    #     for jv in range(n):
    #         fz_jac[iv, jv].simplify().collect(dfx1z[jv])
    fz_jac = Array(fz_jac)
    fz_jac_lambda = lambdify([v["z"], v["y1"], v["Iext1"], v["K"], v["w"], v["a"], v["b"], v["d"], v["tau1"], v["tau0"],
                              v["x_taylor"]], fz_jac, 'numpy')
    return fz_jac_lambda, fz_jac, v
