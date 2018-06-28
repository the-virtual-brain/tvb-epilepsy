# coding=utf-8

from scipy.optimize import root
from tvb_fit.base.utils.data_structures_utils import assert_arrays, shape_to_size
from tvb_fit.base.utils.log_error_utils import initialize_logger, raise_import_error, raise_value_error
from tvb_fit.tvb_epilepsy.base.constants.model_constants import *
from tvb_fit.tvb_epilepsy.base.computation_utils.equations_utils import *


logger = initialize_logger(__name__)

try:
    from sympy import Matrix, lambdify, reshape
    from tvb_fit.tvb_epilepsy.base.computation_utils.symbolic_utils \
        import symbol_eqtn_coupling, symbol_eqtn_x0, \
        symbol_eqtn_fx1, symbol_eqtn_fy1, symbol_eqtn_fz, symbol_calc_fz_jac_square_taylor, symbol_calc_x0cr_r, \
        symbol_calc_fx1y1_6d_diff_x1, symbol_calc_fx1z_2d_x1neg_zpos_jac, symbol_eqtn_fx1z_diff, symbol_eqtn_x0cr_r, \
        symbol_eqtn_fx1z, symbol_calc_2d_taylor, symbol_calc_coupling_diff, symbol_eqtn_fx2, symbol_eqtn_fy2, \
        symbol_eqtn_fg, symbol_eqtn_fx0, symbol_eqtn_fslope, symbol_eqtn_fIext1, symbol_eqtn_fK, symbol_eqnt_dfun, \
        symbol_eqtn_fIext2, symbol_calc_jac, symbol_vars

    SYMBOLIC_IMPORT = True

except ImportError:
    logger.exception("Could not load...")
    logger.warning("Unable to load symbolic_equations module! Symbolic computation_utils are not possible!")
    SYMBOLIC_IMPORT = False


def confirm_calc_mode(calc_mode):
    if np.all(calc_mode == "symbol"):
        if SYMBOLIC_IMPORT:
            logger.info("Executing symbolic computation_utils...")
        else:
            logger.warning("\nNot possible to execute symbolic computation_utils! Turning to non-symbolic ones!..")
            calc_mode = "non_symbol"
    return calc_mode


def calc_coupling(x1, K, w, ix=None, jx=None, shape=None, calc_mode="non_symbol"):
    calc_mode = confirm_calc_mode(calc_mode)
    x1, K = assert_arrays([x1, K], shape)
    n_regions = x1.size
    w = assert_arrays([w], (x1.size, x1.size))
    if ix is None:
        ix = range(n_regions)
    if jx is None:
        jx = range(n_regions)
    if np.all(calc_mode == "symbol"):
        return np.array(symbol_eqtn_coupling(x1.size, ix, jx, shape=x1.shape)[0](x1, K, w))
    else:
        return eqtn_coupling(x1, K, w, ix, jx)


def calc_x0(x1, z, K=0.0, w=0.0, zmode=np.array("lin"), z_pos=True, shape=None, calc_mode="non_symbol"):
    calc_mode = confirm_calc_mode(calc_mode)
    x1, z, K = assert_arrays([x1, z, K], shape)
    w = assert_arrays([w], (z.size, z.size))
    if np.all(calc_mode == "symbol"):
        return np.array(symbol_eqtn_x0(z.size, zmode, z_pos, "K", z.shape)[0](x1, z, K, w))
    else:
        if zmode == np.array("lin") and z_pos is None:
            z_pos = z > 0.0
        return eqtn_x0(x1, z, zmode, z_pos, K, w)


def calc_fx1(x1=0.0, z=0.0, y1=0.0, Iext1=I_EXT1_DEF, slope=SLOPE_DEF, a=A_DEF, b=B_DEF, d=D_DEF, tau1=TAU1_DEF, x2=0.0,
             model="2d", x1_neg=True, shape=None, calc_mode="non_symbol"):
    calc_mode = confirm_calc_mode(calc_mode)
    x1, z, y1, Iext1, slope, a, b, d, tau1 = assert_arrays([x1, z, y1, Iext1, slope, a, b, d, tau1], shape)
    if np.all(calc_mode == "symbol"):
        if np.all(model == "2d"):
            return np.array(symbol_eqtn_fx1(x1.size, model, x1_neg, slope="slope", Iext1="Iext1", shape=x1.shape)[0]
                            (x1, z, y1, Iext1, slope, a, b, d, tau1))
        else:
            x2 = assert_arrays([x2], x1.shape)
            return np.array(symbol_eqtn_fx1(x1.size, model, x1_neg, slope="slope", Iext1="Iext1", shape=x1.shape)[0]
                            (x1, z, y1, x2, Iext1, slope, a, b, d, tau1))
    else:
        if x1_neg is None:
            x1_neg = x1 < 0.0
        if np.all(model == "2d"):
            return eqtn_fx1(x1, z, y1, Iext1, slope, a, b, d, tau1, x1_neg, model, x2=None)
        else:
            x2 = assert_arrays([x2], x1.shape)
            return eqtn_fx1(x1, z, y1, Iext1, slope, a, b, d, tau1, x1_neg, model, x2)


def calc_fy1(x1, yc, y1=0, d=D_DEF, tau1=TAU1_DEF, shape=None, calc_mode="non_symbol"):
    calc_mode = confirm_calc_mode(calc_mode)
    x1, yc, y1, d, tau1 = assert_arrays([x1, yc, y1, d, tau1], shape)
    if np.all(calc_mode == "symbol"):
        return np.array(symbol_eqtn_fy1(x1.size, x1.shape)[0](x1, y1, yc, d, tau1))
    else:
        return eqtn_fy1(x1, yc, y1, d, tau1)


def calc_fz(x1=0.0, z=0, x0=0.0, K=0.0, w=0.0, tau1=TAU1_DEF, tau0=TAU0_DEF, zmode=np.array("lin"), z_pos=True,
            shape=None, calc_mode="non_symbol"):
    calc_mode = confirm_calc_mode(calc_mode)
    x1, z, x0, K, tau1, tau0 = assert_arrays([x1, z, x0, K, tau1, tau0], shape)
    w = assert_arrays([w], (z.size, z.size))
    if np.all(calc_mode == "symbol"):
        return np.array(symbol_eqtn_fz(z.size, zmode, z_pos, x0="x0_values", K="K", shape=z.shape)[0](x1, z, x0,
                                                                                                      K, w, tau1,
                                                                                                      tau0))
    else:
        if zmode == np.array("lin") and z_pos is None:
            z_pos = z > 0.0
        return eqtn_fz(x1, z, x0, tau1, tau0, zmode, z_pos, K, w, coupl=None)


def calc_fx2(x2, y2=0.0, z=0.0, g=0.0, Iext2=I_EXT2_DEF, tau1=TAU1_DEF, shape=None, calc_mode="non_symbol"):
    calc_mode = confirm_calc_mode(calc_mode)
    x2, y2, z, g, Iext2, tau1 = assert_arrays([x2, y2, z, g, Iext2, tau1], shape)
    if np.all(calc_mode == "symbol"):
        return np.array(symbol_eqtn_fx2(x2.size, Iext2="Iext2", shape=x2.shape)[0](x2, y2, z, g, Iext2, tau1))
    else:
        return eqtn_fx2(x2, y2, z, g, Iext2, tau1)


def calc_fy2(x2, y2=0.0, s=S_DEF, tau1=TAU1_DEF, tau2=TAU2_DEF, x2_neg=None, shape=None, calc_mode="non_symbol"):
    calc_mode = confirm_calc_mode(calc_mode)
    x2, y2, s, tau1, tau2 = assert_arrays([x2, y2, s, tau1, tau2], shape)
    if np.any(x2_neg is None):
        try:
            x2_neg = x2 < -0.25
        except:
            x2_neg = False
            logger.warning("\nx2_neg is None and failed to compare x2_neg = x2 < -0.25!" +
                           "\nSetting default x2_neg = False")
    if np.all(calc_mode == "symbol"):
        return np.array(symbol_eqtn_fy2(x2.size, x2_neg=x2_neg, shape=x2.shape)[0](x2, y2, s, tau1, tau2))
    else:
        return eqtn_fy2(x2, y2, s, tau1, tau2, x2_neg)


def calc_fg(x1, g=0.0, gamma=GAMMA_DEF, tau1=TAU1_DEF, shape=None, calc_mode="non_symbol"):
    calc_mode = confirm_calc_mode(calc_mode)
    x1, g, gamma, tau1 = assert_arrays([x1, g, gamma, tau1], shape)
    if np.all(calc_mode == "symbol"):
        return np.array(symbol_eqtn_fg(x1.size, x1.shape)[0](x1, g, gamma, tau1))
    else:
        return eqtn_fg(x1, g, gamma, tau1)


def calc_fx0(x0_var, x0, tau1=TAU1_DEF, shape=None, calc_mode="non_symbol"):
    calc_mode = confirm_calc_mode(calc_mode)
    x0_var, x0, tau1 = assert_arrays([x0_var, x0, tau1], shape)
    if np.all(calc_mode == "symbol"):
        return np.array(symbol_eqtn_fx0(x0.size, shape)[0](x0_var, x0, tau1))
    else:
        return eqtn_fx0(x0_var, x0, tau1)


def calc_fslope(slope_var, slope, z=0.0, g=0.0, tau1=TAU1_DEF, pmode=np.array("const"), shape=None,
                calc_mode="non_symbol"):
    calc_mode = confirm_calc_mode(calc_mode)
    slope_var, slope, tau1 = assert_arrays([slope_var, slope, tau1], shape)
    if np.all(calc_mode == "symbol"):
        if pmode == "z":
            z = assert_arrays([z], slope.shape)
            return np.array(symbol_eqtn_fslope(slope.size, pmode, shape)[0](slope_var, z, tau1))
        elif pmode == "g":
            g = assert_arrays([g], slope.shape)
            return np.array(symbol_eqtn_fslope(slope.size, pmode, shape)[0](slope_var, g, tau1))
        elif pmode == "z*g":
            z = assert_arrays([z], slope.shape)
            g = assert_arrays([g], slope.shape)
            return np.array(symbol_eqtn_fslope(slope.size, pmode, shape)[0](slope_var, z, g, tau1))
        else:
            return np.array(symbol_eqtn_fslope(slope.size, pmode, shape)[0](slope_var, slope, tau1))
    else:
        if pmode == "z" or pmode == "g" or pmode == "z*g":
            z, g = assert_arrays([z, g], slope.shape)
            from tvb_fit.tvb_epilepsy.base.model import EpileptorDPrealistic
            slope = EpileptorDPrealistic.fun_slope_Iext2(z, g, pmode, slope, 0.0)[0]
        return eqtn_fslope(slope_var, slope, tau1)


def calc_fIext1(Iext1_var, Iext1, tau1=TAU1_DEF, tau0=TAU0_DEF, shape=None, calc_mode="non_symbol"):
    calc_mode = confirm_calc_mode(calc_mode)
    Iext1_var, Iext1, tau1, tau0 = assert_arrays([Iext1_var, Iext1, tau1, tau0], shape)
    if np.all(calc_mode == "symbol"):
        return np.array(symbol_eqtn_fIext1(Iext1.size, shape)[0](Iext1_var, Iext1, tau1, tau0))
    else:
        return eqtn_fIext1(Iext1_var, Iext1, tau1, tau0)


def calc_fIext2(Iext2_var, Iext2, z=0.0, g=0.0, tau1=TAU1_DEF, pmode=np.array("const"), shape=None,
                calc_mode="non_symbol"):
    calc_mode = confirm_calc_mode(calc_mode)
    Iext2_var, Iext2, tau1 = assert_arrays([Iext2_var, Iext2, tau1], shape)
    if np.all(calc_mode == "symbol"):
        if pmode == "z":
            z = assert_arrays([z], Iext2.shape)
            return np.array(symbol_eqtn_fIext2(Iext2.size, pmode, shape)[0](Iext2_var, z, tau1))
        elif pmode == "g":
            g = assert_arrays([g], Iext2.shape)
            return np.array(symbol_eqtn_fIext2(Iext2.size, pmode, shape)[0](Iext2_var, g, tau1))
        elif pmode == "z*g":
            z = assert_arrays([z], Iext2.shape)
            g = assert_arrays([g], Iext2.shape)
            return np.array(symbol_eqtn_fIext2(Iext2.size, pmode, shape)[0](Iext2_var, z, g, tau1))
        else:
            return np.array(symbol_eqtn_fIext2(Iext2.size, pmode, shape)[0](Iext2_var, Iext2, tau1))
    else:
        if pmode == "z" or pmode == "g" or pmode == "z*g":
            z, g = assert_arrays([z, g], Iext2.shape)
            from tvb_fit.tvb_epilepsy.base.model import EpileptorDPrealistic
            Iext2 = EpileptorDPrealistic.fun_slope_Iext2(z, g, pmode, 0.0, Iext2)[1]
        return eqtn_fIext2(Iext2_var, Iext2, tau1)


def calc_fK(K_var, K, tau1=TAU1_DEF, tau0=TAU0_DEF, shape=None, calc_mode="non_symbol"):
    calc_mode = confirm_calc_mode(calc_mode)
    K_var, K, tau1, tau0 = assert_arrays([K_var, K, tau1, tau0], shape)
    if np.all(calc_mode == "symbol"):
        return np.array(symbol_eqtn_fK(K.size, shape)[0](K_var, K, tau1, tau0))
    else:
        return eqtn_fK(K_var, K, tau1, tau0)


def calc_dfun(x1, z, yc, Iext1, x0, K, w, model_vars=2,
              zmode="lin", pmode="z", x1_neg=True, z_pos=True, x2_neg=None,
              y1=None, x2=None, y2=None, g=None,
              x0_var=None, slope_var=None, Iext1_var=None, Iext2_var=None, K_var=None,
              slope=SLOPE_DEF, a=A_DEF, b=B_DEF, d=D_DEF, s=S_DEF, Iext2=I_EXT2_DEF, gamma=GAMMA_DEF,
              tau1=TAU1_DEF, tau0=TAU0_DEF, tau2=TAU2_DEF, shape=None, output_mode="array", calc_mode="non_symbol"):
    calc_mode = confirm_calc_mode(calc_mode)
    if model_vars > 2:
        if np.any(x2_neg is None):
            try:
                x2_neg = x2 < -0.25
            except:
                x2_neg = False
                logger.warning("\nx2_neg is None and failed to compare x2_neg = x2 < -0.25!" +
                               "\nSetting default x2_neg = False")
    if output_mode == "array":
        return calc_dfun_array(x1, z, yc, Iext1, x0, K, w, model_vars, zmode, pmode, x1_neg, z_pos, x2_neg,
                               y1, x2, y2, g, x0_var, slope_var, Iext1_var, Iext2_var, K_var,
                               slope, a, b, d, s, Iext2, gamma, tau1, tau0, tau2)
    else:
        if np.all(calc_mode == "symbol"):
            dfun_sym = symbol_eqnt_dfun(x1.size, model_vars, zmode, x1_neg, z_pos, x2_neg, pmode, shape)[0]
            x1, z, yc, Iext1, x0, K, slope, a, b, tau1, tau0 = \
                assert_arrays([x1, z, yc, Iext1, x0, K, slope, a, b, tau1, tau0], shape)
            w = assert_arrays([w], (z.size, z.size))
            if model_vars == 6:
                y1, x2, y2, g, Iext2, s, gamma, tau2 = \
                    assert_arrays([y1, x2, y2, g, Iext2, s, gamma, tau2], z.shape)
                return dfun_sym[0](x1, z, y1, x2, Iext1, slope, a, b, d, tau1), \
                       dfun_sym[1](x1, y1, yc, d, tau1), \
                       dfun_sym[2](x1, z, x0, K, w, tau1, tau0), \
                       dfun_sym[3](x2, y2, z, g, Iext2, tau1), \
                       dfun_sym[4](x2, y2, s, tau1, tau2), \
                       dfun_sym[5](x1, g, gamma, tau1)
            elif model_vars == 11:
                y1, x2, y2, g, x0_var, slope_var, Iext1_var, Iext2_var, K_var, Iext2, s, gamma, tau2 = \
                    assert_arrays([y1, x2, y2, g, x0_var, slope_var, Iext1_var, Iext2_var, K_var, Iext2, s, gamma,
                                   tau2], z.shape)
                dfun = [dfun_sym[0](x1, z, y1, x2, Iext1_var, slope_var, a, b, d, tau1),
                        dfun_sym[1](x1, y1, yc, d, tau1),
                        dfun_sym[2](x1, z, x0_var, K_var, w, tau1, tau0),
                        dfun_sym[3](x2, y2, z, g, Iext2_var, tau1),
                        dfun_sym[4](x2, y2, s, tau1, tau2),
                        dfun_sym[5](x1, g, gamma, tau1),
                        dfun_sym[6](x0_var, x0, tau1)]
                if pmode == "z":
                    dfun7 = dfun_sym[7](slope_var, z, tau1)
                    dfun9 = dfun_sym[9](Iext2_var, z, tau1)
                elif pmode == "g":
                    dfun7 = dfun_sym[7](slope_var, g, tau1)
                    dfun9 = dfun_sym[9](Iext2_var, g, tau1)
                elif pmode == "z*g":
                    dfun7 = dfun_sym[7](slope_var, z, g, tau1)
                    dfun9 = dfun_sym[9](Iext2_var, z, g, tau1)
                else:
                    dfun7 = dfun_sym[7](slope_var, slope, tau1)
                    dfun9 = dfun_sym[9](Iext2_var, Iext2, tau1)
                dfun.append(dfun7)
                dfun.append(dfun_sym[8](Iext1_var, Iext1, tau1, tau0))
                dfun.append(dfun9)
                dfun.append(dfun_sym[10](K_var, K, tau1, tau0))
                return tuple(dfun)
        else:
            if x1_neg is None:
                x1_neg = x1 < 0.0
            if zmode == np.array("lin") and z_pos is None:
                z_pos = z > 0.0
            x1, z, yc, Iext1, x0, K, slope, a, b, d, tau1, tau0 = \
                assert_arrays([x1, z, yc, Iext1, x0, K, slope, a, b, d, tau1, tau0], shape)
            w = assert_arrays([w], (z.size, z.size))
            y1, x2, y2, g, Iext2, s, gamma, tau2 = assert_arrays([y1, x2, y2, g, Iext2, s, gamma, tau2], z.shape)
            if model_vars == 6:
                return eqtn_fx1(x1, z, yc, Iext1, slope, a, b, d, tau1, x1_neg, model="6d", x2=x2), \
                       eqtn_fy1(x1, yc, y1, d, tau1), \
                       eqtn_fz(x1, z, x0, tau1, tau0, zmode=zmode, z_pos=z_pos, K=K, w=w, coupl=None), \
                       eqtn_fx2(x2, y2, z, g, Iext2, tau1), \
                       eqtn_fy2(x2, y2, s, tau1, tau2, x2_neg), \
                       eqtn_fg(x1, g, gamma, tau1)
            elif model_vars == 11:
                x0_var, slope_var, Iext1_var, Iext2_var, K_var = \
                    assert_arrays([x0_var, slope_var, Iext1_var, Iext2_var, K_var], z.shape)
                dfun = (eqtn_fx1(x1, z, yc, Iext1_var, slope_var, a, b, d, tau1, x1_neg, model="6d", x2=x2),
                        eqtn_fy1(x1, yc, y1, d, tau1),
                        eqtn_fz(x1, z, x0_var, tau1, tau0, zmode=zmode, z_pos=z_pos, K=K_var, w=w, coupl=None),
                        eqtn_fx2(x2, y2, z, g, Iext2_var, tau1),
                        eqtn_fy2(x2, y2, s, tau1, tau2, x2_neg),
                        eqtn_fg(x1, g, gamma, tau1),
                        eqtn_fx0(x0_var, x0, tau1))
                if pmode == "z" or pmode == "g" or pmode == "z*g":
                    from tvb_fit.tvb_epilepsy.base.model import EpileptorDPrealistic
                    slope, Iext2 = EpileptorDPrealistic.fun_slope_Iext2(z, g, pmode, slope, Iext2)[1]
                dfun += (eqtn_fslope(slope_var, slope, tau1),
                         eqtn_fIext1(Iext1_var, Iext1, tau1, tau0),
                         eqtn_fIext2(Iext2_var, Iext2, tau1),
                         eqtn_fK(K_var, K, tau1, tau0))
                return dfun


def calc_jac(x1, z, yc, Iext1, x0, K, w, model_vars=2,
             zmode="lin", pmode="z", x1_neg=True, z_pos=True, x2_neg=None,
             y1=None, x2=None, y2=None, g=None,
             x0_var=None, slope_var=None, Iext1_var=None, Iext2_var=None, K_var=None,
             slope=SLOPE_DEF, a=A_DEF, b=B_DEF, d=D_DEF, s=S_DEF, Iext2=I_EXT2_DEF, gamma=GAMMA_DEF,
             tau1=TAU1_DEF, tau0=TAU0_DEF, tau2=TAU2_DEF, calc_mode="non_symbol"):
    calc_mode = confirm_calc_mode(calc_mode)
    if model_vars > 2:
        if np.any(x2_neg is None):
            try:
                x2_neg = x2 < -0.25
            except:
                x2_neg = False
                logger.warning("\nx2_neg is None and failed to compare x2_neg = x2 < -0.25!" +
                               "\nSetting default x2_neg = False")
    n_regions = max(shape_to_size(x1.shape), shape_to_size(z.shape))
    x1, z, yc, Iext1, x0, K, slope, a, b, d, tau1, tau0 = \
        assert_arrays([x1, z, yc, Iext1, x0, K, slope, a, b, d, tau1, tau0], z.shape)
    w = assert_arrays([w], (z.size, z.size))
    if np.all(calc_mode == "symbol"):
        n = model_vars * n_regions
        jac = np.zeros((n, n), dtype=z.dtype)
        ind = lambda x: x * n_regions + np.array(range(n_regions))
        jac_lambda, jac_sym = symbol_calc_jac(n_regions, model_vars, zmode, x1_neg, z_pos, x2_neg, pmode)[:2]
        y1, x2, y2, g, Iext2, s, gamma, tau2 = \
            assert_arrays([y1, x2, y2, g, Iext2, s, gamma, tau2], z.shape)
        if model_vars == 6:
            jac[ind(0), :] = np.array(jac_lambda[0](x1, y1, z, x2, y2, g, Iext1, slope, a, b, d, tau1))
            jac[ind(1), :] = np.array(jac_lambda[1](x1, y1, z, x2, y2, g, yc, d, tau1))
            jac[ind(2), :] = np.array(jac_lambda[2](x1, y1, z, x2, y2, g, x0, K, w, tau1, tau0))
            jac[ind(3), :] = np.array(jac_lambda[3](x1, y1, z, x2, y2, g, Iext2, tau1))
            jac[ind(4), :] = np.array(jac_lambda[4](x1, y1, z, x2, y2, g, s, tau1, tau2))
            jac[ind(5), :] = np.array(jac_lambda[5](x1, y1, z, x2, y2, g, gamma, tau1))
        elif model_vars == 11:
            x0_var, slope_var, Iext1_var, Iext2_var, K_var = \
                assert_arrays([x0_var, slope_var, Iext1_var, Iext2_var, K_var], z.shape)
            jac[ind(0), :] = np.array(jac_lambda[0](x1, y1, z, x2, y2, g,
                                                    x0_var, slope_var, Iext1_var, Iext2_var, K_var,
                                                    a, b, d, tau1))
            jac[ind(1), :] = np.array(jac_lambda[1](x1, y1, z, x2, y2, g,
                                                    x0_var, slope_var, Iext1_var, Iext2_var, K_var,
                                                    yc, d, tau1))
            jac[ind(2), :] = np.array(jac_lambda[2](x1, y1, z, x2, y2, g,
                                                    x0_var, slope_var, Iext1_var, Iext2_var, K_var,
                                                    w, tau1, tau0))
            jac[ind(3), :] = np.array(jac_lambda[3](x1, y1, z, x2, y2, g,
                                                    x0_var, slope_var, Iext1_var, Iext2_var, K_var,
                                                    tau1))
            jac[ind(4), :] = np.array(jac_lambda[4](x1, y1, z, x2, y2, g,
                                                    x0_var, slope_var, Iext1_var, Iext2_var, K_var,
                                                    s, tau1, tau2))
            jac[ind(5), :] = np.array(jac_lambda[5](x1, y1, z, x2, y2, g,
                                                    x0_var, slope_var, Iext1_var, Iext2_var, K_var,
                                                    gamma, tau1))
            jac[ind(6), :] = np.array(jac_lambda[6](x1, y1, z, x2, y2, g,
                                                    x0_var, slope_var, Iext1_var, Iext2_var, K_var,
                                                    x0, tau1))
            jac[ind(7), :] = np.array(jac_lambda[7](x1, y1, z, x2, y2, g,
                                                    x0_var, slope_var, Iext1_var, Iext2_var, K_var,
                                                    slope, tau1))
            jac[ind(8), :] = np.array(jac_lambda[8](x1, y1, z, x2, y2, g,
                                                    x0_var, slope_var, Iext1_var, Iext2_var, K_var,
                                                    Iext1, tau1, tau0))
            jac[ind(9), :] = np.array(jac_lambda[9](x1, y1, z, x2, y2, g,
                                                    x0_var, slope_var, Iext1_var, Iext2_var, K_var,
                                                    Iext2, tau1))
            jac[ind(10), :] = np.array(jac_lambda[10](x1, y1, z, x2, y2, g,
                                                      x0_var, slope_var, Iext1_var, Iext2_var, K_var,
                                                      K, tau1, tau0))
        return jac
    else:
        if model_vars == 2:
            if x1_neg is None:
                x1_neg = x1 < 0.0
            if z_pos is None:
                z_pos = z > 0.0
            return np.concatenate([eqtn_jac_x1_2d(x1, z, slope, a, b, d, tau1, x1_neg),
                                   eqtn_jac_fz_2d(x1, z, tau1, tau0, zmode, z_pos, K, w)])
        else:
            if model_vars == 6:
                sx1, sy1, sz, sx2, sy2, sg = symbol_vars(n_regions, ['x1', 'y1', 'z', 'x2', 'y2', 'g'])[:6]
                dfun_sym = calc_dfun_array(sx1, sz, yc, Iext1, x0, K, w, model_vars,
                                           zmode, pmode, x1_neg, z_pos, x2_neg,
                                           sy1, sx2, sy2, sg,
                                           x0_var, slope_var, Iext1_var, Iext2_var, K_var,
                                           slope, a, b, d, s, Iext2, gamma, tau1, tau0, tau2)
                x = Matrix([sx1, sy1, sz, sx2, sy2, sg]).reshape(6 * n_regions, 1)
                jac_sym = Matrix(dfun_sym.flatten()).jacobian(x)
                jac_lambda = lambdify([x], jac_sym, "numpy")
                return np.array(jac_lambda([x1, y1, z, x2, y2, g])).astype(x1.dtype)
            elif model_vars == 11:
                sx1, sy1, sz, sx2, sy2, sg, sx0_var, sslope_var, sIext1_var, sIext2_var, sK_var = \
                    symbol_vars(n_regions, ['x1', 'y1', 'z', 'x2', 'y2', 'g',
                                            'x0_var', 'slope_var', 'Iext1_var', 'Iext2_var', 'K_var'])[:11]
                dfun_sym = calc_dfun_array(sx1, sz, yc, Iext1, x0, K, w, model_vars,
                                           zmode, pmode, x1_neg, z_pos, x2_neg,
                                           sy1, sx2, sy2, sg,
                                           sx0_var, sslope_var, sIext1_var, sIext2_var, sK_var,
                                           slope, a, b, d, s, Iext2, gamma, tau1, tau0, tau2)
                x = Matrix([sx1, sy1, sz, sx2, sy2, sg, sx0_var, sslope_var, sIext1_var, sIext2_var, sK_var]) \
                    .reshape(11 * n_regions, 1)
                jac_sym = Matrix(dfun_sym.flatten()).jacobian(x)
                jac_lambda = lambdify([x], jac_sym, "numpy")
                return np.array(jac_lambda([x1, y1, z, x2, y2, g, x0_var, slope_var, Iext1_var, Iext2_var, K_var])) \
                    .astype(x1.dtype)


def calc_coupling_diff(K, w, ix=None, jx=None, calc_mode="non_symbol"):
    calc_mode = confirm_calc_mode(calc_mode)
    K = assert_arrays([K])
    n_regions = K.size
    w = assert_arrays([w], (K.size, K.size))
    if ix is None:
        ix = range(n_regions)
    if jx is None:
        jx = range(n_regions)
    if np.all(calc_mode == "symbol"):
        return np.array(symbol_calc_coupling_diff(K.size, ix, jx, K="K")[0](K, w))
    else:
        return eqtn_coupling_diff(K, w, ix, jx)


def calc_fx1_2d_taylor(x1, x_taylor, z=0, y1=0.0, Iext1=I_EXT1_DEF, slope=SLOPE_DEF, a=A_DEF, b=B_DEF, d=D_DEF,
                       tau1=TAU1_DEF, x1_neg=True, order=2, shape=None, calc_mode="non_symbol"):
    calc_mode = confirm_calc_mode(calc_mode)
    x1, x_taylor, z, y1, Iext1, slope, a, b, d, tau1 = \
        assert_arrays([x1, x_taylor, z, y1, Iext1, slope, a, b, d, tau1], shape)
    if np.all(calc_mode == "symbol"):
        return np.array(symbol_calc_2d_taylor(x1.size, order=order, x1_neg=x1_neg, slope="slope", Iext1="Iext1",
                                              shape=shape)[0](x1, x_taylor, z, y1, Iext1, slope, a, b, d, tau1))
    else:
        if x1_neg is None:
            x1_neg = x1 < 0.0
        if order == 2 and np.all(x1_neg == True):
            # Correspondance with EpileptorDP2D
            b = b - d
            fx1lin = np.multiply(Iext1 + 2 * np.multiply(np.power(x_taylor, 3), a)
                                 - np.multiply(np.power(x_taylor, 2), b) + y1 - z +
                                 np.multiply(x1, (-3 * np.multiply(np.power(x_taylor, 2), a)
                                                  + 2 * np.multiply(x_taylor, b))), tau1)
        else:
            try:
                from sympy import Symbol, series
            except:
                raise_import_error("Unable to load symbolic_equations module. Taylor expansion calculation is not  \
                                    supported non-symbolically for order of expansion >2 and/or when any x1 > 0.")
            x = Symbol("x")
            fx1lin = calc_fx1(x1, z, y1, Iext1, slope, a, b, d, tau1, x1_neg).flatten()
            for ix in range(x1.size):
                fx1lin[ix] = series(fx1lin[ix], x=x, x0=x_taylor, n=order).removeO().simplify(). \
                    subs(x, x1.flatten()[ix])
            fx1lin = reshape(fx1lin, shape).astype(x1.dtype)
        return fx1lin


def calc_fx1z(x1, x0, K, w, yc, Iext1, a=A_DEF, b=B_DEF, d=D_DEF, tau1=TAU1_DEF, tau0=TAU0_DEF,
              model="6d", zmode=np.array("lin"), shape=None, calc_mode="non_symbol"):
    # TODO: for the extreme x1_neg = False case where we have to solve for x2 as well
    # slope=SLOPE_DEF, x2=0.0, z_pos=True, x1_neg=True,
    calc_mode = confirm_calc_mode(calc_mode)
    x1, x0, K, yc, Iext1, a, b, d, tau1, tau0 = assert_arrays([x1, x0, K, yc, Iext1, a, b, d, tau1, tau0], shape)
    w = assert_arrays([w], (x1.size, x1.size))
    if np.all(calc_mode == "symbol"):
        if model == "2d":
            return np.array(symbol_eqtn_fx1z(x1.size, model, zmode, x1.shape)[0](x1, x0, K, w, yc, Iext1, a, b,
                                                                                 d, tau1, tau0))
        else:
            return np.array(symbol_eqtn_fx1z(x1.size, model, zmode, x1.shape)[0](x1, x0, K, w, yc, Iext1, a, b, d, tau1,
                                                                                 tau0))
    else:
        # TODO: for the extreme z_pos = False case where we have terms like 0.1 * z ** 7
        if np.all(model == "2d"):
            z = calc_fx1(x1, 0.0, yc, Iext1, 0.0, a, b, tau1=TAU1_DEF, x2=0.0, model=model, x1_neg=True, shape=x1.shape)
            return eqtn_fz(x1, z, x0, tau1, tau0, zmode, z_pos=True, K=K, w=w, coupl=None)
        else:
            y1 = calc_fy1(x1, yc, 0.0, d, tau1=TAU1_DEF, shape=x1.shape)
            z = calc_fx1(x1, 0.0, y1, Iext1, 0.0, a, b, tau1=TAU1_DEF, x2=0.0, model=model, x1_neg=True, shape=x1.shape)
            return eqtn_fz(x1, z, x0, tau1, tau0, zmode, z_pos=True, K=K, w=w, coupl=None)


def calc_fx1z_diff(x1, K, w, a=A_DEF, b=B_DEF, d=D_DEF, tau1=TAU1_DEF, tau0=TAU0_DEF, model="6d", zmode=np.array("lin"),
                   calc_mode="non_symbol"):  # , yc=0.0, Iext1=I_EXT1_DEF, z_pos=True, slope=SLOPE_DEF, x2=0.0, x1_neg=True,
    # TODO: for the extreme x1_neg = False case where we have to solve for x2 as well
    calc_mode = confirm_calc_mode(calc_mode)
    x1, K, a, b, d, tau1, tau0 = assert_arrays([x1, K, a, b, d, tau1, tau0])
    w = assert_arrays([w], (x1.size, x1.size))
    if np.all(calc_mode == "symbol"):
        return np.array(symbol_eqtn_fx1z_diff(x1.size, model, zmode)[0](x1, K, w, a, b, d, tau1, tau0))
    else:
        # TODO: for the extreme z_pos = False case where we have terms like 0.1 * z ** 7
        ix = range(x1.size)
        return eqtn_fx1z_diff(x1, K, w, ix, ix, a, b, d, tau1, tau0, zmode)


def calc_fx1z_2d_x1neg_zpos_jac(x1, z, x0, yc, Iext1, K, w, ix0, iE, a=A_DEF, b=B_DEF, d=D_DEF, tau1=TAU1_DEF,
                                tau0=TAU0_DEF, calc_mode="non_symbol"):
    calc_mode = confirm_calc_mode(calc_mode)
    if np.all(calc_mode == "symbol"):
        x1, z, x0, yc, Iext1, K, a, b, d, tau1, tau0 = \
            assert_arrays([x1, z, x0, yc, Iext1, K, a, b, d, tau1, tau0])
        w = assert_arrays([w], (x1.size, x1.size))
        return np.array(symbol_calc_fx1z_2d_x1neg_zpos_jac(x1.size, ix0, iE)[0](x1, z, x0, yc, Iext1, K, w, a, b, d,
                                                                                tau1, tau0))
    else:
        if x1.shape != (1, x1.size):
            x1 = np.expand_dims(x1.flatten(), 1).T
        x1, z, x0, yc, Iext1, K, a, b, d, tau1, tau0 = \
            assert_arrays([x1, z, x0, yc, Iext1, K, a, b, d, tau1, tau0], x1.shape)
        # Correspondance with EpileptorDP2D
        b = b - d
        w = assert_arrays([w], (x1.size, x1.size))
        tau = np.divide(tau1, tau0)
        no_x0 = len(ix0)
        no_e = len(iE)
        i_x0 = np.ones((no_x0, 1))
        i_e = np.ones((no_e, 1))
        jac_e_x0e = np.diag(np.multiply(tau[:, iE], - 4.0).flatten())
        jac_e_x1o = -np.dot(np.dot(i_e, np.multiply(tau[:, iE], K[:, iE])), w[iE][:, ix0])
        jac_x0_x0e = np.zeros((no_x0, no_e))
        jac_x0_x1o = (np.diag(np.multiply(tau[:, ix0],
                                          (4 + 3 * np.multiply(a[:, ix0], np.power(x1[:, ix0], 2))
                                           - 2 * np.multiply(b[:, ix0], x1[:, ix0]) +
                                           np.multiply(K[:, ix0], np.sum(w[ix0], axis=1)))).flatten()) -
                      np.multiply(np.dot(i_x0, np.multiply(tau[:, ix0], K[:, ix0])).T, w[ix0][:, ix0]))
        jac = np.empty((x1.size, x1.size), dtype=jac_x0_x1o.dtype)
        jac[np.ix_(iE, iE)] = jac_e_x0e
        jac[np.ix_(iE, ix0)] = jac_e_x1o
        jac[np.ix_(ix0, iE)] = jac_x0_x0e
        jac[np.ix_(ix0, ix0)] = jac_x0_x1o
        return jac


def calc_fx1y1_6d_diff_x1(x1, yc, Iext1, a=A_DEF, b=B_DEF, d=D_DEF, tau1=TAU1_DEF, shape=None, calc_mode="non_symbol"):
    calc_mode = confirm_calc_mode(calc_mode)
    x1, yc, Iext1, a, b, d, tau1 = assert_arrays([x1, yc, Iext1, a, b, d, tau1], shape)
    if np.all(calc_mode == "symbol"):
        return np.array(symbol_calc_fx1y1_6d_diff_x1(x1.size, shape)[0](x1, yc, Iext1, a, b, d, tau1))
    else:
        # Correspondance with EpileptorDP2D
        b = b - d
        return np.multiply(np.multiply(-3 * np.multiply(x1, a) + 2 * b, x1), tau1)


def calc_x0cr_r(yc, Iext1, a=A_DEF, b=B_DEF, d=D_DEF, zmode=np.array("lin"), x1_rest=X1_DEF, x1_cr=X1EQ_CR_DEF,
                x0def=X0_DEF, x0cr_def=X0_CR_DEF, test=False, shape=None,
                calc_mode="non_symbol"):  # epileptor_model="2d",
    calc_mode = confirm_calc_mode(calc_mode)
    yc, Iext1, a, b, d, x1_rest, x1_cr, x0def, x0cr_def \
        = assert_arrays([yc, Iext1, a, b, d, x1_rest, x1_cr, x0def, x0cr_def], shape)
    if np.all(calc_mode == "symbol"):
        if test:
            x0cr, r = symbol_calc_x0cr_r(Iext1.size, zmode, Iext1.shape)[0]
        else:
            x0cr, r = symbol_eqtn_x0cr_r(Iext1.size, zmode, Iext1.shape)[0]
        # Calculate x0cr from the lambda function
        x0cr = np.array(x0cr(yc, Iext1, a, b, d, x1_rest, x1_cr, x0def, x0cr_def))
        # r is already given as independent of yc and Iext1
        r = np.array(r(yc, Iext1, a, b, d, x1_rest, x1_cr, x0def, x0cr_def))
    else:
        if test:
            if np.all(Iext1 == Iext1[0]) and np.all(yc == yc[0]) \
                    and np.all(a == a[0]) and np.all(b == b[0] and np.all(d == d[0])):
                Iext1 = Iext1[0]
                yc = yc[0]
                a = a[0]
                b = b[0]
                d = d[0]
            p2 = Iext1.shape
            x1_rest, x1_cr = assert_arrays([x1_rest, x1_cr], p2)
            # Define the z equilibrium expression...
            # if epileptor_model == "2d":
            zeq_rest = calc_fx1(x1_rest, z=0.0, y1=yc, Iext1=Iext1, a=a, b=b, d=d, tau1=1.0, x2=0.0, model="2d",
                                x1_neg=True)
            zeq_cr = calc_fx1(x1_cr, z=0.0, y1=yc, Iext1=Iext1, a=a, b=b, d=d, tau1=1.0, x2=0.0, model="2d",
                              x1_neg=True)
            if zmode == np.array("lin"):
                xinit = np.array([2.460, 0.398])
            else:
                xinit = np.array([3.174, 0.260])
            # else:
            #     zeq_rest = calc_fx1(x1_rest, z=0.0, y1=calc_fy1(x1_rest, yc), Iext1=Iext1, x1_neg=True)
            #     zeq_cr = calc_fx1(x1_cr, z=0.0, y1=calc_fy1(x1_cr, yc), Iext1=Iext1, x1_neg=True)
            #     if zmode == np.array("lin"):
            #         xinit = np.array([5.9320, 1.648])
            #     else:
            #         xinit = np.array([17.063, 5.260])
            # Define the fz expression...
            x0cr = []
            r = []
            for ii in range(Iext1.size):
                # x0 -> r * x0 - x0cr
                fz = lambda x: np.array([calc_fz(x1_rest[ii], z=zeq_rest[ii], x0=x[1] * x0def - x[0], K=0.0, w=0.0,
                                                 tau1=1.0, tau0=1.0, zmode=zmode, z_pos=True, shape=None),
                                         calc_fz(x1_cr[ii], z=zeq_cr[ii], x0=x[1] * x0cr_def - x[0], K=0.0, w=0.0,
                                                 tau1=1.0, tau0=1.0, zmode=zmode, z_pos=True, shape=None)])
                sol = root(fz, xinit, method='lm', tol=10 ** (-12), callback=None, options=None)
                if sol.success:
                    if np.any([np.any(np.isnan(sol.x)), np.any(np.isinf(sol.x))]):
                        raise_value_error("nan or inf values in solution x\n" + sol.message)
                    x0cr.append(sol.x[0])
                    r.append(sol.x[1])
                else:
                    raise_value_error(sol.message)
            if p2 != shape:
                x0cr = np.tile(x0cr[0], shape)
                r = np.tile(r[0], shape)
            else:
                x0cr = reshape(x0cr, shape)
                r = reshape(r, shape)
        else:
            x0cr, r = eqtn_x0cr_r(yc, Iext1, a, b, d, x1_rest, x1_cr, x0def, x0cr_def, zmode=zmode)
    return x0cr, r


def calc_fz_jac_square_taylor(zeq, yc, Iext1, K, w, a=A_DEF, b=B_DEF, d=D_DEF, tau1=TAU1_DEF, tau0=TAU0_DEF,
                              x_taylor=X1EQ_CR_DEF, calc_mode="non_symbol"):
    calc_mode = confirm_calc_mode(calc_mode)
    zeq, yc, Iext1, K, a, b, d, tau1, tau0, x_taylor = \
        assert_arrays([zeq, yc, Iext1, K, a, b, d, tau1, tau0, x_taylor], (1, zeq.size))
    w = assert_arrays([w], (zeq.size, zeq.size))
    if np.all(calc_mode == "symbol"):
        return symbol_calc_fz_jac_square_taylor(zeq.size)[0](zeq, yc, Iext1, K, w, a, b, d, tau1, tau0, x_taylor)
    else:
        return eqtn_fz_square_taylor(zeq, yc, Iext1, K, w, tau1, tau0)


def calc_fpop2(x2, y2=0.0, z=0.0, g=0.0, Iext2=I_EXT2_DEF, s=S_DEF, tau1=TAU1_DEF, tau2=1.0, x2_neg=None, shape=None,
               calc_mode="non_symbol"):
    return calc_fx2(x2, y2, z, g, Iext2, tau1, shape, calc_mode), \
           calc_fy2(x2, y2, s, tau1, tau2, x2_neg, shape, calc_mode)


def calc_fparams_var(x0_var, slope_var, Iext1_var, Iext2_var, K_var, x0, slope, Iext1, Iext2, K, z=0.0, g=0.0,
                     tau1=TAU1_DEF, tau0=1.0, pmode=np.array("const"), shape=None, calc_mode="non_symbol"):
    return calc_fx0(x0_var, x0, tau1, shape, calc_mode), \
           calc_fslope(slope_var, slope, z, g, tau1, pmode, shape, calc_mode), \
           calc_fIext1(Iext1_var, Iext1, tau1, tau0, shape, calc_mode), \
           calc_fIext2(Iext2_var, Iext2, z, g, tau1, pmode, shape, calc_mode), \
           calc_fK(K_var, K, tau1, tau0, shape, calc_mode)


def calc_dfun_array(x1, z, yc, Iext1, x0, K, w, model_vars=2,
                    zmode="lin", pmode="z", x1_neg=None, z_pos=None, x2_neg=None,
                    y1=None, x2=None, y2=None, g=None,
                    x0_var=None, slope_var=None, Iext1_var=None, Iext2_var=None, K_var=None,
                    slope=SLOPE_DEF, a=A_DEF, b=B_DEF, d=D_DEF, s=S_DEF, Iext2=I_EXT2_DEF, gamma=GAMMA_DEF,
                    tau1=TAU1_DEF, tau0=TAU0_DEF, tau2=TAU2_DEF, calc_mode="non_symbol"):
    n_regions = max(shape_to_size(x1.shape), shape_to_size(z.shape))
    shape = (1, n_regions)
    f = np.empty((model_vars, n_regions), dtype=type(x1[0]))
    if model_vars == 2:
        f[0, :] = calc_fx1(x1, z, yc, Iext1, slope, a, b, d, tau1, x2, model="2d", x1_neg=x1_neg, shape=shape,
                           calc_mode=calc_mode)
        f[1, :] = calc_fz(x1, z, x0, K, w, tau1, tau0, zmode, z_pos, shape=shape, calc_mode=calc_mode)
    elif model_vars == 6:
        f[0, :] = calc_fx1(x1, z, y1, Iext1, slope, a, b, d, tau1, x2=x2, model="6d", x1_neg=x1_neg, shape=shape,
                           calc_mode=calc_mode)
        f[1, :] = calc_fy1(x1, yc, y1, d, tau1, shape, calc_mode)
        f[2, :] = calc_fz(x1, z, x0, K, w, tau1, tau0, zmode, z_pos, shape=shape, calc_mode=calc_mode)
        f[3, :] = calc_fx2(x2, y2, z, g, Iext2, tau1, shape, calc_mode)
        f[4, :] = calc_fy2(x2, y2, s, tau1, tau2, x2_neg, shape, calc_mode)
        f[5, :] = calc_fg(x1, g, gamma, tau1, shape, calc_mode)
    elif model_vars == 11:
        f[0, :] = calc_fx1(x1, z, y1, Iext1_var, slope_var, a, b, d, tau1, x2=x2, model="6d", x1_neg=x1_neg,
                           shape=shape, calc_mode=calc_mode)
        f[1, :] = calc_fy1(x1, yc, y1, d, tau1, shape, calc_mode)
        f[2, :] = calc_fz(x1, z, x0_var, K_var, w, tau1, tau0, zmode, z_pos, shape=shape, calc_mode=calc_mode)
        f[3, :] = calc_fx2(x2, y2, z, g, Iext2_var, tau1, shape, calc_mode)
        f[4, :] = calc_fy2(x2, y2, s, tau1, tau2, x2_neg, shape, calc_mode)
        f[5, :] = calc_fg(x1, g, gamma, tau1, shape, calc_mode)
        f[6, :] = calc_fx0(x0_var, x0, tau1, shape, calc_mode)
        f[7, :] = calc_fslope(slope_var, slope, z, g, tau1, pmode, shape, calc_mode)
        f[8, :] = calc_fIext1(Iext1_var, Iext1, tau1, tau0, shape, calc_mode)
        f[9, :] = calc_fIext2(Iext2_var, Iext2, z, g, tau1, pmode, shape, calc_mode)
        f[10, :] = calc_fK(K_var, K, tau1, tau0, shape, calc_mode)
    return f


def calc_x0_val_to_model_x0(x0_values, yc, Iext1, a=A_DEF, b=B_DEF, d=D_DEF, zmode=np.array("lin"), shape=None,
                            calc_mode="non_symbol"):
    hyp_x0, yc, Iext1, a, b, d = \
        assert_arrays([x0_values, yc, Iext1, a, b, d], shape)
    x0cr, r = calc_x0cr_r(yc, Iext1, a, b, d, zmode=zmode, calc_mode=calc_mode)  # epileptor_model="6d",
    return np.multiply(r, x0_values) - x0cr


def calc_model_x0_to_x0_val(x0, yc, Iext1, a=A_DEF, b=B_DEF, d=D_DEF, zmode=np.array("lin"), shape=None,
                            calc_mode="non_symbol"):
    x0, yc, Iext1, a, b, d = assert_arrays([x0, yc, Iext1, a, b, d], shape)
    x0cr, r = calc_x0cr_r(yc, Iext1, a, b, d, zmode=zmode, calc_mode=calc_mode)  # epileptor_model="6d",
    return np.divide(x0 + x0cr, r)
