import warnings
import numpy
from numpy import empty, array, ones, zeros, multiply, dot, power, divide, sum, exp, reshape, diag, expand_dims, where
from tvb_epilepsy.base.constants import X0_DEF, X0_CR_DEF, X1_DEF, X1_EQ_CR_DEF, SYMBOLIC_CALCULATIONS_FLAG
from tvb_epilepsy.base.utils import assert_arrays, shape_to_size
from tvb_epilepsy.base.equations import *
from tvb_epilepsy.base.symbolic import *

# TODO: find out why I cannot import anything from utils here
# from tvb_epilepsy.base.utils import assert_array_shape as sc2arr

if SYMBOLIC_CALCULATIONS_FLAG == "symbolic":

    try:
        from tvb_epilepsy.base.symbolic import *

    except:
        warnings.warn("Unable to load symbolic_equations module. Turning to non symbolic ones.")
        SYMBOLIC_CALCULATIONS_FLAG = False
        from scipy.optimize import root
else:

    from scipy.optimize import root


if SYMBOLIC_CALCULATIONS_FLAG:

    # Symbolic calculations are only used for testing and demonstration of equations, as well as for a few difficult
    # calculations, such as finding roots of calculating jacobians

    def calc_coupling(x1, K, w, ix=None, jx=None, shape=None):

        x1, K = assert_arrays([x1, K], shape)
        n_regions = x1.size

        w = assert_arrays([w], (x1.size, x1.size))

        if ix is None:
            ix = range(n_regions)

        if jx is None:
            jx = range(n_regions)

        return array(symbol_eqtn_coupling(x1.size, ix, jx, shape=x1.shape)[0](x1, K, w))


    def calc_x0(x1, z, K=0.0, w=0.0, x0cr=0.0, r=1.0, model="2d", zmode=array("lin"), z_pos=True, shape=None):

        x1, z,  K = assert_arrays([x1, z,  K], shape)
        w = assert_arrays([w], (z.size, z.size))

        if model == "2d":

            x0cr, r = assert_arrays([x0cr, r], z.shape)

            return array(symbol_eqtn_x0(z.size, zmode, z_pos, model, "K", z.shape)[0](x1, z, x0cr, r, K, w))

        else:

            return array(symbol_eqtn_x0(z.size, zmode, z_pos, model, "K", z.shape)[0](x1, z, K, w))


    def calc_fx1(x1=0.0, z=0.0, y1=0.0, Iext1=0.0, slope=0.0, a=1.0, b=-2.0, tau1=1.0, x2=0.0, model="2d", x1_neg=True,
                 shape=None):

        x1, z, y1, Iext1, slope, a, b, tau1 = assert_arrays([x1, z, y1, Iext1, slope, a, b, tau1], shape)

        if model == "2d":

            return array(symbol_eqtn_fx1(x1.size, model, x1_neg, slope="slope", Iext1="Iext1", shape=x1.shape)[0]
                           (x1, z, y1, Iext1, slope, a, b, tau1))
        else:

            x2 = assert_arrays([x2], x1.shape)

            return array(symbol_eqtn_fx1(x1.size, model, x1_neg, slope="slope", Iext1="Iext1", shape=x1.shape)[0]
                           (x1, z, y1, x2, Iext1, slope, a, b, tau1))


    def calc_fy1(x1, yc, y1=0, d=5.0, tau1=1.0, shape=None):

        x1, yc, y1, d, tau1 = assert_arrays([x1, yc, y1, d, tau1], shape)

        return array(symbol_eqtn_fy1(x1.size, x1.shape)[0](x1, y1, yc, d, tau1))


    def calc_fz(x1=0.0, z=0, x0=0.0, K=0.0, w=0.0, tau1=1.0, tau0=1.0, x0cr=0.0, r=1.0, zmode=array("lin"), z_pos=True,
                   model="2d", shape=None):

        x1, z,  x0, K, tau1, tau0 = assert_arrays([x1, z,  x0, K, tau1, tau0], shape)
        w = assert_arrays([w], (z.size, z.size))

        if model == "2d":

            x0cr, r = assert_arrays([x0cr, r], z.shape)

            return array(symbol_eqtn_fz(z.size, zmode, z_pos, model, x0="x0", K="K", shape=z.shape)[0](x1, z, x0,
                                                                                                         x0cr, r, K, w,
                                                                                                         tau1, tau0))

        else:

            return array(symbol_eqtn_fz(z.size, zmode, z_pos, model, x0="x0", K="K", shape=z.shape)[0](x1, z, x0,
                                                                                                         K, w, tau1,
                                                                                                         tau0))


    def calc_fx2(x2, y2=0.0, z=0.0, g=0.0, Iext2=0.45, tau1=1.0, shape=None):

        x2, y2, z, g, Iext2, tau1 = assert_arrays([x2, y2, z, g, Iext2, tau1], shape)

        return array(symbol_eqtn_fx2(x2.size, Iext2="Iext2", shape=x2.shape)[0](x2, y2, z, g, Iext2, tau1))


    def calc_fy2(x2, y2=0.0, s=6.0, tau1=1.0, tau2=1.0, x2_neg=None, shape=None):

        x2, y2, s, tau1, tau2 = assert_arrays([x2, y2, s, tau1, tau2], shape)

        if numpy.any(x2_neg is None):
            try:
                x2_neg = x2 < -0.25
            except:
                x2_neg = False
                warnings.warn("\nx2_neg is None and failed to compare x2_neg = x2 < -0.25!" +
                              "\nSetting default x2_neg = False")

        return array(symbol_eqtn_fy2(x2.size, x2_neg=x2_neg, shape=x2.shape)[0](x2, y2, s, tau1, tau2))


    def calc_fg(x1, g=0.0, gamma=0.01, tau1=1.0, shape=None):

        x1, g, gamma, tau1 = assert_arrays([x1, g, gamma, tau1], shape)

        return array(symbol_eqtn_fg(x1.size, x1.shape)[0](x1, g, gamma, tau1))


    def calc_fx0(x0_var, x0, tau1=1.0, shape=None):

        x0_var, x0, tau1 = assert_arrays([x0_var, x0, tau1], shape)

        return array(symbol_eqtn_fx0(x0.size, shape)[0](x0_var, x0, tau1))


    def calc_fslope(slope_var, slope, z=0.0, g=0.0, tau1=1.0, pmode=array("const"), shape=None):

        slope_var, slope, tau1 = assert_arrays([slope_var, slope, tau1], shape)

        if pmode == "z":
            z = assert_arrays([z], slope.shape)
            return array(symbol_eqtn_fslope(slope.size, pmode, shape)[0](slope_var, z, tau1))
        elif pmode == "g":
            g = assert_arrays([g], slope.shape)
            return array(symbol_eqtn_fslope(slope.size, pmode, shape)[0](slope_var, g, tau1))
        elif pmode == "z*g":
            z = assert_arrays([z], slope.shape)
            g = assert_arrays([g], slope.shape)
            return array(symbol_eqtn_fslope(slope.size, pmode, shape)[0](slope_var, z, g, tau1))
        else:
            return array(symbol_eqtn_fslope(slope.size, pmode, shape)[0](slope_var, slope, tau1))


    def calc_fIext1(Iext1_var, Iext1, tau1=1.0, tau0=1.0, shape=None):

        Iext1_var, Iext1, tau1, tau0 = assert_arrays([Iext1_var, Iext1, tau1, tau0], shape)

        return array(symbol_eqtn_fIext1(Iext1.size, shape)[0](Iext1_var, Iext1, tau1, tau0))


    def calc_fIext2(Iext2_var, Iext2, z=0.0, g=0.0, tau1=1.0, pmode=array("const"), shape=None):

        Iext2_var, Iext2, tau1 = assert_arrays([Iext2_var, Iext2, tau1], shape)

        if pmode == "z":
            z = assert_arrays([z], Iext2.shape)
            return array(symbol_eqtn_fIext2(Iext2.size, pmode, shape)[0](Iext2_var, z, tau1))
        elif pmode == "g":
            g = assert_arrays([g], Iext2.shape)
            return array(symbol_eqtn_fIext2(Iext2.size, pmode, shape)[0](Iext2_var, g, tau1))
        elif pmode == "z*g":
            z = assert_arrays([z], Iext2.shape)
            g = assert_arrays([g], Iext2.shape)
            return array(symbol_eqtn_fIext2(Iext2.size, pmode, shape)[0](Iext2_var, z, g, tau1))
        else:
            return array(symbol_eqtn_fIext2(Iext2.size, pmode, shape)[0](Iext2_var, Iext2, tau1))


    def calc_fK(K_var, K, tau1=1.0, tau0=1.0, shape=None):

        K_var, K, tau1, tau0 = assert_arrays([K_var, K, tau1, tau0], shape)

        return array(symbol_eqtn_fK(K.size, shape)[0](K_var, K, tau1, tau0))


    def calc_dfun(x1, z, yc, Iext1, x0, K, w, model_vars=2, x0cr=None, r=None,
                  zmode="lin", pmode="const", x1_neg=True, z_pos=True, x2_neg=None,
                  y1=None, x2=None, y2=None, g=None,
                  x0_var=None, slope_var=None, Iext1_var=None, Iext2_var=None, K_var=None,
                  slope=0.0, a=1.0, b=-2.0, d=5.0, s=6.0, Iext2=0.45, gamma=0.01,
                  tau1=1.0, tau0=2857.0, tau2=10.0, shape=None, output_mode="array"):

        if model_vars > 2:
            if numpy.any(x2_neg is None):
                try:
                    x2_neg = x2 < -0.25
                except:
                    x2_neg = False
                    warnings.warn("\nx2_neg is None and failed to compare x2_neg = x2 < -0.25!" +
                                  "\nSetting default x2_neg = False")

        if output_mode == "array":

            return calc_dfun_array(x1, z, yc, Iext1, x0, K, w, model_vars, x0cr, r, zmode, pmode, x1_neg, z_pos, x2_neg,
                                   y1, x2, y2, g, x0_var, slope_var, Iext1_var, Iext2_var, K_var,
                                   slope, a, b, d, s, Iext2, gamma, tau1, tau0, tau2)

        else:

            dfun_sym = symbol_eqnt_dfun(x1.size, model_vars, zmode, x1_neg, z_pos, x2_neg, pmode, shape)[0]

            x1, z,  yc, Iext1, x0, K, slope, a, b, tau1, tau0 = \
                assert_arrays([x1, z,  yc, Iext1, x0, K, slope, a, b, tau1, tau0], shape)

            w = assert_arrays([w], (z.size, z.size))

            if model_vars == 2:

                x0cr, r = assert_arrays([x0cr, r], z.shape)

                return dfun_sym[0](x1, z, yc, Iext1, slope, a, b, tau1), \
                       dfun_sym[1](x1, z, x0, x0cr, r, K, w, tau1, tau0)

            elif model_vars == 6:

                y1, x2, y2, g, Iext2, d, s, gamma, tau2 = \
                    assert_arrays([y1, x2, y2, g, Iext2, d, s, gamma, tau2], z.shape)

                return dfun_sym[0](x1, z, y1, x2, Iext1, slope, a, b, tau1), \
                       dfun_sym[1](x1, y1, yc, d, tau1), \
                       dfun_sym[2](x1, z, x0, K, w, tau1, tau0), \
                       dfun_sym[3](x2, y2, z, g, Iext2, tau1), \
                       dfun_sym[4](x2, y2, s, tau1, tau2), \
                       dfun_sym[5](x1, g, gamma, tau1)

            elif model_vars == 11:

                y1, x2, y2, g, x0_var, slope_var, Iext1_var, Iext2_var, K_var, Iext2, d, s, gamma, tau2 = \
                    assert_arrays([y1, x2, y2, g, x0_var, slope_var, Iext1_var, Iext2_var, K_var, Iext2,
                                   d, s, gamma, tau2], z.shape)

                dfun = [dfun_sym[0](x1, z, y1, x2, Iext1_var, slope_var, a, b, tau1),
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


    def calc_jac(x1, z, yc, Iext1, x0, K, w, model_vars=2, x0cr=None, r=None,
                 zmode="lin", pmode="const", x1_neg=True, z_pos=True, x2_neg=None,
                 y1=None, x2=None, y2=None, g=None,
                 x0_var=None, slope_var=None, Iext1_var=None, Iext2_var=None, K_var=None,
                 slope=0.0, a=1.0, b=-2.0, d=5.0, s=6.0, Iext2=0.45, gamma=0.01,
                 tau1=1.0, tau0=2857.0, tau2=10.0):

        if model_vars > 2:
            if numpy.any(x2_neg is None):
                try:
                    x2_neg = x2 < -0.25
                except:
                    x2_neg = False
                    warnings.warn("\nx2_neg is None and failed to compare x2_neg = x2 < -0.25!" +
                                  "\nSetting default x2_neg = False")

        n_regions = max(shape_to_size(x1.shape), shape_to_size(z.shape))

        x1, z,  yc, Iext1, x0, K, slope, a, b, tau1, tau0 = \
            assert_arrays([x1, z,  yc, Iext1, x0, K, slope, a, b, tau1, tau0], z.shape)

        w = assert_arrays([w], (z.size, z.size))

        n = model_vars * n_regions
        jac = zeros((n,n), dtype=z.dtype)

        ind = lambda x: x * n_regions + array(range(n_regions))

        jac_lambda, jac_sym = symbol_calc_jac(n_regions, model_vars, zmode, x1_neg, z_pos, x2_neg, pmode)[:2]

        if model_vars == 2:

            x0cr, r = assert_arrays([x0cr, r], z.shape)

            jac[ind(0), :] = array(jac_lambda[0](x1, z, yc, Iext1, slope, a, b, tau1))
            jac[ind(1), :] = array(jac_lambda[1](x1, z, x0, x0cr, r, K, w, tau1, tau0))

        else:

            y1, x2, y2, g, Iext2, d, s, gamma, tau2 = \
                assert_arrays([y1, x2, y2, g, Iext2, d, s, gamma, tau2], z.shape)

            if model_vars == 6:

                jac[ind(0), :] = array(jac_lambda[0](x1, y1, z, x2, y2, g, Iext1, slope, a, b, tau1))
                jac[ind(1), :] = array(jac_lambda[1](x1, y1, z, x2, y2, g, yc, d, tau1))
                jac[ind(2), :] = array(jac_lambda[2](x1, y1, z, x2, y2, g, x0, K, w, tau1, tau0))
                jac[ind(3), :] = array(jac_lambda[3](x1, y1, z, x2, y2, g, Iext2, tau1))
                jac[ind(4), :] = array(jac_lambda[4](x1, y1, z, x2, y2, g, s, tau1, tau2))
                jac[ind(5), :] = array(jac_lambda[5](x1, y1, z, x2, y2, g, gamma, tau1))


            elif model_vars == 11:

                x0_var, slope_var, Iext1_var, Iext2_var, K_var = \
                    assert_arrays([x0_var, slope_var, Iext1_var, Iext2_var, K_var], z.shape)

                jac[ind(0), :] = array(jac_lambda[0](x1, y1, z, x2, y2, g,
                                                     x0_var, slope_var, Iext1_var, Iext2_var, K_var,
                                                     a, b, tau1))
                jac[ind(1), :] = array(jac_lambda[1](x1, y1, z, x2, y2, g,
                                                     x0_var, slope_var, Iext1_var, Iext2_var, K_var,
                                                     yc, d, tau1))
                jac[ind(2), :] = array(jac_lambda[2](x1, y1, z, x2, y2, g,
                                                     x0_var, slope_var, Iext1_var, Iext2_var, K_var,
                                                     w, tau1, tau0))
                jac[ind(3), :] = array(jac_lambda[3](x1, y1, z, x2, y2, g,
                                                     x0_var, slope_var, Iext1_var, Iext2_var, K_var,
                                                     tau1))
                jac[ind(4), :] = array(jac_lambda[4](x1, y1, z, x2, y2, g,
                                                     x0_var, slope_var, Iext1_var, Iext2_var, K_var,
                                                     s, tau1, tau2))
                jac[ind(5), :] = array(jac_lambda[5](x1, y1, z, x2, y2, g,
                                                     x0_var, slope_var, Iext1_var, Iext2_var, K_var,
                                                     gamma, tau1))
                jac[ind(6), :] = array(jac_lambda[6](x1, y1, z, x2, y2, g,
                                                     x0_var, slope_var, Iext1_var, Iext2_var, K_var,
                                                     x0, tau1))
                jac[ind(7), :] = array(jac_lambda[7](x1, y1, z, x2, y2, g,
                                                     x0_var, slope_var, Iext1_var, Iext2_var, K_var,
                                                     slope, tau1))
                jac[ind(8), :] = array(jac_lambda[8](x1, y1, z, x2, y2, g,
                                                     x0_var, slope_var, Iext1_var, Iext2_var, K_var,
                                                     Iext1, tau1, tau0))
                jac[ind(9), :] = array(jac_lambda[9](x1, y1, z, x2, y2, g,
                                                     x0_var, slope_var, Iext1_var, Iext2_var, K_var,
                                                     Iext2, tau1))
                jac[ind(10), :] = array(jac_lambda[10](x1, y1, z, x2, y2, g,
                                                       x0_var, slope_var, Iext1_var, Iext2_var, K_var,
                                                       K, tau1, tau0))

        return jac


    def calc_coupling_diff(K, w, ix=None, jx=None):

        K = assert_arrays([K])
        n_regions = K.size

        w = assert_arrays([w], (K.size, K.size))

        if ix is None:
            ix = range(n_regions)

        if jx is None:
            jx = range(n_regions)

        return array(symbol_calc_coupling_diff(K.size, ix, jx, K="K")[0](K, w))


    def calc_fx1_2d_taylor(x1, x_taylor, z=0, y1=0.0, Iext1=0.0, slope=0.0, a=1.0, b=-2.0, tau1=1.0, x1_neg=True,
                           order=2, shape=None):

        x1, x_taylor, z, y1, Iext1, slope, a, b, tau1 = \
            assert_arrays([x1, x_taylor, z, y1, Iext1, slope, a, b, tau1], shape)

        return array(symbol_calc_2d_taylor(x1.size, order=order, x1_neg=x1_neg, slope="slope", Iext1="Iext1",
                                           shape=shape)[0](x1, x_taylor, z, y1, Iext1, slope, a, b, tau1))


    def calc_fx1z(x1, x0, K, w, yc, Iext1, x0cr=0.0, r=1.0, a=1.0, b=-2.0, d=5.0, tau1=1.0, tau0=1.0, model="6d",
                  zmode=array("lin"), shape=None): #slope=0.0, x2=0.0, z_pos=True, x1_neg=True,

        # TODO: for the extreme x1_neg = False case where we have to solve for x2 as well

        x1, x0, K, yc, Iext1, a, b, tau1, tau0 = assert_arrays([x1, x0, K, yc, Iext1, a, b, tau1, tau0], shape)

        w = assert_arrays([w], (x1.size, x1.size))

        if model == "2d":
            x0cr, r = assert_arrays([x0cr, r], x1.shape)
            return array(symbol_eqtn_fx1z(x1.size, model, zmode, x1.shape)[0](x1, x0, K, w, x0cr, r, yc, Iext1, a, b,
                                                                              tau1, tau0))
        else:
            d = assert_arrays([d], x1.shape)
            return array(symbol_eqtn_fx1z(x1.size, model, zmode, x1.shape)[0](x1, x0, K, w, yc, Iext1, a, b, d, tau1,
                                                                              tau0))


    def calc_fx1z_diff(x1, K, w, a=1.0, b=-2.0, d=5.0, tau1=1.0, tau0=1.0, model="6d", zmode=array("lin")): # , yc=0.0,
                       #Iext1=0.0, z_pos=True, slope=0.0, x2=0.0, x1_neg=True,

        # TODO: for the extreme x1_neg = False case where we have to solve for x2 as well

        x1, K, a, b, tau1, tau0 = assert_arrays([x1, K, a, b, tau1, tau0])

        w = assert_arrays([w], (x1.size, x1.size))

        if model == "2d":
            return array(symbol_eqtn_fx1z_diff(x1.size, model, zmode)[0](x1, K, w, a, b, tau1, tau0))
        else:
            d = assert_arrays([d], x1.shape)
            return array(symbol_eqtn_fx1z_diff(x1.size, model, zmode)[0](x1, K, w, a, b, d, tau1, tau0))


    def calc_fx1z_2d_x1neg_zpos_jac(x1, z, x0, x0cr, r, yc, Iext1, K, w, ix0, iE, a=1.0, b=-2.0, tau1=1.0, tau0=1.0):

        x1, z, x0, x0cr, r, yc, Iext1, K, a, b, tau1, tau0 = \
            assert_arrays([x1, z, x0, x0cr, r, yc, Iext1, K, a, b, tau1, tau0])

        w = assert_arrays([w], (x1.size, x1.size))

        return array(symbol_calc_fx1z_2d_x1neg_zpos_jac(x1.size, ix0, iE)[0](x1, z, x0, x0cr, r, yc, Iext1, K, w, a, b,
                                                                             tau1, tau0))


    def calc_fx1y1_6d_diff_x1(x1, yc, Iext1, a=1.0, b=3.0, d=5.0, tau1=1.0, shape=None):

        x1, yc, Iext1, a, b, d, tau1 = assert_arrays([ x1, yc, Iext1, a, b, d, tau1], shape)

        return array(symbol_calc_fx1y1_6d_diff_x1(x1.size, x1.shape)[0](x1, yc, Iext1, a, b, d, tau1))


    def calc_x0cr_r(yc, Iext1, a=1.0, b=-2.0, zmode=array("lin"), x1_rest=X1_DEF, x1_cr=X1_EQ_CR_DEF, x0def=X0_DEF,
                    x0cr_def=X0_CR_DEF, test=False, shape=None):  #epileptor_model="2d",

        yc, Iext1, a, b, x1_rest, x1_cr, x0def, x0cr_def \
            = assert_arrays([yc, Iext1, a, b, x1_rest, x1_cr, x0def, x0cr_def], shape)

        if test:
            x0cr, r = symbol_calc_x0cr_r(Iext1.size, zmode, Iext1.shape)[0]
        else:
            x0cr, r = symbol_eqtn_x0cr_r(Iext1.size, zmode, Iext1.shape)[0]

        # Calculate x0cr from the lambda function
        x0cr = array(x0cr(yc, Iext1, a, b, x1_rest, x1_cr, x0def, x0cr_def))

        # r is already given as independent of yc and Iext1
        r = array(r(yc, Iext1, a, b, x1_rest, x1_cr, x0def, x0cr_def))

        return x0cr, r

else:

    def calc_coupling(x1, K, w, ix=None, jx=None, shape=None):

        x1, K = assert_arrays([x1, K], shape)

        w = assert_arrays([w], (x1.size, x1.size))

        n_regions = x1.size

        if ix is None:
            ix = range(n_regions)

        if jx is None:
            jx = range(n_regions)

        return eqtn_coupling(x1, K, w, ix, jx)


    def calc_x0(x1, z, K=0.0, w=0.0, x0cr=0.0, r=1.0, model="2d", zmode=array("lin"), z_pos=None, shape=None):

        x1, z,  K = assert_arrays([x1, z,  K], shape)
        w = assert_arrays([w], (z.size, z.size))

        if zmode == array("lin") and z_pos is None:
            z_pos = z > 0.0

        if model == "2d":

            x0cr, r = assert_arrays([x0cr, r], z.shape)

            return eqtn_x0(x1, z, model, zmode, z_pos, K, w, coupl=None, x0cr=x0cr, r=r)

        else:

            return eqtn_x0(x1, z, model, zmode, z_pos, K, w)


    def calc_fx1(x1=0.0, z=0.0, y1=0.0, Iext1=0.0, slope=0.0, a=1.0, b=-2.0, tau1=1.0, x2=0.0, model="2d", x1_neg=None,
                 shape=None):

        if x1_neg is None:
            x1_neg = x1 < 0.0

        x1, z, y1, Iext1, slope, a, b, tau1 = assert_arrays([x1, z, y1, Iext1, slope, a, b, tau1], shape)

        if model == "2d":

            return eqtn_fx1(x1, z, y1, Iext1, slope, a, b, tau1, x1_neg, model, x2=None)

        else:

            x2 = assert_arrays([x2], x1.shape)

            return eqtn_fx1(x1, z, y1, Iext1, slope, a, b, tau1, x1_neg, model, x2)


    def calc_fy1(x1, yc, y1=0, d=5.0, tau1=1.0, shape=None):

        x1, yc, y1, d, tau1 = assert_arrays([x1, yc, y1, d, tau1], shape)

        return eqtn_fy1(x1, yc, y1, d, tau1)


    def calc_fz(x1=0.0, z=0, x0=0.0, K=0.0, w=0.0, tau1=1.0, tau0=1.0, x0cr=0.0, r=1.0, zmode=array("lin"), z_pos=None,
                model="2d",  shape=None):

        x1, z,  K, tau1, tau0 = assert_arrays([x1, z,  K, tau1, tau0], shape)

        if zmode == array("lin") and z_pos is None:
            z_pos = z > 0.0

        if model == "2d":

            x0cr, r = assert_arrays([x0cr, r], z.shape)

            return eqtn_fz(x1, z, x0, tau1, tau0, model, zmode, z_pos, K, w, coupl=None, x0cr=x0cr, r=r)

        else:

            return eqtn_fz(x1, z, x0, tau1, tau0, model, zmode, z_pos, K, w, coupl=None, x0cr=x0cr, r=r)


    def calc_fx2(x2, y2=0.0, z=0.0, g=0.0, Iext2=0.45, tau1=1.0, shape=None):

        x2, y2, z, g, Iext2, tau1 = assert_arrays([x2, y2, z, g, Iext2, tau1], shape)

        return eqtn_fx2(x2, y2, z, g, Iext2, tau1)


    def calc_fy2(x2, y2=0.0, s=6.0, tau1=1.0, tau2=1.0, x2_neg=None, shape=None):

        if x2_neg is None:
            x2_neg = x2 < -0.25

        x2, y2, s, tau1, tau2 = assert_arrays([x2, y2, s, tau1, tau2], shape)

        return eqtn_fy2(x2, y2, s, tau1, tau2, x2_neg)


    def calc_fg(x1, g=0.0, gamma=0.01, tau1=1.0, shape=None):

        x1, g, gamma, tau1 = assert_arrays([x1, g, gamma, tau1], shape)

        return eqtn_fg(x1, g, gamma, tau1)


    def calc_fx0(x0_var, x0, tau1=1.0, shape=None):

        x0_var, x0, tau1 = assert_arrays([x0_var, x0, tau1], shape)

        return eqtn_fx0(x0_var, x0, tau1)


    def calc_fslope(slope_var, slope, z=0.0, g=0.0, tau1=1.0, pmode=array("const"), shape=None):

        slope_var, slope, tau1 = assert_arrays([slope_var, slope, tau1], shape)

        if pmode == "z" or pmode == "g" or pmode == "z*g":

            z, g = assert_arrays([z, g], slope.shape)

            from tvb_epilepsy.tvb_api.epileptor_models import EpileptorDPrealistic
            slope = EpileptorDPrealistic.fun_slope_Iext2(z, g, pmode, slope, 0.0)[0]

        return eqtn_fslope(slope_var, slope, tau1)


    def calc_fIext1(Iext1_var, Iext1, tau1=1.0, tau0=1.0, shape=None):

        Iext1_var, Iext1, tau1, tau0 = assert_arrays([Iext1_var, Iext1, tau1, tau0], shape)

        return eqtn_fIext1(Iext1_var, Iext1, tau1, tau0)


    def calc_fIext2(Iext2_var, Iext2, z=0.0, g=0.0, tau1=1.0, pmode=array("const"), shape=None):

        Iext2_var, Iext2, tau1 = assert_arrays([Iext2_var, Iext2, tau1], shape)

        if pmode == "z" or pmode == "g" or pmode == "z*g":

            z, g = assert_arrays([z, g], Iext2.shape)

            from tvb_epilepsy.tvb_api.epileptor_models import EpileptorDPrealistic
            Iext2 = EpileptorDPrealistic.fun_slope_Iext2(z, g, pmode, 0.0, Iext2)[1]

        return eqtn_fIext2(Iext2_var, Iext2, tau1)


    def calc_fK(K_var, K, tau1=1.0, tau0=1.0, shape=None):

        K_var, K, tau1, tau0 = assert_arrays([K_var, K, tau1, tau0], shape)

        return eqtn_fK(K_var, K, tau1, tau0)


    def calc_dfun(x1, z, yc, Iext1, x0, K, w, model_vars=2, x0cr=None, r=None,
                  zmode="lin", pmode="const", x1_neg=None, z_pos=None, x2_neg=None,
                  y1=None, x2=None, y2=None, g=None,
                  x0_var=None, slope_var=None, Iext1_var=None, Iext2_var=None, K_var=None,
                  slope=0.0, a=1.0, b=-2.0, d=5.0, s=6.0, Iext2=0.45, gamma=0.01,
                  tau1=1.0, tau0=2857.0, tau2=10.0, shape=None, output_mode="array"):

        if output_mode == "array":

            return calc_dfun_array(x1, z, yc, Iext1, x0, K, w, model_vars, x0cr, r, zmode, pmode, x1_neg, z_pos, x2_neg,
                                   y1, x2, y2, g, x0_var, slope_var, Iext1_var, Iext2_var, K_var,
                                   slope, a, b, d, s, Iext2, gamma, tau1, tau0, tau2)

        else:

            if x1_neg is None:
                x1_neg = x1 < 0.0

            if zmode == array("lin") and z_pos is None:
                z_pos = z > 0.0

            x1, z,  yc, Iext1, x0, K, slope, a, b, tau1, tau0 = \
                assert_arrays([x1, z,  yc, Iext1, x0, K, slope, a, b, tau1, tau0], shape)

            w = assert_arrays([w], (z.size, z.size))

            if model_vars == 2:

                x0cr, r = assert_arrays([x0cr, r], z.shape)

                return eqtn_fx1(x1, z, yc, Iext1, slope, a, b, tau1, x1_neg, model="2d", x2=None), \
                       eqtn_fz(x1, z, x0, tau1, tau0, model="2d", zmode=zmode, z_pos=z_pos, K=K, w=w, coupl=None,
                               x0cr=x0cr, r=r)

            else:

                if x2_neg is None:
                    x2_neg = x2 < -0.25

                y1, x2, y2, g, Iext2, d, s, gamma, tau2 = \
                    assert_arrays([y1, x2, y2, g, Iext2, d, s, gamma, tau2], z.shape)

                if model_vars == 6:

                    return eqtn_fx1(x1, z, yc, Iext1, slope, a, b, tau1, x1_neg, model="6d", x2=x2), \
                           eqtn_fy1(x1, yc, y1, d, tau1), \
                           eqtn_fz(x1, z, x0, tau1, tau0, model="6d", zmode=zmode, z_pos=z_pos, K=K, w=w, coupl=None,
                                   x0cr=x0cr, r=r), \
                           eqtn_fx2(x2, y2, z, g, Iext2, tau1), \
                           eqtn_fy2(x2, y2, s, tau1, tau2, x2_neg), \
                           eqtn_fg(x1, g, gamma, tau1)

                elif model_vars == 11:

                    x0_var, slope_var, Iext1_var, Iext2_var, K_var = \
                        assert_arrays([x0_var, slope_var, Iext1_var, Iext2_var, K_var], z.shape)

                    dfun = (eqtn_fx1(x1, z, yc, Iext1_var, slope_var, a, b, tau1, x1_neg, model="6d", x2=x2),
                            eqtn_fy1(x1, yc, y1, d, tau1),
                            eqtn_fz(x1, z, x0_var, tau1, tau0, model="6d", zmode=zmode, z_pos=z_pos, K=K_var, w=w, coupl=None,
                                    x0cr=x0cr, r=r),
                            eqtn_fx2(x2, y2, z, g, Iext2_var, tau1),
                            eqtn_fy2(x2, y2, s, tau1, tau2, x2_neg),
                            eqtn_fg(x1, g, gamma, tau1),
                            eqtn_fx0(x0_var, x0, tau1))

                    if pmode == "z" or pmode == "g" or pmode == "z*g":

                        from tvb_epilepsy.tvb_api.epileptor_models import EpileptorDPrealistic
                        slope, Iext2 = EpileptorDPrealistic.fun_slope_Iext2(z, g, pmode, slope, Iext2)[1]

                    dfun += (eqtn_fslope(slope_var, slope, tau1),
                             eqtn_fIext1(Iext1_var, Iext1, tau1, tau0),
                             eqtn_fIext2(Iext2_var, Iext2, tau1),
                             eqtn_fK(K_var, K, tau1, tau0))

                    return dfun


    def calc_jac(x1, z, yc, Iext1, x0, K, w, model_vars=2, x0cr=None, r=None,
                 zmode="lin", pmode="const", x1_neg=None, z_pos=None, x2_neg=None,
                 y1=None, x2=None, y2=None, g=None,
                 x0_var=None, slope_var=None, Iext1_var=None, Iext2_var=None, K_var=None,
                 slope=0.0, a=1.0, b=-2.0, d=5.0, s=6.0, Iext2=0.45, gamma=0.01,
                 tau1=1.0, tau0=2857.0, tau2=10.0, shape=None):

        n_regions = max(shape_to_size(x1.shape), shape_to_size(z.shape))

        if model_vars == 2:

            if x1_neg is None:
                x1_neg = x1 < 0.0

            if z_pos is None:
                z_pos = z > 0.0

            x1, z,  yc, Iext1, x0, K, slope, a, b, tau1, tau0 = \
                assert_arrays([x1, z,  yc, Iext1, x0, K, slope, a, b, tau1, tau0], shape)

            w = assert_arrays([w], (n_regions, n_regions))

            return concatenate([eqtn_jac_x1_2d(x1, z, slope, a, b, tau1, x1_neg),
                                eqtn_jac_fz_2d(x1, z, tau1, tau0, zmode, z_pos, K, w)])

        else:



            if model_vars == 6:

                sx1, sy1, sz, sx2, sy2, sg = symbol_vars(n_regions, ['x1', 'y1', 'z', 'x2', 'y2', 'g'])[:6]

                dfun_sym = calc_dfun_array(sx1, sz, yc, Iext1, x0, K, w, model_vars, x0cr, r,
                                           zmode, pmode, x1_neg, z_pos, x2_neg,
                                           sy1, sx2, sy2, sg,
                                           x0_var, slope_var, Iext1_var, Iext2_var, K_var,
                                           slope, a, b, d, s, Iext2, gamma, tau1, tau0, tau2)

                x = Matrix([sx1, sy1, sz, sx2, sy2, sg]).reshape(6 * n_regions, 1)
                jac_sym = Matrix(dfun_sym.flatten()).jacobian(x)

                jac_lambda = lambdify([x], jac_sym, "numpy")

                return array(jac_lambda([x1, y1, z, x2, y2, g])).astype(x1.dtype)

            elif model_vars == 11:

                sx1, sy1, sz, sx2, sy2, sg, sx0_var, sslope_var, sIext1_var, sIext2_var, sK_var = \
                                symbol_vars(n_regions, ['x1', 'y1', 'z', 'x2', 'y2', 'g',
                                                        'x0_var', 'slope_var', 'Iext1_var', 'Iext2_var', 'K_var'])[:11]

                dfun_sym = calc_dfun_array(sx1, sz, yc, Iext1, x0, K, w, model_vars, x0cr, r,
                                           zmode, pmode, x1_neg, z_pos, x2_neg,
                                           sy1, sx2, sy2, sg,
                                           sx0_var, sslope_var, sIext1_var, sIext2_var, sK_var,
                                           slope, a, b, d, s, Iext2, gamma, tau1, tau0, tau2)

                x = Matrix([sx1, sy1, sz, sx2, sy2, sg, sx0_var, sslope_var, sIext1_var, sIext2_var, sK_var])\
                    .reshape(11 *  n_regions, 1)
                jac_sym = Matrix(dfun_sym.flatten()).jacobian(x)

                jac_lambda = lambdify([x], jac_sym, "numpy")

                return array(jac_lambda([x1, y1, z, x2, y2, g, x0_var, slope_var, Iext1_var, Iext2_var, K_var]))\
                    .astype(x1.dtype)


    def calc_coupling_diff(K, w, ix=None, jx=None):

        K = assert_arrays([K])

        w = assert_arrays([w], (K.size, K.size))

        n_regions = K.size

        if ix is None:
            ix = range(n_regions)

        if jx is None:
            jx = range(n_regions)

        return eqtn_coupling_diff(K, w, ix, jx)


    def calc_fx1_2d_taylor(x1, x_taylor, z=0, y1=0.0, Iext1=0.0, slope=0.0, a=1.0, b=-2.0, tau1=1.0, x1_neg=True,
                           order=2, shape=None):

        x1, x_taylor, z, y1, Iext1, slope, a, b, tau1 = \
            assert_arrays([x1, x_taylor, z, y1, Iext1, slope, a, b, tau1], shape)

        if x1_neg is None:
            x1_neg = x1 < 0.0

        if order == 2 and numpy.all(x1_neg == True):

            fx1lin = multiply(Iext1 + 2 * multiply(power(x_taylor, 3), a) - multiply(power(x_taylor, 2), b) + y1 - z +
                              multiply(x1, (-3 * multiply(power(x_taylor, 2), a) + 2 * multiply(x_taylor, b))), tau1)

        else:

            try:
                from sympy import Symbol, series

            except:
                raise ImportError("Unable to load symbolic_equations module. Taylor expansion calculation is not  \
                                  supported non-symbolically for order of expansion >2 and/or when any x1 > 0.")

            from sympy import Symbol, series

            x = Symbol("x")

            fx1lin = calc_fx1(x1, z, y1, Iext1, slope, a, b, tau1, x1_neg).flatten()

            for ix in range(x1.size):
                fx1lin[ix] = series(fx1lin[ix], x=x, x0=x_taylor, n=order).removeO().simplify(). \
                    subs(x, x1.flatten()[ix])

            fx1lin = reshape(fx1lin, shape).astype(x1.dtype)

        return fx1lin


    def calc_fx1z(x1, x0, K, w, yc, Iext1, x0cr=0.0, r=1.0, a=1.0, b=-2.0, d=5.0, tau1=1.0, tau0=1.0,
                  model="6d", zmode=array("lin"), shape=None):  #, slope=0.0, x2=0.0, x1_neg=None, z_pos=True

        # TODO: for the extreme z_pos = False case where we have terms like 0.1 * z ** 7
        # TODO: for the extreme x1_neg = False case where we have to solve for x2 as well

        x1, x0, K, yc, Iext1, a, b, tau1, tau0 = assert_arrays([x1, x0, K, yc, Iext1, a, b, tau1, tau0], shape)

        w = assert_arrays([w], (x1.size, x1.size))

        if model == "2d":

            x0cr, r = assert_arrays([x0cr, r], x1.shape)

            z = calc_fx1(x1, 0.0, yc, Iext1, 0.0, a, b, tau1=1.0, x2=0.0, model=model, x1_neg=True, shape=x1.shape)

            return eqtn_fz(x1, z, x0, tau1, tau0, model, zmode, z_pos=True, K=K, w=w, coupl=None, x0cr=x0cr, r=r)

        else:

            d = assert_arrays([d], x1.shape)

            y1 = calc_fy1(x1, yc, 0.0, d, tau1=1.0, shape=x1.shape)

            z = calc_fx1(x1, 0.0, y1, Iext1, 0.0, a, b, tau1=1.0, x2=0.0, model=model, x1_neg=True, shape=x1.shape)

            return eqtn_fz(x1, z, x0, tau1, tau0, model, zmode, z_pos=True, K=K, w=w , coupl=None, x0cr=0.0, r=1.0)


    def calc_fx1z_diff(x1, K, w, a=1.0, b=-2.0, d=5.0, tau1=1.0, tau0=1.0, model="6d", zmode=array("lin")):
        #, x1_neg=None, z_pos=True

        # TODO: for the extreme z_pos = False case where we have terms like 0.1 * z ** 7
        # TODO: for the extreme x1_neg = False case where we have to solve for x2 as well

        x1, K, a, b, tau1, tau0 = assert_arrays([x1, K, a, b, tau1, tau0])

        w = assert_arrays([w], (x1.size, x1.size))

        if model != "2d":
            d = assert_arrays([d], x1.shape)

        ix = range(x1.size)

        return eqtn_fx1z_diff(x1, K, w, ix, ix, a, b, d, tau1, tau0, model, zmode)


    def calc_fx1z_2d_x1neg_zpos_jac(x1, z, x0, x0cr, r, yc, Iext1, K, w, ix0, iE, a=1.0, b=-2.0, tau1=1.0, tau0=1.0):

        if x1.shape != (1, x1.size):
            x1 = expand_dims(x1.flatten(), 1).T

        x1, z, x0, x0cr, r, yc, Iext1, K, a, b, tau1, tau0 = \
            assert_arrays([x1, z, x0, x0cr, r, yc, Iext1, K, a, b, tau1, tau0], x1.shape)

        w = assert_arrays([w], (x1.size, x1.size))

        tau = divide(tau1, tau0)

        no_x0 = len(ix0)
        no_e = len(iE)

        i_x0 = ones((no_x0, 1))
        i_e = ones((no_e, 1))

        jac_e_x0e = diag(multiply(tau[:, iE], (- 4 * r[:, iE])).flatten())
        jac_e_x1o = -dot(dot(i_e, multiply(tau[:, iE], K[:, iE])), w[iE][:, ix0])
        jac_x0_x0e = zeros((no_x0, no_e))
        jac_x0_x1o = (diag(multiply(tau[:, ix0],
                                    (4 + 3 * multiply(a[:, ix0], power(x1[:, ix0], 2))
                                     - 2 * multiply(b[:, ix0], x1[:, ix0]) +
                                     multiply(K[:, ix0], sum(w[ix0], axis=1)))).flatten()) -
                      multiply(dot(i_x0, multiply(tau[:, ix0], K[:, ix0])).T, w[ix0][:, ix0]))

        jac = empty((x1.size, x1.size), dtype=jac_e_x0e.dtype)
        jac[numpy.ix_(iE, iE)] = jac_e_x0e
        jac[numpy.ix_(iE, ix0)] = jac_e_x1o
        jac[numpy.ix_(ix0, iE)] = jac_x0_x0e
        jac[numpy.ix_(ix0, ix0)] = jac_x0_x1o

        return jac


    def calc_fx1y1_6d_diff_x1(x1, yc, Iext1, a=1.0, b=3.0, d=5.0, tau1=1.0, shape=None):

        x1 = assert_arrays([x1], shape)
        shape = x1.shape

        if x1.shape != (1, x1.size):
            x1 = expand_dims(x1.flatten(), 1).T

        p = x1.shape

        yc, Iext1, a, b, d, tau1 = assert_arrays([yc, Iext1, a, b, d, tau1], shape)

        dfx1 = multiply(multiply(-3 * multiply(x1, a) + 2 * (b - d), x1), tau1)

        return reshape(dfx1, shape)


    def calc_x0cr_r(yc, Iext1, a=1.0, b=-2.0, zmode=array("lin"), x1_rest=X1_DEF, x1_cr=X1_EQ_CR_DEF, x0def=X0_DEF,
                    x0cr_def=X0_CR_DEF, test=False, shape=None):  # epileptor_model="2d",

        Iext1, yc, a, b = assert_arrays([Iext1, yc, a, b], shape)
        shape = Iext1.shape

        if test:
            if numpy.all(Iext1 == Iext1[0]) and numpy.all(yc == yc[0]) \
                    and numpy.all(a == a[0]) and numpy.all(b == b[0]):
                Iext1 = Iext1[0]
                yc = yc[0]
                a = a[0]
                b = b[0]

            p2 = Iext1.shape
            x1_rest, x1_cr = assert_arrays([x1_rest, x1_cr], p2)

            # Define the z equilibrium expression...
            # if epileptor_model == "2d":
            zeq_rest = calc_fx1(x1_rest, z=0.0, y1=yc, Iext1=Iext1, a=a, b=b, tau1=1.0, x2=0.0, model="2d",
                                x1_neg=True)
            zeq_cr = calc_fx1(x1_cr, z=0.0, y1=yc, Iext1=Iext1, a=a, b=b, tau1=1.0, x2=0.0, model="2d", x1_neg=True)
            if zmode == array("lin"):
                xinit = array([2.460, 0.398])
            else:
                xinit = array([3.174, 0.260])
            # else:
            #     zeq_rest = calc_fx1(x1_rest, z=0.0, y1=calc_fy1(x1_rest, yc), Iext1=Iext1, x1_neg=True)
            #     zeq_cr = calc_fx1(x1_cr, z=0.0, y1=calc_fy1(x1_cr, yc), Iext1=Iext1, x1_neg=True)
            #     if zmode == array("lin"):
            #         xinit = array([5.9320, 1.648])
            #     else:
            #         xinit = array([17.063, 5.260])

            # Define the fz expression...
            x0cr = []
            r = []
            for ii in range(Iext1.size):
                fz = lambda x: array([calc_fz(x1_rest[ii], z=zeq_rest[ii], x0=x0def, K=0.0, w=0.0, tau1=1.0, tau0=1.0,
                                              x0cr=x[0], r=x[1], zmode=zmode, z_pos=True, model="2d", shape=None),
                                      calc_fz(x1_cr[ii], z=zeq_cr[ii], x0=x0cr_def, K=0.0, w=0.0, tau1=1.0, tau0=1.0,
                                              x0cr=x[0], r=x[1], zmode=zmode, z_pos=True, model="2d", shape=None)])

                sol = root(fz, xinit, method='lm', tol=10 ** (-12), callback=None, options=None)

                if sol.success:
                    if numpy.any([numpy.any(numpy.isnan(sol.x)), numpy.any(numpy.isinf(sol.x))]):
                        raise ValueError("nan or inf values in solution x\n" + sol.message)
                    x0cr.append(sol.x[0])
                    r.append(sol.x[1])
                else:
                    raise ValueError(sol.message)

            if p2 != shape:
                x0cr = numpy.tile(x0cr[0], shape)
                r = numpy.tile(r[0], shape)
            else:
                x0cr = reshape(x0cr, shape)
                r = reshape(r, shape)

        else:

            x0cr, r = eqtn_x0cr_r(yc, Iext1, a, b, x1_rest, x1_cr, x0def, x0cr_def, zmode=zmode)

        return x0cr, r


def calc_fpop2(x2, y2=0.0, z=0.0, g=0.0, Iext2=0.45, s=6.0, tau1=1.0, tau2=1.0, x2_neg=None, shape=None):

    return calc_fx2(x2, y2, z, g, Iext2, tau1, shape), calc_fy2(x2, y2, s, tau1, tau2, x2_neg, shape)


def calc_fparams_var(x0_var, slope_var, Iext1_var, Iext2_var, K_var, x0, slope, Iext1, Iext2, K, z=0.0, g=0.0,
                     tau1=1.0, tau0=1.0, pmode=array("const"), shape=None):

    return calc_fx0(x0_var, x0, tau1, shape), \
           calc_fslope(slope_var, slope, z, g, tau1, pmode, shape), \
           calc_fIext1(Iext1_var, Iext1, tau1, tau0, shape), \
           calc_fIext2(Iext2_var, Iext2, z, g, tau1, pmode, shape), \
           calc_fK(K_var, K, tau1, tau0, shape)


def calc_dfun_array(x1, z, yc, Iext1, x0, K, w, model_vars=2, x0cr=None, r=None,
                    zmode="lin", pmode="const", x1_neg=None, z_pos=None, x2_neg=None,
                    y1=None, x2=None, y2=None, g=None,
                    x0_var=None, slope_var=None, Iext1_var=None, Iext2_var=None, K_var=None,
                    slope=0.0, a=1.0, b=-2.0, d=5.0, s=6.0, Iext2=0.45, gamma=0.01,
                    tau1=1.0, tau0=2857.0, tau2=10.0):

    n_regions = max(shape_to_size(x1.shape), shape_to_size(z.shape))

    shape = (1, n_regions)

    f = empty((model_vars, n_regions), dtype=type(x1[0]))

    if model_vars == 2:

        f[0, :] = calc_fx1(x1, z, yc, Iext1, slope, a, b, tau1, x2, model="2d", x1_neg=x1_neg, shape=shape)
        f[1, :] = calc_fz(x1, z, x0, K, w, tau1, tau0, x0cr, r, zmode, z_pos, model="2d", shape=shape)

    elif model_vars == 6:

        f[0, :] = calc_fx1(x1, z, y1, Iext1, slope, a, b, tau1, x2=x2, model="6d", x1_neg=x1_neg, shape=shape)
        f[1, :] = calc_fy1(x1, yc, y1, d, tau1, shape)
        f[2, :] = calc_fz(x1, z, x0, K, w, tau1, tau0, x0cr, r, zmode, z_pos, model="6d", shape=shape)
        f[3, :] = calc_fx2(x2, y2, z, g, Iext2, tau1, shape)
        f[4, :] = calc_fy2(x2, y2, s, tau1, tau2, x2_neg, shape)
        f[5, :] = calc_fg(x1, g, gamma, tau1, shape)

    elif model_vars == 11:

        f[0, :] = calc_fx1(x1, z, y1, Iext1_var, slope_var, a, b, tau1, x2=x2, model="6d", x1_neg=x1_neg,
                           shape=shape)
        f[1, :] = calc_fy1(x1, yc, y1, d, tau1, shape)
        f[2, :] = calc_fz(x1, z, x0_var, K_var, w, tau1, tau0, x0cr, r, zmode, z_pos, model="6d", shape=shape)
        f[3, :] = calc_fx2(x2, y2, z, g, Iext2_var, tau1, shape)
        f[4, :] = calc_fy2(x2, y2, s, tau1, tau2, x2_neg, shape)
        f[5, :] = calc_fg(x1, g, gamma, tau1, shape)
        f[6, :] = calc_fx0(x0_var, x0, tau1, shape)
        f[7, :] = calc_fslope(slope_var, slope, z, g, tau1, pmode, shape)
        f[8, :] = calc_fIext1(Iext1_var, Iext1, tau1, tau0, shape)
        f[9, :] = calc_fIext2(Iext2_var, Iext2, z, g, tau1, pmode, shape)
        f[10, :] = calc_fK(K_var, K, tau1, tau0, shape)

    return f


def rescale_x0(x0_2d, yc, Iext1, a=1.0, b=-2.0, zmode=array("lin"), shape=None):

    x0_2d, yc, Iext1, a, b = assert_arrays([x0_2d, yc, Iext1, a, b ], shape)

    x0cr, r = calc_x0cr_r(yc, Iext1, a, b, zmode=zmode) #epileptor_model="6d",

    return multiply(r, x0_2d) - x0cr