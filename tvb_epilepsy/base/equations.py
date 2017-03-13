import warnings

import numpy

from tvb_epilepsy.base.constants import SYMBOLIC_EQUATIONS_FLAG
# TODO: find out why I cannot import anything from utils here
#from tvb_epilepsy.base.utils import assert_array_shape as sc2arr

if SYMBOLIC_EQUATIONS_FLAG:

    try:
        from tvb_epilepsy.base.symbolic_equations import *

    except:
        warnings.warn("Unable to load symbolic_equations module. Turning to non symbolic ones.")
        SYMBOLIC_EQUATIONS_FLAG = False


def sc2arr(x, shape):

    if isinstance(x, numpy.ndarray):

        if x.shape == shape:
            return x

        else:
            try:
                return numpy.reshape(x, shape)
            except:
                raise ValueError("Input is a numpy.ndarray of shape " + str(x.shape) +
                                 " that cannot be reshaped to the desired shape " + str(shape))

    elif isinstance(x, (float, int, long, complex)):
        return x * numpy.ones(shape, dtype=type(x))

    elif isinstance(x, list):
        # assuming a list of symbols...
        return numpy.array(x, dtype=type(x[0]))

    else:
        try:
            from sympy.core.symbol import Symbol
        except:
            raise ImportError()

        if isinstance(x, Symbol):
            return numpy.tile(x, shape)

        else:
            raise ValueError("Input of type " + str(type(x)) + " is not numeric, of type numpy.ndarray, nor Symbol")


if SYMBOLIC_EQUATIONS_FLAG:

    def calc_coupling(x1, K, w, ix=None, jx=None):

        x1 = numpy.array(x1)

        p = x1.shape

        return numpy.reshape(eqtn_coupling(x1.size, ix, jx)[0](x1, sc2arr(K, p), sc2arr(w, (x1.size, x1.size))))\
               .astype(x1.dtype)


    def calc_x0(x1, z, x0cr, r, K, w, zmode=numpy.array("lin")):

        x1 = numpy.array(x1)

        p = x1.shape

        return numpy.reshape(eqtn_x0(x1.size, zmode)[0](x1, z, x0cr, r, K, w), x1.shape).astype(x1.dtype)


    def calc_fx1_6d(x1, z=0.0, y1=0.0, x2=0.0, Iext1=0.0, slope=0.0, a=1.0, b=3.0, tau1=1.0, x1_neg=None):

        x1 = numpy.array(x1)

        p = x1.shape

        if x1_neg is None:
            x1_neg = x1 < 0.0

        return numpy.reshape(eqtn_fx1_6d(x1.size, x1_neg)[0](x1, sc2arr(z,p), sc2arr(y1,p), sc2arr(x2,p),
                                                                 sc2arr(Iext1, p),sc2arr(slope,p), sc2arr(a, p),
                                                                 sc2arr(b, p), sc2arr(tau1, p)),
                                                             x1.shape).astype(x1.dtype)


    def calc_fx1_2d(x1, z=0.0, yc=0.0, Iext1=0.0, slope=0.0, a=1.0, b=-2.0, tau1=1.0, x1_neg=None):

        x1 = numpy.array(x1)

        p = x1.shape

        if x1_neg is None:
            x1_neg = x1 < 0.0

        return numpy.reshape(eqtn_fx1_2d(x1.size, x1_neg)[0](x1, sc2arr(z,p), sc2arr(yc,p), sc2arr(Iext1,p),
                                                                 sc2arr(slope,p), sc2arr(a, p), sc2arr(b, p),
                                                                 sc2arr(tau1, p)), x1.shape).astype(x1.dtype)


    def calc_fy1(x1, yc, y1=0, d=5.0, tau1=1.0):

        x1 = numpy.array(x1)

        p = x1.shape

        return numpy.reshape(eqtn_fy1(x1.size)[0](x1, sc2arr(y1, p), sc2arr(yc, p),
                                                      sc2arr(d, p), sc2arr(tau1, p)), x1.shape).astype(x1.dtype)


    def calc_fz(x1, x0, x0cr, r, z=0, K=0, w=0.0, tau1=1.0, tau0=1.0, zmode=numpy.array("lin")):

        x1 = numpy.array(x1)

        p = x1.shape

        return numpy.reshape(eqtn_fz(x1.size, zmode)[0](x1, sc2arr(z, p), sc2arr(x0, p), sc2arr(x0cr, p), sc2arr(r, p),
                                                            sc2arr(K, p), sc2arr(w, (x1.size, x1.size)),
                                                            sc2arr(tau1, p), sc2arr(tau0, p)),
                                                            x1.shape).astype(x1.dtype)


    def calc_fpop2(x2, y2=0.0, z=0.0, g=0.0, Iext2=0.45, s=6.0, tau1=1.0, tau2=10.0, x2_neg=None):

        x2 = numpy.array(x2)

        p = x2.shape

        if x2_neg is None:
            x2_neg = x2 < -0.25

        fx2fy2 = eqtn_fpop2(x2.size, x2_neg)[0]

        return numpy.reshape(fx2fy2[0](x2, sc2arr(y2, p), sc2arr(z, p), sc2arr(g, p), sc2arr(Iext2, p), tau1), \
                             x2.shape).astype(x2.dtype), \
               numpy.reshape(fx2fy2[1](x2, sc2arr(y2, p), s, tau1, tau2), x2.shape).astype(x2.dtype)


    def calc_fg(x1, g=0.0, gamma=0.01, tau1=1.0):

        x1 = numpy.array(x1)

        p = x1.shape

        return numpy.reshape(eqtn_fg(x1.size)[0](x1, sc2arr(g, p), sc2arr(gamma, p), sc2arr(tau1, p)),
                             x1.shape).astype(x1.dtype)


    def calc_fparams_var(x0_var, slope_var, Iext1_var, Iext2_var, K_var, x0, slope, Iext1, Iext2, K, z=0.0, g=0.0,
                         tau1=1.0, tau0=1.0, pmode=numpy.array("const")):

        x0_var =  numpy.arra(x0_var)

        p = x0_var.shape

        f = eqtn_fparam_vars(x0_var.size, pmode=numpy.array("const"))[0]

        return numpy.reshape(f[0](sc2arr(x0, p), sc2arr(x0_var, p), sc2arr(tau1, p)),
                             x0_var.shape).astype(x0_var.dtype), \
               numpy.reshape(f[1](sc2arr(z, p), sc2arr(g, p), sc2arr(slope, p), sc2arr(slope_var, p), sc2arr(tau1, p)),
                             slope_var.shape).astype(slope_var.dtype), \
               numpy.reshape(f[2](sc2arr(Iext1, p), sc2arr(Iext1_var, p), sc2arr(tau1, p), sc2arr(tau0, p)),
                             Iext1_var.shape).astype(Iext1_var.dtype), \
               numpy.reshape(f[3](sc2arr(z, p), sc2arr(g, p), sc2arr(Iext2, p), sc2arr(Iext2_var, p), sc2arr(tau1, p)),
                             Iext2_var.shape).astype(Iext2_var.dtype), \
               numpy.reshape(f[4](sc2arr(K, p), sc2arr(K_var, p), sc2arr(tau1, p), sc2arr(tau0, p)),
                             K_var.shape).astype(K_var.dtype)


    def calc_dfun(x1, z, yc, Iext1, x0, x0cr, r, K, w, model_vars=2, zmode="lin", pmode="const", x1_neg=None,
                  y1=None, x2=None, y2=None, g=None, x2_neg=None,
                  x0_var=None, slope_var=None, Iext1_var=None, Iext2_var=None, K_var=None,
                  slope=0.0, a=1.0, b=-2.0, d=5.0, s=6.0, Iext2=0.45, gamma=0.01,
                  tau1=1.0, tau0=2857.0, tau2=10.0):

        x1 = numpy.array(x1)

        p = x1.shape

        if x1_neg is None:
            x1_neg = x1 < 0.0

        if model_vars > 2:
            if x2_neg is None:
                x2_neg = x2 < -0.25


        f = numpy.zeros((model_vars, x1.size), dtype=x1.dtype)

        dfun = eqnt_dfun(x1.size, model_vars, zmode, x1_neg, x2_neg, pmode)[0]

        p = x1.shape

        z = sc2arr(z, p)
        yc = sc2arr(yc, p)
        Iext1 = sc2arr(Iext1, p)
        slope = sc2arr(slope, p)
        x0 = sc2arr(x0, p)
        x0cr = sc2arr(x0cr, p)
        r = sc2arr(r, p)
        K = sc2arr(K, p)
        w = sc2arr(w, (x1.size, x1.size))
        a = sc2arr(a, p)
        b = sc2arr(b, p)
        tau1 = sc2arr(tau1, p)
        tau0 = sc2arr(tau0, p)

        if model_vars == 2:

            f[0,:] = numpy.array(dfun[0](x1, z, yc, Iext1, slope, a, b, tau1), dtype=x1.dtype)
            f[1, :] = numpy.array(dfun[1](x1, z, x0, x0cr, r, K, w, tau1, tau0), dtype=x1.dtype)

        else:

            y1 = sc2arr(y1, p)
            x2 = sc2arr(x2, p)
            y2 = sc2arr(y2, p)
            g = sc2arr(g, p)
            Iext2 = sc2arr(Iext2, p)
            tau2 = sc2arr(tau2, p)
            gamma = sc2arr(gamma, p)
            d = sc2arr(d, p)
            s = sc2arr(s, p)

            if model_vars == 6:

                f[0, :] = numpy.array(dfun[0](x1, z, y1, x2, Iext1, slope, a, b, tau1), dtype=x1.dtype)
                f[1, :] = numpy.array(dfun[1](x1, y1, yc, d, tau1), dtype=x1.dtype)
                f[2, :] = numpy.array(dfun[2](x1, z, x0, x0cr, r, K, w, tau1, tau0), dtype=x1.dtype)
                f[3, :] = numpy.array(dfun[3](x2, y2, z, g, Iext2, tau1), dtype=x1.dtype)
                f[4, :] = numpy.array(dfun[4](x2, y2, s, tau1, tau2), dtype=x1.dtype)
                f[5, :] = numpy.array(dfun[5](x1, g, gamma, tau1), dtype=x1.dtype)

            elif model_vars == 11:

                x0_var = sc2arr(x0_var, p)
                slope_var = sc2arr(slope_var, p)
                Iext1_var = sc2arr(Iext1_var, p)
                Iext2_var = sc2arr(Iext2_var, p)
                K_var = sc2arr(K_var, p)

                f[0, :] = numpy.array(dfun[0](x1, z, y1, x2, Iext1_var, slope_var, a, b, tau1), dtype=x1.dtype)
                f[1, :] = numpy.array(dfun[1](x1, y1, yc, d, tau1), dtype=x1.dtype)
                f[2, :] = numpy.array(dfun[2](x1, z, x0_var, x0cr, r, K, w, tau1, tau0), dtype=x1.dtype)
                f[3, :] = numpy.array(dfun[3](x2, y2, z, g, Iext2_var, tau1), dtype=x1.dtype)
                f[4, :] = numpy.array(dfun[4](x2, y2, s, tau1, tau2), dtype=x1.dtype)
                f[5, :] = numpy.array(dfun[5](x1, g, gamma, tau1), dtype=x1.dtype)
                f[6, :] = numpy.array(dfun[6](x0, x0_var, tau1), dtype=x1.dtype)
                f[7, :] = numpy.array(dfun[7](z, g, slope, slope_var, tau1), dtype=x1.dtype)
                f[8, :] = numpy.array(dfun[8](Iext1, Iext1_var, tau1, tau0), dtype=x1.dtype)
                f[9, :] = numpy.array(dfun[9](z, g, Iext2, Iext2_var, tau1), dtype=x1.dtype)
                f[10, :] = numpy.array(dfun[10](K, K_var, tau1, tau0), dtype=x1.dtype)

        return f


    def calc_jac(x1, z, yc, Iext1, x0, x0cr, r, K, w, model_vars=2, zmode="lin", pmode="const", x1_neg=None,
                 y1=None, x2=None, y2=None, g=None, x2_neg=None,
                 x0_var=None, slope_var=None, Iext1_var=None, Iext2_var=None, K_var=None,
                 slope=0.0, a=1.0, b=-2.0, d=5.0, s=6.0, Iext2=0.45, gamma=0.01,
                 tau1=1.0, tau0=2857.0, tau2=10.0):

        jac = eqnt_jac(x1.size, model_vars, zmode, x1_neg, x2_neg, pmode)[0]

        return jac

else:

    def calc_coupling(x1, K, w, ix=None, jx=None):

        # Only difference coupling for the moment.
        # TODO: Extend for different coupling forms

        x1 = numpy.array(x1)

        n_regions = x1.size

        if ix is None:
            ix = range(n_regions)
            n_ix = n_regions
        else:
            n_ix = len(ix)

        if jx is None:
            jx = range(n_regions)
            n_jx = n_regions
        else:
            n_jx = len(jx)

        i_n = numpy.ones((n_ix, 1), dtype='float32')
        j_n = numpy.ones((n_jx, 1), dtype='float32')

        x1_shape = list(x1.shape)
        x1_shape[numpy.argmax(x1_shape)] = n_ix
        x1_shape = tuple(x1_shape)
        x1 = numpy.expand_dims(x1.squeeze(), 1).T
        K = numpy.reshape(K, x1.shape)

        # Coupling                                                        from                 to
        coupling = K[:, ix]*numpy.sum(numpy.dot(w[ix][:, jx], numpy.dot(i_n, x1[:, jx])
                                                                                - numpy.dot(j_n, x1[:, ix]).T), axis=1)
        return numpy.reshape(coupling, x1.shape).astype(x1.dtype)


    def calc_x0(x1, z, x0cr, r, K, w, zmode=numpy.array("lin")):

        x1 = numpy.array(x1)

        coupl = numpy.array(calc_coupling(x1, K, w))

        if zmode == 'lin':
             x0 =  (x1 + x0cr - (z+coupl) / 4.0) / r

        elif zmode == 'sig':
            x0 = (3.0 / (1.0 + numpy.exp(-10.0 * (x1 + 0.5))) + x0cr - z + coupl) / r

        else:
            raise ValueError('zmode is neither "lin" nor "sig"')

        return numpy.reshape(x0, x1.shape).astype(x1.dtype)




    def calc_fx1_6d(x1, z=0.0, y1=0.0, x2=0.0, Iext1=0.0, slope=0.0, a=1.0, b=3.0, tau1=1.0, x1_neg=None):

        x1 = numpy.array(x1)

        # if_ydot0 = - self.a * y[0] ** 2 + self.b * y[0]
        if_ydot0 = - a * x1 ** 2 + b * x1  # self.a=1.0, self.b=3.0

        # else_ydot0 = self.slope - y[3] + 0.6 * (y[2] - 4.0) ** 2
        else_ydot0 = slope - x2 + 0.6 * (z - 4.0) ** 2

        if x1_neg is None:
            x1_neg = x1 < 0.0

        fx1 = tau1 * (y1 - z + Iext1 + numpy.where(x1_neg, if_ydot0, else_ydot0) * x1)

        return numpy.reshape(fx1, x1.shape).astype(x1.dtype)



    def calc_fx1_2d(x1, z=0, yc=0.0, Iext1=0.0, slope=0.0, a=1.0, b=-2.0, tau1=1.0, x1_neg=None):

        x1 = numpy.array(x1)

        # if_ydot0 = - self.a * y[0] ** 2 + self.b * y[0]
        if_ydot0 = - a * x1 ** 2 + b * x1  # self.a=1.0, self.b=3.0

        # else_ydot0 = self.slope - y[3] + 0.6 * (y[2] - 4.0) ** 2
        else_ydot0 = slope - 5.0*x1 + 0.6 * (z - 4.0) ** 2

        if x1_neg is None:
            x1_neg = x1 < 0.0

        fx1 = tau1 * (yc - z + Iext1 + numpy.where(x1_neg, if_ydot0, else_ydot0) * x1)

        return numpy.reshape(fx1, x1.shape).astype(x1.dtype)



    def calc_fy1(x1, yc, y1=0, d=5.0, tau1=1.0):

        x1 = numpy.array(x1)

        fy1 = tau1 * (yc - d * x1 ** 2 - y1)

        return numpy.reshape(fy1, x1.shape).astype(x1.dtype)


    def calc_fz(x1, x0, x0cr, r, z=0, K=0.0, w=0.0, tau1=1.0, tau0=1.0, zmode=numpy.array("lin")):

        x1 = numpy.array(x1)

        if numpy.any(K> 0.0) and numpy.any(w> 0.0):
            coupl = numpy.array(calc_coupling(x1, K, w))
        else:
            coupl = 0.0

        if zmode == 'lin':
            fz = tau1 * (4 * (x1 - r * x0 + x0cr) - z - coupl) / tau0

        elif zmode == 'sig':
            fz = tau1 * (3 / (1 + numpy.exp(-10.0 * (x1 + 0.5))) - r * x0 + x0cr - z - coupl) / tau0

        else:
            raise ValueError('zmode is neither "lin" nor "sig"')

        return numpy.reshape(fz, x1.shape).astype(x1.dtype)



    def calc_fpop2(x2, y2=0.0, z=0.0, g=0.0, Iext2=0.45, s=6.0, tau1=1.0, tau2=10.0, x2_neg=None):

        x2 = numpy.array(x2)

        # ydot[3] = self.tt * (-y[4] + y[3] - y[3] ** 3 + self.Iext2 + 2 * y[5] - 0.3 * (y[2] - 3.5) + self.Kf * c_pop2)
        fx2 = tau1 * (-y2 + x2 - x2 ** 3 + Iext2 + 2 * g - 0.3 * (z - 3.5))

        # if_ydot4 = 0
        if_ydot4 = 0
        # else_ydot4 = self.aa * (y[3] + 0.25)
        else_ydot4 = s * (x2 + 0.25)  # self.s = 6.0

        if x2_neg is None:
            x2_neg = x2 < -0.25

        # ydot[4] = self.tt * ((-y[4] + where(y[3] < -0.25, if_ydot4, else_ydot4)) / self.tau)
        fy2 = tau1 * ((-y2 + numpy.where(x2_neg, if_ydot4, else_ydot4)) / tau2)

        return numpy.reshape(fx2, x2.shape).astype(x2.dtype), numpy.reshape(fy2, x2.shape).astype(x2.dtype)


    def calc_fg(x1, g=0, gamma=0.01, tau1=1.0):

        x1 = numpy.array(x1)

        #ydot[5] = self.tt * (-0.01 * (y[5] - 0.1 * y[0]))
        fg = -tau1 * gamma * (g - 0.1 * x1)

        return numpy.reshape(fg, x1.shape).astype(x1.dtype)


    def calc_fparams_var(x0_var, slope_var, Iext1_var, Iext2_var, K_var, x0, slope, Iext1, Iext2, K, z=0.0, g=0.0,
                         tau1=1.0, tau0 = 1.0, pmode=numpy.array("const")):

        x0_var = numpy.array(x0_var)

        # ydot[6] = self.tau1 * (-y[6] + self.x0)
        fx0 = tau1 * (-x0_var + x0)

        from tvb_epilepsy.tvb_api.epileptor_models import EpileptorDPrealistic
        slope_eq, Iext2_eq = EpileptorDPrealistic.fun_slope_Iext2(z, g, pmode, slope, Iext2)

        # slope
        # ydot[7] = 10 * self.tau1 * (-y[7] + slope_eq)
        fslope = 10.0 * tau1 * (-slope_var + slope_eq)

        # Iext1
        # ydot[8] = self.tau1 * (-y[8] + self.Iext1) / self.tau0
        fIext1 = tau1 * (-Iext1_var + Iext1) / tau0

        # Iext2
        # ydot[9] = 5 * self.tau1 * (-y[9] + Iext2_eq)
        fIext2 = 5.0 * tau1 * (-Iext2_var + Iext2_eq)

        # K
        # ydot[10] = self.tau1 * (-y[10] + self.K) / self.tau0
        fK = tau1 * (-K_var + K) / tau0

        return numpy.array(fx0).astype(x0.dtype), numpy.array(fslope).astype(slope_var.dtype), \
               numpy.array(fIext1).astype(Iext1_var.dtype), numpy.array(fIext2).astype(Iext2_var.dtype), \
               numpy.array(fK).astype(K_var.dtype)


    def calc_dfun(x1, z, yc, Iext1, x0, x0cr, r, K, w, model_vars=2, zmode="lin", pmode="const", x1_neg=None,
                  y1=None, x2=None, y2=None, g=None, x2_neg=None,
                  x0_var=None, slope_var=None, Iext1_var=None, Iext2_var=None, K_var=None,
                  slope=0.0, a=1.0, b=-2.0, d=5.0, s=6.0, Iext2=0.45, gamma=0.01,
                  tau1=1.0, tau0=2857.0, tau2=10.0):

        x1 = numpy.array(x1)

        f = numpy.empty((model_vars, x0.size), dtype=x1.dtype)

        if model_vars == 2:

            f[0, :] = calc_fx1_2d(x1, z, yc, Iext1=Iext1, slope=slope, a=a, b=b, tau1=tau1, x1_neg=x1_neg)
            iz = 1

        else:
            iz = 2

            f[1, :] = calc_fy1(x1, yc, y1, d, tau1)
            f[5, :] = calc_fg(x1, g, gamma, tau1)

            if model_vars == 6:
                f[0, :] = calc_fx1_6d(x1, z, y1, x2, Iext1, slope, a, b, tau1, x1_neg)
                f[3, :], f[4, :] = calc_fpop2(x2, y2, z, g, Iext2, s, tau1, tau2, x2_neg)

            elif model_vars == 11:
                f[0, :] = calc_fx1_6d(x1, z, y1, x2, Iext1_var, slope_var, a, b, tau1, x1_neg)
                f[3, :], f[4, :] = calc_fpop2(x2, y2, z, g, Iext2_var, s, tau1, tau2, x2_neg)
                f[6, :], f[7, :], f[8, :], f[9, :], f[10, :] = calc_fparams_var(x0_var, slope_var, Iext1_var, Iext2_var,
                                                                                K_var, x0, slope, Iext1, Iext2, K, z,
                                                                                g, tau1, tau0, pmode)
                x0 = x0_var
                K = K_var

        f[iz, :] = calc_fz(x1, x0, x0cr, r, z, K, w, tau1, tau0, zmode)

        return f.astype(x1.dtype)


    def calc_jac(x1, z, yc, Iext1, x0, x0cr, r, K, w, model_vars=2, zmode="lin", pmode="const", x1_neg=None,
                  y1=None, x2=None, y2=None, g=None, x2_neg=None,
                  x0_var=None, slope_var=None, Iext1_var=None, Iext2_var=None, K_var=None,
                  slope=0.0, a=1.0, b=-2.0, d=5.0, s=6.0, Iext2=0.45, gamma=0.01,
                  tau1=1.0, tau0=2857.0, tau2=10.0):

        from tvb_epilepsy.base.symbolic_equations import sym_vars

        if isinstance(x1, (float, int, long, complex, numpy.ndarray)):
            n_regions = x1.size
        else:
            x1 = len(x1)

        if model_vars == 2:

            x = [sym_vars(n_regions, 'x1'), sym_vars(n_regions, 'z')]

            dfun_sym = calc_dfun(x[0], x[1], yc, Iext1, x0, x0cr, r, K, w, 2, zmode, pmode, x1_neg,
                                 y1, x2, y2, g, x2_neg,
                                 x0_var, slope_var, Iext1_var, Iext2_var, K_var,
                                 slope, a, b, d, s, Iext2, gamma,
                                 tau1, tau0, tau2)

            jac_sym = dfun_sym.jacobian(Matrix(x))

            jac_lambda = lambdify(x, jac_sym, "numpy")

            return numpy.array(jac_lambda([x1, z]), dtype=x1.type)


        elif model_vars == 6:

            x = [sym_vars(n_regions, 'x1'), sym_vars(n_regions, 'y1'), sym_vars(n_regions, 'z'),
                 sym_vars(n_regions, 'x2'), sym_vars(n_regions, 'y2'), sym_vars(n_regions, 'g')]

            dfun_sym = calc_dfun(x[0], x[2], yc, Iext1, x0, x0cr, r, K, w, 2, zmode, pmode, x1_neg,
                                 x[1], x[3], x[4], x[5], x2_neg,
                                 x0_var, slope_var, Iext1_var, Iext2_var, K_var,
                                 slope, a, b, d, s, Iext2, gamma,
                                 tau1, tau0, tau2)

            jac_sym = dfun_sym.jacobian(Matrix(x))

            jac_lambda = lambdify(x, jac_sym, "numpy")

            return numpy.array(jac_lambda([x1, y1, z, x2, y2, g]), dtype=x1.type)

        elif model_vars == 11:

            x = [sym_vars(n_regions, 'x1'), sym_vars(n_regions, 'y1'), sym_vars(n_regions, 'z'),
                 sym_vars(n_regions, 'x2'), sym_vars(n_regions, 'y2'), sym_vars(n_regions, 'g'),
                 sym_vars(n_regions, 'x0_var'), sym_vars(n_regions, 'slope_var'), sym_vars(n_regions, 'Iext1_var'),
                 sym_vars(n_regions, 'Iext2_var'), sym_vars(n_regions, 'K_var')]

            dfun_sym = calc_dfun(x[0], x[2], yc, Iext1, x0, x0cr, r, K, w, 2, zmode, pmode, x1_neg,
                                 x[1], x[3], x[4], x[5], x2_neg,
                                 x[6], x[7], x[8], x[9], x[10],
                                 slope, a, b, d, s, Iext2, gamma,
                                 tau1, tau0, tau2)

            jac_sym = dfun_sym.jacobian(Matrix(x))

            jac_lambda = lambdify(x, jac_sym, "numpy")

            return numpy.array(jac_lambda([x1, y1, z, x2, y2, g, x0_var, slope_var, Iext1_var, Iext2_var, K_var]),
                                   dtype=x1.type)
