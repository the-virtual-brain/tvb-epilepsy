import numpy
from numpy import array, empty, ones, zeros, multiply, dot, power, divide, sum, exp, reshape, concatenate, diag, \
                  where, argmax, repeat


def if_ydot0(x1, a, b):
    # if_ydot0 = - self.a * y[0] ** 2 + self.b * y[0]
    return multiply(pow(x1, 2), a) + multiply(x1, b)


def else_ydot0_6d(x2, z, slope):
    # else_ydot0 = self.slope - y[3] + 0.6 * (y[2] - 4.0) ** 2
    return slope - x2 + 0.6 * power(z - 4.0, 2)


def else_ydot0_2d(x1, z, slope):
    # else_ydot0 = self.slope - 5 * y[0] + 0.6 * (y[2] - 4.0) ** 2
    return slope - 5.0 * x1 + 0.6*power(z - 4.0, 2)


def eqtn_coupling(x1, K, w, ix, jx):

    from tvb_epilepsy.base.utils import assert_arrays

    # Only difference coupling for the moment.
    # TODO: Extend for different coupling forms

    shape = x1.shape

    x1, K = assert_arrays([x1, K],  (1, x1.size))

    i_n = ones((len(ix), 1), dtype='float32')
    j_n = ones((len(jx), 1), dtype='float32')

    # Coupling
    #                                                            from (jx)                to (ix)
    coupling = multiply(K[:, ix], sum(multiply(w[ix][:, jx], dot(i_n, x1[:, jx]) - dot(j_n, x1[:, ix]).T), axis=1))

    return reshape(coupling, shape)


def eqtn_x0(x1, z, model="2d", zmode=array("lin"), z_pos=True, K=None, w=None, coupl=None, x0cr=None, r=None):

    if coupl is None:
        if numpy.all(K == 0.0) or numpy.all(w == 0.0) or (K is None) or (w is None):
            coupl = 0.0
        else:
            from tvb_epilepsy.base.calculations import calc_coupling
            coupl = calc_coupling(x1, K, w)

    if model == "2d":
        if zmode == 'lin':
            return divide((x1 + x0cr - -z - (where(z_pos, z, 0.1 * power(z, 7.0)) + coupl) / 4.0), r)

        elif zmode == 'sig':
            return divide((divide(3.0 / (1.0 + power(exp(1), -10.0 * (x1 + 0.5)))) + x0cr - z - coupl), r)

        else:
            raise ValueError('zmode is neither "lin" nor "sig"')

    else:
        if zmode == 'lin':
            return x1 - (z + where(z_pos, z, 0.1 * power(z, 7.0)) + coupl) / 4.0

        elif zmode == 'sig':
            return divide(3.0, (1.0 + power(exp(1), -10.0 * (x1 + 0.5)))) - z - coupl

        else:
            raise ValueError('zmode is neither "lin" nor "sig"')



def eqtn_fx1(x1, z, y1, Iext1, slope, a, b, tau1, x1_neg, model="2d", x2=None):

    if model == "2d":
        return multiply(y1 - z + Iext1 + multiply(x1, where(x1_neg, if_ydot0(x1, a, b), else_ydot0_2d(x1, z, slope))),
                        tau1)
    else:
        return multiply(y1 - z + Iext1 + multiply(x1, where(x1_neg, if_ydot0(x1, a, b), else_ydot0_6d(x2, z, slope))),
                        tau1)


def eqtn_fx1_2d_taylor_lin(x1, x_taylor, z, yc, Iext1, a, b, tau1):

    return multiply(Iext1+ 2 * multiply(power(x_taylor, 3), a) - multiply(power(x_taylor, 2), b) + yc - z +
                    multiply(x1, (-3 * multiply(power(x_taylor, 2), a) + 2 * multiply(x_taylor, b))), tau1)


def eqtn_jac_x1_2d(x1, z, slope, a, b, tau1, x1_neg):


    jac_x1 = numpy.multiply(numpy.where(x1_neg, multiply(-3.0 * multiply(a, x1) - 2.0 * multiply(b, x1), x1),
                                 else_ydot0_2d(x1, z, slope)), tau1)

    jac_z = numpy.multiply(-ones(x1.shape, dtype=x1.dtype), tau1)

    shape = array(x1.shape)
    shape[argmax(shape)] *= 2
    shape = tuple(shape)
    return reshape(concatenate([jac_x1.flatten(), jac_z.flatten()]), shape)



def eqtn_fy1(x1, yc, y1, d, tau1):

    return multiply((yc - multiply(pow(x1, 2), d) - y1), tau1)


def eqtn_fz(x1, z, x0, tau1, tau0, model="2d", zmode=array("lin"), z_pos=True, K=None, w=None, coupl=None, x0cr=None,
            r=None):

    if coupl is None:
        if numpy.all(K == 0.0) or numpy.all(w == 0.0) or (K is None) or (w is None):
            coupl = 0.0
        else:
            from tvb_epilepsy.base.calculations import calc_coupling
            coupl = calc_coupling(x1, K, w)

    if model == "2d":

        if zmode == 'lin':
            return divide(multiply((4 * (x1 - multiply(r, x0) + x0cr) - z - where(z_pos, z, 0.1 * power(z, 7.0))
                                    - coupl), tau1), tau0)

        elif zmode == 'sig':
            return divide(multiply((4 * divide(3.0, (1 + multiply(exp(1), (-10.0 * (x1 + 0.5))))
                          - multiply(r, x0) + x0cr) - z - coupl), tau1), tau0)
        else:
            raise ValueError('zmode is neither "lin" nor "sig"')

    else:

        if zmode == 'lin':
            return divide(multiply((4 * (x1 - x0) - z - where(z_pos, z, 0.1 * power(z, 7.0)) - coupl), tau1), tau0)

        elif zmode == 'sig':
            return divide(multiply((4 * divide(3.0, (1 + multiply(exp(1), (-10.0 * (x1 + 0.5)))) - x0)
                           - z - coupl), tau1), tau0)
        else:
            raise ValueError('zmode is neither "lin" nor "sig"')


def eqtn_jac_fz_2d(x1, z, tau1, tau0, zmode=array("lin"), z_pos=True, K=None, w=None):

    tau = repeat(divide(tau1, tau0).flatten(), (2,))

    jac_z = - ones(z.shape, dtype=z.dtype)

    if zmode == 'lin':

        jac_x1 = 4.0 * ones(z.shape, dtype=z.dtype)

        if not(z_pos):
            jac_z -= 0.7 * power(z, 6.0)

    elif zmode == 'sig':
        jac_x1 = divide(30 * multiply(exp(1), (-10.0 * (x1 + 0.5))), (1 + multiply(exp(1), (-10.0 * (x1 + 0.5))))) - \
                 multiply(K, sum(w, axis=1))
    else:
        raise ValueError('zmode is neither "lin" nor "sig"')

    shape = array(z.shape)
    shape[argmax(shape)] *= 2
    shape = tuple(shape)

    return reshape(multiply(concatenate([jac_x1.flatten(), jac_z.flatten()]), tau.flatten()),  shape)


def eqtn_fx1z_2d_zpos_jac(x1, r, K, w, ix0, iE, a, b, tau1, tau0): #

    from tvb_epilepsy.base.utils import assert_arrays
    p = x1.shape

    x1, r, K, a, b, tau1, tau0 = assert_arrays([x1, r, K, a, b, tau1, tau0], (1, x1.size))

    no_x0 = len(ix0)
    no_e = len(iE)

    i_x0 = ones((no_x0, 1), dtype="float32")
    i_e = ones((no_e, 1), dtype="float32")

    tau = divide(tau1, tau0)

    jac_e_x0e = diag(multiply(tau[:,iE],(- 4 * r[:, iE])).flatten())
    jac_e_x1o = -dot(dot(i_e, multiply(tau[:,iE], K[:, iE])), w[iE][:, ix0])
    jac_x0_x0e = zeros((no_x0, no_e), dtype="float32")
    jac_x0_x1o = (diag(multiply(tau[:,ix0],
                                (4 + 3 * multiply(a[:, ix0], power(x1[:, ix0], 2))
                                 - 2 * multiply(b[:, ix0],  x1[:, ix0]) +
                                 multiply(K[:, ix0], sum(w[ix0], axis=1)))).flatten()) -
                  multiply(dot(i_x0, multiply(tau[:,ix0], K[:, ix0])).T,  w[ix0][:, ix0]))

    jac = empty((x1.size, x1.size), dtype=type(jac_e_x0e))
    jac[numpy.ix_(iE, iE)] = jac_e_x0e
    jac[numpy.ix_(iE, ix0)] = jac_e_x1o
    jac[numpy.ix_(ix0, iE)] = jac_x0_x0e
    jac[numpy.ix_(ix0, ix0)] = jac_x0_x1o

    return jac


def eqtn_fx1y1_6d_diff_x1(x1, a, b, d, tau1):

    return multiply(multiply(-3 * multiply(x1, a) + 2 * (b - d), x1), tau1)


def eqtn_fx2(x2, y2, z, g, Iext2, tau1):

    # ydot[3] = self.tt * (-y[4] + y[3] - y[3] ** 3 + self.Iext2 + 2 * y[5] - 0.3 * (y[2] - 3.5) + self.Kf * c_pop2)
    return multiply(-y2 + x2 - power(x2, 3) + Iext2 + 2 * g - 0.3 * (z - 3.5), tau1)


def eqtn_fy2(x2, y2, s, tau1, tau2, x2_neg=True):

    # ydot[4] = self.tt * ((-y[4] + where(y[3] < -0.25, if_ydot4, else_ydot4)) / self.tau)
    return divide(multiply(-y2 + where(x2_neg, 0.0, multiply(x2 + 0.25, s)), tau1), tau2)


def eqtn_fg(x1, g, gamma, tau1):

    return multiply(multiply(g - 0.1 * x1, gamma), tau1)


def eqtn_fx0(x0_var, x0, tau1):
    # ydot[6] = self.tau1 * (-y[6] + self.x0)
    return multiply(-x0_var + x0, tau1)


def eqtn_fslope(slope_var, slope, tau1):
    # ydot[7] = 10 * self.tau1 * (-y[7] + slope_eq)
    return 10.0 * multiply(-slope_var + slope, tau1)


def eqtn_fIext1(Iext1_var, Iext1, tau1, tau0):
    # ydot[8] = self.tau1 * (-y[8] + self.Iext1) / self.tau0
    return divide(multiply(-Iext1_var + Iext1, tau1), tau0)


def eqtn_fIext2(Iext2_var, Iext2, tau1):
    # ydot[9] = 5 * self.tau1 * (-y[9] + Iext2_eq)
    return 5.0 * multiply(-Iext2_var + Iext2, tau1)


def eqtn_fK(K_var, K, tau1, tau0):
    # ydot[10] = self.tau1 * (-y[10] + self.K) / self.tau0
    return divide(multiply(-K_var + K, tau1), tau0)


def eqtn_fparams_vars(x0_var, slope_var, Iext1_var, Iext2_var, K_var, x0, slope, Iext1, Iext2, K, tau1, tau0,
                      pmode="const", z=None, g=None):

    fx0 = eqtn_fx0(x0_var, x0, tau1)

    from tvb_epilepsy.tvb_api.epileptor_models import EpileptorDPrealistic
    slope_eq, Iext2_eq = EpileptorDPrealistic.fun_slope_Iext2(z, g, pmode, slope, Iext2)

    fslope = eqtn_fslope(slope_var, slope_eq, tau1)

    fIext1 = eqtn_fIext1(Iext1_var, Iext1, tau1, tau0)

    fIext2 = eqtn_fIext2(Iext2_var, Iext2_eq, tau1)

    fK = eqtn_fK(K_var, K, tau1, tau0)

    return fx0, fslope, fIext1, fIext2, fK


def eqtn_dfun(x1, z, yc, Iext1, x0, K, w, model_vars=2, x0cr=None, r=None, zmode="lin", pmode="const", x1_neg=True,
                  y1=None, x2=None, y2=None, g=None, x2_neg=True,
                  x0_var=None, slope_var=None, Iext1_var=None, Iext2_var=None, K_var=None,
                  slope=0.0, a=1.0, b=-2.0, d=5.0, s=6.0, Iext2=0.45, gamma=0.1,
                  tau1=1.0, tau0=2857.0, tau2=10.0):

    if model_vars == 2:

        fx1 = eqtn_fx1(x1, z, y1, Iext1, slope, a, b, tau1, x1_neg, model="2d", x2=None)

        fz = eqtn_fz(x1, z, x0, tau1, tau0, model="2d", zmode=zmode, z_pos=True, K=K, w=w, coupl=None, x0cr=x0cr, r=r)

        return fx1, fz

    elif model_vars == 6:

        fx1 = eqtn_fx1(x1, z, y1, Iext1, slope, a, b, tau1, x1_neg, model="6d", x2=x2)

        fy1 = eqtn_fy1(x1, yc, y1, d, tau1)

        fz = eqtn_fz(x1, z, x0, tau1, tau0, model="6d", zmode=zmode, z_pos=True, K=K, w=w, coupl=None, x0cr=None,
                     r=None)

        fx2 = eqtn_fx2(x2, y2, z, g, Iext2, tau1)

        fy2 = eqtn_fy2(x2, y2, s, tau1, tau2, x2_neg)

        fg = eqtn_fg(x1, g, gamma, tau1)

        return fx1, fy1, fz, fx2, fy2, fg

    elif model_vars == 11:

        fx1 = eqtn_fx1(x1, z, y1, Iext1_var, slope_var, a, b, tau1, x1_neg, model="6d", x2=x2)

        fy1 = eqtn_fy1(x1, yc, y1, d, tau1)

        fz = eqtn_fz(x1, z, x0, tau1, tau0, model="6d", zmode=zmode, z_pos=True, K=K_var, w=w, coupl=None, x0cr=None,
                     r=None)

        fx2 = eqtn_fx2(x2, y2, z, g, Iext2_var, tau1)

        fy2 = eqtn_fy2(x2, y2, s, tau1, tau2, x2_neg)

        fg = eqtn_fg(x1, g, gamma, tau1)

        fx0, fslope, fIext1, fIext2, fK = eqtn_fparams_vars(x0_var, slope_var, Iext1_var, Iext2_var, K_var, x0, slope,
                                                            Iext1, Iext2, K, tau1, tau0, pmode, z, g)

        return fx1, fy1, fz, fx2, fy2, fg, fx0, fslope, fIext1, fIext2, fK


def eqtn_jac_2d(x1, z, K, w, slope, a, b, tau1, tau0, zmode=array("lind"), x1_neg=True, z_pos=True):

    jac_x1 = eqtn_jac_x1_2d(x1, z, slope, a, b, tau1, x1_neg)

    jac_z = eqtn_jac_fz_2d(x1, z, tau1, tau0, zmode, z_pos, K, w)

    return jac_x1, jac_z