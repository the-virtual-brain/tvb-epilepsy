import numpy


def calc_coupling(x1, K, w, ix=None, jx=None):

    # Only difference coupling for the moment.
    # TODO: Extend for different coupling forms

    n_regions = x1.size

    if ix is None:
        ix = range(n_regions)

    if jx is None:
        jx = range(n_regions)

    i_n = numpy.ones((len(ix), 1), dtype='float32')
    j_n = numpy.ones((len(jx), 1), dtype='float32')

    x1 = numpy.expand_dims(x1.squeeze(),1).T
    K = numpy.reshape(K, x1.shape)

    # Coupling                                                     from                        to
    return K[:, ix]*numpy.sum(numpy.dot(w[ix][:, jx], numpy.dot(i_n, x1[:, jx]) - numpy.dot(j_n, x1[:, ix]).T), axis=1)


def calc_x0(x1, z, x0cr, rx0, coupl, zmode=numpy.array("lin")):

    if zmode == 'lin':
        return numpy.array((x1 + x0cr - (z+coupl) / 4.0) / rx0, dtype="float32")

    elif zmode == 'sig':
        return numpy.array((3.0 / (1.0 + numpy.exp(-10.0 * (x1 + 0.5))) + x0cr - z + coupl) / rx0, dtype="float32")

    else:
        raise ValueError('zmode is neither "lin" nor "sig"')



def calc_fx1_6d(x1, z=0.0, y1=0.0, x2=0.0, Iext1=0.0, slope=0.0, a=1.0, b=3.0, tau1=1.0, x1_neg=None):

    # if_ydot0 = - self.a * y[0] ** 2 + self.b * y[0]
    if_ydot0 = - a * x1 ** 2 + b * x1  # self.a=1.0, self.b=3.0

    # else_ydot0 = self.slope - y[3] + 0.6 * (y[2] - 4.0) ** 2
    else_ydot0 = slope - x2 + 0.6 * (z - 4.0) ** 2

    if x1_neg is None:
        x1_neg = x1 < 0.0

    return tau1 * (y1 - z + Iext1 + numpy.where(x1_neg, if_ydot0, else_ydot0) * x1)


def calc_fx1_2d(x1, z=0, yc=0.0, Iext1=0.0, slope=0.0, a=1.0, b=-2.0, tau1=1.0, x1_neg=None):

    # if_ydot0 = - self.a * y[0] ** 2 + self.b * y[0]
    if_ydot0 = - a * x1 ** 2 + b * x1  # self.a=1.0, self.b=3.0

    # else_ydot0 = self.slope - y[3] + 0.6 * (y[2] - 4.0) ** 2
    else_ydot0 = slope - 5.0*x1 + 0.6 * (z - 4.0) ** 2

    if x1_neg is None:
        x1_neg = x1 < 0.0

    return tau1 * (yc - z + Iext1 + numpy.where(x1_neg, if_ydot0, else_ydot0) * x1)


def calc_fy1(x1, yc, y1=0, d=5.0, tau1=1.0):
    return tau1 * (yc - d * x1 ** 2 - y1)


def calc_fz_lin(x1, x0, x0cr, r, z=0, coupl=0, tau0=1.0):
    return (4 * (x1 - r * x0 + x0cr) - z - coupl) / tau0


def calc_fz_sig(x1, x0, x0cr, r, z=0, coupl=0, tau0=1.0):
    return (3/(1 + numpy.exp(-10 * (x1 + 0.5))) - r * x0 + x0cr - z - coupl) / tau0


def calc_fpop2(x2, y2=0.0, g=0.0, z=0.0, Iext2=0.45, s=6.0, tau1=1.0, tau2=10.0, x2_neg=None):

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

    return fx2, fy2


def calc_fg(x1, g=0, gamma=0.01, tau1=1.0):
    #ydot[5] = self.tt * (-0.01 * (y[5] - 0.1 * y[0]))
    return -tau1 * gamma * (g - 0.1 * x1)


def calc_dfun(x1, z, yc, Iext1, x0, x0cr, rx0, K, w, model_vars=2, zmode="lin", pmode="const", x1_neg=None,
              y1=None, x2=None, y2=None, g=None, x2_neg=None,
              x0_var=None, slope_var=None, Iext1_var=None, Iext2_var=None, K_var=None,
              slope=0.0, a=1.0, b=-2.0, d=5.0, s=6.0, Iext2=0.45, gamma=0.01,
              tau1=1.0, tau0=2857.0, tau2=10.0):

    n_regions = x0.size
    x0_type = x0.dtype

    f = numpy.zeros((model_vars, n_regions), dtype=x0_type)

    if model_vars == 2:

        f[0,:] = calc_fx1_2d(x1, z, yc, Iext1=Iext1, slope=slope, a=a, b=b, tau1=tau1, x1_neg=x1_neg)
        iz = 1

    else:
        iz = 2

        f[1, :] = calc_fy1(x1, yc, y1, d, tau1)
        f[5, :] = calc_fg(x1, g, gamma, tau1)

        if model_vars == 6:
            f[0, :] = calc_fx1_6d(x1, z, y1, x2, Iext1, slope, a, b, tau1, x1_neg)
            f[3, :], f[4, :] = calc_fpop2(x2, y2, g, z, Iext2, s, tau1, tau2, x2_neg)

        elif model_vars == 11:
            f[0, :] = calc_fx1_6d(x1, z, y1, x2, Iext1_var, slope_var, a, b, tau1, x1_neg)
            f[3, :], f[4, :] = calc_fpop2(x2, y2, g, z, Iext2_var, s, tau1, tau2, x2_neg)
            from tvb_epilepsy.tvb_api.epileptor_models import EpileptorDPrealistic
            #ydot[6] = self.tau1 * (-y[6] + self.x0)
            f[6, :] = tau1 * (-x0_var + x0)
            slope_eq, Iext2_eq = EpileptorDPrealistic.fun_slope_Iext2(z, g, pmode, slope, Iext2)
            # slope
            #ydot[7] = 10 * self.tau1 * (-y[7] + slope_eq)  # 5*
            f[7, :] = 10 * tau1 * (-slope_var + slope_eq)
            # Iext1
            #ydot[8] = self.tau1 * (-y[8] + self.Iext1) / self.tau0
            f[8, :] = tau1 * (-Iext1_var + Iext1) / tau0
            # Iext2
            #ydot[9] = 5 * self.tau1 * (-y[9] + Iext2_eq)
            f[9, :] = 5 * tau1 * (-Iext2_var + Iext2_eq)
            # K
            #ydot[10] = self.tau1 * (-y[10] + self.K) / self.tau0
            f[10,:] = tau1 * (-K_var + K) / tau0
            x0 = x0_var
            K = K_var

    if zmode == "lin":
        f[iz, :] = calc_fz_lin(x1, x0, x0cr, rx0, z, calc_coupling(x1, K, w), tau0)
    elif zmode == "sig":
        f[iz, :] = calc_fz_sig(x1, x0, x0cr, rx0, z, calc_coupling(x1, K, w), tau0)
    else:
        raise ValueError('zmode is neither "lin" nor "sig"')

    return f


# def calc_dfun_jac_2d_lin(x1, z, yc, Iext1, x0, x0cr, rx0, K, w, x1_neg=True,
#                          slope=0.0, tau1=1.0, tau0=2857.0):
#
#     # Define the symbolic variables we need:
#     x1, z = symbols('x1 z')
#     f1 = calc_fx1_2d(x1, z, yc, Iext1=Iext1, slope=slope, a=1.0, b=-2.0, tau1=tau1, x1_neg=x1_neg)
#     fz = calc_fz_lin(x1, x0, x0cr, rx0, z, calc_coupling(x1, K, w), tau0)
#
#     r = x0.shape[0]
#     c = x0.shape[1]
#     x1v = MatrixSymbol('x1v', r, c)
#     zv = MatrixSymbol('zv', r, c)
#
#     for ii in range(x0.size):
#         f1x1 = diff(f1, x1)
#         f1z = diff(f1, z)
#         fzx1 = diff(fz, x1)
#         fzz = diff(fz, z)
#
#     jac = numpy.concatenate([numpy.hstack([f1x1, f1z]),numpy.hstack([fzx1, fzz])],axis=0)
#     jac = lambdify((x1, z), jac, 'numpy')
#
#     return jac