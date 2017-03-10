import numpy
from sympy import Symbol, symbols, exp, solve, lambdify, MatrixSymbol, diff


def eqtn_coupling(n, ix=None, jx=None, K="K"):

    # Only difference coupling for the moment.
    # TODO: Extend for different coupling forms

    x1 = numpy.array([Symbol('x1_%d' % i_n) for i_n in range(n)])
    K = numpy.array([Symbol(K+'_%d' % i_n) for i_n in range(n)])
    w = []
    for i_n in range(n):
        w.append([Symbol('w_%d_%d' % (i_n, j_n)) for j_n in range(n)])
    w = numpy.array(w)

    if ix is None:
        ix = range(n)

    if jx is None:
        jx = range(n)

    i_n = numpy.ones((len(ix), 1))
    j_n = numpy.ones((len(jx), 1))

    x1 = numpy.expand_dims(x1.squeeze(), 1).T
    K = numpy.reshape(K, x1.shape)

    # Coupling                                                         from           to
    coupling = (K[:, ix]*numpy.sum(numpy.dot(w[ix][:, jx], numpy.dot(i_n, x1[:, jx])
                                                                      - numpy.dot(j_n, x1[:, ix]).T), axis=1)).tolist()

    return lambdify([x1, K, w], coupling, "numpy"), coupling


def eqtn_x0(n, zmode=numpy.array("lin")):

    x1 = numpy.array([Symbol('x1_%d' % i_n) for i_n in range(n)])
    z = numpy.array([Symbol('z_%d' % i_n) for i_n in range(n)])
    x0cr = numpy.array([Symbol('x0cr_%d' % i_n) for i_n in range(n)])
    r = numpy.array([Symbol('r_%d' % i_n) for i_n in range(n)])
    coupl = numpy.array([Symbol('coupl_%d' % i_n) for i_n in range(n)])

    if zmode == 'lin':
        x0 = (x1 + x0cr - (z+coupl) / 4.0) / r

    elif zmode == 'sig':
        x0 = (3.0 / (1.0 + numpy.exp(1) ** (-10.0 * (x1 + 0.5))) + x0cr - z + coupl) / r

    else:
        raise ValueError('zmode is neither "lin" nor "sig"')

    return lambdify([x1, z, x0cr, r, coupl], x0, "numpy"), x0


def eqtn_fx1_6d(n, x1_neg=True, slope="slope", Iext1="Iext1"):

    a, b, tau1 = symbols('a b tau1')

    x1 = numpy.array([Symbol('x1_%d' % i_n) for i_n in range(n)])
    z = numpy.array([Symbol('z_%d' % i_n) for i_n in range(n)])
    x2 = numpy.array([Symbol('x_%d' % i_n) for i_n in range(n)])
    y1 = numpy.array([Symbol('y1_%d' % i_n) for i_n in range(n)])
    slope = numpy.array([Symbol(slope+'_%d' % i_n) for i_n in range(n)])
    Iext1 = numpy.array([Symbol(Iext1+'_%d' % i_n) for i_n in range(n)])

    # if_ydot0 = - self.a * y[0] ** 2 + self.b * y[0]
    if_ydot0 = - a * x1 ** 2 + b * x1  # self.a=1.0, self.b=-2.0

    # else_ydot0 = self.slope - y[3] + 0.6 * (y[2] - 4.0) ** 2
    else_ydot0 = slope - x2 + 0.6 * (z - 4.0) ** 2

    fx1 = (tau1 * (y1 - z + Iext1 + numpy.where(x1_neg, if_ydot0, else_ydot0) * x1)).tolist()

    return lambdify([x1, z, y1, x2, Iext1, slope, a, b, tau1], fx1, "numpy"), fx1


def eqtn_fx1_2d(n, x1_neg=True):

    a, b, tau1 = symbols('a b tau1')

    x1 = numpy.array([Symbol('x1_%d' % i_n) for i_n in range(n)])
    z = numpy.array([Symbol('z_%d' % i_n) for i_n in range(n)])
    yc = numpy.array([Symbol('yc_%d' % i_n) for i_n in range(n)])
    slope = numpy.array([Symbol('slope_%d' % i_n) for i_n in range(n)])
    Iext1 = numpy.array([Symbol('Iext1_%d' % i_n) for i_n in range(n)])

    # if_ydot0 = - self.a * y[0] ** 2 + self.b * y[0]
    if_ydot0 = - a * x1 ** 2 + b * x1  # self.a=1.0, self.b=-2.0

    # else_ydot0 = self.slope - 5 * y[0] + 0.6 * (y[2] - 4.0) ** 2
    else_ydot0 = slope - 5.0 * x1 + 0.6 * (z - 4.0) ** 2

    fx1 = (tau1 * (yc - z + Iext1 + numpy.where(x1_neg, if_ydot0, else_ydot0) * x1)).tolist()

    return lambdify([x1, z, yc, Iext1, slope, a, b, tau1], fx1, "numpy"), fx1


def eqtn_fy1(n):

    d, tau1 = symbols('d tau1')

    x1 = numpy.array([Symbol('x1_%d' % i_n) for i_n in range(n)])
    y1 = numpy.array([Symbol('y1_%d' % i_n) for i_n in range(n)])
    yc = numpy.array([Symbol('yc_%d' % i_n) for i_n in range(n)])

    fy1 = (tau1 * (yc - d * x1 ** 2 - y1)).tolist()

    return lambdify([x1, y1, yc, d, tau1], fy1, "numpy"), fy1


def eqtn_fz(n, zmode=numpy.array("lin"), x0="x0"):

    tau1, tau0 = symbols('tau1 tau2')

    x1 = numpy.array([Symbol('x1_%d' % i_n) for i_n in range(n)])
    z = numpy.array([Symbol('z_%d' % i_n) for i_n in range(n)])
    x0 = numpy.array([Symbol(x0+'_%d' % i_n) for i_n in range(n)])
    x0cr = numpy.array([Symbol('x0cr_%d' % i_n) for i_n in range(n)])
    coupl = numpy.array([Symbol('coupl_%d' % i_n) for i_n in range(n)])
    r = numpy.array([Symbol('r_%d' % i_n) for i_n in range(n)])

    if zmode == 'lin':
        fz = (tau1 * (4 * (x1 - r * x0 + x0cr) - z - coupl) / tau0).tolist()

    elif zmode == 'sig':
        fz = (tau1 * (3/(1 + numpy.exp(1.0) ** (-10.0 * (x1 + 0.5))) - r * x0 + x0cr - z - coupl) / tau0).tolist()
    else:
        raise ValueError('zmode is neither "lin" nor "sig"')

    return lambdify([x1, z, x0, x0cr, r, coupl, tau1, tau0], fz, "numpy"), fz


def eqtn_fpop2(n, x2_neg=True, Iext2="Iext2"):

    s, tau1, tau2 = symbols('s tau1 tau0')

    x2 = numpy.array([Symbol('x2_%d' % i_n) for i_n in range(n)])
    y2 = numpy.array([Symbol('y2_%d' % i_n) for i_n in range(n)])
    z = numpy.array([Symbol('z_%d' % i_n) for i_n in range(n)])
    g = numpy.array([Symbol('g_%d' % i_n) for i_n in range(n)])
    Iext2 = numpy.array([Symbol(Iext2+'%d' % i_n) for i_n in range(n)])

    # ydot[3] = self.tt * (-y[4] + y[3] - y[3] ** 3 + self.Iext2 + 2 * y[5] - 0.3 * (y[2] - 3.5) + self.Kf * c_pop2)
    fx2 = tau1 * (-y2 + x2 - x2 ** 3 + Iext2 + 2 * g - 0.3 * (z - 3.5))

    # if_ydot4 = 0
    if_ydot4 = 0
    # else_ydot4 = self.aa * (y[3] + 0.25)
    else_ydot4 = s * (x2 + 0.25)  # self.s = 6.0

    # ydot[4] = self.tt * ((-y[4] + where(y[3] < -0.25, if_ydot4, else_ydot4)) / self.tau)
    fy2 = (tau1 * ((-y2 + numpy.where(x2_neg, if_ydot4, else_ydot4)) / tau2)).tolist()

    return [lambdify([x2, y2, z, g, Iext2, tau1], fx2, "numpy"), \
            lambdify([x2, y2, s, tau1, tau2], fy2, "numpy")], [fx2, fy2]


def eqtn_fg(n):

    gamma, tau1 = symbols('gamma tau1')

    x1 = numpy.array([Symbol('x1_%d' % i_n) for i_n in range(n)])
    g = numpy.array([Symbol('g_%d' % i_n) for i_n in range(n)])

    #ydot[5] = self.tt * (-0.01 * (y[5] - 0.1 * y[0]))
    fg =(-tau1 * gamma * (g - 0.1 * x1)).tolist()

    return lambdify([x1, g, gamma, tau1], fg, "numpy"), fg


def eqtn_fparam_vars(n, pmode=numpy.array("const")):

    tau1, tau0 = symbols('tau1 tau0')

    z = numpy.array([Symbol('z_%d' % i_n) for i_n in range(n)])
    g = numpy.array([Symbol('g_%d' % i_n) for i_n in range(n)])
    x0_var = numpy.array([Symbol('x0_var_%d' % i_n) for i_n in range(n)])
    slope_var = numpy.array([Symbol('slope_var_%d' % i_n) for i_n in range(n)])
    Iext1_var = numpy.array([Symbol('Iext1_var_%d' % i_n) for i_n in range(n)])
    Iext2_var= numpy.array([Symbol('Iext2_var_%d' % i_n) for i_n in range(n)])
    K_var = numpy.array([Symbol('K_var_%d' % i_n) for i_n in range(n)])
    x0 = numpy.array([Symbol('x0_%d' % i_n) for i_n in range(n)])
    slope = numpy.array([Symbol('slope_%d' % i_n) for i_n in range(n)])
    Iext1 = numpy.array([Symbol('Iext1_%d' % i_n) for i_n in range(n)])
    Iext2 = numpy.array([Symbol('Iext2_%d' % i_n) for i_n in range(n)])
    K = numpy.array([Symbol('K_%d' % i_n) for i_n in range(n)])

    #ydot[5] = self.tt * (-0.01 * (y[5] - 0.1 * y[0]))
    fx0 =(tau1 * (-x0_var + x0)).tolist()

    from tvb_epilepsy.tvb_api.epileptor_models import EpileptorDPrealistic
    slope_eq, Iext2_eq = EpileptorDPrealistic.fun_slope_Iext2(z, g, pmode, slope, Iext2)

    # slope
    # ydot[7] = 10 * self.tau1 * (-y[7] + slope_eq)
    fslope = (10.0 * tau1 * (-slope_var + slope_eq)).tolist()
    # Iext1
    # ydot[8] = self.tau1 * (-y[8] + self.Iext1) / self.tau0
    fIext1 = (tau1 * (-Iext1_var + Iext1) / tau0).tolist()
    # Iext2
    # ydot[9] = 5 * self.tau1 * (-y[9] + Iext2_eq)
    fIext2 = (5.0 * tau1 * (-Iext2_var + Iext2_eq)).tolist()
    # K
    # ydot[10] = self.tau1 * (-y[10] + self.K) / self.tau0
    fK = (tau1 * (-K_var + K) / tau0).tolist()

    return [lambdify([x0, x0_var, tau1], fx0, "numpy"), \
            lambdify([slope, slope_var, tau1], fslope, "numpy"), \
            lambdify([Iext1, Iext1_var, tau1, tau0], fIext1, "numpy"), \
            lambdify([Iext2, Iext2_var, tau1], fIext2, "numpy"), \
            lambdify([K, K_var, tau1, tau0], fK, "numpy")],[fx0, fslope, fIext1, fIext2, fK]


def eqnt_dfun(n_regions, model_vars, zmode=numpy.array("lin"), x1_neg=True, x2_neg=True, pmode=numpy.array("const")):

    f_lambda = []
    f_sym = []

    if model_vars == 2:

        fl, fs = eqtn_fx1_2d(n_regions, x1_neg)
        f_lambda += fl
        f_sym += fs

        fl, fs = eqtn_fz(n_regions, zmode)
        f_lambda += fl
        f_sym += fs

    elif model_vars == 6:

        fl, fs = eqtn_fx1_6d(n_regions, x1_neg)
        f_lambda += fl
        f_sym += fs

        fl, fs = eqtn_fy1(n_regions)
        f_lambda += fl
        f_sym += fs

        fl, fs = eqtn_fpop2(n_regions, x2_neg)
        f_lambda += fl
        f_sym += fs

        fl, fs = eqtn_fg(n_regions)
        f_lambda += fl
        f_sym += fs

    elif model_vars == 11:

        fl, fs = eqtn_fx1_6d(n_regions, x1_neg, "slope_var", "Iext1_var")
        f_lambda += fl
        f_sym += fs

        fl, fs = eqtn_fy1(n_regions)
        f_lambda += fl
        f_sym += fs

        fl, fs = eqtn_fpop2(n_regions, x2_neg, "Iext2_var")
        f_lambda += fl
        f_sym += fs

        fl, fs = eqtn_fparam_vars(n_regions, pmode)
        f_lambda += fl
        f_sym += fs

    return f_lambda, f_sym


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