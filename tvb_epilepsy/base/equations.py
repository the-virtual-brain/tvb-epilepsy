import numpy as np
from tvb_epilepsy.base.utils import assert_arrays


def if_ydot0(x1, a, b):
    # if_ydot0 = - self.a * y[0] ** 2 + self.b * y[0]
    return -np.multiply(np.power(x1, 2), a) + np.multiply(x1, b)


def else_ydot0_6d(x2, z, slope):
    # else_ydot0 = self.slope - y[3] + 0.6 * (y[2] - 4.0) ** 2
    return slope - x2 + 0.6 * np.power(z - 4.0, 2)


def else_ydot0_2d(x1, z, slope):
    # else_ydot0 = self.slope - 5 * y[0] + 0.6 * (y[2] - 4.0) ** 2
    return slope - 5.0 * x1 + 0.6 * np.power(z - 4.0, 2)


def eqtn_coupling(x1, K, w, ix, jx):
    # Only difference coupling for the moment.
    # TODO: Extend for different coupling forms

    shape = x1.shape

    x1, K = assert_arrays([x1, K], (1, x1.size))

    i_n = np.ones((len(ix), 1), dtype='float32')
    j_n = np.ones((len(jx), 1), dtype='float32')

    # Coupling                 from (jx)                to (ix)
    coupling = np.multiply(K[:, ix],
                           sum(np.multiply(w[ix][:, jx], np.dot(i_n, x1[:, jx]) - np.dot(j_n, x1[:, ix]).T), axis=1))

    return np.reshape(coupling, shape)


def eqtn_coupling_diff(K, w, ix, jx):
    # Only difference coupling for the moment.
    # TODO: Extend for different coupling forms

    K = np.reshape(K, (K.size,))

    if K.dtype == "object" or w.dtype == "object":
        dtype = "object"
    else:
        dtype = K.dtype

    dcoupl_dx1 = np.empty((len(ix), len(jx)), dtype=dtype)

    for ii in ix:
        for ij in jx:

            if ii == ij:
                dcoupl_dx1[ii, ij] = -np.multiply(K[ii], sum(w[ii, jx]))
            else:
                dcoupl_dx1[ii, ij] = np.multiply(K[ii], w[ii, ij])

    return dcoupl_dx1


def eqtn_x0cr_r(Iext1, yc, a, b, x1_rest, x1_cr, x0_rest, x0_cr, zmode=np.array("lin")):

    if zmode == 'lin':

        return 0.25*(x0_rest*(a*x1_cr**3 - a*x1_rest**3 - b*x1_cr**2 + b*x1_rest**2 + 4.0*x1_cr - 4.0*x1_rest) +
                     (x0_cr - x0_rest)*(Iext1 - a*x1_rest**3 + b*x1_rest**2 - 4.0*x1_rest + yc))/(x0_cr - x0_rest), \
               0.25 * (a * x1_cr ** 3 - a * x1_rest ** 3 - b * x1_cr ** 2 + b * x1_rest ** 2 + 4.0 * x1_cr -
                       4.0 * x1_rest) / (x0_cr - x0_rest)

    elif zmode == 'sig':

        return (-x0_cr*(3.2e+66*20000000000000.0**(10*x1_cr) + 4.74922109128249e+68*54365636569181.0**(10*x1_cr))
                *(3.2e+66*1.024e+133**x1_rest*(Iext1 - a*x1_rest**3 + b*x1_rest**2 + yc)
                + 4.74922109128249e+68*2.25551009738825e+137**x1_rest*(Iext1 - a*x1_rest**3 + b*x1_rest**2 + yc - 3.0))
                + x0_rest*(3.2e+66*20000000000000.0**(10*x1_rest) +
                4.74922109128249e+68*54365636569181.0**(10*x1_rest))*(3.2e+66*1.024e+133**x1_cr*(Iext1 - a*x1_cr**3 +
                b*x1_cr**2 + yc) + 4.74922109128249e+68*2.25551009738825e+137**x1_cr*(Iext1 - a*x1_cr**3 + b*x1_cr**2 +
                yc - 3.0)))/((3.2e+66*20000000000000.0**(10.0*x1_cr) +
                4.74922109128249e+68*54365636569181.0**(10.0*x1_cr))*(3.2e+66*20000000000000.0**(10.0*x1_rest) +
                4.74922109128249e+68*54365636569181.0**(10.0*x1_rest))*(-x0_cr + x0_rest)), \
                (-(3.2e+66 * 20000000000000.0 ** (10 * x1_cr) +
                   4.74922109128249e+68 * 54365636569181.0 ** (10 * x1_cr)) * (3.2e+66 * 1.024e+133 ** x1_rest * (
                Iext1 - a * x1_rest ** 3 + b * x1_rest ** 2 + yc) +
                4.74922109128249e+68 * 2.25551009738825e+137 ** x1_rest * (
                Iext1 - a * x1_rest ** 3 + b * x1_rest ** 2 + yc - 3.0)) + (
                3.2e+66 * 20000000000000.0 ** (10 * x1_rest) + 4.74922109128249e+68 * 54365636569181.0 ** (
                10 * x1_rest)) * (3.2e+66 * 1.024e+133 ** x1_cr * (Iext1 - a * x1_cr ** 3 + b * x1_cr ** 2 + yc) +
                                     4.74922109128249e+68 * 2.25551009738825e+137 ** x1_cr * (
                                     Iext1 - a * x1_cr ** 3 + b * x1_cr ** 2 + yc - 3.0))) / \
                 ((3.2e+66 * 20000000000000.0 ** (10.0 * x1_cr) + 4.74922109128249e+68 * 54365636569181.0 ** (
                   10.0 * x1_cr)) * (3.2e+66 * 20000000000000.0 ** (10.0 * x1_rest) +
                                   4.74922109128249e+68 * 54365636569181.0 ** (10.0 * x1_rest)) * (-x0_cr + x0_rest))

    else:
        raise ValueError('zmode is neither "lin" nor "sig"')


def eqtn_x0(x1, z, model="2d", zmode=np.array("lin"), z_pos=True, K=None, w=None, coupl=None, x0cr=None, r=None):

    if coupl is None:
        if np.all(K == 0.0) or np.all(w == 0.0) or (K is None) or (w is None):
            coupl = 0.0
        else:
            from tvb_epilepsy.base.calculations import calc_coupling
            coupl = calc_coupling(x1, K, w)

    if model == "2d":
        if zmode == 'lin':
            return np.divide((x1 + x0cr - (np.where(z_pos, z, z + 0.1 * np.power(z, 7.0)) + coupl) / 4.0), r)

        elif zmode == 'sig':
            return np.divide(np.divide(3.0, 1.0 + np.power(np.exp(1), -10.0 * (x1 + 0.5))) + x0cr - z - coupl, r)

        else:
            raise ValueError('zmode is neither "lin" nor "sig"')

    else:
        if zmode == 'lin':
            return x1 - (z + np.where(z_pos, z, 0.1 * np.power(z, 7.0)) + coupl) / 4.0

        elif zmode == 'sig':
            return np.divide(3.0, 1.0 + np.power(np.exp(1), -10.0 * (x1 + 0.5))) - z - coupl

        else:
            raise ValueError('zmode is neither "lin" nor "sig"')


def eqtn_fx1(x1, z, y1, Iext1, slope, a, b, tau1, x1_neg=True, model="2d", x2=0.0):

    if model == "2d":
        return np.multiply(y1 - z + Iext1 + np.multiply(x1, np.where(x1_neg, if_ydot0(x1, a, b),
                                                                     else_ydot0_2d(x1, z, slope))),
                           tau1)
    else:
        return np.multiply(y1 - z + Iext1 + np.multiply(x1, np.where(x1_neg, if_ydot0(x1, a, b),
                                                                     else_ydot0_6d(x2, z, slope))),
                           tau1)


def eqtn_fx1_2d_taylor_lin(x1, x_taylor, z, yc, Iext1, a, b, tau1):

    return np.multiply(Iext1+ 2 * np.multiply(np.power(x_taylor, 3), a) - np.multiply(np.power(x_taylor, 2), b) + yc - z +
                       np.multiply(x1, (-3 * np.multiply(np.power(x_taylor, 2), a) + 2 * np.multiply(x_taylor, b))), tau1)


def eqtn_jac_x1_2d(x1, z, slope, a, b, tau1, x1_neg=True):

    jac_x1 = np.diag(
        np.multiply(np.where(x1_neg, np.multiply(-3.0 * np.multiply(a, x1) + 2.0 * np.multiply(b, 1.0), x1),
                             + else_ydot0_2d(x1, z, slope)), tau1).flatten())

    jac_z = - np.diag(np.multiply(np.ones(x1.shape, dtype=x1.dtype) +
                                  np.where(x1_neg, 0.0, 1.2 * np.multiply(z - 4.0, x1)), tau1).flatten())

    return np.concatenate([jac_x1, jac_z], axis=1)


def eqtn_fx1z_diff(x1, K, w, ix, jx, a, b, d, tau1, tau0, model="6d", zmode=np.array("lin")):  # , z_pos=True

    # TODO: for the extreme z_pos = False case where we have terms like 0.1 * z ** 7. See below eqtn_fz()
    # TODO: for the extreme x1_neg = False case where we have to solve for x2 as well

    shape = x1.shape

    x1, K, ix, jx, a, b, d, tau1, tau0 = assert_arrays([x1, K, ix, jx, a, b, d, tau1, tau0], (x1.size,))

    tau = np.divide(tau1, tau0)

    dcoupl_dx = eqtn_coupling_diff(K, w, ix, jx)

    if zmode == 'lin':
        dfx1_1_dx1 = 4.0 * np.ones(x1[ix].shape)
    elif zmode == 'sig':
        dfx1_1_dx1 = np.divide(30 * np.power(np.exp(1), (-10.0 * (x1[ix] + 0.5))),
                               np.power(1 + np.power(np.exp(1), (-10.0 * (x1[ix] + 0.5))), 2))
    else:
        raise ValueError('zmode is neither "lin" nor "sig"')

    if model == "2d":
        dfx1_3_dx1 = 3 * np.multiply(np.power(x1[ix], 2.0), a[ix]) - 2 * np.multiply(x1[ix], b[ix])
    else:
        dfx1_3_dx1 = 3 * np.multiply(np.power(x1[ix], 2.0), a[ix]) + 2 * np.multiply(x1[ix], d[ix] - b[ix])

    fx1z_diff = np.empty_like(dcoupl_dx, dtype=dcoupl_dx.dtype)
    for xi in ix:
        for xj in jx:
            if xj == xi:
                fx1z_diff[xi, xj] = np.multiply(dfx1_3_dx1[xi] + dfx1_1_dx1[xi] - dcoupl_dx[xi, xj], tau[xi])
            else:
                fx1z_diff[xi, xj] = np.multiply(- dcoupl_dx[xi, xj], tau[xi])

    return fx1z_diff


def eqtn_fy1(x1, yc, y1, d, tau1):

    return np.multiply((yc - np.multiply(pow(x1, 2), d) - y1), tau1)


def eqtn_fz(x1, z, x0, tau1, tau0, model="2d", zmode=np.array("lin"), z_pos=True, K=None, w=None, coupl=None, x0cr=None,
            r=None):

    if coupl is None:
        if np.all(K == 0.0) or np.all(w == 0.0) or (K is None) or (w is None):
            coupl = 0.0
        else:
            from tvb_epilepsy.base.calculations import calc_coupling
            coupl = calc_coupling(x1, K, w)

    tau = np.divide(tau1, tau0)

    if model == "2d":

        if zmode == 'lin':
            return np.multiply(
                (4 * (x1 - np.multiply(r, x0) + x0cr) - np.where(z_pos, z, z + 0.1 * np.power(z, 7.0)) - coupl), tau)

        elif zmode == 'sig':
            return np.multiply(np.divide(3.0, (1 + np.power(np.exp(1), (-10.0 * (x1 + 0.5))))) -
                               np.multiply(r, x0) + x0cr - z - coupl, tau)
        else:
            raise ValueError('zmode is neither "lin" nor "sig"')

    else:

        if zmode == 'lin':
            return np.multiply((4 * (x1 - x0) - np.where(z_pos, z, z + 0.1 * np.power(z, 7.0)) - coupl), tau)

        elif zmode == 'sig':
            return np.multiply(np.divide(3.0, (1 + np.power(np.exp(1), (-10.0 * (x1 + 0.5))))) - x0 - z - coupl, tau)
        else:
            raise ValueError('zmode is neither "lin" nor "sig"')


def eqtn_jac_fz_2d(x1, z, tau1, tau0, zmode=np.array("lin"), z_pos=True, K=None, w=None):
    tau = np.divide(tau1, tau0)

    jac_z = - np.ones(z.shape, dtype=z.dtype)

    if zmode == 'lin':

        jac_x1 = 4.0 * np.ones(z.shape, dtype=z.dtype)

        if not (z_pos):
            jac_z -= 0.7 * np.power(z, 6.0)

    elif zmode == 'sig':
        jac_x1 = np.divide(30 * np.power(np.exp(1), (-10.0 * (x1 + 0.5))),
                           1 + np.power(np.exp(1), (-10.0 * (x1 + 0.5))))
    else:
        raise ValueError('zmode is neither "lin" nor "sig"')

    # Assuming that wii = 0
    jac_x1 += np.multiply(K, sum(w, 1))
    jac_x1 = np.diag(jac_x1.flatten()) - np.multiply(np.repeat(np.reshape(K, (x1.size, 1)), x1.size, axis=1), w)
    jac_x1 *= np.repeat(np.reshape(tau, (x1.size, 1)), x1.size, axis=1)

    jac_z *= tau
    jac_z = np.diag(jac_z.flatten())

    return np.concatenate([jac_x1, jac_z], axis=1)


def eqtn_fx1z_2d_zpos_jac(x1, r, K, w, ix0, iE, a, b, tau1, tau0):

    p = x1.shape

    x1, r, K, a, b, tau1, tau0 = assert_arrays([x1, r, K, a, b, tau1, tau0], (1, x1.size))

    no_x0 = len(ix0)
    no_e = len(iE)

    i_x0 = np.ones((no_x0, 1), dtype="float32")
    i_e = np.ones((no_e, 1), dtype="float32")

    tau = np.divide(tau1, tau0)

    jac_e_x0e = np.diag(np.multiply(tau[:, iE], (- 4 * r[:, iE])).flatten())
    jac_e_x1o = -np.dot(np.dot(i_e, np.multiply(tau[:, iE], K[:, iE])), w[iE][:, ix0])
    jac_x0_x0e = np.zeros((no_x0, no_e), dtype="float32")
    jac_x0_x1o = (np.diag(np.multiply(tau[:, ix0],
                                      (4 + 3 * np.multiply(a[:, ix0], np.power(x1[:, ix0], 2))
                                       - 2 * np.multiply(b[:, ix0], x1[:, ix0]) +
                                       np.multiply(K[:, ix0], sum(w[ix0], axis=1)))).flatten()) -
                  np.multiply(np.dot(i_x0, np.multiply(tau[:, ix0], K[:, ix0])).T, w[ix0][:, ix0]))

    jac = np.empty((x1.size, x1.size), dtype=type(jac_e_x0e))
    jac[np.ix_(iE, iE)] = jac_e_x0e
    jac[np.ix_(iE, ix0)] = jac_e_x1o
    jac[np.ix_(ix0, iE)] = jac_x0_x0e
    jac[np.ix_(ix0, ix0)] = jac_x0_x1o

    return jac


def eqtn_fx1y1_6d_diff_x1(x1, a, b, d, tau1):

    return np.multiply(np.multiply(-3 * np.multiply(x1, a) + 2 * (b - d), x1), tau1)


def eqtn_fx2(x2, y2, z, g, Iext2, tau1):

    # ydot[3] = self.tt * (-y[4] + y[3] - y[3] ** 3 + self.Iext2 + 2 * y[5] - 0.3 * (y[2] - 3.5) + self.Kf * c_pop2)
    return np.multiply(-y2 + x2 - np.power(x2, 3) + Iext2 + 2 * g - 0.3 * (z - 3.5), tau1)


def eqtn_fy2(x2, y2, s, tau1, tau2, x2_neg=False):

    # ydot[4] = self.tt * ((-y[4] + np.where(y[3] < -0.25, if_ydot4, else_ydot4)) / self.tau)
    return np.divide(np.multiply(-y2 + np.where(x2_neg, 0.0, np.multiply(x2 + 0.25, s)), tau1), tau2)


def eqtn_fg(x1, g, gamma, tau1):

    return np.multiply(np.multiply(-g + 0.1 * x1, gamma), tau1)


def eqtn_fx0(x0_var, x0, tau1):
    # ydot[6] = self.tau1 * (-y[6] + self.x0)
    return np.multiply(-x0_var + x0, tau1)


def eqtn_fslope(slope_var, slope, tau1):
    # ydot[7] = 10 * self.tau1 * (-y[7] + slope_eq)
    return 10.0 * np.multiply(-slope_var + slope, tau1)


def eqtn_fIext1(Iext1_var, Iext1, tau1, tau0):
    # ydot[8] = self.tau1 * (-y[8] + self.Iext1) / self.tau0
    return np.divide(np.multiply(-Iext1_var + Iext1, tau1), tau0)


def eqtn_fIext2(Iext2_var, Iext2, tau1):
    # ydot[9] = 5 * self.tau1 * (-y[9] + Iext2_eq)
    return 5.0 * np.multiply(-Iext2_var + Iext2, tau1)


def eqtn_fK(K_var, K, tau1, tau0):
    # ydot[10] = self.tau1 * (-y[10] + self.K) / self.tau0
    return np.divide(np.multiply(-K_var + K, tau1), tau0)


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
              y1=None, x2=None, y2=None, g=None, x2_neg=False,
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


def eqtn_jac_2d(x1, z, K, w, slope, a, b, tau1, tau0, zmode=np.array("lin"), x1_neg=True, z_pos=True):

    jac_fx1 = eqtn_jac_x1_2d(x1, z, slope, a, b, tau1, x1_neg)

    jac_fz = eqtn_jac_fz_2d(x1, z, tau1, tau0, zmode, z_pos, K, w)

    return jac_fx1, jac_fz


def eqtn_fz_square_taylor(zeq, yc, Iext1, K, w, tau1, tau0):

    n_regions = zeq.size

    tau = np.divide(tau1, tau0)
    tau = np.repeat(tau.T, n_regions, 1)

    # The z derivative of the function
    # x1 = F(z) = -4/3 -1/2*sqrt(2(z-yc-Iext1)+64/27)
    dfz = -np.divide(0.5, np.power(2.0 * (zeq - yc - Iext1) + 64.0 / 27.0, 0.5))
    # Tim Proix: dfz = -np.divide(1, np.power(8.0 * zeq - 629.6/27, 0.5))

    try:
        if np.any([np.any(np.isnan(dfz)), np.any(np.isinf(dfz))]):
            raise ValueError("nan or inf values in dfz")
    except:
        pass

    # Jacobian: diagonal elements at first row
    # Diagonal elements: -1 + dfz_i * (4 + K_i * sum_j_not_i{wij})
    # Off diagonal elements: -K_i * wij_not_i * dfz_j_not_i
    i = np.ones((1, n_regions), dtype=np.float32)
    fz_jac = np.diag((-1.0 + np.multiply(dfz, (4.0 + np.multiply(K, np.expand_dims(sum(w, axis=1), 1).T)))).T[:, 0]) \
             - np.multiply(np.multiply(np.dot(K.T, i), np.np.dot(i.T, dfz)), w)

    try:
        if np.any([np.any(np.isnan(fz_jac.flatten())), np.any(np.isinf(fz_jac.flatten()))]):
            raise ValueError("nan or inf values in dfz")
    except:
        pass

    return np.multiply(fz_jac, tau)
