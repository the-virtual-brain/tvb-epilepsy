"""
Module to compute the resting equilibrium point of a Virtual Epileptic Patient module
"""
import warnings
import numpy
from scipy.optimize import root
from sympy import symbols, exp, solve, lambdify
from tvb_epilepsy.base.constants import X1_DEF, X1_EQ_CR_DEF, X0_DEF, X0_CR_DEF


#Currently we assume only difference coupling (permittivity coupling following Proix et al 2014)
#TODO: to generalize for different coupling functions

def def_x1eq(X1_DEF, X1_EQ_CR_DEF, n_regions):
    #The default initial condition for x1 equilibrium search
    return (X1_EQ_CR_DEF + X1_DEF) / 2.0 * numpy.ones((1,n_regions), dtype='float32')


def def_x1lin(X1_DEF, X1_EQ_CR_DEF, n_regions):
    # The point of the linear Taylor expansion
    return (X1_EQ_CR_DEF + X1_DEF) / 2.0 * numpy.ones((1,n_regions), dtype='float32')


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
        return (x1 + x0cr - (z+coupl) / 4.0) / rx0

    elif zmode == 'sig':
        return (3.0 / (1.0 + numpy.exp(-10.0 * (x1 + 0.5))) + x0cr - z + coupl) / rx0

    else:
        raise ValueError('zmode is neither "lin" nor "sig"')


def calc_fx1(x1, z=0, y1=0.0, x2=0, Iext1=0, slope=0.0, a=1.0, b=-2.0, tau1=1.0, x1_neg=None):

    # if_ydot0 = - self.a * y[0] ** 2 + self.b * y[0]
    if_ydot0 = - a * x1 ** 2 + b * x1  # self.a=1.0, self.b=3.0

    # else_ydot0 = self.slope - y[3] + 0.6 * (y[2] - 4.0) ** 2
    else_ydot0 = slope - x2 + 0.6 * (z - 4.0) ** 2

    if x1_neg is None:
        x1_neg = x1 < 0.0

    return tau1 * (y1 - z + Iext1 + numpy.where(x1_neg, if_ydot0, else_ydot0) * x1)


def calc_fx1_6d(x1, z=0.0, y1=0.0, x2=0.0, Iext1=0.0, slope=0.0, a=1.0, b=3.0, tau1=1.0, x1_neg=None):
    return calc_fx1(x1, z, y1, x2, Iext1, slope, a, b, tau1, x1_neg)


def calc_fx1_2d(x1, z=0, y0=0.0, x2=0.0, Iext1=0.0, slope=0.0, a=1.0, b=-2.0, tau1=1.0, x1_neg=None):
    return calc_fx1(x1, z, y0, x2, Iext1, slope, a, b, tau1, x1_neg)


def calc_fy1(x1, yc, y1=0, d=5.0, tau1=1.0):
    return tau1 * (yc - d * x1 ** 2 - y1)


def calc_fz_lin(x1, x0, x0cr, r, z=0, coupl=0, tau0=1.0):
    return (4 * (x1 - r * x0 + x0cr) - z - coupl) / tau0


def calc_fz_sig(x1, x0, x0cr, r, z=0, coupl=0, tau0=1.0):
    return (3/(1 + exp(-10 * (x1 + 0.5))) - r * x0 + x0cr - z - coupl) / tau0


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


def calc_dfun(x1, z, yc, Iext1, x0, x0cr, rx0, K, w, model="2d", zmode="lin", pmode="const", x1_neg=None,
              y1=None, x2=None, y2=None, g=None, x2_neg=None,
              x0_var=None, slope_var=None, Iext1_var=None, Iext2_var=None, K_var=None,
              Iext2=0.45, slope=0.0, a=1.0, b=-2.0, d=5.0, s=6.0,
              tau1=1.0, tau0=2857.0, tau2=10.0):

    n_regions = x1.size
    x1_type = x1.dtype

    if model == "2d":
        f = numpy.zeros((2,n_regions),dtype=x1_type)
        f[0,:] = calc_fx1_2d(x1, z, yc, Iext1=Iext1, slope=slope, a=a, b=b, tau1=tau1, x1_neg=x1_neg)
        iz = 1

    else:
        f = numpy.zeros((6, n_regions), dtype=x1_type)
        f[0, :] = calc_fx1_6d(x1, z, y1, x2, Iext1, slope, a, b, tau1, x1_neg)
        iz = 2
        f[1, :] = calc_fy1(x1, yc, y1, d, tau1)
        f[3, :], f[4, :] = calc_fpop2(x2, y2, g, z, Iext2, s, tau1, tau2, x2_neg)

        if  model=="11d":
            from tvb_epilepsy.tvb_api.epileptor_models import EpileptorDPrealistic
            #ydot[6] = self.tau1 * (-y[6] + self.x0)
            f[6, :] = tau1 * (-x0_var + x0)
            slope_eq, Iext2_eq = EpileptorDPrealistic.fun_slope_Iext2(z, g, pmode)
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

    if zmode == "lin":
        f[iz, :] = calc_fz_lin(x1, x0, x0cr, rx0, z, calc_coupling(x1, K, w), tau0)
    elif zmode == "sig":
        f[iz, :] = calc_fz_sig(x1, x0, x0cr, rx0, z, calc_coupling(x1, K, w), tau0)
    else:
        raise ValueError('zmode is neither "lin" nor "sig"')

    return f


def calc_eq_z_2d(x1eq, y0, Iext1, x1_neg=True):

    return calc_fx1_2d(x1eq, z=0, y0=y0, Iext1=Iext1, x1_neg=x1_neg)


def calc_eq_z_6d(x1eq, y1, Iext1, x1_neg=True):
    return calc_fx1_6d(x1eq, z=0, y1=y1, Iext1=Iext1, x1_neg=x1_neg)


def calc_eq_y1(x1eq, yc, d=5.0):
    return calc_fy1(x1eq, yc, y1=0, d=d, tau1=1.0)


def calc_eq_pop2(x1eq, zeq, Iext2):

    y2eq = numpy.zeros((x1eq.size,), dtype=x1eq.dtype)

    # # -x2eq**3 + x2eq -y2eq+2*g_eq-0.3*(zeq-3.5)+Iext2 =0=> (1),(2)
    # # -x2eq**3 + x2eq +2*0.1*x1eq-0.3*(zeq-3.5)+Iext2 =0=>
    # # p3        p1                   p0
    # # -x2eq**3 + x2eq +0.2*x1eq-0.3*(zeq-3.5)+Iext2 =0
    # p0 = 0.2 * x1eq - 0.3 * (zeq - 3.5) + Iext2
    # x2eq = numpy.zeros(x1eq.shape, dtype=x1eq.dtype)
    # for i in range(shape[1]):
    #     x2eq[0 ,i] = numpy.min(numpy.real(numpy.roots([-1.0, 0.0, 1.0, p0[0, i]])))

    g_eq = numpy.squeeze(calc_eq_g(x1eq))

    (x2, fx2) = symbols('x2 fx2')

    #TODO: use symbolic vectors and functions
    #fx2 = -y2eq + x2 - x2 ** 3 + numpy.squeeze(Iext2) + 2 * g_eq - 0.3 * (numpy.squeeze(zeq) - 3.5)
    fx2 = numpy.squeeze(calc_fpop2(x2, y2eq, g_eq, zeq, Iext2, s=6.0, tau1=1.0, tau2=10.0,
                                  x2_neg=numpy.ones((x1eq.size,), dtype='bool'))[0])

    x2eq = []
    for ii in range(y2eq.size):
        x2eq.append(numpy.min(numpy.real(numpy.array(solve(fx2[ii], x2),dtype="complex"))))

    return numpy.reshape(numpy.array(x2eq, dtype=x1eq.dtype), x1eq.shape),  numpy.reshape(y2eq, x1eq.shape)


def calc_eq_g(x1eq):
    return calc_fg(x1eq, g=0.0, gamma=1.0, tau1=1.0)


def eq_x1_hypo_x0_optimize_fun(x, ix0, iE, x1EQ, zEQ, x0, x0cr, rx0, y0, Iext1, K, w):

    x1_type = x1EQ.dtype

    # #Coupling                        to   from           from                    to
    # no_x0 = len(ix0)
    # no_e = len(iE)
    # i_e = numpy.ones((no_e,1), dtype=x1_type)
    # i_x0 = numpy.ones((no_x0,1), dtype=x1_type)
    # # w_e_to_e = numpy.sum(numpy.dot(w[iE][:,iE],    numpy.dot(i_e, x1EQ[:,iE]) - numpy.dot(i_e, x1EQ[:,iE]).T), axis=1)
    # # w_x0_to_e = numpy.sum(numpy.dot(w[iE][:, ix0], numpy.dot(i_e, x[:,ix0]) - numpy.dot(i_x0, x1EQ[:,iE]).T), axis=1)
    # Coupl_to_e = calc_coupling(x1EQ, K, w, ix=iE)
    #
    # # w_e_to_x0 = numpy.sum(numpy.dot(w[ix0][:, iE],  numpy.dot(i_x0, x1EQ[:, iE]) - numpy.dot(i_e, x[:,ix0]).T), axis=1)
    # # w_x0_to_x0 = numpy.sum(numpy.dot(w[ix0][:,ix0], numpy.dot(i_x0, x[:,ix0]) - numpy.dot(i_x0, x[:,ix0]).T), axis=1)
    # Coupl_to_x0 = calc_coupling(x1EQ, K, w, ix=ix0)
    #
    # fun = numpy.zeros(x1EQ.shape).astype(x1_type)
    # #Known x1eq, unknown x0:
    # # fun[:,iE] = calc_fz_lin(x1EQ[:,iE], x0[:,iE], x0cr[:,iE], rx0[:,iE], z=zEQ[:,iE],
    # #                         coupl=K[:,iE] * (w_e_to_e + w_x0_to_e)).astype(x1_type)
    # fun[:, iE] = calc_fz_lin(x1EQ[:, iE], x0[:, iE], x0cr[:, iE], rx0[:, iE], z=zEQ[:, iE],
    #                          coupl=calc_coupling(x1EQ, K, w, ix=i_x0)).astype(x1_type)
    #
    # # Known x0, unknown x1eq:
    # # fun[:,ix0] = calc_fz_lin(x[:, ix0], x0, x0cr[:, ix0], rx0[:, ix0],
    # #                          z=calc_eq_z_2d(x[:, ix0], y0[:, ix0], Iext1[:, ix0]),
    # #                          coupl=K[:, ix0] * (w_e_to_x0 + w_x0_to_x0)).astype(x1_type)
    # fun[:, ix0] = calc_fz_lin(x1EQ[:, ix0], x0, x0cr[:, ix0], rx0[:, ix0],
    #                           z=calc_eq_z_2d(x[:, ix0], y0[:, ix0], Iext1[:, ix0]),
    #                           coupl=Coupl_to_x0).astype(x1_type)

    # Construct the x1 and z vectors, comprising of the current x1EQ, zEQ values for i_e regions,
    # and the unknown x1 values for x1EQ and respective zEQ for the i_x0 regions
    x1EQ[:, ix0] = numpy.array(x[ix0])
    zEQ[:, ix0] = numpy.array(calc_eq_z_2d(x1EQ[:, ix0], y0[:, ix0], Iext1[:, ix0]))

    # Construct the x0 vector, comprising of the current x0 values for i_x0 regions,
    # and the unknown x0 values for the i_e regions
    x0_dummy = numpy.array(x0)
    x0 = numpy.array(x1EQ)
    x0[:, iE] = numpy.array(x[iE])
    x0[:, ix0] = numpy.array(x0_dummy)
    del x0_dummy

    fun = calc_fz_lin(x1EQ, x0, x0cr, rx0, z=zEQ, coupl=calc_coupling(x1EQ, K, w)).astype(x1_type)

    # if numpy.any([numpy.any(numpy.isnan(x)), numpy.any(numpy.isinf(x)),
    #               numpy.any(numpy.isnan(fun)), numpy.any(numpy.isinf(fun))]):
    #     raise ValueError("nan or inf values in x or fun")

    return numpy.squeeze(fun)


def eq_x1_hypo_x0_optimize_jac(x, ix0, iE, x1EQ, zEQ, x0, x0cr, rx0, y0, Iext1, K, w):

    x = numpy.expand_dims(x, 1).T

    no_x0 = len(ix0)
    no_e = len(iE)

    n_regions = no_e + no_x0

    x1_type = x1EQ.dtype
    i_x0 = numpy.ones((no_x0, 1), dtype=type)
    i_e = numpy.ones((no_e, 1), dtype=type)

    jac_e_x0e = numpy.diag(- 4 * rx0[:, iE]).astype(x1_type)
    jac_e_x1o = -numpy.dot(numpy.dot(i_e, K[:,iE]), w[iE][:,ix0]).astype(x1_type)
    jac_x0_x0e = numpy.zeros((no_x0, no_e)).astype(x1_type)
    jac_x0_x1o = (numpy.diag(4 + 3 * x[:, ix0] ** 2 + 4 * x[:, ix0] + K[:, ix0] * numpy.sum(w[ix0][:,ix0], axis=1)) - \
                  numpy.dot(i_x0, K[:, ix0]) * w[ix0][:, ix0]).astype(x1_type)

    jac = numpy.zeros((n_regions,n_regions), dtype=x1_type)
    jac[numpy.ix_(iE, iE)] = jac_e_x0e
    jac[numpy.ix_(iE, ix0)] = jac_e_x1o
    jac[numpy.ix_(ix0, iE)] = jac_x0_x0e
    jac[numpy.ix_(ix0, ix0)] = jac_x0_x1o

    # if numpy.any([ numpy.any(numpy.isnan(x)), numpy.any(numpy.isnan(x)),
    #                numpy.any(numpy.isnan(jac.flatten())), numpy.any(numpy.isinf(jac.flatten()))]):
    #     raise ValueError("nan or inf values in x or jac")

    return jac


def eq_x1_hypo_x0_optimize(ix0, iE, x1EQ, zEQ, x0, x0cr, rx0, y0, Iext1, K, w):

    xinit = numpy.zeros(x1EQ.shape, dtype = x1EQ.dtype)

    #Set initial conditions for the optimization algorithm, by ignoring coupling (=0)
    # fz = 4 * (x1 - r * x0 + x0cr) - z -coupling = 0
    #x0init = (x1 + x0cr -z/4) / rx0
    xinit[:, iE] = calc_x0(x1EQ[:, iE], zEQ[:, iE], x0cr[:, iE],  rx0[:, iE], 0.0)
    #x1eqinit = rx0 * x0 - x0cr + z / 4
    xinit[:, ix0] = rx0[:, ix0] * x0 - x0cr[:, ix0] + zEQ[:, ix0] / 4

    #Solve:
    sol = root(eq_x1_hypo_x0_optimize_fun, xinit, args=(ix0, iE, x1EQ, zEQ, x0, x0cr, rx0, y0, Iext1, K, w),
               method='lm', jac=eq_x1_hypo_x0_optimize_jac, tol=10**(-6), callback=None, options=None) #method='hybr'

    if sol.success:
        x1EQ[:, ix0] = sol.x[ix0]
        if numpy.any([numpy.any(numpy.isnan(sol.x)), numpy.any(numpy.isinf(sol.x))]):
            raise ValueError("nan or inf values in solution x\n" + sol.message)
        else:
            return x1EQ
    else:
        raise ValueError(sol.message)



def eq_x1_hypo_x0_linTaylor(ix0, iE, x1EQ, zEQ, x0, x0cr, rx0, y0, Iext1, K, w):

    no_x0 = len(ix0)
    no_e = len(iE)

    n_regions = no_e + no_x0

    # The equilibria of the nodes of fixed epileptogenicity
    x1_eq = x1EQ[:, iE]
    z_eq = zEQ[:, iE]

    #Prepare linear system to solve:

    x1_type = x1EQ.dtype

    #The point of the linear Taylor expansion
    x1LIN = def_x1lin(X1_DEF, X1_EQ_CR_DEF, n_regions).astype(x1_type)

    # For regions of fixed equilibria:
    ii_e = numpy.ones((1, no_e), dtype=x1_type)
    we_to_e = numpy.expand_dims(numpy.sum(w[iE][:, iE] * (numpy.dot(ii_e.T, x1_eq) -
                                                          numpy.dot(x1_eq.T, ii_e)), axis=1), 1).T.astype(x1_type)
    wx0_to_e = x1_eq * numpy.expand_dims(numpy.sum(w[ix0][:, iE], axis=0), 0).astype(x1_type)
    be = 4.0 * (x1_eq + x0cr[:, iE]) - z_eq - K[:, iE] * (we_to_e - wx0_to_e)

    # For regions of fixed x0:
    ii_x0 = numpy.ones((1, no_x0), dtype=x1_type)
    we_to_x0 = numpy.expand_dims(numpy.sum(w[ix0][:, iE] * numpy.dot(ii_x0.T, x1_eq), axis=1), 1).T.astype(x1_type)
    bx0 = 4.0 * (x0cr[:, ix0] - rx0[:, ix0] * x0) - y0[:, ix0] - Iext1[:, ix0] \
          - 2.0 * x1LIN[:, ix0] ** 3 - 2.0 * x1LIN[:, ix0] ** 2 - K[:, ix0] * we_to_x0

    # Concatenate B vector:
    b = -numpy.concatenate((be, bx0), axis=1).T.astype(x1_type)

    # From-to Epileptogenicity-fixed regions
    # ae_to_e = -4 * numpy.eye( no_e, dtype=numpy.float32 )
    ae_to_e = -4 * numpy.diag(rx0[0, iE]).astype(x1_type)

    # From x0-fixed regions to Epileptogenicity-fixed regions
    ax0_to_e = -numpy.dot(K[:, iE].T, ii_x0) * w[iE][:, ix0]

    # From Epileptogenicity-fixed regions to x0-fixed regions
    ae_to_x0 = numpy.zeros((no_x0, no_e), dtype=x1_type)

    # From-to x0-fixed regions
    ax0_to_x0 = numpy.diag( (4.0 + 3.0 * x1LIN[:, ix0] ** 2 + 4.0 * x1LIN[:, ix0] +
                K[0, ix0] * numpy.expand_dims(numpy.sum(w[ix0][:, ix0], axis=0), 0)).T[:, 0]) - \
                numpy.dot(K[:, ix0].T, ii_x0) * w[ix0][:, ix0]

    # Concatenate A matrix
    a = numpy.concatenate((numpy.concatenate((ae_to_e, ax0_to_e), axis=1),
                           numpy.concatenate((ae_to_x0, ax0_to_x0), axis=1)), axis=0).astype(x1_type)

    # Solve the system
    x = numpy.dot(numpy.linalg.inv(a), b).T
    if numpy.any([numpy.any(numpy.isnan(x)), numpy.any(numpy.isnan(x))]):
        raise ValueError("nan or inf values in solution x")

    # Unpack solution:
    # The equilibria of the regions with fixed E have not changed:
    # The equilibria of the regions with fixed x0:
    x1EQ[0, ix0] = x[0, no_e:]

    return x1EQ


def calc_x0cr_rx0(y0, Iext1, epileptor_model = "2d", zmode = numpy.array("lin"),
                  x1rest = X1_DEF, x1cr = X1_EQ_CR_DEF, x0def = X0_DEF, x0cr_def = X0_CR_DEF):

    from tvb_epilepsy.tvb_api.epileptor_models import EpileptorDP2D

    #Define the symbolic variables we need:
    (y01, I1, x1, z, x0, r, x0cr, f1, fz) = symbols('y01 I1 x1 z x0 r x0cr f1 fz')

    #Define the fx1(x1) expression (assuming centered x1 in all cases)...
    if isinstance(epileptor_model,EpileptorDP2D) or epileptor_model=="2d":
        #...for the 2D permittivity coupling approximation, Proix et al 2014
        #fx1 = x1 ** 3 + 2 * x1 ** 2
        # #...and the z expression, coming from solving dx1/dt=f1(x1,z)=0
        # z = y01 - fx1 + I1
        z = calc_eq_z_2d(x1, y01, I1)

    else:
        #...or for the original (>=6D) epileptor
        # fx1 = x1 ** 3 - 3 * x1 ** 2
        # #...and the z expression, coming from solving dx1/dt=f1(x1,z)=0
        # y1 = y01 - 5.0 * x1 ** 2
        # z = y1 - fx1 + I1
        z = calc_eq_z_6d(x1, calc_eq_y1(x1, y01, d=5.0), I1)

    #Define the fz expression...
    if zmode == 'lin':
        #...for linear...
        #fz = 4 * (x1 - r * x0 + x0cr) - z
        fz = calc_fz_lin(x1, x0, x0cr, r, z=z, coupl=0)

    elif zmode == 'sig':
        #...and sigmoidal versions
        #fz = 3/(1 + exp(-10 * (x1 + 0.5))) - r * x0 + x0cr - z
        z = calc_fz_sig(x1, x0, x0cr, r, z=z, coupl=0)

    else:
        raise ValueError('zmode is neither "lin" nor "sig"')

    #Solve the fz expression for rx0 and x0cr, assuming the following two points (x1eq,x0) = [(-5/3,0.0),(-4/3,1.0)]...
    #...and WITHOUT COUPLING
    fz_sol = solve([fz.subs([(x1, x1rest), (x0, x0def), (z, z.subs(x1, x1rest))]),
                    fz.subs([(x1, x1cr), (x0, x0cr_def), (z, z.subs(x1, x1cr))])], r, x0cr)

    #Convert the solution of x0cr from expression to function that accepts numpy arrays as inputs:
    x0cr = lambdify((y01,I1), fz_sol[x0cr], 'numpy')

    #Compute the actual x0cr now given the inputs y0 and Iext1
    x0cr = x0cr(y0, Iext1)

    #The rx0 doesn' depend on y0 and Iext1, therefore...
    rx0 = fz_sol[r]*numpy.ones(shape=x0cr.shape)

    return x0cr, rx0


def assert_equilibrium_point(epileptor_model, hypothesis, equilibrium_point):

    from tvb_epilepsy.tvb_api.epileptor_models import EpileptorDPrealistic

    n_dim = equilibrium_point.shape[0]

    coupl = calc_coupling(hypothesis.x1EQ, epileptor_model.K.T, hypothesis.weights)
    coupl = numpy.expand_dims(numpy.r_[coupl, 0.0 * coupl], 2).astype('float32')

    dfun = epileptor_model.dfun(numpy.expand_dims(equilibrium_point, 2).astype('float32'), coupl).squeeze()

    dfun_max = numpy.max(dfun)

    if dfun_max > 10 ** -2:

        if (isinstance(epileptor_model, EpileptorDPrealistic) and
                    epileptor_model.pmode == numpy.array(['g', 'z', 'z*g'])).any():
            warnings.warn("Equilibrium point for initial condition not accurate enough!\n" \
                          + "max(dfun) = " + str(dfun_max))

        else:
            raise ValueError("Equilibrium point for initial condition not accurate enough!\n" \
                             + "max(dfun) = " + str(dfun_max))

    if n_dim == 2:

        dfun2 = calc_dfun(equilibrium_point[0].squeeze(), equilibrium_point[2].squeeze(),
                          epileptor_model.yc.squeeze(), epileptor_model.Iext1.squeeze(), epileptor_model.x0.squeeze(),
                          epileptor_model.x0cr.squeeze(), epileptor_model.r.squeeze(), epileptor_model.K.squeeze(),
                          hypothesis.weights, model=str(n_dim)+"d", zmode=epileptor_model.zmode,
                          slope=epileptor_model.slope.squeeze(), a=1.0, b=-2.0,
                          tau1=epileptor_model.tau1, tau0=epileptor_model.tau0)

    elif n_dim == 6:

        dfun2 = calc_dfun(equilibrium_point[0].squeeze(), equilibrium_point[2].squeeze(),
                          epileptor_model.yc.squeeze(), epileptor_model.Iext1.squeeze(), epileptor_model.x0.squeeze(),
                          epileptor_model.x0cr.squeeze(), epileptor_model.r.squeeze(), epileptor_model.K.squeeze(),
                          hypothesis.weights, model=str(n_dim)+"d", zmode=epileptor_model.zmode,
                          y1=equilibrium_point[1].squeeze(), x2=equilibrium_point[3].squeeze(),
                          y2=equilibrium_point[4].squeeze(), g=equilibrium_point[5].squeeze(),
                          Iext2=epileptor_model.Iext2.squeeze(), slope=epileptor_model.slope.squeeze(), a=1.0, b=3.0,
                          tau1=epileptor_model.tau1, tau0=epileptor_model.tau0, tau2=epileptor_model.tau2)

    elif n_dim == 11:

        dfun2 = calc_dfun(equilibrium_point[0].squeeze(), equilibrium_point[2].squeeze(),
                          epileptor_model.yc.squeeze(), epileptor_model.Iext1.squeeze(), epileptor_model.x0.squeeze(),
                          epileptor_model.x0cr.squeeze(), epileptor_model.r.squeeze(), epileptor_model.K.squeeze(),
                          hypothesis.weights, model=str(n_dim)+"d", zmode=epileptor_model.zmode, pmode=epileptor_model.pmode,
                          y1=equilibrium_point[1].squeeze(), x2=equilibrium_point[3].squeeze(),
                          y2=equilibrium_point[4].squeeze(), g=equilibrium_point[5].squeeze(),
                          x0_var=equilibrium_point[6].squeeze(), slope_var=equilibrium_point[7].squeeze(),
                          Iext1_var=equilibrium_point[8].squeeze(), Iext2_var=equilibrium_point[9].squeeze(),
                          K_var=equilibrium_point[10].squeeze(), Iext2=epileptor_model.Iext2.squeeze(),
                          slope=epileptor_model.slope.squeeze(), a=1.0, b=3.0,
                          tau1=epileptor_model.tau1, tau0=epileptor_model.tau0, tau2=epileptor_model.tau2)

    if numpy.max(numpy.abs(dfun2 - dfun.squeeze())) > 10 ** -2:
        warnings.warn("model dfun and calc_dfun functions do not return the same results!\n"
                      + "model dfun = " + str(dfun) + "\n"
                      + "calc_dfun = " + str(dfun2))


def calc_equilibrium_point(epileptor_model, hypothesis):

    from tvb_epilepsy.tvb_api.epileptor_models import EpileptorDPrealistic, EpileptorDP2D

    if isinstance(epileptor_model,EpileptorDP2D):
        if epileptor_model.zmode == 'sig':
            #2D approximation, Proix et al 2014
            zeq = calc_eq_z_2d(hypothesis.x1EQ, epileptor_model.yc.T, epileptor_model.Iext1.T)
        else:
            zeq = hypothesis.zEQ
        equilibrium_point = numpy.r_[hypothesis.x1EQ, zeq].astype('float32')
    else:
        #all >=6D models
        y1eq = calc_eq_y1(hypothesis.x1EQ, epileptor_model.yc.T)
        zeq = calc_eq_z_6d(hypothesis.x1EQ, y1eq, epileptor_model.Iext1.T)
        if epileptor_model.Iext2.size == 1:
            epileptor_model.Iext2 = epileptor_model.Iext2[0] * numpy.ones((hypothesis.n_regions, 1))
        (x2eq, y2eq) = calc_eq_pop2(hypothesis.x1EQ, zeq, epileptor_model.Iext2.T)
        geq = calc_eq_g(hypothesis.x1EQ)
        if isinstance(epileptor_model, EpileptorDPrealistic):
            # the 11D "realistic" simulations model
            if epileptor_model.slope.size == 1:
                epileptor_model.slope = epileptor_model.slope[0] * numpy.ones((hypothesis.n_regions, 1))
            slope_eq, Iext2_eq = epileptor_model.fun_slope_Iext2(zeq.T, geq.T)
            equilibrium_point = numpy.r_[hypothesis.x1EQ, y1eq, zeq, x2eq, y2eq, geq,
                                          epileptor_model.x0.T, slope_eq.T, epileptor_model.Iext1.T, Iext2_eq.T,
                                          epileptor_model.K.T].astype('float32')
        else:
            #all >=6D models
            equilibrium_point = numpy.r_[hypothesis.x1EQ, y1eq, zeq, x2eq, y2eq, geq].astype('float32')

    assert_equilibrium_point(epileptor_model, hypothesis, equilibrium_point)

    return equilibrium_point