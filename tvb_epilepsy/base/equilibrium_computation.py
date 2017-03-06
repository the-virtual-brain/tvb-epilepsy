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

def x1eq_def(X1_DEF, X1_EQ_CR_DEF, n_regions):
    #The default initial condition for x1 equilibrium search
    return (X1_EQ_CR_DEF + X1_DEF) / 2.0 * numpy.ones((1,n_regions), dtype='float32')


def x1_lin_def(X1_DEF, X1_EQ_CR_DEF, n_regions):
    # The point of the linear Taylor expansion
    return (X1_EQ_CR_DEF + X1_DEF) / 2.0 * numpy.ones((1,n_regions), dtype='float32')


def fx1_calc(x1, z=0, y1=0.0, x2=0, Iext1=0, slope=0.0, a=1.0, b=-2.0, tau1=1.0):

    # if_ydot0 = - self.a * y[0] ** 2 + self.b * y[0]
    if_ydot0 = - a * x1 ** 2 + b * x1  # self.a=1.0, self.b=3.0

    # else_ydot0 = self.slope - y[3] + 0.6 * (y[2] - 4.0) ** 2
    else_ydot0 = slope - x2 + 0.6 * (z - 4.0) ** 2

    return tau1*(y1 - z + Iext1 + numpy.where(x1 < 0.0, if_ydot0, else_ydot0) * x1)


def fx1_6d_calc(x1, z=0.0, y1=0.0, x2=0.0, Iext1=0.0, slope=0.0, a=1.0, b=3.0, tau1=1.0):
    return fx1_calc(x1, z=z, y1=y1, x2=x2, Iext1=Iext1, slope=slope, tau1=tau1, a=a, b=b)


def fx1_2d_calc(x1, z=0, y0=0.0, x2=0.0, Iext1=0.0, slope=0.0, a=1.0, b=-2.0, tau1=1.0):
    return fx1_calc(x1, z=z, y1=y0, x2=x2, Iext1=Iext1, slope=slope, tau1=tau1, a=a, b=b)


def y1_calc(x1, yc, y1=0, d=5.0, tau1=1.0):
    return tau1 * (yc - d * x1 ** 2 - y1)


def fz_lin_calc(x1, x0, x0cr, r, z=0, coupl=0, tau0=1.0):
    return (4 * (x1 - r * x0 + x0cr) - z - coupl) / tau0


def fz_sig_calc(x1, x0, x0cr, r, z=0, coupl=0, tau0=1.0):
    return (3/(1 + exp(-10 * (x1 + 0.5))) - r * x0 + x0cr - z - coupl) / tau0


def pop2_calc(x2, y2=0.0, g=0.0, z=0.0, Iext2=0.45, s=6.0, tau1=1.0, tau2=10.0):

    # ydot[3] = self.tt * (-y[4] + y[3] - y[3] ** 3 + self.Iext2 + 2 * y[5] - 0.3 * (y[2] - 3.5) + self.Kf * c_pop2)
    fx2 = tau1 * (-y2 + x2 - x2 ** 3 + Iext2 + 2 * g - 0.3 * (z - 3.5))

    # if_ydot4 = 0
    if_ydot4 = 0
    # else_ydot4 = self.aa * (y[3] + 0.25)
    else_ydot4 = s * (x2 + 0.25)  # self.s = 6.0

    # ydot[4] = self.tt * ((-y[4] + where(y[3] < -0.25, if_ydot4, else_ydot4)) / self.tau)
    fy2 = tau1 * ((-y2 + numpy.where(x2 < -0.25, if_ydot4, else_ydot4)) / tau2)

    return fx2, fy2


def g_calc(x1, g=0, gamma=0.01, tau1=1.0):
    #ydot[5] = self.tt * (-0.01 * (y[5] - 0.1 * y[0]))
    return -tau1 * gamma * (g - 0.1 * x1)


def dfun(x1, z, yc, Iext1, x0, x0cr, rx0, K, w, model="2d", zmode="lin",
         y1=None, x2=None, y2=None, g=None,
         Iext2=0.45, slope=0.0, a=1.0, b=-2.0, d=5.0, s=6.0,
         tau1=1.0, tau2=10.0, tau0=2857.0):

    n_regions = x1.size
    x1_type = x1.dtype

    if model == "2d":
        f = numpy.zeros((2,n_regions),dtype=x1_type)
        f[0,:] = fx1_2d_calc(x1, z, yc, Iext1, tau1)
        iz = 1

    else:
        f = numpy.zeros((6, n_regions), dtype=x1_type)
        f[0, :] = fx1_6d_calc(x1, z, y1, Iext1, tau1)
        iz = 2
        f[1, :] = y1_calc(x1, yc, y1, d, tau1)
        f[3, :], f[4, :] = pop2_calc(x1, x2, y2, g, z, Iext2, s, tau1, tau2)

    if zmode == "lin":
        f[iz, :] = fz_lin_calc(x1, x0, x0cr, rx0, z, coupling_calc(x1, K, w), tau0)
    elif zmode == "lin":
        f[iz, :] = fz_sig_calc(x1, x0, x0cr, rx0, z, coupling_calc(x1, K, w), tau0)
    else:
        raise ValueError('zmode is neither "lin" nor "sig"')


def zeq_2d_calc(x1eq, y0, Iext1):
    return fx1_2d_calc(x1eq, z=0, y0=y0, Iext1=Iext1)


def zeq_6d_calc(x1eq, y1, Iext1):
    return fx1_6d_calc(x1eq, z=0, y1=y1, Iext1=Iext1)


def y1eq_calc(x1eq, yc, d=5.0):
    return y1_calc(x1eq, yc, y1=0, d=d, tau1=1.0)


def pop2eq_calc(x1eq, zeq, Iext2):

    y2eq = numpy.zeros((x1eq.size,), dtype=x1eq.dtype)

    # # -x2eq**3 + x2eq -y2eq+2*g_eq-0.3*(zeq-3.5)+Iext2 =0=> (1),(2)
    # # -x2eq**3 + x2eq +2*0.1*x1eq-0.3*(zeq-3.5)+Iext2 =0=>
    # # p3        p1                   p0
    # # -x2eq**3 + x2eq +0.2*x1eq-0.3*(zeq-3.5)+Iext2 =0
    # p0 = 0.2 * x1eq - 0.3 * (zeq - 3.5) + Iext2
    # x2eq = numpy.zeros(x1eq.shape, dtype=x1eq.dtype)
    # for i in range(shape[1]):
    #     x2eq[0 ,i] = numpy.min(numpy.real(numpy.roots([-1.0, 0.0, 1.0, p0[0, i]])))

    g_eq = numpy.squeeze(geq_calc(x1eq))

    (x2, fx2) = symbols('x2 fx2')

    #TODO: use symbolic vectors and functions, and define them as negative, so that to resolve the where() statement
    fx2 = -y2eq + x2 - x2 ** 3 + numpy.squeeze(Iext2) + 2 * g_eq - 0.3 * (numpy.squeeze(zeq) - 3.5)

    x2eq = []
    for ii in range(y2eq.size):
        x2eq.append(numpy.min(numpy.real(numpy.array(solve(fx2[ii], x2),dtype="complex"))))

    return numpy.reshape(numpy.array(x2eq, dtype=x1eq.dtype), x1eq.shape),  numpy.reshape(y2eq, x1eq.shape)


def geq_calc(x1eq):
    return g_calc(x1eq, g=0.0, gamma=1.0, tau1=1.0)


def x1eq_x0_hypo_optimize_fun(x, ix0, iE, x1EQ, zEQ, x0, x0cr, rx0, y0, Iext1, K, w):

    x1_type = x1EQ.dtype

    # #Coupling                        to   from           from                    to
    # no_x0 = len(ix0)
    # no_e = len(iE)
    # i_e = numpy.ones((no_e,1), dtype=x1_type)
    # i_x0 = numpy.ones((no_x0,1), dtype=x1_type)
    # # w_e_to_e = numpy.sum(numpy.dot(w[iE][:,iE],    numpy.dot(i_e, x1EQ[:,iE]) - numpy.dot(i_e, x1EQ[:,iE]).T), axis=1)
    # # w_x0_to_e = numpy.sum(numpy.dot(w[iE][:, ix0], (numpy.dot(i_e, x[:,ix0]) - numpy.dot(i_x0, x1EQ[:,iE]).T).T), axis=1)
    # Coupl_to_e = coupling_calc(x1EQ, K, w, ix=iE)
    #
    # # w_e_to_x0 = numpy.sum(numpy.dot(w[ix0][:, iE], (numpy.dot(i_x0, x1EQ[:, iE]) - numpy.dot(i_e, x[:,ix0]).T).T), axis=1)
    # # w_x0_to_x0 = numpy.sum(numpy.dot(w[ix0][:,ix0], numpy.dot(i_x0, x[:,ix0]) - numpy.dot(i_x0, x[:,ix0]).T), axis=1)
    # Coupl_to_x0 = coupling_calc(x1EQ, K, w, ix=ix0)
    #
    # fun = numpy.zeros(x1EQ.shape).astype(x1_type)
    # #Known x1eq, unknown x0:
    # # fun[:,iE] = fz_lin_calc(x1EQ[:,iE], x0[:,iE], x0cr[:,iE], rx0[:,iE], z=zEQ[:,iE],
    # #                         coupl=K[:,iE] * (w_e_to_e + w_x0_to_e)).astype(x1_type)
    # fun[:, iE] = fz_lin_calc(x1EQ[:, iE], x0[:, iE], x0cr[:, iE], rx0[:, iE], z=zEQ[:, iE],
    #                          coupl=coupling_calc(x1EQ, K, w, ix=i_x0)).astype(x1_type)
    #
    # # Known x0, unknown x1eq:
    # # fun[:,ix0] = fz_lin_calc(x[:, ix0], x0, x0cr[:, ix0], rx0[:, ix0],
    # #                          z=zeq_2d_calc(x[:, ix0], y0[:, ix0], Iext1[:, ix0]),
    # #                          coupl=K[:, ix0] * (w_e_to_x0 + w_x0_to_x0)).astype(x1_type)
    # fun[:, ix0] = fz_lin_calc(x1EQ[:, ix0], x0, x0cr[:, ix0], rx0[:, ix0],
    #                           z=zeq_2d_calc(x[:, ix0], y0[:, ix0], Iext1[:, ix0]),
    #                           coupl=Coupl_to_x0).astype(x1_type)

    # Construct the x1 and z vectors, comprising of the current x1EQ, zEQ values for i_e regions,
    # and the unknown x1 values for x1EQ and respective zEQ for the i_x0 regions
    x1EQ[:, ix0] = numpy.array(x[ix0])
    zEQ[:, ix0] = numpy.array(zeq_2d_calc(x1EQ[:, ix0], y0[:, ix0], Iext1[:, ix0]))

    # Construct the x0 vector, comprising of the current x0 values for i_x0 regions,
    # and the unknown x0 values for the i_e regions
    x0_dummy = numpy.array(x0)
    x0 = numpy.array(x1EQ)
    x0[:, iE] = numpy.array(x[iE])
    x0[:, ix0] = numpy.array(x0_dummy)
    del x0_dummy

    fun = fz_lin_calc(x1EQ, x0, x0cr, rx0, z=zEQ, coupl=coupling_calc(x1EQ, K, w)).astype(x1_type)

    # if numpy.any([numpy.any(numpy.isnan(x)), numpy.any(numpy.isinf(x)),
    #               numpy.any(numpy.isnan(fun)), numpy.any(numpy.isinf(fun))]):
    #     raise ValueError("nan or inf values in x or fun")

    return numpy.squeeze(fun)


def x1eq_x0_hypo_optimize_jac(x, ix0, iE, x1EQ, zEQ, x0, x0cr, rx0, y0, Iext1, K, w):

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


def x1eq_x0_hypo_optimize(ix0, iE, x1EQ, zEQ, x0, x0cr, rx0, y0, Iext1, K, w):

    xinit = numpy.zeros(x1EQ.shape, dtype = x1EQ.dtype)

    #Set initial conditions for the optimization algorithm, by ignoring coupling (=0)
    # fz = 4 * (x1 - r * x0 + x0cr) - z -coupling = 0
    #x0init = (x1 + x0cr -z/4) / rx0
    xinit[:, iE] = x0_calc(x1EQ[:, iE], zEQ[:, iE], x0cr[:, iE],  rx0[:, iE], 0.0)
    #x1eqinit = rx0 * x0 - x0cr + z / 4
    xinit[:, ix0] = rx0[:, ix0] * x0 - x0cr[:, ix0] + zEQ[:, ix0] / 4

    #Solve:
    sol = root(x1eq_x0_hypo_optimize_fun, xinit, args=(ix0, iE, x1EQ, zEQ, x0, x0cr, rx0, y0, Iext1, K, w),
               method='lm', jac=x1eq_x0_hypo_optimize_jac, tol=10**(-6), callback=None, options=None) #method='hybr'

    if sol.success:
        x1EQ[:, ix0] = sol.x[ix0]
        if numpy.any([numpy.any(numpy.isnan(sol.x)), numpy.any(numpy.isinf(sol.x))]):
            raise ValueError("nan or inf values in solution x\n" + sol.message)
        else:
            return x1EQ
    else:
        raise ValueError(sol.message)



def x1eq_x0_hypo_linTaylor(ix0, iE, x1EQ, zEQ, x0, x0cr, rx0, y0, Iext1, K, w):

    no_x0 = len(ix0)
    no_e = len(iE)

    n_regions = no_e + no_x0

    # The equilibria of the nodes of fixed epileptogenicity
    x1_eq = x1EQ[:, iE]
    z_eq = zEQ[:, iE]

    #Prepare linear system to solve:

    x1_type = x1EQ.dtype

    #The point of the linear Taylor expansion
    x1LIN = x1_lin_def(X1_DEF, X1_EQ_CR_DEF, n_regions).astype(x1_type)

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


def x0cr_rx0_calc(y0, Iext1, epileptor_model = "2d", zmode = numpy.array("lin"),
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
        z = zeq_2d_calc(x1, y01, I1)

    else:
        #...or for the original (>=6D) epileptor
        # fx1 = x1 ** 3 - 3 * x1 ** 2
        # #...and the z expression, coming from solving dx1/dt=f1(x1,z)=0
        # y1 = y01 - 5.0 * x1 ** 2
        # z = y1 - fx1 + I1
        z = zeq_6d_calc(x1, y1eq_calc(x1, y01, d=5.0), I1)

    #Define the fz expression...
    if zmode == 'lin':
        #...for linear...
        #fz = 4 * (x1 - r * x0 + x0cr) - z
        fz = fz_lin_calc(x1, x0, x0cr, r, z=z, coupl=0)

    elif zmode == 'sig':
        #...and sigmoidal versions
        #fz = 3/(1 + exp(-10 * (x1 + 0.5))) - r * x0 + x0cr - z
        z = fz_sig_calc(x1, x0, x0cr, r, z=z, coupl=0)

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


def coupling_calc(x1, K, w, ix=None, jx=None):
    # Note that for difference coupling it doesn't matter whether we use centered x1 or decentered x1-5/3
    # Only difference coupling for the moment.
    # TODO: Extend for different coupling forms

    n_regions = x1.size

    if ix is None:
        ix = range(n_regions)

    if jx is None:
        jx = range(n_regions)

    i_n = numpy.ones((ix.size, 1), dtype='float32')
    j_n = numpy.ones((jx.size, 1), dtype='float32')

    # Coupling                                                 from                      to
    return K[:,ix]*numpy.sum(numpy.dot(w[ix][:, jx], numpy.dot(i_n, x1[jx]) - numpy.dot(j_n, x1[ix]).T), axis=1)


def x0_calc(x1, z, x0cr, rx0, coupl, zmode=numpy.array("lin")):

    if zmode == 'lin':
        return (x1 + x0cr - (z+coupl) / 4.0) / rx0

    elif zmode == 'sig':
        return (3.0 / (1.0 + numpy.exp(-10.0 * (x1 + 0.5))) + x0cr - z + coupl) / rx0

    else:
        raise ValueError('zmode is neither "lin" nor "sig"')


def assert_equilibrium_point(epileptor_model, hypothesis, equilibrium_point):

    from tvb_epilepsy.tvb_api.epileptor_models import EpileptorDPrealistic

    coupl = coupling_calc(hypothesis.x1EQ, epileptor_model.K.T, hypothesis.weights)
    coupl = numpy.expand_dims(numpy.r_[coupl, 0.0 * coupl], 2).astype('float32')

    dfun_max = numpy.max(epileptor_model.dfun(numpy.expand_dims(equilibrium_point, 2).astype('float32'), coupl))

    if dfun_max > 10 ** -2:

        if (isinstance(epileptor_model, EpileptorDPrealistic) and
                    epileptor_model.pmode == numpy.array(['g', 'z', 'z*g'])).any():
            warnings.warn("Equilibrium point for initial condition not accurate enough!\n" \
                          + "max(dfun) = " + str(dfun_max))

        else:
            raise ValueError("Equilibrium point for initial condition not accurate enough!\n" \
                             + "max(dfun) = " + str(dfun_max))


def calc_equilibrium_point(epileptor_model, hypothesis):

    from tvb_epilepsy.tvb_api.epileptor_models import EpileptorDPrealistic, EpileptorDP2D

    if isinstance(epileptor_model,EpileptorDP2D):
        if epileptor_model.zmode == 'sig':
            #2D approximation, Proix et al 2014
            zeq = zeq_2d_calc(hypothesis.x1EQ, epileptor_model.yc.T, epileptor_model.Iext1.T)
        else:
            zeq = hypothesis.zEQ
        equilibrium_point = numpy.r_[hypothesis.x1EQ, zeq].astype('float32')
    else:
        #all >=6D models
        y1eq = y1eq_calc(hypothesis.x1EQ, epileptor_model.yc.T)
        zeq = zeq_6d_calc(hypothesis.x1EQ, y1eq, epileptor_model.Iext1.T)
        if epileptor_model.Iext2.size == 1:
            epileptor_model.Iext2 = epileptor_model.Iext2[0] * numpy.ones((hypothesis.n_regions, 1))
        (x2eq, y2eq) = pop2eq_calc(hypothesis.x1EQ, zeq, epileptor_model.Iext2.T)
        geq = geq_calc(hypothesis.x1EQ)
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