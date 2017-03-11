"""
Module to compute the resting equilibrium point of a Virtual Epileptic Patient module
"""
import warnings
import numpy
from scipy.optimize import root
from sympy import symbols, Symbol, solve, lambdify
from tvb_epilepsy.base.constants import X1_DEF, X1_EQ_CR_DEF, X0_DEF, X0_CR_DEF
from tvb_epilepsy.base.equations import calc_fx1_6d, calc_fx1_2d, calc_fy1, calc_fz, calc_fpop2, calc_fg, calc_x0, \
                                        calc_coupling, calc_dfun



def def_x1eq(X1_DEF, X1_EQ_CR_DEF, n_regions):
    #The default initial condition for x1 equilibrium search
    return (X1_EQ_CR_DEF + X1_DEF) / 2.0 * numpy.ones((1,n_regions), dtype='float32')


def def_x1lin(X1_DEF, X1_EQ_CR_DEF, n_regions):
    # The point of the linear Taylor expansion
    return (X1_EQ_CR_DEF + X1_DEF) / 2.0 * numpy.ones((1,n_regions), dtype='float32')


def calc_eq_z_6d(x1eq, y1, Iext1, x2=0.0, slope=0.0, a=1.0, b=3.0, x1_neg=True):
    return calc_fx1_6d(x1eq, z=0.0, y1=y1, x2=x2, Iext1=Iext1, slope=slope, a=a, b=b, x1_neg=x1_neg)


def calc_eq_z_2d(x1eq, yc, Iext1, slope=0.0, a=1.0, b=-2.0, x1_neg=True):

    return calc_fx1_2d(x1eq, z=0, yc=yc, Iext1=Iext1, slope=slope, a=a, b=b, tau1=1.0, x1_neg=x1_neg)


def calc_eq_y1(x1eq, yc, d=5.0):
    return calc_fy1(x1eq, yc, y1=0, d=d, tau1=1.0)


def calc_eq_pop2(x1eq, zeq, Iext2, s=6.0, tau1=1.0, tau2=1.0, x2_neg=True):

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

    x2 = numpy.array([Symbol('x2_%d' % i_n) for i_n in range(x1eq.size)])

    #TODO: use symbolic vectors and functions
    #fx2 = -y2eq + x2 - x2 ** 3 + numpy.squeeze(Iext2) + 2 * g_eq - 0.3 * (numpy.squeeze(zeq) - 3.5)
    fx2 = numpy.squeeze(calc_fpop2(x2, y2eq, zeq, g_eq, Iext2, s, tau1, tau2, x2_neg)[0])

    x2eq = []
    for ii in range(y2eq.size):
        x2eq.append(numpy.min(numpy.real(numpy.array(solve(fx2[ii], x2[ii]), dtype="complex"))))

    return numpy.reshape(numpy.array(x2eq, dtype=x1eq.dtype), x1eq.shape),  numpy.reshape(y2eq, x1eq.shape)


def calc_eq_g(x1eq):
    return calc_fg(x1eq, g=0.0, gamma=1.0, tau1=1.0)


def eq_x1_hypo_x0_optimize_fun(x, ix0, iE, x1EQ, zEQ, x0, x0cr, rx0, yc, Iext1, K, w):

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
    # #                          z=calc_eq_z_2d(x[:, ix0], yc[:, ix0], Iext1[:, ix0]),
    # #                          coupl=K[:, ix0] * (w_e_to_x0 + w_x0_to_x0)).astype(x1_type)
    # fun[:, ix0] = calc_fz_lin(x1EQ[:, ix0], x0, x0cr[:, ix0], rx0[:, ix0],
    #                           z=calc_eq_z_2d(x[:, ix0], yc[:, ix0], Iext1[:, ix0]),
    #                           coupl=Coupl_to_x0).astype(x1_type)

    # Construct the x1 and z vectors, comprising of the current x1EQ, zEQ values for i_e regions,
    # and the unknown x1 values for x1EQ and respective zEQ for the i_x0 regions
    x1EQ[:, ix0] = numpy.array(x[ix0])
    zEQ[:, ix0] = numpy.array(calc_eq_z_2d(x1EQ[:, ix0], yc[:, ix0], Iext1[:, ix0]))

    # Construct the x0 vector, comprising of the current x0 values for i_x0 regions,
    # and the unknown x0 values for the i_e regions
    x0_dummy = numpy.array(x0)
    x0 = numpy.array(x1EQ)
    x0[:, iE] = numpy.array(x[iE])
    x0[:, ix0] = numpy.array(x0_dummy)
    del x0_dummy

    fun = calc_fz(x1EQ, x0, x0cr, rx0, z=zEQ, coupl=calc_coupling(x1EQ, K, w)).astype(x1_type)

    # if numpy.any([numpy.any(numpy.isnan(x)), numpy.any(numpy.isinf(x)),
    #               numpy.any(numpy.isnan(fun)), numpy.any(numpy.isinf(fun))]):
    #     raise ValueError("nan or inf values in x or fun")

    return numpy.squeeze(fun)


def eq_x1_hypo_x0_optimize_jac(x, ix0, iE, x1EQ, zEQ, x0, x0cr, rx0, yc, Iext1, K, w):

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


def eq_x1_hypo_x0_optimize(ix0, iE, x1EQ, zEQ, x0, x0cr, rx0, yc, Iext1, K, w):

    xinit = numpy.zeros(x1EQ.shape, dtype = x1EQ.dtype)

    #Set initial conditions for the optimization algorithm, by ignoring coupling (=0)
    # fz = 4 * (x1 - r * x0 + x0cr) - z -coupling = 0
    #x0init = (x1 + x0cr -z/4) / rx0
    xinit[:, iE] = calc_x0(x1EQ[:, iE], zEQ[:, iE], x0cr[:, iE],  rx0[:, iE], 0.0)
    #x1eqinit = rx0 * x0 - x0cr + z / 4
    xinit[:, ix0] = rx0[:, ix0] * x0 - x0cr[:, ix0] + zEQ[:, ix0] / 4

    #Solve:
    sol = root(eq_x1_hypo_x0_optimize_fun, xinit, args=(ix0, iE, x1EQ, zEQ, x0, x0cr, rx0, yc, Iext1, K, w),
               method='lm', jac=eq_x1_hypo_x0_optimize_jac, tol=10**(-6), callback=None, options=None) #method='hybr'

    if sol.success:
        x1EQ[:, ix0] = sol.x[ix0]
        if numpy.any([numpy.any(numpy.isnan(sol.x)), numpy.any(numpy.isinf(sol.x))]):
            raise ValueError("nan or inf values in solution x\n" + sol.message)
        else:
            return x1EQ
    else:
        raise ValueError(sol.message)



def eq_x1_hypo_x0_linTaylor(ix0, iE, x1EQ, zEQ, x0, x0cr, rx0, yc, Iext1, K, w):

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
    bx0 = 4.0 * (x0cr[:, ix0] - rx0[:, ix0] * x0) - yc[:, ix0] - Iext1[:, ix0] \
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


def calc_x0cr_rx0(yc, Iext1, epileptor_model="2d", zmode=numpy.array("lin"),
                  x1rest=X1_DEF, x1cr=X1_EQ_CR_DEF, x0def=X0_DEF, x0cr_def=X0_CR_DEF):

    #Define the symbolic variables we need:
    (yc1, I1, x1, z, x2, x0, r, x0cr, f1, fz) = symbols('yc1 I1 x1 z x2 x0 r x0cr f1 fz')

    #Define the fx1(x1) expression (assuming centered x1 in all cases)...
    if epileptor_model == "2d":
        #...for the 2D permittivity coupling approximation, Proix et al 2014
        #fx1 = x1 ** 3 + 2 * x1 ** 2
        # #...and the z expression, coming from solving dx1/dt=f1(x1,z)=0
        # z = yc1 - fx1 + I1
        z = calc_eq_z_2d(x1, yc1, I1)

    else:
        #...or for the original (>=6D) epileptor
        # fx1 = x1 ** 3 - 3 * x1 ** 2
        # #...and the z expression, coming from solving dx1/dt=f1(x1,z)=0
        # y1 = yc1 - 5.0 * x1 ** 2
        # z = y1 - fx1 + I1
        z = calc_eq_z_6d(x1, calc_eq_y1(x1, yc1, d=5.0), x2, I1)

    #Define the fz expression...
    fz = calc_fz(x1, x0, x0cr, r, z=z, coupl=0, zmode=zmode)

    #Solve the fz expression for rx0 and x0cr, assuming the following two points (x1eq,x0) = [(-5/3,0.0),(-4/3,1.0)]...
    #...and WITHOUT COUPLING
    fz_sol = solve([fz.subs([(x1, x1rest), (x0, x0def), (z, z.subs(x1, x1rest))]),
                    fz.subs([(x1, x1cr), (x0, x0cr_def), (z, z.subs(x1, x1cr))])], r, x0cr)

    #Convert the solution of x0cr from expression to function that accepts numpy arrays as inputs:
    x0cr = lambdify((yc1,I1), fz_sol[x0cr], 'numpy')

    #Compute the actual x0cr now given the inputs yc and Iext1
    x0cr = x0cr(yc, Iext1).astype('float32')

    #The rx0 doesn' depend on yc and Iext1, therefore...
    rx0 = numpy.array(fz_sol[r]*numpy.ones(shape=x0cr.shape), dtype="float32")

    return x0cr, rx0


def assert_equilibrium_point(epileptor_model, hypothesis, equilibrium_point):

    n_dim = equilibrium_point.shape[0]

    coupl = calc_coupling(hypothesis.x1EQ, epileptor_model.K.T, hypothesis.weights)
    coupl = numpy.expand_dims(numpy.r_[coupl, 0.0 * coupl], 2).astype('float32')

    dfun = epileptor_model.dfun(numpy.expand_dims(equilibrium_point, 2).astype('float32'), coupl).squeeze()
    dfun_max = numpy.max(dfun, axis=1)
    dfun_max_cr = 10 ** -6 * numpy.ones(dfun_max.shape)

    if epileptor_model._ui_name == "EpileptorDP2D":
        dfun2 = calc_dfun(equilibrium_point[0].squeeze(), equilibrium_point[1].squeeze(),
                          epileptor_model.yc.squeeze(), epileptor_model.Iext1.squeeze(), epileptor_model.x0.squeeze(),
                          epileptor_model.x0cr.squeeze(), epileptor_model.r.squeeze(), epileptor_model.K.squeeze(),
                          hypothesis.weights, model_vars=n_dim, zmode=epileptor_model.zmode,
                          slope=epileptor_model.slope.squeeze(), a=1.0, b=-2.0,
                          tau1=epileptor_model.tau1, tau0=epileptor_model.tau0)

    elif epileptor_model._ui_name == "EpileptorDP":
        dfun_max_cr[2] = 10 ** -2
        dfun2 = calc_dfun(equilibrium_point[0].squeeze(), equilibrium_point[2].squeeze(),
                          epileptor_model.yc.squeeze(), epileptor_model.Iext1.squeeze(), epileptor_model.x0.squeeze(),
                          epileptor_model.x0cr.squeeze(), epileptor_model.r.squeeze(), epileptor_model.K.squeeze(),
                          hypothesis.weights, model_vars=n_dim, zmode=epileptor_model.zmode,
                          y1=equilibrium_point[1].squeeze(), x2=equilibrium_point[3].squeeze(),
                          y2=equilibrium_point[4].squeeze(), g=equilibrium_point[5].squeeze(),
                          slope=epileptor_model.slope.squeeze(), a=1.0, b=3.0, Iext2=epileptor_model.Iext2.squeeze(),
                          tau1=epileptor_model.tau1, tau0=epileptor_model.tau0, tau2=epileptor_model.tau2)

    elif epileptor_model._ui_name == "EpileptorDPrealistic":
        dfun_max_cr[2] = 10 ** -2
        dfun2 = calc_dfun(equilibrium_point[0].squeeze(), equilibrium_point[2].squeeze(),
                          epileptor_model.yc.squeeze(), epileptor_model.Iext1.squeeze(), epileptor_model.x0.squeeze(),
                          epileptor_model.x0cr.squeeze(), epileptor_model.r.squeeze(), epileptor_model.K.squeeze(),
                          hypothesis.weights, model_vars=n_dim, zmode=epileptor_model.zmode, pmode=epileptor_model.pmode,
                          y1=equilibrium_point[1].squeeze(), x2=equilibrium_point[3].squeeze(),
                          y2=equilibrium_point[4].squeeze(), g=equilibrium_point[5].squeeze(),
                          x0_var=equilibrium_point[6].squeeze(), slope_var=equilibrium_point[7].squeeze(),
                          Iext1_var=equilibrium_point[8].squeeze(), Iext2_var=equilibrium_point[9].squeeze(),
                          K_var=equilibrium_point[10].squeeze(),
                          slope=epileptor_model.slope.squeeze(), a=1.0, b=3.0, Iext2=epileptor_model.Iext2.squeeze(),
                          tau1=epileptor_model.tau1, tau0=epileptor_model.tau0, tau2=epileptor_model.tau2)

    max_dfun_diff  = numpy.max(numpy.abs(dfun2 - dfun.squeeze()), axis=1)
    if numpy.any(max_dfun_diff > dfun_max_cr):
        warnings.warn("model dfun and calc_dfun functions do not return the same results!\n"
                      + "maximum difference = " + str(max_dfun_diff) + "\n"
                      + "model dfun = " + str(dfun) + "\n"
                      + "calc_dfun = " + str(dfun2))

    if numpy.any(dfun_max > dfun_max_cr):
        # raise ValueError("Equilibrium point for initial condition not accurate enough!\n" \
        #                  + "max(dfun) = " + str(dfun_max) + "\n"
        #                  + "model dfun = " + str(dfun))
        warnings.warn("Equilibrium point for initial condition not accurate enough!\n"
                      + "max(dfun) = " + str(dfun_max) + "\n"
                      + "model dfun = " + str(dfun))


def calc_equilibrium_point(epileptor_model, hypothesis):

    if epileptor_model._ui_name == "EpileptorDP2D":
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
        geq = calc_eq_g(hypothesis.x1EQ)

        if epileptor_model._ui_name == "EpileptorDPrealistic":
            # the 11D "realistic" simulations model
            if epileptor_model.slope.size == 1:
                epileptor_model.slope = epileptor_model.slope[0] * numpy.ones((hypothesis.n_regions, 1))
            slope_eq, Iext2_eq = epileptor_model.fun_slope_Iext2(zeq.T, geq.T, epileptor_model.pmode,
                                                                 epileptor_model.slope, epileptor_model.Iext2)
            (x2eq, y2eq) = calc_eq_pop2(hypothesis.x1EQ, zeq, Iext2_eq.T)
            equilibrium_point = numpy.r_[hypothesis.x1EQ, y1eq, zeq, x2eq, y2eq, geq,
                                          epileptor_model.x0.T, slope_eq.T, epileptor_model.Iext1.T, Iext2_eq.T,
                                          epileptor_model.K.T].astype('float32')
        else:
            #all >=6D models
            (x2eq, y2eq) = calc_eq_pop2(hypothesis.x1EQ, zeq, epileptor_model.Iext2.T)
            equilibrium_point = numpy.r_[hypothesis.x1EQ, y1eq, zeq, x2eq, y2eq, geq].astype('float32')

    assert_equilibrium_point(epileptor_model, hypothesis, equilibrium_point)

    return equilibrium_point
