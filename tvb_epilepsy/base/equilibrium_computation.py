"""
Module to compute the resting equilibrium point of a Virtual Epileptic Patient module
"""
import warnings
import numpy
from tvb_epilepsy.base.constants import X1_DEF, X1_EQ_CR_DEF, SYMBOLIC_CALCULATIONS_FLAG
from tvb_epilepsy.base.utils import assert_arrays
from tvb_epilepsy.base.calculations import calc_x0, calc_fx1, calc_fy1, calc_fz, calc_fx2, calc_fg, calc_coupling, \
                                           calc_dfun, calc_fx1z_2d_x1neg_zpos_jac, calc_fx1y1_6d_diff_x1
from tvb_epilepsy.base.symbolic import symbol_calc_eq_x1_6d, symbol_calc_eq_x2


if SYMBOLIC_CALCULATIONS_FLAG :

    try:
        from sympy import solve
        from tvb_epilepsy.base.symbolic import symbol_vars

    except:
        warnings.warn("Unable to load sympy. Turning to scipy.optimization.")
        SYMBOLIC_CALCULATIONS_FLAG = False
        from scipy.optimize import root
else:

    from scipy.optimize import root


def def_x1eq(X1_DEF, X1_EQ_CR_DEF, n_regions):
    #The default initial condition for x1 equilibrium search
    return (X1_EQ_CR_DEF + X1_DEF) / 2.0 * numpy.ones((1,n_regions), dtype='float32')


def def_x1lin(X1_DEF, X1_EQ_CR_DEF, n_regions):
    # The point of the linear Taylor expansion
    return (X1_EQ_CR_DEF + X1_DEF) / 2.0 * numpy.ones((1,n_regions), dtype='float32')


def calc_eq_z_6d(x1eq, y1, Iext1, x2=0.0, slope=0.0, a=1.0, b=3.0, x1_neg=True):

    return calc_fx1(x1eq, z=0.0, y1=y1, Iext1=Iext1, slope=slope, a=a, b=b, tau1=1.0, x2=x2, model="6d", x1_neg=x1_neg)


def calc_eq_z_2d(x1eq, yc, Iext1, slope=0.0, a=1.0, b=-2.0, x1_neg=True):
    return calc_fx1(x1eq, z=0.0, y1=yc, Iext1=Iext1, slope=slope,  a=a, b= b, tau1=1.0, model="2d", x1_neg=x1_neg)


def calc_eq_y1(x1eq, yc, d=5.0):
    return calc_fy1(x1eq, yc, y1=0, d=d, tau1=1.0)


def calc_eq_x1_6d(zeq, yc, Iext1, slope=0.0, a=1.0, b=3.0, d=5.0, x1_neg=True):

    shape = zeq.shape

    zeq, yc, Iext1, slope, a, b, d = assert_arrays([zeq, yc, Iext1, slope, a, b, d], (zeq.size,))

    if SYMBOLIC_CALCULATIONS_FLAG:

        temp = symbol_calc_eq_x1_6d(zeq.size, x1_neg=x1_neg)[0](zeq, yc, Iext1, slope, a, b, d)

        x1eq = []
        for ii in range(zeq.size):

            temp2 = numpy.real(temp[ii][numpy.abs(numpy.imag(temp[ii])) < 10 ** -6])
            x1eq.append(numpy.min(temp2))

    else:

        #            a                          b       c=0           d
        # fx1_6d = - 1.0*a*x1**3 + x1**2*(1.0*b - 1.0*d) + 1.0*Iext1 + 1.0*yc - 1.0*z  = 0
        # which by dividing with -1/a gives us:
        #                               a2         a1=0             a0
        # fx1_6d = x1**3 - x1**2*(1.0*b - 1.0*d)/a - (1.0*Iext1 + 1.0*yc - 1.0*z)/a  = 0

        #From  http://mathworld.wolfram.com/CubicFormula.html for: x^3 + a2x^2 + a1*x + a0 = 0
        a2 = ((d-b) / a).flatten()
        a23 = 1.0/3 * a2
        #a1 = 0
        a0 = ((-Iext1 - yc + zeq) / a).flatten()
        # then p = (3a1-a2^2)/3 and q = (9a1a2 - 27a0 - 2a2^3)/27
        # and Q = 1/3*p and R = 1/2*q, so that:
        # Q = (3a1-a2^2)/9 and R = (9a1a2 - 27a0 - 2a2^3)/54 and for a1=0
        # Q = -a2^2/9 and R = -a0/2 - a2^3/27
        Q = -a2 ** 2 / 9.0
        R = - a0 / 2.0 - (a2 ** 3) / 27.0
        # and the determinant:
        # delta = Q^3 + R^3 =>
        delta = (Q ** 3 + R ** 2).astype("complex")
        S = (R + numpy.sqrt(delta)) ** (1.0 / 3)
        T = (R - numpy.sqrt(delta)) ** (1.0 / 3)
        B = S + T
        A = S - T
        BB = a23 - B / 2
        AA = 1.0 / 2 * numpy.sqrt(3) * A * numpy.ones(A.shape, dtype="complex")
        sol = numpy.concatenate([[a23 + B], [BB + AA], [BB - AA]]).T
        x1eq = []
        for ii in range(delta.size):
            temp = sol[ii, numpy.abs(numpy.imag(sol[ii])) < 10 ** (-6)]
            x1eq.append(numpy.min(numpy.real(temp)))


    # x1eq = []
        #
        # for ii in range(zeq.size):
        #
        #     fx1 = lambda x1: calc_fx1(x1, z=zeq[ii], y1=calc_eq_y1(x1, yc[ii], d=d[ii]), Iext1=Iext1[ii],
        #                               slope=slope[ii], a=a[ii], b=b[ii], tau1=1.0, x2=0.0, model="6d", x1_neg=x1_neg,
        #                               shape=(1, ))
        #
        #     jac = lambda x1: calc_fx1y1_6d_diff_x1(x1, yc[ii], Iext1[ii], a=a[ii], b=b[ii], d=d[ii], tau1=1.0,
        #                                            shape=(1, ))
        #
        #     sol = root(fx1, -2.0, method='lm', jac=jac, tol=10 ** (-6), callback=None, options=None)
        #     #args=(y2eq[ii], zeq[ii], g_eq[ii], Iext2[ii], s, tau1, tau2, x2_neg)  method='hybr'
        #
        #     if sol.success:
        #         x1eq.append(numpy.min(numpy.real(numpy.array(sol.x))))
        #         if numpy.any([numpy.any(numpy.isnan(sol.x)), numpy.any(numpy.isinf(sol.x))]):
        #             raise ValueError("nan or inf values in solution x\n" + sol.message)
        #     else:
        #         raise ValueError(sol.message)

    x1eq = numpy.reshape(x1eq, shape)

    return x1eq


def calc_eq_y2(x2eq, x2_neg=True):

    x2eq = assert_arrays([x2eq])

    return numpy.where(x2_neg, numpy.zeros(x2eq.shape), 6.0 * (x2eq + 0.25))


def calc_eq_g(x1eq, gamma=0.1):
    return calc_fg(x1eq, 0.0, gamma, tau1=1.0)


def calc_eq_x2(zeq, x1eq, Iext2,  y2eq=None, geq=None, s=6.0, gamma=0.1, x2_neg=True):

    shape = zeq.shape

    if geq is None:
        geq = calc_eq_g(x1eq, gamma)

    zeq, x1eq, Iext2, s, geq, gamma = assert_arrays([zeq, x1eq, Iext2, s, geq, gamma], (zeq.size,))

    if SYMBOLIC_CALCULATIONS_FLAG:

        temp = symbol_calc_eq_x2(zeq.size, x2_neg=x2_neg)[0](zeq, geq, Iext2, s)

        x2eq = []
        for ii in range(zeq.size):
            temp2 = numpy.real(temp[ii][numpy.abs(numpy.imag(temp[ii])) < 10 ** -6])
            x2eq.append(numpy.min(temp2))

    else:

        x2eq = []

        x2_neg = numpy.array(x2_neg)
        if x2_neg.size == 1:

            if numpy.all(zeq == zeq[0]) and numpy.all(geq == geq[0]) and numpy.all(Iext2 == Iext2[0]) and \
                numpy.all(s == s[0]):

                zeq = zeq[0]
                geq = geq[0]
                Iext2 = Iext2[0]
                s = s[0]

                n=1

            else:
                n = zeq.size
                x2_neg = numpy.tile(x2_neg, (n,))

        for ii in range(n):

            if y2eq is None:

                fx2 = lambda x2: calc_fx2(x2, y2=calc_eq_y2(x2, x2_neg=x2_neg[ii]), z=zeq[ii], g=geq[ii],
                                          Iext2=Iext2[ii], tau1=1.0, shape=(1,))

                jac = lambda x2: -3 * x2 ** 2 + 1.0 - numpy.where(x2_neg[ii], 0.0, -6.0)

            else:

                fx2 = lambda x2: calc_fx2(x2, y2=0.0, z=zeq[ii], g=geq[ii], Iext2=Iext2[ii], tau1=1.0, shape=(1,))
                jac = lambda x2: -3 * x2 ** 2 + 1.0

            sol = root(fx2, -0.75, method='lm', jac=jac, tol=10 ** (-6), callback=None, options=None)

            if sol.success:

                if numpy.any([numpy.any(numpy.isnan(sol.x)), numpy.any(numpy.isinf(sol.x))]):
                    raise ValueError("nan or inf values in solution x\n" + sol.message)

                x2eq.append(numpy.min(numpy.real(numpy.array(sol.x))))

            else:
                raise ValueError(sol.message)

    x2eq = numpy.reshape(x2eq, shape)

    return x2eq


def calc_eq_pop2(zeq, x1eq, Iext2, geq=None, s=6.0, gamma=0.1, x2_neg=True):

    x2eq = calc_eq_x2(zeq, x1eq, Iext2,  y2eq=None, geq=geq, s=s, gamma=gamma, x2_neg=x2_neg)

    y2eq = calc_eq_y2(x2eq, x2_neg=x2_neg)

    return x2eq, y2eq


def eq_x1_hypo_x0_optimize_fun(x, ix0, iE, x1EQ, zEQ, x0, x0cr, r, yc, Iext1, K, w):

    x1_type = x1EQ.dtype

    # Construct the x1 and z equilibria vectors, comprising of the current x1EQ, zEQ values for i_e regions,
    # and the unknown equilibria x1 and respective z values for the i_x0 regions
    x1EQ[:, ix0] = numpy.array(x[ix0])
    zEQ[:, ix0] = numpy.array(calc_eq_z_2d(x1EQ[:, ix0], yc[:, ix0], Iext1[:, ix0]))

    # Construct the x0 vector, comprising of the current x0 values for i_x0 regions,
    # and the unknown x0 values for the i_e regions
    x0_dummy = numpy.array(x0)
    x0 = numpy.empty_like(x1EQ)
    x0[:, iE] = numpy.array(x[iE])
    x0[:, ix0] = numpy.array(x0_dummy)
    del x0_dummy

    fun = calc_fz(x1EQ, zEQ, x0, K, w, tau1=1.0, tau0=1.0, x0cr=x0cr, r=r, zmode=numpy.array("lin"), z_pos=True,
                  model="2d").astype(x1_type)

    # if numpy.any([numpy.any(numpy.isnan(x)), numpy.any(numpy.isinf(x)),
    #               numpy.any(numpy.isnan(fun)), numpy.any(numpy.isinf(fun))]):
    #     raise ValueError("nan or inf values in x or fun")

    return fun.flatten()


def eq_x1_hypo_x0_optimize_jac(x, ix0, iE, x1EQ, zEQ, x0, x0cr, r, yc, Iext1, K, w):

    # Construct the x1 and z equilibria vectors, comprising of the current x1EQ, zEQ values for i_e regions,
    # and the unknown equilibria x1 and respective z values for the i_x0 regions
    x1EQ[:, ix0] = numpy.array(x[ix0])
    zEQ[:, ix0] = numpy.array(calc_eq_z_2d(x1EQ[:, ix0], yc[:, ix0], Iext1[:, ix0]))

    # Construct the x0 vector, comprising of the current x0 values for i_x0 regions,
    # and the unknown x0 values for the i_e regions
    x0_dummy = numpy.array(x0)
    x0 = numpy.empty_like(x1EQ)
    x0[:, iE] = numpy.array(x[iE])
    x0[:, ix0] = numpy.array(x0_dummy)
    del x0_dummy

    return calc_fx1z_2d_x1neg_zpos_jac(x1EQ, zEQ, x0, x0cr, r, yc, Iext1, K, w, ix0, iE, a=1.0, b=-2.0, tau1=1.0,
                                       tau0=1.0)


def eq_x1_hypo_x0_optimize(ix0, iE, x1EQ, zEQ, x0, x0cr, r, yc, Iext1, K, w):

    x1EQ, zEQ, x0, r, yc, Iext1, K = assert_arrays([x1EQ, zEQ, x0, r, yc, Iext1, K], (1, x1EQ.size))

    w = assert_arrays([w], (x1EQ.size, x1EQ.size))

    xinit = numpy.zeros(x1EQ.shape, dtype = x1EQ.dtype)

    #Set initial conditions for the optimization algorithm, by ignoring coupling (=0)
    # fz = 4 * (x1 - r * x0 + x0cr) - z -coupling = 0
    #x0init = (x1 + x0cr -z/4) / r
    xinit[:, iE] = calc_x0(x1EQ[:, iE], zEQ[:, iE], 0.0, 0.0, x0cr[:, iE], r[:, iE], model="2d",
                           zmode=numpy.array("lin"), z_pos=True, shape=None)
    #x1eqinit = r * x0 - x0cr + z / 4
    xinit[:, ix0] = r[:, ix0] * x0 - x0cr[:, ix0] + zEQ[:, ix0] / 4

    #Solve:
    sol = root(eq_x1_hypo_x0_optimize_fun, xinit, args=(ix0, iE, x1EQ, zEQ, x0, x0cr, r, yc, Iext1, K, w),
               method='lm', jac=eq_x1_hypo_x0_optimize_jac, tol=10**(-6), callback=None, options=None) #method='hybr'

    if sol.success:
        x1EQ[:, ix0] = sol.x[ix0]
        if numpy.any([numpy.any(numpy.isnan(sol.x)), numpy.any(numpy.isinf(sol.x))]):
            raise ValueError("nan or inf values in solution x\n" + sol.message)
        else:
            return x1EQ
    else:
        raise ValueError(sol.message)


def eq_x1_hypo_x0_linTaylor(ix0, iE, x1EQ, zEQ, x0, x0cr, r, yc, Iext1, K, w):

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
    bx0 = 4.0 * (x0cr[:, ix0] - r[:, ix0] * x0) - yc[:, ix0] - Iext1[:, ix0] \
          - 2.0 * x1LIN[:, ix0] ** 3 - 2.0 * x1LIN[:, ix0] ** 2 - K[:, ix0] * we_to_x0

    # Concatenate B vector:
    b = -numpy.concatenate((be, bx0), axis=1).T.astype(x1_type)

    # From-to Epileptogenicity-fixed regions
    # ae_to_e = -4 * numpy.eye( no_e, dtype=numpy.float32 )
    ae_to_e = -4 * numpy.diag(r[0, iE].flatten()).astype(x1_type)

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


def assert_equilibrium_point(epileptor_model, hypothesis, equilibrium_point):

    n_dim = equilibrium_point.shape[0]

    coupl = calc_coupling(equilibrium_point[0], epileptor_model.K.T, hypothesis.weights)
    coupl = numpy.expand_dims(numpy.r_[coupl, 0.0 * coupl], 2).astype('float32')

    dfun = epileptor_model.dfun(numpy.expand_dims(equilibrium_point, 2).astype('float32'), coupl).flatten()
    dfun_max = numpy.max(dfun, axis=1)
    dfun_max_cr = 10 ** -6 * numpy.ones(dfun_max.shape)

    if epileptor_model._ui_name == "EpileptorDP2D":
        dfun2 = calc_dfun(equilibrium_point[0].flatten(), equilibrium_point[1].flatten(),
                          epileptor_model.yc.flatten(), epileptor_model.Iext1.flatten(), epileptor_model.x0.flatten(),
                          epileptor_model.K.flatten(), hypothesis.weights, model_vars=n_dim,
                          x0cr=epileptor_model.x0cr.flatten(), r=epileptor_model.r.flatten(),
                          zmode=epileptor_model.zmode,
                          slope=epileptor_model.slope.flatten(), a=1.0, b=-2.0,
                          tau1=epileptor_model.tau1, tau0=epileptor_model.tau0, output_mode="array")

    elif epileptor_model._ui_name == "EpileptorDP":
        dfun_max_cr[2] = 10 ** -3
        dfun2 = calc_dfun(equilibrium_point[0].flatten(), equilibrium_point[2].flatten(),
                          epileptor_model.yc.flatten(), epileptor_model.Iext1.flatten(), epileptor_model.x0.flatten(),
                          epileptor_model.K.flatten(), hypothesis.weights, model_vars=n_dim,
                          zmode=epileptor_model.zmode,
                          y1=equilibrium_point[1].flatten(), x2=equilibrium_point[3].flatten(),
                          y2=equilibrium_point[4].flatten(), g=equilibrium_point[5].flatten(),
                          slope=epileptor_model.slope.flatten(), a=1.0, b=3.0, Iext2=epileptor_model.Iext2.flatten(),
                          tau1=epileptor_model.tau1, tau0=epileptor_model.tau0, tau2=epileptor_model.tau2,
                          output_mode="array")

    elif epileptor_model._ui_name == "EpileptorDPrealistic":
        dfun_max_cr[2] = 10 ** -3
        dfun2 = calc_dfun(equilibrium_point[0].flatten(), equilibrium_point[2].flatten(),
                          epileptor_model.yc.flatten(), epileptor_model.Iext1.flatten(), epileptor_model.x0.flatten(),
                          epileptor_model.K.flatten(), hypothesis.weights, model_vars=n_dim,
                          zmode=epileptor_model.zmode, pmode=epileptor_model.pmode,
                          y1=equilibrium_point[1].flatten(), x2=equilibrium_point[3].flatten(),
                          y2=equilibrium_point[4].flatten(), g=equilibrium_point[5].flatten(),
                          x0_var=equilibrium_point[6].flatten(), slope_var=equilibrium_point[7].flatten(),
                          Iext1_var=equilibrium_point[8].flatten(), Iext2_var=equilibrium_point[9].flatten(),
                          K_var=equilibrium_point[10].flatten(),
                          slope=epileptor_model.slope.flatten(), a=1.0, b=3.0, Iext2=epileptor_model.Iext2.flatten(),
                          tau1=epileptor_model.tau1, tau0=epileptor_model.tau0, tau2=epileptor_model.tau2,
                          output_mode="array")

    max_dfun_diff  = numpy.max(numpy.abs(dfun2 - dfun.flatten()), axis=1)
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


def calc_eq_6d(zeq, yc, Iext1, Iext2,  slope=0.0, a=1.0, b=3.0, d=5.0, gamma=0.1):

    x1eq = calc_eq_x1_6d(zeq, yc, Iext1, slope=slope, a=a, b=b, d=d)

    y1eq = calc_eq_y1(x1eq, yc, d)
    geq = calc_eq_g(x1eq, gamma)

    (x2eq, y2eq) = calc_eq_pop2(zeq, x1eq, Iext2, geq)

    equilibrium_point = numpy.r_[x1eq, y1eq, zeq, x2eq, y2eq, geq].astype('float32')

    return equilibrium_point


def calc_eq_11d(zeq, yc, Iext1, Iext2, slope, x0, K, fun_slope_Iext2, a=1.0, b=3.0, d=5.0, gamma=0.1, pmode="const"):

    x1eq = calc_eq_x1_6d(zeq, yc, Iext1, slope=slope, a=a, b=b, d=d)

    geq = calc_eq_g(x1eq, gamma)
    y1eq = calc_eq_y1(x1eq, yc, d)

    slope_eq, Iext2_eq = fun_slope_Iext2(zeq, geq, pmode, slope, Iext2)

    (x2eq, y2eq) = calc_eq_pop2(zeq, x1eq, Iext2_eq, geq)

    equilibrium_point = numpy.r_[x1eq, y1eq, zeq, x2eq, y2eq, geq, x0, slope_eq, Iext1, Iext2_eq, K].astype('float32')

    return equilibrium_point, slope_eq, Iext2_eq


def calc_equilibrium_point(epileptor_model, hypothesis):

    # Update zeq given the specific model, and assuming the hypothesis x1eq for the moment in the context of a 2d model:
    # It is assumed that the model.x0 has been adjusted already at the phase of model creation
    zeq = calc_eq_z_2d(hypothesis.x1EQ, epileptor_model.yc.T, epileptor_model.Iext1.T)

    if epileptor_model._ui_name == "EpileptorDP2D":
        equilibrium_point = numpy.r_[hypothesis.x1EQ, zeq].astype('float32')

    else:

        #all >=6D models

        if epileptor_model._ui_name == "EpileptorDPrealistic":

            equilibrium_point = calc_eq_11d(zeq, epileptor_model.yc, epileptor_model.Iext1, epileptor_model.Iext2,
                                                 epileptor_model.slope, epileptor_model.x0, epileptor_model.K,
                                                 epileptor_model.fun_slope_Iext2, epileptor_model.a, epileptor_model.b,
                                                 epileptor_model.d, gamma=0.1, pmode=epileptor_model.pmode)[0]
        else:

            #all >=6D models
            equilibrium_point = calc_eq_6d(zeq, epileptor_model.yc.T, epileptor_model.Iext1.T, epileptor_model.Iext2.T,
                                                epileptor_model.a, epileptor_model.b, epileptor_model.d, gamma=0.1)

    assert_equilibrium_point(epileptor_model, hypothesis, equilibrium_point)

    return equilibrium_point
