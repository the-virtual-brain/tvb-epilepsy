"""
@version $Id: equilibrium_computation.py 2017-02-10 16:08 denis $

Module to compute the resting equilibrium point of a Virtual Epileptic Patient module
"""

import numpy
from scipy.optimize import root
from sympy import symbols, exp, solve, lambdify
from tvb_epilepsy.tvb_api.epileptor_models import *
from tvb_epilepsy.tvb_api.simulator_tvb import *
from tvb.simulator.models import Epileptor


#Currently we assume only difference coupling (permittivity coupling following Proix et al 2014)
#TODO: to generalize for different coupling functions

def x1eq_def(X1_DEF, X1_EQ_CR_DEF, n_regions):
    return numpy.repeat((X1_EQ_CR_DEF - X1_DEF) / 2.0, n_regions)


def fx1_2d_calc(x1):
    return x1**3 + 2*x1**2


def fx1_6d_calc(x1):
    return x1**3 - 3*x1**2


def fz_lin_calc(x1,x0,x0cr,r):
    return 4*(x1-r*x0+x0cr)


def x1eq_x0_hypo_optimize_fun(x, ix0, iE, x1EQ, zEQ, x0, x0cr, rx0, y0, Iext1, K, w):

    no_x0 = len(ix0)
    no_e = len(iE)

    type = x1EQ.dtype
    i_e = numpy.ones((no_e,1), dtype=type)
    i_x0 = numpy.ones((no_x0,1), dtype=type)

    #Coupling                        to   from           from                    to
    w_e_to_e = numpy.sum(numpy.dot(w[iE][:,iE],    numpy.dot(i_e, x1EQ[:,iE]) - numpy.dot(i_e, x1EQ[:,iE]).T), axis=1)
    w_x0_to_e = numpy.sum(numpy.dot(w[iE][:, ix0], numpy.dot(i_e, x0) - numpy.dot(i_x0, x1EQ[:,iE]).T), axis=1)

    w_e_to_x0 = numpy.sum(numpy.dot(w[ix0][:,iE],  numpy.dot(i_x0, x1EQ[:,iE]) - numpy.dot(i_e, x0).T), axis=1)
    w_x0_to_x0 = numpy.sum(numpy.dot(w[ix0][:,ix0], numpy.dot(i_x0, x0) - numpy.dot(i_x0, x0).T), axis=1)

    fun = numpy.array(x1EQ.shape)
    #Known x1eq, unknown x0:
    fun[iE] = fz_lin_calc(x1EQ[iE], x[iE], x0cr[iE], rx0[iE]) - zEQ[iE] \
                                         - K[iE] * (w_e_to_e + w_x0_to_e)
    # Known x0, unknown x1eq:
    fun[ix0] = fz_lin_calc(x[ix0], x0, x0cr[ix0], rx0[ix0]) - zeq_2d_calc(x[ix0]-5.0/3, y0[ix0], Iext1[ix0]) \
                                        - K[ix0] * (w_e_to_x0 + w_x0_to_x0)

    return fun


def x1eq_x0_hypo_optimize_jac(x, ix0, iE, x1EQ, zEQ, x0, x0cr, rx0, y0, Iext1, K, w):

    no_x0 = len(ix0)
    no_e = len(iE)

    n_regions = no_e + no_x0

    type = x1EQ.dtype
    i_x0 = numpy.ones((no_x0, 1), dtype=type)

    jac_e_x0e = numpy.diag(- 4 * rx0[iE])
    jac_e_x1o = -numpy.dot(i_x0, K[:,iE]) * w[iE][:,ix0]
    jac_x0_x0e = numpy.zeros((no_x0,no_e),dtype = type)
    x53 = x[ix0] - 5.0 / 3
    jac_x0_x1o = numpy.diag(4 + 3 * x53 ** 2 + 4 * x53 + K[ix0] * numpy.sum(w[ix0][:,ix0], axis=1)) \
                 - numpy.dot(i_x0, K[:, ix0]) * w[ix0][:, ix0]

    jac = numpy.zeros((n_regions,n_regions), dtype=type)
    jac[iE][:,iE] = jac_e_x0e
    jac[iE][:, ix0] = jac_e_x1o
    jac[ix0][:, iE] = jac_x0_x0e
    jac[ix0][:, ix0] = jac_x0_x1o

    return jac


def x1eq_x0_hypo_optimize(ix0, iE, x1EQ, zEQ, x0, x0cr, rx0, y0, Iext1, K, w):

    xinit = numpy.zeros(x1EQ.shape, dtype = x1EQ.dtype)

    #Set initial conditions for the optimization algorithm, by ignoring coupling
    # fz = 4 * (x1 - r * x0 + x0cr) - z -coupling = 0
    #x0init = x1 + x0cr -z/(4*rx0)
    xinit[:,iE] = x1EQ[:, iE] + x0cr[:, iE] - zEQ[:, iE] / (4 * rx0[:, iE])
    #x1eqinit = x0cr-rx0*x0 +z/4
    xinit[:, ix0] = x0cr[:,ix0] - rx0[:,ix0]*x0 + zEQ[:,ix0]/4

    #Solve:
    sol = root(x1eq_x0_hypo_optimize_fun, xinit, args=(ix0, iE, x1EQ, zEQ, x0, x0cr, rx0, y0, Iext1, K, w),
               method='lm', jac=x1eq_x0_hypo_optimize_jac, tol=10**(-6), callback=None, options=None) #method='hybr'

    if sol.success:
        x1EQ[:,ix0] = sol.x[:,ix0]
        return x1EQ
    else:
        raise ValueError(sol.message)



def x1eq_x0_hypo_linTaylor(ix0,iE,x1EQ,zEQ,x0,x0cr,x1LIN,rx0,y0,Iext1,K,w):

    no_x0 = len(ix0)
    no_e = len(iE)

    # The equilibria of the nodes of fixed epileptogenicity
    x1_eq = x1EQ[:, iE]
    z_eq = zEQ[:, iE]

    #Prepare linear system to solve:

    # For regions of fixed equilibria:
    ii_e = numpy.ones((1, no_e), dtype=numpy.float32)
    we_to_e = numpy.expand_dims(numpy.sum(w[iE][:, iE] * (numpy.dot(ii_e.T, x1_eq) -
                                                    numpy.dot(x1_eq.T, ii_e)), axis=1), 1).T
    wx0_to_e = -x1_eq * numpy.expand_dims(numpy.sum(w[ix0][:, iE], axis=0), 0)
    be = 4.0 * (x1_eq + x0cr[:, iE]) - z_eq - K[:, iE] * (we_to_e + wx0_to_e)

    # For regions of fixed x0:
    ii_x0 = numpy.ones((1, no_x0), dtype=numpy.float32)
    we_to_x0 = numpy.expand_dims(numpy.sum(w[ix0][:, iE] * numpy.dot(ii_x0.T, x1_eq), axis=1), 1).T
    #        bx0 = 4 * (self.x0cr[:, ix0] - x0) - self.y0[:, ix0] - self.Iext1[:, ix0] \
    #            - 2 * self.x1LIN[:, ix0] ** 3 - 2 * self.x1LIN[:, ix0] ** 2 - self.K[:, ix0] * we_to_x0
    bx0 = 4.0 * (x0cr[:, ix0] - rx0[:, ix0] * x0) - y0[:, ix0] - Iext1[:, ix0] \
          - 2.0 * x1LIN[:, ix0] ** 3 + 3.0 * x1LIN[:, ix0] ** 2 + 25.0 / 27.0 - K[:, ix0] * we_to_x0

    # Concatenate B vector:
    b = -numpy.concatenate((be, bx0), axis=1).T

    # From-to Epileptogenicity-fixed regions
    # ae_to_e = -4 * numpy.eye( no_e, dtype=numpy.float32 )
    ae_to_e = -4 * numpy.diag(rx0[0, iE])

    # From x0-fixed regions to Epileptogenicity-fixed regions
    ax0_to_e = -numpy.dot(K[:, iE].T, ii_x0) * w[iE][:, ix0]

    # From Epileptogenicity-fixed regions to x0-fixed regions
    ae_to_x0 = numpy.zeros((no_x0, no_e), dtype=numpy.float32)

    # From-to x0-fixed regions
    #        ax0_to_x0 = numpy.diag((4 + 3 * self.x1LIN[:, ix0] ** 2 + 4 * self.x1LIN[:, ix0]  \
    #                  + self.K[0, ix0] *numpy.expand_dims(numpy.sum(w[ix0][:, ix0], axis=0), 0)).T[:, 0])  \
    #                  - numpy.dot(self.K[:, ix0].T, ii_x0) * w[ix0][:, ix0]
    ax0_to_x0 = numpy.diag((4.0 + 3.0 * (x1LIN[:, ix0] ** 2 - 2.0 * x1LIN[:, ix0] + 5.0 / 9.0) +
                         K[0, ix0] * numpy.expand_dims(numpy.sum(w[ix0][:, ix0], axis=0), 0)).T[:, 0]) \
                - numpy.dot(K[:, ix0].T, ii_x0) * w[ix0][:, ix0]

    # Concatenate A matrix
    a = numpy.concatenate((numpy.concatenate((ae_to_e, ax0_to_e), axis=1),
                        numpy.concatenate((ae_to_x0, ax0_to_x0), axis=1)),
                       axis=0)

    # Solve the system
    x = numpy.dot(numpy.linalg.inv(a), b).T

    # Unpack solution:
    # The equilibria of the regions with fixed E have not changed:
    # The equilibria of the regions with fixed x0:
    x1EQ[0, ix0] = x[0, no_e:]

    return x1EQ


#In all cases below, x1eq is already de-centered, i.e., x1eq - 5/3 -> x1eq


def zeq_2d_calc(x1eq, y0, Iext1):
    return y0 + Iext1 -x1eq**3 - 2*x1eq**2


def zeq_6d_calc(x1eq, y0, Iext1):
    return y0 + Iext1 -x1eq**3 + 3*x1eq**2


def y1eq_calc(x1eq, y0, d=5.0):
    return y0 - d * x1eq ** 2


def pop2eq_calc(x1eq, zeq, Iext2):
    n_regions = len(Iext2)
    # g_eq = 0.1*x1eq (1)
    # y2eq = 0 (2)
    y2eq = numpy.zeros((n_regions, 1))
    # -x2eq**3 + x2eq -y2eq+2*g_eq-0.3*(zeq-3.5)+Iext2 =0=> (1),(2)
    # -x2eq**3 + x2eq +2*0.1*x1eq-0.3*(zeq-3.5)+Iext2 =0=>
    # p3        p1                   p0
    # -x2eq**3 + x2eq +0.2*x1eq-0.3*(zeq-3.5)+Iext2 =0
    p0 = 0.2 * x1eq - 0.3 * (zeq - 3.5) + Iext2
    x2eq = numpy.zeros((n_regions, 1))
    for i in range(n_regions):
        x2eq[i, 0] = numpy.min(numpy.real(numpy.roots([-1.0, 0.0, 1.0, p0[i, 0]])))
    return (x2eq, y2eq)


# def pop2eq_calc(n_regions,x1eq,zeq,Iext2):
#    #g_eq = 0.1*x1eq (1)
#    #y2eq = 6*(x2eq+0.25)*x1eq (2)
#    #-x2eq**3 + x2eq -y2eq+2*g_eq-0.3*(zeq-3.5)+Iext2 =0=> (1),(2)
#    #-x2eq**3 + x2eq -6*(x2eq+0.25)*x1eq+2*0.1*x1eq-0.3*(zeq-3.5)+Iext2 =0=>
#    #-x2eq**3 + (1.0-6*x1eq)*x2eq -1.5*x1eq+ 0.2*x1eq-0.3*(zeq-3.5)+Iext2 =0
#    #p3                p1                           p0
#    #-x2eq**3 + (1.0-6*x1eq)*x2eq -1.3*x1eq -0.3*(zeq-3.5) +Iext2 =0
#    p0 = -1.3*x1eq-0.3*(zeq-3.5)+Iext2
#    p1 = 1.0-6*x1eq
#    x2eq = numpy.zeros((n_regions,1))
#    for i in range(n_regions):
#        x2eq[i,0] = numpy.min( numpy.real( numpy.roots([-1.0, 0.0, p1[i,0], p0[i,0] ]) ) )
#    #(2):
#    y2eq = 6*(x2eq+0.25)*x1eq
#    return (x2eq, y2eq)

def geq_calc(x1eq):
    return 0.1 * x1eq


def x0cr_rx0_calc(y0, Iext1, epileptor_model="2d", zmode=numpy.array("lin")):

    #Define the symbolic variables we need:
    (y01, I1, x1, z, x0, r, x0cr, f1, fz) = symbols('y01 I1 x1 z x0 r x0cr f1 fz')

    #Define the fx1(x1) expression (assuming centered x1 in all cases)...
    if isinstance(epileptor_model,EpileptorDP2D) or  epileptor_model=="2d":
        #...for the 2D permittivity coupling approximation, Proix et al 2014
        fx1 = (x1 - 5.0 / 3) ** 3 + 2 * (x1 - 5.0 / 3) ** 2
    else:
        #...or for the original (>=6D) epileptor
        fx1 = (x1 - 5.0 / 3) ** 3 - 3 * (x1 - 5.0 / 3) ** 2

    #...and the z expression, coming from solving dx1/dt=f1(x1,z)=0
    z = y01 - fx1 + I1

    #Define the fz expression...
    if zmode=='lin':
        #...for linear...
        fz = 4 * (x1 - r * x0 + x0cr ) - z
    elif zmode == 'sig':
        #...and sigmoidal versions
        fz = 4 * ((1+exp(-10*(x1-7.0/6))) - r * x0 + x0cr) - z
    else:
        raise ValueError('zmode is neither "lin" nor "sig"')

    #Solve the fz expression for rx0 and x0cr, assuming the following two points (x1eq,x0) = [(0.0,0.0),(1/3,1.0)]...
    #...and WITHOUT COUPLING
    fz_sol = solve([fz.subs([(x1, 0.0), (x0, 0.0), (z, z.subs(x1, 0.0))]),
                       fz.subs([(x1, 1.0 / 3), (x0, 1.0), (z, z.subs(x1, 1.0 / 3))])], r, x0cr)

    #Convert the solutions from expressions to functions that accept numpy arrays as inputs:
    x0cr = lambdify((y01,I1),fz_sol[x0cr],'numpy')

    #Compute the actual x0cr now given the inputs y0 and Iext1
    x0cr = x0cr(y0, Iext1)

    #The rx0 doesn' depend on y0 and Iext1, therefore...
    rx0 = fz_sol[r]*numpy.ones(shape=x0cr.shape)

    return x0cr, rx0


def coupling_calc(x1, K, w):
    # Only difference coupling for the moment.
    # TODO: Extend for different coupling forms
    n_regions = x1.size
    i_n = numpy.ones((n_regions,1), dtype='f')
    # Coupling                         from                    to
    return K*numpy.sum(numpy.dot(w, numpy.dot(i_n,x1) - numpy.dot(i_n,x1).T), axis=1)


def x0_calc(x1, z, x0cr, rx0, coupl, zmode=numpy.array("lin")):

    if zmode=='lin':
        return x1 + x0cr - (z+coupl) / (4 * rx0)
    elif zmode=='sig':
        return 3/(1+numpy.exp(-10*(x1-7.0/6))) + x0cr - (z + coupl) / (4 * rx0)
    else:
        raise ValueError('zmode is neither "lin" nor "sig"')


def calc_equilibrium_point(epileptor_model, hypothesis):

    #Get the x1 equilibria from the hypothesis:
    x1eq =  hypothesis.x1EQ
    #De-center them:
    x1eq53 = x1eq - 5.0 / 3.0
    if isinstance(epileptor_model,EpileptorDP2D):
        if epileptor_model.zmode=='sig':
            #2D approximation, Proix et al 2014
            zeq = zeq_2d_calc(x1eq53, epileptor_model.y0, epileptor_model.Iext1)
        return (x1eq,zeq)
    else:
        #all other >=6D models
        zeq = zeq_6d_calc(x1eq53, epileptor_model.y0, epileptor_model.Iext1)
        y1eq=y1eq_calc(x1eq53, epileptor_model.y0)
        (x2eq,y2eq)=pop2eq_calc(x1eq53, zeq, epileptor_model.Iext2)
        geq=geq_calc(x1eq)
        if isinstance(epileptor_model, EpileptorDPrealistic):
            #the 11D "realistic" simulations model
            return (x1eq, y1eq, zeq, x2eq, y2eq, epileptor_model.x0, epileptor_model.slope,
                    epileptor_model.Iext1, geq, epileptor_model.Iext2, epileptor_model.K)
        elif isinstance(epileptor_model, Epileptor):
            #the original 6D TVB model with de-centered x1
            return (x1eq53, y1eq, zeq, x2eq, y2eq, geq)
        else:
            #the default 6D model we use for tvb-epilepsy
            return (x1eq, y1eq, zeq, x2eq, y2eq, geq)

