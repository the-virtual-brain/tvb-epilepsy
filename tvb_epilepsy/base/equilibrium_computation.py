"""
@version $Id: equilibrium_computation.py 2017-02-10 16:08 denis $

Module to compute the resting equilibrium point of a Virtual Epileptic Patient module
"""

import numpy
from scipy.optimize import root
from tvb_epilepsy.tvb_api.epileptor_models import *
from tvb_epilepsy.tvb_api.simulator_tvb import *
from tvb.simulator.models import Epileptor


#Currently we assume only difference coupling (permittivity coupling following Proix et al 2014)
#TODO: to generalize for different coupling functions

def x1eq_def(X1_DEF, X1_EQ_CR_DEF, n_regions):
    return numpy.repeat((X1_EQ_CR_DEF - X1_DEF) / 2.0, n_regions)

def x1eq_x0_hypo_optimize(ix0,iE,x1EQ,x1_eq,z_eq,x0,x0cr,rx0,y0,Iext1,K,w):
    #fun =
    pass

def x1eq_x0_hypo_linTaylor(ix0,iE,x1EQ,x1_eq,z_eq,x0,x0cr,x1LIN,rx0,y0,Iext1,K,w):

    no_x0 = len(ix0)
    no_e = len(iE)

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


def zeq_2d_calc(x1eq, y0, Iext1):
    return y0 + Iext1 - x1eq ** 3 + 3.0 * x1eq ** 2 - 5.0 * x1eq/3.0 -25.0/27.0

def zeq_calc(x1eq, y0, Iext1):
    return y0 + Iext1 - x1eq ** 3 + 3.0 * x1eq ** 2 - 5.0 * x1eq/3.0 -25.0/27.0

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


def calc_equilibrium_point(epileptor_model,hypothesis):

    x1eq =  hypothesis.x1EQ

    if isinstance(epileptor_model,EpileptorDP2D):
        zeq = zeq_2d_calc(x1eq, epileptor_model.y0, epileptor_model.Iext1)
        return (x1eq,zeq)
    else:
        x1eq53 = x1eq - 5.0 / 3.0
        zeq = zeq_calc(x1eq53, epileptor_model.y0, epileptor_model.Iext1)
        y1eq=y1eq_calc(x1eq53, epileptor_model.y0)
        (x2eq,y2eq)=pop2eq_calc(x1eq53, zeq, epileptor_model.Iext2)
        geq=geq_calc(x1eq)
        if isinstance(epileptor_model, EpileptorDPrealistic):
            return (x1eq, y1eq, zeq, x2eq, y2eq, epileptor_model.x0, epileptor_model.slope,
                    epileptor_model.Iext1, geq, epileptor_model.Iext2, epileptor_model.K)
        elif isinstance(epileptor_model, Epileptor):
            return (x1eq53, y1eq, zeq, x2eq, y2eq, geq)
        else:
            return (x1eq, y1eq, zeq, x2eq, y2eq, geq)

