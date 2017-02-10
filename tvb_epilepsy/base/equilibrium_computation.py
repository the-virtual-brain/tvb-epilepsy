"""
@version $Id: equilibrium_computation.py 2017-02-10 16:08 denis $

Module to compute the resting equilibrium point of a Virtual Epileptic Patient module
"""

import numpy
from scipy.optimize import root
from tvb_epilepsy.base.constants import *
from tvb.epilpesy.hypothesis import X1_DEF, X1_EQ_CR_DEF
from tvb_epilepsy.tvb_api.epileptor_models import *
from tvb_epilepsy.tvb_api.simulator_tvb import *
from tvb.simulator import coupling


#Currently we assume only difference coupling (permittivity coupling following Proix et al 2014)
#TODO: to generalize for different coupling functions

def x1eq_def(X1_DEF, X1_EQ_CR_DEF, n_regions):
    return numpy.repeat((X1_EQ_CR_DEF - X1_DEF) / 2.0, n_regions)

def zeq_def(x1eq, y0, Iext1):
    return y0 + Iext1 - x1eq ** 3 + 3.0 * x1eq ** 2 - 5.0 * x1eq/3.0 -25.0/27.0

def y1eq_def(x1eq, y0, d=5.0):
    return y0 - d * x1eq ** 2

def pop2eq_def(n_regions, x1eq, zeq, Iext2):
    # g_eq = 0.1*x1eq (1)
    # y2eq = 0 (2)
    y2eq = np.zeros((n_regions, 1))
    # -x2eq**3 + x2eq -y2eq+2*g_eq-0.3*(zeq-3.5)+Iext2 =0=> (1),(2)
    # -x2eq**3 + x2eq +2*0.1*x1eq-0.3*(zeq-3.5)+Iext2 =0=>
    # p3        p1                   p0
    # -x2eq**3 + x2eq +0.2*x1eq-0.3*(zeq-3.5)+Iext2 =0
    p0 = 0.2 * x1eq - 0.3 * (zeq - 3.5) + Iext2
    x2eq = np.zeros((n_regions, 1))
    for i in range(n_regions):
        x2eq[i, 0] = np.min(np.real(np.roots([-1.0, 0.0, 1.0, p0[i, 0]])))
    return (x2eq, y2eq)


# def get_2eq(n_regions,x1eq,zeq,Iext2):
#    #g_eq = 0.1*x1eq (1)
#    #y2eq = 6*(x2eq+0.25)*x1eq (2)
#    #-x2eq**3 + x2eq -y2eq+2*g_eq-0.3*(zeq-3.5)+Iext2 =0=> (1),(2)
#    #-x2eq**3 + x2eq -6*(x2eq+0.25)*x1eq+2*0.1*x1eq-0.3*(zeq-3.5)+Iext2 =0=>
#    #-x2eq**3 + (1.0-6*x1eq)*x2eq -1.5*x1eq+ 0.2*x1eq-0.3*(zeq-3.5)+Iext2 =0
#    #p3                p1                           p0
#    #-x2eq**3 + (1.0-6*x1eq)*x2eq -1.3*x1eq -0.3*(zeq-3.5) +Iext2 =0
#    p0 = -1.3*x1eq-0.3*(zeq-3.5)+Iext2
#    p1 = 1.0-6*x1eq
#    x2eq = np.zeros((n_regions,1))
#    for i in range(n_regions):
#        x2eq[i,0] = np.min( np.real( np.roots([-1.0, 0.0, p1[i,0], p0[i,0] ]) ) )
#    #(2):
#    y2eq = 6*(x2eq+0.25)*x1eq
#    return (x2eq, y2eq)

def geq_def(x1eq):
    return 0.1 * x1eq


def epileptor_with_permittivity_coupling(state_variables, model_fun, weights):
    pass
    #fun_no_coupling = model_fun(state_variables, [])

    #fun_coupling_zx =

    #return fun

def jacobian_with_permittivity_coupling(state_variables, jac_fun, weights):
    pass

   # jac_no_coupling = jac_fun(state_variables,[])

    #jac_coupling_zx =

    #return jac


def compute_equilibrium_point(builder_model,hypothesis):
    epileptor_model = builder_model(hypothesis)
    args=dict()
    args['model_fun'] = epileptor_model.dfun
    args['jac_fun'] = epileptor_model.jacobian
    args['nvar'] = epileptor_model._nvar
    args['weights'] = hypothesis.normalized_weights
    args['n_regions'] = hypothesis.n_regions
    model_fun = epileptor_with_permittivity_coupling
    jac_fun = jacobian_with_permittivity_coupling
    x1o = x1eq_def(X1_DEF, X1_EQ_CR_DEF, hypothesis.n_regions)
    z0 = zeq_def(x1o, epileptor_model.y0, epileptor_model.Iext1)
    if isinstance(epileptor_model,EpileptorDP2D):
        x0 = numpy.concatenate(x1o, z0)
    else:
        x1o53 = x1o - 5.0 / 3.0
        y1o = y1eq_def(x1o53, hypothesis.y0.T)
        (x2o, y2o) = pop2eq_def(hypothesis.n_regions, x1o53, z0, epileptor_model.Iext2)
        g0 = geq_def(x1o53)
        if isinstance(epileptor_model,EpileptorDP):
            x0 = numpy.concatenate([x1o, y1o, z0, x2o, y2o, g0])
        elif isinstance(epileptor_model,EpileptorDPrealistic):
            x0 = numpy.concatenate([x1o, y1o, z0, x2o, y2o, g0, epileptor_model.x0, epileptor_model.slope,
                                    epileptor_model.Iext1, epileptor_model.Iext1, epileptor_model.K ])

    #sol = root(model_fun,x0,args=args,method=,jac=jac_fun,tol=None,options=None)
    return sol.x
