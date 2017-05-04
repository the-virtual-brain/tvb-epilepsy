"""
Mechanism for parameter search exploration for LSA and simulations (it will have TVB or custom implementations)
"""
import numpy

class PSE(object):

    def __init__(self, task, hypothesis, model=None, simulator=None, hypo_params=None, model_params=None, sim_params=None,
                 init_fun=None, run_fun=None, out_fun=None):

        if numpy.any([task is "LSA", task is "ModelSimulation", task is