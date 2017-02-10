"""
@version $Id: hypothesis.py 1588 2016-08-18 23:44:14Z denis $

Class for defining and configuring disease hypothesis (epilepsy hypothesis).
It should contain everything for later configuring an Epileptor Model from this hypothesis.

"""

import numpy as np
from collections import OrderedDict
from tvb_epilepsy.base.equilibrium_computation import zeq_2d_calc, y1eq_calc,\
                                                      x1eq_x0_hypo_linTaylor, x1eq_x0_hypo_optimize
from tvb_epilepsy.base.utils import reg_dict, formal_repr, vector2scalar


# Default model parameters
X0_DEF = 0.0
E_DEF = 0.0
K_DEF = 1.0
I_EXT1_DEF = 3.1
Y0_DEF = 1.0
X1_DEF = 0.0
X1_EQ_CR_DEF = 1.0 / 3.0
X1_LIN_DEF = (X1_EQ_CR_DEF-X1_DEF)/2.0
X1_SQ_DEF = X1_EQ_CR_DEF

#Currently we assume only difference coupling (permittivity coupling following Proix et al 2014
#TODO: to generalize for different coupling functions
class Hypothesis(object):
    def __init__(self, n_regions, normalized_weights, name="", x1eq_mode = "optimize",
                 e_def=E_DEF, k_def=K_DEF, i_ext1_def=I_EXT1_DEF, y0_def=Y0_DEF,
                 x1_eq_cr_def=X1_EQ_CR_DEF, x1_lin_def=X1_LIN_DEF, x1_sq_def=X1_SQ_DEF):

        #TODO: question the course below. Maybe use the opposite one?
        """
        At initalization we follow the course:

            E->equilibria->x0

        Notice that epileptogenicities (and therefore equilibria) can overwrite excitabilities!!
        """
        self.name = name
        self.n_regions = n_regions
        self.weights = normalized_weights

        i = np.ones((1, self.n_regions), dtype=np.float32)
        self.K = k_def * i
        self.Iext1 = i_ext1_def * i
        self.y0 = y0_def * i
        self.x1EQcr = x1_eq_cr_def
        self.x1LIN = x1_lin_def * i
        self.x1SQ = x1_sq_def * i
        self.E = e_def * i

        self.x0cr = self._calculate_critical_x0()
        self.rx0 = self._calculate_x0_scaling()
        self.x1eq_mode = x1eq_mode
        self.x1EQ = self._calculate_equilibria_x1(i)
        self.zEQ = self._calculate_equilibria_z()
        self.Ceq = self._calculate_coupling_at_equilibrium(i, normalized_weights)
        self.x0 = self._calculate_x0()

        # Region indices assumed to start the seizure
        self.seizure_indices = np.array([], dtype=np.int32)
        self.lsa_ps = []

    def __repr__(self):
        d = {"01.name": self.name,
             "02.K": vector2scalar(self.K),
             "03.Iext1": vector2scalar(self.Iext1),
             "04.seizure indices": self.seizure_indices,
             "05. no of seizure nodes": self.n_seizure_nodes,
             "06. x0": reg_dict(self.x0, sort = 'descend'),
             "07. E": reg_dict(self.E, sort = 'descend'),
             "08. PSlsa": reg_dict(self.lsa_ps, sort = 'descend'),
             "09. x1EQ": reg_dict(self.x1EQ, sort = 'descend'),
             "10. zEQ": reg_dict(self.zEQ, sort = 'ascend'),
             "11. Ceq": reg_dict(self.Ceq, sort = 'descend'),
             "12. weights for seizure nodes": self.weights_for_seizure_nodes,
             "13. x1EQcr": vector2scalar(self.x1EQcr),
             "14. x1LIN": vector2scalar(self.x1LIN),
             "15. x11SQ": vector2scalar(self.x1SQ),
             "16. x0cr": vector2scalar(self.x0cr),
             "17. rx0": vector2scalar(self.rx0),
             "18. x1eq_mode": self.x1eq_mode}
        return formal_repr(self, OrderedDict(sorted(d.items(), key=lambda t: t[0]) ))
                                                               

    def __str__(self):
        return self.__repr__()

    @property
    def n_seizure_nodes(self):
        """
        :return: The number of hypothesized epileptogenic regions is also
        the number of eigenectors used for the calculation of the Propagation Strength index
        """
        return len(self.seizure_indices)

    @property
    def weights_for_seizure_nodes(self):
        """
        :return: Connectivity weights from epileptogenic/seizure starting regions to the rest
        """
        return self.weights[:, self.seizure_indices]

    def _calculate_critical_x0(self):
        # At the hypothesis level, we assume linear z function
        return self.Iext1 / 4.0 + self.y0 / 4.0 - 25.0 / 108.0  # for linear z dfun
        # return self.Iext1 / 4 + self.y0 / 4 + 28.0 / 27.0  # for linear z dfun

    def _calculate_x0_scaling(self):
        # At the hypothesis level, we assume linear z function
        return -self.Iext1 / 4.0 - self.y0 / 4.0 + 1537.0 / 1080.0  # for linear z dfun

    def _calculate_equilibria_x1(self, i=None):
        if i is None:
            i = np.ones((1, self.n_regions), dtype=np.float32)
        return (self.E / 3.0) * i
        # return self.E[1, i] / 3.0
        # return ((self.E - 4.0) / 3.0) * i

    def _calculate_equilibria_z(self):
        # y0 + Iext1 - x1eq ** 3 + 3.0 * x1eq ** 2 - 5.0 * x1eq/3.0 -25.0/27.0
        return zeq_2d_calc(self.x1EQ, self.y0, self.Iext1)
        #non centered x1:
        # return self.y0 + self.Iext1 - self.x1EQ ** 3 - 2.0 * self.x1EQ ** 2

    def _calculate_coupling_at_equilibrium(self, i, w):
        return self.K * (np.expand_dims(np.sum(w * ( np.dot(i.T, self.x1EQ) - np.dot(self.x1EQ.T, i)), axis=1), 1).T)

    def _calculate_x0(self):
        return (self.x1EQ + self.x0cr - (self.zEQ + self.Ceq) / 4.0) / self.rx0
        # return self.x1EQ + self.x0cr - (self.zEQ + self.Ceq) / 4.0

    def _calculate_e(self):
        return 3.0 * self.x1EQ

    def get_yeq(self):
        return y1eq_calc(self.x1EQ)


    def _update_parameters(self, seizure_indices):
        """
        Updating hypothesis always starts from a new equilibrium point
        :param seizure_indices: numpy array with conn region indices where we think the seizure starts
        """
        i = np.ones((1, self.n_regions), dtype=np.float32)
        self.x0cr = self._calculate_critical_x0()
        self.rx0 = self._calculate_x0_scaling()
        self.Ceq = self._calculate_coupling_at_equilibrium(i, self.weights)
        self.x0 = self._calculate_x0()
        self.E = self._calculate_e()

        self.seizure_indices = seizure_indices
        if self.n_seizure_nodes > 0:
            self._run_lsa(seizure_indices)


    def _run_lsa(self, seizure_indices):

        self._check_hypothesis(seizure_indices)
        i = np.ones((1, self.n_regions), dtype=np.float32)
        # The z derivative of the x1 = F(z) function
        # dfz = (3.0 / 4.0 * np.sqrt(6 / (27.0 * (self.zEQ - self.y0 - self.Iext1 + 32)))) * i
        dfz = (1.0 / np.sqrt(8 * (self.zEQ - self.y0 - self.Iext1) + 256.0 / 27.0)) * i

        # Jacobian: diagonal elements at first row
        jacobian = np.diag((dfz * 4.0 + self.K * np.expand_dims(np.sum(self.weights, axis=1), 1).T).T[:, 0]) \
            - np.dot(self.K.T, i) * np.dot(i.T, dfz) * (1 - np.eye(self.n_regions))

        # Perform eigenvalue decomposition
        (eigvals, eigvects) = np.linalg.eig(jacobian)
        
        # Sort eigenvalues in descending order... 
        ind = np.argsort(eigvals, kind='mergesort')[::-1]
        self.lsa_eigvals = eigvals[ind]
        #...and eigenvectors accordingly
        self.lsa_eigvects = eigvects[:, ind]
        
        #Calculate the propagation strength index by summing the first n_seizure_nodes eigenvectors
        self.lsa_ps = np.expand_dims(np.sum(np.abs(self.lsa_eigvects[:, :self.n_seizure_nodes]), axis=1), 1).T
        
        #Calculate the propagation strength index by summing all eigenvectors
        self.lsa_ps_tot = np.expand_dims(np.sum(np.abs(self.lsa_eigvects), axis=1), 1).T


    def _check_hypothesis(self, seizure_indices):
        """
         LSA doesn't work well if there are some E>1 (i.e., x1EQ>1/3),
        and at the same time the rest of the equilibria are not negative "enough"
         Suggested correction for the moment to ceil x1EQ to the critical x1EQcr = 1/3,
        and then update the whole hypothesis accordingly. We should ask the user for this..
        """

        temp = self.x1EQ > self.x1EQcr
        if temp.any():
            self.x1EQ[temp] = self.x1EQcr
            self.zEQ = self._calculate_equilibria_z()

            # Now that equilibria are OK, update the hypothesis to get the actual x0, E etc
            self._update_parameters(seizure_indices)


    # The two hypothesis modes below could be combined (but always starting from "E" first, if any)

    def configure_e_hypothesis(self, ie, e, seizure_indices):
        """
        Configure hypothesis starting from Epileptogenicities E
        :param e: new Epileptogenicities E
        :param ie: indices where the new E should be set
        :param seizure_indices: Indices where seizure starts
        """
        self.E[0, ie] = e
        self.x1EQ = self._calculate_equilibria_x1()
        self.zEQ = self._calculate_equilibria_z()

        self._update_parameters(seizure_indices)


    def configure_x0_hypothesis(self, ix0, x0, seizure_indices):
        """
        Hypothesis starting from Excitabilities x0
        :param ix0: indices of regions with a x0 hypothesis
        :param x0: the x0 hypothesis for the regions of ix0 indices
        :param seizure_indices: Indices where seizure starts
        """
        # Create region indices:
        # All regions
        ii = np.array(range(self.n_regions), dtype=np.int32)
        # All regions with an Epileptogenicity hypothesis:
        iE = np.delete(ii, ix0)  # their indices

        # ...and the resulting equilibria
        x1_eq = self.x1EQ[:, iE]
        z_eq = self.zEQ[:, iE]

        if self.self.x1eq_mode=="linTaylor":
            self.x1EQ = x1eq_x0_hypo_linTaylor(ix0, iE, self.x1EQ, x1_eq, z_eq, x0, self.x0cr, self.x1LIN, self.rx0,
                                           self.y0, self.Iext1, self.K, self.weights)
        else:
            self.x1EQ = x1eq_x0_hypo_optimize(ix0, iE, self.x1EQ, x1_eq, z_eq, x0, self.x0cr, self.rx0, self.y0,
                                              self.Iext1, self.K, self.weights)

        self.zEQ = self._calculate_equilibria_z()

        # Now that equilibria are OK, update the hypothesis to get the actual x0, E etc
        self._update_parameters(seizure_indices)
