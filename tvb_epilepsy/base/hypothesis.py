"""
@version $Id: hypothesis.py 1588 2016-08-18 23:44:14Z denis $

Class for defining and configuring disease hypothesis (epilepsy hypothesis).
It should contain everything for later configuring an Epileptor Model from this hypothesis.

"""

import numpy
from collections import OrderedDict
from tvb_epilepsy.base.constants import X0_DEF, E_DEF, K_DEF, I_EXT1_DEF, Y0_DEF, X1_DEF, X1_EQ_CR_DEF
from tvb_epilepsy.base.equilibrium_computation import zeq_2d_calc, coupling_calc, x0_calc, x0cr_rx0_calc, \
                                                      x1eq_x0_hypo_linTaylor, x1eq_x0_hypo_optimize, x1_lin_def
from tvb_epilepsy.base.utils import reg_dict, formal_repr, vector2scalar



#Currently we assume only difference coupling (permittivity coupling following Proix et al 2014
#TODO: to generalize for different coupling functions
class Hypothesis(object):
    def __init__(self, n_regions, normalized_weights, name="", x1eq_mode = "optimize",
                 e_def=E_DEF, k_def=K_DEF, i_ext1_def=I_EXT1_DEF, y0_def=Y0_DEF,
                 x1_eq_cr_def=X1_EQ_CR_DEF):

        #TODO: question the course below. Maybe use the opposite one?
        """
        At initalization we follow the course:

            E->equilibria->x0

        Notice that epileptogenicities (and therefore equilibria) can overwrite excitabilities!!
        """
        self.name = name
        self.n_regions = n_regions
        self.weights = normalized_weights

        i = numpy.ones((1, self.n_regions), dtype=numpy.float32)
        self.K = k_def * i
        self.Iext1 = i_ext1_def * i
        self.y0 = y0_def * i
        self.x1EQcr = x1_eq_cr_def
        self.E = e_def * i

        self.x1LIN = x1_lin_def(X1_DEF, X1_EQ_CR_DEF, n_regions)
        self.x1SQ = X1_EQ_CR_DEF

        (self.x0cr, self.rx0)= self._calculate_critical_x0_scaling()
        self.x1eq_mode = x1eq_mode
        self.x1EQ = self._set_equilibria_x1(i)
        self.zEQ = self._calculate_equilibria_z()
        self.Ceq = self._calculate_coupling_at_equilibrium()
        self.x0 = self._calculate_x0()

        # Region indices assumed to start the seizure
        self.seizure_indices = numpy.array([], dtype=numpy.int32)
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
             "15. x1SQ": vector2scalar(self.x1SQ),
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

    def _calculate_critical_x0_scaling(self):
        return x0cr_rx0_calc(self.y0, self.Iext1, epileptor_model="2d", zmode="lin")

    def _set_equilibria_x1(self, i=None):
        if i is None:
            i = numpy.ones((1, self.n_regions), dtype=numpy.float32)
        return ((self.E - 5.0) / 3.0) * i

    def _calculate_equilibria_z(self):
        return zeq_2d_calc(self.x1EQ, self.y0, self.Iext1)
        #non centered x1:
        # return self.y0 + self.Iext1 - self.x1EQ ** 3 - 2.0 * self.x1EQ ** 2

    def _calculate_coupling_at_equilibrium(self):
        return coupling_calc(self.x1EQ, self.K, self.weights)
        #i = numpy.ones((1, self.n_regions), dtype=numpy.float32)
        #return self.K * (numpy.expand_dims(numpy.sum(self.weights * ( numpy.dot(i.T, self.x1EQ) - numpy.dot(self.x1EQ.T, i)), axis=1), 1).T)

    def _calculate_x0(self):
        return x0_calc(self.x1EQ, self.zEQ, self.x0cr, self.rx0, self.Ceq, zmode="lin")
        #return (self.x1EQ + self.x0cr - (self.zEQ + self.Ceq) / 4.0) / self.rx0

    def _dfz_square_taylor(self):
        # The z derivative of the function
        # x1 = F(z) = -4/3 -1/2*sqrt(2(z-y0-Iext1)+64/27)
        dfz = -(0.5 / numpy.sqrt(2 * (self.zEQ - self.y0 - self.Iext1) + 64.0 / 27.0))
        if numpy.any([numpy.any(numpy.isnan(dfz)), numpy.any(numpy.isinf(dfz))]):
            raise ValueError("nan or inf values in dfz")
        else:
            return dfz

    def _fz_jac(self, dfz):
        i = numpy.ones((1, self.n_regions), dtype=numpy.float32)
        # Jacobian: diagonal elements at first row
        #Diagonal elements: -1 + dfz_i * (4 + K_i * sum_j_not_i{wij})
        fz_jac = numpy.diag(-1 + dfz * (4.0 + self.K * numpy.expand_dims(numpy.sum(self.weights, axis=1), 1).T).T[:, 0]) \
        - numpy.dot(self.K.T, i) * numpy.dot(i.T, dfz) * (1 - numpy.eye(self.n_regions))
        if numpy.any([numpy.any(numpy.isnan(fz_jac.flatten())), numpy.any(numpy.isinf(fz_jac.flatten()))]):
            raise ValueError("nan or inf values in dfz")
        else:
            return fz_jac
        return fz_jac


    def _calculate_e(self):
        return 3.0 * self.x1EQ + 5.0

    def _update_parameters(self, seizure_indices):
        """
        Updating hypothesis always starts from a new equilibrium point
        :param seizure_indices: numpy array with conn region indices where we think the seizure starts
        """
        (self.x0cr, self.rx0) = self._calculate_critical_x0_scaling()
        self.Ceq = self._calculate_coupling_at_equilibrium()
        self.x0 = self._calculate_x0()
        self.E = self._calculate_e()

        self.seizure_indices = seizure_indices
        if self.n_seizure_nodes > 0:
            self._run_lsa(seizure_indices)


    def _run_lsa(self, seizure_indices):

        #TODO: automatically choose the number of eigenvalue to sum via a cutting criterion

        self._check_hypothesis(seizure_indices)

        # The z derivative of the function...
        # x1 = F(z) = -4/3 -1/2*sqrt(2(z-y0-Iext1)+64/27)
        dfz = self._dfz_square_taylor()

        #...and the respective Jacobian
        fz_jac = self._fz_jac(dfz)

        # Perform eigenvalue decomposition
        (eigvals, eigvects) = numpy.linalg.eig(fz_jac)
        
        # Sort eigenvalues in descending order... 
        ind = numpy.argsort(eigvals, kind='mergesort')[::-1]
        self.lsa_eigvals = eigvals[ind]
        #...and eigenvectors accordingly
        self.lsa_eigvects = eigvects[:, ind]
        
        #Calculate the propagation strength index by summing the first n_seizure_nodes eigenvectors
        self.lsa_ps = numpy.expand_dims(numpy.sum(numpy.abs(self.lsa_eigvects[:, :self.n_seizure_nodes]), axis=1), 1).T
        
        #Calculate the propagation strength index by summing all eigenvectors
        self.lsa_ps_tot = numpy.expand_dims(numpy.sum(numpy.abs(self.lsa_eigvects), axis=1), 1).T


    def _check_hypothesis(self, seizure_indices):
        """
         LSA doesn't work well if there are some E>1 (i.e., x1EQ>1/3),
        and at the same time the rest of the equilibria are not negative "enough"
         Suggested correction for the moment to ceil x1EQ to the critical x1EQcr = 1/3,
        and then update the whole hypothesis accordingly. We should ask the user for this..
        """
        #TODO: deal with super-critical equilibria better than the way done below...

        temp = self.x1EQ > self.x1EQcr - 10 ** (-3) #numpy.nextafter(0., 1.)
        if temp.any():
            self.x1EQ[temp] = self.x1EQcr - 10 ** (-3) #numpy.nextafter(0., 1.)
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
        self.x1EQ = self._set_equilibria_x1()
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
        ii = numpy.array(range(self.n_regions), dtype=numpy.int32)
        # All regions with an Epileptogenicity hypothesis:
        iE = numpy.delete(ii, ix0)  # their indices

        #Convert x0 to an array of (1,len(ix0)) shape
        x0 = numpy.expand_dims(numpy.array(x0), 1).T
        
        if self.x1eq_mode=="linTaylor":
            self.x1EQ = x1eq_x0_hypo_linTaylor(ix0, iE, self.x1EQ, self.zEQ, x0, self.x0cr, self.rx0, self.y0,
                                               self.Iext1, self.K, self.weights)
        else:
            self.x1EQ = x1eq_x0_hypo_optimize(ix0, iE, self.x1EQ, self.zEQ, x0, self.x0cr, self.rx0, self.y0,
                                              self.Iext1, self.K, self.weights)

        self.zEQ = self._calculate_equilibria_z()

        # Now that equilibria are OK, update the hypothesis to get the actual x0, E etc
        self._update_parameters(seizure_indices)
