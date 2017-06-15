# coding=utf-8
"""
Class to keep the model configuration values.
This will be used to populate a Model instance needed in order to launch a simulation.
"""
from collections import OrderedDict

from tvb_epilepsy.base.h5_model import prepare_for_h5
from tvb_epilepsy.base.utils import formal_repr
from tvb_epilepsy.base.constants import X0_DEF, K_DEF, YC_DEF, I_EXT1_DEF, A_DEF, B_DEF


class ModelConfiguration(object):

    def __init__(self, yc=YC_DEF, Iext1=I_EXT1_DEF, K=K_DEF, a=A_DEF, b=B_DEF,
                       x0cr=None, rx0=None, x1EQ=None, zEQ=None, Ceq=None, x0_values=X0_DEF, E_values=None):

        # These parameters are used for every Epileptor Model...
        self.x0_values = x0_values
        self.yc = yc
        self.Iext1 = Iext1
        self.K = K
        # ...but these 2 have different values for models with more than 2 dimensions
        self.a = a
        self.b = b

        # These parameters are used only for EpileptorDP2D Model
        self.x0cr = x0cr
        self.rx0 = rx0

        # These parameters are not used for Epileptor Model, but are important to keep (h5 or plotting)
        self.x1EQ = x1EQ
        self.zEQ = zEQ
        self.Ceq = Ceq
        self.E_values = E_values

    def __repr__(self):
        d = {
            "01. Excitability": self.x0_values,
            "02. yc": self.yc,
            "03. Iext1": self.Iext1,
            "04. K": self.K,
            "05. a": self.a,
            "06. b": self.b,
            "07. x0cr": self.x0cr,
            "08. rx0": self.rx0,
            "09. x1EQ": self.x1EQ,
            "10. zEQ": self.zEQ,
            "11. Ceq": self.Ceq,
            "12. Epileptogenicity": self.E_values
        }
        return formal_repr(self, OrderedDict(sorted(d.items()), key=lambda t: t[0]))

    def __str__(self):
        return self.__repr__()

    def prepare_for_h5(self):
        h5_model = prepare_for_h5(self)
        h5_model.add_or_update_metadata_attribute("EPI_Type", "HypothesisModel")
        h5_model.add_or_update_metadata_attribute("Number_of_nodes", len(self.x0_values))

        return h5_model


