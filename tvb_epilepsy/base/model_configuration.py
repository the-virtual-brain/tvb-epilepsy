# coding=utf-8
"""
Class to keep the model configuration values.
This will be used to populate a Model instance needed in order to launch a simulation.
"""
from collections import OrderedDict

from tvb_epilepsy.base.h5_model import prepare_for_h5
from tvb_epilepsy.base.utils import formal_repr


class ModelConfiguration(object):
    def __init__(self, x0_values, yc, Iext1, K, x0cr, rx0, x1EQ, zEQ, Ceq, E_values):
        # These parameters are used for every Epileptor Model
        self.x0_values = x0_values
        self.yc = yc
        self.Iext1 = Iext1
        self.K = K

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
            "01. x0 values": self.x0_values,
            "02. yc": self.yc,
            "03. Iext1": self.Iext1,
            "04. K": self.K,
            "05. x0cr": self.x0cr,
            "06. rx0": self.rx0,
            "07. x1EQ": self.x1EQ,
            "08. zEQ": self.zEQ,
            "09. Ceq": self.Ceq,
            "10. E values": self.E_values
        }
        return formal_repr(self, OrderedDict(sorted(d.items()), key=lambda t: t[0]))

    def __str__(self):
        return self.__repr__()

    def prepare_for_h5(self):
        h5_model = prepare_for_h5(self)
        h5_model.add_or_update_metadata_attribute("EPI_Type", "HypothesisModel")
        h5_model.add_or_update_metadata_attribute("Number_of_nodes", len(self.x0_values))

        return h5_model
