# coding=utf-8
"""
Model configuration POJO class.
"""
from collections import OrderedDict

from tvb_epilepsy.base.h5_model import prepare_for_h5
from tvb_epilepsy.base.utils import formal_repr


class ModelConfiguration(object):
    def __init__(self, model_x0_values, model_yc, model_Iext1, model_K, model_xcr, model_rx0, x1EQ, zEQ, Ceq, E_values):
        self.model_x0_values = model_x0_values
        self.model_yc = model_yc
        self.model_Iext1 = model_Iext1
        self.model_K = model_K
        self.model_xcr = model_xcr
        self.model_rx0 = model_rx0

        # TODO: to keep or not to keep?
        self.x1EQ = x1EQ
        self.zEQ = zEQ
        self.Ceq = Ceq
        self.E_values = E_values

    def __repr__(self):
        d = {
            "01. x0 values": self.model_x0_values,
            "02. yc": self.model_yc,
            "03. Iext1": self.model_Iext1,
            "04. K": self.model_K,
            "05. xcr": self.model_xcr,
            "06. rx0": self.model_rx0
        }
        return formal_repr(self, OrderedDict(sorted(d.items()), key=lambda t: t[0]))

    def __str__(self):
        return self.__repr__()

    def prepare_for_h5(self):
        h5_model = prepare_for_h5(self)
        h5_model.add_or_update_metadata_attribute("EPI_Type", "HypothesisModel")
        h5_model.add_or_update_metadata_attribute("Number_of_nodes", len(self.model_x0_values))

        return h5_model

    def get_equilibrum_points(self):
        return self.x1EQ, self.zEQ, self.Ceq

    def get_x0_values(self):
        return self.model_x0_values

    def get_yc(self):
        return self.model_yc

    def get_Iext1(self):
        return self.model_Iext1

    def get_K(self):
        return self.model_K

    def get_xcr(self):
        return self.model_xcr

    def get_rx0(self):
        return self.model_rx0

    def get_E_values(self):
        return self.E_values
