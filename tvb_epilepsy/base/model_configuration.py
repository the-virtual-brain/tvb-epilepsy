# coding=utf-8
"""
Model configuration POJO class.
"""


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
