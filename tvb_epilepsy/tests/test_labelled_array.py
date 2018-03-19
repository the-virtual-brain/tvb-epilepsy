import numpy as np

from tvb_epilepsy.base.datatypes import LabelledArray

if __name__ == "__main__":

    la = LabelledArray(np.array([[0.0, 1.0], [2.0, 3.0]]),
                       labels=[np.array(["00", "01"]), np.array(["10", "11"])])

    print(la)
    print(la._labels)
    print("print(la['00':None, :])=")
    print(la["00":None, :])
    print("print(la['00':None, ['10', '11']])=")
    print(la['00':None, ['10', '11']])
    print("print(la[..., np.newaxis])=")
    print(la[..., np.newaxis])
    print("la[-1]=")
    print(la[-1])
