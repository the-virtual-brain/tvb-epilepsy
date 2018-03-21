import numpy as np

from tvb_epilepsy.base.datatypes.labelled_array import LabelledArray


if __name__ == "__main__":

    la = LabelledArray(np.array([[0.0, 1.0], [2.0, 3.0]]),
                       labels=[np.array(["a", "b"]), np.array(["0", "1"])])

    print(la)
    print("print(la['a':None, :])=")
    print(la["a":None, :])
    print("print(la['a':None, ['0', '1']])=")
    print(la['a':None, ['0', '1']])
    print("print(la[..., np.newaxis])=")
    print(la[..., np.newaxis])
    print("2*la[-1]=")
    print(2*la[-1])
    print("2*la[::'b']=")
    print(2*la[::'b'])

    print("print(la.slice((slice('b', None), slice()))=")
    print(la.slice((slice('b', None), slice(None))))

