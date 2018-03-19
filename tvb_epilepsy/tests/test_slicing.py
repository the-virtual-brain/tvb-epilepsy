
from tvb_epilepsy.base.utils.slicing_utils import *


class test:

    a = np.array([[1,2],[3,4.0]])

    labels = [["a00", "a01"], ["a10", "a11"]]

    def __getitem__(self, index):
        return self.a[verify_index(index, self.labels)]


if __name__ == "__main__":

    t = test()

    print(t.a)
    print(t.labels)
    print("print(t['a00':None, ...])=")
    print(t["a00":None, ...])
    print("print(t['a00':None, ['a10', 'a11'])=")
    print(t["a00":None, ...])
    print("print(t[..., np.newaxis])=")
    print(t[..., np.newaxis])