from tvb_epilepsy.base.utils import curve_elbow_point
from pylab import ion

#The main function...
if __name__ == "__main__":

    v = [30.0, 20.0, 15.0, 10.0, 5.0, 4.0, 3.0, 2.0, 1.0]

    ion()

    elbow = curve_elbow_point(v, interactive=True)

    print elbow, v[:elbow+1]
