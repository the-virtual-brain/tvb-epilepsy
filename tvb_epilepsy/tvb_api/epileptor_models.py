# coding=utf-8
"""
@version $Id: epileptor_models.py 1651 2016-09-01 17:47:05Z denis $

Extend TVB Models, with new ones, specific for Epilepsy.
"""

import numpy
from tvb.simulator.common import get_logger
import tvb.datatypes.arrays as arrays
import tvb.basic.traits.types_basic as basic
from tvb.simulator.models import Model


LOG = get_logger(__name__)

        
class EpileptorDP(Model):
    r"""
    The Epileptor is a composite neural mass model of six dimensions which
    has been crafted to model the phenomenology of epileptic seizures.
    (see [Jirsaetal_2014]_). 
    ->x0 parameters are shifted for the bifurcation
      to be at x0=1, where x0>1 is the supercritical region.
    ->there is a choice for linear or sigmoidal z dynamics (see [Proixetal_2014]_)
    ->some parameters change their names to be more similar to the equations.

    Equations and default parameters are taken from [Jirsaetal_2014]_.

          +------------------------------------------------------+
          |                         Table 1                      |
          +----------------------+-------------------------------+
          |        Parameter     |           Value               |
          +======================+===============================+
          |         I_ext1      |              3.1              |
          +----------------------+-------------------------------+
          |         I_ext2      |              0.45             |
          +----------------------+-------------------------------+
          |         tau0         |           2857.0              |
          +----------------------+-------------------------------+
          |         x_0          |              0.0              |
          +----------------------+-------------------------------+
          |         slope        |              0.0              |
          +----------------------+-------------------------------+
          |             Integration parameter                    |
          +----------------------+-------------------------------+
          |           dt         |              0.1              |
          +----------------------+-------------------------------+
          |  simulation_length   |              4000             |
          +----------------------+-------------------------------+
          |                    Noise                             |
          +----------------------+-------------------------------+
          |         nsig         | [0., 0., 0., 1e-3, 1e-3, 0.]  |
          +----------------------+-------------------------------+
          |       Jirsa et al. 2014, Proix et al. 2014           |
          +------------------------------------------------------+


    .. figure :: img/Epileptor_01_mode_0_pplane.svg
        :alt: Epileptor phase plane

    .. [Jirsaetal_2014] Jirsa, V. K.; Stacey, W. C.; Quilichini, P. P.;
        Ivanov, A. I.; Bernard, C. *On the nature of seizure dynamics.* Brain,
        2014.

    .. [Proixetal_2014] Proix, T., Bartolomei, F., Chauvel, P., Bernard, C.,
                       & Jirsa, V. K. (2014).
                       Permitivity Coupling across Brain Regions Determines
                       Seizure Recruitment in Partial Epilepsy.
                       Journal of Neuroscience, 34(45), 15009–15021.
                       htau1p://doi.org/10.1523/JNEUROSCI.1570-14.2014

    .. automethod:: EpileptorDP.__init__

    Variables of interest to be used by monitors: -y[0] + y[3]

        .. math::
            \dot{x_{1}} &=& y_{1} - f_{1}(x_{1}, x_{2}) - z + I_{ext1} \\
            \dot{y_{1}} &=& yc - d x_{1}^{2} - y{1} \\
            \dot{z} &=&
            \begin{cases}
            (f_z(x_{1}) - z-0.1 z^{7})/tau0 & \text{if } x<0 \\
            (f_z(x_{1}) - z)/tau0           & \text{if } x \geq 0
            \end{cases} \\
            \dot{x_{2}} &=& -y_{2} + x_{2} - x_{2}^{3} + I_{ext2} + 0.002 g - 0.3 (z-3.5) \\
            \dot{y_{2}} &=& 1 / \tau2 (-y_{2} + f_{2}(x_{2}))\\
            \dot{g} &=& -0.01 (g - 0.1 x_{1} )

    where:
        .. math::
            f_{1}(x_{1}, x_{2}) =
            \begin{cases}
            a x_{1}^{3} - b x_{1}^2 & \text{if } x_{1} <0\\
            ( x_{2} - 0.6(z-4)^2 -slope ) x_{1}  &\text{if }x_{1} \geq 0
            \end{cases}

        .. math::
            f_z(x_{1})  =
            \begin{cases}
            4 * (x_{1} - r_{x0}*x0 + x0_{cr}) & \text{linear} \\
            \frac{3}{1+e^{-10*(x_{1}+0.5)}} - r_{x0}*x0 + x0_{cr} & \text{sigmoidal} \\
            \end{cases}
    and:

        .. math::
            f_{2}(x_{2}) =
            \begin{cases}
            0 & \text{if } x_{2} <-0.25\\
            s*(x_{2} + 0.25) & \text{if } x_{2} \geq -0.25
            \end{cases}
    """

    _ui_name = "EpileptorDP"
    ui_configurable_parameters = ["Iext1", "Iext2", "tau0", "x0", "slope"]

    zmode = arrays.FloatArray(
        label="zmode",
        default=numpy.array("lin"),
        doc="zmode = numpy.array(""lin"") for linear and numpy.array(""sig"") for sigmoidal z dynamics",
        order=-1)

#    a = arrays.FloatArray(
#        label="a",
#        default=numpy.array([1]),
#        doc="Coefficient of the cubic term in the first state variable",
#        order=-1)

#    b = arrays.FloatArray(
#        label="b",
#        default=numpy.array([3]),
#        doc="Coefficient of the squared term in the first state variabel",
#        order=-1)

    yc = arrays.FloatArray(
        label="yc",
        default=numpy.array([1]),
        doc="Additive coefficient for the second state variable",
        order=-1)

#    d = arrays.FloatArray(
#        label="d",
#        default=numpy.array([5]),
#        doc="Coefficient of the squared term in the second state variable",
#        order=-1)

    tau0 = arrays.FloatArray(
        label="r",
        range=basic.Range(lo=100.0, hi=5000, step=10),
        default=numpy.array([2857.0]),
        doc="Temporal scaling in the third state variable",
        order=4)

#    s = arrays.FloatArray(
#        label="s",
#        default=numpy.array([4]),
#        doc="Linear coefficient in the third state variable",
#        order=-1)

    x0 = arrays.FloatArray(
        label="x0",
        range=basic.Range(lo=-0.5, hi=1.5, step=0.1),
        default=numpy.array([0.0]),
        doc="Excitability parameter",
        order=3)

    x0cr = arrays.FloatArray(
        label="x0cr",
        range=basic.Range(lo=-1.0, hi=1.0, step=0.1),
        default=numpy.array([5.93240740740741]),
        doc="Critical excitability parameter",
        order=-1)

    r = arrays.FloatArray(
        label="r",
        range=basic.Range(lo=0.0, hi=1.0, step=0.1),
        default=numpy.array([1.64814814814815]),
        doc="Excitability parameter scaling",
        order=-1)

    Iext1 = arrays.FloatArray(
        label="Iext1",
        range=basic.Range(lo=1.5, hi=5.0, step=0.1),
        default=numpy.array([3.1]),
        doc="External input current to the first population",
        order=1)

    slope = arrays.FloatArray(
        label="slope",
        range=basic.Range(lo=-16.0, hi=6.0, step=0.1),
        default=numpy.array([0.]),
        doc="Linear coefficient in the first state variable",
        order=5)

    Iext2 = arrays.FloatArray(
        label="Iext2",
        range=basic.Range(lo=0.0, hi=1.0, step=0.05),
        default=numpy.array([0.45]),
        doc="External input current to the second population",
        order=2)

    tau2 = arrays.FloatArray(
        label="tau2",
        default=numpy.array([10]),
        doc="Temporal scaling coefficient in fifth state variable",
        order=-1)

    Kvf = arrays.FloatArray(
        label="K_vf",
        default=numpy.array([0.0]),
        range=basic.Range(lo=0.0, hi=4.0, step=0.5),
        doc="Coupling scaling on a very fast time scale.",
        order=6)

    Kf = arrays.FloatArray(
        label="K_f",
        default=numpy.array([0.0]),
        range=basic.Range(lo=0.0, hi=4.0, step=0.5),
        doc="Correspond to the coupling scaling on a fast time scale.",
        order=7)

    K = arrays.FloatArray(
        label="K",
        default=numpy.array([0.0]),
        range=basic.Range(lo=-4.0, hi=4.0, step=0.1),
        doc="Permitau1ivity coupling, that is from the fast time scale toward the slow time scale",
        order=8)

    tau1 = arrays.FloatArray(
        label="tau1",
        default=numpy.array([1.0]),
        range=basic.Range(lo=0.001, hi=10.0, step=0.001),
        doc="Time scaling of the whole system",
        order=9)

    state_variable_range = basic.Dict(
        label="State variable ranges [lo, hi]",
        default={"y0": numpy.array([-2., 2.]),
                 "y1": numpy.array([-20., 2.]),
                 "y2": numpy.array([2.0, 20.0]),
                 "y3": numpy.array([-2., 0.]),
                 "y4": numpy.array([0., 2.]),
                 "y5": numpy.array([-1., 1.])},
        doc="n/a",
        order=16
        )

    variables_of_interest = basic.Enumerate(
        label="Variables watched by Monitors",
        options=["y0", "y1", "y2", "y3", "y4", "y5", "y3 - y0"],
        default=["y3 - y0", "y2"],
        select_multiple=True,
        doc="""default state variables to be monitored""",
        order=-1)

    state_variables = ["y0", "y1", "y2", "y3", "y4", "y5"]

    _nvar = 6
    cvar = numpy.array([0, 3], dtype=numpy.int32)


    def dfun(self, state_variables, coupling, local_coupling=0.0,
             array=numpy.array, where=numpy.where, concat=numpy.concatenate):
        r"""
        Computes the derivatives of the state variables of the Epileptor
        with respect to time.

        Implementation note: we expect this version of the Epileptor to be used
        in a vectorized manner. Concretely, y has a shape of (6, n) where n is
        the number of nodes in the network. An consequence is that
        the original use of if/else is translated by calculated both the true
        and false forms and mixing them using a boolean mask.

        Variables of interest to be used by monitors: -y[0] + y[3]

            .. math::
            \dot{y_{0}} &=& y_{1} - f_{1}(y_{0}, y_{3}) - y_{2} + I_{ext1} \\
            \dot{y_{1}} &=& yc - d y_{0}^{2} - y{1} \\
            \dot{y_{2}} &=&
            \begin{cases}
            (f_z(y_{0}) - y_{2}-0.1 y_{2}^{7})/tau0 & \text{if } y_{0}<0 \\
            (f_z(y_{0}) - y_{2})/tau0           & \text{if } y_{0} \geq 0
            \end{cases} \\
            \dot{y_{3}} &=& -y_{4} + y_{3} - y_{3}^{3} + I_{ext2} + 0.002 y_{5} - 0.3 (y_{2}-3.5) \\
            \dot{y_{4}} &=& 1 / \tau2 (-y_{4} + f_{2}(y_{3}))\\
            \dot{y_{5}} &=& -0.01 (y_{5} - 0.1 y_{0} )

        where:
            .. math::
                f_{1}(y_{0}, y_{3}) =
                \begin{cases}
                a y_{0}^{3} - by_{0}^2 & \text{if } y_{0} <0\\
                (y_{3} - 0.6(y_{2}-4)^2 slope)y_{0} &\text{if }y_{0} \geq 0
                \end{cases}

            .. math::
                f_z(y_{0})  =
                \begin{cases}
                4 * (y_{0} - r*x0 + x0_{cr}) & \text{linear} \\
                \frac{3}{1+e^{-10*(y_{0}+0.5)}} - r*x0 + x0_{cr} & \text{sigmoidal} \\
                \end{cases}
        and:

            .. math::
                f_{2}(y_{3}) =
                \begin{cases}
                0 & \text{if } y_{3} <-0.25\\
                s*(y_{3} + 0.25) & \text{if } y_{3} \geq -0.25
                \end{cases}

        """

        y = state_variables
        ydot = numpy.empty_like(state_variables)

        Iext1 = self.Iext1 + local_coupling * y[0]
        c_pop1 = coupling[0, :]
        c_pop2 = coupling[1, :]

        #TVB Epileptor in commented lines below

        # population 1
        #if_ydot0 = - self.a * y[0] ** 2 + self.b * y[0]
        if_ydot0 = y[0]**2 - 3.0*y[0] #self.a=1.0, self.b=3.0
        # else_ydot0 = self.slope - y[3] + 0.6 * (y[2] - 4.0) ** 2
        else_ydot0 = y[3] - 0.6*(y[2]-4.0)**2 - self.slope
        # ydot[0] = self.tt * (y[1] - y[2] + Iext + self.Kvf * c_pop1 + where(y[0] < 0., if_ydot0, else_ydot0) * y[0])
        ydot[0] = self.tau1*(y[1] - y[2] + Iext1 + self.Kvf*c_pop1 - where(y[0] < 0.0, if_ydot0, else_ydot0) * y[0])
        # ydot[1] = self.tt * (self.c - self.d * y[0] ** 2 - y[1])
        ydot[1] = self.tau1*(self.yc - 5.0*y[0]**2 - y[1]) #self.d=5

        # energy
        #if_ydot2 = - 0.1 * y[2] ** 7
        if_ydot2 = - 0.1 * y[2] ** 7
        #else_ydot2 = 0
        else_ydot2 = 0
        # ydot[2] = self.tt * (
        if self.zmode=='lin':
            # self.r * (4 * (y[0] - self.x0) - y[2]      + where(y[2] < 0., if_ydot2, else_ydot2) + self.Ks * c_pop1))
            fz = 4*(y[0] - self.r * self.x0 + self.x0cr) + where(y[2] < 0., if_ydot2, else_ydot2)
        elif self.zmode=='sig':
            fz = 3 / (1 + numpy.exp(-10*(y[0] + 0.5))) - self.r * self.x0 + self.x0cr
        else:
            print "ERROR: zmode has to be either ""lin"" or ""sig"" for linear and sigmoidal fz(), respectively"
        ydot[2] = self.tau1*((fz - y[2] + self.K*c_pop1)/self.tau0)

        # population 2
        # ydot[3] = self.tt * (-y[4] + y[3] - y[3] ** 3 + self.Iext2 + 2 * y[5] - 0.3 * (y[2] - 3.5) + self.Kf * c_pop2)
        ydot[3] = self.tau1 * (-y[4] + y[3] - y[3] ** 3 + self.Iext2 + 2 * y[5] - 0.3 * (y[2] - 3.5) + self.Kf * c_pop2)
        # if_ydot4 = 0
        if_ydot4 = 0
        # else_ydot4 = self.aa * (y[3] + 0.25)
        else_ydot4 = 6.0 * (y[3] + 0.25) #self.s = 6.0
        # ydot[4] = self.tt * ((-y[4] + where(y[3] < -0.25, if_ydot4, else_ydot4)) / self.tau)
        ydot[4] = self.tau1*((-y[4] + where(y[3] < -0.25, if_ydot4, else_ydot4)) / self.tau2)

        # filter
        #ydot[5] = self.tt * (-0.01 * (y[5] - 0.1 * y[0]))
        ydot[5] = self.tau1*(-0.01 * (y[5] - 0.1 * y[0]))

        return ydot

    def jacobian(self, state_variables, coupling, local_coupling=0.0,
                 array=numpy.array, where=numpy.where, concat=numpy.concatenate):

        return None



class EpileptorDPrealistic(Model):
    r"""
    The Epileptor is a composite neural mass model of six dimensions which
    has been crafted to model the phenomenology of epileptic seizures.
    (see [Jirsaetal_2014]_).
    ->x0 parameters are shifted for the bifurcation
      to be at x0=1, where x0>1 is the supercritical region.
    ->there is a choice for linear or sigmoidal z dynamics (see [Proixetal_2014]_).

    Equations and default parameters are taken from [Jirsaetal_2014]_.

    The realistic Epileptor allows for state variables I_{ext1}, I_{ext2}, x0, slope and K
    to fluctuate as linear dynamical equations, driven by the corresponding
    parameter values. It could be combined with multiplicative and/or pink noise.

          +------------------------------------------------------+
          |                         Table 1                      |
          +----------------------+-------------------------------+
          |        Parameter     |           Value               |
          +======================+===============================+
          |         I_ext1      |              3.1              |
          +----------------------+-------------------------------+
          |         I_ext2      |              0.45             |
          +----------------------+-------------------------------+
          |         tau0         |           2857.0              |
          +----------------------+-------------------------------+
          |         x_0          |              0.0              |
          +----------------------+-------------------------------+
          |         slope        |              0.0              |
          +----------------------+-------------------------------+
          |             Integration parameter                    |
          +----------------------+-------------------------------+
          |           dt         |              0.1              |
          +----------------------+-------------------------------+
          |  simulation_length   |              4000             |
          +----------------------+-------------------------------+
          |                    Noise                             |
          +----------------------+-------------------------------+
          |         nsig         | [0., 0., 0., 1e-3, 1e-3, 0.]  |
          +----------------------+-------------------------------+
          |       Jirsa et al. 2014, Proix et al. 2014           |
          +------------------------------------------------------+


    .. figure :: img/Epileptor_01_mode_0_pplane.svg
        :alt: Epileptor phase plane

    .. [Jirsaetal_2014] Jirsa, V. K.; Stacey, W. C.; Quilichini, P. P.;
        Ivanov, A. I.; Bernard, C. *On the nature of seizure dynamics.* Brain,
        2014.

    .. [Proixetal_2014] Proix, T., Bartolomei, F., Chauvel, P., Bernard, C.,
                       & Jirsa, V. K. (2014).
                       Permitau1ivity Coupling across Brain Regions Determines
                       Seizure Recruitment in Partial Epilepsy.
                       Journal of Neuroscience, 34(45), 15009–15021.
                       htau1p://doi.org/10.1523/JNEUROSCI.1570-14.2014

    .. automethod:: EpileptorDP.__init__

    Variables of interest to be used by monitors: -y[0] + y[3]

        .. math::
            \dot{x_{1}} &=& y_{1} - f_{1}(x_{1}, x_{2}) - z + I_{ext1} \\
            \dot{y_{1}} &=& yc - d x_{1}^{2} - y{1} \\
            \dot{z} &=&
            \begin{cases}
            (f_z(x_{1}) - z-0.1 z^{7})/tau0 & \text{if } x<0 \\
            (f_z(x_{1}) - z)/tau0           & \text{if } x \geq 0
            \end{cases} \\
            \dot{x_{2}} &=& -y_{2} + x_{2} - x_{2}^{3} + I_{ext2} + 0.002 g - 0.3 (z-3.5) \\
            \dot{y_{2}} &=& 1 / \tau2 (-y_{2} + f_{2}(x_{2}))\\
            \dot{g} &=& -0.01 (g - 0.1 x_{1} )

    where:
        .. math::
            f_{1}(x_{1}, x_{2}) =
            \begin{cases}
            a x_{1}^{3} - b x_{1}^2 & \text{if } x_{1} <0\\
            (x_{2} - 0.6(z-4)^2 -slope) x_{1} &\text{if }x_{1} \geq 0
            \end{cases}

        .. math::
            f_z(x_{1})  =
            \begin{cases}
            4 * (x_{1} - r_{x0}*x0 + x0_{cr}) & \text{linear} \\
            \frac{3}{1+e^{-10*(x_{1}+0.5)}} - r_{x0}*x0 + x0_{cr} & \text{sigmoidal} \\
            \end{cases}
    and:

        .. math::
            f_{2}(x_{2}) =
            \begin{cases}
            0 & \text{if } x_{2} <-0.25\\
            s*(x_{2} + 0.25) & \text{if } x_{2} \geq -0.25
            \end{cases}
    """

    _ui_name = "EpileptorDPrealistic"
    ui_configurable_parameters = ["Iext1", "Iext2", "tau0", "x0", "slope"]

    zmode = arrays.FloatArray(
        label="zmode",
        default=numpy.array("lin"),
        doc="zmode = np.array(""lin"") for linear and numpy.array(""sig"") for sigmoidal z dynamics",
        order=-1)
        
    pmode = arrays.FloatArray(
        label="pmode",
        default=numpy.array("const"),
        doc="pmode = numpy.array(""g""), numpy.array(""z""), numpy.array(""z*g"") or numpy.array(""const"") parameters following the g, z, z*g dynamics or staying constamt, respectively",
        order=-1)   

#    a = arrays.FloatArray(
#        label="a",
#        default=numpy.array([1]),
#        doc="Coefficient of the cubic term in the first state variable",
#        order=-1)

#    b = arrays.FloatArray(
#        label="b",
#        default=numpy.array([3]),
#        doc="Coefficient of the squared term in the first state variabel",
#        order=-1)

    yc = arrays.FloatArray(
        label="yc",
        default=numpy.array([1]),
        doc="Additive coefficient for the second state variable",
        order=-1)

#    d = arrays.FloatArray(
#        label="d",
#        default=numpy.array([5]),
#        doc="Coefficient of the squared term in the second state variable",
#        order=-1)

    tau0 = arrays.FloatArray(
        label="r",
        range=basic.Range(lo=100.0, hi=5000, step=10),
        default=numpy.array([4000.0]),
        doc="Temporal scaling in the third state variable",
        order=4)

#    s = arrays.FloatArray(
#        label="s",
#        default=numpy.array([4]),
#        doc="Linear coefficient in the third state variable",
#        order=-1)

    x0 = arrays.FloatArray(
        label="x0",
        range=basic.Range(lo=-0.5, hi=1.5, step=0.1),
        default=numpy.array([0.0]),
        doc="Excitability parameter",
        order=3)

    x0cr = arrays.FloatArray(
        label="x0cr",
        range=basic.Range(lo=-1.0, hi=1.0, step=0.1),
        default=numpy.array([5.93240740740741]),
        doc="Critical excitability parameter",
        order=-1)

    r = arrays.FloatArray(
        label="r",
        range=basic.Range(lo=0.0, hi=1.0, step=0.1),
        default=numpy.array([1.64814814814815]),
        doc="Excitability parameter scaling",
        order=-1)

    Iext1 = arrays.FloatArray(
        label="Iext1",
        range=basic.Range(lo=1.5, hi=5.0, step=0.1),
        default=numpy.array([3.1]),
        doc="External input current to the first population",
        order=1)

    slope = arrays.FloatArray(
        label="slope",
        range=basic.Range(lo=-16.0, hi=6.0, step=0.1),
        default=numpy.array([0.]),
        doc="Linear coefficient in the first state variable",
        order=5)

    Iext2 = arrays.FloatArray(
        label="Iext2",
        range=basic.Range(lo=0.0, hi=1.0, step=0.05),
        default=numpy.array([0.45]),
        doc="External input current to the second population",
        order=2)

    tau2 = arrays.FloatArray(
        label="tau2",
        default=numpy.array([10]),
        doc="Temporal scaling coefficient in fifth state variable",
        order=-1)

    Kvf = arrays.FloatArray(
        label="K_vf",
        default=numpy.array([0.0]),
        range=basic.Range(lo=0.0, hi=4.0, step=0.5),
        doc="Coupling scaling on a very fast time scale.",
        order=6)

    Kf = arrays.FloatArray(
        label="K_f",
        default=numpy.array([0.0]),
        range=basic.Range(lo=0.0, hi=4.0, step=0.5),
        doc="Correspond to the coupling scaling on a fast time scale.",
        order=7)

    K = arrays.FloatArray(
        label="K",
        default=numpy.array([0.0]),
        range=basic.Range(lo=-4.0, hi=4.0, step=0.1),
        doc="Permitau1ivity coupling, that is from the fast time scale toward the slow time scale",
        order=8)

    tau1 = arrays.FloatArray(
        label="tau1",
        default=numpy.array([1.0]),
        range=basic.Range(lo=0.001, hi=10.0, step=0.001),
        doc="Time scaling of the whole system",
        order=9)

    state_variable_range = basic.Dict(
        label="State variable ranges [lo, hi]",
        default={"y0": numpy.array([-2., 2.]), #x1
                 "y1": numpy.array([-20., 2.]), #y1
                 "y2": numpy.array([2.0, 20.0]), #z
                 "y3": numpy.array([-2., 0.]), #x2
                 "y4": numpy.array([0., 2.]), #y2
                 "y5": numpy.array([-1., 1.]), #g
                 "y6": numpy.array([-2, 2]), #x0
                 "y7": numpy.array([-20., 6.]), #slope
                 "y8": numpy.array([1.5, 5.]), #Iext1
                 "y9": numpy.array([0., 1.]), #Iext2
                 "y10": numpy.array([-50., 50.])},#K
        doc="n/a",
        order=16
        )

    variables_of_interest = basic.Enumerate(
        label="Variables watched by Monitors",
        options=["y0", "y1", "y2", "y3", "y4", "y5", "y6", "y7", "y8", "y9", "y10", "y3 - y0"],
        default=["y3 - y0", "y2"],
        select_multiple=True,
        doc="""default state variables to be monitored""",
        order=-1)

    state_variables = ["y0", "y1", "y2", "y3", "y4", "y5", "y6", "y7", "y8", "y9", "y10"]

    _nvar = 11
    cvar = numpy.array([0, 3], dtype=numpy.int32)

    def linear_scaling(self,x,x1,x2,y1,y2):
        scaling_factor = (y2 - y1) / (x2 - x1)
        return y1 + (x - x1) * scaling_factor
        
    def dfun(self, state_variables, coupling, local_coupling=0.0,
             array=numpy.array, where=numpy.where, concat=numpy.concatenate):
        r"""
        Computes the derivatives of the state variables of the Epileptor
        with respect to time.

        Implementation note: we expect this version of the Epileptor to be used
        in a vectorized manner. Concretely, y has a shape of (6, n) where n is
        the number of nodes in the network. An consequence is that
        the original use of if/else is translated by calculated both the true
        and false forms and mixing them using a boolean mask.

        Variables of interest to be used by monitors: -y[0] + y[3]

            .. math::
            \dot{y_{0}} &=& y_{1} - f_{1}(y_{0}, y_{3}) - y_{2} + I_{ext1} \\
            \dot{y_{1}} &=& yc - d (y_{0} -5/3)^{2} - y{1} \\
            \dot{y_{2}} &=&
            \begin{cases}
            (f_z(y_{0}) - y_{2}-0.1 y_{2}^{7})/tau0 & \text{if } y_{0}<5/3 \\
            (f_z(y_{0}) - y_{2})/tau0           & \text{if } y_{0} \geq 5/3
            \end{cases} \\
            \dot{y_{3}} &=& -y_{4} + y_{3} - y_{3}^{3} + I_{ext2} + 0.002 y_{5} - 0.3 (y_{2}-3.5) \\
            \dot{y_{4}} &=& 1 / \tau2 (-y_{4} + f_{2}(y_{3}))\\
            \dot{y_{5}} &=& -0.01 (y_{5} - 0.1 ( y_{0} -5/3 ) )

        where:
            .. math::
                f_{1}(y_{0}, y_{3}) =
                \begin{cases}
                a ( y_{0} -5/3 )^{3} - b ( y_{0} -5/3 )^2 & \text{if } y_{0} <5/3\\
                ( y_{3} - 0.6(y_{2}-4)^2 -slope ) ( y_{0} - 5/3 ) &\text{if }y_{0} \geq 5/3
                \end{cases}

            .. math::
                f_z(y_{0})  =
                \begin{cases}
                4 * (y_{0} - r*x0 + x0_{cr}) & \text{linear} \\
                \frac{3}{1+e^{-10*(y_{0}-7/6)}} - r*x0 + x0_{cr} & \text{sigmoidal} \\
                \end{cases}
        and:

            .. math::
                f_{2}(y_{3}) =
                \begin{cases}
                0 & \text{if } y_{3} <-0.25\\
                s*(y_{3} + 0.25) & \text{if } y_{3} \geq -0.25
                \end{cases}

        """

        y = state_variables
        ydot = numpy.empty_like(state_variables)


        #To use later:
        x0=y[6]
        slope = y[7]
        Iext1 = y[8]
        Iext2 = y[9]
        K = y[10]

        Iext1 = Iext1 + local_coupling * y[0]
        c_pop1 = coupling[0, :]
        c_pop2 = coupling[1, :]

        # population 1
        if_ydot0 = y[0]**2 - 3.0*y[0] #self.a=1.0, self.b=3.0
        else_ydot0 = y[3] - 0.6*(y[2]-4.0)**2 - slope
        ydot[0] = self.tau1*(y[1] - y[2] + Iext1 + self.Kvf*c_pop1 - where(y[0] < 0.0, if_ydot0, else_ydot0) * y[0])
        ydot[1] = self.tau1*(self.yc - 5.0*y[0]**2 - y[1]) #self.d=5

        # energy
        if_ydot2 = - 0.1*y[2]**7
        else_ydot2 = 0
        if self.zmode=='lin':
            fz = 4*(y[0] - self.r * x0 + self.x0cr) + where(y[2] < 0., if_ydot2, else_ydot2)
        elif self.zmode=='sig':
            fz = 3 / (1 + numpy.exp(-10 * (y[0] + 0.5))) - self.r * x0 + self.x0cr
        else:
            print "ERROR: zmode has to be either ""lin"" or ""sig"" for linear and sigmoidal fz(), respectively"
        ydot[2] = self.tau1*((fz - y[2] + K*c_pop1)/self.tau0)

        # population 2
        ydot[3] = self.tau1*(-y[4] + y[3] - y[3]**3 + Iext2 + 2*y[5] - 0.3*(y[2] - 3.5) + self.Kf*c_pop2)
        if_ydot4 = 0
        else_ydot4 = 6.0*(y[3] + 0.25) #self.s = 6.0
        ydot[4] = self.tau1*((-y[4] + where(y[3] < -0.25, if_ydot4, else_ydot4))/self.tau2)

        # filter
        ydot[5] = self.tau1*(-0.01*(y[5] - 0.1*y[0]))
        
        if (self.pmode == numpy.array(['g','z','z*g'])).any():
            if self.pmode == 'g':
                xp = 1/(1+numpy.exp(-10*(y[5]+0.0)))
                xp1 = 0#-0.175
                xp2 = 1#0.025
            elif self.pmode == 'z':
                xp = 1/(1+numpy.exp(-10*(y[2]-3.00)))
                xp1 = 0
                xp2 = 1
            elif self.pmode == 'z*g':
                xp = y[2]*y[5]
                xp1 = -0.7
                xp2 = 0.1    
            slope_eq = self.linear_scaling(xp,xp1,xp2,1.0,self.slope)
            #slope_eq = self.slope    
            Iext2_eq = self.linear_scaling(xp,xp1,xp2,0.0,self.Iext2)    
        else:
             slope_eq = self.slope
             Iext2_eq = self.Iext2                   
            
        
        # x0
        ydot[6] = self.tau1*(-y[6] + self.x0)
        # slope
        ydot[7] = 10*self.tau1*(-y[7] + slope_eq) #5*
        # Iext1
        ydot[8] = self.tau1*(-y[8] + self.Iext1)/self.tau0
        # Iext2
        ydot[9] = 5*self.tau1*(-y[9] + Iext2_eq)
        # K
        ydot[10] = self.tau1*(-y[10] + self.K)/self.tau0

        return ydot


    def jacobian(self, state_variables, coupling, local_coupling=0.0,
                 array=numpy.array, where=numpy.where, concat=numpy.concatenate):

        return None


class EpileptorDP2D(Model):

    r"""
    The Epileptor 2D is a composite neural mass model of two dimensions which
    has been crafted to model the phenomenology of epileptic seizures in a
    reduced form. This model is used for Linear Stability Analysis
    (see [Proixetal_2014]_).
    ->x0 parameters are shifted for the bifurcation
      to be at x0=1, where x0>1 is the supercritical region.
    ->there is a choice for linear or sigmoidal z dynamics (see [Proixetal_2014]_)
    ->some parameters change their names to be more similar to the equations.

    Equations and default parameters are taken from [Jirsaetal_2014]_.

          +------------------------------------------------------+
          |                         Table 1                      |
          +----------------------+-------------------------------+
          |        Parameter     |           Value               |
          +======================+===============================+
          |         I_ext1      |              3.1              |
          +----------------------+-------------------------------+
          |         tau0         |           2857.0              |
          +----------------------+-------------------------------+
          |         x_0          |              0.0              |
          +----------------------+-------------------------------+
          |         slope        |              0.0              |
          +----------------------+-------------------------------+
          |             Integration parameter                    |
          +----------------------+-------------------------------+
          |           dt         |              0.1              |
          +----------------------+-------------------------------+
          |  simulation_length   |              4000             |
          +----------------------+-------------------------------+
          |                    Noise                             |
          +----------------------+-------------------------------+
          |         nsig         | [0., 0., 0., 1e-3, 1e-3, 0.]  |
          +----------------------+-------------------------------+
          |       Jirsa et al. 2014, Proix et al. 2014           |
          +------------------------------------------------------+


    .. figure :: img/Epileptor_01_mode_0_pplane.svg
        :alt: Epileptor phase plane

    .. [Jirsaetal_2014] Jirsa, V. K.; Stacey, W. C.; Quilichini, P. P.;
        Ivanov, A. I.; Bernard, C. *On the nature of seizure dynamics.* Brain,
        2014.

    .. [Proixetal_2014] Proix, T., Bartolomei, F., Chauvel, P., Bernard, C.,
                       & Jirsa, V. K. (2014).
                       Permitau1ivity Coupling across Brain Regions Determines
                       Seizure Recruitment in Partial Epilepsy.
                       Journal of Neuroscience, 34(45), 15009–15021.
                       htau1p://doi.org/10.1523/JNEUROSCI.1570-14.2014

    .. automethod:: EpileptorDP.__init__

    Variables of interest to be used by monitors: -y[0] + y[3]

        .. math::
            \dot{x_{1}} &=& yc - f_{1}(x_{1},z) - z + I_{ext1} \\
            \dot{z} &=&
            \begin{cases}
            (f_z(x_{1}) - z-0.1 z^{7})/tau0 & \text{if } x<5/3 \\
            (f_z(x_{1}) - z)/tau0           & \text{if } x \geq 5/3
            \end{cases} \\

    where:
        .. math::
            f_{1}(x_{1},z) =
            \begin{cases}
            a ( x_{1} -5/3 )^{3} - b ( x_{1} -5/3 )^2 & \text{if } x_{1} <5/3\\
            ( 5*( x_{1} -5/3 ) - 0.6(z-4)^2 -slope) ( x_{1} - 5/3 ) &\text{if }x_{1} \geq 5/3
            \end{cases}

   and:
        .. math::
            f_z(x_{1})  =
            \begin{cases}
            4 * (x_{1} - r_{x0}*x0 + x0_{cr}) & \text{linear} \\
            \frac{3}{1+e^{-10*(x_{1}-7/6)}} - r_{x0}*x0 + x0_{cr} & \text{sigmoidal} \\
            \end{cases}

    """

    _ui_name = "EpileptorDP2D"
    ui_configurable_parameters = ["Iext1", "tau0", "x0", "slope"]

    zmode = arrays.FloatArray(
        label="zmode",
        default=numpy.array("lin"),
        doc="zmode = numpy.array(""lin"") for linear and numpy.array(""sig"") for sigmoidal z dynamics",
        order=-1)

#    a = arrays.FloatArray(
#        label="a",
#        default=numpy.array([1]),
#        doc="Coefficient of the cubic term in the first state variable",
#        order=-1)

#    b = arrays.FloatArray(
#        label="b",
#        default=numpy.array([3]),
#        doc="Coefficient of the squared term in the first state variabel",
#        order=-1)

    yc = arrays.FloatArray(
        label="y0",
        default=numpy.array([1]),
        doc="Additive coefficient for the second state variable",
        order=-1)

#    d = arrays.FloatArray(
#        label="d",
#        default=numpy.array([5]),
#        doc="Coefficient of the squared term in the second state variable",
#        order=-1)

    tau0 = arrays.FloatArray(
        label="tau0",
        range=basic.Range(lo=100.0, hi=5000, step=10),
        default=numpy.array([2857.0]),
        doc="Temporal scaling in the z state variable",
        order=4)

#    s = arrays.FloatArray(
#        label="s",
#        default=numpy.array([4]),
#        doc="Linear coefficient in the third state variable",
#        order=-1)

    x0 = arrays.FloatArray(
        label="x0",
        range=basic.Range(lo=-0.5, hi=1.5, step=0.1),
        default=numpy.array([0.0]),
        doc="Excitability parameter",
        order=3)

    x0cr = arrays.FloatArray(
        label="x0cr",
        range=basic.Range(lo=-1.0, hi=1.0, step=0.1),
        default=numpy.array([2.46018518518519]),
        doc="Critical excitability parameter",
        order=-1)

    r = arrays.FloatArray(
        label="r",
        range=basic.Range(lo=0.0, hi=1.0, step=0.1),
        default=numpy.array([43.0/108.0]),
        doc="Excitability parameter scaling",
        order=-1)

    Iext1 = arrays.FloatArray(
        label="Iext1",
        range=basic.Range(lo=1.5, hi=5.0, step=0.1),
        default=numpy.array([3.1]),
        doc="External input current to the first population",
        order=1)

    slope = arrays.FloatArray(
        label="slope",
        range=basic.Range(lo=-16.0, hi=6.0, step=0.1),
        default=numpy.array([0.]),
        doc="Linear coefficient in the first state variable",
        order=5)

    Kvf = arrays.FloatArray(
        label="K_vf",
        default=numpy.array([0.0]),
        range=basic.Range(lo=0.0, hi=4.0, step=0.5),
        doc="Coupling scaling on a very fast time scale.",
        order=6)

    K = arrays.FloatArray(
        label="K",
        default=numpy.array([0.0]),
        range=basic.Range(lo=-4.0, hi=4.0, step=0.1),
        doc="Permittivity coupling, that is from the fast time scale toward the slow time scale",
        order=8)

    tau1 = arrays.FloatArray(
        label="tau1",
        default=numpy.array([1.0]),
        range=basic.Range(lo=0.001, hi=10.0, step=0.001),
        doc="Time scaling of the whole system",
        order=9)

    state_variable_range = basic.Dict(
        label="State variable ranges [lo, hi]",
        default={"y0": numpy.array([-2., 2.]),
                 "y1": numpy.array([-2.0, 5.0])},
        doc="n/a",
        order=16
        )

    variables_of_interest = basic.Enumerate(
        label="Variables watched by Monitors",
        options=["y0", "y1"],
        default=["y0", "y1"],
        select_multiple=True,
        doc="""default state variables to be monitored""",
        order=-1)

    state_variables = ["y0", "y1"]

    _nvar = 2
    cvar = numpy.array([0, 1], dtype=numpy.int32)

    def dfun(self, state_variables, coupling, local_coupling=0.0,
             array=numpy.array, where=numpy.where, concat=numpy.concatenate):
        r"""
        Computes the derivatives of the state variables of the Epileptor
        with respect to time.

        Implementation note: we expect this version of the Epileptor to be used
        in a vectorized manner. Concretely, y has a shape of (2, n) where n is
        the number of nodes in the network. An consequence is that
        the original use of if/else is translated by calculated both the true
        and false forms and mixing them using a boolean mask.

        Variables of interest to be used by monitors: -y[0] + y[3]

            .. math::
            \dot{y_{0}} &=& yc - f_{1}(y_{0}, y_{1}) - y_{2} + I_{ext1} \\
            \dot{y_{1}} &=&
            \begin{cases}
            (f_z(y_{0}) - y_{1}-0.1 y_{1}^{7})/tau0 & \text{if } y_{0}<5/3 \\
            (f_z(y_{0}) - y_{1})/tau0           & \text{if } y_{0} \geq 5/3
            \end{cases} \\

        where:
            .. math::
                f_{1}(y_{0}, y_{3}) =
                \begin{cases}
                a ( y_{0} -5/3 )^{3} - b ( y_{0} -5/3 )^2 & \text{if } y_{0} <5/3\\
                ( 5*( y_{0} -5/3 ) - 0.6(y_{1}-4)^2 -slope) ( y_{0} - 5/3 ) &\text{if }y_{0} \geq 5/3
                \end{cases}
        and:

            .. math::
                f_z(y_{0})  =
                \begin{cases}
                4 * (y_{0} - r*x0 + x0_{cr}) & \text{linear} \\
                \frac{3}{1+e^{-10*(y_{0}-7/6)}} - r*x0 + x0_{cr} & \text{sigmoidal} \\
                \end{cases}


        """

        y = state_variables
        ydot = numpy.empty_like(state_variables)

        Iext1 = self.Iext1 + local_coupling * y[0]
        c_pop1 = coupling[0, :]

        # population 1
        if_ydot0 = y[0] ** 2 + 2.0 * y[0] #self.a=1.0, self.b=-2.0
        else_ydot0 = 5 * y[0] - 0.6 * (y[1] - 4.0) ** 2 -self.slope
        ydot[0] = self.tau1 * (self.yc - y[1] + Iext1 + self.Kvf*c_pop1 - where(y[0] < 0.0, if_ydot0, else_ydot0) * y[0])

        if numpy.any(ydot[0] == numpy.nan) or numpy.any(ydot[0] == numpy.inf):
            print "error"

        # energy
        if_ydot1 = - 0.1 * y[1] ** 7
        else_ydot1 = 0
        if self.zmode == 'lin':
            fz = 4 * (y[0] - self.r * self.x0 + self.x0cr) + where(y[1] < 0.0, if_ydot1, else_ydot1) #self.x0
        elif self.zmode == 'sig':
            fz = 3 / (1 + numpy.exp(-10*(y[0] + 0.5))) - self.r * self.x0 + self.x0cr
        else:
            raise ValueError('zmode has to be either ""lin"" or ""sig"" for linear and sigmoidal fz(), respectively')
        ydot[1] = self.tau1*(fz - y[1] + self.K * c_pop1)/self.tau0

        return ydot


    def jacobian(self, state_variables, coupling, local_coupling=0.0,
            array=numpy.array, where=numpy.where, concat=numpy.concatenate):
        r"""
        Computes the Jacobian of the state variables of the Epileptor
        with respect to time.

        Implementation note: we expect this version of the Epileptor to be used
        in a vectorized manner. Concretely, y has a shape of (2, n) where n is
        the number of nodes in the network. An consequence is that
        the original use of if/else is translated by calculated both the true
        and false forms and mixing them using a boolean mask.

        Variables of interest to be used by monitors: -y[0] + y[3]

            .. math::
            \dot{y_{0}} &=& yc - f_{1}(y_{0}, y_{1}) - y_{2} + I_{ext1} \\
            \dot{y_{1}} &=&
            \begin{cases}
            (f_z(y_{0}) - y_{1}-0.1 y_{1}^{7})/tau0 & \text{if } y_{0}<5/3 \\
            (f_z(y_{0}) - y_{1})/tau0           & \text{if } y_{0} \geq 5/3
            \end{cases} \\

        where:
            .. math::
                f_{1}(y_{0}, y_{3}) =
                \begin{cases}
                a ( y_{0} -5/3 )^{3} - b ( y_{0} -5/3 )^2 & \text{if } y_{0} <5/3\\
                ( 5*( y_{0} -5/3 ) - 0.6(y_{1}-4)^2 -slope) ( y_{0} - 5/3 ) &\text{if }y_{0} \geq 5/3
                \end{cases}
        and:

            .. math::
                f_z(y_{0})  =
                \begin{cases}
                4 * (y_{0} - r*x0 + x0_{cr}) & \text{linear} \\
                \frac{3}{1+e^{-10*(y_{0}-7/6)}} - r*x0 + x0_{cr} & \text{sigmoidal} \\
                \end{cases}


        """

        y = state_variables

        n_ep = state_variables.shape[1]
        # population 1
        jac_xx = where(y[0] < 0.0, numpy.diag(3*y[0]**2 + 4.0*y[0]), numpy.diag(5*y[0]+0.6*(y[1]-4.0)**2-self.slope))
        jac_xz = where(y[0] < 0.0, numpy.diag(numpy.zeros((n_ep,), dtype=y.dtype)), numpy.diag(1.2*(y[1]-4.0)*y[0]))

        # energy
        # The terms resulting from coupling from other regions, have to be added later on
        if_fz = - 0.1 * y[1] ** 7
        else_fz = 0
        jac_zz = -numpy.diag(numpy.ones((n_ep,)), dtype=y.dtype) / self.tau0
        if self.zmode == 'lin':
            jac_zx = numpy.diag(4.0)/self.tau0
            jac_zz -= numpy.diag(where(y[1] < 0.0, if_fz, else_fz))
        elif self.zmode == 'sig':
            exp_fun = numpy.exp(-10.0*(y[0]+0.5))
            jac_zx = numpy.diag(30.0*exp_fun/(1+exp_fun)**2)/self.tau0
        else:
            raise ValueError('zmode has to be either ""lin"" or ""sig"" for linear and sigmoidal fz(), respectively')

        return concat([numpy.hstack([jac_xx, jac_xz]),numpy.hstack([jac_zx, jac_zz])],axis=0)