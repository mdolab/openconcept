import numpy as np

from openmdao.api import ExplicitComponent

R = 287.058


class DensityComp(ExplicitComponent):
    """
    This component computes density from pressure and temperature.

    Adapted from:
    J.P. Jasa, J.T. Hwang, and J.R.R.A. Martins: Design and Trajectory Optimization of a Morphing Wing Aircraft
    2018 AIAA/ASCE/AHS/ASC Structures, Structural Dynamics, and Materials Conference; AIAA SciTech Forum, January 2018

    Inputs
    ------
    fltcond|p : float
        Pressure at flight condition (vector, Pa)
    fltcond|T : float
        Temperature at flight condition (vector, K)

    Outputs
    -------
    fltcond|rho : float
        Density at flight condition (vector, kg/m^3)

    Options
    -------
    num_nodes : int
        Number of analysis points to run, i.e. length of vector inputs (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare("num_nodes", types=int)

    def setup(self):
        num_points = self.options["num_nodes"]

        self.add_input("fltcond|p", shape=num_points, units="Pa")
        self.add_input("fltcond|T", shape=num_points, units="K")
        self.add_output("fltcond|rho", shape=num_points, units="kg / m**3")

        arange = np.arange(num_points)
        self.declare_partials("fltcond|rho", "fltcond|p", rows=arange, cols=arange)
        self.declare_partials("fltcond|rho", "fltcond|T", rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        p_Pa = inputs["fltcond|p"]
        T_K = inputs["fltcond|T"]

        outputs["fltcond|rho"] = p_Pa / R / T_K

    def compute_partials(self, inputs, partials):
        p_Pa = inputs["fltcond|p"]
        T_K = inputs["fltcond|T"]

        data = 1.0 / R / T_K
        partials["fltcond|rho", "fltcond|p"] = data

        data = -p_Pa / R / T_K**2
        partials["fltcond|rho", "fltcond|T"] = data
