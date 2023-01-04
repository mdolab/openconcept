import numpy as np

from openmdao.api import ExplicitComponent


gamma = 1.4
R = 287.058


class SpeedOfSoundComp(ExplicitComponent):
    """
    This component computes speed of sound from temperature.

    Adapted from:
    J.P. Jasa, J.T. Hwang, and J.R.R.A. Martins: Design and Trajectory Optimization of a Morphing Wing Aircraft
    2018 AIAA/ASCE/AHS/ASC Structures, Structural Dynamics, and Materials Conference; AIAA SciTech Forum, January 2018

    Inputs
    ------
    fltcond|T : float
        Temperature at flight condition (vector, K)

    Outputs
    -------
    fltcond|a : float
        Speed of sound (vector, m/s)

    Options
    -------
    num_nodes : int
        Number of analysis points to run, i.e. length of vector inputs (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare("num_nodes", types=int)

    def setup(self):
        num_points = self.options["num_nodes"]

        self.add_input("fltcond|T", shape=num_points, units="K")
        self.add_output("fltcond|a", shape=num_points, units="m/s")

        arange = np.arange(num_points)
        self.declare_partials("fltcond|a", "fltcond|T", rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        T_K = inputs["fltcond|T"]

        outputs["fltcond|a"] = np.sqrt(gamma * R * T_K)

    def compute_partials(self, inputs, partials):
        T_K = inputs["fltcond|T"]

        data = 0.5 * np.sqrt(gamma * R / T_K)
        partials["fltcond|a", "fltcond|T"] = data
