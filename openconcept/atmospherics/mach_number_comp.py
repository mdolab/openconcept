import numpy as np

from openmdao.api import ExplicitComponent


class MachNumberComp(ExplicitComponent):
    """
    Computes mach number from true airspeed and speed of sound

    Inputs
    ------
    fltcond|a : float
        Speed of sound (vector, m/s)
    fltcond|Utrue : float
        True airspeed (vector, m/s)

    Outputs
    -------
    fltcond|M : float
        Mach number (vector, dimensionless)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length) (default 1)
    """

    def initialize(self):
        self.options.declare("num_nodes", types=int)

    def setup(self):
        num_points = self.options["num_nodes"]

        self.add_input("fltcond|a", units="m / s", shape=num_points)
        self.add_input("fltcond|Utrue", units="m /s", shape=num_points)
        self.add_output("fltcond|M", shape=num_points)

        arange = np.arange(num_points)
        self.declare_partials("fltcond|M", "fltcond|a", rows=arange, cols=arange)
        self.declare_partials("fltcond|M", "fltcond|Utrue", rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        outputs["fltcond|M"] = inputs["fltcond|Utrue"] / inputs["fltcond|a"]

    def compute_partials(self, inputs, partials):
        num_points = self.options["num_nodes"]
        partials["fltcond|M", "fltcond|Utrue"] = np.ones(num_points) / inputs["fltcond|a"]
        partials["fltcond|M", "fltcond|a"] = -inputs["fltcond|Utrue"] / (inputs["fltcond|a"] ** 2)
