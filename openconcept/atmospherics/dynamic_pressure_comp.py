import numpy as np

from openmdao.api import ExplicitComponent


class DynamicPressureComp(ExplicitComponent):
    """
    Calculates dynamic pressure from true airspeed and density.

    Inputs
    ------
    fltcond|Utrue : float
        True airspeed (vector, m/s)
    fltcond|rho : float
        Density (vector, m/s)

    Outputs
    -------
    fltcond|q : float
        Dynamic pressure (vector, Pa)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length) (default 1)
    """

    def initialize(self):
        self.options.declare("num_nodes", types=int)

    def setup(self):
        nn = self.options["num_nodes"]

        self.add_input("fltcond|Utrue", units="m/s", shape=(nn,))
        self.add_input("fltcond|rho", units="kg * m**-3", shape=(nn,))
        self.add_output("fltcond|q", units="N * m**-2", shape=(nn,))

        arange = np.arange(nn)
        self.declare_partials("fltcond|q", "fltcond|rho", rows=arange, cols=arange)
        self.declare_partials("fltcond|q", "fltcond|Utrue", rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        outputs["fltcond|q"] = 0.5 * inputs["fltcond|rho"] * inputs["fltcond|Utrue"] ** 2

    def compute_partials(self, inputs, partials):
        partials["fltcond|q", "fltcond|rho"] = 0.5 * inputs["fltcond|Utrue"] ** 2
        partials["fltcond|q", "fltcond|Utrue"] = inputs["fltcond|rho"] * inputs["fltcond|Utrue"]
