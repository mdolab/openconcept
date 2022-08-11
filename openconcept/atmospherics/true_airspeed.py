import numpy as np

from openmdao.api import ExplicitComponent


class TrueAirspeedComp(ExplicitComponent):
    """
    Computes true airspeed from equivalent airspeed and density

    Inputs
    ------
    fltcond|rho : float
        Density (vector, kg/m3)
    fltcond|Ueas : float
        Equivalent airspeed (vector, m/s)

    Outputs
    -------
    fltcond|Utrue : float
        True airspeed (vector, m/s)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length) (default 1)
    """

    def initialize(self):
        self.options.declare("num_nodes", types=int)

    def setup(self):
        num_points = self.options["num_nodes"]

        self.add_input("fltcond|Ueas", units="m / s", shape=num_points)
        self.add_input("fltcond|rho", units="kg * m**-3", shape=num_points)
        self.add_output("fltcond|Utrue", units="m / s", shape=num_points)

        arange = np.arange(num_points)
        self.declare_partials("fltcond|Utrue", "fltcond|Ueas", rows=arange, cols=arange)
        self.declare_partials("fltcond|Utrue", "fltcond|rho", rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        rho_isa_kgm3 = 1.225
        outputs["fltcond|Utrue"] = inputs["fltcond|Ueas"] * np.sqrt(rho_isa_kgm3 / inputs["fltcond|rho"])

    def compute_partials(self, inputs, partials):
        rho_isa_kgm3 = 1.225
        partials["fltcond|Utrue", "fltcond|Ueas"] = np.sqrt(rho_isa_kgm3 / inputs["fltcond|rho"])
        partials["fltcond|Utrue", "fltcond|rho"] = (
            inputs["fltcond|Ueas"] * np.sqrt(rho_isa_kgm3) * (-1 / 2) * inputs["fltcond|rho"] ** (-3 / 2)
        )


class EquivalentAirspeedComp(ExplicitComponent):
    """
    Computes equivalent airspeed from true airspeed and density

    Inputs
    ------
    fltcond|rho : float
        Density (vector, kg/m3)
    fltcond|Utrue : float
        True airspeed (vector, m/s)

    Outputs
    -------
    fltcond|Ueas : float
        Equivalent airspeed (vector, m/s)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length) (default 1)
    """

    def initialize(self):
        self.options.declare("num_nodes", types=int)

    def setup(self):
        num_points = self.options["num_nodes"]

        self.add_output("fltcond|Ueas", units="m / s", shape=num_points)
        self.add_input("fltcond|rho", units="kg * m**-3", shape=num_points)
        self.add_input("fltcond|Utrue", units="m / s", shape=num_points)

        arange = np.arange(num_points)
        self.declare_partials("fltcond|Ueas", "fltcond|Utrue", rows=arange, cols=arange)
        self.declare_partials("fltcond|Ueas", "fltcond|rho", rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        rho_isa_kgm3 = 1.225
        outputs["fltcond|Ueas"] = inputs["fltcond|Utrue"] * np.sqrt(inputs["fltcond|rho"] / rho_isa_kgm3)

    def compute_partials(self, inputs, partials):
        rho_isa_kgm3 = 1.225
        partials["fltcond|Ueas", "fltcond|Utrue"] = np.sqrt(inputs["fltcond|rho"] / rho_isa_kgm3)
        partials["fltcond|Ueas", "fltcond|rho"] = (
            (1 / 2) * inputs["fltcond|Utrue"] * (inputs["fltcond|rho"] / rho_isa_kgm3) ** (-1 / 2) / rho_isa_kgm3
        )
