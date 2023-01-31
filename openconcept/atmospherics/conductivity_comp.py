import numpy as np
from openmdao.api import ExplicitComponent


class ConductivityComp(ExplicitComponent):
    """
    This component computes thermal conductivity of the air from altitude based on a piecewise linear curve fit of
    data from Engineering ToolBox (https://www.engineeringtoolbox.com/international-standard-atmosphere-d_985.html).

    The fit is valid up to around 65k ft, but still close beyond that. The piecewise curve fit is smoothed using
    a tanh transition between the two phases. This component does not consider any temperature offset.

    Inputs
    ------
    fltcond|h : float
        Altitude (vector, m)

    Outputs
    -------
    fltcond|k : float
        Thermal conductivity at flight condition (vector, W/(m-K))

    Options
    -------
    num_nodes : int
        Number of analysis points to run, i.e. length of vector inputs (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare("num_nodes", types=int)

    def setup(self):
        num_points = self.options["num_nodes"]

        self.add_input("fltcond|h", shape=num_points, units="m")
        self.add_output("fltcond|k", shape=num_points, units="W / (m * K)")

        arange = np.arange(num_points)
        self.declare_partials("fltcond|k", "fltcond|h", rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        h_m = inputs["fltcond|h"]

        # Fits for thermal conductivity in 1e-2 W/m-K
        k_lower_alt = -5.225e-5 * h_m + 2.5349
        k_higher_alt = 1.952

        # Transition between two fits at ~11k m
        tanh_smooth = 0.5 * np.tanh(0.01 * (h_m - 11.13e3)) + 0.5

        outputs["fltcond|k"] = 1e-2 * (k_lower_alt * (1 - tanh_smooth) + k_higher_alt * tanh_smooth)

    def compute_partials(self, inputs, partials):
        h_m = inputs["fltcond|h"]

        k_lower_alt = -5.225e-5 * h_m + 2.5349
        k_higher_alt = 1.952

        # Transition between two fits at ~11k m
        tanh_smooth = 0.5 * np.tanh(0.01 * (h_m - 11.13e3)) + 0.5

        # Partials
        dk_lower_alt_dh = -5.225e-5
        dk_higher_alt_dh = 0.0
        dtanh_smooth_dh = 0.5 * (1 - np.tanh(0.01 * (h_m - 11.13e3)) ** 2) * 0.01

        partials["fltcond|k", "fltcond|h"] = 1e-2 * (
            dk_lower_alt_dh * (1 - tanh_smooth)
            - k_lower_alt * dtanh_smooth_dh
            + dk_higher_alt_dh * tanh_smooth
            + k_higher_alt * dtanh_smooth_dh
        )
