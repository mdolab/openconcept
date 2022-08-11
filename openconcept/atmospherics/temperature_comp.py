import numpy as np

from openmdao.api import ExplicitComponent

from .atmospherics_data import get_mask_arrays, compute_temps, compute_temp_derivs


class TemperatureComp(ExplicitComponent):
    """
    This component computes temperature from altitude using the 1976 Standard Atmosphere.

    Adapted from:
    J.P. Jasa, J.T. Hwang, and J.R.R.A. Martins: Design and Trajectory Optimization of a Morphing Wing Aircraft
    2018 AIAA/ASCE/AHS/ASC Structures, Structural Dynamics, and Materials Conference; AIAA SciTech Forum, January 2018

    Inputs
    ------
    fltcond|h : float
        Altitude (vector, m)
    fltcond|TempIncrement : float
        Offset for temperature; useful for modeling hot (+ increment) or cold (- increment) days (vector, deg C)

    Outputs
    -------
    fltcond|T : float
        Temperature at flight condition (vector, K)

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
        self.add_input("fltcond|TempIncrement", shape=num_points, val=0.0, units="degC")
        self.add_output("fltcond|T", shape=num_points, lower=0.0, units="K")

        arange = np.arange(num_points)
        self.declare_partials("fltcond|T", "fltcond|h", rows=arange, cols=arange)
        self.declare_partials("fltcond|T", "fltcond|TempIncrement", rows=arange, cols=arange, val=1.0)

    def compute(self, inputs, outputs):
        h_m = inputs["fltcond|h"]

        self.tropos_mask, self.strato_mask, self.smooth_mask = get_mask_arrays(h_m)
        temp_K = compute_temps(h_m, self.tropos_mask, self.strato_mask, self.smooth_mask)

        outputs["fltcond|T"] = temp_K + inputs["fltcond|TempIncrement"]

    def compute_partials(self, inputs, partials):
        h_m = inputs["fltcond|h"]

        derivs = compute_temp_derivs(h_m, self.tropos_mask, self.strato_mask, self.smooth_mask)

        partials["fltcond|T", "fltcond|h"] = derivs
