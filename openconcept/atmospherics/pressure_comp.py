import numpy as np

from openmdao.api import ExplicitComponent

from .atmospherics_data import get_mask_arrays, compute_pressures, compute_pressure_derivs


class PressureComp(ExplicitComponent):
    """
    This component computes pressure from altitude using the 1976 Standard Atmosphere.

    Adapted from:
    J.P. Jasa, J.T. Hwang, and J.R.R.A. Martins: Design and Trajectory Optimization of a Morphing Wing Aircraft
    2018 AIAA/ASCE/AHS/ASC Structures, Structural Dynamics, and Materials Conference; AIAA SciTech Forum, January 2018

    Inputs
    ------
    fltcond|h : float
        Altitude (vector, m)

    Outputs
    -------
    fltcond|p : float
        Pressure at flight condition (vector, Pa)

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
        self.add_output("fltcond|p", shape=num_points, units="Pa")

        arange = np.arange(num_points)
        self.declare_partials("fltcond|p", "fltcond|h", rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        h_m = inputs["fltcond|h"]
        self.tropos_mask, self.strato_mask, self.smooth_mask = get_mask_arrays(h_m)
        p_Pa = compute_pressures(h_m, self.tropos_mask, self.strato_mask, self.smooth_mask)

        outputs["fltcond|p"] = p_Pa

    def compute_partials(self, inputs, partials):
        h_m = inputs["fltcond|h"]

        derivs = compute_pressure_derivs(h_m, self.tropos_mask, self.strato_mask, self.smooth_mask)

        partials["fltcond|p", "fltcond|h"] = derivs
