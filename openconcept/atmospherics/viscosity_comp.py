import numpy as np
from openmdao.api import ExplicitComponent


class ViscosityComp(ExplicitComponent):
    """
    This component computes thermal conductivity of the air from altitude based on a piecewise linear fit of
    tabulated data from Digital Dutch (https://www.digitaldutch.com/atmoscalc/table.htm).

    The piecewise curve fit is smoothed using a tanh transition between the two phases. This component does
    not consider any temperature offset other than incorporating changes to density that have already been made.

    Inputs
    ------
    fltcond|h : float
        Altitude (vector, m)
    fltcond|rho : float
        Density (vector, kg/m^3)

    Outputs
    -------
    fltcond|mu : float
        Dynamic viscosity at flight condition (vector, N-s/m^2)
    fltcond|visc_kin : float
        Kinematic viscosity at flight condition (vector, m^2/s)

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
        self.add_input("fltcond|rho", shape=num_points, units="kg/m**3")
        self.add_output("fltcond|mu", shape=num_points, units="N * s / m**2")
        self.add_output("fltcond|visc_kin", shape=num_points, units="m**2 / s")

        arange = np.arange(num_points)
        self.declare_partials("fltcond|mu", "fltcond|h", rows=arange, cols=arange)
        self.declare_partials("fltcond|visc_kin", ["fltcond|h", "fltcond|rho"], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        h_m = inputs["fltcond|h"]

        # Fits for dynamic viscosity in N-s/m^2
        mu_lower_alt = -3.413179e-10 * h_m + 1.813085e-5
        mu_mid_alt = 1.43226e-5
        mu_higher_alt = 5.608883e-11 * h_m + 1.320169e-5

        # Transition between low and mid altitudes at ~11k m and mid to high at 20k m
        low_mid_smooth = 0.5 * np.tanh(0.01 * (h_m - 11.1e3)) + 0.5
        mid_high_smooth = 0.5 * np.tanh(0.01 * (h_m - 20e3)) + 0.5

        outputs["fltcond|mu"] = mu_lower_alt * (1 - low_mid_smooth) + mu_mid_alt * low_mid_smooth * (1 - mid_high_smooth) + mu_higher_alt * mid_high_smooth
        outputs["fltcond|visc_kin"] = outputs["fltcond|mu"] / inputs["fltcond|rho"]

    def compute_partials(self, inputs, partials):
        h_m = inputs["fltcond|h"]

        # Fits for dynamic viscosity in N-s/m^2
        mu_lower_alt = -3.413179e-10 * h_m + 1.813085e-5
        mu_mid_alt = 1.43226e-5
        mu_higher_alt = 5.608883e-11 * h_m + 1.320169e-5

        # Transition between low and mid altitudes at ~11k m and mid to high at 20k m
        low_mid_smooth = 0.5 * np.tanh(0.01 * (h_m - 11.1e3)) + 0.5
        mid_high_smooth = 0.5 * np.tanh(0.01 * (h_m - 20e3)) + 0.5

        mu = mu_lower_alt * (1 - low_mid_smooth) + mu_mid_alt * low_mid_smooth * (1 - mid_high_smooth) + mu_higher_alt * mid_high_smooth

        # Partials
        dmu_lower_alt_dh = -3.413179e-10
        dmu_mid_alt_dh = 0.0
        dmu_higher_alt_dh = 5.608883e-11
        dlow_mid_smooth_dh = 0.5 * (1 - np.tanh(0.01 * (h_m - 11.1e3)) ** 2) * 0.01
        dmid_high_smooth_dh = 0.5 * (1 - np.tanh(0.01 * (h_m - 20e3)) ** 2) * 0.01

        partials["fltcond|mu", "fltcond|h"] = (
            dmu_lower_alt_dh * (1 - low_mid_smooth) - mu_lower_alt * dlow_mid_smooth_dh
            + dmu_mid_alt_dh * low_mid_smooth * (1 - mid_high_smooth) + mu_mid_alt * dlow_mid_smooth_dh * (1 - mid_high_smooth) - mu_mid_alt * low_mid_smooth * dmid_high_smooth_dh
            + dmu_higher_alt_dh * mid_high_smooth + mu_higher_alt * dmid_high_smooth_dh
        )
        partials["fltcond|visc_kin", "fltcond|h"] = partials["fltcond|mu", "fltcond|h"] / inputs["fltcond|rho"]
        partials["fltcond|visc_kin", "fltcond|rho"] = -mu / inputs["fltcond|rho"]**2
