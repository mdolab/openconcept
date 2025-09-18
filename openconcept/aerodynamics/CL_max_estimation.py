"""
@File    :   CLmax_jet_transport.py
@Date    :   2023/03/24
@Author  :   Eytan Adler
@Description : Max lift coefficient estimate for jet transport aircraft
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
import openmdao.api as om

# ==============================================================================
# Extension modules
# ==============================================================================


class CleanCLmax(om.ExplicitComponent):
    """
    Predict the maximum lift coefficient of the clean configuration.
    Method from Raymer (Equation 12.15, 1992 edition).

    Inputs
    ------
    ac|aero|airfoil_Cl_max : float
        Maximum 2D lift coefficient of the wing airfoil (scalar, dimensionless)
    ac|geom|wing|c4sweep : float
        Wing quarter chord sweep angle (scalar, radians)

    Outputs
    -------
    CL_max_clean : float
        Maximum lift coefficient with no flaps or slats (scalar, dimensionless)

    Options
    -------
    fudge_factor : float
        Optional multiplier on resulting lift coefficient, by default 1.0
    """

    def initialize(self):
        self.options.declare("fudge_factor", default=1.0, desc="Multiplier of CL max")

    def setup(self):
        self.add_input("ac|aero|airfoil_Cl_max")
        self.add_input("ac|geom|wing|c4sweep", units="rad")
        self.add_output("CL_max_clean")
        self.declare_partials("CL_max_clean", "*")

    def compute(self, inputs, outputs):
        Cl_max = inputs["ac|aero|airfoil_Cl_max"]
        sweep = inputs["ac|geom|wing|c4sweep"]

        outputs["CL_max_clean"] = self.options["fudge_factor"] * 0.9 * Cl_max * np.cos(sweep)

    def compute_partials(self, inputs, J):
        Cl_max = inputs["ac|aero|airfoil_Cl_max"]
        sweep = inputs["ac|geom|wing|c4sweep"]
        mult = self.options["fudge_factor"]

        J["CL_max_clean", "ac|aero|airfoil_Cl_max"] = mult * 0.9 * np.cos(sweep)
        J["CL_max_clean", "ac|geom|wing|c4sweep"] = -mult * 0.9 * Cl_max * np.sin(sweep)


class FlapCLmax(om.ExplicitComponent):
    """
    Predict the maximum lift coefficient with Fowler flaps and slats
    extended. Method from Roskam Part VI Chapter 8 1989.

    Inputs
    ------
    flap_extension : float
        Flap extension amount (scalar, deg)
    ac|geom|wing|c4sweep : float
        Wing sweep at 25% mean aerodynamic chord (scalar, radians)
    ac|geom|wing|toverc : float
        Wing thickness-to-chord ratio (scalar, dimensionless)
    CL_max_clean : float
        Maximum lift coefficient with no flaps or slats (scalar, dimensionless)

    Outputs
    -------
    CL_max_flap : float
        Maximum lift coefficient with flaps and slats extended (scalar, dimensionless)


    Options
    -------
    flap_chord_frac : float
        Flap chord divided by wing chord, by default 0.2
    wing_area_flapped_frac : float
        Flapped wing area divided by total wing area. Flapped wing area integrates the chord
        over any portions of the span that contain flaps (not just the area of the flap itself).
        By default 0.9.
    slat_chord_frac : float
        Slat chord divided by wing chord, by default 0.1. Set to 0.0 to remove slats.
    slat_span_frac : float
        Fraction of the wing span that has slats, by default 0.8
    fudge_factor : float
        Optional multiplier on resulting lift coefficient, by default 1.0
    """

    def initialize(self):
        self.options.declare("flap_chord_frac", default=0.2, desc="Flap chord / wing chord")
        self.options.declare("wing_area_flapped_frac", default=0.9, desc="Flapped wing area / wing area")
        self.options.declare("slat_chord_frac", default=0.1, desc="Slat chord / wing chord")
        self.options.declare("slat_span_frac", default=0.8, desc="Slat span / wing span")
        self.options.declare("fudge_factor", default=1.0, desc="Multiplier of CL max")

    def setup(self):
        self.add_input("flap_extension", units="deg")
        self.add_input("ac|geom|wing|c4sweep", units="rad")
        self.add_input("CL_max_clean")
        self.add_input("ac|geom|wing|toverc")

        self.add_output("CL_max_flap")

        self.declare_partials("CL_max_flap", ["flap_extension", "ac|geom|wing|c4sweep", "ac|geom|wing|toverc"])
        self.declare_partials("CL_max_flap", "CL_max_clean", val=self.options["fudge_factor"])

    def compute(self, inputs, outputs):
        delta = inputs["flap_extension"]
        sweep = inputs["ac|geom|wing|c4sweep"]
        tc = inputs["ac|geom|wing|toverc"]

        # -------------- Compute the increase in 2D airfoil lift coefficient from flaps --------------
        # See Roskam 1989 Part VI Chapter 8 Equation 8.18
        delta_cl_max_base = 1 + 2.33 * tc - 77.9 * tc**2 + 1120 * tc**3 - 3430 * tc**4
        k1 = 4 * self.options["flap_chord_frac"]
        k2 = 0.4 + 0.0234 * delta - 2.04e-4 * delta**2
        k3 = 1.36 * k2 - 0.389 * k2**2
        delta_cl_max = delta_cl_max_base * k1 * k2 * k3

        # -------------- Compute the increment to the total wing lift coefficient from flaps --------------
        # See Roskam 1989 Part VI Chapter 8 Equation 8.29
        delta_CL_max_flaps = (
            delta_cl_max
            * self.options["wing_area_flapped_frac"]
            * (1 - 0.08 * np.cos(sweep) ** 2)
            * np.cos(sweep) ** 0.75
        )

        # -------------- Compute the increment to the total wing lift coefficient from slats --------------
        # See Roskam 1989 Part VI Chapter 8 Equation 8.30
        delta_CL_max_slats = (
            7.11 * self.options["slat_chord_frac"] * self.options["slat_span_frac"] ** 2 * np.cos(sweep) ** 2
        )

        outputs["CL_max_flap"] = self.options["fudge_factor"] * (
            inputs["CL_max_clean"] + delta_CL_max_flaps + delta_CL_max_slats
        )

    def compute_partials(self, inputs, J):
        delta = inputs["flap_extension"]
        sweep = inputs["ac|geom|wing|c4sweep"]
        tc = inputs["ac|geom|wing|toverc"]

        # -------------- Compute the increase in 2D airfoil lift coefficient from flaps --------------
        # See Roskam 1989 Part VI Chapter 8 Equation 8.18
        delta_cl_max_base = 1 + 2.33 * tc - 77.9 * tc**2 + 1120 * tc**3 - 3430 * tc**4
        k1 = 4 * self.options["flap_chord_frac"]
        k2 = 0.4 + 0.0234 * delta - 2.04e-4 * delta**2
        k3 = 1.36 * k2 - 0.389 * k2**2
        delta_cl_max = delta_cl_max_base * k1 * k2 * k3

        ddclmax_dtc = k1 * k2 * k3 * (2.33 - 2 * 77.9 * tc + 3 * 1120 * tc**2 - 4 * 3430 * tc**3)
        dk2_ddelta = 0.0234 - 2 * 2.04e-4 * delta
        dk3_ddelta = (1.36 - 2 * 0.389 * k2) * dk2_ddelta
        ddclmax_ddelta = delta_cl_max_base * k1 * (k2 * dk3_ddelta + dk2_ddelta * k3)

        # -------------- Compute the increment to the total wing lift coefficient from flaps --------------
        # See Roskam 1989 Part VI Chapter 8 Equation 8.29
        J["CL_max_flap", "ac|geom|wing|c4sweep"] = (
            delta_cl_max
            * self.options["wing_area_flapped_frac"]
            * (np.sin(sweep) * (0.22 * np.cos(sweep) ** 2 - 0.75) / np.cos(sweep) ** 0.25)
        )

        ddclmaxflap_ddclmax = (
            self.options["wing_area_flapped_frac"] * (1 - 0.08 * np.cos(sweep) ** 2) * np.cos(sweep) ** 0.75
        )
        J["CL_max_flap", "flap_extension"] = ddclmaxflap_ddclmax * ddclmax_ddelta
        J["CL_max_flap", "ac|geom|wing|toverc"] = ddclmaxflap_ddclmax * ddclmax_dtc

        # -------------- Compute the increment to the total wing lift coefficient from slats --------------
        # See Roskam 1989 Part VI Chapter 8 Equation 8.30
        J["CL_max_flap", "ac|geom|wing|c4sweep"] -= (
            7.11
            * self.options["slat_chord_frac"]
            * self.options["slat_span_frac"] ** 2
            * 2
            * np.cos(sweep)
            * np.sin(sweep)
        )

        J["CL_max_flap", "flap_extension"] *= self.options["fudge_factor"]
        J["CL_max_flap", "ac|geom|wing|toverc"] *= self.options["fudge_factor"]
        J["CL_max_flap", "ac|geom|wing|c4sweep"] *= self.options["fudge_factor"]
