import numpy as np
import openmdao.api as om


class WingMACTrapezoidal(om.ExplicitComponent):
    """
    Compute the mean aerodynamic chord of a trapezoidal planform.

    Inputs
    ------
    S_ref : float
        Wing planform area (scalar, sq m)
    AR : float
        Wing aspect ratio (scalar, dimensionless)
    taper : float
        Wing taper ratio (scalar, dimensionless)

    Outputs
    -------
    MAC : float
        Mean aerodynamic chord of the trapezoidal planform (scalar, m)
    """

    def setup(self):
        self.add_input("S_ref", units="m**2")
        self.add_input("AR")
        self.add_input("taper")
        self.add_output("MAC", lower=1e-6, units="m")
        self.declare_partials("MAC", "*")

    def compute(self, inputs, outputs):
        S = inputs["S_ref"]
        AR = inputs["AR"]
        taper = inputs["taper"]

        c_root = np.sqrt(S / AR) * 2 / (1 + taper)
        c_tip = taper * c_root
        outputs["MAC"] = 2 / 3 * (c_root + c_tip - c_root * c_tip / (c_root + c_tip))

    def compute_partials(self, inputs, J):
        S = inputs["S_ref"]
        AR = inputs["AR"]
        taper = inputs["taper"]

        c_root = np.sqrt(S / AR) * 2 / (1 + taper)
        dcr_dS = 0.5 / np.sqrt(S * AR) * 2 / (1 + taper)
        dcr_dAR = -0.5 * S**0.5 / AR**1.5 * 2 / (1 + taper)
        dcr_dtaper = -np.sqrt(S / AR) * 2 / (1 + taper) ** 2

        c_tip = taper * c_root

        dMAC_dcr = 2 / 3 * (1 - c_tip**2 / (c_root + c_tip) ** 2)
        dMAC_dct = 2 / 3 * (1 - c_root**2 / (c_root + c_tip) ** 2)

        J["MAC", "S_ref"] = (dMAC_dcr + dMAC_dct * taper) * dcr_dS
        J["MAC", "AR"] = (dMAC_dcr + dMAC_dct * taper) * dcr_dAR
        J["MAC", "taper"] = (dMAC_dcr + dMAC_dct * taper) * dcr_dtaper + dMAC_dct * c_root


class WingSpan(om.ExplicitComponent):
    """
    Compute the wing span as the square root of wing area times aspect ratio.

    Inputs
    ------
    S_ref : float
        Wing planform area (scalar, sq m)
    AR : float
        Wing aspect ratio (scalar, dimensionless)

    Outputs
    -------
    span : float
        Wing span (scalar, m)
    """

    def setup(self):
        self.add_input("S_ref", units="m**2")
        self.add_input("AR")

        self.add_output("span", units="m")
        self.declare_partials(["span"], ["*"])

    def compute(self, inputs, outputs):
        b = inputs["S_ref"] ** 0.5 * inputs["AR"] ** 0.5
        outputs["span"] = b

    def compute_partials(self, inputs, J):
        J["span", "S_ref"] = 0.5 * inputs["S_ref"] ** (0.5 - 1) * inputs["AR"] ** 0.5
        J["span", "AR"] = inputs["S_ref"] ** 0.5 * 0.5 * inputs["AR"] ** (0.5 - 1)
