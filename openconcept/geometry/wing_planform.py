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
        self.add_output("MAC", units="m")
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
        dcr_dtaper = -np.sqrt(S / AR) * 2 / (1 + taper)**2

        c_tip = taper * c_root

        dMAC_dcr = 2 / 3 * (1 - c_tip**2 / (c_root + c_tip)**2)
        dMAC_dct = 2 / 3 * (1 - c_root**2 / (c_root + c_tip)**2)

        J["MAC", "S_ref"] = (dMAC_dcr + dMAC_dct * taper) * dcr_dS
        J["MAC", "AR"] = (dMAC_dcr + dMAC_dct * taper) * dcr_dAR
        J["MAC", "taper"] = (dMAC_dcr + dMAC_dct * taper) * dcr_dtaper


# class WingRoot_LinearTaper(om.ExplicitComponent):
#     """Inputs: ac|geom|wing|S_ref, ac|geom|wing|AR, ac|geom|wing|taper
#     Outputs: C_root
#     """

#     def setup(self):
#         self.add_input("ac|geom|wing|S_ref", units="ft**2", desc="Reference wing area in sq ft")
#         self.add_input("ac|geom|wing|AR", desc="Wing Aspect Ratio")
#         self.add_input("ac|geom|wing|taper", desc="Main Wing Taper Ratio")
#         self.add_output("C_root", units="ft")
#         self.declare_partials(["C_root"], ["*"])

#     def compute(self, inputs, outputs):
#         root_chord = (
#             2
#             * inputs["ac|geom|wing|S_ref"]
#             / (np.sqrt(inputs["ac|geom|wing|S_ref"] * inputs["ac|geom|wing|AR"]) * (1 + inputs["ac|geom|wing|taper"]))
#         )
#         outputs["C_root"] = root_chord

#     def compute_partials(self, inputs, J):
#         J["C_root", "ac|geom|wing|S_ref"] = 2 / (
#             (1 + inputs["ac|geom|wing|taper"]) * np.sqrt(inputs["ac|geom|wing|AR"] * inputs["ac|geom|wing|S_ref"])
#         ) - (inputs["ac|geom|wing|AR"] * inputs["ac|geom|wing|S_ref"]) / (
#             (1 + inputs["ac|geom|wing|taper"]) * (inputs["ac|geom|wing|AR"] * inputs["ac|geom|wing|S_ref"]) ** 1.5
#         )
#         J["C_root", "ac|geom|wing|AR"] = -inputs["ac|geom|wing|S_ref"] ** 2 / (
#             (inputs["ac|geom|wing|taper"] + 1) * (inputs["ac|geom|wing|AR"] * inputs["ac|geom|wing|S_ref"]) ** 1.5
#         )
#         J["C_root", "ac|geom|wing|taper"] = (
#             -2
#             * inputs["ac|geom|wing|S_ref"]
#             / (
#                 (inputs["ac|geom|wing|taper"] + 1) ** 2
#                 * (inputs["ac|geom|wing|AR"] * inputs["ac|geom|wing|S_ref"]) ** 0.5
#             )
#         )

# class WingMAC_Trapezoidal(om.ExplicitComponent):
#     """Inputs: ac|geom|wing|root_chord, ac|geom|wing|taper
#     Outputs: MAC"""

#     def setup(self):
#         self.add_input("ac|geom|wing|root_chord", units="ft", desc="Main wing root chord")
#         self.add_input("ac|geom|wing|taper", desc="Main wing taper ratio")
#         self.add_output("MAC", units="ft")
#         self.declare_partials(["MAC"], ["*"])

#     def compute(self, inputs, outputs):
#         meanAeroChord = (
#             (2 / 3)
#             * inputs["ac|geom|wing|root_chord"]
#             * (1 + inputs["ac|geom|wing|taper"] + inputs["ac|geom|wing|taper"] ** 2)
#             / (1 + inputs["ac|geom|wing|taper"])
#         )
#         outputs["MAC"] = meanAeroChord

#     def compute_partials(self, inputs, J):
#         J["MAC", "ac|geom|wing|root_chord"] = (
#             (2 / 3)
#             * (1 + inputs["ac|geom|wing|taper"] + inputs["ac|geom|wing|taper"] ** 2)
#             / (1 + inputs["ac|geom|wing|taper"])
#         )
#         J["MAC", "ac|geom|wing|taper"] = (2 / 3) * inputs["ac|geom|wing|root_chord"] * (
#             2 * inputs["ac|geom|wing|taper"] + 1
#         ) / (inputs["ac|geom|wing|taper"] + 1) - (2 / 3) * inputs["ac|geom|wing|root_chord"] * (
#             1 + inputs["ac|geom|wing|taper"] + inputs["ac|geom|wing|taper"] ** 2
#         ) / (
#             1 + inputs["ac|geom|wing|taper"]
#         ) ** 2


# class WingSpan(om.ExplicitComponent):
#     def setup(self):
#         self.add_input("ac|geom|wing|S_ref", units="m**2")
#         self.add_input("ac|geom|wing|AR")

#         self.add_output("span", units="m")
#         self.declare_partials(["span"], ["*"])

#     def compute(self, inputs, outputs):
#         b = inputs["ac|geom|wing|S_ref"] ** 0.5 * inputs["ac|geom|wing|AR"] ** 0.5
#         outputs["span"] = b

#     def compute_partials(self, inputs, J):
#         J["span", "ac|geom|wing|S_ref"] = (
#             0.5 * inputs["ac|geom|wing|S_ref"] ** (0.5 - 1) * inputs["ac|geom|wing|AR"] ** 0.5
#         )
#         J["span", "ac|geom|wing|AR"] = (
#             inputs["ac|geom|wing|S_ref"] ** 0.5 * 0.5 * inputs["ac|geom|wing|AR"] ** (0.5 - 1)
#         )

