from __future__ import division
from matplotlib import units
import numpy as np
from openmdao.api import ExplicitComponent, IndepVarComp
from openmdao.api import Group
from openconcept.utilities.math import AddSubtractComp, ElementMultiplyDivideComp
import math


class WingRoot_LinearTaper(ExplicitComponent):
    """Inputs: ac|geom|wing|S_ref, ac|geom|wing|AR, ac|geom|wing|taper
    Outputs: C_root
    """

    def setup(self):
        self.add_input("ac|geom|wing|S_ref", units="ft**2", desc="Reference wing area in sq ft")
        self.add_input("ac|geom|wing|AR", desc="Wing Aspect Ratio")
        self.add_input("ac|geom|wing|taper", desc="Main Wing Taper Ratio")
        self.add_output("C_root", units="ft")
        self.declare_partials(["C_root"], ["*"])

    def compute(self, inputs, outputs):
        root_chord = (
            2
            * inputs["ac|geom|wing|S_ref"]
            / (np.sqrt(inputs["ac|geom|wing|S_ref"] * inputs["ac|geom|wing|AR"]) * (1 + inputs["ac|geom|wing|taper"]))
        )
        outputs["C_root"] = root_chord

    def compute_partials(self, inputs, J):
        J["C_root", "ac|geom|wing|S_ref"] = 2 / (
            (1 + inputs["ac|geom|wing|taper"]) * np.sqrt(inputs["ac|geom|wing|AR"] * inputs["ac|geom|wing|S_ref"])
        ) - (inputs["ac|geom|wing|AR"] * inputs["ac|geom|wing|S_ref"]) / (
            (1 + inputs["ac|geom|wing|taper"]) * (inputs["ac|geom|wing|AR"] * inputs["ac|geom|wing|S_ref"]) ** 1.5
        )
        J["C_root", "ac|geom|wing|AR"] = -inputs["ac|geom|wing|S_ref"] ** 2 / (
            (inputs["ac|geom|wing|taper"] + 1) * (inputs["ac|geom|wing|AR"] * inputs["ac|geom|wing|S_ref"]) ** 1.5
        )
        J["C_root", "ac|geom|wing|taper"] = (
            -2
            * inputs["ac|geom|wing|S_ref"]
            / (
                (inputs["ac|geom|wing|taper"] + 1) ** 2
                * (inputs["ac|geom|wing|AR"] * inputs["ac|geom|wing|S_ref"]) ** 0.5
            )
        )

class WingMAC_Trapezoidal(ExplicitComponent):
    """Inputs: ac|geom|wing|root_chord, ac|geom|wing|taper
    Outputs: MAC"""

    def setup(self):
        self.add_input("ac|geom|wing|root_chord", units="ft", desc="Main wing root chord")
        self.add_input("ac|geom|wing|taper", desc="Main wing taper ratio")
        self.add_output("MAC", units="ft")
        self.declare_partials(["MAC"], ["*"])

    def compute(self, inputs, outputs):
        meanAeroChord = (
            (2 / 3)
            * inputs["ac|geom|wing|root_chord"]
            * (1 + inputs["ac|geom|wing|taper"] + inputs["ac|geom|wing|taper"] ** 2)
            / (1 + inputs["ac|geom|wing|taper"])
        )
        outputs["MAC"] = meanAeroChord

    def compute_partials(self, inputs, J):
        J["MAC", "ac|geom|wing|root_chord"] = (
            (2 / 3)
            * (1 + inputs["ac|geom|wing|taper"] + inputs["ac|geom|wing|taper"] ** 2)
            / (1 + inputs["ac|geom|wing|taper"])
        )
        J["MAC", "ac|geom|wing|taper"] = (2 / 3) * inputs["ac|geom|wing|root_chord"] * (
            2 * inputs["ac|geom|wing|taper"] + 1
        ) / (inputs["ac|geom|wing|taper"] + 1) - (2 / 3) * inputs["ac|geom|wing|root_chord"] * (
            1 + inputs["ac|geom|wing|taper"] + inputs["ac|geom|wing|taper"] ** 2
        ) / (
            1 + inputs["ac|geom|wing|taper"]
        ) ** 2


class WingSpan(ExplicitComponent):
    def setup(self):
        self.add_input("ac|geom|wing|S_ref", units="m**2")
        self.add_input("ac|geom|wing|AR")

        self.add_output("span", units="m")
        self.declare_partials(["span"], ["*"])

    def compute(self, inputs, outputs):
        b = inputs["ac|geom|wing|S_ref"] ** 0.5 * inputs["ac|geom|wing|AR"] ** 0.5
        outputs["span"] = b

    def compute_partials(self, inputs, J):
        J["span", "ac|geom|wing|S_ref"] = (
            0.5 * inputs["ac|geom|wing|S_ref"] ** (0.5 - 1) * inputs["ac|geom|wing|AR"] ** 0.5
        )
        J["span", "ac|geom|wing|AR"] = (
            inputs["ac|geom|wing|S_ref"] ** 0.5 * 0.5 * inputs["ac|geom|wing|AR"] ** (0.5 - 1)
        )

