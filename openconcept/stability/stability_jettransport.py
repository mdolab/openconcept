from __future__ import division
from matplotlib import units
import numpy as np
from openmdao.api import ExplicitComponent, IndepVarComp
from openmdao.api import Group
from openconcept.utilities.math import AddSubtractComp, ElementMultiplyDivideComp
import math


class HStabSizing_JetTransport(ExplicitComponent):
    """Inputs: ac|geom|wing|S_ref, ac|geom|wing|MAC, ac|geom|hstab|c4_to_wing_c4
    Outputs: hstab_area
    Metadata: C_ht, (volume coefficients for single engine general aviation aircraft)

    """

    def initialize(self):
        self.options.declare(
            "C_ht",
            default=1.0,
            desc="Tail colume coefficient for vertical stabilizer, nondimensionless, single engine airplanes",
        )

    def setup(self):
        self.add_input("ac|geom|wing|S_ref", units="ft**2", desc="Reference wing area in sq ft")
        self.add_input("ac|geom|wing|MAC", units="ft", desc="Wing mean aerodynamic chord")
        self.add_input(
            "ac|geom|hstab|c4_to_wing_c4",
            units="ft",
            desc="Distance from wing c/4 to horiz stab c/4 (tail arm distance)",
        )

        self.add_output("hstab_area", units="ft**2")
        self.declare_partials(["hstab_area"], ["*"])

    def compute(self, inputs, outputs):
        C_ht = self.options["C_ht"]
        hstabarea = (C_ht * inputs["ac|geom|wing|MAC"] * inputs["ac|geom|wing|S_ref"]) / inputs[
            "ac|geom|hstab|c4_to_wing_c4"
        ]
        outputs["hstab_area"] = hstabarea

    def compute_partials(self, inputs, J):
        C_ht = self.options["C_ht"]
        J["hstab_area", "ac|geom|wing|S_ref"] = (C_ht * inputs["ac|geom|wing|MAC"]) / inputs[
            "ac|geom|hstab|c4_to_wing_c4"
        ]
        J["hstab_area", "ac|geom|wing|MAC"] = (C_ht * inputs["ac|geom|wing|S_ref"]) / inputs[
            "ac|geom|hstab|c4_to_wing_c4"
        ]
        J["hstab_area", "ac|geom|hstab|c4_to_wing_c4"] = (
            -C_ht * inputs["ac|geom|wing|MAC"] * inputs["ac|geom|wing|S_ref"]
        ) / (inputs["ac|geom|hstab|c4_to_wing_c4"] ** 2)


class VStabSizing_JetTransport(ExplicitComponent):
    """Inputs: ac|geom|wing|S_ref, ac|geom|wing|AR, ac|geom|hstab|c4_to_wing_c4
    Outputs: vstab_area
    Metadata: C_vt (volume coefficients for single engine general aviation aircraft)

    """

    def initialize(self):
        self.options.declare(
            "C_vt",
            default=0.090,
            desc="Tail colume coefficient for vertical stabilizer, nondimensionless, single engine airplanes",
        )

    def setup(self):
        self.add_input("ac|geom|wing|S_ref", units="ft**2", desc="Reference wing area in sq ft")
        self.add_input("ac|geom|wing|AR", desc="Wing aspect ratio")
        self.add_input(
            "ac|geom|hstab|c4_to_wing_c4",
            units="ft",
            desc="Distance from wing c/4 to horiz stab c/4 (tail arm distance)",
        )

        self.add_output("vstab_area", units="ft**2")
        self.declare_partials(["vstab_area"], ["*"])

    def compute(self, inputs, outputs):
        C_vt = self.options["C_vt"]
        vstabarea = (
            C_vt
            * np.sqrt(inputs["ac|geom|wing|AR"])
            * (inputs["ac|geom|wing|S_ref"] ** (1.5))
            / inputs["ac|geom|hstab|c4_to_wing_c4"]
        )
        outputs["vstab_area"] = vstabarea

    def compute_partials(self, inputs, J):
        C_vt = self.options["C_vt"]
        J["vstab_area", "ac|geom|wing|S_ref"] = (
            1.5
            * (C_vt * np.sqrt(inputs["ac|geom|wing|AR"] * inputs["ac|geom|wing|S_ref"]))
            / inputs["ac|geom|hstab|c4_to_wing_c4"]
        )
        J["vstab_area", "ac|geom|wing|AR"] = 0.5 * (
            C_vt
            * (inputs["ac|geom|wing|AR"] ** (-0.5))
            * (inputs["ac|geom|wing|S_ref"] ** (1.5))
            / inputs["ac|geom|hstab|c4_to_wing_c4"]
        )
        J["vstab_area", "ac|geom|hstab|c4_to_wing_c4"] = (
            -C_vt * np.sqrt(inputs["ac|geom|wing|AR"]) * inputs["ac|geom|wing|S_ref"] ** (1.5)
        ) / (inputs["ac|geom|hstab|c4_to_wing_c4"] ** 2)


