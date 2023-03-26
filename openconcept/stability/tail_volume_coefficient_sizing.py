import numpy as np
import openmdao.api as om


class HStabVolumeCoefficientSizing(om.ExplicitComponent):
    """
    Computes horizontal stabilizer reference area using tail volume coefficient
    method from Raymer (see Equation 6.27 in Section 6.4 of 1992 edition).

    Inputs
    ------
    ac|geom|wing|S_ref : float
        Wing planform area (scalar, sq ft)
    ac|geom|wing|MAC : float
        Wing mean aerodynamic chord (scalar, ft)
    ac|geom|hstab|c4_to_wing_c4 : float
        Distance from the horizontal stabilizer's quarter chord (of the MAC) to the wing's quarter chord (scalar, ft)

    Outputs
    -------
    ac|geom|hstab|S_ref : float
        Horizontal stabilizer reference area (scalar, sq ft)

    Options
    -------
    C_ht : float
        Tail volume coefficient for horizontal stabilizer, by default 1.00 from Table 6.4 in Raymer 1992
        for jet transport aircraft. See the table for other values (twin turboprop is 0.9).
    """

    def initialize(self):
        self.options.declare(
            "C_ht",
            default=1.0,
            desc="Horizontal tail volume coefficient",
        )

    def setup(self):
        self.add_input("ac|geom|wing|S_ref", units="ft**2", desc="Reference wing area in sq ft")
        self.add_input("ac|geom|wing|MAC", units="ft", desc="Wing mean aerodynamic chord")
        self.add_input(
            "ac|geom|hstab|c4_to_wing_c4",
            units="ft",
            desc="Distance from wing c/4 to horiz stab c/4 (tail arm distance)",
        )

        self.add_output("ac|geom|hstab|S_ref", units="ft**2")
        self.declare_partials(["ac|geom|hstab|S_ref"], ["*"])

    def compute(self, inputs, outputs):
        C_ht = self.options["C_ht"]
        outputs["ac|geom|hstab|S_ref"] = (C_ht * inputs["ac|geom|wing|MAC"] * inputs["ac|geom|wing|S_ref"]) / inputs[
            "ac|geom|hstab|c4_to_wing_c4"
        ]

    def compute_partials(self, inputs, J):
        C_ht = self.options["C_ht"]
        J["ac|geom|hstab|S_ref", "ac|geom|wing|S_ref"] = (C_ht * inputs["ac|geom|wing|MAC"]) / inputs[
            "ac|geom|hstab|c4_to_wing_c4"
        ]
        J["ac|geom|hstab|S_ref", "ac|geom|wing|MAC"] = (C_ht * inputs["ac|geom|wing|S_ref"]) / inputs[
            "ac|geom|hstab|c4_to_wing_c4"
        ]
        J["ac|geom|hstab|S_ref", "ac|geom|hstab|c4_to_wing_c4"] = (
            -C_ht * inputs["ac|geom|wing|MAC"] * inputs["ac|geom|wing|S_ref"]
        ) / (inputs["ac|geom|hstab|c4_to_wing_c4"] ** 2)


class VStabVolumeCoefficientSizing(om.ExplicitComponent):
    """
    Computes vertical stabilizer reference area using tail volume coefficient
    method from Raymer (see Equation 6.26 in Section 6.4 of 1992 edition).

    Inputs
    ------
    ac|geom|wing|S_ref : float
        Wing planform area (scalar, sq ft)
    ac|geom|wing|AR : float
        Wing aspect ratio (scalar, dimensionless)
    ac|geom|vstab|c4_to_wing_c4 : float
        Distance from the vertical stabilizer's quarter chord (of the MAC) to the wing's quarter chord (scalar, ft)

    Outputs
    -------
    ac|geom|vstab|S_ref : float
        Vertical stabilizer reference area (scalar, sq ft)

    Options
    -------
    C_vt : float
        Tail volume coefficient for vertical stabilizer, by default 0.09 from Table 6.4 in Raymer 1992
        for jet transport aircraft. See the table for other values (twin turboprop is 0.08).
    """

    def initialize(self):
        self.options.declare(
            "C_vt",
            default=0.09,
            desc="Vertical tail volume coefficient",
        )

    def setup(self):
        self.add_input("ac|geom|wing|S_ref", units="ft**2", desc="Reference wing area in sq ft")
        self.add_input("ac|geom|wing|AR", desc="Wing aspect ratio")
        self.add_input(
            "ac|geom|vstab|c4_to_wing_c4",
            units="ft",
            desc="Distance from wing c/4 to vertical stab c/4 (tail arm distance)",
        )

        self.add_output("ac|geom|vstab|S_ref", units="ft**2")
        self.declare_partials(["ac|geom|vstab|S_ref"], ["*"])

    def compute(self, inputs, outputs):
        C_vt = self.options["C_vt"]
        outputs["ac|geom|vstab|S_ref"] = (
            C_vt
            * np.sqrt(inputs["ac|geom|wing|AR"])
            * (inputs["ac|geom|wing|S_ref"] ** (1.5))
            / inputs["ac|geom|vstab|c4_to_wing_c4"]
        )

    def compute_partials(self, inputs, J):
        C_vt = self.options["C_vt"]
        J["ac|geom|vstab|S_ref", "ac|geom|wing|S_ref"] = (
            1.5
            * (C_vt * np.sqrt(inputs["ac|geom|wing|AR"] * inputs["ac|geom|wing|S_ref"]))
            / inputs["ac|geom|vstab|c4_to_wing_c4"]
        )
        J["ac|geom|vstab|S_ref", "ac|geom|wing|AR"] = 0.5 * (
            C_vt
            * (inputs["ac|geom|wing|AR"] ** (-0.5))
            * (inputs["ac|geom|wing|S_ref"] ** (1.5))
            / inputs["ac|geom|vstab|c4_to_wing_c4"]
        )
        J["ac|geom|vstab|S_ref", "ac|geom|vstab|c4_to_wing_c4"] = (
            -C_vt * np.sqrt(inputs["ac|geom|wing|AR"]) * inputs["ac|geom|wing|S_ref"] ** (1.5)
        ) / (inputs["ac|geom|vstab|c4_to_wing_c4"] ** 2)
