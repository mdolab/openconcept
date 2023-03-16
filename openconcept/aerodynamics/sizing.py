from __future__ import division
from multiprocessing.context import ForkContext
from matplotlib import units
import numpy as np
from openmdao.api import ExplicitComponent, IndepVarComp
from openmdao.api import Group
from openconcept.utilities.math import AddSubtractComp, ElementMultiplyDivideComp


class WettedArea_JetTransport(ExplicitComponent):
    def initialize(self):
        self.options.declare("c", default=0.0199, desc="Roskam regression constant c")
        self.options.declare("d", default=0.7531, desc="Roskam regression constant d")

    def setup(self):
        self.add_input("ac|weights|MTOW", units="lb", desc="Maximum rated takeoff weight")
        self.add_output("S_wet", units="ft**2", desc="Estimate of total wetted area Roskam Eqn 3.22")
        self.declare_partials(["S_wet"], ["*"])

    def compute(self, inputs, outputs):

        c = self.options["c"]
        d = self.options["d"]

        S_wet_roskam = 10 ** c * inputs["ac|weights|MTOW"] ** d
        outputs["S_wet"] = S_wet_roskam

    def compute_partials(self, inputs, J):
        c = self.options["c"]
        d = self.options["d"]
        J["S_wet", "ac|weights|MTOW"] = 10 ** c * d * inputs["ac|weights|MTOW"] ** (d - 1)


class WettedAreaNonWingGeom_JetTransport(ExplicitComponent):
    def setup(self):
        self.add_input("ac|geom|fuselage|S_wet", units="ft**2", desc="Maximum rated takeoff weight")
        self.add_input("ac|geom|hstab|S_ref", units="ft**2", desc="Horizontal stabilizer reference area")
        self.add_input("ac|geom|vstab|S_ref", units="ft**2", desc="Vertical stabilizer reference area")
        self.add_output("S_wet", units="ft**2", desc="Estimate of total wetted area Roskam Eqn 3.22")
        self.declare_partials(["S_wet"], ["*"])

    def compute(self, inputs, outputs):

        S_wet_geometric = (
            0.9 * inputs["ac|geom|fuselage|S_wet"]
            + 2 * inputs["ac|geom|vstab|S_ref"]
            + 2 * inputs["ac|geom|hstab|S_ref"]
        )
        outputs["S_wet"] = S_wet_geometric

    def compute_partials(self, inputs, J):
        J["S_wet", "ac|geom|fuselage|S_wet"] = 0.9
        J["S_wet", "ac|geom|hstab|S_ref"] = 2
        J["S_wet", "ac|geom|vstab|S_ref"] = 2


class SkinFriction_JetTransport(ExplicitComponent):
    def initialize(self):
        self.options.declare(
            "C_fe", default=0.0030, desc="Equivalent skin friction coefficient for civil transport, Roskam"
        )

    def setup(self):
        self.add_input("S_wet", units="ft**2", desc="Estimate of total wetted area")
        self.add_input("ac|geom|wing|S_ref", units="ft**2", desc="Total wing area")

        self.add_output("C_d0", desc="Parasitic drag coefficient estimate")
        self.declare_partials(["C_d0"], ["*"])

    def compute(self, inputs, outputs):
        C_fe = self.options["C_fe"]
        Cd0_roskam = (C_fe * inputs["S_wet"]) / inputs["ac|geom|wing|S_ref"]
        outputs["C_d0"] = Cd0_roskam

    def compute_partials(self, inputs, J):
        C_fe = self.options["C_fe"]
        J["C_d0", "S_wet"] = C_fe / inputs["ac|geom|wing|S_ref"]
        J["C_d0", "ac|geom|wing|S_ref"] = -(C_fe * inputs["S_wet"]) / inputs["ac|geom|wing|S_ref"] ** 2


class SkinFrictionNonWing_JetTransport(ExplicitComponent):
    def initialize(self):
        self.options.declare(
            "C_fe", default=0.0030, desc="Equivalent skin friction coefficient for civil transport, Roskam"
        )

    def setup(self):
        self.add_input("S_wet", units="ft**2", desc="Estimate of total wetted area")
        self.add_input("ac|geom|wing|S_ref", units="ft**2", desc="Total wing area")

        self.add_output("C_d0", desc="Parasitic drag coefficient estimate")
        self.declare_partials(["C_d0"], ["*"])

    def compute(self, inputs, outputs):
        C_fe = self.options["C_fe"]
        Cd0_roskam = (C_fe * inputs["S_wet"]) / inputs["ac|geom|wing|S_ref"] #- C_fe
        outputs["C_d0"] = Cd0_roskam

    def compute_partials(self, inputs, J):
        C_fe = self.options["C_fe"]
        J["C_d0", "S_wet"] = C_fe / inputs["ac|geom|wing|S_ref"]
        J["C_d0", "ac|geom|wing|S_ref"] = -(C_fe * inputs["S_wet"]) / inputs["ac|geom|wing|S_ref"] ** 2


class CleanParasiticDrag_JetTransport(Group):
    def setup(self):
        self.add_subsystem(
            "total_wetted_area", WettedArea_JetTransport(), promotes_inputs=["*"], promotes_outputs=["*"]
        )
        self.add_subsystem("cd0", SkinFriction_JetTransport(), promotes_inputs=["*"], promotes_outputs=["*"])


class Cd0_NonWing_JetTransport(Group):
    def setup(self):
        self.add_subsystem(
            "nonwing_wetted_area", WettedAreaNonWingGeom_JetTransport(), promotes_inputs=["*"], promotes_outputs=["*"]
        )
        self.add_subsystem(
            "nonwing_cd0", SkinFrictionNonWing_JetTransport(), promotes_inputs=["*"], promotes_outputs=["*"]
        )


class CL_MAX_cruise(ExplicitComponent):
    def setup(self):
        self.add_input("ac|aero|Cl_max")
        self.add_input("ac|geom|wing|c4sweep", units="rad")

        self.add_output("Wing_CL_max")
        self.declare_partials(["Wing_CL_max"], ["*"])

    def compute(self, inputs, outputs):
        cLmax = 0.9 * inputs["ac|aero|Cl_max"] * np.cos(inputs["ac|geom|wing|c4sweep"])
        outputs["Wing_CL_max"] = cLmax

    def compute_partials(self, inputs, J):
        J["Wing_CL_max", "ac|aero|Cl_max"] = 0.9 * np.cos(inputs["ac|geom|wing|c4sweep"])
        J["Wing_CL_max", "ac|geom|wing|c4sweep"] = (
            -0.9 * inputs["ac|aero|Cl_max"] * np.sin(inputs["ac|geom|wing|c4sweep"])
        )


if __name__ == "__main__":
    from openmdao.api import IndepVarComp, Problem

    prob = Problem()
    prob.model = Group()
    dvs = prob.model.add_subsystem("dvs", IndepVarComp(), promotes_outputs=["*"])
    dvs.add_output("ac|weights|MTOW", 79002, units="kg")
    dvs.add_output("ac|geom|wing|S_ref", 124.6, units="m**2")
    dvs.add_output("ac|geom|hstab|S_ref", 32.78, units="m**2")
    dvs.add_output("ac|geom|vstab|S_ref", 26.44, units="m**2")
    dvs.add_output("ac|geom|fuselage|S_wet", 859, units="m**2")

    prob.model.add_subsystem("clean_skin_drag", Cd0_NonWing_JetTransport(), promotes_inputs=["*"])

    prob.setup(force_alloc_complex=True)
    prob.run_model()
    print("Nonwing wetted area:")
    print(prob["clean_skin_drag.S_wet"])
    print("Cd0 Estimate:")
    print(prob["clean_skin_drag.C_d0"])

    data = prob.check_partials(method="cs", compact_print=True, show_only_incorrect=False)