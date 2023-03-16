from __future__ import division
from multiprocessing.context import ForkContext
from xml.dom.minidom import Element
from matplotlib import units
import numpy as np
from openmdao.api import ExplicitComponent, IndepVarComp
from openmdao.api import Group
from openconcept.utilities.math import AddSubtractComp, ElementMultiplyDivideComp


class WingWeight_JetTrasport(ExplicitComponent):
    """Inputs: MTOW, ac|geom|wing|S_ref, ac|geom|wing|AR, ac|geom|wing|c4sweep, ac|geom|wing|taper, ac|geom|wing|toverc, V_H (max SL speed)
    Outputs: W_wing
    Metadata: n_ult (ult load factor)

    """

    def initialize(self):
        # self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')
        # define configuration parameters
        self.options.declare("n_ult", default=3.8 * 1.5, desc="Ultimate load factor (dimensionless)")

    def setup(self):
        # nn = self.options['num_nodes']
        self.add_input("ac|weights|MTOW", units="lb", desc="Maximum rated takeoff weight")
        self.add_input("ac|geom|wing|S_ref", units="ft**2", desc="Reference wing area in sq ft")
        self.add_input("ac|geom|wing|AR", desc="Wing aspect ratio")
        self.add_input("ac|geom|wing|c4sweep", units="rad", desc="Quarter-chord sweep angle")
        self.add_input("ac|geom|wing|taper", desc="Wing taper ratio")
        self.add_input("ac|geom|wing|toverc", desc="Wing max thickness to chord ratio")

        self.add_output("W_wing", units="lb", desc="Wing weight")
        self.declare_partials(["W_wing"], ["*"])

    def compute(self, inputs, outputs):
        n_ult = self.options["n_ult"]
        W_wing_Raymer = (
            0.0051
            * (inputs["ac|weights|MTOW"] * n_ult) ** 0.557
            * (inputs["ac|geom|wing|S_ref"]) ** (0.649 + 0.1)
            * (inputs["ac|geom|wing|AR"]) ** 0.5
            * (inputs["ac|geom|wing|toverc"]) ** -0.4
            * (1 + inputs["ac|geom|wing|taper"]) ** 0.1
            * np.cos(inputs["ac|geom|wing|c4sweep"]) ** -1
            * (0.12 ** 0.1)
        )  # *(inputs['ac|geom|wing|S_ref'])**0.1

        outputs["W_wing"] = W_wing_Raymer

    def compute_partials(self, inputs, J):  # TO DO
        n_ult = self.options["n_ult"]
        # J['W_wing','ac|weights|MTOW'] = (0.0023*inputs['ac|geom|wing|AR']**(1/2)*n_ult*inputs['ac|geom|wing|S_ref']**0.7490*(inputs['ac|geom|wing|taper'] + 1)**0.1000)/(inputs['ac|geom|wing|toverc']**0.4000*np.cos(inputs['ac|geom|wing|c4sweep'])*(inputs['ac|weights|MTOW']*n_ult)**0.4430)
        # J['W_wing','ac|weights|MTOW'] = (0.0023*inputs['ac|geom|wing|AR']**(1/2)*n_ult*inputs['ac|geom|wing|S_ref']**0.7490*(inputs['ac|geom|wing|taper'] + 1)**0.1000)/(inputs['ac|geom|wing|toverc']**0.4000*np.cos(inputs['ac|geom|wing|c4sweep'])*(inputs['ac|weights|MTOW']*n_ult)**0.4430)
        J["W_wing", "ac|weights|MTOW"] = (
            (0.0051 * 0.557)
            * (inputs["ac|weights|MTOW"] ** (0.557 - 1))
            * n_ult ** 0.557
            * (inputs["ac|geom|wing|S_ref"]) ** 0.649
            * (inputs["ac|geom|wing|AR"]) ** 0.5
            * (inputs["ac|geom|wing|toverc"]) ** -0.4
            * (1 + inputs["ac|geom|wing|taper"]) ** 0.1
            * np.cos(inputs["ac|geom|wing|c4sweep"]) ** -1
            * (0.12 * inputs["ac|geom|wing|S_ref"]) ** 0.1
        )
        J["W_wing", "ac|geom|wing|S_ref"] = (
            0.0051
            * (inputs["ac|weights|MTOW"] * n_ult) ** 0.557
            * (0.649 + 0.1)
            * (inputs["ac|geom|wing|S_ref"]) ** (0.649 + 0.1 - 1)
            * (inputs["ac|geom|wing|AR"]) ** 0.5
            * (inputs["ac|geom|wing|toverc"]) ** -0.4
            * (1 + inputs["ac|geom|wing|taper"]) ** 0.1
            * np.cos(inputs["ac|geom|wing|c4sweep"]) ** -1
            * (0.12 ** 0.1)
        )
        J["W_wing", "ac|geom|wing|AR"] = (
            0.0051
            * (inputs["ac|weights|MTOW"] * n_ult) ** 0.557
            * (inputs["ac|geom|wing|S_ref"]) ** 0.649
            * 0.5
            * (inputs["ac|geom|wing|AR"]) ** (0.5 - 1)
            * (inputs["ac|geom|wing|toverc"]) ** -0.4
            * (1 + inputs["ac|geom|wing|taper"]) ** 0.1
            * np.cos(inputs["ac|geom|wing|c4sweep"]) ** -1
            * (0.12 ** 0.1)
            * (inputs["ac|geom|wing|S_ref"]) ** 0.1
        )
        J["W_wing", "ac|geom|wing|c4sweep"] = (
            0.0051
            * (inputs["ac|weights|MTOW"] * n_ult) ** 0.557
            * (inputs["ac|geom|wing|S_ref"]) ** (0.649 + 0.1)
            * (inputs["ac|geom|wing|AR"]) ** 0.5
            * (inputs["ac|geom|wing|toverc"]) ** -0.4
            * (1 + inputs["ac|geom|wing|taper"]) ** 0.1
            * -1
            * np.cos(inputs["ac|geom|wing|c4sweep"]) ** -2
            * (-1 * np.sin(inputs["ac|geom|wing|c4sweep"]))
            * (0.12 ** 0.1)
        )
        J["W_wing", "ac|geom|wing|taper"] = (
            0.0051
            * (inputs["ac|weights|MTOW"] * n_ult) ** 0.557
            * (inputs["ac|geom|wing|S_ref"]) ** 0.649
            * (inputs["ac|geom|wing|AR"]) ** 0.5
            * (inputs["ac|geom|wing|toverc"]) ** -0.4
            * 0.1
            * (1 + inputs["ac|geom|wing|taper"]) ** (0.1 - 1)
            * np.cos(inputs["ac|geom|wing|c4sweep"]) ** -1
            * (0.12 ** 0.1)
            * (inputs["ac|geom|wing|S_ref"]) ** 0.1
        )
        J["W_wing", "ac|geom|wing|toverc"] = (
            0.0051
            * (inputs["ac|weights|MTOW"] * n_ult) ** 0.557
            * (inputs["ac|geom|wing|S_ref"]) ** 0.649
            * (inputs["ac|geom|wing|AR"]) ** 0.5
            * -0.4
            * (inputs["ac|geom|wing|toverc"]) ** (-0.4 - 1)
            * (1 + inputs["ac|geom|wing|taper"]) ** 0.1
            * np.cos(inputs["ac|geom|wing|c4sweep"]) ** -1
            * (0.12 ** 0.1)
            * (inputs["ac|geom|wing|S_ref"]) ** 0.1
        )


class HstabConst_JetTransport(ExplicitComponent):
    def setup(self):
        self.add_input("ac|geom|hstab|S_ref", units="ft**2", desc="Horizontal stabizer reference area")
        self.add_input("ac|geom|hstab|AR", desc="horizontal stabilizer aspect ratio")
        self.add_output("HstabConst")
        self.declare_partials(["HstabConst"], ["*"])

    def compute(self, inputs, outputs):
        const = 1 + 0.6 * 13 * inputs["ac|geom|hstab|S_ref"] ** -0.5 * inputs["ac|geom|hstab|AR"] ** -0.5
        outputs["HstabConst"] = const

    def compute_partials(self, inputs, J):
        J["HstabConst", "ac|geom|hstab|S_ref"] = (
            0.6 * 13 * -0.5 * inputs["ac|geom|hstab|S_ref"] ** -1.5 * inputs["ac|geom|hstab|AR"] ** -0.5
        )
        J["HstabConst", "ac|geom|hstab|AR"] = (
            0.6 * 13 * -0.5 * inputs["ac|geom|hstab|S_ref"] ** -0.5 * inputs["ac|geom|hstab|AR"] ** -1.5
        )


class HstabWeight_JetTransport(ExplicitComponent):
    def initialize(self):
        self.options.declare("K_uht", default=1.143, desc="Scaling for all moving stabilizer")
        self.options.declare("n_ult", default=3.8 * 1.5, desc="Ultimate load factor (dimensionless)")

    def setup(self):
        self.add_input("ac|weights|MTOW", units="lb", desc="Maximum rated takeoff weight")
        self.add_input("ac|geom|hstab|S_ref", units="ft**2", desc="Reference wing area in sq ft")
        self.add_input("ac|geom|hstab|AR", desc="Wing aspect ratio")
        self.add_input("ac|geom|hstab|c4sweep", units="rad", desc="Quarter-chord sweep angle")
        self.add_input(
            "ac|geom|hstab|c4_to_wing_c4", units="m", desc="Tail quarter-chord to wing quarter chord disnp.tance"
        )
        self.add_input("HstabConst", desc="Constant multiplier")

        self.add_output("W_hstab", units="lb", desc="Hstab weight")
        self.declare_partials(["W_hstab"], ["*"])

    def compute(self, inputs, outputs):
        n_ult = self.options["n_ult"]
        K_uht = self.options["K_uht"]

        W_hstab_raymer = (
            0.0379
            * K_uht
            * inputs["HstabConst"] ** -0.25
            * inputs["ac|weights|MTOW"] ** 0.639
            * n_ult ** 0.10
            * inputs["ac|geom|hstab|S_ref"] ** 0.75
            * (0.3 ** 0.704)
            * inputs["ac|geom|hstab|c4_to_wing_c4"] ** -0.296
            * np.cos(inputs["ac|geom|hstab|c4sweep"]) ** -1
            * inputs["ac|geom|hstab|AR"] ** 0.166
            * (1 + (0.2)) ** 0.1
        )
        outputs["W_hstab"] = W_hstab_raymer

    def compute_partials(self, inputs, J):
        n_ult = self.options["n_ult"]
        K_uht = self.options["K_uht"]

        # J['W_hstab','ac|weights|MTOW'] = (0.0247*K_uht*inputs['ac|geom|hstab|AR']**0.1660*n_ult**0.1000*inputs['ac|geom|hstab|S_ref']**0.7500*(0.3000*inputs['ac|geom|hstab|c4_to_wing_c4'])**0.7040)/(inputs['ac|geom|hstab|c4_to_wing_c4']**inputs['ac|weights|MTOW']**0.3610*np.cos(inputs['ac|geom|hstab|c4sweep'])**(7.8000/(inputs['ac|geom|hstab|AR']*inputs['ac|geom|hstab|S_ref'])**0.5000 + 1)**0.2500)
        # J['W_hstab','ac|weights|MTOW'] = (0.0247*K_uht*inputs['ac|geom|hstab|AR']**0.1660*n_ult**0.1000*inputs['ac|geom|hstab|S_ref']**0.7500*(0.3000*inputs['ac|geom|hstab|c4_to_wing_c4'])**0.7040)/(inputs['ac|geom|hstab|c4_to_wing_c4']**0.3610*np.cos(inputs['ac|geom|hstab|c4sweep'])*(7.8000/('ac|geom|hstab|AR'*inputs['ac|geom|hstab|S_ref'])**0.5000+1)**0.2500)
        J["W_hstab", "ac|weights|MTOW"] = (
            0.0379
            * K_uht
            * inputs["HstabConst"] ** -0.25
            * 0.639
            * inputs["ac|weights|MTOW"] ** (0.639 - 1)
            * n_ult ** 0.10
            * inputs["ac|geom|hstab|S_ref"] ** 0.75
            * inputs["ac|geom|hstab|c4_to_wing_c4"] ** -1
            * (0.3 * inputs["ac|geom|hstab|c4_to_wing_c4"]) ** 0.704
            * np.cos(inputs["ac|geom|hstab|c4sweep"]) ** -1
            * inputs["ac|geom|hstab|AR"] ** 0.166
            * (1 + (0.2)) ** 0.1
        )
        J["W_hstab", "ac|geom|hstab|S_ref"] = (
            0.0379
            * K_uht
            * inputs["HstabConst"] ** -0.25
            * inputs["ac|weights|MTOW"] ** 0.639
            * n_ult ** 0.10
            * 0.75
            * inputs["ac|geom|hstab|S_ref"] ** (0.75 - 1)
            * (0.3 ** 0.704)
            * inputs["ac|geom|hstab|c4_to_wing_c4"] ** -0.296
            * np.cos(inputs["ac|geom|hstab|c4sweep"]) ** -1
            * inputs["ac|geom|hstab|AR"] ** 0.166
            * (1 + (0.2)) ** 0.1
        )
        J["W_hstab", "ac|geom|hstab|AR"] = (
            0.0379
            * K_uht
            * inputs["HstabConst"] ** -0.25
            * inputs["ac|weights|MTOW"] ** 0.639
            * n_ult ** 0.10
            * inputs["ac|geom|hstab|S_ref"] ** 0.75
            * (0.3 ** 0.704)
            * inputs["ac|geom|hstab|c4_to_wing_c4"] ** -0.296
            * np.cos(inputs["ac|geom|hstab|c4sweep"]) ** -1
            * 0.166
            * inputs["ac|geom|hstab|AR"] ** (0.166 - 1)
            * (1 + (0.2)) ** 0.1
        )
        J["W_hstab", "ac|geom|hstab|c4sweep"] = (
            0.0379
            * K_uht
            * inputs["HstabConst"] ** -0.25
            * inputs["ac|weights|MTOW"] ** 0.639
            * n_ult ** 0.10
            * inputs["ac|geom|hstab|S_ref"] ** 0.75
            * inputs["ac|geom|hstab|c4_to_wing_c4"] ** -1
            * (0.3 * inputs["ac|geom|hstab|c4_to_wing_c4"]) ** 0.704
            * (np.sin(inputs["ac|geom|hstab|c4sweep"]))
            * np.cos(inputs["ac|geom|hstab|c4sweep"]) ** (-1 - 1)
            * inputs["ac|geom|hstab|AR"] ** 0.166
            * (1 + (0.2)) ** 0.1
        )
        J["W_hstab", "ac|geom|hstab|c4_to_wing_c4"] = (
            0.0379
            * K_uht
            * inputs["HstabConst"] ** -0.25
            * inputs["ac|weights|MTOW"] ** 0.639
            * n_ult ** 0.10
            * inputs["ac|geom|hstab|S_ref"] ** 0.75
            * (0.3 ** 0.704)
            * -0.296
            * inputs["ac|geom|hstab|c4_to_wing_c4"] ** (-0.296 - 1)
            * np.cos(inputs["ac|geom|hstab|c4sweep"]) ** -1
            * inputs["ac|geom|hstab|AR"] ** 0.166
            * (1 + (0.2)) ** 0.1
        )
        J["W_hstab", "HstabConst"] = (
            0.0379
            * K_uht
            * -0.25
            * inputs["HstabConst"] ** (-0.25 - 1)
            * inputs["ac|weights|MTOW"] ** 0.639
            * n_ult ** 0.10
            * inputs["ac|geom|hstab|S_ref"] ** 0.75
            * (0.3 ** 0.704)
            * inputs["ac|geom|hstab|c4_to_wing_c4"] ** -0.296
            * np.cos(inputs["ac|geom|hstab|c4sweep"]) ** -1
            * inputs["ac|geom|hstab|AR"] ** 0.166
            * (1 + (0.2)) ** 0.1
        )


class VstabWeight_JetTransport(ExplicitComponent):
    def initialize(self):
        self.options.declare("n_ult", default=3.8 * 1.5, desc="Ultimate load factor (dimensionless)")

    def setup(self):
        self.add_input("ac|weights|MTOW", units="lb", desc="Maximum rated takeoff weight")
        self.add_input("ac|geom|vstab|S_ref", units="ft**2", desc="Reference vtail area in sq ft")
        self.add_input("ac|geom|vstab|AR", desc="vtail aspect ratio")
        self.add_input("ac|geom|vstab|c4sweep", units="rad", desc="Quarter-chord sweep angle")
        self.add_input(
            "ac|geom|hstab|c4_to_wing_c4", units="m", desc="Tail quarter-chord to wing quarter chord disnp.tance"
        )
        self.add_input("ac|geom|vstab|toverc", desc="root t/c of v-tail, estimated same as wing")

        self.add_output("W_vstab", units="lb", desc="Vstab weight")
        self.declare_partials(["W_vstab"], ["*"])

    def compute(self, inputs, outputs):
        n_ult = self.options["n_ult"]

        W_vstab_raymer = (
            0.0026
            * inputs["ac|weights|MTOW"] ** 0.556
            * n_ult ** 0.536
            * inputs["ac|geom|hstab|c4_to_wing_c4"] ** (-0.5 + 0.875)
            * inputs["ac|geom|vstab|S_ref"] ** 0.5
            * np.cos(inputs["ac|geom|vstab|c4sweep"]) ** -1
            * inputs["ac|geom|vstab|AR"] ** 0.35
            * inputs["ac|geom|vstab|toverc"] ** -0.5
        )
        outputs["W_vstab"] = W_vstab_raymer

    def compute_partials(self, inputs, J):
        n_ult = self.options["n_ult"]

        J["W_vstab", "ac|weights|MTOW"] = (
            0.0026
            * 0.556
            * inputs["ac|weights|MTOW"] ** (0.556 - 1)
            * n_ult ** 0.536
            * inputs["ac|geom|hstab|c4_to_wing_c4"] ** (-0.5 + 0.875)
            * inputs["ac|geom|vstab|S_ref"] ** 0.5
            * np.cos(inputs["ac|geom|vstab|c4sweep"]) ** -1
            * inputs["ac|geom|vstab|AR"] ** 0.35
            * inputs["ac|geom|vstab|toverc"] ** -0.5
        )
        J["W_vstab", "ac|geom|vstab|S_ref"] = (
            0.0013
            * inputs["ac|geom|vstab|AR"] ** 0.3500
            * inputs["ac|geom|hstab|c4_to_wing_c4"] ** 0.3750
            * inputs["ac|weights|MTOW"] ** 0.5560
            * n_ult ** 0.5360
        ) / (
            inputs["ac|geom|vstab|S_ref"] ** 0.5000
            * inputs["ac|geom|vstab|toverc"] ** 0.5000
            * np.cos(inputs["ac|geom|vstab|c4sweep"])
        )
        J["W_vstab", "ac|geom|vstab|AR"] = (
            9.1000e-04
            * inputs["ac|geom|hstab|c4_to_wing_c4"] ** 0.3750
            * inputs["ac|weights|MTOW"] ** 0.5560
            * n_ult ** 0.5360
            * inputs["ac|geom|vstab|S_ref"] ** (1 / 2)
        ) / (
            inputs["ac|geom|vstab|AR"] ** 0.6500
            * inputs["ac|geom|vstab|toverc"] ** 0.5000
            * np.cos(inputs["ac|geom|vstab|c4sweep"])
        )
        J["W_vstab", "ac|geom|vstab|c4sweep"] = -(
            0.0026
            * inputs["ac|geom|vstab|AR"] ** 0.3500
            * inputs["ac|geom|hstab|c4_to_wing_c4"] ** 0.3750
            * inputs["ac|weights|MTOW"] ** 0.5560
            * n_ult ** 0.5360
            * inputs["ac|geom|vstab|S_ref"] ** (1 / 2)
            * np.sin(inputs["ac|geom|vstab|c4sweep"])
        ) / (inputs["ac|geom|vstab|toverc"] ** 0.5000 * (np.sin(inputs["ac|geom|vstab|c4sweep"]) ** 2 - 1))
        J["W_vstab", "ac|geom|hstab|c4_to_wing_c4"] = (
            9.7500e-04
            * inputs["ac|geom|vstab|AR"] ** 0.3500
            * inputs["ac|weights|MTOW"] ** 0.5560
            * n_ult ** 0.5360
            * inputs["ac|geom|vstab|S_ref"] ** (1 / 2)
        ) / (
            inputs["ac|geom|hstab|c4_to_wing_c4"] ** 0.6250
            * inputs["ac|geom|vstab|toverc"] ** 0.5000
            * np.cos(inputs["ac|geom|vstab|c4sweep"])
        )
        J["W_vstab", "ac|geom|vstab|toverc"] = -(
            0.0013
            * inputs["ac|geom|vstab|AR"] ** 0.3500
            * inputs["ac|geom|hstab|c4_to_wing_c4"] ** 0.3750
            * inputs["ac|weights|MTOW"] ** 0.5560
            * n_ult ** 0.5360
            * inputs["ac|geom|vstab|S_ref"] ** (1 / 2)
        ) / (inputs["ac|geom|vstab|toverc"] ** 1.5000 * np.cos(inputs["ac|geom|vstab|c4sweep"]))


class FuselageConst1_JetTransport(ExplicitComponent):
    def setup(self):
        self.add_input("ac|geom|wing|taper")
        self.add_output("FuseConst1")
        self.declare_partials(["FuseConst1"], ["*"])

    def compute(self, inputs, outputs):
        const = 0.75 * 1 + 0.75 * (2 * inputs["ac|geom|wing|taper"]) / (1 + inputs["ac|geom|wing|taper"])
        outputs["FuseConst1"] = const

    def compute_partials(self, inputs, J):
        J["FuseConst1", "ac|geom|wing|taper"] = 0.75 * (2) / ((1 + inputs["ac|geom|wing|taper"]) ** 2)


class FuselageConst2_JetTransport(ExplicitComponent):
    def setup(self):
        self.add_input("FuseConst1", desc="Fuselage taper ratio constant")
        self.add_input("ac|geom|wing|S_ref", units="ft**2", desc="Wing reference area")
        self.add_input("ac|geom|wing|AR", desc="Wing aspect ratio")
        self.add_input("ac|geom|wing|c4sweep", units="rad", desc="Wing Aspect Ratio")
        self.add_input("ac|geom|fuselage|length", units="ft", desc="Fuselage structural length")
        self.add_output("K_ws", desc="Fuselage constant Kws defined in Raymer")
        self.declare_partials(["K_ws"], ["*"])

    def compute(self, inputs, outputs):
        Kws_raymer = (
            0.75
            * inputs["FuseConst1"]
            * inputs["ac|geom|wing|S_ref"] ** 0.5
            * inputs["ac|geom|wing|AR"] ** 0.5
            * np.tan(inputs["ac|geom|wing|c4sweep"])
            * inputs["ac|geom|fuselage|length"] ** -1
        )
        outputs["K_ws"] = Kws_raymer

    def compute_partials(self, inputs, J):
        J["K_ws", "FuseConst1"] = (
            0.75
            * inputs["ac|geom|wing|S_ref"] ** 0.5
            * inputs["ac|geom|wing|AR"] ** 0.5
            * np.tan(inputs["ac|geom|wing|c4sweep"])
            * inputs["ac|geom|fuselage|length"] ** -1
        )
        J["K_ws", "ac|geom|wing|S_ref"] = (
            0.75
            * inputs["FuseConst1"]
            * 0.5
            * inputs["ac|geom|wing|S_ref"] ** (0.5 - 1)
            * inputs["ac|geom|wing|AR"] ** 0.5
            * np.tan(inputs["ac|geom|wing|c4sweep"])
            * inputs["ac|geom|fuselage|length"] ** -1
        )
        J["K_ws", "ac|geom|wing|AR"] = (
            0.75
            * inputs["FuseConst1"]
            * inputs["ac|geom|wing|S_ref"] ** 0.5
            * 0.5
            * inputs["ac|geom|wing|AR"] ** (0.5 - 1)
            * np.tan(inputs["ac|geom|wing|c4sweep"])
            * inputs["ac|geom|fuselage|length"] ** -1
        )
        J["K_ws", "ac|geom|wing|c4sweep"] = (
            0.75
            * inputs["FuseConst1"]
            * inputs["ac|geom|wing|S_ref"] ** 0.5
            * inputs["ac|geom|wing|AR"] ** 0.5
            * (1 / np.cos(inputs["ac|geom|wing|c4sweep"])) ** 2
            * inputs["ac|geom|fuselage|length"] ** -1
        )
        J["K_ws", "ac|geom|fuselage|length"] = (
            0.75
            * inputs["FuseConst1"]
            * inputs["ac|geom|wing|S_ref"] ** 0.5
            * inputs["ac|geom|wing|AR"] ** 0.5
            * np.tan(inputs["ac|geom|wing|c4sweep"])
            * -1
            * inputs["ac|geom|fuselage|length"] ** (-1 - 1)
        )


class FuselageWeight_JetTransport(ExplicitComponent):
    def initialize(self):
        self.options.declare("n_ult", default=3.8 * 1.5, desc="Ultimate load factor (dimensionless)")
        self.options.declare("k_door", default=1, desc="Ultimate load factor (dimensionless)")
        self.options.declare("k_lg", default=1.12, desc="Ultimate load factor (dimensionless)")

    def setup(self):
        self.add_input("ac|weights|MTOW", units="lb", desc="Maximum rated takeoff weight")
        self.add_input("ac|geom|fuselage|length", units="ft", desc="fuselage structural length")
        self.add_input("ac|geom|fuselage|S_wet", units="ft**2", desc="fuselage wetted area")
        self.add_input("ac|aero|LoverD", desc="Design L/D of aircraft")
        self.add_input("K_ws")

        self.add_output("W_fuselage", units="lb", desc="fuselage weight")
        self.declare_partials(["W_fuselage"], ["*"])

    def compute(self, inputs, outputs):
        n_ult = self.options["n_ult"]
        k_door = self.options["k_door"]
        k_lg = self.options["k_lg"]

        W_fuselage_raymer = (
            0.3280
            * k_door
            * k_lg
            * inputs["ac|weights|MTOW"] ** 0.5
            * n_ult ** 0.5
            * inputs["ac|geom|fuselage|length"] ** 0.25
            * inputs["ac|geom|fuselage|S_wet"] ** 0.302
            * (1 + inputs["K_ws"]) ** 0.04
            * inputs["ac|aero|LoverD"] ** 0.10
        )
        outputs["W_fuselage"] = W_fuselage_raymer

    def compute_partials(self, inputs, J):
        n_ult = self.options["n_ult"]
        k_door = self.options["k_door"]
        k_lg = self.options["k_lg"]

        J["W_fuselage", "ac|weights|MTOW"] = (
            0.3280
            * k_door
            * k_lg
            * 0.5
            * inputs["ac|weights|MTOW"] ** (0.5 - 1)
            * n_ult ** 0.5
            * inputs["ac|geom|fuselage|length"] ** 0.25
            * inputs["ac|geom|fuselage|S_wet"] ** 0.302
            * (1 + inputs["K_ws"]) ** 0.04
            * inputs["ac|aero|LoverD"] ** 0.10
        )
        J["W_fuselage", "ac|geom|fuselage|length"] = (
            0.3280
            * k_door
            * k_lg
            * inputs["ac|weights|MTOW"] ** 0.5
            * n_ult ** 0.5
            * 0.25
            * inputs["ac|geom|fuselage|length"] ** (0.25 - 1)
            * inputs["ac|geom|fuselage|S_wet"] ** 0.302
            * (1 + inputs["K_ws"]) ** 0.04
            * inputs["ac|aero|LoverD"] ** 0.10
        )
        J["W_fuselage", "ac|geom|fuselage|S_wet"] = (
            0.3280
            * k_door
            * k_lg
            * inputs["ac|weights|MTOW"] ** 0.5
            * n_ult ** 0.5
            * inputs["ac|geom|fuselage|length"] ** 0.25
            * 0.302
            * inputs["ac|geom|fuselage|S_wet"] ** (0.302 - 1)
            * (1 + inputs["K_ws"]) ** 0.04
            * inputs["ac|aero|LoverD"] ** 0.10
        )
        J["W_fuselage", "ac|aero|LoverD"] = (
            0.3280
            * k_door
            * k_lg
            * inputs["ac|weights|MTOW"] ** 0.5
            * n_ult ** 0.5
            * inputs["ac|geom|fuselage|length"] ** 0.25
            * inputs["ac|geom|fuselage|S_wet"] ** 0.302
            * (1 + inputs["K_ws"]) ** 0.04
            * 0.10
            * inputs["ac|aero|LoverD"] ** (0.10 - 1)
        )
        J["W_fuselage", "K_ws"] = (
            0.3280
            * k_door
            * k_lg
            * inputs["ac|weights|MTOW"] ** 0.5
            * n_ult ** 0.5
            * inputs["ac|geom|fuselage|length"] ** 0.25
            * inputs["ac|geom|fuselage|S_wet"] ** 0.302
            * 0.04
            * (1 + inputs["K_ws"]) ** (0.04 - 1)
            * inputs["ac|aero|LoverD"] ** 0.10
        )


# class FuselageWeight_JetTransport(ExplicitComponent):

#     def initialize(self):
#         self.options.declare('n_ult', default=3.8*1.5, desc='Ultimate load factor (dimensionless)')
#         self.options.declare('k_door', default=1, desc='Ultimate load factor (dimensionless)')
#         self.options.declare('k_lg', default=1.12, desc='Ultimate load factor (dimensionless)')

#     def setup(self):
#         self.add_input('ac|weights|MTOW', units='lb', desc='Maximum rated takeoff weight')
#         self.add_input('ac|geom|wing|S_ref', units='ft**2', desc='Reference wing area in sq ft')
#         self.add_input('ac|geom|wing|AR', desc='Wing aspect ratio')
#         self.add_input('ac|geom|wing|c4sweep', units='rad', desc='Quarter-chord sweep angle')
#         self.add_input('ac|geom|wing|taper', desc='Wing taper ratio')
#         self.add_input('ac|geom|fuselage|length', units='ft', desc='fuselage structural length')
#         self.add_input('ac|geom|fuselage|S_wet', units='ft**2', desc='fuselage wetted area')
#         self.add_input('ac|aero|LoverD', desc='Design L/D of aircraft')

#         self.add_output('W_fuselage', units='lb', desc='fuselage weight')
#         self.declare_partials(['W_fuselage'],['*'])

#     def compute(self, inputs, outputs):
#         n_ult = self.options['n_ult']
#         k_door = self.options['k_door']
#         k_lg = self.options['k_lg']

#         W_fuselage_raymer = 0.3280*k_door*k_lg*(inputs['ac|weights|MTOW']*n_ult)**0.5*inputs['ac|geom|fuselage|length']**0.25*inputs['ac|geom|fuselage|S_wet']**0.302*(1+0.75*(1+(2*inputs['ac|geom|wing|taper'])/(1+inputs['ac|geom|wing|taper']))*(inputs['ac|geom|wing|S_ref']*inputs['ac|geom|wing|AR'])**0.5*np.tan(inputs['ac|geom|wing|c4sweep']/inputs['ac|geom|fuselage|length']))**0.04*inputs['ac|aero|LoverD']**0.10
#         outputs['W_fuselage'] = W_fuselage_raymer

#     def compute_partials(self, inputs, J):
#         n_ult = self.options['n_ult']
#         k_door = self.options['k_door']
#         k_lg = self.options['k_lg']

#         J['W_fuselage','ac|weights|MTOW'] = 0.3280*k_door*k_lg*0.5*inputs['ac|weights|MTOW']**-0.5*n_ult**0.5*inputs['ac|geom|fuselage|length']**0.25*inputs['ac|geom|fuselage|S_wet']**0.302*(1+0.75*(1+(2*inputs['ac|geom|wing|taper'])/(1+inputs['ac|geom|wing|taper']))*(inputs['ac|geom|wing|S_ref']*inputs['ac|geom|wing|AR'])**0.5*np.tan(inputs['ac|geom|wing|c4sweep']/inputs['ac|geom|fuselage|length']))**0.04*inputs['ac|aero|LoverD']**0.10
#         J['W_fuselage','ac|geom|wing|S_ref'] = (0.0186*inputs['ac|geom|wing|AR']*k_door*k_lg*inputs['ac|geom|fuselage|length']**0.2500*inputs['ac|aero|LoverD']**0.1000*inputs['ac|geom|fuselage|S_wet']**0.3020*np.tan(inputs['ac|geom|wing|c4sweep']/inputs['ac|geom|fuselage|length'])*(3*inputs['ac|geom|wing|taper'] + 1)*(inputs['ac|weights|MTOW']*n_ult)**(1/2))/((inputs['ac|geom|wing|AR']*inputs['ac|geom|wing|S_ref'])**0.5000*((3*np.tan(inputs['ac|geom|wing|c4sweep']/inputs['ac|geom|fuselage|length'])*(3*inputs['ac|geom|wing|taper'] + 1)*(inputs['ac|geom|wing|AR']*inputs['ac|geom|wing|S_ref'])**(1/2))/(inputs['ac|geom|wing|taper'] + 1) + 4)**0.9600*(inputs['ac|geom|wing|taper'] + 1))
#         J['W_fuselage','ac|geom|wing|AR'] = (0.0186*k_door*k_lg*inputs['ac|geom|fuselage|length']**0.2500*inputs['ac|aero|LoverD']**0.1000*inputs['ac|geom|wing|S_ref']*inputs['ac|geom|fuselage|S_wet']**0.3020*np.tan(inputs['ac|geom|wing|c4sweep']/inputs['ac|geom|fuselage|length'])*(3*inputs['ac|geom|wing|taper'] + 1)*(inputs['ac|weights|MTOW']*n_ult)**(1/2))/((inputs['ac|geom|wing|AR']*inputs['ac|geom|wing|S_ref'])**0.5000*((3*np.tan(inputs['ac|geom|wing|c4sweep']/inputs['ac|geom|fuselage|length'])*(3*inputs['ac|geom|wing|taper'] + 1)*(inputs['ac|geom|wing|AR']*inputs['ac|geom|wing|S_ref'])**(1/2))/(inputs['ac|geom|wing|taper'] + 1) + 4)**0.9600*(inputs['ac|geom|wing|taper'] + 1))
#         J['W_fuselage','ac|geom|wing|c4sweep'] = (0.0372*k_door*k_lg*inputs['ac|aero|LoverD']**0.1000*inputs['ac|geom|fuselage|S_wet']**0.3020*(3*inputs['ac|geom|wing|taper'] + 1)*(inputs['ac|geom|wing|AR']*inputs['ac|geom|wing|S_ref'])**(1/2)*(inputs['ac|weights|MTOW']*n_ult)**(1/2)*(np.tan(inputs['ac|geom|wing|c4sweep']/inputs['ac|geom|fuselage|length'])**2 + 1))/(inputs['ac|geom|fuselage|length']**0.7500*((3*np.tan(inputs['ac|geom|wing|c4sweep']/inputs['ac|geom|fuselage|length'])*(3*inputs['ac|geom|wing|taper'] + 1)*(inputs['ac|geom|wing|AR']*inputs['ac|geom|wing|S_ref'])**(1/2))/(inputs['ac|geom|wing|taper'] + 1) + 4)**0.9600*(inputs['ac|geom|wing|taper'] + 1))
#         J['W_fuselage','ac|geom|wing|taper'] = (0.0745*k_door*k_lg*inputs['ac|geom|fuselage|length']**0.2500*inputs['ac|aero|LoverD']**0.1000*inputs['ac|geom|fuselage|S_wet']**0.3020*np.tan(inputs['ac|geom|wing|c4sweep']/inputs['ac|geom|fuselage|length'])*(inputs['ac|geom|wing|AR']*inputs['ac|geom|wing|S_ref'])**(1/2)*(inputs['ac|weights|MTOW']*n_ult)**(1/2))/(((3*np.tan(inputs['ac|geom|wing|c4sweep']/inputs['ac|geom|fuselage|length'])*(3*inputs['ac|geom|wing|taper'] + 1)*(inputs['ac|geom|wing|AR']*inputs['ac|geom|wing|S_ref'])**(1/2))/(inputs['ac|geom|wing|taper'] + 1) + 4)**0.9600*(inputs['ac|geom|wing|taper'] + 1)**2)
#         J['W_fuselage','ac|geom|fuselage|length'] = 0.3280*k_door*k_lg*(inputs['ac|weights|MTOW']*n_ult)**0.5*0.25*inputs['ac|geom|fuselage|length']**(0.25-1)*inputs['ac|geom|fuselage|S_wet']**0.302*(1+0.75*(1+(2*inputs['ac|geom|wing|taper'])/(1+inputs['ac|geom|wing|taper']))*(inputs['ac|geom|wing|S_ref']*inputs['ac|geom|wing|AR'])**0.5*np.tan(inputs['ac|geom|wing|c4sweep']/inputs['ac|geom|fuselage|length']))**0.04*inputs['ac|aero|LoverD']**0.10
#         J['W_fuselage','ac|geom|fuselage|S_wet'] = 0.3280*k_door*k_lg*(inputs['ac|weights|MTOW']*n_ult)**0.5*inputs['ac|geom|fuselage|length']**0.25*0.302*inputs['ac|geom|fuselage|S_wet']**(0.302-1)*(1+0.75*(1+(2*inputs['ac|geom|wing|taper'])/(1+inputs['ac|geom|wing|taper']))*(inputs['ac|geom|wing|S_ref']*inputs['ac|geom|wing|AR'])**0.5*np.tan(inputs['ac|geom|wing|c4sweep']/inputs['ac|geom|fuselage|length']))**0.04*inputs['ac|aero|LoverD']**0.10
#         J['W_fuselage','ac|aero|LoverD'] = 0.3280*k_door*k_lg*(inputs['ac|weights|MTOW']*n_ult)**0.5*inputs['ac|geom|fuselage|length']**0.25*inputs['ac|geom|fuselage|S_wet']**0.302*(1+0.75*(1+(2*inputs['ac|geom|wing|taper'])/(1+inputs['ac|geom|wing|taper']))*(inputs['ac|geom|wing|S_ref']*inputs['ac|geom|wing|AR'])**0.5*np.tan(inputs['ac|geom|wing|c4sweep']/inputs['ac|geom|fuselage|length']))**0.04*0.10*inputs['ac|aero|LoverD']**(0.10-1)


class MainLandingGear_JetTransport(ExplicitComponent):
    def initialize(self):
        self.options.declare("n_ult", default=3 * 1.5, desc="ultimate landing load factor, N_gear*1.5")

    def setup(self):
        self.add_input("ac|geom|maingear|length", units="inch", desc="main landing gear length")
        self.add_input("ac|weights|MLW", units="lb", desc="max landing weight")
        self.add_input("ac|geom|maingear|n_wheels", desc="numer of main landing gear wheels")
        self.add_input("ac|aero|Vstall_land", units="kn", desc="stall speed in max landing configuration")

        self.add_output("W_mlg", units="lb", desc="maingear weight")
        self.declare_partials(["W_mlg"], ["*"])

    def compute(self, inputs, outputs):
        n_ult = self.options["n_ult"]

        W_maingear_raymer = (
            0.0106
            * inputs["ac|weights|MLW"] ** 0.888
            * n_ult ** 0.25
            * inputs["ac|geom|maingear|length"] ** 0.4
            * inputs["ac|geom|maingear|n_wheels"] ** 0.321
            * inputs["ac|aero|Vstall_land"] ** 0.1
        )
        outputs["W_mlg"] = W_maingear_raymer

    def compute_partials(self, inputs, J):
        n_ult = self.options["n_ult"]
        J["W_mlg", "ac|weights|MLW"] = (
            0.0106
            * 0.888
            * inputs["ac|weights|MLW"] ** (0.888 - 1)
            * n_ult ** 0.25
            * inputs["ac|geom|maingear|length"] ** 0.4
            * inputs["ac|geom|maingear|n_wheels"] ** 0.321
            * inputs["ac|aero|Vstall_land"] ** 0.1
        )
        J["W_mlg", "ac|geom|maingear|length"] = (
            0.0106
            * inputs["ac|weights|MLW"] ** 0.888
            * n_ult ** 0.25
            * 0.4
            * inputs["ac|geom|maingear|length"] ** (0.4 - 1)
            * inputs["ac|geom|maingear|n_wheels"] ** 0.321
            * inputs["ac|aero|Vstall_land"] ** 0.1
        )
        J["W_mlg", "ac|geom|maingear|n_wheels"] = (
            0.0106
            * inputs["ac|weights|MLW"] ** 0.888
            * n_ult ** 0.25
            * inputs["ac|geom|maingear|length"] ** 0.4
            * 0.321
            * inputs["ac|geom|maingear|n_wheels"] ** (0.321 - 1)
            * inputs["ac|aero|Vstall_land"] ** 0.1
        )
        J["W_mlg", "ac|aero|Vstall_land"] = (
            0.0106
            * inputs["ac|weights|MLW"] ** 0.888
            * n_ult ** 0.25
            * inputs["ac|geom|maingear|length"] ** 0.4
            * inputs["ac|geom|maingear|n_wheels"] ** 0.321
            * 0.1
            * inputs["ac|aero|Vstall_land"] ** (0.1 - 1)
        )


class NoseLandingGear_JetTransport(ExplicitComponent):
    def initialize(self):
        self.options.declare("n_ult", default=3 * 1.5, desc="ultimate landing load factor, N_gear*1.5")

    def setup(self):
        self.add_input("ac|geom|nosegear|length", units="inch", desc="nose landing gear length")
        self.add_input("ac|weights|MLW", units="lb", desc="max landing weight")
        self.add_input("ac|geom|nosegear|n_wheels", desc="numer of nose landing gear wheels")

        self.add_output("W_nlg", units="lb", desc="nosegear weight")
        self.declare_partials(["W_nlg"], ["*"])

    def compute(self, inputs, outputs):
        n_ult = self.options["n_ult"]
        W_nosegear_raymer = (
            0.032
            * inputs["ac|weights|MLW"] ** 0.646
            * n_ult ** 0.2
            * inputs["ac|geom|nosegear|length"] ** 0.5
            * inputs["ac|geom|nosegear|n_wheels"] ** 0.45
        )
        outputs["W_nlg"] = W_nosegear_raymer

    def compute_partials(self, inputs, J):
        n_ult = self.options["n_ult"]
        J["W_nlg", "ac|weights|MLW"] = (
            0.032
            * 0.646
            * inputs["ac|weights|MLW"] ** (0.646 - 1)
            * n_ult ** 0.2
            * inputs["ac|geom|nosegear|length"] ** 0.5
            * inputs["ac|geom|nosegear|n_wheels"] ** 0.45
        )
        J["W_nlg", "ac|geom|nosegear|length"] = (
            0.0160
            * n_ult ** 0.2000
            * inputs["ac|geom|nosegear|n_wheels"] ** 0.4500
            * inputs["ac|weights|MLW"] ** 0.6460
        ) / inputs["ac|geom|nosegear|length"] ** 0.5000
        J["W_nlg", "ac|geom|nosegear|n_wheels"] = (
            0.0144 * n_ult ** 0.2000 * inputs["ac|weights|MLW"] ** 0.6460 * inputs["ac|geom|nosegear|length"] ** (1 / 2)
        ) / inputs["ac|geom|nosegear|n_wheels"] ** 0.5500


# class Engine_JetTransport(ExplicitComponent):
#     # Uses Raymer method
#     def initialize(self):
#         self.options.declare('n_ult', default=3*1.5, desc='ultimate landing load factor, N_gear*1.5')
#     def setup(self):
#         self.add_input('ac|propulsion|engine|rating', units='N', desc='Rated thrust per engine')
#         self.add_input('ac|propulsion|engine|BPR', desc='engine bypass ratio')
#         self.add_output('W_engines', units='kg')
#         self.declare_partials(['W_engines'],['*'])

#     def compute(self, inputs, outputs):
#         W_engine_raymer = 14.7*(inputs['ac|propulsion|engine|rating']/1000)**1.1*np.exp(-0.045*inputs['ac|propulsion|engine|BPR'])
#         # W_engine_raymer = 2370*2
#         outputs['W_engines'] = W_engine_raymer
#     def compute_partials(self, inputs, J):
#         J['W_engines', 'ac|propulsion|engine|rating'] = 0.0162*np.exp(-0.0450*inputs['ac|propulsion|engine|BPR'])*(0.0010*inputs['ac|propulsion|engine|rating'])**0.1000
#         J['W_engines', 'ac|propulsion|engine|BPR'] = -0.6615*np.exp(-0.0450*inputs['ac|propulsion|engine|BPR'])*(0.0010*inputs['ac|propulsion|engine|BPR'])**1.1000
#         # J['W_engines', 'ac|propulsion|engine|rating'] = 0
#         # J['W_engines', 'ac|propulsion|engine|BPR'] = 0


class Engine_JetTransport(ExplicitComponent):
    # Uses regression method in Roskam
    def initialize(self):
        self.options.declare("n_ult", default=3 * 1.5, desc="ultimate landing load factor, N_gear*1.5")

    def setup(self):
        self.add_input("ac|propulsion|engine|rating", units="lbf", desc="Rated thrust per engine")
        self.add_output("W_engine", units="lb")
        self.declare_partials(["W_engine"], ["*"])

    def compute(self, inputs, outputs):
        W_engine_roskam = 0.521 * (inputs["ac|propulsion|engine|rating"]) ** 0.9
        outputs["W_engine"] = W_engine_roskam

    def compute_partials(self, inputs, J):
        J["W_engine", "ac|propulsion|engine|rating"] = 0.4689 / inputs["ac|propulsion|engine|rating"] ** 0.1000


class EngineSystems_JetTransport(ExplicitComponent):
    def setup(self):
        self.add_input("ac|propulsion|engine|rating", units="lbf", desc="rated thrust per engine")
        self.add_input("W_engine", units="lb", desc="Estimate of engine weight")
        self.add_output("W_engineSystems", units="lb")
        self.declare_partials(["W_engineSystems"], ["*"])

    def compute(self, inputs, outputs):
        W_engineSystems_Roskam = (
            9.3300 * 0.0010 ** 1.0780 * inputs["W_engine"] ** 1.0780
            + 0.2600 * inputs["ac|propulsion|engine|rating"] ** (1 / 2)
            + 0.0820 * inputs["ac|propulsion|engine|rating"] ** 0.6500
            + 0.0340 * inputs["ac|propulsion|engine|rating"]
        )
        outputs["W_engineSystems"] = W_engineSystems_Roskam

    def compute_partials(self, inputs, J):
        J["W_engineSystems", "ac|propulsion|engine|rating"] = (
            0.0533 / inputs["ac|propulsion|engine|rating"] ** 0.3500
            + 0.1300 / inputs["ac|propulsion|engine|rating"] ** 0.5000
            + 0.0340
        )
        J["W_engineSystems", "W_engine"] = 9.3300 * 0.0010 ** 1.0780 * 1.0780 * inputs["W_engine"] ** 0.0780


class JetTransportEmptyWeight(Group):
    def setup(self):
        const = self.add_subsystem("const", IndepVarComp(), promotes_outputs=["*"])
        # const.add_output('W_fluids', val=20, units='kg')
        const.add_output("structural_fudge", val=1.6, units="m/m")
        const.add_output("n_engines", val=2, units="m/m")
        self.add_subsystem("wing", WingWeight_JetTrasport(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("hstabconst", HstabConst_JetTransport(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("hstab", HstabWeight_JetTransport(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("vstab", VstabWeight_JetTransport(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem(
            "fuselage_constant1", FuselageConst1_JetTransport(), promotes_inputs=["*"], promotes_outputs=["*"]
        )
        self.add_subsystem(
            "fuselage_constant2", FuselageConst2_JetTransport(), promotes_inputs=["*"], promotes_outputs=["*"]
        )
        self.add_subsystem("fuselage", FuselageWeight_JetTransport(), promotes_inputs=["*"], promotes_outputs=["*"])
        # self.add_subsystem('nacelle',Nacelle_JetTransport(),promotes_inputs=["*"],promotes_outputs=["*"])
        self.add_subsystem("mlg", MainLandingGear_JetTransport(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("nlg", NoseLandingGear_JetTransport(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("engine_dry", Engine_JetTransport(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem(
            "engine_systems", EngineSystems_JetTransport(), promotes_inputs=["*"], promotes_outputs=["*"]
        )
        self.add_subsystem(
            "single_engine_wet",
            AddSubtractComp(output_name="W_singleEngineWet", input_names=["W_engine", "W_engineSystems"], units="lb"),
            promotes_outputs=["*"],
            promotes_inputs=["*"],
        )
        self.add_subsystem(
            "total_engines",
            ElementMultiplyDivideComp(
                output_name="W_engines_total", input_names=["W_singleEngineWet", "n_engines"], input_units=["lb", "m/m"]
            ),
            promotes_outputs=["*"],
            promotes_inputs=["*"],
        )
        self.add_subsystem(
            "structural",
            AddSubtractComp(
                output_name="W_structure",
                input_names=["W_wing", "W_hstab", "W_vstab", "W_fuselage", "W_mlg", "W_nlg"],
                units="lb",
            ),
            promotes_outputs=["*"],
            promotes_inputs=["*"],
        )
        self.add_subsystem(
            "structural_fudge",
            ElementMultiplyDivideComp(
                output_name="W_structure_adjusted",
                input_names=["W_structure", "structural_fudge"],
                input_units=["lb", "m/m"],
            ),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )
        self.add_subsystem(
            "totalempty",
            AddSubtractComp(output_name="OEW", input_names=["W_structure_adjusted", "W_engines_total"], units="lb"),
            promotes_outputs=["*"],
            promotes_inputs=["*"],
        )


if __name__ == "__main__":
    from openmdao.api import IndepVarComp, Problem

    prob = Problem()
    prob.model = Group()
    dvs = prob.model.add_subsystem("dvs", IndepVarComp(), promotes_outputs=["*"])
    dvs.add_output("ac|weights|MTOW", 79002, units="kg")
    dvs.add_output("ac|geom|wing|S_ref", 124.6, units="m**2")
    dvs.add_output("ac|geom|wing|AR", 9.45)
    dvs.add_output("ac|geom|wing|c4sweep", 25, units="deg")
    dvs.add_output("ac|geom|wing|taper", 0.159)
    dvs.add_output("ac|geom|wing|toverc", 0.12)
    # dvs.add_output('V_H',255, units='kn')

    dvs.add_output("ac|geom|hstab|S_ref", 37.28, units="m**2")
    dvs.add_output("ac|geom|hstab|AR", 4.13)
    dvs.add_output("ac|geom|hstab|c4_to_wing_c4", 17.9, units="m")
    dvs.add_output("ac|geom|hstab|taper", 0.203)
    dvs.add_output("ac|geom|hstab|c4sweep", 30, units="deg")
    dvs.add_output("ac|geom|vstab|S_ref", 26.44, units="m**2")
    dvs.add_output("ac|geom|vstab|c4sweep", 35, units="deg")
    dvs.add_output("ac|geom|vstab|toverc", 0.12)
    dvs.add_output("ac|geom|vstab|AR", 1.94)

    dvs.add_output("ac|geom|fuselage|length", 39.12, units="m")
    dvs.add_output("ac|geom|fuselage|height", 4.01, units="m")
    dvs.add_output("ac|geom|fuselage|width", 3.76, units="m")
    dvs.add_output("ac|geom|fuselage|S_wet", 1077, units="m**2")
    # dvs.add_output('V_C',201, units='kn') #IAS (converted from 315kt true at 28,000 )
    # dvs.add_output('V_MO',266, units='kn')
    dvs.add_output("ac|propulsion|engine|rating", 27000, units="lbf")
    dvs.add_output("ac|weights|W_fuel_max", 2000, units="lb")
    dvs.add_output("ac|weights|MLW", 66349, units="kg")
    dvs.add_output("ac|geom|nosegear|length", 3, units="ft")
    dvs.add_output("ac|geom|nosegear|n_wheels", 2)
    dvs.add_output("ac|geom|maingear|length", 4, units="ft")
    dvs.add_output("ac|geom|maingear|n_wheels", 4)
    dvs.add_output("ac|aero|Vstall_land", 110, units="kn")
    dvs.add_output("ac|aero|LoverD", 17)

    prob.model.add_subsystem("OEW", JetTransportEmptyWeight(), promotes_inputs=["*"])

    prob.setup(force_alloc_complex=True)
    prob.run_model()
    print("Wing weight:")
    print(prob["OEW.W_wing"])
    print("Fuselage weight:")
    print(prob["OEW.W_fuselage"])
    print("Hstab Weight:")
    print(prob["OEW.W_hstab"])
    print("Vstab weight:")
    print(prob["OEW.W_vstab"])
    print("Main landing gear weight")
    print(prob["OEW.W_mlg"])
    print("Nose landing gear weight")
    print(prob["OEW.W_nlg"])
    print("Total engine weight")
    print(prob["OEW.W_engines_total"])
    print("Operating empty weight:")
    print(prob["OEW.OEW"])
    data = prob.check_partials(method="cs", compact_print=True, show_only_incorrect=False)