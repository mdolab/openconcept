import numpy as np
from openmdao.api import ExplicitComponent, IndepVarComp
from openmdao.api import Group
from openconcept.utilities import AddSubtractComp, ElementMultiplyDivideComp
import math

# TODO: add fuel system weight back in (depends on Wf, which depends on MTOW and We, and We depends on fuel system weight)


class WingWeight_SmallTurboprop(ExplicitComponent):
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
        self.add_input("ac|weights|W_fuel_max", units="lb", desc="Fuel weight")
        self.add_input("ac|geom|wing|S_ref", units="ft**2", desc="Reference wing area in sq ft")
        self.add_input("ac|geom|wing|AR", desc="Wing aspect ratio")
        self.add_input("ac|geom|wing|c4sweep", units="rad", desc="Quarter-chord sweep angle")
        self.add_input("ac|geom|wing|taper", desc="Wing taper ratio")
        self.add_input("ac|geom|wing|toverc", desc="Wing max thickness to chord ratio")
        # self.add_input('V_H', units='kn', desc='Max sea-level speed')
        self.add_input("ac|q_cruise", units="lb*ft**-2")

        # self.add_output('heat_out', units='W', desc='Waste heat out',shape=(nn,))
        self.add_output("W_wing", units="lb", desc="Wing weight")

        self.declare_partials(["W_wing"], ["*"])

    def compute(self, inputs, outputs):
        n_ult = self.options["n_ult"]
        # USAF method, Roskam PVC5pg68eq5.4
        # W_wing_USAF = 96.948*((inputs['ac|weights|MTOW']*n_ult/1e5)**0.65 * (inputs['ac|geom|wing|AR']/math.cos(inputs['ac|geom|wing|c4sweep']))**0.57 * (inputs['ac|geom|wing|S_ref']/100)**0.61 * ((1+inputs['ac|geom|wing|taper'])/2/inputs['ac|geom|wing|toverc'])**0.36 * (1+inputs['V_H']/500)**0.5)**0.993
        # Torenbeek, Roskam PVC5p68eq5.5
        # b = math.sqrt(inputs['ac|geom|wing|S_ref']*inputs['ac|geom|wing|AR'])
        # root_chord = 2*inputs['ac|geom|wing|S_ref']/b/(1+inputs['ac|geom|wing|taper'])
        # tr = root_chord * inputs['ac|geom|wing|toverc']
        # c2sweep_wing = inputs['ac|geom|wing|c4sweep'] # a hack for now
        # W_wing_Torenbeek = 0.00125*inputs['ac|weights|MTOW'] * (b/math.cos(c2sweep_wing))**0.75 * (1+ (6.3*math.cos(c2sweep_wing)/b)**0.5) * n_ult**0.55 * (b*inputs['ac|geom|wing|S_ref']/tr/inputs['ac|weights|MTOW']/math.cos(c2sweep_wing))**0.30

        W_wing_Raymer = (
            0.036
            * inputs["ac|geom|wing|S_ref"] ** 0.758
            * inputs["ac|weights|W_fuel_max"] ** 0.0035
            * (inputs["ac|geom|wing|AR"] / math.cos(inputs["ac|geom|wing|c4sweep"]) ** 2) ** 0.6
            * inputs["ac|q_cruise"] ** 0.006
            * inputs["ac|geom|wing|taper"] ** 0.04
            * (100 * inputs["ac|geom|wing|toverc"] / math.cos(inputs["ac|geom|wing|c4sweep"])) ** -0.3
            * (n_ult * inputs["ac|weights|MTOW"]) ** 0.49
        )

        outputs["W_wing"] = W_wing_Raymer

    def compute_partials(self, inputs, J):
        n_ult = self.options["n_ult"]
        J["W_wing", "ac|weights|MTOW"] = (
            0.036
            * inputs["ac|geom|wing|S_ref"] ** 0.758
            * inputs["ac|weights|W_fuel_max"] ** 0.0035
            * (inputs["ac|geom|wing|AR"] / math.cos(inputs["ac|geom|wing|c4sweep"]) ** 2) ** 0.6
            * inputs["ac|q_cruise"] ** 0.006
            * inputs["ac|geom|wing|taper"] ** 0.04
            * (100 * inputs["ac|geom|wing|toverc"] / math.cos(inputs["ac|geom|wing|c4sweep"])) ** -0.3
            * (n_ult * inputs["ac|weights|MTOW"]) ** (0.49 - 1)
            * n_ult
            * 0.49
        )
        J["W_wing", "ac|weights|W_fuel_max"] = (
            0.036
            * inputs["ac|geom|wing|S_ref"] ** 0.758
            * 0.0035
            * inputs["ac|weights|W_fuel_max"] ** (0.0035 - 1)
            * (inputs["ac|geom|wing|AR"] / math.cos(inputs["ac|geom|wing|c4sweep"]) ** 2) ** 0.6
            * inputs["ac|q_cruise"] ** 0.006
            * inputs["ac|geom|wing|taper"] ** 0.04
            * (100 * inputs["ac|geom|wing|toverc"] / math.cos(inputs["ac|geom|wing|c4sweep"])) ** -0.3
            * (n_ult * inputs["ac|weights|MTOW"]) ** 0.49
        )
        J["W_wing", "ac|geom|wing|S_ref"] = (
            0.036
            * inputs["ac|geom|wing|S_ref"] ** (0.758 - 1)
            * 0.758
            * inputs["ac|weights|W_fuel_max"] ** 0.0035
            * (inputs["ac|geom|wing|AR"] / math.cos(inputs["ac|geom|wing|c4sweep"]) ** 2) ** 0.6
            * inputs["ac|q_cruise"] ** 0.006
            * inputs["ac|geom|wing|taper"] ** 0.04
            * (100 * inputs["ac|geom|wing|toverc"] / math.cos(inputs["ac|geom|wing|c4sweep"])) ** -0.3
            * (n_ult * inputs["ac|weights|MTOW"]) ** 0.49
        )
        J["W_wing", "ac|geom|wing|AR"] = (
            0.036
            * inputs["ac|geom|wing|S_ref"] ** 0.758
            * inputs["ac|weights|W_fuel_max"] ** 0.0035
            * 0.6
            * (inputs["ac|geom|wing|AR"] / math.cos(inputs["ac|geom|wing|c4sweep"]) ** 2) ** (0.6 - 1)
            / math.cos(inputs["ac|geom|wing|c4sweep"]) ** 2
            * inputs["ac|q_cruise"] ** 0.006
            * inputs["ac|geom|wing|taper"] ** 0.04
            * (100 * inputs["ac|geom|wing|toverc"] / math.cos(inputs["ac|geom|wing|c4sweep"])) ** -0.3
            * (n_ult * inputs["ac|weights|MTOW"]) ** 0.49
        )
        c4const = (
            0.036
            * inputs["ac|geom|wing|S_ref"] ** 0.758
            * inputs["ac|weights|W_fuel_max"] ** 0.0035
            * inputs["ac|q_cruise"] ** 0.006
            * inputs["ac|geom|wing|taper"] ** 0.04
            * (n_ult * inputs["ac|weights|MTOW"]) ** 0.49
        )
        c4multa = (inputs["ac|geom|wing|AR"] / math.cos(inputs["ac|geom|wing|c4sweep"]) ** 2) ** 0.6
        c4multb = (100 * inputs["ac|geom|wing|toverc"] / math.cos(inputs["ac|geom|wing|c4sweep"])) ** -0.3
        dc4multa = (
            0.6
            * (inputs["ac|geom|wing|AR"] / math.cos(inputs["ac|geom|wing|c4sweep"]) ** 2) ** (0.6 - 1)
            * (-2 * inputs["ac|geom|wing|AR"] / math.cos(inputs["ac|geom|wing|c4sweep"]) ** 3)
            * (-math.sin(inputs["ac|geom|wing|c4sweep"]))
        )
        dc4multb = (
            -0.3
            * (100 * inputs["ac|geom|wing|toverc"] / math.cos(inputs["ac|geom|wing|c4sweep"])) ** (-0.3 - 1)
            * -100
            * inputs["ac|geom|wing|toverc"]
            / math.cos(inputs["ac|geom|wing|c4sweep"]) ** 2
            * (-math.sin(inputs["ac|geom|wing|c4sweep"]))
        )
        J["W_wing", "ac|geom|wing|c4sweep"] = c4const * (c4multa * dc4multb + c4multb * dc4multa)
        J["W_wing", "ac|geom|wing|taper"] = (
            0.036
            * inputs["ac|geom|wing|S_ref"] ** 0.758
            * inputs["ac|weights|W_fuel_max"] ** 0.0035
            * (inputs["ac|geom|wing|AR"] / math.cos(inputs["ac|geom|wing|c4sweep"]) ** 2) ** 0.6
            * inputs["ac|q_cruise"] ** 0.006
            * 0.04
            * inputs["ac|geom|wing|taper"] ** (0.04 - 1)
            * (100 * inputs["ac|geom|wing|toverc"] / math.cos(inputs["ac|geom|wing|c4sweep"])) ** -0.3
            * (n_ult * inputs["ac|weights|MTOW"]) ** 0.49
        )
        J["W_wing", "ac|geom|wing|toverc"] = (
            0.036
            * inputs["ac|geom|wing|S_ref"] ** 0.758
            * inputs["ac|weights|W_fuel_max"] ** 0.0035
            * (inputs["ac|geom|wing|AR"] / math.cos(inputs["ac|geom|wing|c4sweep"]) ** 2) ** 0.6
            * inputs["ac|q_cruise"] ** 0.006
            * inputs["ac|geom|wing|taper"] ** 0.04
            * -0.3
            * (100 * inputs["ac|geom|wing|toverc"] / math.cos(inputs["ac|geom|wing|c4sweep"])) ** (-0.3 - 1)
            * (100 / math.cos(inputs["ac|geom|wing|c4sweep"]))
            * (n_ult * inputs["ac|weights|MTOW"]) ** 0.49
        )
        J["W_wing", "ac|q_cruise"] = (
            0.036
            * inputs["ac|geom|wing|S_ref"] ** 0.758
            * inputs["ac|weights|W_fuel_max"] ** 0.0035
            * (inputs["ac|geom|wing|AR"] / math.cos(inputs["ac|geom|wing|c4sweep"]) ** 2) ** 0.6
            * 0.006
            * inputs["ac|q_cruise"] ** (0.006 - 1)
            * inputs["ac|geom|wing|taper"] ** 0.04
            * (100 * inputs["ac|geom|wing|toverc"] / math.cos(inputs["ac|geom|wing|c4sweep"])) ** -0.3
            * (n_ult * inputs["ac|weights|MTOW"]) ** 0.49
        )


class EmpennageWeight_SmallTurboprop(ExplicitComponent):
    """Inputs: MTOW, ac|geom|wing|S_ref, ac|geom|wing|AR, ac|geom|wing|c4sweep, ac|geom|wing|taper, ac|geom|wing|toverc, V_H (max SL speed)
    Outputs: W_wing
    Metadata: n_ult (ult load factor)

    """

    def initialize(self):
        self.options.declare("n_ult", default=3.8 * 1.5, desc="Ultimate load factor (dimensionless)")

    def setup(self):
        self.add_input("ac|geom|hstab|S_ref", units="ft**2", desc="Projected horiz stab area in sq ft")
        self.add_input("ac|geom|vstab|S_ref", units="ft**2", desc="Projected vert stab area in sq ft")
        # self.add_input('ac|geom|hstab|c4_to_wing_c4', units='ft', desc='Distance from wing c/4 to horiz stab c/4 (tail arm distance)')
        # self.add_input('ac|weights|MTOW', units='lb', desc='Maximum rated takeoff weight')
        # self.add_input('AR_h', desc='Horiz stab aspect ratio')
        # self.add_input('AR_v', units='rad', desc='Vert stab aspect ratio')
        # self.add_input('troot_h', units='ft', desc='Horiz stab root thickness (ft)')
        # self.add_input('troot_v', units='ft', desc='Vert stab root thickness (ft)')
        # self.add_input('ac|q_cruise', units='lb*ft**-2', desc='Cruise dynamic pressure')

        self.add_output("W_empennage", units="lb", desc="Empennage weight")
        self.declare_partials(["W_empennage"], ["*"])

    def compute(self, inputs, outputs):
        n_ult = self.options["n_ult"]
        # USAF method, Roskam PVC5pg72eq5.14/15
        # bh = math.sqrt(inputs['ac|geom|hstab|S_ref']*inputs['AR_h'])
        # bv = math.sqrt(inputs['ac|geom|vstab|S_ref']*inputs['AR_v'])
        # # Wh = 127 * ((inputs['ac|weights|MTOW']*n_ult/1e5)**0.87 * (inputs['ac|geom|hstab|S_ref']/100)**1.2 * 0.289*(inputs['ac|geom|hstab|c4_to_wing_c4']/10)**0.483 * (bh/inputs['troot_h'])**0.5)**0.458
        # # #Wh_raymer = 0.016 * (n_ult*inputs['ac|weights|MTOW'])**0.414 * inputs['ac|q_cruise']**0.168 * inputs['ac|geom|hstab|S_ref']**0.896 * (100 * 0.18)**-0.12 * (inputs['AR_h'])**0.043 * 0.7**-0.02
        # # Wv = 98.5 * ((inputs['ac|weights|MTOW']*n_ult/1e5)**0.87 * (inputs['ac|geom|vstab|S_ref']/100)**1.2 * 0.289 * (bv/inputs['troot_v'])**0.5)**0.458

        # # Wemp_USAF = Wh + Wv

        # Torenbeek, Roskam PVC5p73eq5.16
        Wemp_Torenbeek = 0.04 * (n_ult * (inputs["ac|geom|vstab|S_ref"] + inputs["ac|geom|hstab|S_ref"]) ** 2) ** 0.75
        outputs["W_empennage"] = Wemp_Torenbeek

    def compute_partials(self, inputs, J):
        n_ult = self.options["n_ult"]
        J["W_empennage", "ac|geom|vstab|S_ref"] = (
            0.75
            * 0.04
            * (n_ult * (inputs["ac|geom|vstab|S_ref"] + inputs["ac|geom|hstab|S_ref"]) ** 2) ** (0.75 - 1)
            * (n_ult * 2 * (inputs["ac|geom|vstab|S_ref"] + inputs["ac|geom|hstab|S_ref"]))
        )
        J["W_empennage", "ac|geom|hstab|S_ref"] = (
            0.75
            * 0.04
            * (n_ult * (inputs["ac|geom|vstab|S_ref"] + inputs["ac|geom|hstab|S_ref"]) ** 2) ** (0.75 - 1)
            * (n_ult * 2 * (inputs["ac|geom|vstab|S_ref"] + inputs["ac|geom|hstab|S_ref"]))
        )


class FuselageWeight_SmallTurboprop(ExplicitComponent):
    def initialize(self):
        # self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')
        # define configuration parameters
        self.options.declare("n_ult", default=3.8 * 1.5, desc="Ultimate load factor (dimensionless)")

    def setup(self):
        # nn = self.options['num_nodes']
        self.add_input("ac|weights|MTOW", units="lb", desc="Maximum rated takeoff weight")
        self.add_input("ac|geom|fuselage|length", units="ft", desc="Fuselage length (not counting nacelle")
        self.add_input("ac|geom|fuselage|height", units="ft", desc="Fuselage height")
        self.add_input("ac|geom|fuselage|width", units="ft", desc="Fuselage weidth")
        # self.add_input('V_C', units='kn', desc='Indicated cruise airspeed (KEAS)')
        # self.add_input('V_MO', units='kn', desc='Max operating speed (indicated)')
        self.add_input("ac|geom|fuselage|S_wet", units="ft**2", desc="Fuselage shell area")
        self.add_input("ac|geom|hstab|c4_to_wing_c4", units="ft", desc="Horiz tail arm")
        self.add_input("ac|q_cruise", units="lb*ft**-2", desc="Dynamic pressure at cruise")

        # self.add_output('heat_out', units='W', desc='Waste heat out',shape=(nn,))
        self.add_output("W_fuselage", units="lb", desc="Fuselage weight")
        self.declare_partials(["W_fuselage"], ["*"])

    def compute(self, inputs, outputs):
        n_ult = self.options["n_ult"]
        # USAF method, Roskam PVC5pg76eq5.25
        # W_fuselage_USAF = 200*((inputs['ac|weights|MTOW']*n_ult/1e5)**0.286 * (inputs['ac|geom|fuselage|length']/10)**0.857 * (inputs['ac|geom|fuselage|width']+inputs['ac|geom|fuselage|height'])/10 * (inputs['V_C']/100)**0.338)**1.1
        # print(W_fuselage_USAF)

        # W_fuselage_Torenbeek = 0.021 * 1.08 * ((inputs['V_MO']*inputs['ac|geom|hstab|c4_to_wing_c4']/(inputs['ac|geom|fuselage|width']+inputs['ac|geom|fuselage|height']))**0.5 * inputs['ac|geom|fuselage|S_wet']**1.2)
        W_press = (
            11.9
            * (
                math.pi
                * (inputs["ac|geom|fuselage|width"] + inputs["ac|geom|fuselage|height"])
                / 2
                * inputs["ac|geom|fuselage|length"]
                * 0.8
                * 8
            )
            ** 0.271
        )
        W_fuselage_Raymer = (
            0.052
            * inputs["ac|geom|fuselage|S_wet"] ** 1.086
            * (n_ult * inputs["ac|weights|MTOW"]) ** 0.177
            * inputs["ac|geom|hstab|c4_to_wing_c4"] ** -0.051
            * (inputs["ac|geom|fuselage|length"] / inputs["ac|geom|fuselage|height"]) ** -0.072
            * inputs["ac|q_cruise"] ** 0.241
            + W_press
        )
        outputs["W_fuselage"] = W_fuselage_Raymer

    def compute_partials(self, inputs, J):
        n_ult = self.options["n_ult"]
        J["W_fuselage", "ac|weights|MTOW"] = (
            0.052
            * inputs["ac|geom|fuselage|S_wet"] ** 1.086
            * 0.177
            * n_ult
            * (n_ult * inputs["ac|weights|MTOW"]) ** (0.177 - 1)
            * inputs["ac|geom|hstab|c4_to_wing_c4"] ** -0.051
            * (inputs["ac|geom|fuselage|length"] / inputs["ac|geom|fuselage|height"]) ** -0.072
            * inputs["ac|q_cruise"] ** 0.241
        )
        J["W_fuselage", "ac|geom|fuselage|width"] = (
            0.271
            * 11.9
            * (
                math.pi
                * (inputs["ac|geom|fuselage|width"] + inputs["ac|geom|fuselage|height"])
                / 2
                * inputs["ac|geom|fuselage|length"]
                * 0.8
                * 8
            )
            ** (0.271 - 1)
            * (math.pi / 2 * inputs["ac|geom|fuselage|length"] * 0.8 * 8)
        )
        J["W_fuselage", "ac|geom|fuselage|height"] = (
            0.271
            * 11.9
            * (
                math.pi
                * (inputs["ac|geom|fuselage|width"] + inputs["ac|geom|fuselage|height"])
                / 2
                * inputs["ac|geom|fuselage|length"]
                * 0.8
                * 8
            )
            ** (0.271 - 1)
            * (math.pi / 2 * inputs["ac|geom|fuselage|length"] * 0.8 * 8)
            + 0.052
            * inputs["ac|geom|fuselage|S_wet"] ** 1.086
            * (n_ult * inputs["ac|weights|MTOW"]) ** 0.177
            * inputs["ac|geom|hstab|c4_to_wing_c4"] ** -0.051
            * -0.072
            * (inputs["ac|geom|fuselage|length"] / inputs["ac|geom|fuselage|height"]) ** (-0.072 - 1)
            * (-inputs["ac|geom|fuselage|length"] / inputs["ac|geom|fuselage|height"] ** 2)
            * inputs["ac|q_cruise"] ** 0.241
        )
        J["W_fuselage", "ac|geom|fuselage|length"] = (
            0.271
            * 11.9
            * (
                math.pi
                * (inputs["ac|geom|fuselage|width"] + inputs["ac|geom|fuselage|height"])
                / 2
                * inputs["ac|geom|fuselage|length"]
                * 0.8
                * 8
            )
            ** (0.271 - 1)
            * (math.pi * (inputs["ac|geom|fuselage|width"] + inputs["ac|geom|fuselage|height"]) / 2 * 0.8 * 8)
            + 0.052
            * inputs["ac|geom|fuselage|S_wet"] ** 1.086
            * (n_ult * inputs["ac|weights|MTOW"]) ** 0.177
            * inputs["ac|geom|hstab|c4_to_wing_c4"] ** -0.051
            * -0.072
            * (inputs["ac|geom|fuselage|length"] / inputs["ac|geom|fuselage|height"]) ** (-0.072 - 1)
            * (1 / inputs["ac|geom|fuselage|height"])
            * inputs["ac|q_cruise"] ** 0.241
        )
        J["W_fuselage", "ac|geom|fuselage|S_wet"] = (
            0.052
            * 1.086
            * inputs["ac|geom|fuselage|S_wet"] ** (1.086 - 1)
            * (n_ult * inputs["ac|weights|MTOW"]) ** 0.177
            * inputs["ac|geom|hstab|c4_to_wing_c4"] ** -0.051
            * (inputs["ac|geom|fuselage|length"] / inputs["ac|geom|fuselage|height"]) ** -0.072
            * inputs["ac|q_cruise"] ** 0.241
        )
        J["W_fuselage", "ac|geom|hstab|c4_to_wing_c4"] = (
            0.052
            * inputs["ac|geom|fuselage|S_wet"] ** 1.086
            * (n_ult * inputs["ac|weights|MTOW"]) ** 0.177
            * -0.051
            * inputs["ac|geom|hstab|c4_to_wing_c4"] ** (-0.051 - 1)
            * (inputs["ac|geom|fuselage|length"] / inputs["ac|geom|fuselage|height"]) ** -0.072
            * inputs["ac|q_cruise"] ** 0.241
        )
        J["W_fuselage", "ac|q_cruise"] = (
            0.052
            * inputs["ac|geom|fuselage|S_wet"] ** 1.086
            * (n_ult * inputs["ac|weights|MTOW"]) ** 0.177
            * inputs["ac|geom|hstab|c4_to_wing_c4"] ** -0.051
            * (inputs["ac|geom|fuselage|length"] / inputs["ac|geom|fuselage|height"]) ** -0.072
            * 0.241
            * inputs["ac|q_cruise"] ** (0.241 - 1)
        )


class NacelleWeight_SmallSingleTurboprop(ExplicitComponent):
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
        self.add_input("P_TO", units="hp", desc="Takeoff power")

        # self.add_output('heat_out', units='W', desc='Waste heat out',shape=(nn,))
        self.add_output("W_nacelle", units="lb", desc="Nacelle weight")
        self.declare_partials(["W_nacelle"], ["*"])

    def compute(self, inputs, outputs):
        # Torenbeek method, Roskam PVC5pg78eq5.30
        W_nacelle = 2.5 * inputs["P_TO"] ** 0.5
        outputs["W_nacelle"] = W_nacelle

    def compute_partials(self, inputs, J):
        # Torenbeek method, Roskam PVC5pg78eq5.30
        J["W_nacelle", "P_TO"] = 0.5 * 2.5 * inputs["P_TO"] ** (0.5 - 1)


class NacelleWeight_MultiTurboprop(ExplicitComponent):
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
        self.add_input("P_TO", units="hp", desc="Takeoff power")

        # self.add_output('heat_out', units='W', desc='Waste heat out',shape=(nn,))
        self.add_output("W_nacelle", units="lb", desc="Nacelle weight")
        self.declare_partials(["W_nacelle"], ["*"])

    def compute(self, inputs, outputs):
        # Torenbeek method, Roskam PVC5pg78eq5.33
        W_nacelle = 0.14 * inputs["P_TO"]
        outputs["W_nacelle"] = W_nacelle

    def compute_partials(self, inputs, J):
        # Torenbeek method, Roskam PVC5pg78eq5.30
        J["W_nacelle", "P_TO"] = 0.14


class LandingGearWeight_SmallTurboprop(ExplicitComponent):
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
        # self.add_input('ac|weights|MTOW', units='lb',desc='Max takeoff weight')
        self.add_input("ac|weights|MLW", units="lb", desc="Max landing weight")
        self.add_input("ac|geom|maingear|length", units="ft", desc="Main landing gear extended length")
        self.add_input("ac|geom|nosegear|length", units="ft", desc="Nose gear extended length")

        # self.add_output('heat_out', units='W', desc='Waste heat out',shape=(nn,))
        self.add_output("W_gear", units="lb", desc="Gear weight (nose and main)")
        self.declare_partials(["W_gear"], ["*"])

    def compute(self, inputs, outputs):
        n_ult = self.options["n_ult"]
        # Torenbeek method, Roskam PVC5pg82eq5.42
        # W_gear_Torenbeek_main = 33.0+0.04*inputs['ac|weights|MTOW']**0.75 + 0.021*inputs['ac|weights|MTOW']
        # W_gear_Torenbeek_nose = 12.0+0.06*inputs['ac|weights|MTOW']**0.75

        W_gear_Raymer_main = (
            0.095 * (n_ult * inputs["ac|weights|MLW"]) ** 0.768 * (inputs["ac|geom|maingear|length"] / 12) ** 0.409
        )
        W_gear_Raymer_nose = (
            0.125 * (n_ult * inputs["ac|weights|MLW"]) ** 0.566 * (inputs["ac|geom|nosegear|length"] / 12) ** 0.845
        )

        W_gear = W_gear_Raymer_main + W_gear_Raymer_nose
        outputs["W_gear"] = W_gear

    def compute_partials(self, inputs, J):
        n_ult = self.options["n_ult"]
        J["W_gear", "ac|weights|MLW"] = (
            0.095
            * (n_ult * inputs["ac|weights|MLW"]) ** (0.768 - 1)
            * 0.768
            * n_ult
            * (inputs["ac|geom|maingear|length"] / 12) ** 0.409
            + 0.125
            * (n_ult * inputs["ac|weights|MLW"]) ** (0.566 - 1)
            * 0.566
            * n_ult
            * (inputs["ac|geom|nosegear|length"] / 12) ** 0.845
        )
        J["W_gear", "ac|geom|maingear|length"] = (
            0.095
            * (n_ult * inputs["ac|weights|MLW"]) ** 0.768
            * (inputs["ac|geom|maingear|length"] / 12) ** (0.409 - 1)
            * (1 / 12)
            * 0.409
        )
        J["W_gear", "ac|geom|nosegear|length"] = (
            0.125
            * (n_ult * inputs["ac|weights|MLW"]) ** 0.566
            * (inputs["ac|geom|nosegear|length"] / 12) ** (0.845 - 1)
            * (1 / 12)
            * 0.845
        )


class FuelSystemWeight_SmallTurboprop(ExplicitComponent):
    def initialize(self):
        # self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')
        # define configuration parameters
        self.options.declare("Kfsp", default=6.55, desc="Fuel density (lbs/gal)")
        # self.options.declare('num_tanks', default=2, desc='Number of fuel tanks')
        # self.options.declare('num_engines', default=1, desc='Number of engines')

    def setup(self):
        # nn = self.options['num_nodes']
        self.add_input("ac|weights|W_fuel_max", units="lb", desc="Full fuel weight")

        # self.add_output('heat_out', units='W', desc='Waste heat out',shape=(nn,))
        self.add_output("W_fuelsystem", units="lb", desc="Fuel system weight")
        self.declare_partials("W_fuelsystem", "ac|weights|W_fuel_max")

    def compute(self, inputs, outputs):
        # n_t = self.options['num_tanks']
        # n_e = self.options['num_engines']
        Kfsp = self.options["Kfsp"]
        # Torenbeek method, Roskam PVC6pg92eq6.24
        # W_fs_Cessna = 0.4 * inputs['ac|weights|W_fuel_max'] / Kfsp
        # W_fs_Torenbeek = 80*(n_e+n_t-1) + 15*n_t**0.5 * (inputs['ac|weights|W_fuel_max']/Kfsp)**0.333
        # print(W_fs_Torenbeek)
        # W_fs_USAF = 2.49* ((inputs['ac|weights|W_fuel_max']/Kfsp)**0.6 * n_t**0.20 * n_e**0.13)**1.21
        # print(W_fs_USAF)
        W_fs_Raymer = 2.49 * (inputs["ac|weights|W_fuel_max"] * 1.0 / Kfsp) ** 0.726 * (0.5) ** 0.363
        outputs["W_fuelsystem"] = W_fs_Raymer

    def compute_partials(self, inputs, J):
        Kfsp = self.options["Kfsp"]
        J["W_fuelsystem", "ac|weights|W_fuel_max"] = (
            2.49 * 0.726 * (inputs["ac|weights|W_fuel_max"] * 1.0 / Kfsp) ** (0.726 - 1) * (1.0 / Kfsp) * (0.5) ** 0.363
        )


class EquipmentWeight_SmallTurboprop(ExplicitComponent):
    def setup(self):
        self.add_input("ac|weights|MTOW", units="lb", desc="Max takeoff weight")
        self.add_input("ac|num_passengers_max", desc="Number of passengers")
        self.add_input("ac|geom|fuselage|length", units="ft", desc="fuselage width")
        self.add_input("ac|geom|wing|AR", desc="Wing aspect ratio")
        self.add_input("ac|geom|wing|S_ref", units="ft**2", desc="Wing reference area")
        self.add_input("W_fuelsystem", units="lb", desc="Fuel system weight")
        self.add_output("W_equipment", units="lb", desc="Equipment weight")
        self.declare_partials(["W_equipment"], ["*"])

    def compute(self, inputs, outputs):
        b = math.sqrt(inputs["ac|geom|wing|S_ref"] * inputs["ac|geom|wing|AR"])

        # Flight control system (unpowered)
        # Roskam PVC7p98eq7.2
        # Wfc_USAF = 1.066*inputs['ac|weights|MTOW']**0.626
        Wfc_Torenbeek = 0.23 * inputs["ac|weights|MTOW"] ** 0.666
        # Hydraulic system weight included in flight controls and LG weight
        Whydraulics = 0.2673 * 1 * (inputs["ac|geom|fuselage|length"] * b) ** 0.937

        # Guesstimate of avionics weight
        # This is a guess for a single turboprop class airplane (such as TBM, Pilatus, etc)
        Wavionics = 2.117 * (np.array([110])) ** 0.933
        # Electrical system weight (NOT including elec propulsion)
        Welec = 12.57 * (inputs["W_fuelsystem"] + Wavionics) ** 0.51

        # pressurization and air conditioning from Roskam
        Wapi = (
            0.265
            * inputs["ac|weights|MTOW"] ** 0.52
            * inputs["ac|num_passengers_max"] ** 0.68
            * Wavionics**0.17
            * 0.95
        )
        Woxygen = 30 + 1.2 * inputs["ac|num_passengers_max"]
        # furnishings (Cessna method)
        Wfur = 0.412 * inputs["ac|num_passengers_max"] ** 1.145 * inputs["ac|weights|MTOW"] ** 0.489
        Wpaint = 0.003 * inputs["ac|weights|MTOW"]

        outputs["W_equipment"] = Wfc_Torenbeek + Welec + Wavionics + Wapi + Woxygen + Wfur + Wpaint + Whydraulics

    def compute_partials(self, inputs, J):
        b = math.sqrt(inputs["ac|geom|wing|S_ref"] * inputs["ac|geom|wing|AR"])
        Wavionics = 2.117 * (np.array([110])) ** 0.933
        J["W_equipment", "ac|weights|MTOW"] = (
            0.23 * inputs["ac|weights|MTOW"] ** (0.666 - 1) * 0.666
            + 0.52
            * 0.265
            * inputs["ac|weights|MTOW"] ** (0.52 - 1)
            * inputs["ac|num_passengers_max"] ** 0.68
            * Wavionics**0.17
            * 0.95
            + 0.412 * inputs["ac|num_passengers_max"] ** 1.145 * inputs["ac|weights|MTOW"] ** (0.489 - 1) * 0.489
            + 0.003
        )
        J["W_equipment", "ac|num_passengers_max"] = (
            0.265
            * inputs["ac|weights|MTOW"] ** 0.52
            * 0.68
            * inputs["ac|num_passengers_max"] ** (0.68 - 1)
            * Wavionics**0.17
            * 0.95
            + 1.2
            + 0.412 * 1.145 * inputs["ac|num_passengers_max"] ** (1.145 - 1) * inputs["ac|weights|MTOW"] ** 0.489
        )
        J["W_equipment", "ac|geom|fuselage|length"] = (
            0.2673 * 1 * 0.937 * (inputs["ac|geom|fuselage|length"] * b) ** (0.937 - 1) * b
        )
        J["W_equipment", "W_fuelsystem"] = 12.57 * (inputs["W_fuelsystem"] + Wavionics) ** (0.51 - 1) * 0.51
        J["W_equipment", "ac|geom|wing|S_ref"] = (
            0.2673
            * 1
            * (inputs["ac|geom|fuselage|length"] * b) ** (0.937 - 1)
            * 0.937
            * inputs["ac|geom|fuselage|length"]
            * (1 / 2)
            * 1
            / b
            * inputs["ac|geom|wing|AR"]
        )
        J["W_equipment", "ac|geom|wing|AR"] = (
            0.2673
            * 1
            * (inputs["ac|geom|fuselage|length"] * b) ** (0.937 - 1)
            * 0.937
            * inputs["ac|geom|fuselage|length"]
            * (1 / 2)
            / b
            * inputs["ac|geom|wing|S_ref"]
        )


class SingleTurboPropEmptyWeight(Group):
    def setup(self):
        const = self.add_subsystem("const", IndepVarComp(), promotes_outputs=["*"])
        const.add_output("W_fluids", val=20, units="kg")
        const.add_output("structural_fudge", val=1.6, units="m/m")
        self.add_subsystem("wing", WingWeight_SmallTurboprop(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("empennage", EmpennageWeight_SmallTurboprop(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("fuselage", FuselageWeight_SmallTurboprop(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem(
            "nacelle", NacelleWeight_SmallSingleTurboprop(), promotes_inputs=["*"], promotes_outputs=["*"]
        )
        self.add_subsystem("gear", LandingGearWeight_SmallTurboprop(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem(
            "fuelsystem", FuelSystemWeight_SmallTurboprop(), promotes_inputs=["*"], promotes_outputs=["*"]
        )
        self.add_subsystem("equipment", EquipmentWeight_SmallTurboprop(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem(
            "structural",
            AddSubtractComp(
                output_name="W_structure",
                input_names=["W_wing", "W_fuselage", "W_nacelle", "W_empennage", "W_gear"],
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
            AddSubtractComp(
                output_name="OEW",
                input_names=[
                    "W_structure_adjusted",
                    "W_fuelsystem",
                    "W_equipment",
                    "W_engine",
                    "W_propeller",
                    "W_fluids",
                ],
                units="lb",
            ),
            promotes_outputs=["*"],
            promotes_inputs=["*"],
        )


if __name__ == "__main__":
    from openmdao.api import Problem

    prob = Problem()
    prob.model = Group()
    dvs = prob.model.add_subsystem("dvs", IndepVarComp(), promotes_outputs=["*"])
    AR = 41.5**2 / 193.75
    dvs.add_output("ac|weights|MTOW", 7394.0, units="lb")
    dvs.add_output("ac|geom|wing|S_ref", 193.75, units="ft**2")
    dvs.add_output("ac|geom|wing|AR", AR)
    dvs.add_output("ac|geom|wing|c4sweep", 1.0, units="deg")
    dvs.add_output("ac|geom|wing|taper", 0.622)
    dvs.add_output("ac|geom|wing|toverc", 0.16)
    # dvs.add_output('V_H',255, units='kn')

    dvs.add_output("ac|geom|hstab|S_ref", 47.5, units="ft**2")
    # dvs.add_output('AR_h',4.13)
    dvs.add_output("ac|geom|vstab|S_ref", 31.36, units="ft**2")
    # dvs.add_output('AR_v',1.2)
    # dvs.add_output('troot_h',0.8, units='ft')
    # dvs.add_output('troot_v',0.8, units='ft')
    dvs.add_output("ac|geom|hstab|c4_to_wing_c4", 17.9, units="ft")

    dvs.add_output("ac|geom|fuselage|length", 27.39, units="ft")
    dvs.add_output("ac|geom|fuselage|height", 5.555, units="ft")
    dvs.add_output("ac|geom|fuselage|width", 4.58, units="ft")
    dvs.add_output("ac|geom|fuselage|S_wet", 392, units="ft**2")
    # dvs.add_output('V_C',201, units='kn') #IAS (converted from 315kt true at 28,000 )
    # dvs.add_output('V_MO',266, units='kn')
    dvs.add_output("P_TO", 850, units="hp")
    dvs.add_output("ac|weights|W_fuel_max", 2000, units="lb")
    dvs.add_output("ac|num_passengers_max", 6)
    dvs.add_output("ac|q_cruise", 135.4, units="lb*ft**-2")
    dvs.add_output("ac|weights|MLW", 7000, units="lb")
    dvs.add_output("ac|geom|nosegear|length", 3, units="ft")
    dvs.add_output("ac|geom|maingear|length", 4, units="ft")
    dvs.add_output("W_engine", 475, units="lb")
    dvs.add_output("W_propeller", 150, units="lb")

    prob.model.add_subsystem("OEW", SingleTurboPropEmptyWeight(), promotes_inputs=["*"])

    # prob.model.add_subsystem('wing',WingWeight_SmallTurboprop(),promotes_inputs=["*"])
    # prob.model.add_subsystem('empennage',EmpennageWeight_SmallTurboprop(),promotes_inputs=["*"])
    # prob.model.add_subsystem('fuselage',FuselageWeight_SmallTurboprop(),promotes_inputs=["*"])
    # prob.model.add_subsystem('nacelle',NacelleWeight_SmallSingleTurboprop(),promotes_inputs=["*"])
    # prob.model.add_subsystem('gear',LandingGearWeight_SmallTurboprop(),promotes_inputs=["*"])
    # prob.model.add_subsystem('fuelsystem', FuelSystemWeight_SmallTurboprop(), promotes_inputs=["*"])
    # prob.model.add_subsystem('equipment',EquipmentWeight_SmallTurboprop(), promotes_inputs=["*"])

    prob.setup()
    prob.run_model()
    print("Wing weight:")
    print(prob["OEW.W_wing"])
    print("Fuselage weight:")
    print(prob["OEW.W_fuselage"])
    print("Empennage weight:")
    print(prob["OEW.W_empennage"])
    print("Nacelle weight:")
    print(prob["OEW.W_nacelle"])
    print("Fuel system weight")
    print(prob["OEW.W_fuelsystem"])
    print("Gear weight")
    print(prob["OEW.W_gear"])
    print("Equipment weight")
    print(prob["OEW.W_equipment"])
    print("Operating empty weight:")
    print(prob["OEW.OEW"])
    data = prob.check_partials(compact_print=True)
