import openmdao.api as om
import numpy as np

from openconcept.utilities import LinearInterpolator
from .thermal import PerfectHeatTransferComp


class LinearSelector(om.ExplicitComponent):
    """
    Averages thermal parameters given bypass

    Inputs
    ------
    T_in_hot : float
        Incoming coolant temperature on the hot side (vector, K)
    T_in_cold : float
        Incoming coolant temperature on the cold side (vector, K)
    T_out_refrig_cold : float
        Coolant temperature in chiller outlet on the cold side (vector, K)
    T_out_refrig_hot : float
        Coolant temperature in chiller outlet on the hot side (vector, K)
    power_rating : float
        Rated electric power (scalar, W)
    bypass : float
        Bypass parameter in range 0 - 1 (inclusive); 0 represents full
        refrig and no bypass, 1 all bypass no refrig (vector, None)

    Outputs
    -------
    T_out_cold : float
        Outgoing coolant temperature on the cold side (vector, K)
    T_out_hot : float
        Outgoing coolant temperature on the hot side (vector, K)
    elec_load : float
        Electrical load (vector, W)

    Options
    -------
    num_nodes : int
        The number of analysis points to run
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of analysis points")

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("T_in_cold", val=300 * np.ones((nn,)), units="K")
        self.add_input("T_in_hot", val=305 * np.ones((nn,)), units="K")
        self.add_input("T_out_refrig_cold", val=290 * np.ones((nn,)), units="K")
        self.add_input("T_out_refrig_hot", val=310 * np.ones((nn,)), units="K")
        self.add_input("power_rating", units="W")
        self.add_input("bypass", val=np.ones((nn,)), units=None)

        self.add_output("elec_load", val=np.ones((nn,)) * 1, units="W")
        self.add_output("T_out_cold", val=290 * np.ones((nn,)), units="K")
        self.add_output("T_out_hot", val=310 * np.ones((nn,)), units="K")

        self.declare_partials(
            "T_out_cold", ["bypass", "T_in_hot", "T_out_refrig_cold"], rows=np.arange(nn), cols=np.arange(nn)
        )
        self.declare_partials(
            "T_out_hot", ["bypass", "T_in_cold", "T_out_refrig_hot"], rows=np.arange(nn), cols=np.arange(nn)
        )
        self.declare_partials("elec_load", "bypass", rows=np.arange(nn), cols=np.arange(nn))
        self.declare_partials("elec_load", "power_rating", rows=np.arange(nn), cols=np.zeros(nn))

    def compute(self, inputs, outputs):
        bypass_side = inputs["bypass"]
        refrig_side = 1 - inputs["bypass"]
        outputs["T_out_cold"] = bypass_side * inputs["T_in_hot"] + refrig_side * inputs["T_out_refrig_cold"]
        outputs["T_out_hot"] = bypass_side * inputs["T_in_cold"] + refrig_side * inputs["T_out_refrig_hot"]
        outputs["elec_load"] = refrig_side * inputs["power_rating"] / 0.95

    def compute_partials(self, inputs, J):
        T_in_hot = inputs["T_in_hot"]
        T_out_hot = inputs["T_out_refrig_hot"]
        T_in_cold = inputs["T_in_cold"]
        T_out_cold = inputs["T_out_refrig_cold"]
        power_rating = inputs["power_rating"]
        bypass = inputs["bypass"]

        J["T_out_cold", "bypass"] = T_in_hot - T_out_cold
        J["T_out_cold", "T_in_hot"] = bypass
        J["T_out_cold", "T_out_refrig_cold"] = 1 - bypass
        J["T_out_hot", "bypass"] = T_in_cold - T_out_hot
        J["T_out_hot", "T_in_cold"] = bypass
        J["T_out_hot", "T_out_refrig_hot"] = 1 - bypass
        J["elec_load", "bypass"] = -power_rating / 0.95
        J["elec_load", "power_rating"] = (1 - bypass) / 0.95


class COPHeatPump(om.ExplicitComponent):
    """
    Models heat transfer to coolant loop assuming zero thermal resistance.

    Inputs
    ------
    COP : float
        Coeff of performance set by optimizer (vector, dimensionless)
    power_rating : float
        Shaft work in the refrigerator (scalar, W)

    Outputs
    -------
    q_in_1 : float
        Heat transfer rate INTO side 1 (vector, W)
    q_in_2 : float
        Heat transfer rate INTO side 2 (vector, W)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, default 1)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of analysis points")

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("COP", val=np.ones((nn,)), units=None)
        self.add_input("power_rating", units="W", val=1000)

        self.add_output("q_in_1", val=np.zeros((nn,)), units="W", shape=(nn,))
        self.add_output("q_in_2", val=np.zeros((nn,)), units="W", shape=(nn,))

        self.declare_partials(["q_in_1", "q_in_2"], "COP", rows=np.arange(nn), cols=np.arange(nn))
        self.declare_partials(["q_in_1", "q_in_2"], "power_rating", rows=np.arange(nn), cols=np.zeros(nn))

    def compute(self, inputs, outputs):
        outputs["q_in_1"] = -inputs["COP"] * inputs["power_rating"]
        outputs["q_in_2"] = (inputs["COP"] + 1) * inputs["power_rating"]

    def compute_partials(self, inputs, J):
        COP = inputs["COP"]
        power_rating = inputs["power_rating"]

        J["q_in_1", "COP"] = -power_rating
        J["q_in_1", "power_rating"] = -COP
        J["q_in_2", "COP"] = power_rating
        J["q_in_2", "power_rating"] = COP + 1


class HeatPumpWeight(om.ExplicitComponent):
    """
    Computes weight and power metrics for the vapor cycle machine.
    Defaults based on limited published data and guesswork.

    Inputs
    ------
    power_rating : float
        Rated electric power (scalar, W)
    specific_power : float
        Power per weight (scalar, W/kg)

    Outputs
    -------
    component_weight : float
        Component weight (including coolants + motor) (scalar, kg)
    """

    def setup(self):
        self.add_input("power_rating", val=1000.0, units="W")
        self.add_input("specific_power", val=200.0, units="W/kg")
        self.add_output("component_weight", val=0.0, units="kg")
        self.declare_partials("component_weight", ["power_rating", "specific_power"])

    def compute(self, inputs, outputs):
        outputs["component_weight"] = inputs["power_rating"] / inputs["specific_power"]

    def compute_partials(self, inputs, J):
        J["component_weight", "power_rating"] = 1 / inputs["specific_power"]
        J["component_weight", "specific_power"] = -inputs["power_rating"] / inputs["specific_power"] ** 2


class HeatPumpWithIntegratedCoolantLoop(om.Group):
    """
    Models chiller with integrated coolant inputs and outputs
    on the hot and cold sides. Can bypass the chiller using
    linearly interpolated control points control.bypass_start
    and control.bypass_end outputs (0 is full refrigerator,
    1 is full bypass, continuous in between).

    Inputs
    ------
    T_in_hot : float
        Incoming coolant temperature on the hot side (vector, K)
    T_in_cold : float
        Incoming coolant temperature on the cold side (vector, K)
    mdot_coolant : float
        Coolant mass flow rate (vector, kg/s)
    power_rating : float
        Rated electric power, default 1 kW (scalar, W)
    specific_power : float
        Heat pump power per weight, default 200 W/kg (scalar, W/kg)
    eff_factor : float
        Heat pump Carnot efficiency factor, default 0.4 (scalar, None)
    control.bypass_start : float
        Bypass value (in range 0-1) at beginning used for linear interpolation,
        0 is full refrigerator and 1 is full bypass; must access via
        control component (i.e. with "control.bypass_start") (scalar, None)
    control.bypass_end : float
        Bypass value (in range 0-1) at end used for linear interpolation,
        0 is full refrigerator and 1 is full bypass; must access via
        control component (i.e. with "control.bypass_end") (scalar, None)

    Outputs
    -------
    T_out_hot : float
        Outgoing coolant temperature on the hot side (vector, K)
    T_out_cold : float
        Outgoing coolant temperature on the cold side (vector, K)
    component_weight : float
        Component weight (including coolants + motor) (scalar, kg)
    elec_load : float
        Electrical load (vector, W)

    Options
    -------
    num_nodes : int
        The number of analysis points to run
    specific_heat : float
        Specific heat of the coolant (scalar, J/kg/K, default 3801 glycol/water)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of analysis points")
        self.options.declare("specific_heat", default=3801.0, desc="Specific heat in J/kg/K")

    def setup(self):
        nn = self.options["num_nodes"]
        nn_ones = np.ones((nn,))
        spec_heat = self.options["specific_heat"]

        iv = self.add_subsystem("control", om.IndepVarComp())
        iv.add_output("bypass_start", val=1.0)
        iv.add_output("bypass_end", val=1.0)

        self.add_subsystem("li", LinearInterpolator(num_nodes=nn, units=None), promotes_outputs=[("vec", "bypass")])
        self.connect("control.bypass_start", "li.start_val")
        self.connect("control.bypass_end", "li.end_val")
        self.add_subsystem(
            "weightpower",
            HeatPumpWeight(),
            promotes_inputs=["power_rating", "specific_power"],
            promotes_outputs=["component_weight"],
        )

        self.add_subsystem(
            "hot_side",
            PerfectHeatTransferComp(num_nodes=nn, specific_heat=spec_heat),
            promotes_inputs=[("T_in", "T_in_hot"), "mdot_coolant"],
        )
        self.add_subsystem(
            "cold_side",
            PerfectHeatTransferComp(num_nodes=nn, specific_heat=spec_heat),
            promotes_inputs=[("T_in", "T_in_cold"), "mdot_coolant"],
        )
        self.add_subsystem(
            "copmatch", COPExplicit(num_nodes=nn), promotes_inputs=["eff_factor"], promotes_outputs=["COP"]
        )
        self.connect("hot_side.T_out", "copmatch.T_h")
        self.connect("cold_side.T_out", "copmatch.T_c")
        self.add_subsystem("heat_pump", COPHeatPump(num_nodes=nn), promotes_inputs=["power_rating", "COP"])
        self.connect("heat_pump.q_in_1", "cold_side.q")
        self.connect("heat_pump.q_in_2", "hot_side.q")
        # Set the default set points and T_in defaults for continuity
        self.set_input_defaults("T_in_hot", val=400.0 * nn_ones, units="K")
        self.set_input_defaults("T_in_cold", val=400.0 * nn_ones, units="K")

        self.add_subsystem(
            "bypass_comp",
            LinearSelector(num_nodes=nn),
            promotes_inputs=["T_in_cold", "T_in_hot", "power_rating"],
            promotes_outputs=["T_out_cold", "T_out_hot", "elec_load"],
        )
        self.connect("hot_side.T_out", "bypass_comp.T_out_refrig_hot")
        self.connect("cold_side.T_out", "bypass_comp.T_out_refrig_cold")
        self.connect("bypass", "bypass_comp.bypass")


class COPExplicit(om.ExplicitComponent):
    """
    Computes "soft" coefficient of performance (COP) that
    doesn't blow up as T_h - T_c approaches zero

    Inputs
    ------
    T_c : float
        Cold side temperature (vector, K)
    T_h : float
        Hot side temperature (vector, K)
    eff_factor : float
        Efficiency factor (scalar, None)

    Outputs
    -------
    COP : float
        Coefficient of performance (vector, None)

    Options
    -------
    num_nodes : int
        The number of analysis points to run
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1)

    def setup(self):
        nn = self.options["num_nodes"]

        self.add_input("T_c", val=300.0, units="K", shape=(nn,))
        self.add_input("T_h", val=400.0, units="K", shape=(nn,))
        self.add_input("eff_factor", units=None, val=0.4)

        self.add_output("COP", units=None, shape=(nn,), val=0.0)

        self.declare_partials(["COP"], ["T_c", "T_h", "eff_factor"], method="cs")

    def compute(self, inputs, outputs):
        delta_T = inputs["T_h"] - inputs["T_c"]
        COP_raw = inputs["T_c"] / (delta_T)
        alpha = -1.5
        a = COP_raw * np.tanh(delta_T) * inputs["eff_factor"] + (1 + np.tanh(-(delta_T + 3))) / 2 * 10
        b = 10.0
        COP_soft = (a * np.exp(alpha * a) + b * np.exp(alpha * b)) / (np.exp(alpha * a) + np.exp(alpha * b))
        outputs["COP"] = COP_soft
