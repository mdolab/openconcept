import openmdao.api as om
import numpy as np


class SimplePump(om.ExplicitComponent):
    """
    A pump that circulates coolant against pressure.
    The default parameters are based on a survey of commercial
    airplane fuel pumps of a variety of makes and models.

    Inputs
    ------
    power_rating : float
        Maximum rated electrical power (scalar, W)
    mdot_coolant : float
        Coolant mass flow rate (vector, kg/s)
    rho_coolant : float
        Coolant density (vector, kg/m3)
    delta_p : float
        Pressure rise provided by the pump (vector, kg/s)

    Outputs
    -------
    elec_load : float
        Electricity used by the pump (vector, W)
    component_weight : float
        Pump weight (scalar, kg)
    component_sizing_margin : float
        Fraction of total power rating used via elec_load (vector, dimensionless)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length; default 1)
    efficiency : float
        Pump electrical + mech efficiency. Sensible range 0.0 to 1.0 (default 0.35)
    weight_base : float
        Base weight of pump, doesn't change with power rating (default 0)
    weight_inc : float
        Incremental weight of pump, scales linearly with power rating (default 1/450 kg/W)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of flight/control conditions")
        self.options.declare("efficiency", default=0.35, desc="Efficiency (dimensionless)")
        self.options.declare("weight_base", default=0.0, desc="Pump base weight")
        self.options.declare("weight_inc", default=1 / 450, desc="Incremental pump weight (kg/W)")

    def setup(self):
        nn = self.options["num_nodes"]
        weight_inc = self.options["weight_inc"]

        self.add_input("power_rating", units="W", desc="Pump electrical power rating")
        self.add_input("mdot_coolant", units="kg/s", desc="Coolant mass flow rate", val=np.ones((nn,)))
        self.add_input("delta_p", units="Pa", desc="Pump pressure rise", val=np.ones((nn,)))
        self.add_input("rho_coolant", units="kg/m**3", desc="Coolant density", val=np.ones((nn,)))

        self.add_output("elec_load", units="W", desc="Pump electrical load", val=np.ones((nn,)))
        self.add_output("component_weight", units="kg", desc="Pump weight")
        self.add_output("component_sizing_margin", units=None, val=np.ones((nn,)), desc="Comp sizing margin")

        self.declare_partials(
            ["elec_load", "component_sizing_margin"],
            ["rho_coolant", "delta_p", "mdot_coolant"],
            rows=np.arange(nn),
            cols=np.arange(nn),
        )
        self.declare_partials(["component_sizing_margin"], ["power_rating"], rows=np.arange(nn), cols=np.zeros(nn))
        self.declare_partials(["component_weight"], ["power_rating"], val=weight_inc)

    def compute(self, inputs, outputs):
        eta = self.options["efficiency"]
        weight_inc = self.options["weight_inc"]
        weight_base = self.options["weight_base"]

        outputs["component_weight"] = weight_base + weight_inc * inputs["power_rating"]

        # compute the fluid power
        vol_flow_rate = inputs["mdot_coolant"] / inputs["rho_coolant"]  # m3/s
        fluid_power = vol_flow_rate * inputs["delta_p"]
        outputs["elec_load"] = fluid_power / eta
        outputs["component_sizing_margin"] = outputs["elec_load"] / inputs["power_rating"]

    def compute_partials(self, inputs, J):
        eta = self.options["efficiency"]

        J["elec_load", "mdot_coolant"] = inputs["delta_p"] / inputs["rho_coolant"] / eta
        J["elec_load", "delta_p"] = inputs["mdot_coolant"] / inputs["rho_coolant"] / eta
        J["elec_load", "rho_coolant"] = -inputs["mdot_coolant"] * inputs["delta_p"] / inputs["rho_coolant"] ** 2 / eta
        for in_var in ["mdot_coolant", "delta_p", "rho_coolant"]:
            J["component_sizing_margin", in_var] = J["elec_load", in_var] / inputs["power_rating"]
        J["component_sizing_margin", "power_rating"] = (
            -inputs["mdot_coolant"] * inputs["delta_p"] / inputs["rho_coolant"] / eta / inputs["power_rating"] ** 2
        )
