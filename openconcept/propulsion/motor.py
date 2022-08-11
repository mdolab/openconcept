import numpy as np
from openmdao.api import ExplicitComponent


class SimpleMotor(ExplicitComponent):
    """
    A simple motor which creates shaft power and draws electrical load.

    Inputs
    ------
    throttle : float
        Power control setting. Should be [0, 1]. (vector, dimensionless)
    elec_power_rating: float
        Electric (not mech) design power. (scalar, W)

    Outputs
    -------
    shaft_power_out : float
        Shaft power output from motor (vector, W)
    elec_load : float
        Electrical load consumed by motor (vector, W)
    heat_out : float
        Waste heat produced (vector, W)
    component_cost : float
        Nonrecurring cost of the component (scalar, USD)
    component_weight : float
        Weight of the component (scalar, kg)
    component_sizing_margin : float
        Equal to 1 when producing full rated power (vector, dimensionless)


    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length; default 1)
    efficiency : float
        Shaft power efficiency. Sensible range 0.0 to 1.0 (default 1)
    weight_inc : float
        Weight per unit rated power (default 1/5000, kg/W)
    weight_base : float
        Base weight (default 0, kg)
    cost_inc : float
        Cost per unit rated power (default 0.134228, USD/W)
    cost_base : float
        Base cost (default 1 USD) B
    """

    def initialize(self):
        # define technology factors
        self.options.declare("num_nodes", default=1, desc="Number of flight/control conditions")
        self.options.declare("efficiency", default=1.0, desc="Efficiency (dimensionless)")
        self.options.declare("weight_inc", default=1 / 5000, desc="kg/W")  # 5kW/kg
        self.options.declare("weight_base", default=0.0, desc="kg base weight")
        self.options.declare("cost_inc", default=100 / 745, desc="$ cost per watt")
        self.options.declare("cost_base", default=1.0, desc="$ cost base")

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("throttle", desc="Throttle input (Fractional)", shape=(nn,))
        self.add_input("elec_power_rating", units="W", desc="Rated electrical power (load)")

        # outputs and partials
        weight_inc = self.options["weight_inc"]
        cost_inc = self.options["cost_inc"]

        self.add_output("shaft_power_out", units="W", desc="Output shaft power", shape=(nn,))
        self.add_output("heat_out", units="W", desc="Waste heat out", shape=(nn,))
        self.add_output("elec_load", units="W", desc="Electrical load consumed", shape=(nn,))
        self.add_output("component_cost", units="USD", desc="Motor component cost")
        self.add_output("component_weight", units="kg", desc="Motor component weight")
        self.add_output("component_sizing_margin", desc="Fraction of rated power", shape=(nn,))
        self.declare_partials("shaft_power_out", "elec_power_rating")
        self.declare_partials("shaft_power_out", "throttle", rows=range(nn), cols=range(nn))
        self.declare_partials("heat_out", "elec_power_rating")
        self.declare_partials("heat_out", "throttle", "elec_power_rating", rows=range(nn), cols=range(nn))
        self.declare_partials("elec_load", "elec_power_rating")
        self.declare_partials("elec_load", "throttle", rows=range(nn), cols=range(nn))
        self.declare_partials("component_cost", "elec_power_rating", val=cost_inc)
        self.declare_partials("component_weight", "elec_power_rating", val=weight_inc)
        self.declare_partials(
            "component_sizing_margin", "throttle", val=1.0 * np.ones(nn), rows=range(nn), cols=range(nn)
        )

    def compute(self, inputs, outputs):
        eta_m = self.options["efficiency"]
        weight_inc = self.options["weight_inc"]
        weight_base = self.options["weight_base"]
        cost_inc = self.options["cost_inc"]
        cost_base = self.options["cost_base"]
        outputs["shaft_power_out"] = inputs["throttle"] * inputs["elec_power_rating"] * eta_m
        outputs["heat_out"] = inputs["throttle"] * inputs["elec_power_rating"] * (1 - eta_m)
        outputs["elec_load"] = inputs["throttle"] * inputs["elec_power_rating"]
        outputs["component_cost"] = inputs["elec_power_rating"] * cost_inc + cost_base
        outputs["component_weight"] = inputs["elec_power_rating"] * weight_inc + weight_base
        outputs["component_sizing_margin"] = inputs["throttle"]

    def compute_partials(self, inputs, J):
        eta_m = self.options["efficiency"]
        J["shaft_power_out", "throttle"] = inputs["elec_power_rating"] * eta_m
        J["shaft_power_out", "elec_power_rating"] = inputs["throttle"] * eta_m
        J["heat_out", "throttle"] = inputs["elec_power_rating"] * (1 - eta_m)
        J["heat_out", "elec_power_rating"] = inputs["throttle"] * (1 - eta_m)
        J["elec_load", "throttle"] = inputs["elec_power_rating"]
        J["elec_load", "elec_power_rating"] = inputs["throttle"]
