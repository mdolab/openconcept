import numpy as np
from openmdao.api import ExplicitComponent


class SimpleGenerator(ExplicitComponent):
    """
    A simple generator which transforms shaft power into electrical power.

    Inputs
    ------
    shaft_power_in : float
        Shaft power in to the generator (vector, W)
    elec_power_rating: float
        Electric (not mech) design power (scalar, W)

    Outputs
    -------
    elec_power_out : float
        Electric power produced by the generator (vector, W)
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
        self.options.declare("num_nodes", default=1, desc="Number of flight/control conditions")

        # define technology factors
        self.options.declare("efficiency", default=1.0, desc="Efficiency (dimensionless)")
        self.options.declare("weight_inc", default=1 / 5000, desc="kg/W")
        self.options.declare("weight_base", default=0.0, desc="kg base weight")
        self.options.declare("cost_inc", default=100.0 / 745.0, desc="$ cost per watt")
        self.options.declare("cost_base", default=1.0, desc="$ cost base")

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("shaft_power_in", units="W", desc="Input shaft power", shape=(nn,))
        self.add_input("elec_power_rating", units="W", desc="Rated output power")

        # outputs and partials
        eta_g = self.options["efficiency"]
        weight_inc = self.options["weight_inc"]
        cost_inc = self.options["cost_inc"]

        self.add_output("elec_power_out", units="W", desc="Output electric power", shape=(nn,))
        self.add_output("heat_out", units="W", desc="Waste heat out", shape=(nn,))
        self.add_output("component_cost", units="USD", desc="Generator component cost")
        self.add_output("component_weight", units="kg", desc="Generator component weight")
        self.add_output("component_sizing_margin", desc="Fraction of rated power", shape=(nn,))

        self.declare_partials(
            "elec_power_out", "shaft_power_in", val=eta_g * np.ones(nn), rows=range(nn), cols=range(nn)
        )
        self.declare_partials(
            "heat_out", "shaft_power_in", val=(1 - eta_g) * np.ones(nn), rows=range(nn), cols=range(nn)
        )
        self.declare_partials("component_cost", "elec_power_rating", val=cost_inc)
        self.declare_partials("component_weight", "elec_power_rating", val=weight_inc)
        self.declare_partials("component_sizing_margin", "shaft_power_in", rows=range(nn), cols=range(nn))
        self.declare_partials("component_sizing_margin", "elec_power_rating")

    def compute(self, inputs, outputs):
        eta_g = self.options["efficiency"]
        weight_inc = self.options["weight_inc"]
        weight_base = self.options["weight_base"]
        cost_inc = self.options["cost_inc"]
        cost_base = self.options["cost_base"]

        outputs["elec_power_out"] = inputs["shaft_power_in"] * eta_g
        outputs["heat_out"] = inputs["shaft_power_in"] * (1 - eta_g)
        outputs["component_cost"] = inputs["elec_power_rating"] * cost_inc + cost_base
        outputs["component_weight"] = inputs["elec_power_rating"] * weight_inc + weight_base
        outputs["component_sizing_margin"] = inputs["shaft_power_in"] * eta_g / inputs["elec_power_rating"]

    def compute_partials(self, inputs, J):
        eta_g = self.options["efficiency"]
        J["component_sizing_margin", "shaft_power_in"] = eta_g / inputs["elec_power_rating"]
        J["component_sizing_margin", "elec_power_rating"] = -(
            eta_g * inputs["shaft_power_in"] / inputs["elec_power_rating"] ** 2
        )
