import numpy as np
from openmdao.api import ExplicitComponent


class PowerSplit(ExplicitComponent):
    """
    A power split mechanism for mechanical or electrical power.

    Inputs
    ------
    power_in : float
        Power fed to the splitter. (vector, W)
    power_rating : float
        Maximum rated power of the split mechanism. (scalar, W)
    power_split_fraction:
        If ``'rule'`` is set to ``'fraction'``, sets percentage of input power directed
        to Output A (minus losses). (vector, dimensionless)
    power_split_amount:
        If ``'rule'`` is set to ``'fixed'``, sets amount of input power to Output A (minus
        losses). (vector, W)

    Outputs
    -------
    power_out_A : float
        Power sent to first output (vector, W)
    power_out_B : float
        Power sent to second output (vector, W)
    heat_out : float
        Waste heat produced (vector, W)
    component_cost : float
        Nonrecurring cost of the component (scalar, USD)
    component_weight : float
        Weight of the component (scalar, kg)
    component_sizing_margin : float
        Equal to 1 when fed full rated power (vector, dimensionless)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length; default 1)
    rule : str
        Power split control rule to use; either ``'fixed'`` where a set
        amount of power is sent to Output A or ``'fraction'`` where a
        fraction of the total power is sent to Output A
    efficiency : float
        Component efficiency (default 1)
    weight_inc : float
        Weight per unit rated power
        (default 0, kg/W)
    weight_base : float
        Base weight
        (default 0, kg)
    cost_inc : float
        Nonrecurring cost per unit power
        (default 0, USD/W)
    cost_base : float
        Base cost
        (default 0 USD)
    """

    def initialize(self):
        # define control rules
        self.options.declare("num_nodes", default=1, desc="Number of flight/control conditions")
        self.options.declare("rule", default="fraction", desc="Control strategy - fraction or fixed power")

        self.options.declare("efficiency", default=1.0, desc="Efficiency (dimensionless)")
        self.options.declare("weight_inc", default=0.0, desc="kg per input watt")
        self.options.declare("weight_base", default=0.0, desc="kg base weight")
        self.options.declare("cost_inc", default=0.0, desc="$ cost per input watt")
        self.options.declare("cost_base", default=0.0, desc="$ cost base")

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("power_in", units="W", desc="Input shaft power or incoming electrical load", shape=(nn,))
        self.add_input("power_rating", val=99999999, units="W", desc="Split mechanism power rating")

        rule = self.options["rule"]
        if rule == "fraction":
            self.add_input("power_split_fraction", val=0.5, desc="Fraction of power to output A", shape=(nn,))
        elif rule == "fixed":
            self.add_input("power_split_amount", units="W", desc="Raw amount of power to output A", shape=(nn,))
        else:
            msg = 'Specify either "fraction" or "fixed" as power split control rule'
            raise ValueError(msg)

        eta = self.options["efficiency"]
        weight_inc = self.options["weight_inc"]
        cost_inc = self.options["cost_inc"]

        self.add_output("power_out_A", units="W", desc="Output power or load to A", shape=(nn,))
        self.add_output("power_out_B", units="W", desc="Output power or load to B", shape=(nn,))
        self.add_output("heat_out", units="W", desc="Waste heat out", shape=(nn,))
        self.add_output("component_cost", units="USD", desc="Splitter component cost")
        self.add_output("component_weight", units="kg", desc="Splitter component weight")
        self.add_output("component_sizing_margin", desc="Fraction of rated power", shape=(nn,))

        if rule == "fraction":
            self.declare_partials(
                ["power_out_A", "power_out_B"], ["power_in", "power_split_fraction"], rows=range(nn), cols=range(nn)
            )
        elif rule == "fixed":
            self.declare_partials(
                ["power_out_A", "power_out_B"], ["power_in", "power_split_amount"], rows=range(nn), cols=range(nn)
            )
        self.declare_partials("heat_out", "power_in", val=(1 - eta) * np.ones(nn), rows=range(nn), cols=range(nn))
        self.declare_partials("component_cost", "power_rating", val=cost_inc)
        self.declare_partials("component_weight", "power_rating", val=weight_inc)
        self.declare_partials("component_sizing_margin", "power_in", rows=range(nn), cols=range(nn))
        self.declare_partials("component_sizing_margin", "power_rating")

    def compute(self, inputs, outputs):
        nn = self.options["num_nodes"]
        rule = self.options["rule"]
        eta = self.options["efficiency"]
        weight_inc = self.options["weight_inc"]
        weight_base = self.options["weight_base"]
        cost_inc = self.options["cost_inc"]
        cost_base = self.options["cost_base"]

        if rule == "fraction":
            outputs["power_out_A"] = inputs["power_in"] * inputs["power_split_fraction"] * eta
            outputs["power_out_B"] = inputs["power_in"] * (1 - inputs["power_split_fraction"]) * eta
        elif rule == "fixed":
            # check to make sure enough power is available
            # if inputs['power_in'] < inputs['power_split_amount']:
            not_enough_idx = np.where(inputs["power_in"] < inputs["power_split_amount"])
            po_A = np.zeros(nn)
            po_B = np.zeros(nn)
            po_A[not_enough_idx] = inputs["power_in"][not_enough_idx] * eta
            po_B[not_enough_idx] = np.zeros(nn)[not_enough_idx]
            # else:
            enough_idx = np.where(inputs["power_in"] >= inputs["power_split_amount"])
            po_A[enough_idx] = inputs["power_split_amount"][enough_idx] * eta
            po_B[enough_idx] = (inputs["power_in"][enough_idx] - inputs["power_split_amount"][enough_idx]) * eta
            outputs["power_out_A"] = po_A
            outputs["power_out_B"] = po_B
        outputs["heat_out"] = inputs["power_in"] * (1 - eta)
        outputs["component_cost"] = inputs["power_rating"] * cost_inc + cost_base
        outputs["component_weight"] = inputs["power_rating"] * weight_inc + weight_base
        outputs["component_sizing_margin"] = inputs["power_in"] / inputs["power_rating"]

    def compute_partials(self, inputs, J):
        nn = self.options["num_nodes"]
        rule = self.options["rule"]
        eta = self.options["efficiency"]
        if rule == "fraction":
            J["power_out_A", "power_in"] = inputs["power_split_fraction"] * eta
            J["power_out_A", "power_split_fraction"] = inputs["power_in"] * eta
            J["power_out_B", "power_in"] = (1 - inputs["power_split_fraction"]) * eta
            J["power_out_B", "power_split_fraction"] = -inputs["power_in"] * eta
        elif rule == "fixed":
            not_enough_idx = np.where(inputs["power_in"] < inputs["power_split_amount"])
            enough_idx = np.where(inputs["power_in"] >= inputs["power_split_amount"])
            # if inputs['power_in'] < inputs['power_split_amount']:
            Jpo_A_pi = np.zeros(nn)
            Jpo_A_ps = np.zeros(nn)
            Jpo_B_pi = np.zeros(nn)
            Jpo_B_ps = np.zeros(nn)
            Jpo_A_pi[not_enough_idx] = eta * np.ones(nn)[not_enough_idx]
            Jpo_A_ps[not_enough_idx] = np.zeros(nn)[not_enough_idx]
            Jpo_B_pi[not_enough_idx] = np.zeros(nn)[not_enough_idx]
            Jpo_B_ps[not_enough_idx] = np.zeros(nn)[not_enough_idx]
            # else:
            Jpo_A_ps[enough_idx] = eta * np.ones(nn)[enough_idx]
            Jpo_A_pi[enough_idx] = np.zeros(nn)[enough_idx]
            Jpo_B_ps[enough_idx] = -eta * np.ones(nn)[enough_idx]
            Jpo_B_pi[enough_idx] = eta * np.ones(nn)[enough_idx]
            J["power_out_A", "power_in"] = Jpo_A_pi
            J["power_out_A", "power_split_amount"] = Jpo_A_ps
            J["power_out_B", "power_in"] = Jpo_B_pi
            J["power_out_B", "power_split_amount"] = Jpo_B_ps
        J["component_sizing_margin", "power_in"] = 1 / inputs["power_rating"]
        J["component_sizing_margin", "power_rating"] = -(inputs["power_in"] / inputs["power_rating"] ** 2)
