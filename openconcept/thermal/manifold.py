import numpy as np
from openmdao.api import ExplicitComponent


class FlowSplit(ExplicitComponent):
    """
    Split incoming flow from one inlet into two outlets at a fractional ratio.

    Inputs
    ------
    mdot_in : float
        Mass flow rate of incoming fluid (vector, kg/s)
    mdot_split_fraction : float
        Fraction of incoming mass flow directed to output A, must be in
        range 0-1 inclusive (vector, dimensionless)

    Outputs
    -------
    mdot_out_A : float
        Mass flow rate directed to first output (vector, kg/s)
    mdot_out_B : float
        Mass flow rate directed to second output (vector, kg/s)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length; default 1)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of analysis points")

    def setup(self):
        nn = self.options["num_nodes"]
        rng = np.arange(0, nn)

        self.add_input("mdot_in", units="kg/s", shape=(nn,))
        self.add_input("mdot_split_fraction", units=None, shape=(nn,), val=0.5)

        self.add_output("mdot_out_A", units="kg/s", shape=(nn,))
        self.add_output("mdot_out_B", units="kg/s", shape=(nn,))

        self.declare_partials(["mdot_out_A"], ["mdot_in", "mdot_split_fraction"], rows=rng, cols=rng)
        self.declare_partials(["mdot_out_B"], ["mdot_in", "mdot_split_fraction"], rows=rng, cols=rng)

    def compute(self, inputs, outputs):
        if np.any(inputs["mdot_split_fraction"] < 0) or np.any(inputs["mdot_split_fraction"] > 1):
            raise RuntimeWarning(
                f"mdot_split_fraction of {inputs['mdot_split_fraction']} has at least one element out of range [0, 1]"
            )
        outputs["mdot_out_A"] = inputs["mdot_in"] * inputs["mdot_split_fraction"]
        outputs["mdot_out_B"] = inputs["mdot_in"] * (1 - inputs["mdot_split_fraction"])

    def compute_partials(self, inputs, J):
        J["mdot_out_A", "mdot_in"] = inputs["mdot_split_fraction"]
        J["mdot_out_A", "mdot_split_fraction"] = inputs["mdot_in"]

        J["mdot_out_B", "mdot_in"] = 1 - inputs["mdot_split_fraction"]
        J["mdot_out_B", "mdot_split_fraction"] = -inputs["mdot_in"]


class FlowCombine(ExplicitComponent):
    """
    Combines two incoming flows into a single outgoing flow and does a weighted average
    of their temperatures based on the mass flow rate of each to compute the outlet temp.

    Inputs
    ------
    mdot_in_A : float
        Mass flow rate of fluid from first inlet, should be nonegative (vector, kg/s)
    mdot_in_B : float
        Mass flow rate of fluid from second inlet, should be nonnegative (vector, kg/s)
    T_in_A : float
        Temperature of fluid from first inlet (vector, K)
    T_in_B : float
        Temperature of fluid from second inlet (vector, K)

    Outputs
    -------
    mdot_out : float
        Outgoing fluid mass flow rate (vector, kg/s)
    T_out : float
        Outgoing fluid temperature (vector, K)

    Options
    -------
    num_nodes : int
        Number of analysis points (scalar, default 1)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of analysis points")

    def setup(self):
        nn = self.options["num_nodes"]
        rng = np.arange(0, nn)

        self.add_input("mdot_in_A", units="kg/s", shape=(nn,))
        self.add_input("mdot_in_B", units="kg/s", shape=(nn,))
        self.add_input("T_in_A", units="K", shape=(nn,))
        self.add_input("T_in_B", units="K", shape=(nn,))

        self.add_output("mdot_out", units="kg/s", shape=(nn,))
        self.add_output("T_out", units="K", shape=(nn,))

        self.declare_partials(["mdot_out"], ["mdot_in_A", "mdot_in_B"], rows=rng, cols=rng)
        self.declare_partials(["T_out"], ["mdot_in_A", "mdot_in_B", "T_in_A", "T_in_B"], rows=rng, cols=rng)

    def compute(self, inputs, outputs):
        mdot_A = inputs["mdot_in_A"]
        mdot_B = inputs["mdot_in_B"]
        outputs["mdot_out"] = mdot_A + mdot_B
        # Weighted average of temperatures for output temperature
        outputs["T_out"] = (mdot_A * inputs["T_in_A"] + mdot_B * inputs["T_in_B"]) / (mdot_A + mdot_B)

    def compute_partials(self, inputs, J):
        nn = self.options["num_nodes"]
        J["mdot_out", "mdot_in_A"] = np.ones((nn,))
        J["mdot_out", "mdot_in_B"] = np.ones((nn,))

        mdot_A = inputs["mdot_in_A"]
        mdot_B = inputs["mdot_in_B"]
        mdot = mdot_A + mdot_B
        T_A = inputs["T_in_A"]
        T_B = inputs["T_in_B"]
        J["T_out", "mdot_in_A"] = (mdot * T_A - mdot_A * T_A - mdot_B * T_B) / (mdot**2)
        J["T_out", "mdot_in_B"] = (mdot * T_B - mdot_A * T_A - mdot_B * T_B) / (mdot**2)
        J["T_out", "T_in_A"] = mdot_A / mdot
        J["T_out", "T_in_B"] = mdot_B / mdot
