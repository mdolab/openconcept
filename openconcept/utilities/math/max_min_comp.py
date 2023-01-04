import openmdao.api as om
import numpy as np


class MaxComp(om.ExplicitComponent):
    """
    Takes in a vector and outputs a scalar that is the value of the maximum element in the input.

    Inputs
    ------
    array : any type that supports comparison (<, >, etc.)
        Array of which the maximum is found (vector)

    Outputs
    -------
    max : same as data type of input array
        The maximum value of the input array (scalar)

    Options
    -------
    num_nodes : int
        Length of all inputs and outputs (scalar, default 1)
    units : string
        OpenMDAO-style units of input and output
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Length of all input and output arrays")
        self.options.declare("units", default=None, desc="Units of input array")

    def setup(self):
        nn = self.options["num_nodes"]
        unit = self.options["units"]

        self.add_input("array", shape=(nn,), units=unit)
        self.add_output("max", units=unit)

        self.declare_partials("max", "array", rows=np.zeros(nn), cols=np.arange(0, nn))

    def compute(self, inputs, outputs):
        outputs["max"] = np.amax(inputs["array"])

    def compute_partials(self, inputs, J):
        J["max", "array"] = np.where(inputs["array"] == np.amax(inputs["array"]), 1, 0)


class MinComp(om.ExplicitComponent):
    """
    Takes in a vector and outputs a scalar that is the value of the minimum element in the input.

    Inputs
    ------
    array : any type that supports comparison (<, >, etc.)
        Array of which the minimum is found (vector)

    Outputs
    -------
    min : same as data type of input array
        The minimum value of the input array (scalar)

    Options
    -------
    num_nodes : int
        Length of all inputs and outputs (scalar, default 1)
    units : string
        OpenMDAO-style units of input and output
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Length of all input and output arrays")
        self.options.declare("units", default=None, desc="Units of input array")

    def setup(self):
        nn = self.options["num_nodes"]
        unit = self.options["units"]

        self.add_input("array", shape=(nn,), units=unit)
        self.add_output("min", units=unit)

        self.declare_partials("min", "array", rows=np.zeros(nn), cols=np.arange(0, nn))

    def compute(self, inputs, outputs):
        print(inputs["array"])
        outputs["min"] = np.amin(inputs["array"])
        print(outputs["min"])

    def compute_partials(self, inputs, J):
        J["min", "array"] = np.where(inputs["array"] == np.amin(inputs["array"]), 1, 0)
