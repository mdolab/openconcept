import openmdao.api as om
import numpy as np


class SelectorComp(om.ExplicitComponent):
    """
    Selects an output from the set of user-specified inputs
    based on the selector input at runtime.

    For example, if the inputs argument is ['A', 'B', 'C'], then a selector value of 0
    would return input 'A', a selector value of 1 would return input 'B', and a selector
    value of 2 would return input 'C'.

    In practice, the inputs may be vectors. Suppose 'A' has the value [10, 10, 10, 10],
    'B' has the value [5, 5, 5, 5], and 'C' has the value [7, 7, 7, 7], then a selector
    of [0, 1, 2, 1] would make the 'result' output take on the value [10, 5, 7, 5].

    If the selector value is out of range, a warning is raised and zero is returned
    where the selector value is invalid.

    Inputs
    ------
    selector : int
        Selects which input to route to the output based on the order they were specified; must be
        in the range [0, # of inputs) and if the data type is not already an int, it is rounded to
        the nearest integer value (vector, default 0)
    user defined inputs : any
        The data inputs must be specified by the user using the input_names option
        and all inputs must have the same units, if none are specified error is raised (vector)

    Outputs
    -------
    result : same as selected input
        The same value as the input selected by the selector input (vector)

    Options
    -------
    num_nodes : int
        Length of all inputs and outputs (scalar, default 1)
    input_names : iterable of strings
        List of the names of the the user-specified inputs
    units : string
        OpenMDAO-style units of the inputs; all inputs should have these units
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Length of all input and output arrays")
        self.options.declare("input_names", default=[], desc="List of input names")
        self.options.declare("units", default=None, desc="Units of inputs (should all be the same units)")

    def setup(self):
        nn = self.options["num_nodes"]
        names = list(self.options["input_names"])
        unit = self.options["units"]

        # Add user-specified inputs
        if len(names) < 1:
            raise ValueError("input_names option must have at least one input name")
        for name in names:
            self.add_input(name, shape=(nn,), units=unit)

        self.add_input("selector", np.zeros(nn, dtype=int), shape=(nn,))
        self.add_output("result", shape=(nn,), units=unit)

        arange = np.arange(0, nn)
        self.declare_partials("result", names, rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        input_names = list(self.options["input_names"])
        num_inputs = len(input_names)
        nn = self.options["num_nodes"]

        outputs["result"] = np.zeros((nn,))
        selector = np.around(inputs["selector"])

        if np.any(selector < 0) or np.any(selector >= num_inputs):
            raise RuntimeWarning("selector input values must be in the range [0, # of inputs)")

        for i_input in range(num_inputs):
            mask = np.where(selector == i_input, 1, 0)
            outputs["result"] += inputs[input_names[i_input]] * mask

    def compute_partials(self, inputs, J):
        input_names = list(self.options["input_names"])

        selector = np.around(inputs["selector"])

        for i_input, input_name in enumerate(input_names):
            J["result", input_name] = np.where(selector == i_input, 1, 0)
