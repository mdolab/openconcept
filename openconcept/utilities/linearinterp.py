from openmdao.api import ExplicitComponent
import numpy as np


class LinearInterpolator(ExplicitComponent):
    """
    Create a linearly interpolated set of points **including** two end points

    Inputs
    ------
    start_val : float
        Starting value (scalar; units set from "units" option)
    end_val : float
        Ending value (scalar; units set from "units" option)

    Outputs
    -------
    vec : float
        Vector of linearly interpolated points (scalar; units set from "units" opt)

    Options
    -------
    units : str, None
        Units for inputs and outputs
    num_nodes : int
        Number of linearly interpolated points to produce (minimum/default 2)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=2, desc="Number of nodes")
        self.options.declare("units", default=None, desc="Units")

    def setup(self):
        nn = self.options["num_nodes"]
        units = self.options["units"]
        self.add_input("start_val", units=units)
        self.add_input("end_val", units=units)
        self.add_output("vec", units=units, shape=(nn,))
        arange = np.arange(0, nn)
        self.declare_partials("vec", "start_val", rows=arange, cols=np.zeros(nn), val=np.linspace(1, 0, nn))
        self.declare_partials("vec", "end_val", rows=arange, cols=np.zeros(nn), val=np.linspace(0, 1, nn))

    def compute(self, inputs, outputs):
        nn = self.options["num_nodes"]
        outputs["vec"] = np.linspace(inputs["start_val"], inputs["end_val"], nn).reshape((nn,))
