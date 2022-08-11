from openmdao.api import ExplicitComponent
import numpy as np


class DVLabel(ExplicitComponent):
    """
    Helper component that is needed when variables must be passed directly from
    input to output of an element with no other component in between.

    This component is adapted from Justin Gray's pyCycle software.

    Inputs
    ------
    Inputs to this component are set upon initialization.

    Outputs
    -------
    Outputs from this component are set upon initialization.

    Options
    -------
    vars_list : iterable
        A list of lists. One outer list entry per variable.
        *Format:* [['input name','output name','val','units']]

    """

    def __init__(self, vars_list):
        super(DVLabel, self).__init__()
        self.vars_list = vars_list
        for i, var_list in enumerate(self.vars_list):
            val = var_list[2]
            if isinstance(val, (float, int)) or np.isscalar(val):
                size = 1
            else:
                size = np.prod(val.shape)
            self.vars_list[i].append(size)

    def setup(self):
        for var_list in self.vars_list:
            i_var = var_list[0]
            o_var = var_list[1]
            val = var_list[2]
            units = var_list[3]
            size = var_list[4]
            if units is None:
                self.add_input(i_var, val)
                self.add_output(o_var, val)
            else:
                self.add_input(i_var, val, units=units)
                self.add_output(o_var, val, units=units)
            # partial derivs setup
            row_col = np.arange(size)
            self.declare_partials(of=o_var, wrt=i_var, val=np.ones(size), rows=row_col, cols=row_col)

    def compute(self, inputs, outputs):
        for var_list in self.vars_list:
            i_var = var_list[0]
            o_var = var_list[1]
            outputs[o_var] = inputs[i_var]

    def compute_partials(self, inputs, J):
        pass
