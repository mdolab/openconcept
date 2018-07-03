from openmdao.api import ExplicitComponent
import numpy as np
from six import string_types

class Sum(ExplicitComponent):
    """Calculates the total of a vector quantity and provides partials
    inputs: any number of vectors vector of length nn (name them in the options)
    outputs: any number of scalars (name them in the options, same number as inputs)
    Specify a string for single inputs / outputs / units or a list/set/etc for multiple
    num_nodes is same for all inputs
    options: num_nodes, input_names, output_names, units, scaling_factors
    """

    def initialize(self):
        self.options.declare('num_nodes',default=1, desc="Length of input vector")
        self.options.declare('input_names', desc="Name of the input vector")
        self.options.declare('units',default=None,  desc='Units of the summation quantity')
        self.options.declare('output_names', desc="Name of output total quantity")
        self.options.declare('scaling_factors',default=1, desc='Scale the totals by a number (e.g. flip positive to negative')


    def setup(self):
        nn = self.options['num_nodes']
        input_names = self.options['input_names']
        output_names = self.options['output_names']
        units = self.options['units']
        scaling_factors = self.options['scaling_factors']
        if not isinstance(input_names, string_types):
            #multiple inputs and outputs, need to handle the default (singluar) options
            if units is None:
                units = [None for i in range(len(input_names))]
            if scaling_factors == 1:
                scaling_factors = [1 for i in range(len(input_names))]

        if isinstance(input_names, string_types) and isinstance(output_names, string_types):
            #only one input/output/sum to calculate
            input_names = [input_names]
            output_names = [output_names]
            units=[units]
            scaling_factors = [scaling_factors]

        elif isinstance(input_names, string_types) or isinstance(output_names, string_types):
            raise ValueError('Need to provide either one set of i/os or the same number of inputs and outputs')
        elif not(len(input_names) == len(output_names) == len(units) == len(scaling_factors)):
            raise ValueError('The input names, output names, scaling factors (if specified) and units (if specified) all need to be the same length')

        for i, input_name in enumerate(input_names):
            self.add_input(input_name, units=units[i], shape=(nn,))
            self.add_output(output_names[i], units=units[i])
            self.declare_partials([output_names[i]], [input_name], val=scaling_factors[i]*np.ones((1,nn)))

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        input_names = self.options['input_names']
        output_names = self.options['output_names']
        scaling_factors = self.options['scaling_factors']
        if not isinstance(input_names, string_types):
            #multiple inputs and outputs, need to handle the default (singluar) options
            if scaling_factors == 1:
                scaling_factors = [1 for i in range(len(input_names))]

        if isinstance(input_names, string_types) and isinstance(output_names, string_types):
            #only one input/output/sum to calculate
            input_names = [input_names]
            output_names = [output_names]
            scaling_factors = [scaling_factors]


        for i, input_name in enumerate(input_names):
            outputs[output_names[i]] = np.sum(inputs[input_name])*scaling_factors[i]
