"""Definition of the Element Multiply Component."""

import collections
import numpy as np
from scipy import sparse as sp
from six import string_types

from openmdao.core.explicitcomponent import ExplicitComponent


class ElementMultiplyDivideComp(ExplicitComponent):
    r"""
    Compute a vectorized element-wise multiplication and/or division.

    Use the add_equation method to define any number of mult/div relations
    User defines the names of the input and output variables using
    add_equation(output_name='my_output', input_names=['a','b', 'c', ...],
        divide=[False,False,True,...])

    .. math::
        result = (a * b / c ....) * \textrm{scaling factor}

    where all inputs  shape (vec_size, n)
          b is of shape (vec_size, n)
          c is of shape (vec_size, n)
          result is of shape (vec_size, n)

    All input vectors must be of the same shape, specified by the options 'vec_size' and 'length'.
    Use scaling factor -1 for subtraction.

    Attributes
    ----------
    _add_systems : list
    List of equation systems to be initialized with the system.
    """

    def __init__(self, output_name=None, input_names=None, vec_size=1, length=1,
                 val=1.0, scaling_factor=1, divide=None, input_units=None, **kwargs):
        """
        Allow user to create an multiplication system with one-liner.

        Parameters
        ----------
        output_name : str
            (required) name of the result variable in this component's namespace.
        input_names : iterable of str
            (required) names of the input variables for this system
        vec_size : int
            Length of the first dimension of the input and output vectors
            (i.e number of rows, or vector length for a 1D vector)
            Default is 1
        length : int
            Length of the second dimension of the input and ouptut vectors (i.e. number of columns)
            Default is 1 which results in input/output vectors of size (vec_size,)
        scaling_factor :  numeric
            Scaling factor to apply to the whole system
            Default is 1
        divide : iterable of bool or None
            True to use division operator, False to use mult operator
            Default is None which results in mult of every input
            Length is same as number of inputs
        val : float or list or tuple or ndarray
            The initial value of the variable being added in user-defined units. Default is 1.0.
        input_units : iterable of str
            Units for each of the input vectors in order.
            Output units will be dimensionally consistent.
        **kwargs : str
            Any other arguments to pass to the addition system
            (same as add_output method for ExplicitComponent)
            Examples include units (str or None), desc (str)
        """
        complexify = kwargs.pop('complex', False)
        super(ElementMultiplyDivideComp, self).__init__(complex=complexify)

        self._add_systems = []

        if isinstance(output_name, string_types):
            self._add_systems.append((output_name, input_names, vec_size, length, val,
                                      scaling_factor, divide, input_units, kwargs))
        elif isinstance(output_name, collections.Iterable):
            raise NotImplementedError('Declaring multiple systems '
                                      'on initiation is not implemented.'
                                      'Use a string to name a single addition relationship or use '
                                      'multiple add_equation calls')
        elif output_name is None:
            pass
        else:
            raise ValueError(
                "first argument to init must be either of type "
                "`str' or 'None'")

    def initialize(self):
        """
        Declare options.

        Parameters
        ----------
        complex : Boolean
            Set True to enable complex math (e.g. for complex step verification)
        """
        self.options.declare('complex', default=False,
                             desc="Allocate as complex (e.g. for complex-step verification)")

    def add_equation(self, output_name, input_names, vec_size=1, length=1, val=1.0,
                     res_units=None, desc='', lower=None, upper=None, ref=1.0,
                     ref0=0.0, res_ref=None, var_set=0, scaling_factor=1,
                     divide=None, input_units=None):
        """
        Add a multiplication relation.

        Parameters
        ----------
        output_name : str
            (required) name of the result variable in this component's namespace.
        input_names : iterable of str
            (required) names of the input variables for this system
        vec_size : int
            Length of the first dimension of the input and output vectors
            (i.e number of rows, or vector length for a 1D vector)
            Default is 1
        length : int
            Length of the second dimension of the input and ouptut vectors (i.e. number of columns)
            Default is 1 which results in input/output vectors of size (vec_size,)
        scaling_factor :  numeric
            Scaling factor to apply to the whole system
            Default is 1
        divide : iterable of bool or None
            True to use division operator, False to use mult operator
            Default is None which results in mult of every input
            Length is same as number of inputs
        val : float or list or tuple or ndarray
            The initial value of the variable being added in user-defined units. Default is 1.0.
        input_units : iterable of str
            Units for each of the input vectors in order.
            Output units will be dimensionally consistent.
        res_units : str or None
            Units in which the residuals of this output will be given to the user when requested.
            Default is None, which means it has no units.
        desc : str
            description of the variable.
        lower : float or list or tuple or ndarray or Iterable or None
            lower bound(s) in user-defined units. It can be (1) a float, (2) an array_like
            consistent with the shape arg (if given), or (3) an array_like matching the shape of
            val, if val is array_like. A value of None means this output has no lower bound.
            Default is None.
        upper : float or list or tuple or ndarray or or Iterable None
            upper bound(s) in user-defined units. It can be (1) a float, (2) an array_like
            consistent with the shape arg (if given), or (3) an array_like matching the shape of
            val, if val is array_like. A value of None means this output has no upper bound.
            Default is None.
        ref : float or ndarray
            Scaling parameter. The value in the user-defined units of this output variable when
            the scaled value is 1. Default is 1.
        ref0 : float or ndarray
            Scaling parameter. The value in the user-defined units of this output variable when
            the scaled value is 0. Default is 0.
        res_ref : float or ndarray
            Scaling parameter. The value in the user-defined res_units of this output's residual
            when the scaled value is 1. Default is 1.
        var_set : hashable object
            For advanced users only. ID or color for this variable, relevant for reconfigurability.
            Default is 0.
        """
        kwargs = {'res_units': res_units, 'desc': desc,
                  'lower': lower, 'upper': upper, 'ref': ref, 'ref0': ref0,
                  'res_ref': res_ref, 'var_set': var_set}
        self._add_systems.append((output_name, input_names, vec_size, length, val,
                                  scaling_factor, divide, input_units, kwargs))

    def add_output(self):
        """
        Use add_equation instead of add_output to define equation systems.
        """
        raise NotImplementedError('Use add_equation method, not add_output method'
                                  'to create an multliplication/division relation')

    def setup(self):
        """
        Set up the addition/subtraction system at run time.
        """
        for (output_name, input_names, vec_size, length, val,
             scaling_factor, divide, input_units, kwargs) in self._add_systems:
            if isinstance(input_names, string_types):
                input_names = [input_names]
            desc = kwargs.get('desc', '')
            var_set = kwargs.get('var_set', 0)

            if divide is None:
                divide = [False for k in range(len(input_names))]
            if input_units is None:
                input_units = [None for k in range(len(input_names))]

            if len(divide) != len(input_names):
                raise ValueError('Division bool list needs to be same length as input names')
            if len(input_units) != len(input_names):
                raise ValueError('Input units list needs to be same length as input names')

            if length == 1:
                shape = (vec_size,)
            else:
                shape = (vec_size, length)

            output_units_assemble = []

            for i, input_name in enumerate(input_names):
                self.add_input(input_name, shape=shape, units=input_units[i],
                               desc=desc + '_inp_' + input_name, var_set=var_set)
                self.declare_partials([output_name], [input_name],
                                      rows=np.arange(0, vec_size * length),
                                      cols=np.arange(0, vec_size * length))
                # derive the units of the output vector from the inputs
                if input_units[i] is not None:
                    if divide[i]:
                        if i == 0:
                            output_units_assemble.append('(' + input_units[i] + ')**-1 ')
                        else:
                            output_units_assemble.append('/ (' + input_units[i] + ') ')
                    else:
                        if i == 0:
                            output_units_assemble.append(input_units[i] + ' ')
                        else:
                            output_units_assemble.append('* (' + input_units[i] + ') ')
                    output_units = ''.join(output_units_assemble)
            if len(output_units_assemble) == 0:
                output_units = None
            super(ElementMultiplyDivideComp, self).add_output(output_name, val,
                                                              shape=shape, units=output_units,
                                                              **kwargs)

    def compute(self, inputs, outputs):
        """
        Compute the element wise multiplication or division of inputs using numpy.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        """
        complexify = self.options['complex']
        for (output_name, input_names, vec_size, length, val, scaling_factor, divide,
             input_units, kwargs) in self._add_systems:
            if isinstance(input_names, string_types):
                input_names = [input_names]

            if divide is None:
                divide = [False for k in range(len(input_names))]
            if length == 1:
                shape = (vec_size,)
            else:
                shape = (vec_size, length)

            if complexify:
                temp = np.ones(shape, dtype=np.complex_)
            else:
                temp = np.ones(shape)

            for i, input_name in enumerate(input_names):
                if divide[i]:
                    temp = temp / inputs[input_name]
                else:
                    temp = temp * inputs[input_name]

            outputs[output_name] = temp * scaling_factor

    def compute_partials(self, inputs, J):
        for (output_name, input_names, vec_size, length, val, scaling_factor, divide,
             input_units, kwargs) in self._add_systems:
            if isinstance(input_names, string_types):
                input_names = [input_names]

            if divide is None:
                divide = [False for k in range(len(input_names))]
            if length == 1:
                shape = (vec_size,)
            else:
                shape = (vec_size, length)

            for j, input_name in enumerate(input_names):
                temp = np.ones(shape)
                for i, input_name_partial in enumerate(input_names):
                    if input_name_partial != input_name:
                        if divide[i]:
                            temp = temp / inputs[input_name_partial]
                        else:
                            temp = temp * inputs[input_name_partial]
                    else:
                        # if i is the differentiated variable
                        if divide[i]:
                            temp = - temp / inputs[input_name_partial] ** 2
                        else:
                            pass
                temp = temp * scaling_factor
                J[output_name, input_name] = temp.flatten()
