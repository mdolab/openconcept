"""Definition of the Element Summation Component."""

from collections.abc import Iterable
import numpy as np
from scipy import sparse as sp
from six import string_types

from openmdao.core.explicitcomponent import ExplicitComponent


class SumComp(ExplicitComponent):
    r"""
    Compute a vectorized summation.

    Use the add_equation method to define any number of summations
    User defines the names of the input and output variables using
    add_equation(output_name='my_output', input_name='my_input')

    Use option axis = None to sum over all array elements. Default
    behavior sums along the columns.

    .. math::

        \textrm{result}_j = \sum_{i=1} ^\text{vec_size} a_{ij} * \textrm{scaling factor}

    where
        - a is shape (vec_size, n)
        - b is of shape (vec_size, n)
        - c is of shape (vec_size, n)

    Result is of shape (1, n) or (1, )

    Attributes
    ----------
    _add_systems : list
        List of equation systems to be initialized with the system.
    """

    def __init__(self, output_name=None, input_name=None, vec_size=1, length=1,
                 val=1.0, scaling_factor=1, **kwargs):
        """
        Allow user to create an multiplication system with one-liner.

        Parameters
        ----------
        output_name : str
            (required) name of the result variable in this component's namespace.
        input_name : str
            (required) name of the input variable for this system
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
        val : float or list or tuple or ndarray
            The initial value of the variable being added in user-defined units. Default is 1.0.
        **kwargs : str
            Any other arguments to pass to the addition system
            (same as add_output method for ExplicitComponent)
            Examples include units (str or None), desc (str)
        """
        axis = kwargs.pop('axis', 0)
        super(SumComp, self).__init__(axis=axis)

        self._add_systems = []

        if isinstance(output_name, string_types):
            self._add_systems.append((output_name, input_name, vec_size, length, val,
                                      scaling_factor, kwargs))
        elif isinstance(output_name, Iterable):
            raise NotImplementedError('Declaring multiple systems '
                                      'on initiation is not implemented.'
                                      'Use a string to name a single addition relationship or use '
                                      'multiple add_equation calls')
        elif output_name is None:
            pass
        else:
            raise ValueError(
                "first argument to init must be either of type "
                "'str' or 'None'")

    def initialize(self):
        """
        Declare options.

        Parameters
        ----------
        axis : int or None
            Sum along this axis. Default 0 sums along first dimension.
            None sums all elements into a scalar.
            1 sums along rows.
        """
        self.options.declare('axis', default=0,
                             desc="Axis along which to sum")

    def add_equation(self, output_name, input_name, vec_size=1, length=1, val=1.0,
                     units=None, res_units=None, desc='', lower=None, upper=None, ref=1.0,
                     ref0=0.0, res_ref=None,  scaling_factor=1):
        """
        Add a multiplication relation.

        Parameters
        ----------
        output_name : str
            (required) name of the result variable in this component's namespace.
        input_name : iterable of str
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
        val : float or list or tuple or ndarray
            The initial value of the variable being added in user-defined units. Default is 1.0.
        units : str or None
            Units in which the output variables will be provided to the component during execution.
            Default is None, which means it has no units.
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
        """
        kwargs = {'units': units, 'res_units': res_units, 'desc': desc,
                  'lower': lower, 'upper': upper, 'ref': ref, 'ref0': ref0,
                  'res_ref': res_ref}
        self._add_systems.append((output_name, input_name, vec_size, length, val,
                                  scaling_factor, kwargs))

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
        axis = self.options['axis']

        for (output_name, input_name, vec_size, length, val,
             scaling_factor, kwargs) in self._add_systems:

            units = kwargs.get('units', None)
            desc = kwargs.get('desc', '')

            if length == 1:
                shape = (vec_size,)
            else:
                shape = (vec_size, length)

            self.add_input(input_name, shape=shape, units=units,
                           desc=desc + '_inp_' + input_name)
            if axis is None:
                rowidx = np.zeros(vec_size * length)
                output_shape = (1,)
            elif axis == 0:
                output_arange = np.arange(0, length)
                rowidx = np.tile(output_arange, vec_size)
                if length == 1:
                    output_shape = (1,)
                else:
                    output_shape = (1, length)
            elif axis == 1:
                output_arange = np.arange(0, vec_size)
                rowidx = np.repeat(output_arange, length)
                output_shape = (vec_size,)
            else:
                raise ValueError('Summation is allowed only over axis=0, 1 or None')

            colidx = np.arange(0, vec_size * length)
            self.declare_partials([output_name], [input_name],
                                  rows=rowidx, cols=colidx,
                                  val=scaling_factor * np.ones(vec_size * length))
            super(SumComp, self).add_output(output_name, val,
                                            shape=output_shape,
                                            **kwargs)

    def compute(self, inputs, outputs):
        """
        Compute the  summation using numpy.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        """
        axis = self.options['axis']

        for (output_name, input_name, vec_size, length, val, scaling_factor,
             kwargs) in self._add_systems:

            if axis is None:
                output_shape = (1,)
            elif axis == 0:
                if length == 1:
                    output_shape = (1,)
                else:
                    output_shape = (1, length)
            elif axis == 1:
                output_shape = (vec_size,)

            result = np.sum(inputs[input_name], axis=axis) * scaling_factor
            outputs[output_name] = result.reshape(output_shape)
