"""Definition of the Vector Combiner/Splitter Component."""

from collections.abc import Iterable
import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent


class VectorConcatenateComp(ExplicitComponent):
    r"""
    Concatenate one or more sets of more than one vector into one or more output vectors.

    Use the add_relation method to define any number of concat relationships
    User defines the names of the input and output variables using
    add_relation(output_name='my_output', input_names=['a','b', 'c', ...],vec_sizes=[10,10,5,...])

    For each relation declared:
    All input vectors must be of the same second dimension, specified by the option 'length'.
    The number of vec_sizes given must match the number of inputs declared.
    Input units must be compatible with output units for each relation.

    Attributes
    ----------
    _add_systems : list
        List of equation systems to be initialized with the system.
    """

    def __init__(self, output_name=None, input_names=None, vec_sizes=None, length=1, val=1.0, **kwargs):
        """
        Allow user to create an addition/subtracton system with one-liner.

        Parameters
        ----------
        output_name : str
            (required) name of the result variable in this component's namespace.
        input_names : iterable of str
            (required) names of the input variables for this system
        vec_sizes : iterable of int
            (required) Lengths of the first dimension of each input vector
            (i.e number of rows, or vector length for a 1D vector)
        length : int
            Length of the second dimension of the input and ouptut vectors (i.e. number of columns)
            Default is 1 (i.e. a vector of scalars)
        val : float or list or tuple or ndarray
            The initial value of the variable being added in user-defined units. Default is 1.0.
        **kwargs : str
            Any other arguments to pass to the addition system
            (same as add_output method for ExplicitComponent)
            Examples include units (str or None), desc (str)
        """
        super(VectorConcatenateComp, self).__init__()

        self._add_systems = []

        if isinstance(output_name, str):
            if not isinstance(input_names, Iterable) or not isinstance(vec_sizes, Iterable):
                raise ValueError("User must provide list of input name(s)" "and list of vec_sizes for each input")

            self._add_systems.append((output_name, input_names, vec_sizes, length, val, kwargs))
        elif isinstance(output_name, Iterable):
            raise NotImplementedError(
                "Declaring multiple relations "
                "on initiation is not implemented."
                "Use a string to name a single addition relationship or use "
                "multiple add_relation calls"
            )
        elif output_name is None:
            pass
        else:
            raise ValueError("First argument to init must be either of type " "'str' or 'None'")

    def add_relation(
        self,
        output_name,
        input_names,
        vec_sizes,
        length=1,
        val=1.0,
        units=None,
        res_units=None,
        desc="",
        lower=None,
        upper=None,
        ref=1.0,
        ref0=0.0,
        res_ref=None,
    ):
        """
        Add a concatenation relation.

        Parameters
        ----------
        output_name : str
            (required) name of the result variable in this component's namespace.
        input_names : iterable of str
            (required) names of the input variables for this system
        vec_sizes : iterable of int
            (required) Lengths of the first dimension of each input vector
            (i.e number of rows, or vector length for a 1D vector)
        length : int
            Length of the second dimension of the input and ouptut vectors (i.e. number of columns)
            Default is 1 (i.e. a vector of scalars)
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
        kwargs = {
            "units": units,
            "res_units": res_units,
            "desc": desc,
            "lower": lower,
            "upper": upper,
            "ref": ref,
            "ref0": ref0,
            "res_ref": res_ref,
        }

        if not isinstance(input_names, Iterable) or not isinstance(vec_sizes, Iterable):
            raise ValueError("User must provide list of input name(s)" "and list of vec_sizes for each input")

        self._add_systems.append((output_name, input_names, vec_sizes, length, val, kwargs))

    def add_output(self):
        """
        Use add_relation instead of add_output to define concatenate relations.
        """
        raise NotImplementedError("Use add_relation method, not add_output method" "to create an concatenate relation")

    def setup(self):
        """
        Set up the component at run time from both add_relation and __init__.
        """
        for output_name, input_names, vec_sizes, length, val, kwargs in self._add_systems:
            if isinstance(input_names, str):
                input_names = [input_names]

            units = kwargs.get("units", None)
            desc = kwargs.get("desc", "")

            if len(vec_sizes) != len(input_names):
                raise ValueError("vec_sizes list needs to be same length as input names list")
            output_size = np.sum(vec_sizes)
            if length == 1:
                output_shape = (output_size,)
            else:
                output_shape = (output_size, length)

            super(VectorConcatenateComp, self).add_output(output_name, val, shape=output_shape, **kwargs)

            for i, input_name in enumerate(input_names):
                if length == 1:
                    input_shape = (vec_sizes[i],)
                else:
                    input_shape = (vec_sizes[i], length)
                self.add_input(input_name, shape=input_shape, units=units, desc=desc + "_inp_" + input_name)
                if i == 0:
                    start_idx = 0
                else:
                    start_idx = np.sum(vec_sizes[0:i])
                end_idx = np.sum(vec_sizes[0 : i + 1])
                rowidxs = np.arange(start_idx * length, end_idx * length)

                self.declare_partials(
                    [output_name],
                    [input_name],
                    rows=rowidxs,
                    cols=np.arange(0, vec_sizes[i] * length),
                    val=np.ones(vec_sizes[i] * length),
                )

    def compute(self, inputs, outputs):
        """
        Concatenate the vector(s) using numpy.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        """
        for output_name, input_names, _, length, _, _ in self._add_systems:
            if isinstance(input_names, str):
                input_names = [input_names]

            if self.under_complex_step:
                dtype = np.complex_
            else:
                dtype = np.float64
            if length == 1:
                temp = np.array([], dtype=dtype)
            else:
                temp = np.empty([0, length], dtype=np.dtype)

            for input_name in input_names:
                temp = np.concatenate((temp, inputs[input_name]))
            outputs[output_name] = temp


class VectorSplitComp(ExplicitComponent):
    r"""
    Splits one or more vectors into one or more sets of 2+ vectors.

    Use the add_relation method to define any number of splitter relationships
    User defines the names of the input and output variables using
    add_relation(output_names=['a','b', 'c', ...],input_name='my_input',vec_sizes=[10,10,5,...])

    For each relation declared:
    All output vectors must be of the same second dimension, specified by the option 'length'.
    The first dim length of the input vector must equal the sum of the first dim
    lengths of the output vectors.
    The number of vec_sizes given must match the number of outputs declared.
    Input units must be compatible with output units for each relation.

    Attributes
    ----------
    _add_systems : list
        List of equation systems to be initialized with the system.
    """

    def __init__(self, output_names=None, input_name=None, vec_sizes=None, length=1, val=1.0, **kwargs):
        """
        Allow user to create an addition/subtracton system with one-liner.

        Parameters
        ----------
        output_names : iterable of str
            (required) names of the output (split) variables in this component's namespace.
        input_name : str
            (required) names of the input variable for this system
        vec_sizes : iterable of int
            (required) Lengths of the first dimension of each input vector
            (i.e number of rows, or vector length for a 1D vector)
        length : int
            Length of the second dimension of the input and ouptut vectors (i.e. number of columns)
            Default is 1 (i.e. a vector of scalars)
        val : float or list or tuple or ndarray
            The initial value of the variable being added in user-defined units. Default is 1.0.
        **kwargs : str
            Any other arguments to pass to the addition system
            (same as add_output method for ExplicitComponent)
            Examples include units (str or None), desc (str)
        """
        super(VectorSplitComp, self).__init__()

        self._add_systems = []

        if isinstance(input_name, str):
            if not isinstance(output_names, Iterable) or not isinstance(vec_sizes, Iterable):
                raise ValueError("User must provide list of output name(s)" "and list of vec_sizes for each input")

            self._add_systems.append((output_names, input_name, vec_sizes, length, val, kwargs))
        elif isinstance(input_name, Iterable):
            raise NotImplementedError(
                "Declaring multiple relations "
                "on initiation is not implemented."
                "Use a string to name a single addition relationship or use "
                "multiple add_relation calls"
            )
        elif input_name is None:
            pass
        else:
            raise ValueError("input_name argument to init must be either of type " "'str' or 'None'")

    def add_relation(
        self,
        output_names,
        input_name,
        vec_sizes,
        length=1,
        val=1.0,
        units=None,
        res_units=None,
        desc="",
        lower=None,
        upper=None,
        ref=1.0,
        ref0=0.0,
        res_ref=None,
    ):
        """
        Add a concatenation relation.

        Parameters
        ----------
        output_names : iterable of str
            (required) names of the output (split) variables in this component's namespace.
        input_name : str
            (required) names of the input variable for this system
        vec_sizes : iterable of int
            (required) Lengths of the first dimension of each input vector
            (i.e number of rows, or vector length for a 1D vector)
        length : int
            Length of the second dimension of the input and ouptut vectors (i.e. number of columns)
            Default is 1 (i.e. a vector of scalars)
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
        kwargs = {
            "units": units,
            "res_units": res_units,
            "desc": desc,
            "lower": lower,
            "upper": upper,
            "ref": ref,
            "ref0": ref0,
            "res_ref": res_ref,
        }

        if not isinstance(output_names, Iterable) or not isinstance(vec_sizes, Iterable):
            raise ValueError("User must provide list of output name(s)" "and list of vec_sizes for each input")

        self._add_systems.append((output_names, input_name, vec_sizes, length, val, kwargs))

    def add_output(self):
        """
        Use add_relation instead of add_output to define split relations.
        """
        raise NotImplementedError("Use add_relation method, not add_output method" "to create a split relation")

    def setup(self):
        """
        Set up the component at run time from both add_relation and __init__.
        """
        for output_names, input_name, vec_sizes, length, val, kwargs in self._add_systems:
            if isinstance(output_names, str):
                output_names = [output_names]

            units = kwargs.get("units", None)
            desc = kwargs.get("desc", "")

            if len(vec_sizes) != len(output_names):
                raise ValueError("vec_sizes list needs to be same length as output names list")
            input_size = np.sum(vec_sizes)
            if length == 1:
                input_shape = (input_size,)
            else:
                input_shape = (input_size, length)
            self.add_input(input_name, shape=input_shape, units=units, desc=desc + "_inp_" + input_name)

            for i, output_name in enumerate(output_names):
                if length == 1:
                    output_shape = (vec_sizes[i],)
                else:
                    output_shape = (vec_sizes[i], length)
                super(VectorSplitComp, self).add_output(output_name, val, shape=output_shape, **kwargs)
                if i == 0:
                    start_idx = 0
                else:
                    start_idx = np.sum(vec_sizes[0:i])
                end_idx = np.sum(vec_sizes[0 : i + 1])
                colidx = np.arange(start_idx * length, end_idx * length)

                self.declare_partials(
                    [output_name],
                    [input_name],
                    rows=np.arange(0, vec_sizes[i] * length),
                    cols=colidx,
                    val=np.ones(vec_sizes[i] * length),
                )

    def compute(self, inputs, outputs):
        """
        Split the vector(s) using numpy.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        """
        for output_names, input_name, vec_sizes, length, _, _ in self._add_systems:
            if isinstance(output_names, str):
                output_names = [output_names]

            for i, output_name in enumerate(output_names):
                if i == 0:
                    start_idx = 0
                else:
                    start_idx = np.sum(vec_sizes[0:i])
                end_idx = np.sum(vec_sizes[0 : i + 1])
                if length == 1:
                    outputs[output_name] = inputs[input_name][start_idx:end_idx]
                else:
                    outputs[output_name] = inputs[input_name][start_idx:end_idx, :]
