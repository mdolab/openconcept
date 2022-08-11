from openmdao.api import IndepVarComp
import numpy as np
import numbers


class DictIndepVarComp(IndepVarComp):
    r"""
    Create indep variables from an external file with a Python dictionary.

    Outputs from this component are read from a Python dictionary and given
    a name matching their location in the data tree.

    For example, let's assume we have stored some data about a vehicle in a dictionary
    which can be accessed using the Python expression ``vehicledata['wheels']['diameter']``.
    The structured_name in this case is ``'wheels|diameter'``.

    The user instantiates a component as ``DictIndepVarComp(vehicledata)``
    and adds an output as follows:
    ``component_instance.add_output_from_dict('wheels|diameter')``.

    Outputs are created after initialization and are user-defined.

    Attributes
    ----------
    _data_dict : dict
        A structured dictionary object with input data to read from.
    """

    def __init__(self, data_dict, **kwargs):
        """
        Initialize the component and store the data dictionary as an attribute.

        Parameters
        ----------
        data_dict : dict
            A structured dictionary object with input data to read from
        """
        super(DictIndepVarComp, self).__init__(**kwargs)
        self._data_dict = data_dict

    def add_output_from_dict(self, structured_name, separator="|", **kwargs):
        """
        Create a new output based on data from the data dictionary

        Parameters
        ----------
        structured_name : string
            A string matching the file structure in the dictionary object
            Pipe symbols indicate treeing down one level
            Example 'aero:CLmax_flaps30' accesses data_dict['aero']['CLmax_flaps30']
        separator : string
            Separator to tree down into the data dict. Default '|' probably
            shouldn't be overridden
        """
        # tree down to the appropriate item in the tree
        split_names = structured_name.split(separator)
        data_dict_tmp = self._data_dict
        for sub_name in split_names:
            try:
                data_dict_tmp = data_dict_tmp[sub_name]
            except KeyError:
                raise KeyError('"%s" does not exist in the data dictionary' % structured_name)
        try:
            val = data_dict_tmp["value"]
        except KeyError:
            raise KeyError('Data dict entry "%s" must have a "value" key' % structured_name)
        units = data_dict_tmp.get("units", None)

        if isinstance(val, numbers.Number):
            val = np.array([val])

        super(DictIndepVarComp, self).add_output(name=structured_name, val=val, units=units, shape=val.shape)


class DymosDesignParamsFromDict:
    r"""
    Create Dymos parameters from an external file with a Python dictionary.


    Attributes
    ----------
    _data_dict : dict
        A structured dictionary object with input data to read from.
    """

    def __init__(self, data_dict, dymos_traj):
        """
        Initialize the component and store the data dictionary as an attribute.

        Parameters
        ----------
        data_dict : dict
            A structured dictionary object with input data to read from
        dymos_traj : Dymos trajectory
            A Dymos trajectory object with phases already added
        """
        self._data_dict = data_dict
        self._dymos_traj = dymos_traj

    def add_output_from_dict(self, structured_name, separator="|", opt=False, dynamic=False, **kwargs):
        """
        Create a new output based on data from the data dictionary

        Parameters
        ----------
        structured_name : string
            A string matching the file structure in the dictionary object
            Pipe symbols indicate treeing down one level
            Example 'aero:CLmax_flaps30' accesses data_dict['aero']['CLmax_flaps30']
        separator : string
            Separator to tree down into the data dict. Default '|' probably
            shouldn't be overridden
        """
        # tree down to the appropriate item in the tree
        split_names = structured_name.split(separator)
        data_dict_tmp = self._data_dict
        for sub_name in split_names:
            try:
                data_dict_tmp = data_dict_tmp[sub_name]
            except KeyError:
                raise KeyError('"%s" does not exist in the data dictionary' % structured_name)
        try:
            val = data_dict_tmp["value"]
        except KeyError:
            raise KeyError('Data dict entry "%s" must have a "value" key' % structured_name)
        units = data_dict_tmp.get("units", None)

        if isinstance(val, numbers.Number):
            val = np.array([val])

        targets = {phase: [structured_name] for phase in self._dymos_traj._phases.keys()}

        self._dymos_traj.add_design_parameter(
            structured_name, units=units, val=val, opt=opt, targets=targets, dynamic=dynamic
        )
