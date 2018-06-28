from openmdao.api import IndepVarComp
import numpy as np
import numbers


class DictIndepVarComp(IndepVarComp):
    r"""
    Create indep variable inputs from an external file with a Python dictionary.

    Add more here later

    Attributes
    ----------
    _data_dict : dict
    A structured dictionary object with input data to read from.
    """

    def __init__(self, data_dict, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        data_dict : dict
            A structured dictionary object with input data to read from
        """
        super(DictIndepVarComp, self).__init__(**kwargs)
        self._data_dict = data_dict

    def add_output_from_dict(self, structured_name, separator=':', **kwargs):
        """
        Create a new output based on data from the data dictionary

        Parameters
        ----------
        structured_name : string
            A string matching the file structure in the dictionary object
            Pipe symbols indicate treeing down one level
            Example 'aero:CLmax_flaps30' accesses data_dict['aero']['CLmax_flaps30']
        separator : string
            Separator to tree down into the data dict. Default ':' probably
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
            val = data_dict_tmp['value']
        except KeyError:
            raise KeyError('Data dict entry "%s" must have a "value" key' % structured_name)
        units = data_dict_tmp.get('units', None)

        if isinstance(val, numbers.Number):
            val = np.array([val])

        super(DictIndepVarComp, self).add_output(name=structured_name, val=val, units=units, shape=val.shape)