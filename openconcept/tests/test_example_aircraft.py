
from __future__ import division
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_rel_error, assert_check_partials
from openmdao.api import IndepVarComp, Group, Problem
from examples.B738 import run_738_analysis

class B738TestCase(unittest.TestCase):
    def setUp(self):
        self.prob = run_738_analysis()
    
    def test_values_B738(self):
        prob = self.prob
        # block fuel
        assert_rel_error(self, prob.get_val('descent.fuel_used_final', units='lbm'), 28688.32933661591, tolerance=1e-5)
        # total fuel
        assert_rel_error(self, prob.get_val('loiter.fuel_used_final', units='lbm'), 34555.31347454542, tolerance=1e-5)
