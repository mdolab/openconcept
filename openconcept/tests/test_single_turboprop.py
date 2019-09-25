from __future__ import division
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_rel_error, assert_check_partials
from openmdao.api import IndepVarComp, Group, Problem
from examples.TBM850 import run_tbm_analysis
class TBMAnalysisTestCase(unittest.TestCase):
    def setUp(self):
        self.prob = run_tbm_analysis()

    def test_values_TBM(self):
        prob = self.prob
        assert_rel_error(self, prob.get_val('climb.OEW', units='lb'), 4756.772140709275, tolerance=1e-5)
        assert_rel_error(self, prob.get_val('rotate.range_final', units='ft'), 2489.7142498884746, tolerance=1e-5)
        assert_rel_error(self, prob.get_val('engineoutclimb.gamma',units='deg'), 8.78263, tolerance=1e-5)
        assert_rel_error(self, prob.get_val('descent.fuel_used_final', units='lb'), 1605.32123542, tolerance=1e-5)
