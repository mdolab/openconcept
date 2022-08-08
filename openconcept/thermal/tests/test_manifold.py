from __future__ import division
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.api import IndepVarComp, Group, Problem
from openconcept.thermal import FlowSplit, FlowCombine


class FlowSplitTestCase(unittest.TestCase):
    """
    Test the FlowSplit component
    """
    def test_default_settings(self):
        p = Problem()
        p.model.add_subsystem('test', FlowSplit(), promotes=['*'])
        p.setup(check=True, force_alloc_complex=True)
        p.run_model()
        assert_near_equal(p['mdot_out_A'], np.array([0.5]))
        assert_near_equal(p['mdot_out_B'], np.array([0.5]))

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_nondefault_settings(self):
        nn = 4
        p = Problem()
        p.model.add_subsystem('test', FlowSplit(num_nodes=nn), promotes=['*'])
        p.setup(check=True, force_alloc_complex=True)
        
        p['mdot_in'] = np.array([-10., 0., 10., 10.])
        p['mdot_split_fraction'] = np.array([0., 0.4, 0.4, 1.])

        p.run_model()

        assert_near_equal(p['mdot_out_A'], np.array([0., 0., 4., 10.]))
        assert_near_equal(p['mdot_out_B'], np.array([-10., 0., 6., 0.]))

    def test_warnings(self):
        nn = 4
        p = Problem()
        p.model.add_subsystem('test', FlowSplit(num_nodes=nn), promotes=['*'])
        p.setup(check=True, force_alloc_complex=True)
        
        p['mdot_in'] = np.array([-10., 0., 10., 10.])
        p['mdot_split_fraction'] = np.array([-0.0001, 0.4, 0.4, 1.])
        with self.assertRaises(RuntimeWarning):
            p.run_model()
        

        p['mdot_split_fraction'] = np.array([1.0001, 0.4, 0.4, 1.])
        with self.assertRaises(RuntimeWarning):
            p.run_model()

class FlowCombineTestCase(unittest.TestCase):
    """
    Test the FlowCombine component
    """
    def test_default_settings(self):
        p = Problem()
        p.model.add_subsystem('test', FlowCombine(), promotes=['*'])
        p.setup(check=True, force_alloc_complex=True)
        p.run_model()
        assert_near_equal(p['mdot_out'], np.array([2.]))
        assert_near_equal(p['T_out'], np.array([1.]))

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_nondefault_settings(self):
        nn = 4
        p = Problem()
        p.model.add_subsystem('test', FlowCombine(num_nodes=nn), promotes=['*'])
        p.setup(check=True, force_alloc_complex=True)
        
        p['mdot_in_A'] = np.array([0., 5., 10., 10.])
        p['mdot_in_B'] = np.array([1., 0., 5., 10.])
        p['T_in_A'] = np.array([1., 10., 30., 500.])
        p['T_in_B'] = np.array([1., 150., 60., 100.])

        p.run_model()

        assert_near_equal(p['mdot_out'], np.array([1., 5., 15., 20.]))
        assert_near_equal(p['T_out'], np.array([1., 10., 40., 300.]))
