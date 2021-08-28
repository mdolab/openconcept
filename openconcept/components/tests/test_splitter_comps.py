from __future__ import division
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.api import IndepVarComp, Group, Problem
from openconcept.components import PowerSplit, FlowSplit, FlowCombine

class PowerSplitTestCase(unittest.TestCase):
    def test_default_settings(self):
        p = Problem()
        p.model = PowerSplit()
        p.setup(check=True, force_alloc_complex=True)
        p.run_model()

        assert_near_equal(p['power_out_A'], np.array([0.5]))
        assert_near_equal(p['power_out_B'], np.array([0.5]))
        assert_near_equal(p['heat_out'], np.array([0.]))
        assert_near_equal(p['component_cost'], np.array([0.]))
        assert_near_equal(p['component_weight'], np.array([0.]))
        assert_near_equal(p['component_sizing_margin'], np.array([1/99999999]))

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_fraction(self):
        p = Problem()
        p.model = PowerSplit(num_nodes=3, efficiency=0.95, weight_inc=0.01,
                             weight_base=1., cost_inc=0.02, cost_base=2.)
        p.setup(check=True, force_alloc_complex=True)

        p.set_val('power_in', np.array([1, 2, 3]), units='W')
        p.set_val('power_rating', 10., units='W')
        p.set_val('power_split_fraction', np.array([0.2, 0.4, 0.3]))

        p.run_model()

        assert_near_equal(p['power_out_A'], 0.95 * np.array([0.2, 0.8, 0.9]))
        assert_near_equal(p['power_out_B'], 0.95 * np.array([0.8, 1.2, 2.1]))
        assert_near_equal(p['heat_out'], 0.05 * np.array([1., 2., 3.]))
        assert_near_equal(p['component_cost'], 2.2)
        assert_near_equal(p['component_weight'], 1.1)
        assert_near_equal(p['component_sizing_margin'], np.array([.1, .2, .3]))

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_fixed(self):
        p = Problem()
        p.model = PowerSplit(num_nodes=3, rule='fixed', efficiency=0.95, weight_inc=0.01,
                             weight_base=1., cost_inc=0.02, cost_base=2.)
        p.setup(check=True)

        p.set_val('power_in', np.array([1, 2, 3]), units='W')
        p.set_val('power_rating', 10., units='W')
        p.set_val('power_split_amount', np.array([0.95, 1., 1.]))

        p.run_model()

        assert_near_equal(p['power_out_A'], 0.95 * np.array([0.95, 1., 1.]))
        assert_near_equal(p['power_out_B'], 0.95 * (np.array([1., 2., 3.]) - np.array([0.95, 1., 1.])))
        assert_near_equal(p['heat_out'], 0.05 * np.array([1., 2., 3.]))
        assert_near_equal(p['component_cost'], 2.2)
        assert_near_equal(p['component_weight'], 1.1)
        assert_near_equal(p['component_sizing_margin'], np.array([.1, .2, .3]))

        partials = p.check_partials(method='fd',compact_print=True)  # for some reason this one
                                                                     # doesn't work with complex step
        assert_check_partials(partials)

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
