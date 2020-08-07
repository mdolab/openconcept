from __future__ import division
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.api import Problem
from openconcept.utilities.math.max_min_comp import MaxComp, MinComp

class MaxCompTestCase(unittest.TestCase):
    """
    Test the MaxComp component
    """
    def test_one_input(self):
        p = Problem()
        p.model.add_subsystem('test', MaxComp(), promotes=['*'])
        p.setup(check=True, force_alloc_complex=True)
        p.set_val('array', np.array([42.]))
        p.run_model()
        assert_near_equal(p['max'], 42.)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_multiple_inputs(self):
        nn = 5
        p = Problem()
        p.model.add_subsystem('test', MaxComp(num_nodes=nn), promotes=['*'])
        p.setup(check=True, force_alloc_complex=True)
        p.set_val('array', np.array([42., 12., -3., 58., 7.]))
        p.run_model()
        assert_near_equal(p['max'], 58.)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_multiple_max(self):
        nn = 5
        p = Problem()
        p.model.add_subsystem('test', MaxComp(num_nodes=nn), promotes=['*'])
        p.setup(check=True, force_alloc_complex=True)
        p.set_val('array', np.array([42., 58., -3., 58., 7.]))
        p.run_model()
        assert_near_equal(p['max'], 58.)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_multiple_very_close_max(self):
        nn = 5
        p = Problem()
        p.model.add_subsystem('test', MaxComp(num_nodes=nn), promotes=['*'])
        p.setup(check=True, force_alloc_complex=True)
        p.set_val('array', np.array([-2., 2e-45, -3., 1e-45, -7.]))
        p.run_model()
        assert_near_equal(p['max'], 2e-45, tolerance=1e-50)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_units(self):
        nn = 5
        p = Problem()
        p.model.add_subsystem('max_comp', MaxComp(num_nodes=nn, units='N'), promotes=['*'])
        p.setup(check=True, force_alloc_complex=True)
        p.set_val('array', np.array([42., 58., -3., 3., 7.]), units='N')
        p.run_model()
        assert_near_equal(p['max'], 58.)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)

class MinCompTestCase(unittest.TestCase):
    """
    Test the MinComp component
    """
    def test_one_input(self):
        p = Problem()
        p.model.add_subsystem('test', MinComp(), promotes=['*'])
        p.setup(check=True, force_alloc_complex=True)
        p.set_val('array', np.array([42.]))
        p.run_model()
        assert_near_equal(p['min'], 42.)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_multiple_inputs(self):
        nn = 5
        p = Problem()
        p.model.add_subsystem('test', MinComp(num_nodes=nn), promotes=['*'])
        p.setup(check=True, force_alloc_complex=True)
        p.set_val('array', np.array([42., 12., -3., 58., 7.]))
        p.run_model()
        assert_near_equal(p['min'], -3.)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_multiple_min(self):
        nn = 5
        p = Problem(MinComp(num_nodes=nn))
        p.setup(check=True, force_alloc_complex=True)
        p.set_val('array', np.array([42., 7., 30., 58., 7.]))
        p.run_model()
        assert_near_equal(p['min'], 7.)

        # Need to use fd here because cs doesn't support backward and forward does
        # not capture behavior of minimum
        partials = p.check_partials(method='fd',form='backward',compact_print=True)
        assert_check_partials(partials)
    
    def test_multiple_very_close_min(self):
        nn = 5
        p = Problem()
        p.model.add_subsystem('test', MinComp(num_nodes=nn), promotes=['*'])
        p.setup(check=True, force_alloc_complex=True)
        p.set_val('array', np.array([2., 2e-45, 3., 1e-45, 7.]))
        p.run_model()
        assert_near_equal(p['min'], 1e-45, tolerance=1e-50)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_units(self):
        nn = 5
        p = Problem()
        p.model.add_subsystem('min_comp', MinComp(num_nodes=nn, units='N'), promotes=['*'])
        p.setup(check=True, force_alloc_complex=True)
        p.set_val('array', np.array([42., 58., -3., 3., 7.]), units='N')
        p.run_model()
        assert_near_equal(p['min'], -3.)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)