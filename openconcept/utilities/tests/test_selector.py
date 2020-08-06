from __future__ import division
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.api import Problem, IndepVarComp
from openconcept.utilities.selector import SelectorComp

class SelectorCompTestCase(unittest.TestCase):
    """
    Test the SelectorComp component
    """
    def test_one_input(self):
        p = Problem(SelectorComp(input_names=['A']))
        p.setup(check=True, force_alloc_complex=True)
        p.set_val('A', np.array([5.7]))
        p.set_val('selector', np.array([0]))
        p.run_model()
        assert_near_equal(p['result'], np.array([5.7]))

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)

        p.set_val('selector', np.array([1]))
        with self.assertRaises(RuntimeWarning):
            p.run_model()
    
    def test_two_inputs(self):
        nn = 5
        p = Problem(SelectorComp(num_nodes=nn, input_names=['A', 'B']))
        p.setup(check=True, force_alloc_complex=True)
        p.set_val('A', np.array([5.7, 2.3, -10., 42., 77.]))
        p.set_val('B', np.array([-1., -1., -1., -1., -2.]))
        p.set_val('selector', np.array([0, 1, 1, 0, 1]))
        p.run_model()
        assert_near_equal(p['result'], np.array([5.7, -1., -1., 42., -2.]))
        
        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)

        p.set_val('A', np.ones(nn))
        p.set_val('B', np.zeros(nn))
        p.set_val('selector', np.zeros(nn))
        p.run_model()
        assert_near_equal(p['result'], np.ones(nn))

        p.set_val('selector', np.array([0, 1, -1, 0, 0]))
        with self.assertRaises(RuntimeWarning):
            p.run_model()
        
        p.set_val('selector', np.array([0, 1, 2, 0, 0]))
        with self.assertRaises(RuntimeWarning):
            p.run_model()
    
    def test_three_inputs(self):
        nn = 5
        p = Problem()
        # Need to use an IndepVarComp here unlike the other two tests because
        # setting an input wasn't working when units were included
        iv = p.model.add_subsystem('iv', IndepVarComp(), promotes=['*'])
        iv.add_output('A', np.array([5.7, 2.3, -10., 2., 77.]), units='g')
        iv.add_output('B', np.array([-1., -1., -1., -1., -2.]), units='kg')
        iv.add_output('C', 42.*np.ones(nn), units='g')
        iv.add_output('selector', np.array([0, 1, 2, 0, 2]))
        p.model.add_subsystem('selector', SelectorComp(num_nodes=nn, input_names=['A', 'B', 'C'], units='g'),
                            promotes=['*'])
        p.setup(check=True, force_alloc_complex=True)
        p.run_model()
        assert_near_equal(p['result'], np.array([5.7, -1000., 42., 2., 42.]))
        
        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)

        p.set_val('A', 5.*np.ones(nn), units='g')
        p.set_val('B', 6.*np.ones(nn), units='g')
        p.set_val('C', 7.*np.ones(nn), units='g')
        p.set_val('selector', np.zeros(nn))
        p.run_model()
        assert_near_equal(p['result'], 5.*np.ones(nn))
        
        p.set_val('selector', np.ones(nn))
        p.run_model()
        assert_near_equal(p['result'], 6.*np.ones(nn))

        p.set_val('selector', 2.*np.ones(nn))
        p.run_model()
        assert_near_equal(p['result'], 7.*np.ones(nn))

        p.set_val('selector', np.array([-1, 1, 0, 2, 0]))
        with self.assertRaises(RuntimeWarning):
            p.run_model()
        
        p.set_val('selector', np.array([0, 1, -1, 2, 3]))
        with self.assertRaises(RuntimeWarning):
            p.run_model()