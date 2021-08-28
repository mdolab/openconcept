from __future__ import division
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.api import Problem, NewtonSolver, DirectSolver
import openconcept.components.thermal as thermal

class SimpleEngineTestCase(unittest.TestCase):
    """
    Test the SimpleEngine component
    """
    def test_default_settings(self):
        nn = 11
        prob = Problem()
        prob.model.add_subsystem('test', thermal.SimpleEngine(num_nodes=nn), promotes=['*'])
        prob.setup(check=True, force_alloc_complex=True)
        prob.run_model()
        assert_near_equal(prob['eta_thermal'], np.ones(nn)*2./15.)
        assert_near_equal(prob['q_h'], np.ones(nn)*7500.)
        assert_near_equal(prob['q_c'], np.ones(nn)*6500.)

        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_nondefault_settings(self):
        nn = 3
        T_h = np.array([400., 700., 1000.])
        T_c = np.array([300., 400., 500.])
        Wdot = np.array([1000., 500., 250.])
        eff_factor = 0.8
        prob = Problem()
        prob.model.add_subsystem('test', thermal.SimpleEngine(num_nodes=nn), promotes=['*'])
        prob.setup(check=True, force_alloc_complex=True)
        prob['T_h'] = T_h
        prob['T_c'] = T_c
        prob['Wdot'] = Wdot
        prob['eff_factor'] = eff_factor
        prob.run_model()
        assert_near_equal(prob['eta_thermal'], np.array([0.2, 12./35., 0.4]))
        assert_near_equal(prob['q_h'], np.array([5000., 500.*35./12., 625.]))
        assert_near_equal(prob['q_c'], np.array([4000., 500.*23./12., 375.]))

        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)

class SimpleHeatPumpTestCase(unittest.TestCase):
    """
    Test the SimpleHeatPump component
    """
    def test_default_settings(self):
        nn = 11
        prob = Problem()
        prob.model.add_subsystem('test', thermal.SimpleHeatPump(num_nodes=nn), promotes=['*'])
        prob.setup(check=True, force_alloc_complex=True)
        prob.run_model()
        assert_near_equal(prob['COP_cooling'], np.ones(nn)*0.8)
        assert_near_equal(prob['q_h'], np.ones(nn)*1800.)
        assert_near_equal(prob['q_c'], np.ones(nn)*(-800.))

        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_nondefault_settings(self):
        nn = 3
        T_h = np.array([400., 700., 1000.])
        T_c = np.array([300., 400., 500.])
        Wdot = np.array([1000., 500., 250.])
        eff_factor = 0.1
        prob = Problem()
        prob.model.add_subsystem('test', thermal.SimpleHeatPump(num_nodes=nn), promotes=['*'])
        prob.setup(check=True, force_alloc_complex=True)
        prob['T_h'] = T_h
        prob['T_c'] = T_c
        prob['Wdot'] = Wdot
        prob['eff_factor'] = eff_factor
        prob.run_model()
        assert_near_equal(prob['COP_cooling'], np.array([0.3, 2./15., 0.1]))
        assert_near_equal(prob['q_h'], np.array([1300., 1000./15.+500., 275.]))
        assert_near_equal(prob['q_c'], np.array([-300., -1000./15., -25.]))

        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)

    def test_partials_near_zero_thermal_lift(self):
        nn = 3
        T_h = np.array([300.1, 300., 100.])
        T_c = np.array([300., 300.1, 500.])
        prob = Problem()
        prob.model.add_subsystem('test', thermal.SimpleHeatPump(num_nodes=nn), promotes=['*'])
        prob.setup(check=True, force_alloc_complex=True)
        prob['T_h'] = T_h
        prob['T_c'] = T_c
        prob.run_model()

        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)

class PerfectHeatTransferCompTestCase(unittest.TestCase):
    """
    Test the PerfectHeatTransferComp component
    """
    def test_comp(self):
        num_nodes = 3
        prob = Problem()
        prob.model.add_subsystem('test', thermal.PerfectHeatTransferComp(num_nodes=num_nodes), promotes=['*'])
        prob.setup(check=True, force_alloc_complex=True)

        # Set the values
        prob['T_in'] = np.array([300., 350., 400.])
        prob['q'] = np.array([10000., 0., -10000.])
        prob['mdot_coolant'] = np.array([1., 1., 1.])

        prob.run_model()

        dT_coolant = 10000./3801.

        assert_near_equal(prob['T_out'], np.array([300 + dT_coolant, 350., 400 - dT_coolant]))
        assert_near_equal(prob['T_average'], np.array([300 + dT_coolant/2, 350., 400 - dT_coolant/2]))

        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)