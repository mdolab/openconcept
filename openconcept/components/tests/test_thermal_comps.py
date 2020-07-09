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
        prob = Problem(thermal.SimpleEngine(num_nodes=nn))
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
        prob = Problem(thermal.SimpleEngine(num_nodes=nn))
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
        prob = Problem(thermal.SimpleHeatPump(num_nodes=nn))
        prob.setup(check=True, force_alloc_complex=True)
        prob.run_model()
        assert_near_equal(prob['COP_cooling'], np.ones(nn)*0.8)
        assert_near_equal(prob['q_h'], np.ones(nn)*1800.)
        assert_near_equal(prob['q_c'], np.ones(nn)*800.)

        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_nondefault_settings(self):
        nn = 3
        T_h = np.array([400., 700., 1000.])
        T_c = np.array([300., 400., 500.])
        Wdot = np.array([1000., 500., 250.])
        eff_factor = 0.1
        prob = Problem(thermal.SimpleHeatPump(num_nodes=nn))
        prob.setup(check=True, force_alloc_complex=True)
        prob['T_h'] = T_h
        prob['T_c'] = T_c
        prob['Wdot'] = Wdot
        prob['eff_factor'] = eff_factor
        prob.run_model()
        assert_near_equal(prob['COP_cooling'], np.array([0.3, 2./15., 0.1]))
        assert_near_equal(prob['q_h'], np.array([1300., 1000./15.+500., 275.]))
        assert_near_equal(prob['q_c'], np.array([300., 1000./15., 25.]))

        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)

class SimpleTMSTestCase(unittest.TestCase):
    """
    Test the convergence of the SimpleTMS Group
    """
    def test_default_settings(self):
        # Set up the SimpleTMS problem with default values
        prob = Problem()
        prob.model = thermal.SimpleTMS()
        prob.model.linear_solver = DirectSolver()
        prob.model.nonlinear_solver = NewtonSolver()
        prob.model.nonlinear_solver.options['solve_subsystems'] = True
        prob.setup()
        prob.run_model()

        # Check that the solvers properly converged the BalanceComp so
        # the heat produced by the motor equals the heat extracted
        # by the refrigerator
        q_fridge = prob['refrigerator.q_c']
        q_motor = prob['motor.heat_out']
        relative_error_met = (q_fridge - q_motor)/q_motor < 1e-9
        self.assertTrue(relative_error_met.all())
    
    def test_vectorized(self):
        # Set up the SimpleTMS problem with 11 evaluation points
        nn = 11
        prob = Problem()
        prob.model = thermal.SimpleTMS(num_nodes=nn)
        prob.model.linear_solver = DirectSolver()
        prob.model.nonlinear_solver = NewtonSolver()
        prob.model.nonlinear_solver.options['solve_subsystems'] = True
        prob.setup()
        prob.set_val('throttle', np.linspace(0.01, 0.99, nn), units=None)
        prob.set_val('motor_elec_power_rating', 6., units='MW')
        prob.run_model()

        # Check that the solvers properly converged the BalanceComp so
        # the heat produced by the motor equals the heat extracted
        # by the refrigerator
        q_fridge = prob['refrigerator.q_c']
        q_motor = prob['motor.heat_out']
        relative_error_met = (q_fridge - q_motor)/q_motor < 1e-9
        self.assertTrue(relative_error_met.all())
    
    def test_zero_throttle(self):
        # Set up the SimpleTMS problem with throttle at zero
        prob = Problem()
        prob.model = thermal.SimpleTMS()
        prob.model.linear_solver = DirectSolver()
        prob.model.nonlinear_solver = NewtonSolver()
        prob.model.nonlinear_solver.options['solve_subsystems'] = True
        prob.setup()
        prob.set_val('throttle', 0.)
        prob.set_val('motor_elec_power_rating', 10., units='kW')
        prob.run_model()

        # Check that the solvers properly converged the BalanceComp so
        # the heat produced by the motor equals the heat extracted
        # by the refrigerator (they should both be zero)
        q_fridge = prob['refrigerator.q_c']
        q_motor = prob['motor.heat_out']
        self.assertAlmostEqual(q_fridge, 0.)
        self.assertAlmostEqual(q_motor, 0.)