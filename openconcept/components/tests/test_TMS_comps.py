from __future__ import division
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.api import Problem, NewtonSolver, DirectSolver
import openconcept.components.tests.TMS as TMS

class SimpleTMSTestCase(unittest.TestCase):
    """
    Test the convergence of the SimpleTMS Group
    """
    def test_default_settings(self):
        # Set up the SimpleTMS problem with default values
        prob = Problem()
        prob.model = TMS.SimpleTMS()
        prob.model.linear_solver = DirectSolver()
        prob.model.nonlinear_solver = NewtonSolver()
        prob.model.nonlinear_solver.options['solve_subsystems'] = True
        prob.setup()
        prob.run_model()

        # Check that the solvers properly converged the BalanceComp so
        # the heat taken by the cold plate equals the heat extracted
        # by the refrigerator
        q_fridge = prob['refrigerator.q_c']
        q_plate = prob['refrigerator_cold_plate.q']
        relative_error_met = (q_fridge - q_plate)/q_plate < 1e-9
        self.assertTrue(relative_error_met.all())
    
    def test_vectorized(self):
        # Set up the SimpleTMS problem with 11 evaluation points
        nn = 11
        prob = Problem()
        prob.model = TMS.SimpleTMS(num_nodes=nn)
        prob.model.linear_solver = DirectSolver()
        prob.model.nonlinear_solver = NewtonSolver()
        prob.model.nonlinear_solver.options['solve_subsystems'] = True
        prob.setup()
        prob.set_val('throttle', np.linspace(0.01, 0.99, nn), units=None)
        prob.set_val('motor_elec_power_rating', 6., units='MW')
        prob.run_model()

        # Check that the solvers properly converged the BalanceComp so
        # the heat taken by the cold plate equals the heat extracted
        # by the refrigerator
        q_fridge = prob['refrigerator.q_c']
        q_plate = prob['refrigerator_cold_plate.q']
        relative_error_met = (q_fridge - q_plate)/q_plate < 1e-9
        self.assertTrue(relative_error_met.all())
    
    def test_zero_work(self):
        # Set up the SimpleTMS problem with throttle at zero
        prob = Problem()
        prob.model = TMS.SimpleTMS()
        prob.model.linear_solver = DirectSolver()
        prob.model.nonlinear_solver = NewtonSolver()
        prob.model.nonlinear_solver.options['solve_subsystems'] = True
        prob.setup()
        prob.set_val('Wdot', 0.)
        prob.set_val('motor_elec_power_rating', 10., units='kW')
        prob.run_model()

        # Check that the solvers properly converged the BalanceComp so
        # the heat taken by the cold plate equals the heat extracted
        # by the refrigerator
        q_fridge = prob['refrigerator.q_c']
        q_plate = prob['refrigerator_cold_plate.q']
        fridge_abs_error_met = np.abs(q_fridge) < 1e-9
        plate_abs_error_met = np.abs(q_plate) < 1e-9
        self.assertTrue(fridge_abs_error_met.all())
        self.assertTrue(plate_abs_error_met.all())
