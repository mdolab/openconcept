from __future__ import division
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.api import Problem, NewtonSolver, DirectSolver
from openconcept.components.hydrogen.tank.GH2_reservoir import *

class GH2ReservoirTestCase(unittest.TestCase):
    def test_simple(self):
        p = Problem()
        p.model = GH2Reservoir()
        p.model.linear_solver = DirectSolver()
        p.model.nonlinear_solver = NewtonSolver()
        p.model.nonlinear_solver.options['err_on_non_converge'] = True
        p.model.nonlinear_solver.options['solve_subsystems'] = True
        p.model.nonlinear_solver.options['maxiter'] = 20
        p.setup(force_alloc_complex=True)

        p.set_val('V', 5., units='m**3')
        p.set_val('V_dot', .1, units='m**3/s')
        p.set_val('m', 2.14, units='kg')
        p.set_val('m_dot_out', .1, units='kg/s')
        p.set_val('m_dot_in', .15, units='kg/s')
        p.set_val('T_in', 25., units='K')
        p.set_val('Q_dot', 125., units='W')

        p.run_model()

        assert_near_equal(p.get_val('T', units='K'), 90., tolerance=1e-9)
        assert_near_equal(p.get_val('P', units='Pa'), 158856.78571429, tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_vectorized(self):
        nn = 5
        p = Problem()
        p.model = GH2Reservoir(num_nodes=nn, vector_V=True)
        p.model.linear_solver = DirectSolver()
        p.model.nonlinear_solver = NewtonSolver()
        p.model.nonlinear_solver.options['err_on_non_converge'] = True
        p.model.nonlinear_solver.options['solve_subsystems'] = True
        p.model.nonlinear_solver.options['maxiter'] = 20
        p.setup(force_alloc_complex=True)

        p.set_val('V', 5.*np.ones(nn), units='m**3')
        p.set_val('V_dot', .0*np.ones(nn), units='m**3/s')
        p.set_val('m', 2.14*np.ones(nn), units='kg')
        p.set_val('m_dot_out', .1*np.ones(nn), units='kg/s')
        p.set_val('m_dot_in', .15*np.ones(nn), units='kg/s')
        p.set_val('T_in', 25.*np.ones(nn), units='K')
        p.set_val('Q_dot', 125.*np.ones(nn), units='W')
        p.set_val('integ.duration', 1, units='min')

        p.run_model()

        assert_near_equal(p.get_val('T', units='K'), np.array([90., 59.88838369,
            50.36978025, 49.0379504, 49.11183131]), tolerance=1e-9)
        assert_near_equal(p.get_val('P', units='Pa'), np.array([158856.78571429,
            105707.51261441, 88906.45986877, 86555.6797673, 86686.08514129]), tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)

class GH2ReservoirODETestCase(unittest.TestCase):
    def test_simple(self):
        p = Problem()
        p.model = GH2ReservoirODE()
        p.model.linear_solver = DirectSolver()
        p.model.nonlinear_solver = NewtonSolver()
        p.model.nonlinear_solver.options['err_on_non_converge'] = True
        p.model.nonlinear_solver.options['solve_subsystems'] = True
        p.model.nonlinear_solver.options['maxiter'] = 20
        p.setup(force_alloc_complex=True)

        p.set_val('V', 5., units='m**3')
        p.set_val('V_dot', .1, units='m**3/s')
        p.set_val('T', 250., units='K')
        p.set_val('m', 2.14, units='kg')
        p.set_val('m_dot_out', .1, units='kg/s')
        p.set_val('m_dot_in', .15, units='kg/s')
        p.set_val('T_in', 25., units='K')
        p.set_val('Q_dot', 125., units='W')

        p.run_model()

        assert_near_equal(p.get_val('T_dot', units='K/s'), -18.87262369, tolerance=1e-9)
        assert_near_equal(p.get_val('P', units='Pa'), 441268.84920635, tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_vectorized(self):
        nn = 3
        p = Problem()
        p.model = GH2ReservoirODE(num_nodes=nn, vector_V=True)
        p.model.linear_solver = DirectSolver()
        p.model.nonlinear_solver = NewtonSolver()
        p.model.nonlinear_solver.options['err_on_non_converge'] = True
        p.model.nonlinear_solver.options['solve_subsystems'] = True
        p.model.nonlinear_solver.options['maxiter'] = 20
        p.setup(force_alloc_complex=True)

        p.set_val('V', np.linspace(0.5, 1, nn), units='m**3')
        p.set_val('V_dot', 0.1*np.ones(nn), units='m**3/s')
        p.set_val('T', np.linspace(200, 300, nn), units='K')
        p.set_val('m', 2.14*np.ones(nn), units='kg')
        p.set_val('m_dot_out', .1*np.ones(nn), units='kg/s')
        p.set_val('m_dot_in', .15*np.ones(nn), units='kg/s')
        p.set_val('T_in', 25.*np.ones(nn), units='K')
        p.set_val('Q_dot', 125.*np.ones(nn), units='W')

        p.run_model()

        assert_near_equal(p.get_val('T_dot', units='K/s'),
                np.array([-30.26235053, -30.55527765, -32.41498778]), tolerance=1e-9)
        assert_near_equal(p.get_val('P', units='Pa'),
                np.array([3530150.79365079, 2941792.32804233, 2647613.0952381]), tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_scalar_V(self):
        nn = 3
        p = Problem()
        p.model = GH2ReservoirODE(num_nodes=nn, vector_V=False)
        p.model.linear_solver = DirectSolver()
        p.model.nonlinear_solver = NewtonSolver()
        p.model.nonlinear_solver.options['err_on_non_converge'] = True
        p.model.nonlinear_solver.options['solve_subsystems'] = True
        p.model.nonlinear_solver.options['maxiter'] = 20
        p.setup(force_alloc_complex=True)

        p.set_val('V', 1., units='m**3')
        p.set_val('V_dot', np.zeros(nn), units='m**3/s')
        p.set_val('T', np.linspace(200, 300, nn), units='K')
        p.set_val('m', 2.14*np.ones(nn), units='kg')
        p.set_val('m_dot_out', .1*np.ones(nn), units='kg/s')
        p.set_val('m_dot_in', .15*np.ones(nn), units='kg/s')
        p.set_val('T_in', 25.*np.ones(nn), units='K')
        p.set_val('Q_dot', 125.*np.ones(nn), units='W')

        p.run_model()

        assert_near_equal(p.get_val('T_dot', units='K/s'),
                np.array([-12.8614586,  -16.81097888, -20.65282288]), tolerance=1e-9)
        assert_near_equal(p.get_val('P', units='Pa'),
                np.array([1765075.3968254, 2206344.24603175, 2647613.0952381]), tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)

if __name__ == "__main__":
    unittest.main()
