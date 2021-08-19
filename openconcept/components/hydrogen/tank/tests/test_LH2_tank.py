from __future__ import division
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.api import Problem, NewtonSolver, DirectSolver
from openconcept.components.hydrogen.tank.LH2_tank import *

class SimpleLH2TankTestCase(unittest.TestCase):
    def test_simple(self):
        p = Problem()
        p.model = SimpleLH2Tank()
        p.model.linear_solver = DirectSolver()
        p.model.nonlinear_solver = NewtonSolver()
        p.model.nonlinear_solver.options['err_on_non_converge'] = True
        p.model.nonlinear_solver.options['solve_subsystems'] = True
        p.model.nonlinear_solver.options['maxiter'] = 20
        p.setup(force_alloc_complex=True)

        p.set_val('design_pressure', 2., units='bar')
        p.set_val('m_dot', 0., units='kg/s')
        p.set_val('radius', 3., units='ft')
        p.set_val('length', 3., units='ft')
        p.set_val('T_inf', 300., units='K')
        p.set_val('insulation_thickness', 5., units='inch')
        p.set_val('LH2_mass_integrator.duration', 100, units='h')

        p.run_model()

        assert_near_equal(p.get_val('W_LH2', units='kg'), 377.2235258383, tolerance=1e-9)
        assert_near_equal(p.get_val('weight', units='kg'), 505.01634881, tolerance=1e-9)
        assert_near_equal(p.get_val('m_boil_off', units='kg/s'), 8.807768372409e-04, tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_vectorized(self):
        nn = 5
        p = Problem()
        p.model = SimpleLH2Tank(num_nodes=nn, safety_factor=2., init_fill_level=0.5, T_surf_guess=90.)
        p.model.linear_solver = DirectSolver()
        p.model.nonlinear_solver = NewtonSolver()
        p.model.nonlinear_solver.options['err_on_non_converge'] = True
        p.model.nonlinear_solver.options['solve_subsystems'] = True
        p.model.nonlinear_solver.options['maxiter'] = 20
        p.setup(force_alloc_complex=True)

        p.set_val('design_pressure', 2., units='bar')
        p.set_val('m_dot', np.array([0., 1., 2., 3., 4.]), units='kg/s')
        p.set_val('radius', 3., units='ft')
        p.set_val('length', 3., units='ft')
        p.set_val('T_inf', np.array([100., 200., 250., 300., 200.]), units='K')
        p.set_val('insulation_thickness', 5., units='inch')
        p.set_val('LH2_mass_integrator.duration', 1, units='min')

        p.run_model()

        assert_near_equal(p.get_val('W_LH2', units='kg'), np.array([198.53869781, 191.03487744,
                          168.53002484, 131.02487675, 78.52132717]), tolerance=1e-9)
        assert_near_equal(p.get_val('weight', units='kg'), np.array([323.33781373, 315.83399336,
                          293.32914076, 255.82399267, 203.32044309]), tolerance=1e-9)
        assert_near_equal(p.get_val('m_boil_off', units='kg/s'), np.array([1.370775640222e-04,
                          2.972846236531e-04, 3.415420179051e-04, 3.366850729953e-04,
                          1.412654526302e-04]), tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)


if __name__ == "__main__":
    unittest.main()
