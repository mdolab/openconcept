from __future__ import division
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.api import Problem, NewtonSolver, DirectSolver, ScipyOptimizeDriver
from openconcept.components.hydrogen.tank.LH2_tank import *

class LH2TankTestCase(unittest.TestCase):
    def test_simple(self):
        p = Problem()
        p.model = LH2Tank()
        p.model.linear_solver = DirectSolver()
        p.model.nonlinear_solver = NewtonSolver()
        p.model.nonlinear_solver.options['err_on_non_converge'] = True
        p.model.nonlinear_solver.options['solve_subsystems'] = True
        p.model.nonlinear_solver.options['maxiter'] = 20
        p.setup(force_alloc_complex=True)

        p.run_model()

        assert_near_equal(p.get_val('W_LH2', units='kg'), 2678.40146873, tolerance=1e-9)
        assert_near_equal(p.get_val('W_GH2', units='kg'), 0.5431713185, tolerance=1e-9)
        assert_near_equal(p.get_val('weight', units='kg'), 3188.48956232, tolerance=1e-9)
        assert_near_equal(p.get_val('ullage_P_residual', units='Pa'), 198675., tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_vectorized(self):
        nn = 5
        p = Problem()
        p.model = LH2Tank(num_nodes=nn)
        p.model.linear_solver = DirectSolver()
        p.model.nonlinear_solver = NewtonSolver()
        p.model.nonlinear_solver.options['err_on_non_converge'] = True
        p.model.nonlinear_solver.options['solve_subsystems'] = True
        p.model.nonlinear_solver.options['maxiter'] = 20
        p.setup(force_alloc_complex=True)

        p.set_val('design_pressure', 2., units='bar')
        p.set_val('m_dot', 0.*np.ones(nn), units='kg/s')
        p.set_val('radius', 1., units='m')
        p.set_val('length', .2, units='m')
        p.set_val('T_inf', np.linspace(100., 100., nn), units='K')
        p.set_val('insulation_thickness', .1, units='inch')
        p.set_val('duration', 5, units='h')

        p.run_model()

        assert_near_equal(p.get_val('W_LH2', units='kg'), np.array([324.22754621,
            292.74492385, 263.83377006, 237.39924829, 213.32789699]), tolerance=1e-9)
        assert_near_equal(p.get_val('W_GH2', units='kg'), np.array([6.57523175e-02,
            3.15483740e+01, 6.04595248e+01, 8.68940375e+01, 1.10965371e+02]), tolerance=1e-8)
        assert_near_equal(p.get_val('weight', units='kg'), np.array([370.80974416,
            370.80974346, 370.80974044, 370.80973145, 370.8097133]), tolerance=1e-9)
        assert_near_equal(p.get_val('ullage.P', units='bar'), np.array([1.01325,
            157.64154789, 182.03687431, 192.48001122, 199.57467472]), tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_vent_and_heat_add(self):
        nn = 5
        p = Problem()
        p.model = LH2Tank(num_nodes=nn)
        p.model.linear_solver = DirectSolver()
        p.model.nonlinear_solver = NewtonSolver()
        p.model.nonlinear_solver.options['err_on_non_converge'] = True
        p.model.nonlinear_solver.options['solve_subsystems'] = True
        p.model.nonlinear_solver.options['maxiter'] = 20
        p.setup(force_alloc_complex=True)

        p.set_val('design_pressure', 2., units='bar')
        p.set_val('m_dot', 0.01*np.ones(nn), units='kg/s')
        p.set_val('radius', 1., units='m')
        p.set_val('length', .2, units='m')
        p.set_val('T_inf', np.linspace(300., 300., nn), units='K')
        p.set_val('insulation_thickness', .1, units='inch')
        p.set_val('m_dot_vent_start', 0.01, units='kg/s')
        p.set_val('m_dot_vent_end', 0.0085, units='kg/s')
        p.set_val('LH2_heat_added_start', 1000., units='W')
        p.set_val('LH2_heat_added_end', 5000., units='W')
        p.set_val('duration', 3, units='h')

        p.run_model()

        assert_near_equal(p.get_val('W_LH2', units='kg'), np.array([324.22754621,
            263.49284599, 204.91502886, 149.30863368, 97.20622145]), tolerance=1e-9)
        assert_near_equal(p.get_val('W_GH2', units='kg'), np.array([0.06575231751,
            7.30670254, 13.40326967, 17.54091486, 19.18707708]), tolerance=1e-9)
        assert_near_equal(p.get_val('weight', units='kg'), np.array([370.80974416,
            317.31599416, 264.83474416, 213.36599416, 162.90974416]), tolerance=1e-9)
        assert_near_equal(p.get_val('ullage.P', units='bar'), np.array([1.01325,
            8.43770982, 8.51901306, 9.61319044, 8.98249862]), tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)

    def test_options(self):
        nn = 5
        p = Problem()
        p.model = LH2Tank(num_nodes=nn, safety_factor=1.5, init_fill_level=0.85, ullage_T_init=60., ullage_P_init=1.5*101325)
        p.model.linear_solver = DirectSolver()
        p.model.nonlinear_solver = NewtonSolver()
        p.model.nonlinear_solver.options['err_on_non_converge'] = True
        p.model.nonlinear_solver.options['solve_subsystems'] = True
        p.model.nonlinear_solver.options['maxiter'] = 20
        p.setup(force_alloc_complex=True)

        p.set_val('design_pressure', 2., units='bar')
        p.set_val('m_dot', 0.01*np.ones(nn), units='kg/s')
        p.set_val('radius', 1., units='m')
        p.set_val('length', .2, units='m')
        p.set_val('T_inf', np.linspace(300., 300., nn), units='K')
        p.set_val('insulation_thickness', .1, units='inch')
        p.set_val('m_dot_vent_start', 0.01, units='kg/s')
        p.set_val('m_dot_vent_end', 0.0085, units='kg/s')
        p.set_val('LH2_heat_added_start', 1000., units='W')
        p.set_val('LH2_heat_added_end', 5000., units='W')
        p.set_val('duration', 3, units='h')

        p.run_model()

        assert_near_equal(p.get_val('W_LH2', units='kg'), np.array([290.09833082,
            233.44580731, 179.11000665, 127.61113991, 79.26614387]), tolerance=1e-9)
        assert_near_equal(p.get_val('W_GH2', units='kg'), np.array([0.44382814,
            3.60260166, 5.45715232, 5.48726906, 3.3760151]), tolerance=1e-9)
        assert_near_equal(p.get_val('weight', units='kg'), np.array([332.7513486,
            279.2575986, 226.7763486, 175.3075986, 124.8513486]), tolerance=1e-9)
        assert_near_equal(p.get_val('ullage.P', units='bar'), np.array([1.519875,
            2.72061633, 2.9359135, 2.58837453, 1.43995542]), tolerance=1e-8)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_optimization(self):
        nn = 9
        p = Problem()
        p.model = LH2Tank(num_nodes=nn, safety_factor=2.25, init_fill_level=0.9, ullage_T_init=50)
        p.model.linear_solver = DirectSolver()
        p.model.nonlinear_solver = NewtonSolver()
        p.model.nonlinear_solver.options['solve_subsystems'] = True
        # p.model.nonlinear_solver.options['err_on_non_converge'] = True
        p.model.nonlinear_solver.options['maxiter'] = 20
        
        p.driver = ScipyOptimizeDriver()
        p.driver.options['optimizer'] = 'SLSQP'

        p.model.add_objective('mass_integ.GH2_vent_final')
        p.model.add_design_var('m_dot_vent_start', lower=0.)
        p.model.add_design_var('m_dot_vent_end', lower=0.)
        p.model.add_design_var('LH2_heat_added_start', lower=0.)
        p.model.add_design_var('LH2_heat_added_end', lower=0.)
        p.model.add_constraint('ullage_P_residual', lower=0., scaler=1e-7)
        p.model.add_constraint('W_GH2', lower=0.)

        p.setup(force_alloc_complex=True)

        p.set_val('design_pressure', 2., units='bar')
        p.set_val('m_dot', 0.01*np.ones(nn), units='kg/s')
        p.set_val('radius', 1., units='m')
        p.set_val('length', .2, units='m')
        p.set_val('T_inf', np.linspace(300., 300., nn), units='K')
        p.set_val('insulation_thickness', .1, units='inch')
        p.set_val('duration', 3, units='h')

        p.run_driver()

        assert_near_equal(p.get_val('W_LH2', units='kg'), np.array([307.16293852,
            281.37457711, 257.15538659, 234.52996554, 213.50290057,
            194.05103704, 176.12650805, 159.66356383, 144.5850991]), tolerance=1e-9)
        assert_near_equal(p.get_val('W_GH2', units='kg'), np.array([0.23670834,
            1.2291291, 2.03486867, 2.62932847, 3.00792188, 3.19380354,
            3.23484037, 3.19678212, 3.15673409]), tolerance=1e-9)
        assert_near_equal(p.get_val('weight', units='kg'), np.array([351.76226195,
            326.9663213, 303.55287035, 281.5219091, 260.87343754, 241.60745568,
            223.72396351, 207.22296104, 192.10444827]), tolerance=1e-9)
        assert_near_equal(p.get_val('ullage_P_residual', units='bar'), np.array([0.98675,
            0.51782939, 0.21889399, 0.02843507, 0.01389464, 0.03447158, 0.05635223,
            0.06509295, 0.02684462]), tolerance=1e-8)
        assert_near_equal(p.get_val('ullage.P', units='bar'), np.array([1.01325,
            1.48217061, 1.78110601, 1.97156493, 1.98610536, 1.96552842, 1.94364777,
            1.93490705, 1.97315538]), tolerance=1e-8)
        assert_near_equal(p.get_val('ullage.T', units='K'), np.array([50.,
            24.72837567, 25.20460278, 27.39834702, 28.8782397, 31.01279343,
            33.96448875, 37.62460363, 42.08078915]), tolerance=1e-8)
        assert_near_equal(p.get_val('m_dot_vent_start', units='kg/s'), 0.0088793966651, tolerance=1e-8)
        assert_near_equal(p.get_val('m_dot_vent_end', units='kg/s'), 0.000686865127, tolerance=1e-8)
        assert_near_equal(p.get_val('LH2_heat_added_start', units='W'), 7.65753052e-12, tolerance=1e-8)
        assert_near_equal(p.get_val('LH2_heat_added_end', units='W'), 1.06952593e-11, tolerance=1e-8)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)

if __name__ == "__main__":
    unittest.main()
