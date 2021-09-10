from __future__ import division
import unittest
from matplotlib import pyplot
import numpy as np
from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.api import Problem, NewtonSolver, DirectSolver, ScipyOptimizeDriver
from openconcept.components.hydrogen.tank.LH2_tank import *

class LH2TankTestCase(unittest.TestCase):
    def test_simple(self):
        p = Problem()
        p.model = LH2Tank(ullage_P_init=101325., init_fill_level=0.95)
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
        p.model = LH2Tank(num_nodes=nn, ullage_P_init=101325., init_fill_level=0.95)
        p.model.linear_solver = DirectSolver()
        p.model.nonlinear_solver = NewtonSolver()
        p.model.nonlinear_solver.options['err_on_non_converge'] = True
        p.model.nonlinear_solver.options['solve_subsystems'] = True
        p.model.nonlinear_solver.options['maxiter'] = 20
        p.setup(force_alloc_complex=True)

        p.set_val('design_pressure', 2., units='bar')
        p.set_val('m_dot_gas', 0.*np.ones(nn), units='kg/s')
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
        p.model = LH2Tank(num_nodes=nn, ullage_P_init=101325., init_fill_level=0.95)
        p.model.linear_solver = DirectSolver()
        p.model.nonlinear_solver = NewtonSolver()
        p.model.nonlinear_solver.options['err_on_non_converge'] = True
        p.model.nonlinear_solver.options['solve_subsystems'] = True
        p.model.nonlinear_solver.options['maxiter'] = 20
        p.setup(force_alloc_complex=True)

        p.set_val('design_pressure', 2., units='bar')
        p.set_val('m_dot_gas', 0.005*np.ones(nn), units='kg/s')
        p.set_val('m_dot_liq', 0.006*np.ones(nn), units='kg/s')
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

        assert_near_equal(p.get_val('W_LH2', units='kg'), np.array([324.22754621, 248.11209355,
            176.71720854, 110.79433692, 50.6849193]), tolerance=1e-9)
        assert_near_equal(p.get_val('W_GH2', units='kg'), np.array([0.06575232, 19.98745498,
            36.20108999, 47.95521161, 54.90837923]), tolerance=1e-9)
        assert_near_equal(p.get_val('weight', units='kg'), np.array([370.80974416, 314.61599416,
            259.43474416, 205.26599416, 152.10974416]), tolerance=1e-9)
        assert_near_equal(p.get_val('ullage.P', units='bar'), np.array([1.01325, 27.99677337,
            25.91405023, 28.25625532, 28.72683992]), tolerance=1e-9)

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
        p.set_val('m_dot_gas', 0.009*np.ones(nn), units='kg/s')
        p.set_val('m_dot_liq', 0.003*np.ones(nn), units='kg/s')
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

        assert_near_equal(p.get_val('W_LH2', units='kg'), np.array([290.09833082, 225.83086207,
            165.1636196, 108.55825113, 56.2251017]), tolerance=1e-9)
        assert_near_equal(p.get_val('W_GH2', units='kg'), np.array([0.44382814, 5.8175469,
            8.60353936, 8.34015783, 4.81705727]), tolerance=1e-9)
        assert_near_equal(p.get_val('weight', units='kg'), np.array([332.7513486, 273.8575986,
            215.9763486, 159.1075986, 103.2513486]), tolerance=1e-9)
        assert_near_equal(p.get_val('ullage.P', units='bar'), np.array([1.519875, 4.46311316,
            4.50352318, 3.82109948, 2.02291237]), tolerance=1e-8)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_optimization(self):
        nn = 9
        p = Problem()
        p.model = LH2Tank(num_nodes=nn, safety_factor=2.25, init_fill_level=0.9, ullage_T_init=50, ullage_P_init=101325.)
        p.model.linear_solver = DirectSolver()
        p.model.nonlinear_solver = NewtonSolver()
        p.model.nonlinear_solver.options['solve_subsystems'] = True
        # p.model.nonlinear_solver.options['err_on_non_converge'] = True
        p.model.nonlinear_solver.options['maxiter'] = 20
       

        p.model.add_objective('mass_integ.GH2_vent_final')
        p.model.add_design_var('m_dot_vent_start', lower=0.)
        p.model.add_design_var('m_dot_vent_end', lower=0.)
        p.model.add_design_var('LH2_heat_added_start', lower=0.)
        p.model.add_design_var('LH2_heat_added_end', lower=0.)
        p.model.add_constraint('ullage_P_residual', lower=0., scaler=1e-7)
        p.model.add_constraint('W_GH2', lower=0.)

        p.setup(force_alloc_complex=True)

        p.set_val('design_pressure', 2., units='bar')
        p.set_val('m_dot_gas', 0.01*np.ones(nn), units='kg/s')
        p.set_val('m_dot_liq', np.zeros(nn), units='kg/s')
        p.set_val('radius', 1., units='m')
        p.set_val('length', .2, units='m')
        p.set_val('T_inf', np.linspace(300., 300., nn), units='K')
        p.set_val('insulation_thickness', .1, units='inch')
        p.set_val('duration', 3, units='h')
 
        # If the SNOPT optimizer is installed use it for this test,
        # otherwise skip the test (SciPy's SLSQP is inconsistent)
        try:
            p.driver = om.pyOptSparseDriver()
            p.driver.options['optimizer'] = 'SNOPT'
            p.run_driver()
        except:
            self.skipTest("SNOPT optimizer and pyOptSparseDriver not installed")

        assert_near_equal(p.get_val('W_LH2', units='kg'), np.array([307.16293852,
            281.37457711, 257.15538659, 234.52996554, 213.50290057,
            194.05103704, 176.12650805, 159.66356383, 144.5850991]), tolerance=1e-9)
        assert_near_equal(p.get_val('W_GH2', units='kg'), np.array([0.23670834, 1.23375388, 2.04429119,
            2.64372168, 3.02745875, 3.21865703, 3.26518344, 3.23278773, 3.19857519]), tolerance=1e-8)
        assert_near_equal(p.get_val('weight', units='kg'), np.array([351.76226195, 326.97094608,
            303.56229287, 281.53630231, 260.89297441, 241.63230917, 223.75430658,
            207.25896665, 192.14628938]), tolerance=1e-9)
        assert_near_equal(p.get_val('ullage_P_residual', units='bar'), np.array([9.86750000e-01,
            5.11545668e-01, 2.09918905e-01, 1.66351753e-02, 7.41632940e-11, 1.83171164e-02,
            3.73230905e-02, 4.25445171e-02, -3.44565304e-09]), tolerance=1e-8)
        assert_near_equal(p.get_val('ullage.P', units='bar'), np.array([1.01325, 1.48845433, 1.7900811,
            1.98336482, 2., 1.98168288, 1.96267691, 1.95745548, 2.]), tolerance=1e-8)
        assert_near_equal(p.get_val('ullage.T', units='K'), np.array([50., 24.74012426, 25.21485205,
            27.41226958, 28.89260806, 31.0262446, 33.97829641, 37.63912943, 42.09533993]), tolerance=1e-8)
        assert_near_equal(p.get_val('m_dot_vent_start', units='kg/s'), 0.008876034963, tolerance=1e-8)
        assert_near_equal(p.get_val('m_dot_vent_end', units='kg/s'), 0.000682478477, tolerance=1e-8)
        assert_near_equal(p.get_val('LH2_heat_added_start', units='W'), 0., tolerance=1e-8)
        assert_near_equal(p.get_val('LH2_heat_added_end', units='W'), 0., tolerance=1e-8)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)

if __name__ == "__main__":
    unittest.main()
