from __future__ import division
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.api import Problem, NewtonSolver, DirectSolver
from openconcept.components.hydrogen.tank.thermal import *

class HeatTransferTestCase(unittest.TestCase):
    def test_simple(self):
        p = Problem()
        p.model = HeatTransfer()
        p.model.linear_solver = DirectSolver()
        p.model.nonlinear_solver = NewtonSolver()
        p.model.nonlinear_solver.options['solve_subsystems'] = True
        p.model.nonlinear_solver.options['err_on_non_converge'] = True
        p.setup(force_alloc_complex=True)

        p.set_val('radius', 2., units='m')
        p.set_val('length', .5, units='m')
        p.set_val('T_liquid', 20., units='K')
        p.set_val('T_inf', 300., units='K')
        p.set_val('composite_thickness', 6., units='inch')
        p.set_val('insulation_thickness', 5., units='inch')
        p.set_val('fill_level', 0.5)

        p.run_model()

        # Check that it has converged
        assert_near_equal(p.get_val('Q_wall.heat_into_walls', units='W'),
                          p.get_val('Q_LH2.heat_total', units='W'), tolerance=1e-6)

        assert_near_equal(p.get_val('heat_into_liquid', units='W'), 819.94055824, tolerance=1e-9)
        assert_near_equal(p.get_val('heat_into_vapor', units='W'), 273.31351941, tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_vectorized(self):
        nn = 5
        p = Problem()
        p.model = HeatTransfer(num_nodes=nn)
        p.model.linear_solver = DirectSolver()
        p.model.nonlinear_solver = NewtonSolver()
        p.model.nonlinear_solver.options['solve_subsystems'] = True
        p.model.nonlinear_solver.options['err_on_non_converge'] = True
        p.setup(force_alloc_complex=True)

        p.set_val('radius', 2., units='m')
        p.set_val('length', .5, units='m')
        p.set_val('T_liquid', 20.*np.ones(nn), units='K')
        p.set_val('T_inf', 300.*np.ones(nn), units='K')
        p.set_val('composite_thickness', 6., units='inch')
        p.set_val('insulation_thickness', 5., units='inch')
        p.set_val('fill_level', 0.5*np.ones(nn))

        p.run_model()

        # Check that it has converged
        assert_near_equal(p.get_val('Q_wall.heat_into_walls', units='W'),
                          p.get_val('Q_LH2.heat_total', units='W'), tolerance=1e-6)

        assert_near_equal(p.get_val('heat_into_liquid', units='W'), 819.94055824*np.ones(nn), tolerance=1e-9)
        assert_near_equal(p.get_val('heat_into_vapor', units='W'), 273.31351941*np.ones(nn), tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_big_range(self):
        nn = 7
        p = Problem()
        p.model = HeatTransfer(num_nodes=nn)
        p.model.linear_solver = DirectSolver()
        p.model.nonlinear_solver = NewtonSolver()
        p.model.nonlinear_solver.options['solve_subsystems'] = True
        p.model.nonlinear_solver.options['maxiter'] = 20
        p.model.nonlinear_solver.options['err_on_non_converge'] = True
        p.setup(force_alloc_complex=True)

        p.set_val('radius', 2., units='m')
        p.set_val('length', .5, units='m')
        p.set_val('T_liquid', 20.*np.ones(nn), units='K')
        p.set_val('T_inf', np.linspace(60, 500, nn), units='K')
        p.set_val('composite_thickness', 6., units='inch')
        p.set_val('insulation_thickness', 5., units='inch')
        p.set_val('fill_level', np.linspace(0.95, 0.2, nn))

        p.run_model()

        # Check that it has converged
        assert_near_equal(p.get_val('Q_wall.heat_into_walls', units='W'),
                          p.get_val('Q_LH2.heat_total', units='W'), tolerance=1e-6)

        assert_near_equal(p.get_val('heat_into_liquid', units='W'), np.array([223.38323324,
                          542.00909991, 758.03210165, 873.624113, 881.83826926,
                          779.61896958, 567.28669575]), tolerance=1e-9)
        assert_near_equal(p.get_val('heat_into_vapor', units='W'), np.array([5.59857727e-01,
                          1.71234338e+01, 7.49702079e+01, 1.92583805e+02, 3.82445988e+02,
                          6.52516910e+02, 1.00850968e+03]), tolerance=1e-8)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)


class FillLevelCalcTestCase(unittest.TestCase):
    def test_defaults(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model = FillLevelCalc()
        p.setup(force_alloc_complex=True)

        p.run_model()

        assert_near_equal(p.get_val('fill_level'), .3546891722881, tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_vectorized(self):
        nn = 5
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model = FillLevelCalc(num_nodes=nn)
        p.setup(force_alloc_complex=True)

        p.run_model()

        assert_near_equal(p.get_val('fill_level'), .3546891722881*np.ones(nn), tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_zero_fill(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model = FillLevelCalc()
        p.setup(force_alloc_complex=True)

        p.set_val('W_liquid', 0., units='kg')

        p.run_model()

        assert_near_equal(p.get_val('fill_level'), 0., tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_full(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model = FillLevelCalc()
        p.setup(force_alloc_complex=True)

        p.set_val('radius', 2, units='m')
        p.set_val('length', .5, units='m')
        p.set_val('density', 1., units='kg/m**3')
        p.set_val('W_liquid', 4/3*np.pi*2**3 + np.pi*2**2*0.5, units='kg')

        p.run_model()

        assert_near_equal(p.get_val('fill_level'), 1., tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)


class COPVThermalResistanceTestCase(unittest.TestCase):
    def test_defaults(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model = COPVThermalResistance()
        p.setup(force_alloc_complex=True)

        p.run_model()

        assert_near_equal(p.get_val('thermal_resistance', units='K/W'), 0.7494029867002, tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_sphere(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model = COPVThermalResistance()
        p.setup(force_alloc_complex=True)

        p.set_val('length', 0., units='m')

        p.run_model()

        assert_near_equal(p.get_val('thermal_resistance', units='K/W'), 2.0047489854, tolerance=1e-9)

        partials = p.check_partials(method='cs', compact_print=True)
        assert_check_partials(partials)
    
    def test_only_liner(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model = COPVThermalResistance(liner_cond=200., liner_thickness=1.6e-3)
        p.setup(force_alloc_complex=True)

        p.set_val('radius', 0.7, units='m')
        p.set_val('length', 1.6, units='m')
        p.set_val('composite_thickness', 0., units='m')
        p.set_val('insulation_thickness', 0., units='m')

        p.run_model()

        assert_near_equal(p.get_val('thermal_resistance', units='K/W'), 6.05290104e-7, tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_only_composite(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model = COPVThermalResistance(liner_thickness=0., composite_cond=0.86)
        p.setup(force_alloc_complex=True)

        p.set_val('radius', 1., units='m')
        p.set_val('length', 1., units='m')
        p.set_val('composite_thickness', 0.6, units='m')
        p.set_val('insulation_thickness', 0., units='m')

        p.run_model()

        assert_near_equal(p.get_val('thermal_resistance', units='K/W'), 2.480424483950e-02, tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_only_insulation(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model = COPVThermalResistance(liner_thickness=0., insulation_cond=0.032)
        p.setup(force_alloc_complex=True)

        p.set_val('radius', 1., units='m')
        p.set_val('length', 1., units='m')
        p.set_val('composite_thickness', 0., units='m')
        p.set_val('insulation_thickness', 0.3, units='m')

        p.run_model()

        assert_near_equal(p.get_val('thermal_resistance', units='K/W'), 0.39858372, tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)

class COPVHeatFromEnvironmentIntoTankWallsTestCase(unittest.TestCase):
    def test_defaults(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model = COPVHeatFromEnvironmentIntoTankWalls()
        p.setup(force_alloc_complex=True)

        p.run_model()

        assert_near_equal(p.get_val('heat_into_walls', units='W'), 19331.92474297, tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_vectorized(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model = COPVHeatFromEnvironmentIntoTankWalls(num_nodes=3)
        p.setup(force_alloc_complex=True)

        p.set_val('T_surface', np.array([150., 200., 250.]), units='K')
        p.set_val('T_inf', np.array([300., 290., 280.]), units='K')

        p.run_model()

        assert_near_equal(p.get_val('heat_into_walls', units='W'),
                          np.array([14250.44520777, 8313.22871615, 2487.49146981]), tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_radiation_only(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model = COPVHeatFromEnvironmentIntoTankWalls(air_cond=0)
        p.setup(force_alloc_complex=True)

        p.run_model()

        assert_near_equal(p.get_val('heat_into_walls', units='W'), 3671.95195277, tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_convection_only(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model = COPVHeatFromEnvironmentIntoTankWalls(surface_emissivity=0)
        p.setup(force_alloc_complex=True)

        p.run_model()

        assert_near_equal(p.get_val('heat_into_walls', units='W'), 15659.97279019, tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_T_inf_less_than_T_surface(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model = COPVHeatFromEnvironmentIntoTankWalls(num_nodes=4)
        p.setup(force_alloc_complex=True)

        # Mix in one T_inf > T_surf to make sure indexing is correct
        p.set_val('T_surface', np.array([300., 300., 300., 300.]), units='K')
        p.set_val('T_inf', np.array([200., 299., 305., 290.]), units='K')

        p.run_model()

        assert_near_equal(p.get_val('heat_into_walls', units='W'),
                          np.array([-14252.58526491, -65.76729809, 381.65602934,
                                    -806.02771188]), tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)

class COPVHeatFromWallsIntoPropellantTestCase(unittest.TestCase):
    def test_defaults(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model = COPVHeatFromWallsIntoPropellant()
        p.setup(force_alloc_complex=True)

        p.run_model()

        assert_near_equal(p.get_val('heat_into_liquid', units='W'), 40., tolerance=1e-9)
        assert_near_equal(p.get_val('heat_into_vapor', units='W'), 13.33333333, tolerance=1e-9)
        assert_near_equal(p.get_val('heat_total', units='W'), 53.3333333333, tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_vectorized(self):
        nn = 5
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model = COPVHeatFromWallsIntoPropellant(num_nodes=nn)
        p.setup(force_alloc_complex=True)

        p.run_model()

        assert_near_equal(p.get_val('heat_into_liquid', units='W'), 40.*np.ones(nn), tolerance=1e-9)
        assert_near_equal(p.get_val('heat_into_vapor', units='W'), 13.33333333*np.ones(nn), tolerance=1e-9)
        assert_near_equal(p.get_val('heat_total', units='W'), 53.3333333333*np.ones(nn), tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_nearly_full(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model = COPVHeatFromWallsIntoPropellant()
        p.setup(force_alloc_complex=True)

        p.set_val('fill_level', 0.99)

        p.run_model()

        assert_near_equal(p.get_val('heat_into_liquid', units='W'), 79.2, tolerance=1e-9)
        assert_near_equal(p.get_val('heat_into_vapor', units='W'), 7.9207920792e-3, tolerance=1e-9)
        assert_near_equal(p.get_val('heat_total', units='W'), 79.20792079, tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_nearly_empty(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model = COPVHeatFromWallsIntoPropellant()
        p.setup(force_alloc_complex=True)

        p.set_val('fill_level', 0.01)

        p.run_model()

        assert_near_equal(p.get_val('heat_into_liquid', units='W'), 0.8, tolerance=1e-9)
        assert_near_equal(p.get_val('heat_into_vapor', units='W'), 39.40100503, tolerance=1e-9)
        assert_near_equal(p.get_val('heat_total', units='W'), 40.20100503, tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)


if __name__ == "__main__":
    unittest.main()
