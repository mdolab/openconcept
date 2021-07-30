from __future__ import division
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.api import Problem, NewtonSolver, DirectSolver
from openconcept.components.hydrogen.tank.thermal import *

class COPVThermalResistanceTestCase(unittest.TestCase):
    def test_defaults(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model = COPVThermalResistance()
        p.setup(force_alloc_complex=True)

        p.run_model()

        assert_near_equal(p.get_val('R_cylinder', units='K/W'), 1.19677354195816, tolerance=1e-9)
        assert_near_equal(p.get_val('R_sphere', units='K/W'), 2.00474898539084, tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
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

        assert_near_equal(p.get_val('R_cylinder', units='K/W'), 1.13552377444205e-6, tolerance=1e-9)
        assert_near_equal(p.get_val('R_sphere', units='K/W'), 1.29626114262823e-6, tolerance=1e-9)

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

        assert_near_equal(p.get_val('R_cylinder', units='K/W'), 0.08698069868092, tolerance=1e-9)
        assert_near_equal(p.get_val('R_sphere', units='K/W'), 0.03469947887178, tolerance=1e-9)

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

        assert_near_equal(p.get_val('R_cylinder', units='K/W'), 1.30489279939595, tolerance=1e-9)
        assert_near_equal(p.get_val('R_sphere', units='K/W'), 0.57387599672558, tolerance=1e-9)

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

if __name__ == "__main__":
    unittest.main()
