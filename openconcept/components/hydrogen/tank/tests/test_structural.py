from __future__ import division
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.api import Problem, NewtonSolver, DirectSolver
from openconcept.components.hydrogen.tank.structural import CompositeOverwrap, COPVLinerWeight, COPVInsulationWeight

class CompositeOverwrapTestCase(unittest.TestCase):
    def test_defaults(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model = CompositeOverwrap()
        p.setup(force_alloc_complex=True)

        p.set_val('design_pressure', 70e6, units='Pa')
        p.set_val('radius', .5, units='m')
        p.set_val('length', 2., units='m')

        p.run_model()

        assert_near_equal(p.get_val('thickness', units='m'), 0.068986481277726, tolerance=1e-9)
        assert_near_equal(p.get_val('weight', units='kg'), 1123.952851354024688, tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_zero_pressure(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model = CompositeOverwrap()
        p.setup(force_alloc_complex=True)

        p.set_val('design_pressure', 0, units='Pa')
        p.set_val('radius', .5, units='m')
        p.set_val('length', 2., units='m')

        p.run_model()

        assert_near_equal(p.get_val('thickness', units='m'), 0, tolerance=1e-9)
        assert_near_equal(p.get_val('weight', units='kg'), 0, tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)

    def test_zero_radius(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model = CompositeOverwrap()
        p.setup(force_alloc_complex=True)

        p.set_val('design_pressure', 70e6, units='Pa')
        p.set_val('radius', 0., units='m')
        p.set_val('length', 2., units='m')

        p.run_model()

        assert_near_equal(p.get_val('thickness', units='m'), 0, tolerance=1e-9)
        assert_near_equal(p.get_val('weight', units='kg'), 0, tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_different_options(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model = CompositeOverwrap(safety_factor=2.,
                                         yield_stress = 8e9,
                                         density=1e3,
                                         fiber_volume_fraction=0.5)
        p.setup(force_alloc_complex=True)

        p.set_val('design_pressure', 70e6, units='Pa')
        p.set_val('radius', .5, units='m')
        p.set_val('length', 2., units='m')

        p.run_model()

        assert_near_equal(p.get_val('thickness', units='mm'), 26.25, tolerance=1e-9)
        assert_near_equal(p.get_val('weight', units='kg'), 256.1352026, tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)


class COPVLinerWeightTestCase(unittest.TestCase):
    def test_defaults(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model = COPVLinerWeight()
        p.setup(force_alloc_complex=True)
        
        p.set_val('radius', .5, units='m')
        p.set_val('length', 2., units='m')

        p.run_model()

        assert_near_equal(p.get_val('weight', units='kg'), 12.72345025, tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_options(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()

        rho = 42.
        thickness = 0.7e-3
        p.model = COPVLinerWeight(density=rho, thickness=thickness)

        p.setup(force_alloc_complex=True)
        
        p.set_val('radius', 1., units='m')
        p.set_val('length', 1., units='m')

        p.run_model()

        assert_near_equal(p.get_val('weight', units='kg'), 6*np.pi*rho*thickness, tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)


class COPVInsulationWeightTestCase(unittest.TestCase):
    def test_defaults(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model = COPVInsulationWeight()
        p.setup(force_alloc_complex=True)
        
        p.set_val('radius', .5, units='m')
        p.set_val('length', 2., units='m')
        p.set_val('thickness', 0.05, units='m')

        p.run_model()

        assert_near_equal(p.get_val('weight', units='kg'), 16.1520274, tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_radius(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model = COPVInsulationWeight()
        p.setup(force_alloc_complex=True)
        
        p.set_val('radius', .8, units='m')
        p.set_val('length', 2., units='m')
        p.set_val('thickness', 0.05, units='m')

        p.run_model()

        assert_near_equal(p.get_val('weight', units='kg'), 30.37118991, tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_length(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model = COPVInsulationWeight()
        p.setup(force_alloc_complex=True)
        
        p.set_val('radius', .5, units='m')
        p.set_val('length', 3., units='m')
        p.set_val('thickness', 0.05, units='m')

        p.run_model()

        assert_near_equal(p.get_val('weight', units='kg'), 21.44639639, tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_thickness(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model = COPVInsulationWeight()
        p.setup(force_alloc_complex=True)
        
        p.set_val('radius', .5, units='m')
        p.set_val('length', 2., units='m')
        p.set_val('thickness', 0.5, units='m')

        p.run_model()

        assert_near_equal(p.get_val('weight', units='kg'), 268.9203311, tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_density(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model = COPVInsulationWeight(density=42.)
        p.setup(force_alloc_complex=True)
        
        p.set_val('radius', .5, units='m')
        p.set_val('length', 2., units='m')
        p.set_val('thickness', 0.05, units='m')

        p.run_model()

        assert_near_equal(p.get_val('weight', units='kg'), 21.13349379, tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)

if __name__ == "__main__":
    unittest.main()
