from __future__ import division
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.api import Problem, NewtonSolver, DirectSolver
from openconcept.components.hydrogen_tank import TankCompositeThickness

class TankCompositeThicknessTestCase(unittest.TestCase):
    def test_defaults(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model = TankCompositeThickness()
        p.setup(force_alloc_complex=True)

        p.set_val('alpha', 45., units='deg')
        p.set_val('design_pressure', 70e6, units='Pa')
        p.set_val('radius', .5, units='m')
        p.set_val('length', 2., units='m')

        p.run_model()

        assert_near_equal(p.get_val('thickness', units='m'), 0.0383959044, tolerance=1e-9)
        assert_near_equal(p.get_val('composite_weight', units='kg'), 571.7591408, tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_zero_pressure(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model = TankCompositeThickness()
        p.setup(force_alloc_complex=True)

        p.set_val('alpha', 45., units='deg')
        p.set_val('design_pressure', 0, units='Pa')
        p.set_val('radius', .5, units='m')
        p.set_val('length', 2., units='m')

        p.run_model()

        assert_near_equal(p.get_val('thickness', units='m'), 0, tolerance=1e-9)
        assert_near_equal(p.get_val('composite_weight', units='kg'), 0, tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)

    def test_zero_radius(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model = TankCompositeThickness()
        p.setup(force_alloc_complex=True)

        p.set_val('alpha', 45., units='deg')
        p.set_val('design_pressure', 70e6, units='Pa')
        p.set_val('radius', 0., units='m')
        p.set_val('length', 2., units='m')

        p.run_model()

        assert_near_equal(p.get_val('thickness', units='m'), 0, tolerance=1e-9)
        assert_near_equal(p.get_val('composite_weight', units='kg'), 0, tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
    
    def test_different_options(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model = TankCompositeThickness(safety_factor=2.,
                                         yield_stress = 8e9,
                                         density=1e3,
                                         fiber_volume_fraction=0.5)
        p.setup(force_alloc_complex=True)

        p.set_val('alpha', 45., units='deg')
        p.set_val('design_pressure', 70e6, units='Pa')
        p.set_val('radius', .5, units='m')
        p.set_val('length', 2., units='m')

        p.run_model()

        assert_near_equal(p.get_val('thickness', units='mm'), 26.25, tolerance=1e-9)
        assert_near_equal(p.get_val('composite_weight', units='kg'), 247.4004215, tolerance=1e-9)

        partials = p.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)


if __name__ == "__main__":
    unittest.main()