from __future__ import division
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.api import Problem, NewtonSolver, DirectSolver
from openconcept.energy_storage.hydrogen.structural import CompositeOverwrap, LinerWeight, InsulationWeight


class CompositeOverwrapTestCase(unittest.TestCase):
    def test_defaults(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model = CompositeOverwrap()
        p.setup(force_alloc_complex=True)

        p.set_val("design_pressure", 70e6, units="Pa")
        p.set_val("radius", 0.5, units="m")
        p.set_val("length", 2.0, units="m")

        p.run_model()

        assert_near_equal(p.get_val("thickness", units="m"), 0.068986481277726, tolerance=1e-9)
        assert_near_equal(p.get_val("weight", units="kg"), 1123.952851354024688, tolerance=1e-9)

        partials = p.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)

    def test_zero_pressure(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model = CompositeOverwrap()
        p.setup(force_alloc_complex=True)

        p.set_val("design_pressure", 0, units="Pa")
        p.set_val("radius", 0.5, units="m")
        p.set_val("length", 2.0, units="m")

        p.run_model()

        assert_near_equal(p.get_val("thickness", units="m"), 0, tolerance=1e-9)
        assert_near_equal(p.get_val("weight", units="kg"), 0, tolerance=1e-9)

        partials = p.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)

    def test_zero_radius(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model = CompositeOverwrap()
        p.setup(force_alloc_complex=True)

        p.set_val("design_pressure", 70e6, units="Pa")
        p.set_val("radius", 0.0, units="m")
        p.set_val("length", 2.0, units="m")

        p.run_model()

        assert_near_equal(p.get_val("thickness", units="m"), 0, tolerance=1e-9)
        assert_near_equal(p.get_val("weight", units="kg"), 0, tolerance=1e-9)

        partials = p.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)

    def test_different_options(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model = CompositeOverwrap(safety_factor=2.0, yield_stress=8e9, density=1e3, fiber_volume_fraction=0.5)
        p.setup(force_alloc_complex=True)

        p.set_val("design_pressure", 70e6, units="Pa")
        p.set_val("radius", 0.5, units="m")
        p.set_val("length", 2.0, units="m")

        p.run_model()

        assert_near_equal(p.get_val("thickness", units="mm"), 26.25, tolerance=1e-9)
        assert_near_equal(p.get_val("weight", units="kg"), 256.1352026, tolerance=1e-9)

        partials = p.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)


class LinerWeightTestCase(unittest.TestCase):
    def test_defaults(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model = LinerWeight()
        p.setup(force_alloc_complex=True)

        p.set_val("radius", 0.5, units="m")
        p.set_val("length", 2.0, units="m")

        p.run_model()

        assert_near_equal(p.get_val("weight", units="kg"), 12.72345025, tolerance=1e-9)

        partials = p.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)

    def test_options(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()

        rho = 42.0
        thickness = 0.7e-3
        p.model = LinerWeight(density=rho, thickness=thickness)

        p.setup(force_alloc_complex=True)

        p.set_val("radius", 1.0, units="m")
        p.set_val("length", 1.0, units="m")

        p.run_model()

        assert_near_equal(p.get_val("weight", units="kg"), 6 * np.pi * rho * thickness, tolerance=1e-9)

        partials = p.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)


class InsulationWeightTestCase(unittest.TestCase):
    def test_defaults(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model = InsulationWeight()
        p.setup(force_alloc_complex=True)

        p.set_val("radius", 0.5, units="m")
        p.set_val("length", 2.0, units="m")
        p.set_val("thickness", 0.05, units="m")

        p.run_model()

        assert_near_equal(p.get_val("weight", units="kg"), 30.12155895, tolerance=1e-9)

        partials = p.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)

    def test_radius(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model = InsulationWeight()
        p.setup(force_alloc_complex=True)

        p.set_val("radius", 0.8, units="m")
        p.set_val("length", 2.0, units="m")
        p.set_val("thickness", 0.05, units="m")

        p.run_model()

        assert_near_equal(p.get_val("weight", units="kg"), 56.1390355, tolerance=1e-9)

        partials = p.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)

    def test_length(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model = InsulationWeight()
        p.setup(force_alloc_complex=True)

        p.set_val("radius", 0.5, units="m")
        p.set_val("length", 3.0, units="m")
        p.set_val("thickness", 0.05, units="m")

        p.run_model()

        assert_near_equal(p.get_val("weight", units="kg"), 39.92222847, tolerance=1e-9)

        partials = p.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)

    def test_thickness(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model = InsulationWeight()
        p.setup(force_alloc_complex=True)

        p.set_val("radius", 0.5, units="m")
        p.set_val("length", 2.0, units="m")
        p.set_val("thickness", 0.5, units="m")

        p.run_model()

        assert_near_equal(p.get_val("weight", units="kg"), 301.69342571, tolerance=1e-9)

        partials = p.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)

    def test_density(self):
        p = Problem()
        p.model.linear_solver = DirectSolver()
        p.model = InsulationWeight(density=42.0)
        p.setup(force_alloc_complex=True)

        p.set_val("radius", 0.5, units="m")
        p.set_val("length", 2.0, units="m")
        p.set_val("thickness", 0.05, units="m")

        p.run_model()

        assert_near_equal(p.get_val("weight", units="kg"), 35.10302534, tolerance=1e-9)

        partials = p.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)


if __name__ == "__main__":
    unittest.main()
