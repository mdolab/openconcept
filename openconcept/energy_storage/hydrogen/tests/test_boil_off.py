from __future__ import division
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
import openmdao.api as om
from openconcept.energy_storage.hydrogen.boil_off import *


class SimpleBoilOffTestCase(unittest.TestCase):
    def test_defaults(self):
        p = om.Problem()
        p.model.linear_solver = om.DirectSolver()
        p.model = SimpleBoilOff()
        p.setup(force_alloc_complex=True)

        p.run_model()

        assert_near_equal(p.get_val("m_boil_off", units="kg/s"), 2.239180280883e-04, tolerance=1e-9)

        partials = p.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)

    def test_vectorized(self):
        nn = 5
        p = om.Problem()
        p.model.linear_solver = om.DirectSolver()
        p.model = SimpleBoilOff(num_nodes=nn)
        p.setup(force_alloc_complex=True)

        p.run_model()

        assert_near_equal(p.get_val("m_boil_off", units="kg/s"), 2.239180280883e-04 * np.ones(nn), tolerance=1e-9)

        partials = p.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)

    def test_adding_heat(self):
        nn = 5
        p = om.Problem()
        p.model.linear_solver = om.DirectSolver()
        p.model = SimpleBoilOff(num_nodes=nn)
        p.setup(force_alloc_complex=True)

        p.set_val("LH2_heat_added", np.linspace(1.0, 10.0, nn), units="W")

        p.run_model()

        assert_near_equal(
            p.get_val("m_boil_off", units="kg/s"),
            np.array([0.0002261572084, 0.000231195364, 0.0002362335196, 0.0002412716753, 0.0002463098309]),
            tolerance=1e-9,
        )

        partials = p.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)


class LiquidHeightTestCase(unittest.TestCase):
    def setUp(self):
        self.nn = nn = 7
        self.p = p = om.Problem()
        p.model.add_subsystem("model", LiquidHeight(num_nodes=nn), promotes=["*"])
        p.model.nonlinear_solver = om.NewtonSolver(
            atol=1e-14, rtol=1e-14, solve_subsystems=True, iprint=2, err_on_non_converge=True
        )
        p.model.linear_solver = om.DirectSolver()
        p.setup(force_alloc_complex=True)

    def test_simple(self):
        r = 0.5
        L = 0.3

        # Define height and work backwards to fill level so we
        # can recompute it and check against the original height
        off = 1e-3
        theta = np.linspace(off, 2 * np.pi - off, self.nn)
        h = r * (1 - np.cos(theta / 2))
        V_fill = r**2 / 2 * (theta - np.sin(theta)) * L + np.pi * h**2 / 3 * (3 * r - h)
        V_tank = 4 / 3 * np.pi * r**3 + np.pi * r**2 * L
        fill = V_fill / V_tank

        self.p.set_val("fill_level", fill)
        self.p.set_val("radius", r, units="m")
        self.p.set_val("length", L, units="m")

        self.p.run_model()

        assert_near_equal(self.p.get_val("h_liq", units="m"), h, tolerance=1e-8)

    def test_derivatives(self):
        self.p.set_val("fill_level", np.linspace(0.1, 0.9, self.nn))
        self.p.set_val("radius", 0.5, units="m")
        self.p.set_val("length", 1.2, units="m")

        self.p.run_model()

        partials = self.p.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)


class BoilOffGeometryTestCase(unittest.TestCase):
    def setup_model(self, nn):
        self.p = p = om.Problem()
        p.model.add_subsystem("model", BoilOffGeometry(num_nodes=nn), promotes=["*"])
        p.setup(force_alloc_complex=True)

        self.r = 0.5
        self.L = 0.3

        self.p.set_val("radius", self.r, units="m")
        self.p.set_val("length", self.L, units="m")

    def test_empty(self):
        self.setup_model(1)

        self.p.set_val("h_liq", 0, units="m")

        self.p.run_model()

        A_tank = 4 * np.pi * self.r**2 + 2 * np.pi * self.r * self.L

        assert_near_equal(self.p.get_val("A_interface", units="m**2").item(), 0.0, tolerance=1e-8)
        assert_near_equal(self.p.get_val("L_interface", units="m").item(), 0.0, tolerance=1e-8)
        assert_near_equal(self.p.get_val("A_wet", units="m**2").item(), 0.0, tolerance=1e-8)
        assert_near_equal(self.p.get_val("A_dry", units="m**2").item(), A_tank, tolerance=1e-8)

    def test_full(self):
        self.setup_model(1)

        self.p.set_val("h_liq", 2 * self.r, units="m")

        self.p.run_model()

        A_tank = 4 * np.pi * self.r**2 + 2 * np.pi * self.r * self.L

        assert_near_equal(self.p.get_val("A_interface", units="m**2").item(), 0.0, tolerance=1e-8)
        assert_near_equal(self.p.get_val("L_interface", units="m").item(), 0.0, tolerance=1e-8)
        assert_near_equal(self.p.get_val("A_wet", units="m**2").item(), A_tank, tolerance=1e-8)
        assert_near_equal(self.p.get_val("A_dry", units="m**2").item(), 0.0, tolerance=1e-8)

    def test_half(self):
        self.setup_model(1)

        self.p.set_val("h_liq", self.r, units="m")

        self.p.run_model()

        A_tank = 4 * np.pi * self.r**2 + 2 * np.pi * self.r * self.L

        assert_near_equal(
            self.p.get_val("A_interface", units="m**2").item(),
            np.pi * self.r**2 + 2 * self.r * self.L,
            tolerance=1e-8,
        )
        assert_near_equal(self.p.get_val("L_interface", units="m").item(), 2 * self.r, tolerance=1e-8)
        assert_near_equal(self.p.get_val("A_wet", units="m**2").item(), A_tank / 2, tolerance=1e-8)
        assert_near_equal(self.p.get_val("A_dry", units="m**2").item(), A_tank / 2, tolerance=1e-8)

    def test_regression(self):
        nn = 5
        self.setup_model(nn)

        self.p.set_val("h_liq", np.linspace(0, 2 * self.r, nn), units="m")

        self.p.run_model()

        A_wet = np.array([0.0, 1.09955743, 2.04203522, 2.98451302, 4.08407045])
        A_tank = 4 * np.pi * self.r**2 + 2 * np.pi * self.r * self.L

        assert_near_equal(
            self.p.get_val("A_interface", units="m**2"),
            np.array([0.0, 0.84885624, 1.08539816, 0.84885624, 0.0]),
            tolerance=1e-8,
        )
        assert_near_equal(
            self.p.get_val("L_interface", units="m"), np.array([0.0, 0.8660254, 1.0, 0.8660254, 0.0]), tolerance=1e-8
        )
        assert_near_equal(self.p.get_val("A_wet", units="m**2"), A_wet, tolerance=1e-8)
        assert_near_equal(self.p.get_val("A_dry", units="m**2"), A_tank - A_wet, tolerance=1e-8)

    def test_derivatives(self):
        nn = 7
        self.setup_model(nn)

        off = 1e-6
        self.p.set_val("h_liq", np.linspace(off, 2 * self.r - off, nn), units="m")

        partials = self.p.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)


if __name__ == "__main__":
    unittest.main()
