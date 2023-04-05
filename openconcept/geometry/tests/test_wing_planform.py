import unittest
import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openconcept.geometry import WingMACTrapezoidal, WingSpan, WingAspectRatio, WingSweepFromSections, WingAreaFromSections


class WingMACTrapezoidalTestCase(unittest.TestCase):
    def test_rectangular(self):
        """
        Test a rectangular wing.
        """
        p = om.Problem()
        p.model.add_subsystem("comp", WingMACTrapezoidal(), promotes=["*"])
        p.setup(force_alloc_complex=True)

        b = 10
        c = 1

        p.set_val("S_ref", b * c, units="m**2")
        p.set_val("AR", b / c)
        p.set_val("taper", 1.0)

        p.run_model()

        assert_near_equal(p.get_val("MAC", units="m"), c)

        p = p.check_partials(method="cs", out_stream=None)
        assert_check_partials(p)

    def test_tapered(self):
        """
        Test a tapered wing.
        """
        p = om.Problem()
        p.model.add_subsystem("comp", WingMACTrapezoidal(), promotes=["*"])
        p.setup(force_alloc_complex=True)

        p.set_val("S_ref", 10, units="m**2")
        p.set_val("AR", 10)
        p.set_val("taper", 0.3)

        p.run_model()

        assert_near_equal(p.get_val("MAC", units="m"), 1.09664694, tolerance=1e-8)

        p = p.check_partials(method="cs", out_stream=None)
        assert_check_partials(p)


class WingSpanTestCase(unittest.TestCase):
    def test(self):
        p = om.Problem()
        p.model.add_subsystem("comp", WingSpan(), promotes=["*"])
        p.setup(force_alloc_complex=True)

        p.set_val("S_ref", 10, units="m**2")
        p.set_val("AR", 2.5)

        p.run_model()

        assert_near_equal(p.get_val("span", units="m"), 5.0)

        p = p.check_partials(method="cs", out_stream=None)
        assert_check_partials(p)


class WingAspectRatioTestCase(unittest.TestCase):
    def test(self):
        p = om.Problem()
        p.model.add_subsystem("comp", WingAspectRatio(), promotes=["*"])
        p.setup(force_alloc_complex=True)

        p.set_val("S_ref", 10, units="m**2")
        p.set_val("span", 5.0, units="m")

        p.run_model()

        assert_near_equal(p.get_val("AR"), 2.5)

        p = p.check_partials(method="cs", out_stream=None)
        assert_check_partials(p)


class WingSweepFromSectionsTestCase(unittest.TestCase):
    def test_no_sweep(self):
        p = om.Problem()
        p.model.add_subsystem("comp", WingSweepFromSections(num_sections=2), promotes=["*"])
        p.setup(force_alloc_complex=True)

        p.set_val("x_LE_sec", [0, 0], units="m")
        p.set_val("y_sec", -1.0, units="m")
        p.set_val("chord_sec", [1.0, 1.0], units="m")

        p.run_model()

        assert_near_equal(p.get_val("c4sweep", units="deg"), 0.0)

    def test_sweep(self):
        p = om.Problem()
        p.model.add_subsystem("comp", WingSweepFromSections(num_sections=2), promotes=["*"])
        p.setup(force_alloc_complex=True)

        p.set_val("x_LE_sec", [1, 0], units="m")
        p.set_val("y_sec", -1.0, units="m")
        p.set_val("chord_sec", [0.5, 1.0], units="m")

        p.run_model()

        assert_near_equal(p.get_val("c4sweep", units="deg"), np.rad2deg(np.arctan(0.875)))

    def test_sweep(self):
        p = om.Problem()
        p.model.add_subsystem("comp", WingSweepFromSections(num_sections=3), promotes=["*"])
        p.setup(force_alloc_complex=True)

        p.set_val("x_LE_sec", [0, 1, 0], units="m")
        p.set_val("y_sec", [-2, -1.0], units="m")
        p.set_val("chord_sec", [0.1, 0.5, 1.0], units="m")

        p.run_model()

        assert_near_equal(p.get_val("c4sweep", units="deg"), 15.78242912, tolerance=1e-8)

    def test_indices(self):
        p = om.Problem()
        p.model.add_subsystem(
            "comp", WingSweepFromSections(num_sections=4, idx_sec_start=1, idx_sec_end=2), promotes=["*"]
        )
        p.setup(force_alloc_complex=True)

        p.set_val("x_LE_sec", [1, 0, 1, 0], units="m")
        p.set_val("y_sec", [-3, -2, -1], units="m")
        p.set_val("chord_sec", [0.1, 1.0, 1.0, 1.5], units="m")

        p.run_model()

        assert_near_equal(p.get_val("c4sweep", units="deg"), -45)


class WingAreaFromSectionsTestCase(unittest.TestCase):
    def test_no_sweep(self):
        p = om.Problem()
        p.model.add_subsystem("comp", WingAreaFromSections(num_sections=2), promotes=["*"])
        p.setup(force_alloc_complex=True)

        p.set_val("y_sec", -1.0, units="m")
        p.set_val("chord_sec", [1.0, 1.0], units="m")

        p.run_model()

        assert_near_equal(p.get_val("S", units="m**2"), 2.0)

    def test_sweep(self):
        p = om.Problem()
        p.model.add_subsystem("comp", WingAreaFromSections(num_sections=2), promotes=["*"])
        p.setup(force_alloc_complex=True)

        p.set_val("y_sec", -1.0, units="m")
        p.set_val("chord_sec", [0.5, 1.0], units="m")

        p.run_model()

        assert_near_equal(p.get_val("S", units="m**2"), 1.5)

    def test_sweep(self):
        p = om.Problem()
        p.model.add_subsystem("comp", WingAreaFromSections(num_sections=3), promotes=["*"])
        p.setup(force_alloc_complex=True)

        p.set_val("y_sec", [-2, -1.0], units="m")
        p.set_val("chord_sec", [0.1, 0.5, 1.0], units="m")

        p.run_model()

        assert_near_equal(p.get_val("S", units="m**2"), 2.1)

    def test_indices(self):
        p = om.Problem()
        p.model.add_subsystem(
            "comp", WingAreaFromSections(num_sections=4, idx_sec_start=1, idx_sec_end=2, chord_frac_start=0.1, chord_frac_end=0.3), promotes=["*"]
        )
        p.setup(force_alloc_complex=True)

        p.set_val("y_sec", [-2.5, -2, -1], units="m")
        p.set_val("chord_sec", [0.1, 1.0, 1.0, 1.5], units="m")

        p.run_model()

        assert_near_equal(p.get_val("S", units="m**2"), 0.4)


if __name__ == "__main__":
    unittest.main()
