import unittest
import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openconcept.geometry import (
    WingMACTrapezoidal,
    WingSpan,
    WingAspectRatio,
    WingSweepFromSections,
    WingAreaFromSections,
    WingMACFromSections,
)


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

        partials = p.check_partials(method="cs", step=1e-125, out_stream=None)
        assert_check_partials(partials)

    def test_sweep(self):
        p = om.Problem()
        p.model.add_subsystem("comp", WingSweepFromSections(num_sections=2), promotes=["*"])
        p.setup(force_alloc_complex=True)

        p.set_val("x_LE_sec", [1, 0], units="m")
        p.set_val("y_sec", -1.0, units="m")
        p.set_val("chord_sec", [0.5, 1.0], units="m")

        p.run_model()

        assert_near_equal(p.get_val("c4sweep", units="deg"), np.rad2deg(np.arctan(0.875)))

        partials = p.check_partials(method="cs", step=1e-125, out_stream=None)
        assert_check_partials(partials)

    def test_sweep_three_sections(self):
        p = om.Problem()
        p.model.add_subsystem("comp", WingSweepFromSections(num_sections=3), promotes=["*"])
        p.setup(force_alloc_complex=True)

        p.set_val("x_LE_sec", [0, 1, 0], units="m")
        p.set_val("y_sec", [-2, -1.0], units="m")
        p.set_val("chord_sec", [0.1, 0.5, 1.0], units="m")

        p.run_model()

        assert_near_equal(p.get_val("c4sweep", units="deg"), 43.1348109, tolerance=1e-8)

        partials = p.check_partials(method="cs", step=1e-125, out_stream=None)
        assert_check_partials(partials)

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

        assert_near_equal(p.get_val("c4sweep", units="deg"), 45)

        partials = p.check_partials(method="cs", step=1e-125, out_stream=None)
        assert_check_partials(partials)


class WingAreaFromSectionsTestCase(unittest.TestCase):
    def test_no_sweep(self):
        p = om.Problem()
        p.model.add_subsystem("comp", WingAreaFromSections(num_sections=2), promotes=["*"])
        p.setup(force_alloc_complex=True)

        p.set_val("y_sec", -1.0, units="m")
        p.set_val("chord_sec", [1.0, 1.0], units="m")

        p.run_model()

        assert_near_equal(p.get_val("S", units="m**2"), 2.0)

        partials = p.check_partials(method="cs", step=1e-125, out_stream=None)
        assert_check_partials(partials)

    def test_sweep(self):
        p = om.Problem()
        p.model.add_subsystem("comp", WingAreaFromSections(num_sections=2), promotes=["*"])
        p.setup(force_alloc_complex=True)

        p.set_val("y_sec", -1.0, units="m")
        p.set_val("chord_sec", [0.5, 1.0], units="m")

        p.run_model()

        assert_near_equal(p.get_val("S", units="m**2"), 1.5)

        partials = p.check_partials(method="cs", step=1e-125, out_stream=None)
        assert_check_partials(partials)

    def test_sweep_three_sections(self):
        p = om.Problem()
        p.model.add_subsystem("comp", WingAreaFromSections(num_sections=3), promotes=["*"])
        p.setup(force_alloc_complex=True)

        p.set_val("y_sec", [-2, -1.0], units="m")
        p.set_val("chord_sec", [0.1, 0.5, 1.0], units="m")

        p.run_model()

        assert_near_equal(p.get_val("S", units="m**2"), 2.1)

        partials = p.check_partials(method="cs", step=1e-125, out_stream=None)
        assert_check_partials(partials)

    def test_indices(self):
        p = om.Problem()
        p.model.add_subsystem(
            "comp",
            WingAreaFromSections(
                num_sections=4, idx_sec_start=1, idx_sec_end=2, chord_frac_start=0.1, chord_frac_end=0.3
            ),
            promotes=["*"],
        )
        p.setup(force_alloc_complex=True)

        p.set_val("y_sec", [-2.5, -2, -1], units="m")
        p.set_val("chord_sec", [0.1, 1.0, 1.0, 1.5], units="m")

        p.run_model()

        assert_near_equal(p.get_val("S", units="m**2"), 0.4)

        partials = p.check_partials(method="cs", step=1e-125, out_stream=None)
        assert_check_partials(partials)


class WingMACFromSectionsTestCase(unittest.TestCase):
    def test_rectangular(self):
        """
        MAC of rectangular wing is just chord and quarter MAC is at quarter chord.
        """
        p = om.Problem()
        p.model.add_subsystem("comp", WingMACFromSections(num_sections=2), promotes=["*"])
        p.setup(force_alloc_complex=True)

        p.set_val("x_LE_sec", [0, 0], units="m")
        p.set_val("y_sec", -1.0, units="m")
        p.set_val("chord_sec", [1.0, 1.0], units="m")

        p.run_model()

        assert_near_equal(p.get_val("MAC", units="m"), 1.0)
        assert_near_equal(p.get_val("x_c4MAC", units="m"), 0.25)

        partials = p.check_partials(method="cs", step=1e-125, out_stream=None)
        assert_check_partials(partials)

    def test_tapered(self):
        p = om.Problem()
        p.model.add_subsystem("comp", WingMACFromSections(num_sections=2), promotes=["*"])
        p.setup(force_alloc_complex=True)

        Cr = 1.0
        taper = 0.5
        b = 4

        p.set_val("x_LE_sec", [Cr * taper / 2, 0], units="m")
        p.set_val("y_sec", -b / 2, units="m")
        p.set_val("chord_sec", [Cr * taper, Cr], units="m")

        p.run_model()

        # Test against equation for trapezoidal wing MAC
        MAC = 2 / 3 * Cr * (1 + taper + taper**2) / (1 + taper)
        y_MAC = b / 6 * (1 + 2 * taper) / (1 + taper)
        x_c4MAC = (y_MAC / (b / 2)) * (Cr * taper / 2) + 0.25 * MAC

        assert_near_equal(p.get_val("MAC", units="m"), MAC)
        assert_near_equal(p.get_val("x_c4MAC", units="m"), x_c4MAC)

        partials = p.check_partials(method="cs", step=1e-125, out_stream=None)
        assert_check_partials(partials)

    def test_tapered_swept(self):
        p = om.Problem()
        p.model.add_subsystem("comp", WingMACFromSections(num_sections=2), promotes=["*"])
        p.setup(force_alloc_complex=True)

        Cr = 1.5
        taper = 0.5
        b = 4

        p.set_val("x_LE_sec", [Cr, 0], units="m")
        p.set_val("y_sec", -b / 2, units="m")
        p.set_val("chord_sec", [Cr * taper, Cr], units="m")

        p.run_model()

        # Test against equation for trapezoidal wing MAC
        MAC = 2 / 3 * Cr * (1 + taper + taper**2) / (1 + taper)
        y_MAC = b / 6 * (1 + 2 * taper) / (1 + taper)
        x_c4MAC = (y_MAC / (b / 2)) * (Cr) + 0.25 * MAC

        assert_near_equal(p.get_val("MAC", units="m"), MAC)
        assert_near_equal(p.get_val("x_c4MAC", units="m"), x_c4MAC)

        partials = p.check_partials(method="cs", step=1e-125, out_stream=None)
        assert_check_partials(partials)

    def test_indices(self):
        p = om.Problem()
        p.model.add_subsystem(
            "comp", WingMACFromSections(num_sections=4, idx_sec_start=1, idx_sec_end=2), promotes=["*"]
        )
        p.setup(force_alloc_complex=True)

        Cr = 1.5
        taper = 0.5
        b = 4

        p.set_val("x_LE_sec", [-10, Cr, 0, 4], units="m")
        p.set_val("y_sec", [-2 * b, -b, -b / 2], units="m")
        p.set_val("chord_sec", [5, Cr * taper, Cr, 0.1], units="m")

        p.run_model()

        # Test against equation for trapezoidal wing MAC
        MAC = 2 / 3 * Cr * (1 + taper + taper**2) / (1 + taper)
        y_MAC = b / 6 * (1 + 2 * taper) / (1 + taper)
        x_c4MAC = (y_MAC / (b / 2)) * (Cr) + 0.25 * MAC

        assert_near_equal(p.get_val("MAC", units="m"), MAC)
        assert_near_equal(p.get_val("x_c4MAC", units="m"), x_c4MAC)

        partials = p.check_partials(method="cs", step=1e-125, out_stream=None)
        assert_check_partials(partials)


if __name__ == "__main__":
    unittest.main()
