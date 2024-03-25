import unittest
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openconcept.aerodynamics import CleanCLmax, FlapCLmax


class CleanCLmaxTestCase(unittest.TestCase):
    def test_B738(self):
        """
        Test roughly B738 parameters.
        """
        p = om.Problem()
        p.model.add_subsystem("comp", CleanCLmax(), promotes=["*"])
        p.setup(force_alloc_complex=True)

        p.set_val("ac|aero|airfoil_Cl_max", 1.75)
        p.set_val("ac|geom|wing|c4sweep", 25, units="deg")

        p.run_model()

        assert_near_equal(p.get_val("CL_max_clean"), 1.42743476, tolerance=1e-8)

        p = p.check_partials(method="cs", out_stream=None)
        assert_check_partials(p)

    def test_fudge_factor(self):
        p = om.Problem()
        p.model.add_subsystem("comp", CleanCLmax(fudge_factor=2.0), promotes=["*"])
        p.setup(force_alloc_complex=True)

        p.set_val("ac|aero|airfoil_Cl_max", 1.75)
        p.set_val("ac|geom|wing|c4sweep", 25, units="deg")

        p.run_model()

        assert_near_equal(p.get_val("CL_max_clean"), 2 * 1.42743476, tolerance=1e-8)

        p = p.check_partials(method="cs", out_stream=None)
        assert_check_partials(p)


class FlapCLmaxTestCase(unittest.TestCase):
    def test_B738(self):
        """
        Test roughly B738 parameters.
        """
        p = om.Problem()
        p.model.add_subsystem("comp", FlapCLmax(), promotes=["*"])
        p.setup(force_alloc_complex=True)

        p.set_val("flap_extension", 40, units="deg")
        p.set_val("ac|geom|wing|c4sweep", 25, units="deg")
        p.set_val("CL_max_clean", 1.42743476)
        p.set_val("ac|geom|wing|toverc", 0.12)

        p.run_model()

        assert_near_equal(p.get_val("CL_max_flap"), 2.65255284, tolerance=1e-8)

        p = p.check_partials(method="cs", out_stream=None)
        assert_check_partials(p)

    def test_options(self):
        p = om.Problem()
        p.model.add_subsystem(
            "comp",
            FlapCLmax(
                flap_chord_frac=0.25,
                wing_area_flapped_frac=0.8,
                slat_chord_frac=0.01,
                slat_span_frac=0.7,
                fudge_factor=1.1,
            ),
            promotes=["*"],
        )
        p.setup(force_alloc_complex=True)

        p.set_val("flap_extension", 15, units="deg")
        p.set_val("ac|geom|wing|c4sweep", 25, units="deg")
        p.set_val("CL_max_clean", 1.42743476)
        p.set_val("ac|geom|wing|toverc", 0.12)

        p.run_model()

        assert_near_equal(p.get_val("CL_max_flap"), 2.17133794, tolerance=1e-8)

        p = p.check_partials(method="cs", out_stream=None)
        assert_check_partials(p)


if __name__ == "__main__":
    unittest.main()
