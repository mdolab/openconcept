"""
@File    :   test_wave_drag.py
@Date    :   2023/04/17
@Author  :   Eytan Adler
@Description : Test the wave drag utilities
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import unittest

# ==============================================================================
# External Python modules
# ==============================================================================
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials

# ==============================================================================
# Extension modules
# ==============================================================================
from openconcept.aerodynamics.openaerostruct import WaveDragFromSections


class WaveDragFromSectionsTestCase(unittest.TestCase):
    def test_verify_OAS(self):
        """
        Verify against OpenAeroStruct
        """
        p = om.Problem()
        p.model.add_subsystem("comp", WaveDragFromSections(num_sections=4), promotes=["*"])
        p.setup()

        p.set_val("fltcond|M", 0.95)
        p.set_val("fltcond|CL", 0.2997707162790314)

        p.set_val("toverc_sec", [0.08, 0.11, 0.15, 0.15])
        p.set_val("y_sec", [-105, -41, -23], units="ft")
        p.set_val("chord_sec", [8, 25, 75, 104], units="ft")
        p.set_val("c4sweep", 37.69966129860842, units="deg")

        p.run_model()

        assert_near_equal(p.get_val("CD_wave"), 0.011138076370115135, tolerance=1e-3)

        partials = p.check_partials(method="fd", out_stream=None)
        assert_check_partials(partials, atol=1e-4, rtol=1e-4)

    def test_indices(self):
        """
        Test the indices by adding a section to the beginning and end of the previous test but ignore them using the indices.
        """
        p = om.Problem()
        p.model.add_subsystem(
            "comp", WaveDragFromSections(num_sections=6, idx_sec_start=1, idx_sec_end=4), promotes=["*"]
        )
        p.setup()

        p.set_val("fltcond|M", 0.95)
        p.set_val("fltcond|CL", 0.2997707162790314)

        p.set_val("toverc_sec", [0.01, 0.08, 0.11, 0.15, 0.15, 0.2])
        p.set_val("y_sec", [-110, -105, -41, -23, 0], units="ft")
        p.set_val("chord_sec", [8, 8, 25, 75, 104, 8], units="ft")
        p.set_val("c4sweep", 37.69966129860842, units="deg")

        p.run_model()

        assert_near_equal(p.get_val("CD_wave"), 0.011138076370115135, tolerance=1e-3)

        partials = p.check_partials(method="fd", out_stream=None)
        assert_check_partials(partials, atol=1e-4, rtol=1e-4)

    def test_area_norm(self):
        """
        Test the area normalization with a different area.
        """
        p = om.Problem()
        p.model.add_subsystem("comp", WaveDragFromSections(num_sections=4, specify_area_norm=True), promotes=["*"])
        p.setup()

        p.set_val("fltcond|M", 0.95)
        p.set_val("fltcond|CL", 0.2997707162790314)

        S_orig = 4014.5 * 2
        S_new = 1e4
        p.set_val("S_ref", S_new, units="ft**2")
        p.set_val("toverc_sec", [0.08, 0.11, 0.15, 0.15])
        p.set_val("y_sec", [-105, -41, -23], units="ft")
        p.set_val("chord_sec", [8, 25, 75, 104], units="ft")
        p.set_val("c4sweep", 37.69966129860842, units="deg")

        p.run_model()

        assert_near_equal(p.get_val("CD_wave"), 0.011138076370115135 * S_new / S_orig, tolerance=1e-3)

        partials = p.check_partials(method="fd", out_stream=None)
        assert_check_partials(partials, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()

    """
    Below is the code used to compute OpenAeroStruct's wave drag coefficient
    against which we compare the values out of OpenConcept.
    """
    # from openconcept.aerodynamics.openaerostruct import SectionPlanformMesh, VLM
    # from openconcept.aerodynamics.openaerostruct import ThicknessChordRatioInterp
    # from openconcept.geometry import WingSweepFromSections
    # p = om.Problem()

    # nx = 1
    # ny_sec = 61  # many spanwise panels make OpenAeroStruct's t/c averaging more accurate
    # p.model.add_subsystem("mesh", SectionPlanformMesh(num_sections=4, num_x=nx, num_y=ny_sec, scale_area=False), promotes=["*"])
    # p.model.add_subsystem("tc", ThicknessChordRatioInterp(num_y=ny_sec, num_sections=4, cos_spacing=True), promotes=["*"])
    # p.model.add_subsystem("sweep", WingSweepFromSections(num_sections=4), promotes=["*"])
    # p.model.add_subsystem("oas", VLM(num_x=nx, num_y=ny_sec * 3), promotes=["*"])
    # p.model.add_subsystem(
    #     "comp", WaveDragFromSections(num_sections=4), promotes=["*"]
    # )
    # p.model.connect("mesh", "ac|geom|wing|OAS_mesh")
    # p.model.connect("panel_toverc", "ac|geom|wing|toverc")

    # p.model.set_input_defaults("fltcond|M", 0.95)
    # p.model.set_input_defaults("fltcond|alpha", 5)
    # for var in ["toverc_sec", "section_toverc"]:
    #     p.model.set_input_defaults(var, [0.08, 0.11, 0.15, 0.15])
    # p.model.set_input_defaults("x_LE_sec", [95, 52, 29, 0], units="ft")
    # p.model.set_input_defaults("y_sec", [-105, -41, -23], units="ft")
    # p.model.set_input_defaults("chord_sec", [8, 25, 75, 104], units="ft")

    # p.setup()
    # p.run_model()

    # print(f"OpenAeroStruct CL   {p.get_val('fltcond|CL').item()}")
    # print(f"OpenConcept sweep   {p.get_val('c4sweep', units='deg').item()} deg")
    # print(f"OpenAeroStruct CDw  {p.get_val('fltcond|CDw').item()}")
    # print(f"OpenConcept CD_wave {p.get_val('CD_wave').item()}")
