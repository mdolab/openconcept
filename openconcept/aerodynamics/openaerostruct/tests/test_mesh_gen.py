import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
import openmdao.api as om
from openconcept.aerodynamics.openaerostruct.mesh_gen import (
    TrapezoidalPlanformMesh,
    SectionPlanformMesh,
    ThicknessChordRatioInterp,
    SectionLinearInterp,
    cos_space,
    cos_space_deriv_start,
    cos_space_deriv_end,
)


class TrapezoidalPlanformMeshTestCase(unittest.TestCase):
    def test_easy(self):
        nx = 3
        ny = 5
        p = om.Problem()
        p.model.add_subsystem("comp", TrapezoidalPlanformMesh(num_x=nx - 1, num_y=ny - 1), promotes=["*"])
        p.setup()
        p.set_val("S", 2, units="m**2")
        p.set_val("AR", 2)
        p.set_val("taper", 1.0)
        p.set_val("sweep", 0.0, units="deg")
        p.run_model()

        mesh = np.zeros((nx, ny, 3))
        mesh[:, :, 0], mesh[:, :, 1] = np.meshgrid(np.linspace(-0.25, 0.75, nx), np.linspace(-1, 0, ny), indexing="ij")

        assert_near_equal(p.get_val("mesh", units="m"), mesh)

        partials = p.check_partials(out_stream=None, compact_print=True, show_only_incorrect=True)
        assert_check_partials(partials)

    def test_S_AR(self):
        nx = 3
        ny = 5
        p = om.Problem()
        p.model.add_subsystem("comp", TrapezoidalPlanformMesh(num_x=nx - 1, num_y=ny - 1), promotes=["*"])
        p.setup(force_alloc_complex=True)
        p.set_val("S", 48, units="m**2")
        p.set_val("AR", 3)
        p.set_val("taper", 1.0)
        p.set_val("sweep", 0.0, units="deg")
        p.run_model()

        mesh = np.zeros((nx, ny, 3))
        mesh[:, :, 0], mesh[:, :, 1] = np.meshgrid(np.linspace(-1, 3, nx), np.linspace(-6, 0, ny), indexing="ij")

        assert_near_equal(p.get_val("mesh", units="m"), mesh)

        partials = p.check_partials(out_stream=None, form="central")
        assert_check_partials(partials)

    def test_taper(self):
        nx = 2
        ny = 3
        p = om.Problem()
        p.model.add_subsystem("comp", TrapezoidalPlanformMesh(num_x=nx - 1, num_y=ny - 1), promotes=["*"])
        p.setup()
        p.set_val("S", 1.3, units="m**2")
        p.set_val("AR", 4 / 1.3)  # pick S and AR for half span and root chord of 1
        p.set_val("taper", 0.3)
        p.set_val("sweep", 0.0, units="deg")
        p.run_model()

        mesh = np.zeros((nx, ny, 3))
        mesh[:, :, 0] = np.array([[-0.075, -0.1625, -0.25], [0.225, 0.4875, 0.75]])
        mesh[:, :, 1] = np.array([[-1, -0.5, 0], [-1, -0.5, 0]])

        assert_near_equal(p.get_val("mesh", units="m"), mesh)

        partials = p.check_partials(out_stream=None, compact_print=True, show_only_incorrect=True)
        assert_check_partials(partials)

    def test_sweep(self):
        nx = 3
        ny = 3
        p = om.Problem()
        p.model.add_subsystem("comp", TrapezoidalPlanformMesh(num_x=nx - 1, num_y=ny - 1), promotes=["*"])
        p.setup()
        p.set_val("S", 2, units="m**2")
        p.set_val("AR", 2)
        p.set_val("taper", 1.0)
        p.set_val("sweep", 45.0, units="deg")
        p.run_model()

        mesh = np.zeros((nx, ny, 3))
        mesh[:, :, 0], mesh[:, :, 1] = np.meshgrid(np.linspace(-0.25, 0.75, nx), np.linspace(-1, 0, ny), indexing="ij")

        mesh[:, 0, 0] += 1
        mesh[:, 1, 0] += 0.5

        assert_near_equal(p.get_val("mesh", units="m"), mesh)

        partials = p.check_partials(out_stream=None, compact_print=True, show_only_incorrect=True)
        assert_check_partials(partials)

    def test_taper_sweep(self):
        nx = 2
        ny = 3
        p = om.Problem()
        p.model.add_subsystem("comp", TrapezoidalPlanformMesh(num_x=nx - 1, num_y=ny - 1), promotes=["*"])
        p.setup()
        p.set_val("S", 1.3, units="m**2")
        p.set_val("AR", 4 / 1.3)  # pick S and AR for half span and root chord of 1
        p.set_val("taper", 0.3)
        p.set_val("sweep", 45.0, units="deg")
        p.run_model()

        mesh = np.zeros((nx, ny, 3))
        mesh[:, :, 0] = np.array([[-0.075, -0.1625, -0.25], [0.225, 0.4875, 0.75]])
        mesh[:, :, 1] = np.array([[-1, -0.5, 0], [-1, -0.5, 0]])
        mesh[:, 0, 0] += 1
        mesh[:, 1, 0] += 0.5

        assert_near_equal(p.get_val("mesh", units="m"), mesh)

        partials = p.check_partials(out_stream=None, compact_print=True, show_only_incorrect=True)
        assert_check_partials(partials)

    def test_777ish_regression(self):
        nx = 3
        ny = 4
        p = om.Problem()
        p.model.add_subsystem("comp", TrapezoidalPlanformMesh(num_x=nx - 1, num_y=ny - 1), promotes=["*"])
        p.setup()
        p.set_val("S", 427.8, units="m**2")
        p.set_val("AR", 9.82)
        p.set_val("taper", 0.149)
        p.set_val("sweep", 31.6, units="deg")
        p.run_model()

        mesh = np.array(
            [
                [
                    [19.50929722, -32.40754542, 0.0],
                    [12.04879827, -21.60503028, 0.0],
                    [4.58829932, -10.80251514, 0.0],
                    [-2.87219963, 0.0, 0.0],
                ],
                [
                    [20.36521271, -32.40754542, 0.0],
                    [14.53420835, -21.60503028, 0.0],
                    [8.70320399, -10.80251514, 0.0],
                    [2.87219963, 0.0, 0.0],
                ],
                [
                    [21.2211282, -32.40754542, 0.0],
                    [17.01961843, -21.60503028, 0.0],
                    [12.81810866, -10.80251514, 0.0],
                    [8.61659889, 0.0, 0.0],
                ],
            ]
        )

        assert_near_equal(p.get_val("mesh", units="m"), mesh, tolerance=1e-10)

        partials = p.check_partials(out_stream=None, compact_print=True, show_only_incorrect=True)
        assert_check_partials(partials, atol=2e-5)


class SectionPlanformMeshTestCase(unittest.TestCase):
    def test_hershey_bar(self):
        """
        A simple rectangular wing with a span of two and chord of one.
        """
        nx = 2
        ny = 2
        x_mesh, y_mesh = np.meshgrid([0, 0.5, 1.0], [-1, -0.5, 0], indexing="ij")

        p = om.Problem()
        p.model.add_subsystem("comp", SectionPlanformMesh(num_x=nx, num_y=ny, num_sections=2), promotes=["*"])
        p.setup(force_alloc_complex=True)

        p.set_val("S", 2.0, units="m**2")
        p.set_val("x_LE_sec", 0)
        p.set_val("y_sec", -1.0)
        p.set_val("chord_sec", 1.0)

        p.run_model()

        assert_near_equal(p.get_val("mesh", units="m")[:, :, 0], x_mesh)
        assert_near_equal(p.get_val("mesh", units="m")[:, :, 1], y_mesh)
        assert_near_equal(p.get_val("mesh", units="m")[:, :, 2], np.zeros_like(x_mesh))

        partials = p.check_partials(method="cs", out_stream=None, compact_print=True, show_only_incorrect=True)
        assert_check_partials(partials)

    def test_hershey_bar_two_sections(self):
        """
        A simple rectangular wing with a span of two and chord of one, but divided into two sections.
        """
        nx = 2
        ny = 2
        x_mesh, y_mesh = np.meshgrid(np.linspace(0, 1, nx + 1), np.linspace(-1, 0, ny * 2 + 1), indexing="ij")

        p = om.Problem()
        p.model.add_subsystem("comp", SectionPlanformMesh(num_x=nx, num_y=ny, num_sections=3), promotes=["*"])
        p.setup(force_alloc_complex=True)

        p.set_val("S", 2.0, units="m**2")
        p.set_val("x_LE_sec", 0)
        p.set_val("y_sec", [-1.0, -0.5])
        p.set_val("chord_sec", 1.0)

        p.run_model()

        assert_near_equal(p.get_val("mesh", units="m")[:, :, 0], x_mesh)
        assert_near_equal(p.get_val("mesh", units="m")[:, :, 1], y_mesh)
        assert_near_equal(p.get_val("mesh", units="m")[:, :, 2], np.zeros_like(x_mesh))

        partials = p.check_partials(method="cs", out_stream=None, compact_print=True, show_only_incorrect=True)
        assert_check_partials(partials)

    def test_scale_area_off(self):
        p = om.Problem()
        p.model.add_subsystem(
            "comp", SectionPlanformMesh(num_x=3, num_y=3, num_sections=3, scale_area=False), promotes=["*"]
        )
        p.setup(force_alloc_complex=True)

        p.set_val("S", 2.0, units="m**2")
        p.set_val("x_LE_sec", [2, 0.2, 0.0], units="m")
        p.set_val("y_sec", [-1.0, -0.2], units="m")
        p.set_val("chord_sec", [0.2, 0.4, 0.4], units="m")

        p.run_model()

        assert_near_equal(p.get_val("S", units="m**2"), 2 * (0.3 * 0.8 + 0.4 * 0.2))

        partials = p.check_partials(method="cs", out_stream=None, compact_print=True, show_only_incorrect=True)
        assert_check_partials(partials)

    def test_chordwise_cos_spacing(self):
        nx = 7
        ny = 2
        x_mesh, y_mesh = np.meshgrid(cos_space(0, 1, nx + 1), [-1, -0.5, 0], indexing="ij")

        p = om.Problem()
        p.model.add_subsystem("comp", SectionPlanformMesh(num_x=nx, num_y=ny, num_sections=2), promotes=["*"])
        p.setup(force_alloc_complex=True)

        p.set_val("S", 2.0, units="m**2")
        p.set_val("x_LE_sec", 0)
        p.set_val("y_sec", -1.0)
        p.set_val("chord_sec", 1.0)

        p.run_model()

        assert_near_equal(p.get_val("mesh", units="m")[:, :, 0], x_mesh)
        assert_near_equal(p.get_val("mesh", units="m")[:, :, 1], y_mesh)
        assert_near_equal(p.get_val("mesh", units="m")[:, :, 2], np.zeros_like(x_mesh))

        partials = p.check_partials(method="cs", out_stream=None, compact_print=True, show_only_incorrect=True)
        assert_check_partials(partials)

    def test_spanwise_cos_spacing(self):
        nx = 2
        ny = 7
        x_mesh, y_mesh = np.meshgrid([0.0, 0.5, 1.0], cos_space(-1, 0, ny + 1), indexing="ij")

        p = om.Problem()
        p.model.add_subsystem("comp", SectionPlanformMesh(num_x=nx, num_y=ny, num_sections=2), promotes=["*"])
        p.setup(force_alloc_complex=True)

        p.set_val("S", 2.0, units="m**2")
        p.set_val("x_LE_sec", 0)
        p.set_val("y_sec", -1.0)
        p.set_val("chord_sec", 1.0)

        p.run_model()

        assert_near_equal(p.get_val("mesh", units="m")[:, :, 0], x_mesh)
        assert_near_equal(p.get_val("mesh", units="m")[:, :, 1], y_mesh)
        assert_near_equal(p.get_val("mesh", units="m")[:, :, 2], np.zeros_like(x_mesh))

        partials = p.check_partials(method="cs", out_stream=None, compact_print=True, show_only_incorrect=True)
        assert_check_partials(partials)

    def test_area_scaling(self):
        nx = 2
        ny = 2
        x_mesh, y_mesh = np.meshgrid([0, 1.5, 3.0], [-3, -1.5, 0], indexing="ij")

        p = om.Problem()
        p.model.add_subsystem("comp", SectionPlanformMesh(num_x=nx, num_y=ny, num_sections=2), promotes=["*"])
        p.setup(force_alloc_complex=True)

        p.set_val("S", 18.0, units="m**2")
        p.set_val("x_LE_sec", 0)
        p.set_val("y_sec", -1.0)
        p.set_val("chord_sec", 1.0)

        p.run_model()

        assert_near_equal(p.get_val("mesh", units="m")[:, :, 0], x_mesh)
        assert_near_equal(p.get_val("mesh", units="m")[:, :, 1], y_mesh)
        assert_near_equal(p.get_val("mesh", units="m")[:, :, 2], np.zeros_like(x_mesh))

        partials = p.check_partials(method="cs", out_stream=None, compact_print=True, show_only_incorrect=True)
        assert_check_partials(partials)

    def test_BWBish_regression(self):
        nx = 3
        num_sec = 4
        x_pos = np.array([480 + 30, 302 + 50, 255, 0])
        y_pos = np.array([-550, -235, -123])
        chord = np.array([30, 100, 577 - 255, 577.0])
        ny = np.array([3, 3, 3])

        # fmt: off
        x_mesh = np.array([[0.88602799, 0.81740425, 0.68015678, 0.61153304, 0.56940328, 0.48514376, 0.443014  , 0.3322605 , 0.1107535 , 0],
                           [0.89905781, 0.83803481, 0.7159888 , 0.65496579, 0.6369412 , 0.60089202, 0.58286743, 0.49980231, 0.33367206, 0.25060694],
                           [0.92511746, 0.87929592, 0.78765282, 0.74183128, 0.77201704, 0.83238855, 0.86257431, 0.83488593, 0.77950918, 0.75182081],
                           [0.93814728, 0.89992647, 0.82348484, 0.78526402, 0.83955495, 0.94813682, 1.00242775, 1.00242775, 1.00242775, 1.00242775]])
        y_mesh = np.array([[-0.95552038, -0.81870724, -0.54508095, -0.4082678 , -0.35962313, -0.26233378, -0.2136891 , -0.16026683, -0.05342228,  0],
                           [-0.95552038, -0.81870724, -0.54508095, -0.4082678 , -0.35962313, -0.26233378, -0.2136891 , -0.16026683, -0.05342228,  0],
                           [-0.95552038, -0.81870724, -0.54508095, -0.4082678 , -0.35962313, -0.26233378, -0.2136891 , -0.16026683, -0.05342228,  0],
                           [-0.95552038, -0.81870724, -0.54508095, -0.4082678 , -0.35962313, -0.26233378, -0.2136891 , -0.16026683, -0.05342228,  0]])
        # fmt: on

        p = om.Problem()
        p.model.add_subsystem("mesh", SectionPlanformMesh(num_x=nx, num_y=ny, num_sections=num_sec), promotes=["*"])
        p.setup(force_alloc_complex=True)
        p.set_val("S", 0.6, units="m**2")
        p.set_val("x_LE_sec", x_pos)
        p.set_val("y_sec", y_pos)
        p.set_val("chord_sec", chord)

        p.run_model()

        assert_near_equal(p.get_val("mesh", units="m")[:, :, 0], x_mesh, tolerance=1e-6)
        assert_near_equal(p.get_val("mesh", units="m")[:, :, 1], y_mesh, tolerance=1e-6)
        assert_near_equal(p.get_val("mesh", units="m")[:, :, 2], np.zeros_like(x_mesh))

        partials = p.check_partials(method="cs", out_stream=None, compact_print=True, show_only_incorrect=True)
        assert_check_partials(partials)


class ThicknessChordRatioInterpTestCase(unittest.TestCase):
    def test_simple(self):
        ny = 4
        p = om.Problem()
        p.model.add_subsystem(
            "comp", ThicknessChordRatioInterp(num_y=ny, num_sections=2, cos_spacing=False), promotes=["*"]
        )
        p.setup(force_alloc_complex=True)

        p.set_val("section_toverc", [1, 3])
        p.run_model()

        assert_near_equal(p.get_val("panel_toverc"), 1 + 2 * np.array([0.125, 0.375, 0.625, 0.875]))

        partials = p.check_partials(method="cs")
        assert_check_partials(partials)

    def test_cos_spacing(self):
        ny = 4
        p = om.Problem()
        p.model.add_subsystem(
            "comp", ThicknessChordRatioInterp(num_y=ny, num_sections=2, cos_spacing=True), promotes=["*"]
        )
        p.setup(force_alloc_complex=True)

        p.set_val("section_toverc", [3, 1])
        p.run_model()

        cos_spacing = cos_space(0, 1, ny + 1)
        panel_toverc = 3 - (cos_spacing[:-1] + cos_spacing[1:])
        assert_near_equal(p.get_val("panel_toverc"), panel_toverc)

        partials = p.check_partials(method="cs")
        assert_check_partials(partials)

    def test_multiple_sections(self):
        ny = np.array([2, 4, 1])
        n_sec = 4
        p = om.Problem()
        p.model.add_subsystem(
            "comp", ThicknessChordRatioInterp(num_y=ny, num_sections=n_sec, cos_spacing=True), promotes=["*"]
        )
        p.setup(force_alloc_complex=True)

        sec_toverc = [0, 2, -2, 1]
        p.set_val("section_toverc", sec_toverc)
        p.run_model()

        panel_toverc = []
        for i_sec in range(n_sec - 1):
            cos_spacing = cos_space(sec_toverc[i_sec], sec_toverc[i_sec + 1], ny[i_sec] + 1)
            panel_toverc += list(0.5 * (cos_spacing[:-1] + cos_spacing[1:]))
        assert_near_equal(p.get_val("panel_toverc"), panel_toverc)

        partials = p.check_partials(method="cs")
        assert_check_partials(partials)


class SectionLinearInterpTestCase(unittest.TestCase):
    def test_cos_spacing(self):
        ny = np.array([3, 2, 1])
        n_sec = 4
        p = om.Problem()
        p.model.add_subsystem(
            "comp", SectionLinearInterp(num_y=ny, num_sections=n_sec, units="deg", cos_spacing=True), promotes=["*"]
        )
        p.setup(force_alloc_complex=True)

        prop = [0, 2, -2, 1]
        p.set_val("property_sec", prop)
        p.run_model()

        prop_node = np.hstack((cos_space(0, 2, 4), cos_space(2, -2, 3)[1:], [1]))
        assert_near_equal(p.get_val("property_node"), prop_node)

        partials = p.check_partials(method="cs")
        assert_check_partials(partials)

    def test_linear_spacing(self):
        ny = np.array([3, 2, 1])
        n_sec = 4
        p = om.Problem()
        p.model.add_subsystem(
            "comp", SectionLinearInterp(num_y=ny, num_sections=n_sec, units="deg", cos_spacing=False), promotes=["*"]
        )
        p.setup(force_alloc_complex=True)

        prop = [0, 2, -2, 1]
        p.set_val("property_sec", prop)
        p.run_model()

        prop_node = np.hstack((np.linspace(0, 2, 4), np.linspace(2, -2, 3)[1:], [1]))
        assert_near_equal(p.get_val("property_node"), prop_node)

        partials = p.check_partials(method="cs")
        assert_check_partials(partials)


class CosSpacingTestCase(unittest.TestCase):
    def test_simple(self):
        num = 7
        correct = (1 - np.cos(np.linspace(0, np.pi, num))) * 0.5
        assert_near_equal(cos_space(0, 1, num), correct)

    def test_shifted(self):
        start = 1
        end = -4
        num = 8
        correct = (1 - np.cos(np.linspace(0, np.pi, num))) * 0.5  # 0-1 range
        correct = correct * (end - start) + start  # stretch and shift
        assert_near_equal(cos_space(start, end, num), correct)

    def test_derivs_simple(self):
        start = 2
        end = 6
        num = 5
        step = 1e-200
        deriv_start = np.imag(cos_space(start + step * 1j, end, num, dtype=complex)) / step
        deriv_end = np.imag(cos_space(start, end + step * 1j, num, dtype=complex)) / step

        assert_near_equal(cos_space_deriv_start(num), deriv_start)
        assert_near_equal(cos_space_deriv_end(num), deriv_end)


if __name__ == "__main__":
    unittest.main()
