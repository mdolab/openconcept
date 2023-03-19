import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
import openmdao.api as om

# Only run if OpenAeroStruct is installed
try:
    from openaerostruct.geometry.geometry_group import Geometry
    from openaerostruct.aerodynamics.aero_groups import AeroPoint
    from openconcept.aerodynamics.openaerostruct import TrapezoidalPlanformMesh
    from openconcept.aerodynamics.openaerostruct.drag_polar import (
        VLMDataGen,
        VLM,
        VLMDragPolar,
        example_usage,
    )

    OAS_installed = True
except ImportError:
    OAS_installed = False


@unittest.skipIf(not OAS_installed, "OpenAeroStruct is not installed")
class VLMDragPolarTestCase(unittest.TestCase):
    def tearDown(self):
        # Get rid of any specified surface options in the VLMDataGen
        # class after every test. This is necessary because the class
        # stores the surface options as a "static" variable and
        # prevents multiple VLMDataGen instances with different
        # surface options. Doing this prevents that error when doing
        # multiple tests with different surface options.
        del VLMDataGen.surf_options

    def test(self):
        twist = np.array([-1, -0.5, 2])

        p = om.Problem(
            VLMDragPolar(
                num_nodes=1,
                num_x=2,
                num_y=4,
                num_twist=twist.size,
                Mach_train=np.linspace(0.1, 0.8, 3),
                alpha_train=np.linspace(-11, 15, 3),
                alt_train=np.linspace(0, 15e3, 2),
            )
        )
        p.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        p.model.linear_solver = om.DirectSolver()
        p.setup()
        p.set_val("fltcond|TempIncrement", 0, units="degC")
        p.set_val("ac|geom|wing|S_ref", 100, units="m**2")
        p.set_val("ac|geom|wing|AR", 10)
        p.set_val("ac|geom|wing|taper", 0.1)
        p.set_val("ac|geom|wing|c4sweep", 20, units="deg")
        p.set_val("ac|geom|wing|twist", twist, units="deg")
        p.set_val("ac|geom|wing|toverc", [0.07, 0.12])
        p.set_val("ac|aero|CD_nonwing", 0.01)
        p.set_val("fltcond|q", 5e3, units="Pa")
        p.set_val("fltcond|M", 0.45)
        p.set_val("fltcond|h", 7.5e3, units="m")
        p.run_model()

        # Generate mesh to pass to OpenAeroStruct
        mesh = om.Problem(VLM(num_x=2, num_y=4))
        mesh.setup()
        mesh.set_val("ac|geom|wing|OAS_mesh", p.get_val("twisted_mesh"))
        mesh.set_val("ac|geom|wing|toverc", p.get_val("t_over_c_interp.panel_toverc"))
        mesh.set_val("fltcond|M", 0.45)
        mesh.set_val("fltcond|h", 7.5e3, units="m")
        mesh.set_val("fltcond|alpha", 2, units="deg")
        mesh.run_model()

        p.set_val("fltcond|CL", mesh.get_val("fltcond|CL"))
        p.run_model()

        # Test on training point
        assert_near_equal(
            mesh.get_val("fltcond|CL"), p.get_val("aero_surrogate.CL"), tolerance=1e-10
        )  # check convergence
        assert_near_equal(2, p.get_val("alpha_bal.alpha", units="deg"), tolerance=1e-7)
        assert_near_equal(mesh.get_val("fltcond|CD") + 0.01, p.get_val("aero_surrogate.CD"), tolerance=2e-2)
        assert_near_equal(p.get_val("drag", units="N"), p.get_val("aero_surrogate.CD") * 100 * 5e3, tolerance=2e-2)

        # Test off training point
        mesh.set_val("fltcond|M", 0.3)
        mesh.set_val("fltcond|h", 4e3, units="m")
        mesh.set_val("fltcond|alpha", 6, units="deg")
        mesh.run_model()

        p.set_val("fltcond|M", 0.3)
        p.set_val("fltcond|h", 4e3, units="m")
        p.set_val("fltcond|CL", mesh.get_val("fltcond|CL"))
        p.run_model()

        assert_near_equal(
            mesh.get_val("fltcond|CL"), p.get_val("aero_surrogate.CL"), tolerance=1e-10
        )  # check convergence
        assert_near_equal(6, p.get_val("alpha_bal.alpha", units="deg"), tolerance=1e-2)
        assert_near_equal(mesh.get_val("fltcond|CD") + 0.01, p.get_val("aero_surrogate.CD"), tolerance=6e-2)
        assert_near_equal(p.get_val("drag", units="N"), p.get_val("aero_surrogate.CD") * 100 * 5e3, tolerance=5e-2)

    def test_surf_options(self):
        nn = 1
        twist = np.array([-1, 0, 1])
        p = om.Problem(
            VLMDragPolar(
                num_nodes=nn,
                num_x=2,
                num_y=4,
                num_twist=twist.size,
                Mach_train=np.linspace(0.1, 0.8, 2),
                alpha_train=np.linspace(-11, 15, 2),
                alt_train=np.linspace(0, 15e3, 2),
                surf_options={"k_lam": 0.9},
            )
        )
        p.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        p.model.linear_solver = om.DirectSolver()
        p.setup()
        p.set_val("fltcond|TempIncrement", 0, units="degC")
        p.set_val("ac|geom|wing|S_ref", 100, units="m**2")
        p.set_val("ac|geom|wing|AR", 10)
        p.set_val("ac|geom|wing|taper", 0.1)
        p.set_val("ac|geom|wing|c4sweep", 20, units="deg")
        p.set_val("ac|geom|wing|twist", twist, units="deg")
        p.set_val("ac|aero|CD_nonwing", 0.01)
        p.set_val("fltcond|q", 5e3 * np.ones(nn), units="Pa")
        p.set_val("fltcond|M", 0.5 * np.ones(nn))
        p.set_val("fltcond|h", 7.5e3 * np.ones(nn), units="m")
        p.set_val("fltcond|CL", 0.5 * np.ones(nn))
        p.run_model()

        # Ensure they're all the same
        assert_near_equal(p.get_val("drag", units="N"), 34905.69308752 * np.ones(nn), tolerance=1e-10)

    def test_vectorized(self):
        nn = 7
        twist = np.array([-1, 0, 1])
        p = om.Problem(
            VLMDragPolar(
                num_nodes=nn,
                num_x=2,
                num_y=4,
                num_twist=twist.size,
                Mach_train=np.linspace(0.1, 0.8, 2),
                alpha_train=np.linspace(-11, 15, 2),
                alt_train=np.linspace(0, 15e3, 2),
            )
        )
        p.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        p.model.linear_solver = om.DirectSolver()
        p.setup()
        p.set_val("fltcond|TempIncrement", 0, units="degC")
        p.set_val("ac|geom|wing|S_ref", 100, units="m**2")
        p.set_val("ac|geom|wing|AR", 10)
        p.set_val("ac|geom|wing|taper", 0.1)
        p.set_val("ac|geom|wing|c4sweep", 20, units="deg")
        p.set_val("ac|geom|wing|twist", twist, units="deg")
        p.set_val("ac|aero|CD_nonwing", 0.01)
        p.set_val("fltcond|q", 5e3 * np.ones(nn), units="Pa")
        p.set_val("fltcond|M", 0.5 * np.ones(nn))
        p.set_val("fltcond|h", 7.5e3 * np.ones(nn), units="m")
        p.set_val("fltcond|CL", 0.5 * np.ones(nn))
        p.run_model()

        # Ensure they're all the same
        assert_near_equal(p.get_val("drag", units="N"), 37615.14285108 * np.ones(nn), tolerance=1e-10)

    def test_section_geometry(self):
        nn = 1
        p = om.Problem(
            VLMDragPolar(
                num_nodes=nn,
                num_x=1,
                num_y=[2, 3],
                num_sections=3,
                geometry="section",
                Mach_train=np.linspace(0.1, 0.11, 2),
                alpha_train=np.linspace(1, 1.1, 2),
                alt_train=np.linspace(0, 2, 2),
            )
        )
        p.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        p.model.linear_solver = om.DirectSolver()
        p.setup()
        p.set_val("fltcond|TempIncrement", 0, units="degC")
        p.set_val("ac|geom|wing|S_ref", 100, units="m**2")
        p.set_val("ac|geom|wing|x_LE_sec", [1, 0.2, 0.0])
        p.set_val("ac|geom|wing|y_sec", [-1, 0.6])
        p.set_val("ac|geom|wing|chord_sec", [0.2, 0.3, 0.3])
        p.set_val("ac|geom|wing|twist", [-1, 0, 1], units="deg")
        p.set_val("ac|geom|wing|toverc", [0.1, 0.17, 0.2])
        p.set_val("ac|aero|CD_nonwing", 0.01)
        p.set_val("fltcond|q", 5e3, units="Pa")
        p.set_val("fltcond|M", 0.105)
        p.set_val("fltcond|h", 1, units="m")
        p.set_val("fltcond|CL", 0.15579806)
        p.run_model()

        vlm = om.Problem(VLM(num_x=1, num_y=5))
        vlm.setup()
        vlm.set_val("ac|geom|wing|OAS_mesh", p.get_val("twisted_mesh", units="m"), units="m")
        vlm.set_val("fltcond|alpha", 1.05, units="deg")
        vlm.set_val("ac|geom|wing|toverc", p.get_val("t_over_c_interp.panel_toverc"))
        vlm.set_val("fltcond|M", p.get_val("fltcond|M"))
        vlm.set_val("fltcond|h", p.get_val("fltcond|h"))
        vlm.run_model()

        # Ensure they're all the same
        assert_near_equal(vlm.get_val("fltcond|CL"), 0.15579806, tolerance=1e-3)
        assert_near_equal(vlm.get_val("fltcond|CD") + 0.01, p.get_val("aero_surrogate.CD"), tolerance=1e-3)

    def test_mesh_geometry_option(self):
        nn = 1
        p = om.Problem()
        p.model.add_subsystem(
            "mesher",
            TrapezoidalPlanformMesh(num_x=1, num_y=2),
            promotes=["*"],
        )
        p.model.add_subsystem(
            "vlm",
            VLMDragPolar(
                num_nodes=nn,
                num_x=1,
                num_y=2,
                geometry="mesh",
                num_twist=3,
                Mach_train=np.linspace(0.1, 0.11, 2),
                alpha_train=np.linspace(1, 1.1, 2),
                alt_train=np.linspace(0, 2, 2),
            ),
            promotes=["*"],
        )
        p.model.connect("mesh", "ac|geom|wing|OAS_mesh")
        p.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        p.model.linear_solver = om.DirectSolver()
        p.setup()

        p.set_val("fltcond|TempIncrement", 0, units="degC")
        p.set_val("S", 100, units="m**2")
        p.set_val("AR", 100)
        p.set_val("taper", 0.1)
        p.set_val("sweep", 20, units="deg")
        p.set_val("ac|aero|CD_nonwing", 0.001)
        p.set_val("ac|geom|wing|toverc", [0.1, 0.17])
        p.set_val("fltcond|q", 5e3, units="Pa")
        p.set_val("fltcond|M", 0.105)
        p.set_val("fltcond|h", 1, units="m")
        p.set_val("fltcond|CL", 0.10634777)
        p.run_model()

        vlm = om.Problem(VLM(num_x=1, num_y=2))
        vlm.setup()
        vlm.set_val("ac|geom|wing|OAS_mesh", p.get_val("ac|geom|wing|OAS_mesh", units="m"), units="m")
        vlm.set_val("fltcond|alpha", 1.05, units="deg")
        vlm.set_val("ac|geom|wing|toverc", p.get_val("ac|geom|wing|toverc"))
        vlm.set_val("fltcond|M", p.get_val("fltcond|M"))
        vlm.set_val("fltcond|h", p.get_val("fltcond|h"))
        vlm.run_model()

        # Ensure they're all the same
        assert_near_equal(vlm.get_val("fltcond|CL"), 0.10634777, tolerance=1e-6)
        assert_near_equal(vlm.get_val("fltcond|CD") + 0.001, p.get_val("aero_surrogate.CD"), tolerance=1e-4)


@unittest.skipIf(not OAS_installed, "OpenAeroStruct is not installed")
class VLMDataGenTestCase(unittest.TestCase):
    def tearDown(self):
        # Get rid of any specified surface options in the VLMDataGen
        # class after every test. This is necessary because the class
        # stores the surface options as a "static" variable and
        # prevents multiple VLMDataGen instances with different
        # surface options. Doing this prevents that error when doing
        # multiple tests with different surface options.
        del VLMDataGen.surf_options

    def test_defaults(self):
        # Regression test
        p = om.Problem()
        p.model.add_subsystem(
            "mesher",
            TrapezoidalPlanformMesh(num_x=1, num_y=2),
            promotes=["*"],
        )
        p.model.add_subsystem(
            "comp",
            VLMDataGen(
                num_x=1,
                num_y=2,
                Mach_train=np.linspace(0.1, 0.85, 2),
                alpha_train=np.linspace(-10, 15, 2),
                alt_train=np.linspace(0, 15e3, 2),
            ),
            promotes=["*"],
        )
        p.model.connect("mesh", "ac|geom|wing|OAS_mesh")
        p.setup()
        p.set_val("fltcond|TempIncrement", 0, units="degC")
        p.set_val("S", 100, units="m**2")
        p.set_val("AR", 10)
        p.set_val("taper", 0.1)
        p.set_val("sweep", 20, units="deg")
        p.set_val("ac|aero|CD_nonwing", 0.01)
        p.run_model()

        CL = np.array(
            [
                [[-0.86481567, -0.86481567], [1.28852469, 1.28852469]],
                [[-0.86481567, -0.86481567], [1.28852469, 1.28852469]],
            ]
        )
        CD = np.array(
            [[[0.03547695, 0.03770253], [0.05900183, 0.0612274]], [[0.03537478, 0.03719636], [0.18710518, 0.18892676]]]
        )

        assert_near_equal(CL, p.get_val("CL_train"), tolerance=1e-7)
        assert_near_equal(CD, p.get_val("CD_train"), tolerance=1e-7)

        partials = p.check_partials(out_stream=None, form="central")
        assert_check_partials(partials, atol=6e-5, rtol=2e-5)

    def test_different_surf_options(self):
        # Test that when there are different surf_options within a single model it catches it
        p = om.Problem()
        p.model.add_subsystem("one", VLMDataGen(surf_options={"a": 1.13521, "b": np.linspace(0, 1, 10)}))
        p.model.add_subsystem("two", VLMDataGen(surf_options={"a": 1.13521, "b": np.linspace(0, 1, 10)}))
        p.model.add_subsystem("three", VLMDataGen(surf_options={"a": 1.13521, "b": np.linspace(0, 1, 10)}))
        p.setup()

        p = om.Problem()
        p.model.add_subsystem("one", VLMDataGen(surf_options={"a": 1.13521}))
        p.model.add_subsystem("two", VLMDataGen(surf_options={"a": 1.1352}))
        self.assertRaises(ValueError, p.setup)

        p = om.Problem()
        p.model.add_subsystem("one", VLMDataGen(surf_options={"a": 1.13521, "b": np.linspace(0, 1, 10)}))
        p.model.add_subsystem("two", VLMDataGen(surf_options={"a": 1.13521, "b": np.linspace(0, 1.0001, 10)}))
        p.model.add_subsystem("three", VLMDataGen(surf_options={"a": 1.13521, "b": np.linspace(0, 1, 10)}))
        self.assertRaises(ValueError, p.setup)

        p = om.Problem()
        p.model.add_subsystem("one", VLMDataGen())
        p.model.add_subsystem("two", VLMDataGen(surf_options={"boof": True}))
        self.assertRaises(ValueError, p.setup)


@unittest.skipIf(not OAS_installed, "OpenAeroStruct is not installed")
class VLMTestCase(unittest.TestCase):
    def test_defaults(self):
        p = om.Problem()
        p.model.add_subsystem("mesh", TrapezoidalPlanformMesh(num_x=2, num_y=4), promotes=["*"])
        p.model.add_subsystem("vlm", VLM(num_x=2, num_y=4), promotes=["*"])
        p.model.connect("mesh", "ac|geom|wing|OAS_mesh")
        p.setup()
        p.set_val("fltcond|alpha", 2, units="deg")
        p.set_val("fltcond|M", 0.6)
        p.set_val("fltcond|h", 5e3, units="m")
        p.set_val("fltcond|TempIncrement", 0, units="degC")
        p.set_val("S", 100, units="m**2")
        p.set_val("AR", 10)
        p.set_val("taper", 0.1)
        p.set_val("sweep", 20, units="deg")

        p.run_model()

        # Run OpenAeroStruct with the same inputs
        inputs = {}
        inputs["mesh"] = p.get_val("mesh", units="m")
        inputs["twist"] = np.zeros(1)
        inputs["v"] = p.get_val("airspeed.Utrue", units="m/s")
        inputs["alpha"] = p.get_val("fltcond|alpha", units="deg")
        inputs["Mach_number"] = p.get_val("fltcond|M")
        inputs["re"] = p.get_val("Re_calc.re", units="1/m")
        inputs["rho"] = p.get_val("density.fltcond|rho", units="kg/m**3")

        exact = run_OAS(inputs)

        assert_near_equal(exact["CL"], p.get_val("fltcond|CL"))
        assert_near_equal(exact["CD"], p.get_val("fltcond|CD"))

    def test_wave_drag(self):
        p = om.Problem()
        p.model.add_subsystem("mesh", TrapezoidalPlanformMesh(num_x=2, num_y=4), promotes=["*"])
        p.model.add_subsystem("vlm", VLM(num_x=2, num_y=4, surf_options={"with_wave": False}), promotes=["*"])
        p.model.connect("mesh", "ac|geom|wing|OAS_mesh")
        p.setup()
        p.set_val("fltcond|alpha", 2, units="deg")
        p.set_val("fltcond|M", 0.85)
        p.set_val("fltcond|h", 5e3, units="m")
        p.set_val("fltcond|TempIncrement", 0, units="degC")
        p.set_val("S", 100, units="m**2")
        p.set_val("AR", 10)
        p.set_val("taper", 0.1)
        p.set_val("sweep", 20, units="deg")

        p.run_model()

        # Run OpenAeroStruct with the same inputs
        inputs = {}
        inputs["mesh"] = p.get_val("mesh", units="m")
        inputs["twist"] = np.zeros(1)
        inputs["v"] = p.get_val("airspeed.Utrue", units="m/s")
        inputs["alpha"] = p.get_val("fltcond|alpha", units="deg")
        inputs["Mach_number"] = p.get_val("fltcond|M")
        inputs["re"] = p.get_val("Re_calc.re", units="1/m")
        inputs["rho"] = p.get_val("density.fltcond|rho", units="kg/m**3")

        exact = run_OAS(inputs, with_wave=False)

        assert_near_equal(exact["CL"], p.get_val("fltcond|CL"))
        assert_near_equal(exact["CD"], p.get_val("fltcond|CD"))

    def test_viscous_drag(self):
        p = om.Problem()
        p.model.add_subsystem("mesh", TrapezoidalPlanformMesh(num_x=2, num_y=4), promotes=["*"])
        p.model.add_subsystem("vlm", VLM(num_x=2, num_y=4, surf_options={"with_viscous": False}), promotes=["*"])
        p.model.connect("mesh", "ac|geom|wing|OAS_mesh")
        p.setup()
        p.set_val("fltcond|alpha", 2, units="deg")
        p.set_val("fltcond|M", 0.85)
        p.set_val("fltcond|h", 5e3, units="m")
        p.set_val("fltcond|TempIncrement", 0, units="degC")
        p.set_val("S", 100, units="m**2")
        p.set_val("AR", 10)
        p.set_val("taper", 0.1)
        p.set_val("sweep", 20, units="deg")

        p.run_model()

        # Run OpenAeroStruct with the same inputs
        inputs = {}
        inputs["mesh"] = p.get_val("mesh", units="m")
        inputs["twist"] = np.zeros(1)
        inputs["v"] = p.get_val("airspeed.Utrue", units="m/s")
        inputs["alpha"] = p.get_val("fltcond|alpha", units="deg")
        inputs["Mach_number"] = p.get_val("fltcond|M")
        inputs["re"] = p.get_val("Re_calc.re", units="1/m")
        inputs["rho"] = p.get_val("density.fltcond|rho", units="kg/m**3")

        exact = run_OAS(inputs, with_viscous=False)

        assert_near_equal(exact["CL"], p.get_val("fltcond|CL"))
        assert_near_equal(exact["CD"], p.get_val("fltcond|CD"))

    def test_t_over_c(self):
        p = om.Problem()
        p.model.add_subsystem("mesh", TrapezoidalPlanformMesh(num_x=2, num_y=2), promotes=["*"])
        p.model.add_subsystem("vlm", VLM(num_x=2, num_y=2), promotes=["*"])
        p.model.connect("mesh", "ac|geom|wing|OAS_mesh")
        p.setup()
        p.set_val("fltcond|alpha", 2, units="deg")
        p.set_val("fltcond|M", 0.85)
        p.set_val("fltcond|h", 5e3, units="m")
        p.set_val("fltcond|TempIncrement", 0, units="degC")
        p.set_val("S", 100, units="m**2")
        p.set_val("AR", 10)
        p.set_val("taper", 0.1)
        p.set_val("sweep", 20, units="deg")
        p.set_val("ac|geom|wing|toverc", np.array([0.1, 0.2]))

        p.run_model()

        # Run OpenAeroStruct with the same inputs
        inputs = {}
        inputs["mesh"] = p.get_val("mesh", units="m")
        inputs["twist"] = np.zeros(1)
        inputs["v"] = p.get_val("airspeed.Utrue", units="m/s")
        inputs["alpha"] = p.get_val("fltcond|alpha", units="deg")
        inputs["Mach_number"] = p.get_val("fltcond|M")
        inputs["re"] = p.get_val("Re_calc.re", units="1/m")
        inputs["rho"] = p.get_val("density.fltcond|rho", units="kg/m**3")

        exact = run_OAS(inputs, t_over_c=np.array([0.1, 0.2]))

        assert_near_equal(exact["CL"], p.get_val("fltcond|CL"))
        assert_near_equal(exact["CD"], p.get_val("fltcond|CD"))


def run_OAS(inputs, with_viscous=True, with_wave=True, t_over_c=None):
    """
    Runs OpenAeroStruct with flight condition and mesh inputs.

    Inputs
    ------
    inputs : dict
        Input dictionary containing
            mesh : ndarray
                Flat wing mesh (m)
            twist : ndarray
                Twist control points (deg)
            v : float
                Flight speed (m/s)
            alpha : float
                Angle of attack (deg)
            Mach_number : float
                Mach number
            re : float
                Dimensional Reynolds number (1/m)
            rho : float
                Flow density (kg/m^3)
    with_viscous : bool (optional)
        Include viscous drag
    with_wave : bool (optional)
        Include wave drag
    t_over_c : float (optional)
        Thickness to chord ratio of the airfoil

    Outputs
    -------
    outputs : dict
        Output dictionary containing
            CL : float
                Lift coefficient
            CD : float
                Drag coefficient
    """
    if t_over_c is None:
        t_over_c = np.array([0.12])

    # Create a dictionary with info and options about the aerodynamic
    # lifting surface
    surface = {
        # Wing definition
        "name": "wing",  # name of the surface
        "symmetry": True,  # if true, model one half of wing
        # reflected across the plane y = 0
        "S_ref_type": "projected",  # how we compute the wing area,
        # can be 'wetted' or 'projected'
        "twist_cp": inputs["twist"],
        "mesh": inputs["mesh"],
        # Aerodynamic performance of the lifting surface at
        # an angle of attack of 0 (alpha=0).
        # These CL0 and CD0 values are added to the CL and CD
        # obtained from aerodynamic analysis of the surface to get
        # the total CL and CD.
        # These CL0 and CD0 values do not vary wrt alpha.
        "CL0": 0.0,  # CL of the surface at alpha=0
        "CD0": 0.0,  # CD of the surface at alpha=0
        # Airfoil properties for viscous drag calculation
        "k_lam": 0.05,  # percentage of chord with laminar
        # flow, used for viscous drag
        "t_over_c_cp": t_over_c,  # thickness over chord ratio (NACA0015)
        "c_max_t": 0.37,  # chordwise location of maximum (NACA0015)
        # thickness
        "with_viscous": with_viscous,  # if true, compute viscous drag
        "with_wave": with_wave,  # if true, compute wave drag
    }

    # Create the OpenMDAO problem
    prob = om.Problem()

    # Create an independent variable component that will supply the flow
    # conditions to the problem.
    indep_var_comp = om.IndepVarComp()
    indep_var_comp.add_output("v", val=inputs["v"], units="m/s")
    indep_var_comp.add_output("alpha", val=inputs["alpha"], units="deg")
    indep_var_comp.add_output("Mach_number", val=inputs["Mach_number"])
    indep_var_comp.add_output("re", val=inputs["re"], units="1/m")
    indep_var_comp.add_output("rho", val=inputs["rho"], units="kg/m**3")
    indep_var_comp.add_output("cg", val=np.zeros((3)), units="m")

    # Add this IndepVarComp to the problem model
    prob.model.add_subsystem("prob_vars", indep_var_comp, promotes=["*"])

    # Create and add a group that handles the geometry for the
    # aerodynamic lifting surface
    geom_group = Geometry(surface=surface)
    prob.model.add_subsystem(surface["name"], geom_group)

    # Create the aero point group, which contains the actual aerodynamic
    # analyses
    aero_group = AeroPoint(surfaces=[surface])
    point_name = "aero_point_0"
    prob.model.add_subsystem(point_name, aero_group, promotes_inputs=["v", "alpha", "Mach_number", "re", "rho", "cg"])

    name = surface["name"]

    # Connect the mesh from the geometry component to the analysis point
    prob.model.connect(name + ".mesh", point_name + "." + name + ".def_mesh")

    # Perform the connections with the modified names within the
    # 'aero_states' group.
    prob.model.connect(name + ".mesh", point_name + ".aero_states." + name + "_def_mesh")
    prob.model.connect(name + ".t_over_c", point_name + "." + name + "_perf." + "t_over_c")

    # Set up and run the model
    prob.setup()
    prob.run_model()
    outputs = {}
    outputs["CL"] = prob["aero_point_0.wing_perf.CL"]
    outputs["CD"] = prob["aero_point_0.wing_perf.CD"]
    return outputs


@unittest.skipIf(not OAS_installed, "OpenAeroStruct is not installed")
class ExampleUsageTestCase(unittest.TestCase):
    def test(self):
        # Test that it runs with no errors
        example_usage()

        # Get rid of any specified surface options in the VLMDataGen
        # class after every test. This is necessary because the class
        # stores the surface options as a "static" variable and
        # prevents multiple VLMDataGen instances with different
        # surface options. Doing this prevents that error when doing
        # multiple tests with different surface options.
        del VLMDataGen.surf_options


if __name__ == "__main__":
    unittest.main()
