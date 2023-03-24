import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
import openmdao.api as om

# Only run if OpenAeroStruct is installed
try:
    from openconcept.aerodynamics.openaerostruct.aerostructural import (
        OASDataGen,
        Aerostruct,
        AerostructDragPolar,
        AerostructDragPolarExact,
        example_usage,
    )

    OAS_installed = True
except ImportError:
    OAS_installed = False


@unittest.skipIf(not OAS_installed, "OpenAeroStruct is not installed")
class AerostructDragPolarTestCase(unittest.TestCase):
    def tearDown(self):
        # Get rid of any specified surface options in the OASDataGen
        # class after every test. This is necessary because the class
        # stores the surface options as a "static" variable and
        # prevents multiple OASDataGen instances with different
        # surface options. Doing this prevents that error when doing
        # multiple tests with different surface options.
        del OASDataGen.surf_options

    def test(self):
        S = 100  # m^2
        AR = 10
        taper = 0.1
        sweep = 20  # deg

        twist = np.array([-1, -0.5, 2])  # deg
        toverc = np.array([0.05, 0.1, 0.12])
        t_skin = np.array([5, 13, 15])  # mm
        t_spar = np.array([5, 13, 15])  # mm

        M = 0.45
        h = 7.5e3  # m
        alpha = 2.0  # deg
        q = 5e3  # Pa

        CD_nonwing = 0.01

        # Generate mesh to pass to OpenAeroStruct
        mesh = om.Problem(
            Aerostruct(
                num_x=2,
                num_y=6,
                num_twist=twist.size,
                num_toverc=toverc.size,
                num_skin=t_skin.size,
                num_spar=t_spar.size,
            )
        )
        mesh.setup()
        mesh.set_val("ac|geom|wing|S_ref", S, units="m**2")
        mesh.set_val("ac|geom|wing|AR", AR)
        mesh.set_val("ac|geom|wing|taper", taper)
        mesh.set_val("ac|geom|wing|c4sweep", sweep, units="deg")
        mesh.set_val("ac|geom|wing|twist", twist, units="deg")
        mesh.set_val("ac|geom|wing|toverc", toverc)
        mesh.set_val("ac|geom|wing|skin_thickness", t_skin, units="mm")
        mesh.set_val("ac|geom|wing|spar_thickness", t_spar, units="mm")
        mesh.set_val("fltcond|M", M)
        mesh.set_val("fltcond|h", h, units="m")
        mesh.set_val("fltcond|alpha", alpha, units="deg")
        mesh.run_model()

        p = om.Problem(
            AerostructDragPolar(
                num_nodes=1,
                num_x=2,
                num_y=6,
                num_twist=twist.size,
                num_toverc=toverc.size,
                num_skin=t_skin.size,
                num_spar=t_spar.size,
                Mach_train=np.linspace(0.1, 0.8, 3),
                alpha_train=np.linspace(-11, 15, 3),
                alt_train=np.linspace(0, 15e3, 2),
            )
        )
        p.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        p.model.linear_solver = om.DirectSolver()
        p.setup()
        p.set_val("fltcond|TempIncrement", 0, units="degC")
        p.set_val("ac|geom|wing|S_ref", S, units="m**2")
        p.set_val("ac|geom|wing|AR", AR)
        p.set_val("ac|geom|wing|taper", taper)
        p.set_val("ac|geom|wing|c4sweep", sweep, units="deg")
        p.set_val("ac|geom|wing|twist", twist, units="deg")
        p.set_val("ac|geom|wing|toverc", toverc)
        p.set_val("ac|geom|wing|skin_thickness", t_skin, units="mm")
        p.set_val("ac|geom|wing|spar_thickness", t_spar, units="mm")
        p.set_val("ac|aero|CD_nonwing", CD_nonwing)
        p.set_val("fltcond|q", q, units="Pa")
        p.set_val("fltcond|M", M)
        p.set_val("fltcond|h", h, units="m")
        p.set_val("fltcond|CL", mesh.get_val("fltcond|CL"))
        p.run_model()

        # Test on training point
        assert_near_equal(
            mesh.get_val("fltcond|CL"), p.get_val("aero_surrogate.CL"), tolerance=1e-10
        )  # check convergence
        assert_near_equal(alpha, p.get_val("alpha_bal.alpha", units="deg"), tolerance=2e-2)
        assert_near_equal(mesh.get_val("fltcond|CD") + CD_nonwing, p.get_val("aero_surrogate.CD"), tolerance=2e-2)
        assert_near_equal(p.get_val("drag", units="N"), p.get_val("aero_surrogate.CD") * S * q, tolerance=2e-2)

        # Test off training point
        M = 0.3
        h = 4e3  # m
        alpha = 6.0  # deg

        mesh.set_val("fltcond|M", M)
        mesh.set_val("fltcond|h", h, units="m")
        mesh.set_val("fltcond|alpha", alpha, units="deg")
        mesh.run_model()

        p.set_val("fltcond|M", M)
        p.set_val("fltcond|h", h, units="m")
        p.set_val("fltcond|CL", mesh.get_val("fltcond|CL"))
        p.run_model()

        assert_near_equal(
            mesh.get_val("fltcond|CL"), p.get_val("aero_surrogate.CL"), tolerance=1e-10
        )  # check convergence
        assert_near_equal(alpha, p.get_val("alpha_bal.alpha", units="deg"), tolerance=2e-2)
        assert_near_equal(mesh.get_val("fltcond|CD") + 0.01, p.get_val("aero_surrogate.CD"), tolerance=5e-2)
        assert_near_equal(p.get_val("drag", units="N"), p.get_val("aero_surrogate.CD") * S * q, tolerance=5e-2)

    def test_surf_options(self):
        nn = 1
        twist = np.array([-1, -0.5, 2])  # deg
        toverc = np.array([0.05, 0.1, 0.12])
        t_skin = np.array([5, 13, 15])  # mm
        t_spar = np.array([5, 13, 15])  # mm
        p = om.Problem(
            AerostructDragPolar(
                num_nodes=nn,
                num_x=2,
                num_y=6,
                num_twist=twist.size,
                num_toverc=toverc.size,
                num_skin=t_skin.size,
                num_spar=t_spar.size,
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
        p.set_val("ac|geom|wing|toverc", toverc)
        p.set_val("ac|geom|wing|skin_thickness", t_skin, units="mm")
        p.set_val("ac|geom|wing|spar_thickness", t_spar, units="mm")
        p.set_val("ac|aero|CD_nonwing", 0.01)
        p.set_val("fltcond|q", 5e3 * np.ones(nn), units="Pa")
        p.set_val("fltcond|M", 0.5 * np.ones(nn))
        p.set_val("fltcond|h", 7.5e3 * np.ones(nn), units="m")
        p.set_val("fltcond|CL", 0.5 * np.ones(nn))
        p.run_model()

        # Ensure they're all the same
        assert_near_equal(p.get_val("drag", units="N"), 33058.43316461 * np.ones(nn), tolerance=1e-10)

    def test_vectorized(self):
        nn = 7
        twist = np.array([-1, -0.5, 2])  # deg
        toverc = np.array([0.05, 0.1, 0.12])
        t_skin = np.array([5, 13, 15])  # mm
        t_spar = np.array([5, 13, 15])  # mm
        p = om.Problem(
            AerostructDragPolar(
                num_nodes=nn,
                num_x=2,
                num_y=6,
                num_twist=twist.size,
                num_toverc=toverc.size,
                num_skin=t_skin.size,
                num_spar=t_spar.size,
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
        p.set_val("ac|geom|wing|toverc", toverc)
        p.set_val("ac|geom|wing|skin_thickness", t_skin, units="mm")
        p.set_val("ac|geom|wing|spar_thickness", t_spar, units="mm")
        p.set_val("ac|aero|CD_nonwing", 0.01)
        p.set_val("fltcond|q", 5e3 * np.ones(nn), units="Pa")
        p.set_val("fltcond|M", 0.5 * np.ones(nn))
        p.set_val("fltcond|h", 7.5e3 * np.ones(nn), units="m")
        p.set_val("fltcond|CL", 0.5 * np.ones(nn))
        p.run_model()

        # Ensure they're all the same
        assert_near_equal(p.get_val("drag", units="N"), 35692.26543182 * np.ones(nn), tolerance=1e-10)


@unittest.skipIf(not OAS_installed, "OpenAeroStruct is not installed")
class OASDataGenTestCase(unittest.TestCase):
    def tearDown(self):
        # Get rid of any specified surface options in the OASDataGen
        # class after every test. This is necessary because the class
        # stores the surface options as a "static" variable and
        # prevents multiple OASDataGen instances with different
        # surface options. Doing this prevents that error when doing
        # multiple tests with different surface options.
        del OASDataGen.surf_options

    def test_defaults(self):
        # Regression test
        twist = np.array([-1, -0.5, 2])  # deg
        toverc = np.array([0.05, 0.1, 0.12])
        t_skin = np.array([5, 13, 15])  # mm
        t_spar = np.array([5, 13, 15])  # mm
        p = om.Problem()
        p.model.add_subsystem(
            "comp",
            OASDataGen(
                num_x=2,
                num_y=6,
                num_twist=twist.size,
                num_toverc=toverc.size,
                num_skin=t_skin.size,
                num_spar=t_spar.size,
                Mach_train=np.linspace(0.1, 0.85, 2),
                alpha_train=np.linspace(-10, 15, 2),
                alt_train=np.linspace(0, 15e3, 2),
            ),
            promotes=["*"],
        )
        p.setup()
        p.set_val("fltcond|TempIncrement", 0, units="degC")
        p.set_val("ac|geom|wing|S_ref", 100, units="m**2")
        p.set_val("ac|geom|wing|AR", 10)
        p.set_val("ac|geom|wing|taper", 0.1)
        p.set_val("ac|geom|wing|c4sweep", 20, units="deg")
        p.set_val("ac|geom|wing|twist", twist, units="deg")
        p.set_val("ac|geom|wing|toverc", toverc)
        p.set_val("ac|geom|wing|skin_thickness", t_skin, units="mm")
        p.set_val("ac|geom|wing|spar_thickness", t_spar, units="mm")
        p.set_val("ac|aero|CD_nonwing", 0.01)
        p.run_model()

        # Check that the values don't change
        CL = np.array(
            [
                [[-0.79243052, -0.79557334], [1.30041913, 1.30407159]],
                [[-0.62943499, -0.76690382], [1.08568083, 1.26980739]],
            ]
        )
        CD = np.array(
            [[[0.04196198, 0.04421198], [0.07526711, 0.07758053]], [[0.03631444, 0.04259311], [0.11894307, 0.158035]]]
        )

        assert_near_equal(p.get_val("CL_train"), CL, tolerance=1e-7)
        assert_near_equal(p.get_val("CD_train"), CD, tolerance=1e-7)

        # The fundamental problem with the derivative inaccuracy is rooted in OpenAeroStruct,
        # not this code. There are no analytic derivatives explicitly implemented in any of
        # the components in aerostructural.py, so the purpose for this check is just to
        # ensure that the derivatives are assembled properly.
        partials = p.check_partials(step=1e-7)
        assert_check_partials(partials, atol=1e-2, rtol=1.5e-2)

    def test_different_surf_options(self):
        # Test that when there are different surf_options within a single model it catches it
        p = om.Problem()
        p.model.add_subsystem("one", OASDataGen(surf_options={"a": 1.13521, "b": np.linspace(0, 1, 10)}))
        p.model.add_subsystem("two", OASDataGen(surf_options={"a": 1.13521, "b": np.linspace(0, 1, 10)}))
        p.model.add_subsystem("three", OASDataGen(surf_options={"a": 1.13521, "b": np.linspace(0, 1, 10)}))
        p.setup()

        p = om.Problem()
        p.model.add_subsystem("one", OASDataGen(surf_options={"a": 1.13521}))
        p.model.add_subsystem("two", OASDataGen(surf_options={"a": 1.1352}))
        self.assertRaises(ValueError, p.setup)

        p = om.Problem()
        p.model.add_subsystem("one", OASDataGen(surf_options={"a": 1.13521, "b": np.linspace(0, 1, 10)}))
        p.model.add_subsystem("two", OASDataGen(surf_options={"a": 1.13521, "b": np.linspace(0, 1.0001, 10)}))
        p.model.add_subsystem("three", OASDataGen(surf_options={"a": 1.13521, "b": np.linspace(0, 1, 10)}))
        self.assertRaises(ValueError, p.setup)

        p = om.Problem()
        p.model.add_subsystem("one", OASDataGen())
        p.model.add_subsystem("two", OASDataGen(surf_options={"boof": True}))
        self.assertRaises(ValueError, p.setup)


@unittest.skipIf(not OAS_installed, "OpenAeroStruct is not installed")
class AerostructTestCase(unittest.TestCase):
    def get_prob(self, surf_dict={}):
        p = om.Problem(
            Aerostruct(num_x=2, num_y=4, num_twist=2, num_toverc=2, num_skin=2, num_spar=2, surf_options=surf_dict)
        )
        p.setup()
        p.set_val("fltcond|alpha", 3.0, units="deg")
        p.set_val("fltcond|M", 0.85)
        p.set_val("fltcond|h", 7.5e3, units="m")
        p.set_val("fltcond|TempIncrement", 1.0, units="degC")
        p.set_val("ac|geom|wing|S_ref", 427.8, units="m**2")
        p.set_val("ac|geom|wing|AR", 9.82)
        p.set_val("ac|geom|wing|taper", 0.149)
        p.set_val("ac|geom|wing|c4sweep", 31.6, units="deg")
        p.set_val("ac|geom|wing|twist", np.array([-1, 1]), units="deg")
        p.set_val("ac|geom|wing|toverc", np.array([0.12, 0.12]))
        p.set_val("ac|geom|wing|skin_thickness", np.array([0.005, 0.025]), units="m")
        p.set_val("ac|geom|wing|spar_thickness", np.array([0.004, 0.01]), units="m")

        return p

    def test_defaults(self):
        p = self.get_prob()
        p.run_model()

        # Use values computed offline from an OAS wingbox case with the same inputs
        assert_near_equal(p.get_val("fltcond|CL"), 0.22369546, tolerance=1e-6)
        assert_near_equal(p.get_val("fltcond|CD"), 0.015608634462089457, tolerance=1e-6)
        assert_near_equal(p.get_val("failure"), -0.64781499, tolerance=1e-6)
        assert_near_equal(p.get_val("ac|weights|W_wing", units="kg"), 29322.10058108, tolerance=1e-6)

    def test_wave_drag(self):
        p = self.get_prob(surf_dict={"with_wave": False})
        p.run_model()

        # Use values computed offline from an OAS wingbox case with the same inputs
        assert_near_equal(p.get_val("fltcond|CL"), 0.22369546, tolerance=1e-6)
        assert_near_equal(p.get_val("fltcond|CD"), 0.015457034121371742, tolerance=1e-6)
        assert_near_equal(p.get_val("failure"), -0.64781499, tolerance=1e-6)
        assert_near_equal(p.get_val("ac|weights|W_wing", units="kg"), 29322.10058108, tolerance=1e-6)

    def test_viscous_drag(self):
        p = self.get_prob(surf_dict={"with_viscous": False})
        p.run_model()

        # Use values computed offline from an OAS wingbox case with the same inputs
        assert_near_equal(p.get_val("fltcond|CL"), 0.22369546, tolerance=1e-6)
        assert_near_equal(p.get_val("fltcond|CD"), 0.009647318876399, tolerance=1e-6)
        assert_near_equal(p.get_val("failure"), -0.64781499, tolerance=1e-6)
        assert_near_equal(p.get_val("ac|weights|W_wing", units="kg"), 29322.10058108, tolerance=1e-6)


@unittest.skipIf(not OAS_installed, "OpenAeroStruct is not installed")
class AerostructDragPolarExactTestCase(unittest.TestCase):
    def test_defaults(self):
        S = 427.8
        CD0 = 0.01
        q = 0.5 * 0.55427276 * 264.20682682**2
        nn = 3
        p = om.Problem(
            AerostructDragPolarExact(num_nodes=nn, num_x=2, num_y=4, num_twist=2, num_toverc=2, num_skin=2, num_spar=2)
        )
        p.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, atol=1e-8, rtol=1e-10)
        p.model.linear_solver = om.DirectSolver()
        p.model.set_input_defaults(
            "fltcond|CL",
            np.array([0.094142402327027, 0.158902999486838, 0.223695460208479]),
        )
        p.model.set_input_defaults("fltcond|M", np.full(nn, 0.85))
        p.model.set_input_defaults("fltcond|h", np.full(nn, 7.5e3), units="m")
        p.model.set_input_defaults("fltcond|q", np.full(nn, q), units="Pa")
        p.model.set_input_defaults("ac|geom|wing|S_ref", S, units="m**2")
        p.setup()
        p.set_val("fltcond|TempIncrement", 1.0, units="degC")
        p.set_val("ac|geom|wing|AR", 9.82)
        p.set_val("ac|geom|wing|taper", 0.149)
        p.set_val("ac|geom|wing|c4sweep", 31.6, units="deg")
        p.set_val("ac|geom|wing|twist", np.array([-1, 1]), units="deg")
        p.set_val("ac|geom|wing|toverc", np.array([0.12, 0.12]))
        p.set_val("ac|geom|wing|skin_thickness", np.array([0.005, 0.025]), units="m")
        p.set_val("ac|geom|wing|spar_thickness", np.array([0.004, 0.01]), units="m")
        p.set_val("ac|aero|CD_nonwing", CD0)

        p.run_model()

        # Use values computed offline from an OAS wingbox case with the same inputs
        CD = CD0 + np.array([0.014130134503259, 0.014710068221375, 0.015608634461878])
        assert_near_equal(p.get_val("drag"), q * S * CD, tolerance=1e-6)
        assert_near_equal(p.get_val("failure"), np.array([-0.89649433, -0.77578479, -0.64781499]), tolerance=1e-6)
        assert_near_equal(p.get_val("ac|weights|W_wing", units="kg"), 29322.10058108, tolerance=1e-6)


@unittest.skipIf(not OAS_installed, "OpenAeroStruct is not installed")
class ExampleUsageTestCase(unittest.TestCase):
    def test(self):
        # Test that it runs with no errors
        example_usage()


if __name__ == "__main__":
    unittest.main()
