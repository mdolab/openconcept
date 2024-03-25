import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
import openmdao.api as om
from openconcept.energy_storage.hydrogen.structural import (
    VacuumTankWeight,
    VacuumWallThickness,
    PressureVesselWallThickness,
    MLIWeight,
)


class VacuumTankWeightTestCase(unittest.TestCase):
    def test_simple(self):
        """
        Regression test with some reasonable values that also checks the partials.
        """
        p = om.Problem()
        p.model.add_subsystem("model", VacuumTankWeight(), promotes=["*"])

        p.setup(force_alloc_complex=True)

        p.set_val("environment_design_pressure", 1, units="atm")
        p.set_val("max_expected_operating_pressure", 3, units="atm")
        p.set_val("vacuum_gap", 2, units="inch")
        p.set_val("radius", 1.0, units="m")
        p.set_val("length", 1.0, units="m")
        p.set_val("N_layers", 10)

        p.run_model()

        assert_near_equal(p.get_val("weight", units="kg"), 369.22641856, tolerance=1e-8)

        partials = p.check_partials(method="cs")
        assert_check_partials(partials, atol=1e-12, rtol=1e-12)

    def test_different_options(self):
        """
        Regression test with some reasonable values that also checks the partials.
        """
        p = om.Problem()
        p.model.add_subsystem(
            "model",
            VacuumTankWeight(
                weight_fudge_factor=1.0, stiffening_multiplier=0.4, inner_safety_factor=2.0, outer_safety_factor=3.0
            ),
            promotes=["*"],
        )

        p.setup(force_alloc_complex=True)

        p.set_val("environment_design_pressure", 1.2, units="atm")
        p.set_val("max_expected_operating_pressure", 5, units="atm")
        p.set_val("vacuum_gap", 1, units="inch")
        p.set_val("radius", 1.0, units="m")
        p.set_val("length", 0.0, units="m")
        p.set_val("N_layers", 30)

        p.run_model()

        assert_near_equal(p.get_val("weight", units="kg"), 149.06278353, tolerance=1e-8)

        partials = p.check_partials(method="cs")
        assert_check_partials(partials, atol=1e-12, rtol=1e-12)


class PressureVesselWallThicknessTestCase(unittest.TestCase):
    def test_simple(self):
        """
        Regression test with some reasonable values that also checks the partials.
        """
        P = 3e5  # Pa
        r = 1  # m
        L = 0.8  # m
        SF = 1.7
        yield_stress = 1e7
        rho = 1.0

        p = om.Problem()
        p.model.add_subsystem(
            "model",
            PressureVesselWallThickness(safety_factor=SF, yield_stress=yield_stress, density=rho),
            promotes=["*"],
        )

        p.setup(force_alloc_complex=True)

        p.set_val("design_pressure_differential", P, units="Pa")
        p.set_val("radius", r, units="m")
        p.set_val("length", L, units="m")

        p.run_model()

        t = P * r * SF / yield_stress
        W = (4 * np.pi * r**2 + 2 * np.pi * r * L) * t * rho
        assert_near_equal(p.get_val("thickness", units="m"), t, tolerance=1e-13)
        assert_near_equal(p.get_val("weight", units="kg"), W, tolerance=1e-13)

        partials = p.check_partials(method="cs")
        assert_check_partials(partials, atol=1e-12, rtol=1e-12)


class VacuumWallThicknessTestCase(unittest.TestCase):
    def test_sphere(self):
        """
        Test a sphere to check the case where the hemispheres buckle first.
        """
        p = om.Problem()
        p.model.add_subsystem("model", VacuumWallThickness(), promotes=["*"])

        p.setup(force_alloc_complex=True)

        p.set_val("design_pressure_differential", 1, units="atm")
        p.set_val("radius", 1.0, units="m")
        p.set_val("length", 0.0, units="m")

        p.run_model()

        assert_near_equal(p.get_val("thickness", units="m"), 0.002107520779403, tolerance=1e-8)
        assert_near_equal(p.get_val("weight", units="kg"), 71.48001153, tolerance=1e-8)

        partials = p.check_partials(method="cs")
        assert_check_partials(partials, atol=1e-12, rtol=1e-12)

    def test_cylinder(self):
        """
        Test a cylinder to check the case where the cylndrical portion buckles first.
        """
        p = om.Problem()
        p.model.add_subsystem("model", VacuumWallThickness(), promotes=["*"])

        p.setup(force_alloc_complex=True)

        p.set_val("design_pressure_differential", 1, units="atm")
        p.set_val("radius", 1.0, units="m")
        p.set_val("length", 10.0, units="m")

        p.run_model()

        assert_near_equal(p.get_val("thickness", units="m"), 0.011996028931134, tolerance=1e-8)
        assert_near_equal(p.get_val("weight", units="kg"), 2441.189557, tolerance=1e-8)

        partials = p.check_partials(method="cs")
        assert_check_partials(partials, atol=1e-12, rtol=1e-12)


class MLIWeightTestCase(unittest.TestCase):
    def test_simple(self):
        rho_spacer = 0.5  # kg/m^2
        rho_foil = 1.0  # kg/m^2
        r = 1.0  # m
        L = 0.5  # m
        N = 10

        p = om.Problem()
        p.model.add_subsystem(
            "model", MLIWeight(foil_layer_areal_weight=rho_foil, spacer_layer_areal_weight=rho_spacer), promotes=["*"]
        )

        p.setup(force_alloc_complex=True)

        p.set_val("N_layers", N)
        p.set_val("radius", r, units="m")
        p.set_val("length", L, units="m")

        p.run_model()

        W = (4 * np.pi * r**2 + 2 * np.pi * r * L) * N * (rho_spacer + rho_foil)
        assert_near_equal(p.get_val("weight", units="kg"), W, tolerance=1e-13)

        partials = p.check_partials(method="cs")
        assert_check_partials(partials, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
