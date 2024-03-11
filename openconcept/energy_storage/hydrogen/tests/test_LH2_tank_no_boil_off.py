import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
import openmdao.api as om
from openconcept.energy_storage.hydrogen.LH2_tank_no_boil_off import LH2TankNoBoilOff, InitialLH2MassModification


class LH2TankTestCase(unittest.TestCase):
    def test_simple(self):
        p = om.Problem()
        p.model = LH2TankNoBoilOff(fill_level_init=0.95)
        p.setup(force_alloc_complex=True)

        p.run_model()

        assert_near_equal(p.get_val("m_liq", units="kg"), 387.66337047, tolerance=1e-9)
        assert_near_equal(p.get_val("fill_level"), 0.95, tolerance=1e-9)
        assert_near_equal(p.get_val("tank_weight", units="kg"), 252.70942027, tolerance=1e-9)
        assert_near_equal(
            p.get_val("total_weight", units="kg"),
            p.get_val("tank_weight", units="kg") + p.get_val("m_liq", units="kg"),
            tolerance=1e-9,
        )

    def test_time_history(self):
        nn = 5

        p = om.Problem()
        p.model = LH2TankNoBoilOff(num_nodes=nn, fill_level_init=0.95)
        p.setup(force_alloc_complex=True)

        duration = nn - 1  # sec

        p.set_val("radius", 0.7, units="m")
        p.set_val("length", 1.7, units="m")
        p.set_val("m_dot_liq", 1.0, units="kg/s")
        p.set_val("integ.duration", duration, units="s")

        p.run_model()

        assert_near_equal(p.get_val("m_liq", units="kg"), 272.84452856480556 - np.arange(nn), tolerance=1e-9)
        assert_near_equal(
            p.get_val("fill_level"), np.array([0.95, 0.94651816, 0.94303633, 0.93955449, 0.93607265]), tolerance=1e-8
        )
        assert_near_equal(p.get_val("tank_weight", units="kg"), 263.38260155, tolerance=1e-9)
        assert_near_equal(
            p.get_val("total_weight", units="kg"),
            p.get_val("tank_weight", units="kg") + p.get_val("m_liq", units="kg"),
            tolerance=1e-9,
        )

        partials = p.check_partials(method="cs")
        assert_check_partials(partials)


class InitialLH2MassModificationTestCase(unittest.TestCase):
    def test_init_values(self):
        p = om.Problem()

        fill = 0.6
        rho = 70.0

        p.model.add_subsystem(
            "model",
            InitialLH2MassModification(
                num_nodes=1,
                fill_level_init=fill,
                LH2_density=rho,
            ),
            promotes=["*"],
        )

        p.setup()

        r = 1.3  # m
        L = 0.7  # m

        p.set_val("radius", r, units="m")
        p.set_val("length", L, units="m")

        # Set input to zero so we can test the computed initial values
        p.set_val("delta_m_liq", 0.0, units="kg")

        p.run_model()

        V_tank = 4 / 3 * np.pi * r**3 + np.pi * r**2 * L
        m_liq = V_tank * fill * rho

        assert_near_equal(p.get_val("m_liq", units="kg"), m_liq, tolerance=1e-12)
        assert_near_equal(p.get_val("fill_level"), fill, tolerance=1e-12)

    def test_vectorized(self):
        p = om.Problem()

        nn = 5
        fill = 0.6
        rho = 70.0

        p.model.add_subsystem(
            "model",
            InitialLH2MassModification(
                num_nodes=nn,
                fill_level_init=fill,
                LH2_density=rho,
            ),
            promotes=["*"],
        )

        p.setup()

        r = 1.3  # m
        L = 0.7  # m

        p.set_val("radius", r, units="m")
        p.set_val("length", L, units="m")

        # Set a specified change to liquid mass
        val = np.linspace(-10, 10, nn)
        p.set_val("delta_m_liq", val, units="kg")

        p.run_model()

        V_tank = 4 / 3 * np.pi * r**3 + np.pi * r**2 * L
        m_liq = V_tank * fill * rho

        assert_near_equal(p.get_val("m_liq", units="kg"), m_liq - val, tolerance=1e-12)
        assert_near_equal(p.get_val("fill_level"), fill - val / rho / V_tank, tolerance=1e-12)

    def test_partials(self):
        p = om.Problem()

        nn = 5

        p.model.add_subsystem("model", InitialLH2MassModification(num_nodes=nn), promotes=["*"])

        p.setup(force_alloc_complex=True)

        p.set_val("radius", 1.6, units="m")
        p.set_val("length", 0.3, units="m")

        # Set a specified change to liquid mass
        val = np.linspace(-10, 10, nn)
        p.set_val("delta_m_liq", val, units="kg")

        p.run_model()

        partials = p.check_partials(method="cs")
        assert_check_partials(partials)


if __name__ == "__main__":
    unittest.main()
