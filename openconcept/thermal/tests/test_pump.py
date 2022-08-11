import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
import openmdao.api as om
from openconcept.thermal import SimplePump


class TestPump(unittest.TestCase):
    """
    Test the coolant pump component
    """

    def generate_model(self, nn):
        prob = om.Problem()

        efficiency = 0.35
        spec_power = 1 / 450
        rho_coolant = 1020 * np.ones(nn)
        mdot_coolant = np.linspace(0.6, 1.2, nn)
        delta_p = np.linspace(2e4, 4e4, nn)
        power_rating = 1000
        ivc = prob.model.add_subsystem("ivc", om.IndepVarComp(), promotes_outputs=["*"])
        ivc.add_output("power_rating", val=power_rating, units="W")
        ivc.add_output("delta_p", val=delta_p, units="Pa")
        ivc.add_output("mdot_coolant", val=mdot_coolant, units="kg/s")
        ivc.add_output("rho_coolant", val=rho_coolant, units="kg/m**3")
        prob.model.add_subsystem("pump", SimplePump(num_nodes=nn), promotes_inputs=["*"])
        prob.setup(check=True, force_alloc_complex=True)

        fluid_power = (mdot_coolant / rho_coolant) * delta_p
        weight = power_rating * spec_power
        elec_load = fluid_power / efficiency
        margin = elec_load / power_rating

        return prob, elec_load, weight, margin

    def test_scalar(self):
        prob, elec_load, weight, margin = self.generate_model(nn=1)
        prob.run_model()
        assert_near_equal(prob.get_val("pump.elec_load", units="W"), elec_load, tolerance=1e-10)
        assert_near_equal(prob.get_val("pump.component_weight", units="kg"), weight, tolerance=1e-10)
        assert_near_equal(prob.get_val("pump.component_sizing_margin", units=None), margin, tolerance=1e-10)
        partials = prob.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)

    def test_vector(self):
        prob, elec_load, weight, margin = self.generate_model(nn=11)
        prob.run_model()
        assert_near_equal(prob.get_val("pump.elec_load", units="W"), elec_load, tolerance=1e-10)
        assert_near_equal(prob.get_val("pump.component_weight", units="kg"), weight, tolerance=1e-10)
        assert_near_equal(prob.get_val("pump.component_sizing_margin", units=None), margin, tolerance=1e-10)
        partials = prob.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)


if __name__ == "__main__":
    unittest.main()
