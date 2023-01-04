import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
import openmdao.api as om
from openconcept.thermal import SimpleHose


class TestHose(unittest.TestCase):
    """
    Test the coolant hose component
    """

    def generate_model(self, nn):
        prob = om.Problem()

        hose_diam = 0.02
        hose_length = 16.0
        hose_design_pressure = 1e6
        mdot_coolant = np.linspace(0.6, 1.2, nn)
        rho_coolant = 1020 * np.ones((nn,))
        mu_coolant = 1.68e-3
        sigma = 2.07e6
        rho_hose = 1356.3

        ivc = prob.model.add_subsystem("ivc", om.IndepVarComp(), promotes_outputs=["*"])
        ivc.add_output("hose_diameter", val=hose_diam, units="m")
        ivc.add_output("hose_length", val=hose_length, units="m")
        ivc.add_output("hose_design_pressure", val=hose_design_pressure, units="Pa")
        ivc.add_output("mdot_coolant", val=mdot_coolant, units="kg/s")
        ivc.add_output("rho_coolant", val=rho_coolant, units="kg/m**3")
        ivc.add_output("mu_coolant", val=mu_coolant, units="kg/m/s")
        prob.model.add_subsystem("hose", SimpleHose(num_nodes=nn), promotes_inputs=["*"])
        prob.setup(check=True, force_alloc_complex=True)

        xs_area = np.pi * (hose_diam / 2) ** 2
        U = mdot_coolant / rho_coolant / xs_area
        Redh = rho_coolant * U * hose_diam / mu_coolant
        f = 0.3164 * Redh ** (-1 / 4)
        dp = f * rho_coolant / 2 * hose_length * U**2 / hose_diam

        wall_thickness = hose_design_pressure * (hose_diam / 2) / sigma
        hose_weight = wall_thickness * np.pi * (hose_diam + wall_thickness) * rho_hose * hose_length
        fluid_weight = xs_area * rho_coolant[0] * hose_length
        return prob, dp, (hose_weight + fluid_weight)

    def test_scalar(self):
        prob, dp, weight = self.generate_model(nn=1)
        prob.run_model()
        assert_near_equal(prob.get_val("hose.delta_p", units="Pa"), dp, tolerance=1e-10)
        assert_near_equal(prob.get_val("hose.component_weight", units="kg"), weight, tolerance=1e-10)
        partials = prob.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)

    def test_vector(self):
        prob, dp, weight = self.generate_model(nn=11)
        prob.run_model()
        assert_near_equal(prob.get_val("hose.delta_p", units="Pa"), dp, tolerance=1e-10)
        assert_near_equal(prob.get_val("hose.component_weight", units="kg"), weight, tolerance=1e-10)
        partials = prob.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)


if __name__ == "__main__":
    unittest.main()
