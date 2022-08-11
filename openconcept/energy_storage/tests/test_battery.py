import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.api import IndepVarComp, Group, Problem
from openconcept.energy_storage import SimpleBattery


class BatteryTestGroup(Group):
    """
    Test the battery component
    """

    def initialize(self):
        self.options.declare("vec_size", default=1, desc="Number of mission analysis points to run")
        self.options.declare("efficiency", default=1.0, desc="Efficiency (dimensionless)")
        self.options.declare("p", default=5000.0, desc="Battery specific power (W/kg)")
        self.options.declare("e", default=300.0, desc="Battery spec energy CAREFUL: (Wh/kg)")
        self.options.declare("cost_inc", default=50.0, desc="$ cost per kg")
        self.options.declare("cost_base", default=1.0, desc="$ cost base")
        self.options.declare("use_defaults", default=True)

    def setup(self):
        use_defaults = self.options["use_defaults"]
        nn = self.options["vec_size"]
        if not use_defaults:
            eta_b = self.options["efficiency"]
            p = self.options["p"]
            e = self.options["e"]
            ci = self.options["cost_inc"]
            cb = self.options["cost_base"]
            self.add_subsystem(
                "battery",
                SimpleBattery(
                    num_nodes=nn, efficiency=eta_b, specific_power=p, specific_energy=e, cost_inc=ci, cost_base=cb
                ),
            )
        else:
            self.add_subsystem("battery", SimpleBattery(num_nodes=nn))

        iv = self.add_subsystem("iv", IndepVarComp())
        iv.add_output("battery_weight", val=100, units="kg")
        iv.add_output("elec_load", val=np.ones(nn) * 100, units="kW")
        self.connect("iv.battery_weight", "battery.battery_weight")
        self.connect("iv.elec_load", "battery.elec_load")


class SimpleBatteryTestCase(unittest.TestCase):
    def test_default_settings(self):
        prob = Problem(BatteryTestGroup(vec_size=10, use_defaults=True))
        prob.setup(check=True, force_alloc_complex=True)
        prob.run_model()
        assert_near_equal(prob["battery.heat_out"], np.ones(10) * 100 * 0.0, tolerance=1e-15)
        assert_near_equal(prob["battery.component_sizing_margin"], np.ones(10) * 0.20, tolerance=1e-15)
        assert_near_equal(prob["battery.component_cost"], 5001, tolerance=1e-15)
        assert_near_equal(prob.get_val("battery.max_energy", units="W*h"), 300 * 100, tolerance=1e-15)

        partials = prob.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)

    def test_nondefault_settings(self):
        prob = Problem(
            BatteryTestGroup(vec_size=10, use_defaults=False, efficiency=0.95, p=3000, e=500, cost_inc=100, cost_base=0)
        )
        prob.setup(check=True, force_alloc_complex=True)
        prob.run_model()
        assert_near_equal(prob.get_val("battery.heat_out", units="kW"), np.ones(10) * 100 * 0.05, tolerance=1e-15)
        assert_near_equal(prob["battery.component_sizing_margin"], np.ones(10) / 3, tolerance=1e-15)
        assert_near_equal(prob["battery.component_cost"], 10000, tolerance=1e-15)
        assert_near_equal(prob.get_val("battery.max_energy", units="W*h"), 500 * 100, tolerance=1e-15)

        partials = prob.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)
