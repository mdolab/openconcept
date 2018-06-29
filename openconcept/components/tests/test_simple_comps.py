import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_rel_error, assert_check_partials
from openmdao.api import IndepVarComp, Group, Problem
from openconcept.components import SimpleBattery, SimpleGearbox, SimpleGenerator, SimpleMotor, SimplePropeller, SimpleTurboshaft, PowerSplit

class BatteryTestGroup(Group):
    """
    Test the battery component
    """
    def initialize(self):
        self.options.declare('vec_size',default=1,desc="Number of mission analysis points to run")
        self.options.declare('batt_efficiency', default=1., desc='Efficiency (dimensionless)')
        self.options.declare('batt_p', default=5000., desc='Battery specific power (W/kg)' )
        self.options.declare('batt_e', default=300., desc='Battery spec energy CAREFUL: (Wh/kg)')
        self.options.declare('batt_cost_inc', default=50., desc='$ cost per kg')
        self.options.declare('batt_cost_base', default=1., desc= '$ cost base')
        self.options.declare('use_defaults', default=True)

    def setup(self):
        use_defaults = self.options['use_defaults']
        nn = self.options['vec_size']
        if not use_defaults:
            eta_b = self.options['batt_efficiency']
            p = self.options['batt_p']
            e = self.options['batt_e']
            ci = self.options['batt_cost_inc']
            cb = self.options['batt_cost_base']
            self.add_subsystem('battery', SimpleBattery(num_nodes=nn,
                                                        efficiency=eta_b,
                                                        specific_power=p,
                                                        specific_energy=e,
                                                        cost_inc=ci,
                                                        cost_base=cb))
        else:
            self.add_subsystem('battery', SimpleBattery(num_nodes=nn))

        iv = self.add_subsystem('iv', IndepVarComp())
        iv.add_output('battery_weight', val=100, units='kg')
        iv.add_output('elec_load', val=np.ones(nn) * 100, units='kW')
        self.connect('iv.battery_weight','battery.battery_weight')
        self.connect('iv.elec_load','battery.elec_load')

class SimpleBatteryTestCase(unittest.TestCase):

    def test_defaults(self):
        prob = Problem(BatteryTestGroup(vec_size=10, use_defaults=True))
        prob.setup(check=True,force_alloc_complex=True)
        prob.run_model()
        assert_rel_error(self, prob['battery.heat_out'], np.ones(10)*100*0.0, tolerance=1e-15)
        assert_rel_error(self, prob['battery.component_sizing_margin'], np.ones(10)*0.20, tolerance=1e-15)
        assert_rel_error(self, prob['battery.component_cost'], 5001, tolerance=1e-15)

        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)
