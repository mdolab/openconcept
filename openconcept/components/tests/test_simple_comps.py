from __future__ import division
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_rel_error, assert_check_partials
from openmdao.api import IndepVarComp, Group, Problem
from openconcept.components import SimpleBattery, SimpleGenerator, SimpleMotor, SimplePropeller, SimpleTurboshaft, PowerSplit

class BatteryTestGroup(Group):
    """
    Test the battery component
    """
    def initialize(self):
        self.options.declare('vec_size',default=1,desc="Number of mission analysis points to run")
        self.options.declare('efficiency', default=1., desc='Efficiency (dimensionless)')
        self.options.declare('p', default=5000., desc='Battery specific power (W/kg)' )
        self.options.declare('e', default=300., desc='Battery spec energy CAREFUL: (Wh/kg)')
        self.options.declare('cost_inc', default=50., desc='$ cost per kg')
        self.options.declare('cost_base', default=1., desc= '$ cost base')
        self.options.declare('use_defaults', default=True)

    def setup(self):
        use_defaults = self.options['use_defaults']
        nn = self.options['vec_size']
        if not use_defaults:
            eta_b = self.options['efficiency']
            p = self.options['p']
            e = self.options['e']
            ci = self.options['cost_inc']
            cb = self.options['cost_base']
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

class MotorTestGroup(Group):
    """
    Test the motor component
    """
    def initialize(self):
        self.options.declare('vec_size',default=1,desc="Number of mission analysis points to run")
        self.options.declare('efficiency', default=1., desc='Efficiency (dimensionless)')
        self.options.declare('weight_inc', default=1./5000, desc='kg/W')  # 5kW/kg motors have been demoed
        self.options.declare('weight_base', default=0., desc='kg base weight')
        self.options.declare('cost_inc', default=100.0/745.0, desc='$ cost per watt')
        self.options.declare('cost_base', default=1., desc= '$ cost base')
        self.options.declare('use_defaults', default=True)

    def setup(self):
        use_defaults = self.options['use_defaults']
        nn = self.options['vec_size']
        if not use_defaults:
            eta_b = self.options['efficiency']
            wi = self.options['weight_inc']
            wb = self.options['weight_base']
            ci = self.options['cost_inc']
            cb = self.options['cost_base']
            self.add_subsystem('motor', SimpleMotor(num_nodes=nn,
                                                    efficiency=eta_b,
                                                    weight_inc=wi,
                                                    weight_base=wb,
                                                    cost_inc=ci,
                                                    cost_base=cb))
        else:
            self.add_subsystem('motor', SimpleMotor(num_nodes=nn))

        iv = self.add_subsystem('iv', IndepVarComp())
        iv.add_output('elec_power_rating', val=100, units='kW')
        iv.add_output('throttle', val=np.ones(nn)*0.9)
        self.connect('iv.elec_power_rating','motor.elec_power_rating')
        self.connect('iv.throttle','motor.throttle')

class GeneratorTestGroup(Group):
    """
    Test the generator component
    """
    def initialize(self):
        self.options.declare('vec_size',default=1,desc="Number of mission analysis points to run")
        self.options.declare('efficiency', default=1., desc='Efficiency (dimensionless)')
        self.options.declare('weight_inc', default=1./5000, desc='kg/W')
        self.options.declare('weight_base', default=0., desc='kg base weight')
        self.options.declare('cost_inc', default=100.0/745.0, desc='$ cost per watt')
        self.options.declare('cost_base', default=1., desc= '$ cost base')
        self.options.declare('use_defaults', default=True)

    def setup(self):
        use_defaults = self.options['use_defaults']
        nn = self.options['vec_size']
        if not use_defaults:
            eta_b = self.options['efficiency']
            wi = self.options['weight_inc']
            wb = self.options['weight_base']
            ci = self.options['cost_inc']
            cb = self.options['cost_base']
            self.add_subsystem('generator', SimpleGenerator(num_nodes=nn,
                                                    efficiency=eta_b,
                                                    weight_inc=wi,
                                                    weight_base=wb,
                                                    cost_inc=ci,
                                                    cost_base=cb))
        else:
            self.add_subsystem('generator', SimpleGenerator(num_nodes=nn))

        iv = self.add_subsystem('iv', IndepVarComp())
        iv.add_output('elec_power_rating', val=100, units='kW')
        iv.add_output('shaft_power_in', val=np.ones(nn)*90, units='kW')
        self.connect('iv.elec_power_rating','generator.elec_power_rating')
        self.connect('iv.shaft_power_in','generator.shaft_power_in')

class TurboshaftTestGroup(Group):
    """
    Test the turboshaft component
    """
    def initialize(self):
        self.options.declare('vec_size',default=1,desc="Number of mission analysis points to run")
        self.options.declare('psfc', default=0.6 * 1.68965774e-7, desc='power specific fuel consumption')
        self.options.declare('weight_inc', default=0., desc='kg per watt')
        self.options.declare('weight_base', default=0., desc='kg base weight')
        self.options.declare('cost_inc', default=1.04, desc='$ cost per watt')
        self.options.declare('cost_base', default=0., desc='$ cost base')
        self.options.declare('use_defaults', default=True)

    def setup(self):
        use_defaults = self.options['use_defaults']
        nn = self.options['vec_size']
        if not use_defaults:
            psfc = self.options['psfc']
            wi = self.options['weight_inc']
            wb = self.options['weight_base']
            ci = self.options['cost_inc']
            cb = self.options['cost_base']
            self.add_subsystem('turboshaft', SimpleTurboshaft(num_nodes=nn,
                                                             psfc=psfc,
                                                             weight_inc=wi,
                                                             weight_base=wb,
                                                             cost_inc=ci,
                                                             cost_base=cb))
        else:
            self.add_subsystem('turboshaft', SimpleTurboshaft(num_nodes=nn))

        iv = self.add_subsystem('iv', IndepVarComp())
        iv.add_output('shaft_power_rating', val=1000, units='hp')
        iv.add_output('throttle', val=np.ones(nn)*0.90)
        self.connect('iv.shaft_power_rating','turboshaft.shaft_power_rating')
        self.connect('iv.throttle','turboshaft.throttle')

class SimpleBatteryTestCase(unittest.TestCase):

    def test_default_settings(self):
        prob = Problem(BatteryTestGroup(vec_size=10, use_defaults=True))
        prob.setup(check=True,force_alloc_complex=True)
        prob.run_model()
        assert_rel_error(self, prob['battery.heat_out'], np.ones(10)*100*0.0, tolerance=1e-15)
        assert_rel_error(self, prob['battery.component_sizing_margin'], np.ones(10)*0.20, tolerance=1e-15)
        assert_rel_error(self, prob['battery.component_cost'], 5001, tolerance=1e-15)
        assert_rel_error(self, prob.get_val('battery.max_energy', units='W*h'), 300*100, tolerance=1e-15)

        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)

    def test_nondefault_settings(self):
        prob = Problem(BatteryTestGroup(vec_size=10,
                                        use_defaults=False,
                                        efficiency=0.95,
                                        p=3000,
                                        e=500,
                                        cost_inc=100,
                                        cost_base=0))
        prob.setup(check=True,force_alloc_complex=True)
        prob.run_model()
        assert_rel_error(self, prob.get_val('battery.heat_out', units='kW'), np.ones(10)*100*0.05, tolerance=1e-15)
        assert_rel_error(self, prob['battery.component_sizing_margin'], np.ones(10)/3, tolerance=1e-15)
        assert_rel_error(self, prob['battery.component_cost'], 10000, tolerance=1e-15)
        assert_rel_error(self, prob.get_val('battery.max_energy', units='W*h'), 500*100, tolerance=1e-15)

        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)

class SimpleMotorTestCase(unittest.TestCase):

    def test_default_settings(self):
        prob = Problem(MotorTestGroup(vec_size=10,
                                      use_defaults=True))
        prob.setup(check=True,force_alloc_complex=True)
        prob.run_model()
        assert_rel_error(self, prob.get_val('motor.shaft_power_out', units='kW'), np.ones(10)*90, tolerance=1e-15)
        assert_rel_error(self, prob.get_val('motor.elec_load', units='kW'), np.ones(10)*90, tolerance=1e-15)
        assert_rel_error(self, prob.get_val('motor.heat_out', units='kW'), np.ones(10)*0.0, tolerance=1e-15)
        assert_rel_error(self, prob.get_val('motor.component_sizing_margin'), np.ones(10)*0.90, tolerance=1e-15)
        assert_rel_error(self, prob.get_val('motor.component_cost', units='USD'), 13423.818791946309, tolerance=1e-10)
        assert_rel_error(self, prob.get_val('motor.component_weight', units='kg'), 20, tolerance=1e-15)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)

    def test_nondefault_settings(self):
        prob = Problem(MotorTestGroup(vec_size=10,
                                      use_defaults=False,
                                      efficiency=0.95,
                                      weight_inc=1/3000,
                                      weight_base=2,
                                      cost_inc=1/500,
                                      cost_base=3,
                                      ))
        prob.setup(check=True,force_alloc_complex=True)
        prob.run_model()
        assert_rel_error(self, prob.get_val('motor.shaft_power_out', units='kW'), np.ones(10)*90*0.95, tolerance=1e-15)
        assert_rel_error(self, prob.get_val('motor.elec_load', units='kW'), np.ones(10)*90, tolerance=1e-15)
        assert_rel_error(self, prob.get_val('motor.heat_out', units='kW'), np.ones(10)*90*0.05, tolerance=1e-15)
        assert_rel_error(self, prob.get_val('motor.component_sizing_margin'), np.ones(10)*0.90, tolerance=1e-15)
        assert_rel_error(self, prob.get_val('motor.component_cost', units='USD'), 203, tolerance=1e-15)
        assert_rel_error(self, prob.get_val('motor.component_weight', units='kg'), 35.333333333333333333, tolerance=1e-10)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)

class SimpleGeneratorTestCase(unittest.TestCase):

    def test_default_settings(self):
        prob = Problem(GeneratorTestGroup(vec_size=10,
                                      use_defaults=True))
        prob.setup(check=True,force_alloc_complex=True)
        prob.run_model()
        assert_rel_error(self, prob.get_val('generator.elec_power_out', units='kW'), np.ones(10)*90, tolerance=1e-15)
        assert_rel_error(self, prob.get_val('generator.heat_out', units='kW'), np.ones(10)*0.0, tolerance=1e-15)
        assert_rel_error(self, prob.get_val('generator.component_sizing_margin'), np.ones(10)*0.90, tolerance=1e-15)
        assert_rel_error(self, prob.get_val('generator.component_cost', units='USD'), 13423.818791946309, tolerance=1e-10)
        assert_rel_error(self, prob.get_val('generator.component_weight', units='kg'), 20, tolerance=1e-15)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)

    def test_nondefault_settings(self):
        prob = Problem(GeneratorTestGroup(vec_size=10,
                                      use_defaults=False,
                                      efficiency=0.95,
                                      weight_inc=1/3000,
                                      weight_base=2,
                                      cost_inc=1/500,
                                      cost_base=3,
                                      ))
        prob.setup(check=True,force_alloc_complex=True)
        prob.run_model()
        assert_rel_error(self, prob.get_val('generator.elec_power_out', units='kW'), np.ones(10)*90*0.95, tolerance=1e-15)
        assert_rel_error(self, prob.get_val('generator.heat_out', units='kW'), np.ones(10)*90*0.05, tolerance=1e-15)
        assert_rel_error(self, prob.get_val('generator.component_sizing_margin'), np.ones(10)*0.90*0.95, tolerance=1e-15)
        assert_rel_error(self, prob.get_val('generator.component_cost', units='USD'), 203, tolerance=1e-15)
        assert_rel_error(self, prob.get_val('generator.component_weight', units='kg'), 35.333333333333333333, tolerance=1e-10)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)

class SimpleTurboshaftTestCase(unittest.TestCase):

    def test_default_settings(self):
        prob = Problem(TurboshaftTestGroup(vec_size=10,
                                      use_defaults=True))
        prob.setup(check=True,force_alloc_complex=True)
        prob.run_model()
        assert_rel_error(self, prob.get_val('turboshaft.shaft_power_out', units='hp'), np.ones(10)*1000*0.9, tolerance=1e-6)
        assert_rel_error(self, prob.get_val('turboshaft.fuel_flow', units='lbm/h'), np.ones(10)*0.6*1000*0.9, tolerance=1e-6)
        assert_rel_error(self, prob.get_val('turboshaft.component_sizing_margin'), np.ones(10)*0.90, tolerance=1e-15)
        assert_rel_error(self, prob.get_val('turboshaft.component_cost', units='USD'), 775528., tolerance=1e-7)
        assert_rel_error(self, prob.get_val('turboshaft.component_weight', units='kg'), 0, tolerance=1e-15)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)

    def test_nondefault_settings(self):
        prob = Problem(TurboshaftTestGroup(vec_size=10,
                                      use_defaults=False,
                                      psfc=0.4*1.68965774e-7,
                                      weight_inc=1/745.7, # 1 kg/hp
                                      weight_base=50,
                                      cost_inc=1000/745.7, # 1000 / hp
                                      cost_base=50000))
        prob.setup(check=True,force_alloc_complex=True)
        prob.run_model()
        assert_rel_error(self, prob.get_val('turboshaft.shaft_power_out', units='hp'), np.ones(10)*1000*0.9, tolerance=1e-6)
        assert_rel_error(self, prob.get_val('turboshaft.fuel_flow', units='lbm/h'), np.ones(10)*0.4*1000*0.9, tolerance=1e-6)
        assert_rel_error(self, prob.get_val('turboshaft.component_sizing_margin'), np.ones(10)*0.90, tolerance=1e-15)
        assert_rel_error(self, prob.get_val('turboshaft.component_cost', units='USD'), 1e6+50000., tolerance=1e-6)
        assert_rel_error(self, prob.get_val('turboshaft.component_weight', units='kg'), 1050, tolerance=1e-10)
        partials = prob.check_partials(method='cs',compact_print=True)
        assert_check_partials(partials)