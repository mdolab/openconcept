import numpy as np
from openmdao.api import ExplicitComponent
from openmdao.api import Group


class SimpleBattery(ExplicitComponent):
    """Inputs: battery_weight, elec_load, SOC
    Outputs: heat_out, component_cost, component_sizing_margin
    Metadata: efficiency, specific_power, specific_energy, cost_inc, cost_base

    spec power in W/kg
    spec energy in Wh/kg (CAREFUL! not J)
    cost in $/kWhr
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')
        #define technology factors
        self.options.declare('efficiency', default=1., desc='Efficiency (dimensionless)')
        self.options.declare('specific_power', default=5000., desc='Battery specific power (W/kg)' )
        self.options.declare('specific_energy', default=300., desc='Battery spec energy CAREFUL: (Wh/kg)')
        self.options.declare('cost_inc', default=50., desc='$ cost per kg')
        self.options.declare('cost_base', default=1., desc= '$ cost base')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('battery_weight', units='kg', desc='Total battery pack weight')
        self.add_input('elec_load', units='W', desc='Electrical load drawn',shape=(nn,))

        #outputs and partials
        eta_b = self.options['efficiency']
        e_b = self.options['specific_energy']
        p_b = self.options['specific_power']
        cost_inc = self.options['cost_inc']
        cost_base = self.options['cost_base']

        self.add_output('heat_out', units='W', desc='Waste heat out',shape=(nn,))
        self.add_output('component_cost', units='USD', desc='Battery cost')
        self.add_output('component_sizing_margin', desc='Load fraction of capable power',shape=(nn,))
        self.add_output('max_energy',units='W*h')

        self.declare_partials('heat_out', 'elec_load', val=(1-eta_b)*np.ones(nn),rows=range(nn),cols=range(nn))
        self.declare_partials('component_cost','battery_weight', val=cost_inc)
        self.declare_partials('component_sizing_margin','battery_weight')
        self.declare_partials('component_sizing_margin','elec_load',rows=range(nn),cols=range(nn))
        self.declare_partials('max_energy','battery_weight',val=e_b)

    def compute(self, inputs, outputs):
        eta_b = self.options['efficiency']
        p_b = self.options['specific_power']
        e_b = self.options['specific_energy']
        cost_inc = self.options['cost_inc']
        cost_base = self.options['cost_base']

        outputs['heat_out'] = inputs['elec_load'] * (1 - eta_b)
        outputs['component_cost'] = inputs['battery_weight'] * cost_inc + cost_base
        outputs['component_sizing_margin'] = inputs['elec_load'] / (p_b * inputs['battery_weight'])
        outputs['max_energy'] = inputs['battery_weight'] * e_b

    def compute_partials(self, inputs, J):
        eta_b = self.options['efficiency']
        p_b = self.options['specific_power']
        J['component_sizing_margin','elec_load'] = 1 / (p_b * inputs['battery_weight'])
        J['component_sizing_margin','battery_weight'] = - inputs['elec_load'] / (p_b * inputs['battery_weight'] ** 2)
