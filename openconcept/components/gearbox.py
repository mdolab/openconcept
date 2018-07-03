import numpy as np
from openmdao.api import ExplicitComponent
from openmdao.api import Group


class SimpleGearbox(ExplicitComponent):
    """Inputs: shaft_power_in, shaft_power_rating
    Outputs: shaft_power_out, heat_out, component_cost, component_weight, component_sizing_margin
    Metadata: efficiency, weight_inc, weight_base, cost_inc, cost_base

    Weights in kg/W, cost in $/W
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')
        #define technology factors
        self.options.declare('efficiency', default=1., desc='Efficiency (dimensionless)')
        self.options.declare('weight_inc', default=0., desc='kg per W')
        self.options.declare('weight_base', default=0., desc='kg base weight')
        self.options.declare('cost_inc', default=0., desc='$ cost per watt')
        self.options.declare('cost_base', default=0., desc= '$ cost base')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('shaft_power_in', units='W', desc='Input shaft power',shape=(nn,))
        self.add_input('shaft_power_rating', units='W', desc='Rated shaft power')

        #outputs and partials
        eta_gb = self.options['efficiency']
        weight_inc = self.options['weight_inc']
        weight_base = self.options['weight_base']
        cost_inc = self.options['cost_inc']
        cost_base = self.options['cost_base']

        self.add_output('shaft_power_out', units='W', desc='Output shaft power',shape=(nn,))
        self.add_output('heat_out', units='W', desc='Waste heat out',shape=(nn,))
        self.add_output('component_cost', units='USD', desc='Gearbox component cost',shape=(nn,))
        self.add_output('component_weight', units='kg', desc='Gearbox component weight')
        self.add_output('component_sizing_margin', desc='Fraction of rated power',shape=(nn,))

        self.declare_partials('shaft_power_out','shaft_power_in', val=eta_gb*np.ones(nn), rows=range(nn), cols=range(nn))
        self.declare_partials('heat_out', 'shaft_power_in', val=(1-eta_gb)*np.ones(nn), rows=range(nn), cols=range(nn))
        self.declare_partials('component_cost','shaft_power_rating', val=cost_inc)
        self.declare_partials('component_weight','shaft_power_rating', val=weight_inc)
        self.declare_partials('component_sizing_margin','shaft_power_in', rows=range(nn), cols=range(nn))
        self.declare_partials('component_sizing_margin','shaft_power_rating')




    def compute(self, inputs, outputs):
        eta_gb = self.options['efficiency']
        weight_inc = self.options['weight_inc']
        weight_base = self.options['weight_base']
        cost_inc = self.options['cost_inc']
        cost_base = self.options['cost_base']

        outputs['shaft_power_out'] = inputs['shaft_power_in'] * eta_gb
        outputs['heat_out'] = inputs['shaft_power_in'] * (1 - eta_gb)
        outputs['component_cost'] = inputs['shaft_power_rating'] * cost_inc + cost_base
        outputs['component_weight'] = inputs['shaft_power_rating'] * weight_inc + weight_base
        outputs['component_sizing_margin'] = inputs['shaft_power_in'] / inputs['shaft_power_rating']

    def compute_partials(self, inputs, J):
        J['component_sizing_margin','shaft_power_in'] = 1 / inputs['shaft_power_rating']
        J['component_sizing_margin','shaft_power_rating'] = - inputs['shaft_power_in'] / inputs['shaft_power_rating'] ** 2