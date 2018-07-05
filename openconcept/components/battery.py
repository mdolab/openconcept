from __future__ import division
import numpy as np
from openmdao.api import ExplicitComponent
from openmdao.api import Group


class SimpleBattery(ExplicitComponent):
    """
    A simple battery which tracks power limits and generates heat.

    Specific energy assumption INCLUDING internal losses should be used
    The efficiency parameter only generates heat

    Inputs
    ------
    battery_weight : float
        Weight of the battery pack (scalar, kg)
    elec_load: float
        Electric power draw upstream (vector, W)

    Outputs
    -------
    max_energy : float
        Total energy in the battery at 100% SOC (scalar, Wh)
    heat_out : float
        Waste heat produced (vector, W)
    component_cost : float
        Nonrecurring cost of the component (scalar, USD)
    component_sizing_margin : float
        Equal to 1 when producing full rated power (vector, dimensionless)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length; default 1)
    efficiency : float
        Shaft power efficiency. Sensible range 0.0 to 1.0 (default 1.0)
    specific_power : float
        Rated power per unit weight (default 5000, W/kg)
    specific_energy : float
        Battery energy per unit weight **NOTE UNITS** (default 300, !!!! Wh/kg)
    cost_inc : float
        Cost per unit weight (default 50, USD/kg)
    cost_base : float
        Base cost (default 1 USD)
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')
        self.options.declare('efficiency', default=1., desc='Efficiency (dimensionless)')
        self.options.declare('specific_power', default=5000., desc='Battery specific power (W/kg)')
        self.options.declare('specific_energy', default=300., desc='Battery spec energy')
        self.options.declare('cost_inc', default=50., desc='$ cost per kg')
        self.options.declare('cost_base', default=1., desc='$ cost base')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('battery_weight', units='kg', desc='Total battery pack weight')
        self.add_input('elec_load', units='W', desc='Electrical load drawn', shape=(nn,))

        eta_b = self.options['efficiency']
        e_b = self.options['specific_energy']
        p_b = self.options['specific_power']
        cost_inc = self.options['cost_inc']
        cost_base = self.options['cost_base']

        self.add_output('heat_out', units='W', desc='Waste heat out', shape=(nn,))
        self.add_output('component_cost', units='USD', desc='Battery cost')
        self.add_output('component_sizing_margin',
                        desc='Load fraction of capable power', shape=(nn,))
        self.add_output('max_energy', units='W*h')

        self.declare_partials('heat_out', 'elec_load', val=(1 - eta_b) * np.ones(nn),
                              rows=range(nn), cols=range(nn))
        self.declare_partials('component_cost', 'battery_weight', val=cost_inc)
        self.declare_partials('component_sizing_margin', 'battery_weight')
        self.declare_partials('component_sizing_margin', 'elec_load',
                              rows=range(nn), cols=range(nn))
        self.declare_partials('max_energy', 'battery_weight', val=e_b)

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
        J['component_sizing_margin', 'elec_load'] = 1 / (p_b * inputs['battery_weight'])
        J['component_sizing_margin', 'battery_weight'] = - (inputs['elec_load'] /
                                                            (p_b * inputs['battery_weight'] ** 2))
