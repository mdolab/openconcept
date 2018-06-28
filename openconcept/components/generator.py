import numpy as np
from openmdao.api import ExplicitComponent
from openmdao.api import Group


class SimpleGenerator(ExplicitComponent):

    """Inputs: shaft_power_in, elec_power_rating
    Outputs: elec_power_out, heat_out, component_cost, component_weight, component_sizing_margin
    Metadata: efficiency, weight_inc, weight_base, cost_inc, cost_base

    Weights in kg/w, cost in $/W
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')

        #define technology factors
        self.options.declare('efficiency', default=1., desc='Efficiency (dimensionless)')
        self.options.declare('weight_inc', default=1./5000, desc='kg/W')
        self.options.declare('weight_base', default=0., desc='kg base weight')
        self.options.declare('cost_inc', default=100.0/745.0, desc='$ cost per watt')
        self.options.declare('cost_base', default=1., desc= '$ cost base')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('shaft_power_in', units='W', desc='Input shaft power',shape=(nn,))
        self.add_input('elec_power_rating', units='W', desc='Rated output power')

        #outputs and partials
        eta_g = self.options['efficiency']
        weight_inc = self.options['weight_inc']
        weight_base = self.options['weight_base']
        cost_inc = self.options['cost_inc']
        cost_base = self.options['cost_base']

        self.add_output('elec_power_out', units='W', desc='Output electric power',shape=(nn,))
        self.add_output('heat_out', units='W', desc='Waste heat out',shape=(nn,))
        self.add_output('component_cost', units='USD', desc='Generator component cost')
        self.add_output('component_weight', units='kg', desc='Generator component weight')
        self.add_output('component_sizing_margin', desc='Fraction of rated power',shape=(nn,))

        self.declare_partials('elec_power_out','shaft_power_in', val=eta_g*np.ones(nn),rows=range(nn),cols=range(nn))
        self.declare_partials('heat_out', 'shaft_power_in', val=(1-eta_g)*np.ones(nn),rows=range(nn),cols=range(nn))
        self.declare_partials('component_cost','elec_power_rating', val=cost_inc)
        self.declare_partials('component_weight','elec_power_rating', val=weight_inc)
        self.declare_partials('component_sizing_margin','shaft_power_in',rows=range(nn),cols=range(nn))
        self.declare_partials('component_sizing_margin','elec_power_rating')



            
    def compute(self, inputs, outputs):
        eta_g = self.options['efficiency']
        weight_inc = self.options['weight_inc']
        weight_base = self.options['weight_base']
        cost_inc = self.options['cost_inc']
        cost_base = self.options['cost_base']

        outputs['elec_power_out'] = inputs['shaft_power_in'] * eta_g
        outputs['heat_out'] = inputs['shaft_power_in'] * (1 - eta_g)
        outputs['component_cost'] = inputs['elec_power_rating'] * cost_inc + cost_base
        outputs['component_weight'] = inputs['elec_power_rating'] * weight_inc + weight_base
        outputs['component_sizing_margin'] = inputs['shaft_power_in'] * eta_g / inputs['elec_power_rating']
        
    def compute_partials(self, inputs, J):
        eta_g = self.options['efficiency']
        J['component_sizing_margin','shaft_power_in'] = eta_g / inputs['elec_power_rating']
        J['component_sizing_margin','elec_power_rating'] = - eta_g * inputs['shaft_power_in'] / inputs['elec_power_rating'] ** 2


if __name__ == "__main__":
    from openmdao.api import IndepVarComp, Problem
    prob = Problem()
    prob.model = Group()
    prob.model.add_subsystem('P_in',IndepVarComp('P_i',val=100.,units='kW'))
    prob.model.add_subsystem('P_rated',IndepVarComp('P_r',val=150.,units='kW'))
    prob.model.add_subsystem('gen',SimpleGenerator(efficiency=0.98,weight_inc=0.2,weight_base=20,cost_inc=0.05,cost_base=10000.))
    prob.model.connect('P_in.P_i','gen.shaft_power_in')
    prob.model.connect('P_rated.P_r','gen.elec_power_rating')
    prob.setup()
    prob.run_model()
    print(prob['gen.component_cost'])
    print(prob['gen.elec_power_out'])
    data = prob.check_partials()

