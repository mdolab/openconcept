import numpy as np
from openmdao.api import ExplicitComponent
from openmdao.api import Group


class SimpleMotor(ExplicitComponent):

    """Inputs: throttle, elec_power_rating
    Outputs: shaft_power_out, heat_out, elec_load, component_cost, component_weight, component_sizing_margin
    Metadata: efficiency, weight_inc, weight_base, cost_inc, cost_base

    Weights in kg/W, cost in USD/W
    """
    def initialize(self):
        #define technology factors
        self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')

        self.options.declare('efficiency', default=1., desc='Efficiency (dimensionless)')
        self.options.declare('weight_inc', default=1./5000, desc='kg/W')  # 5kW/kg motors have been demoed
        self.options.declare('weight_base', default=0., desc='kg base weight')
        self.options.declare('cost_inc', default=100.0/745.0, desc='$ cost per watt')
        self.options.declare('cost_base', default=1., desc= '$ cost base')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('throttle', desc='Throttle input (Fractional)',shape=(nn,))
        self.add_input('elec_power_rating', units='W', desc='Rated electrical power (load)')

        #outputs and partials
        eta_m = self.options['efficiency']
        weight_inc = self.options['weight_inc']
        weight_base = self.options['weight_base']
        cost_inc = self.options['cost_inc']
        cost_base = self.options['cost_base']

        self.add_output('shaft_power_out', units='W', desc='Output shaft power',shape=(nn,))
        self.add_output('heat_out', units='W', desc='Waste heat out',shape=(nn,))
        self.add_output('elec_load',units='W', desc='Electrical load consumed',shape=(nn,))
        self.add_output('component_cost', units='USD', desc='Motor component cost')
        self.add_output('component_weight', units='kg', desc='Motor component weight')
        self.add_output('component_sizing_margin', desc='Fraction of rated power',shape=(nn,))
        self.declare_partials(["*"],["*"],dependent=False)
        self.declare_partials('shaft_power_out','elec_power_rating')
        self.declare_partials('shaft_power_out','throttle','elec_power_rating',rows=range(nn),cols=range(nn))
        self.declare_partials('heat_out','elec_power_rating')
        self.declare_partials('heat_out', 'throttle','elec_power_rating',rows=range(nn),cols=range(nn))
        self.declare_partials('elec_load','elec_power_rating')
        self.declare_partials('elec_load','throttle',rows=range(nn),cols=range(nn))
        self.declare_partials('component_cost','elec_power_rating', val=cost_inc)
        self.declare_partials('component_weight','elec_power_rating', val=weight_inc)
        self.declare_partials('component_sizing_margin','throttle',val=1.0*np.ones(nn),rows=range(nn),cols=range(nn))


            
    def compute(self, inputs, outputs):
        eta_m = self.options['efficiency']
        weight_inc = self.options['weight_inc']
        weight_base = self.options['weight_base']
        cost_inc = self.options['cost_inc']
        cost_base = self.options['cost_base']
        #print('Throttle: '+ str(inputs['throttle']))

        outputs['shaft_power_out'] = inputs['throttle']*inputs['elec_power_rating'] * eta_m
        outputs['heat_out'] = inputs['throttle']*inputs['elec_power_rating'] * (1 - eta_m)
        outputs['elec_load'] = inputs['throttle']*inputs['elec_power_rating']
        outputs['component_cost'] = inputs['elec_power_rating'] * cost_inc + cost_base
        outputs['component_weight'] = inputs['elec_power_rating'] * weight_inc + weight_base
        outputs['component_sizing_margin'] = inputs['throttle']
        #print('Shaft power out: '+ str(outputs['shaft_power_out']))
    def compute_partials(self, inputs, J):
        eta_m = self.options['efficiency']
        J['shaft_power_out','throttle'] = inputs['elec_power_rating'] * eta_m
        J['shaft_power_out','elec_power_rating'] = inputs['throttle'] * eta_m
        J['heat_out','throttle'] = inputs['elec_power_rating'] * (1 - eta_m)
        J['heat_out','elec_power_rating'] = inputs['throttle'] * (1 - eta_m)   
        J['elec_load','throttle'] = inputs['elec_power_rating']
        J['elec_load','elec_power_rating'] = inputs['throttle']  


if __name__ == "__main__":
    from openmdao.api import IndepVarComp, Problem
    prob = Problem()
    prob.model = Group()
    prob.model.add_subsystem('throttle',IndepVarComp('throttle',val=1.1))
    prob.model.add_subsystem('P_rated',IndepVarComp('P_m',val=200.,units='kW'))
    prob.model.add_subsystem('motor',SimpleMotor(efficiency=0.98,weight_inc=0.2,weight_base=20,cost_inc=0.05,cost_base=10000.))
    prob.model.connect('throttle.throttle','motor.throttle')
    prob.model.connect('P_rated.P_m','motor.elec_power_rating')
    prob.setup()
    prob.run_model()
    print(prob['motor.elec_load'])
    print(prob['motor.shaft_power_out'])
    #data = prob.check_partials()

