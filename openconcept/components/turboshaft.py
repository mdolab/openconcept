import numpy as np
from openmdao.api import ExplicitComponent
from openmdao.api import Group


class SimpleTurboshaft(ExplicitComponent):

    """Inputs: throttle, shaft_power_rating
    Outputs: shaft_power_out, fuel_flow, component_cost, component_weight, component_sizing_margin
    Metadata: psfc, weight_inc, weight_base, cost_inc, cost_base

    Weights in kg/W, cost in $/W, psfc in kgfuel/s/Wsh

    """

    def initialize(self):
        #define technology factors
        #psfc conversion from g/kW/hr to kg/W/s = 2.777e-10
        #psfc conversion from lbfuel/hp/hr to kg/W/s = 1.690e-7
        #set to modern turboprop default
        self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')
        self.options.declare('psfc', default=0.6*1.69e-7, desc='power specific fuel consumption (kg fuel per second per shaft Watt')
        self.options.declare('weight_inc', default=0., desc='kg per watt')
        self.options.declare('weight_base', default=0., desc='kg base weight')
        self.options.declare('cost_inc', default=1.04, desc='$ cost per watt')
        self.options.declare('cost_base', default=0., desc= '$ cost base')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('throttle', desc='Throttle input (Fractional)',shape=(nn,))
        self.add_input('shaft_power_rating', units='W', desc='Rated shaft power')

        #outputs and partials
        psfc = self.options['psfc']
        weight_inc = self.options['weight_inc']
        weight_base = self.options['weight_base']
        cost_inc = self.options['cost_inc']
        cost_base = self.options['cost_base']

        self.add_output('shaft_power_out', units='W', desc='Output shaft power',shape=(nn,))
        self.add_output('fuel_flow', units='kg/s', desc='Fuel flow in (kg fuel / s)',shape=(nn,))
        self.add_output('component_cost', units='USD', desc='Motor component cost')
        self.add_output('component_weight', units='kg', desc='Motor component weight')
        self.add_output('component_sizing_margin', desc='Fraction of rated power',shape=(nn,))

        self.declare_partials('shaft_power_out','shaft_power_rating')
        self.declare_partials('shaft_power_out','throttle',rows=range(nn),cols=range(nn))

        self.declare_partials('fuel_flow','shaft_power_rating')
        self.declare_partials('fuel_flow','throttle',rows=range(nn),cols=range(nn))

        self.declare_partials('component_cost','shaft_power_rating', val=cost_inc)
        self.declare_partials('component_weight','shaft_power_rating', val=weight_inc)
        self.declare_partials('component_sizing_margin','throttle',val=1.0*np.ones(nn),rows=range(nn),cols=range(nn))


            
    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        psfc = self.options['psfc']
        weight_inc = self.options['weight_inc']
        weight_base = self.options['weight_base']
        cost_inc = self.options['cost_inc']
        cost_base = self.options['cost_base']

        a = inputs['throttle']
        b = inputs['shaft_power_rating']
        c = a*b
        outputs['shaft_power_out'] = inputs['throttle']*inputs['shaft_power_rating']
        outputs['fuel_flow'] = -inputs['throttle']*inputs['shaft_power_rating']*psfc
        outputs['component_cost'] = inputs['shaft_power_rating'] * cost_inc + cost_base
        outputs['component_weight'] = inputs['shaft_power_rating'] * weight_inc + weight_base
        outputs['component_sizing_margin'] = inputs['throttle']
        #print('Rating:'+str(inputs['shaft_power_rating']))
        #print('Eng sizing:'+str(outputs['component_sizing_margin']))
        
    def compute_partials(self, inputs, J):
        nn = self.options['num_nodes']
        psfc = self.options['psfc']
        J['shaft_power_out','throttle'] = inputs['shaft_power_rating'] * np.ones(nn)
        J['shaft_power_out','shaft_power_rating'] = inputs['throttle']
        J['fuel_flow','throttle'] = -inputs['shaft_power_rating'] * psfc * np.ones(nn)
        J['fuel_flow','shaft_power_rating'] = -inputs['throttle'] * psfc  



if __name__ == "__main__":
    from openmdao.api import IndepVarComp, Problem
    prob = Problem()
    prob.model = Group()
    prob.model.add_subsystem('throttle',IndepVarComp('throttle',val=1.1))
    prob.model.add_subsystem('P_rated',IndepVarComp('P_m',val=200.,units='kW'))
    prob.model.add_subsystem('turboshaft',SimpleTurboshaft(psfc=0.05,weight_inc=0.2,weight_base=20,cost_inc=0.05,cost_base=10000.))
    prob.model.connect('throttle.throttle','turboshaft.throttle')
    prob.model.connect('P_rated.P_m','turboshaft.shaft_power_rating')
    prob.setup()
    prob.run_model()
    print(prob['turboshaft.fuel_flow'])
    print(prob['turboshaft.shaft_power_out'])
    #data = prob.check_partials()

