from __future__ import division
from openconcept.components.turboshaft import SimpleTurboshaft
from openconcept.components.propeller import SimplePropeller
from openconcept.utilities.dvlabel import DVLabel
from openconcept.utilities.math import AddSubtractComp
from openmdao.api import Group, IndepVarComp, ExplicitComponent

class TurbopropPropulsionSystem(Group):
    """This is an example model of the simplest possible propulsion system
        consisting of a constant-speed prop and a turboshaft.

        This is the Pratt and Whitney Canada PT6A-66D with 4-bladed
        propeller used by the SOCATA-DAHER TBM-850.

        Inputs
        ------
        ac|propulsion|engine|rating : float
            The maximum rated shaft power of the engine
        ac|propulsion|propeller|diameter : float
            Diameter of the propeller

        Options
        -------
        num_nodes : float
            Number of analysis points to run (default 1)
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc="Number of mission analysis points to run")

    def setup(self):
        nn = self.options['num_nodes']

        # rename incoming design variables
        dvlist = [['ac|propulsion|engine|rating', 'eng1_rating', 850, 'hp'],
                  ['ac|propulsion|propeller|diameter', 'prop1_diameter', 2.3, 'm']]
        self.add_subsystem('dvs', DVLabel(dvlist),
                           promotes_inputs=["*"], promotes_outputs=["*"])

        # introduce model components
        self.add_subsystem('eng1',
                           SimpleTurboshaft(num_nodes=nn, weight_inc=0.14 / 1000, weight_base=104),
                           promotes_inputs=["throttle"], promotes_outputs=["fuel_flow"])
        self.add_subsystem('prop1',
                           SimplePropeller(num_nodes=nn, num_blades=4,
                                           design_J=2.2, design_cp=0.55),
                           promotes_inputs=["fltcond|*"], promotes_outputs=["thrust"])

        # connect design variables to model component inputs
        self.connect('eng1_rating', 'eng1.shaft_power_rating')
        self.connect('eng1_rating', 'prop1.power_rating')
        self.connect('prop1_diameter', 'prop1.diameter')

        # connect components to each other
        self.connect('eng1.shaft_power_out', 'prop1.shaft_power_in')


class TwinTurbopropPropulsionSystem(Group):
    """This is an example model multiple constant-speed props and turboshafts.
        These are two P&W Canada PT6A-135A with 4-bladed Hartzell propellers used by the Beechcraft King Air C90GT
        https://www.easa.europa.eu/sites/default/files/dfu/TCDS_EASA-IM-A-503_C90-Series%20issue%206.pdf
        INPUTS: ac|propulsion|engine|rating - the maximum rated shaft power of the engine (each engine)
            dv_prop1_diameter - propeller diameter
    """
    def initialize(self):
        self.options.declare('num_nodes',default=1,desc="Number of mission analysis points to run")

    def setup(self):
        #define design variables that are independent of flight condition or control states
        dvlist = [['ac|propulsion|engine|rating','eng_rating',750,'hp'],
                    ['ac|propulsion|propeller|diameter','prop_diameter',2.28,'m'],
                    ]
        self.add_subsystem('dvs',DVLabel(dvlist),promotes_inputs=["*"],promotes_outputs=["*"])
        nn = self.options['num_nodes']
        #introduce model components
        self.add_subsystem('eng1',SimpleTurboshaft(num_nodes=nn,weight_inc=0.14/1000,weight_base=104))
        self.add_subsystem('prop1',SimplePropeller(num_nodes=nn,num_blades=4,design_J=2.2,design_cp=0.55),promotes_inputs=["fltcond|*"])
        self.add_subsystem('eng2',SimpleTurboshaft(num_nodes=nn,weight_inc=0.14/1000,weight_base=104))
        self.add_subsystem('prop2',SimplePropeller(num_nodes=nn,num_blades=4,design_J=2.2,design_cp=0.55),promotes_inputs=["fltcond|*"])

        #connect design variables to model component inputs
        self.connect('eng_rating','eng1.shaft_power_rating')
        self.connect('eng_rating','eng2.shaft_power_rating')
        self.connect('eng_rating','prop1.power_rating')
        self.connect('eng_rating','prop2.power_rating')
        self.connect('prop_diameter','prop1.diameter')
        self.connect('prop_diameter','prop2.diameter')


        #connect components to each other
        self.connect('eng1.shaft_power_out','prop1.shaft_power_in')
        self.connect('eng2.shaft_power_out','prop2.shaft_power_in')

        #add up the weights, thrusts and fuel flows
        add1 = AddSubtractComp(output_name='fuel_flow',input_names=['eng1_fuel_flow','eng2_fuel_flow'],vec_size=nn, units='kg/s')
        add1.add_equation(output_name='thrust',input_names=['prop1_thrust','prop2_thrust'],vec_size=nn, units='N')
        add1.add_equation(output_name='engines_weight',input_names=['eng1_weight','eng2_weight'], units='kg')
        add1.add_equation(output_name='propellers_weight',input_names=['prop1_weight','prop2_weight'], units='kg')
        self.add_subsystem('adder',subsys=add1,promotes_inputs=["*"],promotes_outputs=["*"])
        self.connect('prop1.thrust','prop1_thrust')
        self.connect('prop2.thrust','prop2_thrust')
        self.connect('eng1.fuel_flow','eng1_fuel_flow')
        self.connect('eng2.fuel_flow','eng2_fuel_flow')
        self.connect('prop1.component_weight','prop1_weight')
        self.connect('prop2.component_weight','prop2_weight')
        self.connect('eng1.component_weight','eng1_weight')
        self.connect('eng2.component_weight','eng2_weight')