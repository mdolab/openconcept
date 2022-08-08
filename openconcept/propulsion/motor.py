from __future__ import division
import numpy as np
from openmdao.api import ExplicitComponent, Group, BalanceComp
from openconcept.utilities import Integrator
from openconcept.thermal import MotorCoolingJacket


class LiquidCooledMotor(Group):
    """A component (heat producing) with thermal mass
    cooled by a cold plate.

    Inputs
    ------
    q_in : float
        Heat produced by the operating component (vector, W)
    mdot_coolant : float
        Coolant mass flow rate (vector, kg/s)
    T_in : float
        Instantaneous coolant inflow temperature (vector, K)
    motor_weight : float
        Object mass (only required in thermal mass mode) (scalar, kg)
    T_initial : float
        Initial temperature of the cold plate (only required in thermal mass mode) / object (scalar, K)
    duration : float
        Duration of mission segment, only required in unsteady mode
    power_rating : float
        Rated power of the motor (scalar, kW)

    Outputs
    -------
    T_out : float
        Instantaneous coolant outlet temperature (vector, K)
    T: float
        Windings temperature (vector, K)

    Options
    -------
    motor_specific_heat : float
        Specific heat capacity of the object in J / kg / K (default 921 = aluminum)
    coolant_specific_heat : float
        Specific heat capacity of the coolant in J / kg / K (default 3801, glycol/water)
    num_nodes : int
        Number of analysis points to run
    quasi_steady : bool
        Whether or not to treat the component as having thermal mass
    case_cooling_coefficient : float
        Watts of heat transfer per square meter of case surface area per K
        temperature differential (default 1100 W/m^2/K)
    """

    def initialize(self):
        self.options.declare('motor_specific_heat', default=921.0, desc='Specific heat in J/kg/K')
        self.options.declare('coolant_specific_heat', default=3801, desc='Specific heat in J/kg/K')
        self.options.declare('quasi_steady', default=False, desc='Treat the component as quasi-steady or with thermal mass')
        self.options.declare('num_nodes', default=1, desc='Number of quasi-steady points to runs')
        self.options.declare('case_cooling_coefficient', default=1100.)

    def setup(self):
        nn = self.options['num_nodes']
        quasi_steady = self.options['quasi_steady']
        self.add_subsystem('hex',
                           MotorCoolingJacket(num_nodes=nn, coolant_specific_heat=self.options['coolant_specific_heat'],
                                              motor_specific_heat=self.options['motor_specific_heat'],
                                              case_cooling_coefficient=self.options['case_cooling_coefficient']),
                           promotes_inputs=['q_in','T_in', 'T','power_rating','mdot_coolant','motor_weight'],
                           promotes_outputs=['T_out', 'dTdt'])
        if not quasi_steady:
            ode_integ = self.add_subsystem('ode_integ', Integrator(num_nodes=nn, diff_units='s', method='simpson', time_setup='duration'),
                                           promotes_outputs=['*'], promotes_inputs=['*'])
            ode_integ.add_integrand('T', rate_name='dTdt', units='K', lower=1e-10)
        else:
            self.add_subsystem('thermal_bal',
                               BalanceComp('T', eq_units='K/s', lhs_name='dTdt', rhs_val=0.0, units='K', lower=1.0, val=299.*np.ones((nn,))),
                               promotes_inputs=['dTdt'],
                               promotes_outputs=['T'])


class SimpleMotor(ExplicitComponent):
    """
    A simple motor which creates shaft power and draws electrical load.

    Inputs
    ------
    throttle : float
        Power control setting. Should be [0, 1]. (vector, dimensionless)
    elec_power_rating: float
        Electric (not mech) design power. (scalar, W)

    Outputs
    -------
    shaft_power_out : float
        Shaft power output from motor (vector, W)
    elec_load : float
        Electrical load consumed by motor (vector, W)
    heat_out : float
        Waste heat produced (vector, W)
    component_cost : float
        Nonrecurring cost of the component (scalar, USD)
    component_weight : float
        Weight of the component (scalar, kg)
    component_sizing_margin : float
        Equal to 1 when producing full rated power (vector, dimensionless)


    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length; default 1)
    efficiency : float
        Shaft power efficiency. Sensible range 0.0 to 1.0 (default 1)
    weight_inc : float
        Weight per unit rated power (default 1/5000, kg/W)
    weight_base : float
        Base weight (default 0, kg)
    cost_inc : float
        Cost per unit rated power (default 0.134228, USD/W)
    cost_base : float
        Base cost (default 1 USD) B
    """

    def initialize(self):
        # define technology factors
        self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')
        self.options.declare('efficiency', default=1., desc='Efficiency (dimensionless)')
        self.options.declare('weight_inc', default=1 / 5000, desc='kg/W')  # 5kW/kg
        self.options.declare('weight_base', default=0., desc='kg base weight')
        self.options.declare('cost_inc', default=100 / 745, desc='$ cost per watt')
        self.options.declare('cost_base', default=1., desc='$ cost base')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('throttle', desc='Throttle input (Fractional)', shape=(nn,))
        self.add_input('elec_power_rating', units='W', desc='Rated electrical power (load)')

        # outputs and partials
        eta_m = self.options['efficiency']
        weight_inc = self.options['weight_inc']
        weight_base = self.options['weight_base']
        cost_inc = self.options['cost_inc']
        cost_base = self.options['cost_base']

        self.add_output('shaft_power_out', units='W', desc='Output shaft power', shape=(nn,))
        self.add_output('heat_out', units='W', desc='Waste heat out', shape=(nn,))
        self.add_output('elec_load', units='W', desc='Electrical load consumed', shape=(nn,))
        self.add_output('component_cost', units='USD', desc='Motor component cost')
        self.add_output('component_weight', units='kg', desc='Motor component weight')
        self.add_output('component_sizing_margin', desc='Fraction of rated power', shape=(nn,))
        self.declare_partials('shaft_power_out', 'elec_power_rating')
        self.declare_partials('shaft_power_out', 'throttle',
                              rows=range(nn), cols=range(nn))
        self.declare_partials('heat_out', 'elec_power_rating')
        self.declare_partials('heat_out', 'throttle', 'elec_power_rating',
                              rows=range(nn), cols=range(nn))
        self.declare_partials('elec_load', 'elec_power_rating')
        self.declare_partials('elec_load', 'throttle', rows=range(nn), cols=range(nn))
        self.declare_partials('component_cost', 'elec_power_rating', val=cost_inc)
        self.declare_partials('component_weight', 'elec_power_rating', val=weight_inc)
        self.declare_partials('component_sizing_margin', 'throttle',
                              val=1.0 * np.ones(nn), rows=range(nn), cols=range(nn))

    def compute(self, inputs, outputs):
        eta_m = self.options['efficiency']
        weight_inc = self.options['weight_inc']
        weight_base = self.options['weight_base']
        cost_inc = self.options['cost_inc']
        cost_base = self.options['cost_base']
        outputs['shaft_power_out'] = inputs['throttle'] * inputs['elec_power_rating'] * eta_m
        outputs['heat_out'] = inputs['throttle'] * inputs['elec_power_rating'] * (1 - eta_m)
        outputs['elec_load'] = inputs['throttle'] * inputs['elec_power_rating']
        outputs['component_cost'] = inputs['elec_power_rating'] * cost_inc + cost_base
        outputs['component_weight'] = inputs['elec_power_rating'] * weight_inc + weight_base
        outputs['component_sizing_margin'] = inputs['throttle']

    def compute_partials(self, inputs, J):
        eta_m = self.options['efficiency']
        J['shaft_power_out', 'throttle'] = inputs['elec_power_rating'] * eta_m
        J['shaft_power_out', 'elec_power_rating'] = inputs['throttle'] * eta_m
        J['heat_out', 'throttle'] = inputs['elec_power_rating'] * (1 - eta_m)
        J['heat_out', 'elec_power_rating'] = inputs['throttle'] * (1 - eta_m)
        J['elec_load', 'throttle'] = inputs['elec_power_rating']
        J['elec_load', 'elec_power_rating'] = inputs['throttle']
