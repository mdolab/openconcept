from __future__ import division
from openmdao.api import Problem, Group, IndepVarComp, BalanceComp, NewtonSolver, DirectSolver, BoundsEnforceLS
from openmdao.api import ScipyOptimizeDriver, ExplicitComponent, ImplicitComponent, ExecComp

import numpy as np
import scipy.sparse as sp
import sys, os
sys.path.insert(0,os.getcwd())
from openconcept.components.ducts import ImplicitCompressibleDuct
from openconcept.components.motor import SimpleMotor
from openconcept.utilities.selector import SelectorComp
from openconcept.utilities.math.integrals import Integrator
from openconcept.utilities.math.derivatives import FirstDerivative
from openconcept.utilities.math import AddSubtractComp, ElementMultiplyDivideComp, VectorConcatenateComp, VectorSplitComp
from openconcept.analysis.atmospherics.compute_atmos_props import ComputeAtmosphericProperties

# Define sigmoid function and its derivative for later use
def sigmoid(x):
    """
    Logistic sigmoid function f(x) = 1/(1 + exp(-x))

    Inputs
    ------
    x : float
        Input value, any shape
    
    Outputs
    -------
    f : float
        Sigmoid function evaluated at x, same shape as x
    """
    f = 1 / (1 + np.exp(-x))
    return f

def sigmoid_deriv(x):
    """
    Derivative of logistic sigmoid function w.r.t. x

    Inputs
    ------
    x : float
        Input value, any shape
    
    Outputs
    -------
    df : float
        Sigmoid derivitive w.r.t. x evaluated at x, same shape as x
    """
    df = 1 / (np.exp(-x) + 2 + 1/np.exp(-x))
    return df


"""Analysis routines for simulating thermal management of aircraft components"""

class SimpleEngine(ExplicitComponent):
    """
    Convert heat to work based on an assumed fraction of the
    Carnot efficiency. 

    Inputs
    ------
    T_h : float
        Temperature of the hot heat input (vector, K)
    T_c : float
        Temperature of the cold heat input (vector, K)
    Wdot : float
        Work generation rate (vector, W)
    eff_factor : float
        Percentage of the Carnot efficiency (scalar, dimensionless)

    Outputs
    -------
    q_h : float
        Heat extracted from the hot side (vector, W)
    q_c : float
        Waste heat sent to cold side (vector, W)
    eta_thermal : float
        Overall thermal efficiency (vector, dimensionless)

    Options
    -------
    num_nodes : int
        The number of analysis points to run
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1)

    def setup(self):
        nn_tot = self.options['num_nodes']
        arange = np.arange(0, nn_tot)

        self.add_input('T_h', units='K', shape=(nn_tot,), val=600.)
        self.add_input('T_c', units='K', shape=(nn_tot,), val=400.)
        self.add_input('Wdot', units='W', shape=(nn_tot,), val=1000.)
        self.add_input('eff_factor', units=None, val=0.4)

        self.add_output('q_h', units='W', shape=(nn_tot,))
        self.add_output('q_c', units='W', shape=(nn_tot,))
        self.add_output('eta_thermal', units=None, shape=(nn_tot,))

        self.declare_partials(['q_h'], ['T_h', 'T_c', 'Wdot'], rows=arange, cols=arange)
        self.declare_partials(['q_c'], ['T_h', 'T_c', 'Wdot'], rows=arange, cols=arange)
        self.declare_partials(['eta_thermal'], ['T_h', 'T_c'], rows=arange, cols=arange)

        self.declare_partials(['q_h'], ['eff_factor'], rows=arange, cols=np.zeros((nn_tot,)))
        self.declare_partials(['q_c'], ['eff_factor'], rows=arange, cols=np.zeros((nn_tot,)))
        self.declare_partials(['eta_thermal'], ['eff_factor'], rows=arange, cols=np.zeros((nn_tot,)))

    def compute(self, inputs, outputs):
        # compute carnot efficiency
        # 1 - Tc/Th
        eta_carnot = 1 - inputs['T_c'] / inputs['T_h']
        eta_thermal = inputs['eff_factor'] * eta_carnot
        outputs['eta_thermal'] = eta_thermal
        # compute the heats
        outputs['q_h'] = inputs['Wdot'] / eta_thermal
        outputs['q_c'] = inputs['Wdot'] / eta_thermal - inputs['Wdot'] 

    def compute_partials(self, inputs, J):
        eta_carnot = 1 - inputs['T_c'] / inputs['T_h']
        eta_thermal = inputs['eff_factor'] * eta_carnot

        J['eta_thermal', 'T_h'] = inputs['eff_factor'] * inputs['T_c'] / inputs['T_h'] ** 2
        J['eta_thermal', 'T_c'] = inputs['eff_factor'] * (-1 / inputs['T_h'])
        J['eta_thermal', 'eff_factor'] = eta_carnot

        J['q_h', 'T_h'] = - inputs['Wdot'] / eta_thermal ** 2 * (inputs['eff_factor'] * inputs['T_c'] / inputs['T_h'] ** 2)
        J['q_h', 'T_c'] = - inputs['Wdot'] / eta_thermal ** 2 * (inputs['eff_factor'] * (-1 / inputs['T_h']))
        J['q_h', 'Wdot'] = 1 / eta_thermal
        J['q_h', 'eff_factor'] = - inputs['Wdot'] / eta_thermal ** 2 * (eta_carnot)

        J['q_c', 'T_h'] = - inputs['Wdot'] / eta_thermal ** 2 * (inputs['eff_factor'] * inputs['T_c'] / inputs['T_h'] ** 2)
        J['q_c', 'T_c'] = - inputs['Wdot'] / eta_thermal ** 2 * (inputs['eff_factor'] * (-1 / inputs['T_h']))
        J['q_c', 'Wdot'] = (1 / eta_thermal - 1)
        J['q_c', 'eff_factor'] = - inputs['Wdot'] / eta_thermal ** 2 * (eta_carnot)

class SimpleHeatPump(ExplicitComponent):
    """
    Pumps heat from cold source to hot sink with work input
    based on assumed fraction of Carnot efficiency.

    Inputs
    ------
    T_h : float
        Temperature of the hot heat input (vector, K)
    T_c : float
        Temperature of the cold heat input (vector, K)
    Wdot : float
        Work usage rate (vector, W)
    eff_factor : float
        Percentage of the Carnot efficiency (scalar, dimensionless)
    
    Outputs
    -------
    q_c : float
        Heat extracted from the cold side (vector, W)
    q_h : float
        Heat sent to hot side (vector, W)
    COP_cooling : float
        Cooling coefficient of performance, heat removed from cold side
        divided by work used (vector, dimensionless)

    Options
    -------
    num_nodes : int
        The number of analysis points to run
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of analysis points to run')
    
    def setup(self):
        nn_tot = self.options['num_nodes']
        arange = np.arange(0, nn_tot)

        self.add_input('T_h', units='K', shape=(nn_tot,), val=600.)
        self.add_input('T_c', units='K', shape=(nn_tot,), val=400.)
        self.add_input('Wdot', units='W', shape=(nn_tot,), val=1000.)
        self.add_input('eff_factor', units=None, val=0.4)

        self.add_output('q_c', units='W', shape=(nn_tot,))
        self.add_output('q_h', units='W', shape=(nn_tot,))
        self.add_output('COP_cooling', units=None, shape=(nn_tot,))

        self.declare_partials(['q_c'], ['T_h', 'T_c', 'Wdot'], rows=arange, cols=arange)
        self.declare_partials(['q_h'], ['T_h', 'T_c', 'Wdot'], rows=arange, cols=arange)
        self.declare_partials(['COP_cooling'], ['T_h', 'T_c'], rows=arange, cols=arange)

        self.declare_partials(['q_c'], ['eff_factor'], rows=arange, cols=np.zeros((nn_tot,)))
        self.declare_partials(['q_h'], ['eff_factor'], rows=arange, cols=np.zeros((nn_tot,)))
        self.declare_partials(['COP_cooling'], ['eff_factor'], rows=arange, cols=np.zeros((nn_tot,)))
    
    def compute(self, inputs, outputs):
        # Cooling coefficient of performance, use sigmoid to avoid negative COPs
        delta_T = inputs['T_h'] - inputs['T_c']
        shift = 2  # shift so sigmoid is very close to zero once delta_T < 0
        COP_cooling = sigmoid(delta_T - shift) * inputs['eff_factor'] * inputs['T_c'] / (delta_T)
        outputs['COP_cooling'] = COP_cooling

        # Heat transfer
        outputs['q_c'] = - COP_cooling * inputs['Wdot']
        outputs['q_h'] = (1 + COP_cooling) * inputs['Wdot']
    
    def compute_partials(self, inputs, J):
        # Assign inputs to variables for readability
        T_h = inputs['T_h']
        T_c = inputs['T_c']
        Wdot = inputs['Wdot']
        eff_factor = inputs['eff_factor']
        shift = 2

        J['COP_cooling', 'T_h'] = - sigmoid(T_h - T_c - shift) * eff_factor * T_c / (T_h - T_c) ** 2 + \
                                    sigmoid_deriv(T_h - T_c - shift) * eff_factor * T_c / (T_h - T_c)
        J['COP_cooling', 'T_c'] = sigmoid(T_h - T_c - shift) * eff_factor * T_h / (T_h - T_c) ** 2 - \
                                  sigmoid_deriv(T_h - T_c - shift) * eff_factor * T_c / (T_h - T_c)
        J['COP_cooling', 'eff_factor'] = sigmoid(T_h - T_c - shift) * T_c / (T_h - T_c)

        J['q_c', 'T_h'] = - J['COP_cooling', 'T_h'] * Wdot
        J['q_c', 'T_c'] = - J['COP_cooling', 'T_c'] * Wdot
        J['q_c', 'Wdot'] = - sigmoid(T_h - T_c - shift) * eff_factor * T_c / (T_h - T_c)
        J['q_c', 'eff_factor'] = - J['COP_cooling', 'eff_factor'] * Wdot

        J['q_h', 'T_h'] = J['COP_cooling', 'T_h'] * Wdot
        J['q_h', 'T_c'] = J['COP_cooling', 'T_c'] * Wdot
        J['q_h', 'Wdot'] = 1 + sigmoid(T_h - T_c - shift) * eff_factor * T_c / (T_h - T_c)
        J['q_h', 'eff_factor'] = J['COP_cooling', 'eff_factor'] * Wdot

class SimpleTMS(Group):
    """
    Models a thermal management system (TMS) cooling an electric motor
    with a heat pump (refrigerator). The motor (with thermal mass) is linked
    with a cold plate that exchanges heat with the heat pump. The heat pump's
    waste heat is sent to a constant temperature reservoir via a cold plate.

    Inputs
    ------
    throttle : float
        Motor power control setting. Should be [0, 1]. (vector, dimensionless)
    motor_elec_power_rating: float
        Motor electric (not mech) design power. (scalar, W)
    duration : float
        Duration of mission segment, only required in unsteady mode. (scalar, sec)
    channel_width_motor : float
        Width of coolant channels in motor's cold plate (scalar, m)
    channel_height_motor : float
        Height of coolant channels in motor's cold plate (scalar, m)
    channel_length_motor : float
        Length of coolant channels in motor's cold plate (scalar, m)
    n_parallel_motor : float
        Number of identical coolant channels in motor's cold plate (scalar, dimensionless)
    channel_width_refrig : float
        Width of coolant channels in refrigerator's cold plate (scalar, m)
    channel_height_refrig : float
        Height of coolant channels in refrigerator's cold plate (scalar, m)
    channel_length_refrig : float
        Length of coolant channels in refrigerator's cold plate (scalar, m)
    n_parallel_refrig : float
        Number of identical coolant channels in refrigerator's cold plate (scalar, dimensionless)
    Wdot : float
        Heat pump work usage rate (vector, W)
    eff_factor : float
        Heat pump's percentage of the Carnot efficiency (scalar, dimensionless)
    
    Outputs
    -------
    shaft_power_out : float
        Shaft power output from motor (vector, W)
    motor_elec_load : float
        Electrical load consumed by motor (vector, W)
    motor_cost : float
        Nonrecurring cost of the motor (scalar, USD)
    motor_weight : float
        Weight of the motor (scalar, kg)
    motor_sizing_margin : float
        Equal to 1 when producing full rated power (vector, dimensionless)
    motor_T : float
        Temperature of the motor (vector, K)
    motor_T_final : float
        Final temperature of the motor (scalar, K)
    mdot_coolant : float
        Coolant mass flow rate in cold plates (vector, kg/s)
    q_h : float
        Heat sent to hot side of refrigerator (vector, W)
    COP_cooling : float
        Cooling coefficient of performance of refrigerator, heat removed from motor
        divided by work used (vector, dimensionless)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length; default 1)
    motor_efficiency : float
        Motor shaft power efficiency. Sensible range 0.0 to 1.0 (default 0.93, same as magni500)
    motor_weight_inc : float
        Motor weight per unit rated power (default 2.411e-4, same as magni500, kg/W)
    motor_weight_base : float
        Motor base weight (default 0, kg)
    motor_cost_inc : float
        Motor cost per unit rated power (default 0.134228, USD/W)
    motor_cost_base : float
        Motor base cost (default 1 USD) B
    motor_specific_heat : float
        Specific heat capacity of the object in J / kg / K (default 921 = aluminum)
    motor_T_init : float
        Initial temperature of the motor in K (default 294 K = room temperature, 70 deg F)
    coolant_rho : float
        Density of coolant in cold plates in kg/m**3 (default 0.997, water)
    coolant_k : float
        Thermal conductivity of the coolant in cold plates (W/m/K) (default 0.405, glycol/water)
    coolant_nusselt : float
        Hydraulic diameter Nusselt number of the coolant in the cold plate's channels
        (default 7.54 for constant temperature infinite parallel plate)
    coolant_specific_heat : float
        Specific heat of the coolant in cold plates (J/kg/K) (default 3801, glycol/water)
    heat_sink_T : float
        Temperature of the heat sink where waste heat is dumped (K) (default 294 K = room temperature, 70 deg F)
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of analysis points to run')
        self.options.declare('motor_efficiency', default=0.93, desc='Motor efficiency (dimensionless)')
        self.options.declare('motor_weight_inc', default=135./560e3, desc='Motor kg/W')
        self.options.declare('motor_weight_base', default=0., desc='Motor base weight kg')
        self.options.declare('motor_cost_inc', default=100./745., desc='Motor cost per watt $/W')
        self.options.declare('motor_cost_base', default=1., desc='Motor cost base $')
        self.options.declare('motor_specific_heat', default=921., desc='Specific heat of motor in J/kg/K - default 921 for Al')
        self.options.declare('motor_T_init', default=294., desc='Initial motor temperature in K - default to room temp')
        self.options.declare('coolant_rho', default=997.0, desc='Fluid density in kg/m3')
        self.options.declare('coolant_k', default=0.405, desc='Thermal conductivity of the fluid in W / mK')
        self.options.declare('coolant_nusselt', default=7.54, desc='Hydraulic diameter Nusselt number')
        self.options.declare('coolant_specific_heat', default=3801, desc='Specific heat in J/kg/K')
        self.options.declare('heat_sink_T', default=294., desc='Heat sink temperature in K')
    
    def setup(self):
        nn = self.options['num_nodes']
        # Add the electric motor and its thermal mass component (with an integrator to track temperature)
        self.add_subsystem('motor', SimpleMotor(num_nodes=nn,
                                                efficiency=self.options['motor_efficiency'],
                                                weight_inc=self.options['motor_weight_inc'],
                                                weight_base=self.options['motor_weight_base'],
                                                cost_inc=self.options['motor_cost_inc'],
                                                cost_base=self.options['motor_cost_base']),
                            promotes_inputs=['throttle', ('elec_power_rating', 'motor_elec_power_rating')],
                            promotes_outputs=['shaft_power_out', ('elec_load', 'motor_elec_load'),
                                              ('component_cost', 'motor_cost'), ('component_weight', 'motor_weight'),
                                              ('component_sizing_margin', 'motor_sizing_margin')])
        
        self.add_subsystem('motor_thermal_mass',
                           ThermalComponentWithMass(num_nodes=nn,
                                                    specific_heat=self.options['motor_specific_heat']))

        ivc = self.add_subsystem('ivc', IndepVarComp(), promotes_outputs=['mdot_coolant'])
        ivc.add_output('motor_T_init', val=self.options['motor_T_init'], units='K')

        int_temp = self.add_subsystem('motor_temp_integrator', Integrator(num_nodes=nn,
                                                               diff_units='s',
                                                               method='simpson',
                                                               time_setup='duration'),
                            promotes_inputs=['duration'],
                            promotes_outputs=['motor_T', 'motor_T_final'])
        int_temp.add_integrand('motor_T', start_name='motor_T_init', end_name='motor_T_final',
                               rate_name='dTdt', val=1.0, units='K')
        
        # Add the cold plate to interface the motor and refrigerator
        self.add_subsystem('motor_cold_plate',
                           ConstantSurfaceTemperatureColdPlate_NTU(num_nodes=nn,
                                                                   fluid_rho=self.options['coolant_rho'],
                                                                   fluid_k=self.options['coolant_k'],
                                                                   nusselt=self.options['coolant_nusselt'],
                                                                   specific_heat=self.options['coolant_specific_heat']),
                            promotes_inputs=[('channel_width', 'channel_width_motor'), ('channel_height', 'channel_height_motor'),
                                             ('channel_length', 'channel_length_motor'), ('n_parallel', 'n_parallel_motor'), 'mdot_coolant'])
        
        self.add_subsystem('refrigerator_cold_plate',
                           ConstantSurfaceTemperatureColdPlate_NTU(num_nodes=nn,
                                                                   fluid_rho=self.options['coolant_rho'],
                                                                   fluid_k=self.options['coolant_k'],
                                                                   nusselt=self.options['coolant_nusselt'],
                                                                   specific_heat=self.options['coolant_specific_heat']),
                            promotes_inputs=[('channel_width', 'channel_width_refrig'), ('channel_height', 'channel_height_refrig'),
                                             ('channel_length', 'channel_length_refrig'), ('n_parallel', 'n_parallel_refrig'), 'mdot_coolant'])
        

        self.add_subsystem('refrigerator', SimpleHeatPump(num_nodes=nn), promotes_inputs=['Wdot', 'eff_factor'],
                           promotes_outputs=['q_h', 'COP_cooling'])
        
        # Connect the motor to its thermal mass and cold plate and connect the integrator to det T from dT/dt
        self.connect('motor_weight', 'motor_thermal_mass.mass')
        self.connect('motor.heat_out', 'motor_thermal_mass.q_in')
        self.connect('motor_cold_plate.q', 'motor_thermal_mass.q_out')
        self.connect('motor_T', 'motor_cold_plate.T_surface')
        self.connect('ivc.motor_T_init', 'motor_temp_integrator.motor_T_init')
        self.connect('motor_thermal_mass.dTdt', 'motor_temp_integrator.dTdt')

        # Connect the two cold plates to each other
        self.connect('motor_cold_plate.T_out', 'refrigerator_cold_plate.T_in')
        self.connect('refrigerator_cold_plate.T_out', 'motor_cold_plate.T_in')
        ivc.add_output('mdot_coolant', shape=(nn,), units='kg/s')

        # Use a BalanceComp to set the surface temperature of the cold plate such that the heat extracted from
        # the coolant equals the heat in on the cold side of the refrigerator
        self.add_subsystem('refrigerator_plate_bal',
                           BalanceComp(name='T_surface', units='K', eq_units='W', val=np.ones(nn),
                                       lhs_name='plate_q_out', rhs_name='refrigerator_q_c'))

        self.connect('refrigerator_cold_plate.q', 'refrigerator_plate_bal.plate_q_out')
        self.connect('refrigerator.q_c', 'refrigerator_plate_bal.refrigerator_q_c')
        self.connect('refrigerator_plate_bal.T_surface', 'refrigerator_cold_plate.T_surface')
        self.connect('refrigerator_plate_bal.T_surface', 'refrigerator.T_c')

        # Set the heat sink temperature to be constant and connect it to the heat pump
        ivc.add_output('heat_sink_T', val=self.options['heat_sink_T'], units='K', shape=(nn,))
        self.connect('ivc.heat_sink_T', 'refrigerator.T_h')

class PerfectHeatTransferComp(ExplicitComponent):
    """
    Models heat transfer to coolant loop assuming zero thermal resistance.

    Inputs
    ------
    T_in : float
        Incoming coolant temperature (vector, K)
    q : float
        Heat flow into fluid stream; positive is heat addition (vector, W)
    mdot_coolant : float
        Coolant mass flow (vector, kg/s)
    
    Outputs
    -------
    T_out : float
        Outgoing coolant temperature (vector, K)
    T_average : float
        Average coolant temperature (vector K)
    
    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, default 1)
    specific_heat : float
        Specific heat of the coolant (scalar, J/kg/K, default 3801 glycol/water)
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of analysis points')
        self.options.declare('specific_heat', default=3801., desc='Specific heat in J/kg/K')
    
    def setup(self):
        nn = self.options['num_nodes']
        arange = np.arange(0, nn)

        self.add_input('T_in', desc='Incoming coolant temp', units='K', shape=(nn,))
        self.add_input('q', desc='Heat INTO the fluid stream (positive is heat addition)', units='W', shape=(nn,))
        self.add_input('mdot_coolant', desc='Mass flow rate of coolant', units='kg/s', shape=(nn,))
        self.add_output('T_out', desc='Outgoing coolant temp', val=np.random.uniform(300, 330), units='K', shape=(nn,))
        self.add_output('T_average', desc='Average temp of fluid', val=np.random.uniform(300, 330), units='K', shape=(nn,))

        self.declare_partials(['T_out', 'T_average'], ['q', 'mdot_coolant'], rows=arange, cols=arange)
        self.declare_partials('T_out', 'T_in', rows=arange, cols=arange, val=np.ones((nn,)))
        self.declare_partials('T_average', 'T_in', rows=arange, cols=arange, val=np.ones((nn,)))
    
    def compute(self, inputs, outputs):
        outputs['T_out'] = inputs['T_in'] + inputs['q'] / self.options['specific_heat'] / inputs['mdot_coolant']
        outputs['T_average'] = (inputs['T_in'] + outputs['T_out']) / 2
    
    def compute_partials(self, inputs, J):
        J['T_out', 'q'] = 1 / self.options['specific_heat'] / inputs['mdot_coolant']
        J['T_out', 'mdot_coolant'] = - inputs['q'] / self.options['specific_heat'] / inputs['mdot_coolant']**2

        J['T_average', 'q'] = J['T_out', 'q'] / 2
        J['T_average', 'mdot_coolant'] = J['T_out', 'mdot_coolant'] / 2

class HeatPumpWithIntegratedCoolantLoop(Group):
    """
    SimpleHeatPump connected to two PerfectHeatTransferComp components on either side. This setup
    takes in cold and hot side temperature set points, along with other inputs, and solves for
    the remaining variables including heat pump work usage rate.
    
    One nuance of this group is the hot_side_balance_param output, which must be connected to
    an input that can modulate the hot side coolant temperature. The hot_side_balance_param
    is then adjusted to set the coolant temperature (T_out_hot) to the specified hot side
    temperature set point (T_h_set).

    Inputs
    ------
    T_in_hot : float
        Incoming coolant temperature on the hot side (vector, K)
    T_in_cold : float
        Incoming coolant temperature on the cold side (vector, K)
    mdot_coolant_cold : float
        Coolant mass flow rate on cold side (vector, kg/s)
    mdot_coolant_hot : float
        Coolant mass flow rate on hot side (vector, kg/s)
    T_h_set : float
        Heat pump hot side temperature set point (vector, K)
    T_c_set : float
        Heat pump cold side temperature set point (vector, K)
    eff_factor : float
        Heat pump percentage of Carnot efficiency (scalar, dimensionaless)
    bypass_heat_pump : int (either 1 or 0)
        If 1, heat pump is removed from coolant loop and coolant flows; if 0, heat pump
        is kept in the loop (vector, default all ones)

    Outputs
    -------
    T_out_hot : float
        Outgoing coolant temperature on the hot side (vector, K)
    T_out_cold : float
        Outgoing coolant temperature on the cold side (vector, K)
    Wdot : float
        Heat pump work usage rate (vector, W)
    hot_side_balance_param : float
        Parameter set (by BalanceComp) so that outgoing coolant on the
        hot side has temp of T_h_set, one common example would be to connect this
        to mdot_cold of a heat exchanger on the hot side to modulate the coolant
        temperature (vector, unit varies)

    Options
    -------
    num_nodes : int
        The number of analysis points to run
    specific_heat : float
        Specific heat of the coolant (scalar, J/kg/K, default 3801 glycol/water)
    hot_side_balance_param_units : string
        Units of parameter being used to modulate hot side coolant temp (default None)
    hot_side_balance_param_lower : float
        Lower bound on hot_side_balance_param in the BalanceComp (scalar, default -inf)
    hot_side_balance_param_upper : float
        Upper bound on hot_side_balance_param in the BalanceComp (scalar, default inf)
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of analysis points')
        self.options.declare('specific_heat', default=3801., desc='Specific heat in J/kg/K')
        self.options.declare('hot_side_balance_param_units', default=None,
                             desc='Units of input hot_side_balance_param is connected to in outer model')
        self.options.declare('hot_side_balance_param_lower', default=-np.inf, desc='Lower bound on hot_side_balance_param')
        self.options.declare('hot_side_balance_param_upper', default=np.inf, desc='Upper bound on hot_side_balance_param')
    
    def setup(self):
        nn = self.options['num_nodes']
        nn_ones = np.ones((nn,))
        spec_heat = self.options['specific_heat']
        
        self.add_subsystem('hot_side', PerfectHeatTransferComp(num_nodes=nn, specific_heat=spec_heat),
                           promotes_inputs=[('T_in', 'T_in_hot'), ('mdot_coolant', 'mdot_coolant_hot')])
        self.add_subsystem('cold_side', PerfectHeatTransferComp(num_nodes=nn, specific_heat=spec_heat),
                           promotes_inputs=[('T_in', 'T_in_cold'), ('mdot_coolant', 'mdot_coolant_cold')])
        self.add_subsystem('heat_pump', SimpleHeatPump(num_nodes=nn),
                           promotes_inputs=['eff_factor', ('T_c', 'T_c_set'), ('T_h', 'T_h_set')])

        # Set the work usage rate of the heat pump such that the cold side coolant outlet temperature
        # set point is maintained
        self.add_subsystem('cold_side_bal', BalanceComp('Wdot', eq_units='K', lhs_name='T_c', rhs_name='T_c_set',
                                                   units='kW', val=10.*nn_ones, lower=0.*nn_ones), 
                           promotes_inputs=['T_c_set'])
        self.connect('cold_side_bal.Wdot', 'heat_pump.Wdot')
        # Use a selector to prevent the BalanceComp from solving if bypass mode is switched on
        # by setting T_c in the BalanceComp to T_c_set so they'll match on the first iteration
        self.add_subsystem('cold_bal_selector', SelectorComp(num_nodes=nn, input_names=['T_out', 'Wdot'], units='K'),
                           promotes_inputs=[('selector', 'bypass_heat_pump')])
        self.connect('cold_side.T_out', 'cold_bal_selector.T_out')
        self.connect('cold_bal_selector.result', 'cold_side_bal.T_c')
        # Feed the BalanceComp outputs directly back into the lhs when bypass is on
        self.add_subsystem('bal_dummy', ExecComp('T_c = Wdot', T_c={'units':'K', 'shape':(nn,)}, 
                                                 Wdot={'units':'W', 'shape':(nn,)}))
        self.connect('cold_side_bal.Wdot', 'bal_dummy.Wdot')
        self.connect('bal_dummy.T_c', 'cold_bal_selector.Wdot')

        # Set the hot side balance parameter such that the hot side coolant temperature
        # set point is maintained
        self.add_subsystem('hot_side_bal', BalanceComp('hot_side_balance_param', eq_units='K', lhs_name='T_h',
                                                       rhs_name='T_h_set', val=nn_ones,
                                                       units=self.options['hot_side_balance_param_units'],
                                                       lower=self.options['hot_side_balance_param_lower']*nn_ones,
                                                       upper=self.options['hot_side_balance_param_upper']*nn_ones), 
                           promotes_inputs=['T_h_set'], promotes_outputs=['hot_side_balance_param'])
        # If bypass, use the hot side T_in instead of T_out
        self.add_subsystem('hot_bal_selector', SelectorComp(num_nodes=nn, input_names=['T_out_hot', 'T_in_hot'],
                                                            units='K'),
                           promotes_inputs=['T_in_hot', ('selector', 'bypass_heat_pump')])
        self.connect('hot_side.T_out', 'hot_bal_selector.T_out_hot')
        self.connect('hot_bal_selector.result', 'hot_side_bal.T_h')

        # Connect the heat transfers on either side of the heat pump
        self.connect('heat_pump.q_c', 'cold_side.q')
        self.connect('heat_pump.q_h', 'hot_side.q')

        # Use selectors to control the I/O routing to bypass the heat pump if specified
        # T_out_cold selector
        self.add_subsystem('cold_side_selector', SelectorComp(num_nodes=nn, input_names=['T_out', 'T_in_hot'], units='K'),
                           promotes_inputs=[('selector', 'bypass_heat_pump'), 'T_in_hot'],
                           promotes_outputs=[('result', 'T_out_cold')])
        self.connect('cold_side.T_out', 'cold_side_selector.T_out')
        # T_out_hot selector
        self.add_subsystem('hot_side_selector', SelectorComp(num_nodes=nn, input_names=['T_out', 'T_in_cold'], units='K'),
                           promotes_inputs=[('selector', 'bypass_heat_pump'), 'T_in_cold'],
                           promotes_outputs=[('result', 'T_out_hot')])
        self.connect('hot_side.T_out', 'hot_side_selector.T_out')
        # Wdot selector
        self.add_subsystem('Wdot_selector', SelectorComp(num_nodes=nn, input_names=['Wdot', 'zero'], units='W'),
                           promotes_inputs=[('selector', 'bypass_heat_pump')],
                           promotes_outputs=[('result', 'Wdot')])
        self.connect('cold_side_bal.Wdot', 'Wdot_selector.Wdot')

        # Set the default set points and T_in defaults for continuity
        self.set_input_defaults('Wdot_selector.zero', val=np.zeros((nn,)), units='W')
        self.set_input_defaults('T_c_set', val=300.*nn_ones, units='K')
        self.set_input_defaults('T_h_set', val=500.*nn_ones, units='K')
        self.set_input_defaults('T_in_hot', val=400.*nn_ones, units='K')
        self.set_input_defaults('T_in_cold', val=400.*nn_ones, units='K')

class ThermalComponentWithMass(ExplicitComponent):
    """
    Computes thermal residual of a component with heating, cooling, and thermal mass

    Inputs
    ------
    q_in : float
        Heat generated by the component (vector, W)
    q_out : float
        Heat to waste stream (vector, W)
    mass : float
        Thermal mass (scalar, kg)

    Outputs
    -------
    dTdt : float
        First derivative of temperature (vector, K/s)

    Options
    -------
    specific_heat : float
        Specific heat capacity of the object in J / kg / K (default 921 = aluminum)
    num_nodes : int
        The number of analysis points to run
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1)
        self.options.declare('specific_heat', default=921, desc='Specific heat in J/kg/K - default 921 for aluminum')

    def setup(self):
        nn_tot = self.options['num_nodes']
        arange = np.arange(0, nn_tot)

        self.add_input('q_in', units='W', shape=(nn_tot,))
        self.add_input('q_out', units='W', shape=(nn_tot,))
        self.add_input('mass', units='kg')
        self.add_output('dTdt', units='K/s', shape=(nn_tot,))

        self.declare_partials(['dTdt'], ['q_in'], rows=arange, cols=arange)
        self.declare_partials(['dTdt'], ['q_out'], rows=arange, cols=arange)
        self.declare_partials(['dTdt'], ['mass'], rows=arange, cols=np.zeros((nn_tot,)))

    def compute(self, inputs, outputs):
        spec_heat = self.options['specific_heat']
        outputs['dTdt'] = (inputs['q_in'] - inputs['q_out']) / inputs['mass'] / spec_heat

    def compute_partials(self, inputs, J):
        nn_tot = self.options['num_nodes']
        spec_heat = self.options['specific_heat']

        J['dTdt','mass'] = - (inputs['q_in'] - inputs['q_out']) / inputs['mass']**2 / spec_heat
        J['dTdt','q_in'] = 1 / inputs['mass'] / spec_heat
        J['dTdt','q_out'] = - 1 / inputs['mass'] / spec_heat

class CoolantReservoirRate(ExplicitComponent):
    """
    Computes dT/dt of a coolant reservoir based on inflow and current temps and flow rate

    Inputs
    ------
    T_in : float
        Coolant stream in (vector, K)
    T_out : float
        Temperature of the reservoir (vector, K)
    mass : float
        Total quantity of coolant (scalar, kg)
    mdot_coolant : float
        Mass flow rate of the coolant (vector, kg/s)

    Outputs
    -------
    dTdt : float
        First derivative of temperature (vector, K/s)

    Options
    -------
    num_nodes : int
        The number of analysis points to run
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1)

    def setup(self):
        nn_tot = self.options['num_nodes']
        arange = np.arange(0, nn_tot)

        self.add_input('T_in', units='K', shape=(nn_tot,))
        self.add_input('T_out', units='K', shape=(nn_tot,))
        self.add_input('mdot_coolant', units='kg/s', shape=(nn_tot,))
        self.add_input('mass', units='kg')
        self.add_output('dTdt', units='K/s', shape=(nn_tot,))

        self.declare_partials(['dTdt'], ['T_in','T_out','mdot_coolant'], rows=arange, cols=arange)
        self.declare_partials(['dTdt'], ['mass'], rows=arange, cols=np.zeros((nn_tot,)))

    def compute(self, inputs, outputs):
        outputs['dTdt'] = inputs['mdot_coolant'] / inputs['mass'] * (inputs['T_in'] - inputs['T_out'])

    def compute_partials(self, inputs, J):
        J['dTdt','mass'] = - inputs['mdot_coolant'] / inputs['mass']**2 * (inputs['T_in'] - inputs['T_out'])
        J['dTdt','mdot_coolant'] = 1 / inputs['mass'] * (inputs['T_in'] - inputs['T_out'])
        J['dTdt','T_in'] = inputs['mdot_coolant'] / inputs['mass']
        J['dTdt','T_out'] = - inputs['mdot_coolant'] / inputs['mass']

class ThermalComponentMassless(ImplicitComponent):
    """
    Computes thermal residual of a component with heating, cooling, and thermal mass

    Inputs
    ------
    q_in : float
        Heat generated by the component (vector, W)
    q_out : float
        Heat to waste stream (vector, W)

    Outputs
    -------
    T_object : float
        Object temperature (vector, K/s)

    Options
    -------
    num_nodes : int
        The number of analysis points to run
    """
    def initialize(self):
        self.options.declare('num_nodes',default=1)

    def setup(self):
        nn_tot = self.options['num_nodes']
        arange = np.arange(0, nn_tot)

        self.add_input('q_in', units='W', shape=(nn_tot,))
        self.add_input('q_out', units='W', shape=(nn_tot,))
        self.add_output('T_object', units='K', shape=(nn_tot,))

        self.declare_partials(['T_object'], ['q_in'], rows=arange, cols=arange, val=np.ones((nn_tot,)))
        self.declare_partials(['T_object'], ['q_out'], rows=arange, cols=arange, val=-np.ones((nn_tot,)))

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['T_object'] = inputs['q_in'] - inputs['q_out']

class ConstantSurfaceTemperatureColdPlate_NTU(ExplicitComponent):
    """
    Computes heat rejection to fluid stream of a microchannel cold plate
    with uniform temperature

    Inputs
    ------
    T_in : float
        Coolant inlet temperature (vector, K)
    T_surface : float
        Temperature of the cold plate (vector, K)
    mdot_coolant : float
        Mass flow rate of the coolant (vector, kg/s)
    channel_length : float
        Length of each microchannel (scalar, m)
    channel_width : float
        Width of each microchannel (scalar, m)
    channel_height : float
        Height of each microchannel (scalar, m)
    n_parallel : float
        Number of fluid channels (scalar, dimensionless)

    Outputs
    -------
    q : float
        Heat transfer rate from the plate to the fluid (vector, W)
    T_out : float
        Outlet fluid temperature (vector, K)

    Options
    -------
    num_nodes : int
        The number of analysis points to run
    fluid_rho : float
        Coolant density in kg/m**3 (default 0.997, water)
    fluid_k : float
        Thermal conductivity of the fluid (W/m/K) (default 0.405, glycol/water)
    nusselt : float
        Hydraulic diameter Nusselt number of the coolant in the channels
        (default 7.54 for constant temperature infinite parallel plate)
    specific_heat : float
        Specific heat of the coolant (J/kg/K) (default 3801, glycol/water)
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of analysis points')
        self.options.declare('fluid_rho', default=997.0, desc='Fluid density in kg/m3')
        self.options.declare('fluid_k', default=0.405, desc='Thermal conductivity of the fluid in W / mK')
        self.options.declare('nusselt', default=7.54, desc='Hydraulic diameter Nusselt number')
        self.options.declare('specific_heat', default=3801, desc='Specific heat in J/kg/K')

    def setup(self):
        nn_tot = self.options['num_nodes']
        arange = np.arange(0, nn_tot)

        self.add_input('T_in', units='K', shape=(nn_tot,))
        self.add_input('T_surface', units='K', shape=(nn_tot,))
        self.add_input('channel_width', units='m')
        self.add_input('channel_height', units='m')
        self.add_input('channel_length', units='m')
        self.add_input('n_parallel')
        self.add_input('mdot_coolant', units='kg/s', shape=(nn_tot,))

        self.add_output('q', units='W', shape=(nn_tot,))
        self.add_output('T_out', units='K', shape=(nn_tot,))

        self.declare_partials(['q','T_out'], ['T_in','T_surface','mdot_coolant'], method='cs')
        self.declare_partials(['q','T_out'], ['channel_width','channel_height','channel_length','n_parallel'], method='cs')

    def compute(self, inputs, outputs):
        Ts = inputs['T_surface']
        Ti = inputs['T_in']

        Cmin = inputs['mdot_coolant'] * self.options['specific_heat']

        #cross_section_area = inputs['channel_width'] * inputs['channel_height'] * inputs['n_parallel']
        #flow_rate = inputs['mdot_coolant'] / self.options['rho'] / cross_section_area # m/s
        surface_area = 2 * (inputs['channel_width']*inputs['channel_length'] +
                            inputs['channel_height'] * inputs['channel_length']) * inputs['n_parallel']
        d_h = 2 * inputs['channel_width'] * inputs['channel_height'] / (inputs['channel_width'] + inputs['channel_height'])

        h = self.options['nusselt'] * self.options['fluid_k'] / d_h
        ntu = surface_area * h / Cmin
        effectiveness = 1 - np.exp(-ntu)
        outputs['q'] = effectiveness * Cmin * (Ts - Ti)
        outputs['T_out'] = inputs['T_in'] + outputs['q'] / inputs['mdot_coolant'] / self.options['specific_heat']

class LiquidCooledComp(Group):
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
    mass : float
        Object mass (only required in thermal mass mode) (scalar, kg)
    T_initial : float
        Initial temperature of the cold plate (only required in thermal mass mode) / object (scalar, K)
    duration : float
        Duration of mission segment, only required in unsteady mode
    channel_width : float
        Width of coolant channels (scalar, m)
    channel_height : float
        Height of coolant channels (scalar, m)
    channel_length : float
        Length of coolant channels (scalar, m)
    n_parallel : float
        Number of identical coolant channels (scalar, dimensionless)

    Outputs
    -------
    T_out : float
        Instantaneous coolant outlet temperature (vector, K)
    T: float
        Object temperature (vector, K)

    Options
    -------
    specific_heat_object : float
        Specific heat capacity of the object in J / kg / K (default 921 = aluminum)
    specific_heat_coolant : float
        Specific heat capacity of the coolant in J / kg / K (default 3801, glycol/water)
    num_nodes : int
        Number of analysis points to run
    quasi_steady : bool
        Whether or not to treat the component as having thermal mass
    """

    def initialize(self):
        self.options.declare('specific_heat_object', default=921.0, desc='Specific heat in J/kg/K')
        self.options.declare('specific_heat_coolant', default=3801, desc='Specific heat in J/kg/K')
        self.options.declare('quasi_steady', default=False, desc='Treat the component as quasi-steady or with thermal mass')
        self.options.declare('num_nodes', default=1, desc='Number of quasi-steady points to runs')

    def setup(self):
        nn = self.options['num_nodes']
        quasi_steady = self.options['quasi_steady']
        if not quasi_steady:
            self.add_subsystem('base',
                               ThermalComponentWithMass(specific_heat=self.options['specific_heat_object'],
                                                        num_nodes=nn),
                                                        promotes_inputs=['q_in', 'mass'])
            ode_integ = self.add_subsystem('ode_integ', Integrator(num_nodes=nn, diff_units='s', method='simpson', time_setup='duration'),
                                           promotes_outputs=['*'], promotes_inputs=['*'])
            # TODO lower limit 0
            ode_integ.add_integrand('T', rate_name='dTdt', units='K')
            self.connect('base.dTdt','dTdt')
        else:
            self.add_subsystem('base',
                               ThermalComponentMassless(num_nodes=nn),
                               promotes_inputs=['q_in'],
                               promotes_outputs=['T'])
        self.add_subsystem('hex',
                           ConstantSurfaceTemperatureColdPlate_NTU(num_nodes=nn, specific_heat=self.options['specific_heat_coolant']),
                                                                   promotes_inputs=['T_in', ('T_surface','T'),'n_parallel','channel*','mdot_coolant'],
                                                                   promotes_outputs=['T_out'])
        self.connect('hex.q','base.q_out')

class CoolantReservoir(Group):
    """A reservoir of coolant capable of buffering temperature

    Inputs
    ------
    mdot_coolant : float
        Coolant mass flow rate (vector, kg/s)
    T_in : float
        Coolant inflow temperature (vector, K)
    mass : float
        Object mass (only required in thermal mass mode) (scalar, kg)
    T_initial : float
        Initial temperature of the coolant reservoir(only required in thermal mass mode) / object (scalar, K)
    duration : float
        Time step of each mission segment (one for each segment) (scalar, s)
        If a single segment is provided (by default) this variable will be called just 'dt'
        only required in thermal mass mode

    Outputs
    -------
    T_out : float
        Coolant outlet temperature (vector, K)

    Options
    -------
    num_nodes : int
        Number of analysis points to run
    """

    def initialize(self):
        self.options.declare('num_nodes',default=5)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_subsystem('rate',
                           CoolantReservoirRate(num_nodes=nn),
                           promotes_inputs=['T_in', 'T_out', 'mass', 'mdot_coolant'])

        ode_integ = self.add_subsystem('ode_integ', Integrator(num_nodes=nn, diff_units='s', method='simpson', time_setup='duration'),
                                           promotes_outputs=['*'], promotes_inputs=['*'])
        # TODO lower limit 0
        ode_integ.add_integrand('T_out', rate_name='dTdt', start_name='T_initial', end_name='T_final', units='K')
        self.connect('rate.dTdt','dTdt')

class LiquidCoolantTestGroup(Group):
    """A component (heat producing) with thermal mass
    cooled by a cold plate.
    """

    def initialize(self):
        self.options.declare('num_nodes',default=11)
        self.options.declare('quasi_steady', default=False, desc='Treat the component as quasi-steady or with thermal mass')

    def setup(self):
        quasi_steady = self.options['quasi_steady']
        nn = self.options['num_nodes']

        iv = self.add_subsystem('iv',IndepVarComp(), promotes_outputs=['*'])
        #iv.add_output('q_in', val=10*np.concatenate([np.ones((nn,)),0.5*np.ones((nn,)),0.2*np.ones((nn,))]), units='kW')
        throttle_profile = np.ones((nn,))
        iv.add_output('q_in',val=10*throttle_profile, units='kW')
        #iv.add_output('T_in', val=40*np.ones((nn_tot,)), units='degC')
        iv.add_output('mdot_coolant', val=0.1*np.ones((nn,)), units='kg/s')
        iv.add_output('rho_coolant', val=997*np.ones((nn,)),units='kg/m**3')
        iv.add_output('motor_mass', val=50., units='kg')
        iv.add_output('coolant_mass', val=10., units='kg')
        iv.add_output('T_motor_initial', val=15, units='degC')
        iv.add_output('T_res_initial', val=15.1, units='degC')
        iv.add_output('duration', val=800, units='s')
        iv.add_output('channel_width', val=1, units='mm')
        iv.add_output('channel_height', val=20, units='mm')
        iv.add_output('channel_length', val=0.2, units='m')
        iv.add_output('n_parallel', val=20)
        Ueas = np.ones((nn))*150
        h = np.concatenate([np.linspace(0,25000,nn)])
        iv.add_output('fltcond|Ueas', val=Ueas, units='kn' )
        iv.add_output('fltcond|h', val=h, units='ft')


        self.add_subsystem('atmos',
                           ComputeAtmosphericProperties(num_nodes=nn),
                           promotes_inputs=["fltcond|h",
                                            "fltcond|Ueas"])

        if not quasi_steady:
            lc_promotes = ['q_in',('mass','motor_mass'),'duration','channel_*','n_parallel']
        else:
            lc_promotes = ['q_in','channel_*','n_parallel']

        self.add_subsystem('component',
                           LiquidCooledComp(num_nodes=nn,
                                            quasi_steady=quasi_steady),
                                            promotes_inputs=lc_promotes)
        self.add_subsystem('duct',
                           ImplicitCompressibleDuct(num_nodes=nn))

        self.connect('atmos.fltcond|p','duct.p_inf')
        self.connect('atmos.fltcond|T','duct.T_inf')
        self.connect('atmos.fltcond|Utrue','duct.Utrue')

        self.connect('component.T_out','duct.T_in_hot')
        self.connect('rho_coolant','duct.rho_hot')
        if quasi_steady:
            self.connect('duct.T_out_hot','component.T_in')
            self.connect('mdot_coolant',['component.mdot_coolant','duct.mdot_hot'])
        else:
            self.add_subsystem('reservoir',
                               CoolantReservoir(num_nodes=nn),
                                                promotes_inputs=['duration',('mass','coolant_mass')])
            self.connect('duct.T_out_hot','reservoir.T_in')
            self.connect('reservoir.T_out','component.T_in')
            self.connect('mdot_coolant',['component.mdot_coolant','duct.mdot_hot','reservoir.mdot_coolant'])
            self.connect('T_motor_initial','component.T_initial')
            self.connect('T_res_initial','reservoir.T_initial')




if __name__ == '__main__':
    # run this script from the root openconcept directory like so:
    # python .\openconcept\components\ducts.py
    quasi_steady = False
    nn = 11
    prob = Problem(LiquidCoolantTestGroup(quasi_steady=quasi_steady, num_nodes=nn))
    prob.model.options['assembled_jac_type'] = 'csc'
    prob.model.nonlinear_solver=NewtonSolver(iprint=2)
    prob.model.linear_solver = DirectSolver(assemble_jac=True)
    prob.model.nonlinear_solver.options['solve_subsystems'] = True
    prob.model.nonlinear_solver.options['maxiter'] = 20
    prob.model.nonlinear_solver.options['atol'] = 1e-8
    prob.model.nonlinear_solver.options['rtol'] = 1e-8
    prob.model.nonlinear_solver.linesearch = BoundsEnforceLS(bound_enforcement='scalar',print_bound_enforce=True)

    prob.setup(check=True,force_alloc_complex=True)

    prob.run_model()
    #print(prob['duct.inlet.M'])
    print(np.max(prob['component.T']-273.15))
    print(np.max(-prob['duct.force.F_net']))

    prob.check_partials(method='cs', compact_print=True)

    #prob.model.list_outputs(units=True, print_arrays=True)
    if quasi_steady:
        np.save('quasi_steady',prob['component.T'])

    # prob.run_driver()
    # prob.model.list_inputs(units=True)
    t = np.linspace(0,800,nn)/60

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(t, prob['component.T'] - 273.15)
    plt.xlabel('time (min)')
    plt.ylabel('motor temp (C)')
    plt.figure()
    plt.plot(prob['fltcond|h'], prob['component.T'] - 273.15)
    plt.xlabel('altitude (ft)')
    plt.ylabel('motor temp (C)')
    plt.figure()
    plt.plot(t, prob['duct.inlet.M'])
    plt.xlabel('Mach number')
    plt.ylabel('steady state motor temp (C)')
    plt.figure()
    plt.plot(prob['duct.inlet.M'], prob['duct.force.F_net'])
    plt.xlabel('M_inf')
    plt.ylabel('drag N')
    plt.figure()
    plt.plot(prob['duct.inlet.M'], prob['duct.mdot']/prob['atmos.fltcond|rho']/prob.get_val('atmos.fltcond|Utrue',units='m/s')/prob.get_val('duct.area_nozzle',units='m**2'))
    plt.xlabel('M_inf')
    plt.ylabel('mdot / rho / U / A_nozzle')
    plt.figure()
    plt.plot(prob['duct.inlet.M'],prob['duct.nozzle.M'])
    plt.xlabel('M_inf')
    # plt.ylabel('M_nozzle')
    plt.show()
