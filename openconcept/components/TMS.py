from __future__ import division
from openmdao.api import Problem, Group, IndepVarComp, BalanceComp, NewtonSolver, DirectSolver, BoundsEnforceLS
from openmdao.api import ScipyOptimizeDriver, ExplicitComponent, ImplicitComponent, ExecComp

import numpy as np
import scipy.sparse as sp
import sys, os
sys.path.insert(0,os.getcwd())
from openconcept.components.splitter import FlowSplit, FlowCombine
from openconcept.utilities.math import AddSubtractComp
from openconcept.components.heat_exchanger import HXGroup
from openconcept.components.ducts import ExplicitIncompressibleDuct
from openconcept.components.motor import SimpleMotor
from openconcept.components.battery import SimpleBattery
from openconcept.utilities.math.integrals import Integrator
from openconcept.components.thermal import ThermalComponentWithMass, ConstantSurfaceTemperatureColdPlate_NTU
from openconcept.components.thermal import SimpleHeatPump, HeatPumpWithIntegratedCoolantLoop, LiquidCooledComp
from openconcept.analysis.atmospherics.compute_atmos_props import ComputeAtmosphericProperties
from battery_cooling import LiquidCooledBattery ################################################### THIS WILL CHANGE #######################################

class ParallelTMS(Group):
    """
    Models a thermal management system (TMS) cooling an electric motor and battery
    with two parallel cooling loops. The parallel loops converge and pass through
    a single heat pump. The coolant is split based on the temperature constraints
    of the battery and motor. The following schematic shows the setup of the system:

    ,--> Splitter --> Motor --> Combiner -->,
    |       |                      |        |
    |       '----> Battery ------->'        |
    |                                       |
    '<--------    (Cold)    <---------------'
    ---------> Refrigerator
    ,-------->    (Hot)     --------->,
    |                                 |
    '<--- Heat exchanger and duct <---'

    Two parameters are varied so that the battery and motor temperatures
    are at their desired temperature (set using a combination of the limit
    temperatures and temperature buffer). Those two parameters are the following:
        1. The refrigerator cold side set temperature is adjusted to
           set the battery temperature
        2. The amount of coolant that goes to the motor vs. the battery (think
           of this as how open the splitter valve is) is used to set the
           motor temperature

    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length; default 1)
    motor_T_limit : float
        Upper temperature limit of motor (deg C; default 90 deg C)
    battery_T_limit : float
        Upper temperature limit of battery (deg C; default 40 deg C)
    T_limit_buffer : float
        Degrees below limit temp to set battery and motor temp (deg C; default 5 deg C)
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of analysis points to run')
        self.options.declare('motor_T_limit', default=90., desc='Upper limit on motor temperature (degC)')
        self.options.declare('battery_T_limit', default=40., desc='Upper limit on battery temperature (degC)')
        self.options.declare('T_limit_buffer', default=5., desc='Buffer below limit to set motor and battery temp (degC)')

    def setup(self):
        nn = self.options['num_nodes']
        nn_ones = np.ones((nn,))

        ######### Declare variables needed for components to simplify the TMS's I/O #########
        iv = IndepVarComp()
        # Flight condition
        iv.add_output('throttle', val=0.5, shape=(nn,))
        iv.add_output('fltcond|h', val=20e3, shape=(nn,), units='ft')
        iv.add_output('fltcond|Ueas', val=250., shape=(nn,), units='kn')
        # Refrigerator settings
        iv.add_output('refrig_T_h_set', val=100., shape=(nn,), units='degC')
        iv.add_output('eff_factor', val=0.4, units=None)
        iv.add_output('bypass_heat_pump', val=0., shape=(nn,), units=None)
        # Coolant parameters
        iv.add_output('mdot_coolant_hot', val=1., shape=(nn,), units='kg/s')
        iv.add_output('mdot_coolant_cold', val=1., shape=(nn,), units='kg/s')
        iv.add_output('rho_coolant', val=1000., shape=(nn,), units='kg/m**3')
        # Battery
        iv.add_output('battery_weight', val=500., units='kg')
        iv.add_output('q_batt', val=5., shape=(nn,), units='kW')
        iv.add_output('batt_T_limit', val=self.options['battery_T_limit'] - self.options['T_limit_buffer'],
                      shape=(nn,), units='degC')  # set this via options
        # Motor
        iv.add_output('q_motor', val=3., shape=(nn,), units='kW')
        iv.add_output('motor_T_limit', val=self.options['motor_T_limit'] - self.options['T_limit_buffer'],
                      shape=(nn,), units='degC')  # set this via options
        # Cold plate/bandolier geometry
        iv.add_output('channel_width', val=1., units='mm')
        iv.add_output('channel_length', val=.2, units='m')
        iv.add_output('channel_height', val=50., units='mm')
        iv.add_output('n_parallel', val=15)
        iv.add_output('cells_per_bandolier', val=21)
        
        self.add_subsystem('iv', iv, promotes_outputs=['*'])


        ######### Add components to model #########
        # Motor and battery run in massless mode
        self.add_subsystem('motor_heat_sink', LiquidCooledComp(num_nodes=nn, quasi_steady=True),
                           promotes_inputs=[('q_in', 'q_motor'), 'channel_length', 'channel_width', 'channel_height', 'n_parallel'])
        self.add_subsystem('battery_heat_sink', LiquidCooledBattery(num_nodes=nn, quasi_steady=True),
                           promotes_inputs=[('q_in', 'q_batt'), 'battery_weight', ('n_cpb', 'cells_per_bandolier'),
                                            ('t_channel', 'channel_width')])
        
        # Use splitter and combiner to cool motor and battery in parallel
        self.add_subsystem('coolant_splitter', FlowSplit(num_nodes=nn),
                           promotes_inputs=[('mdot_in', 'mdot_coolant_cold'), 'mdot_split_fraction'])
        self.add_subsystem('coolant_combiner', FlowCombine(num_nodes=nn))

        # Hot side balance param will be set to the cooling duct nozzle area
        self.add_subsystem('refrig', HeatPumpWithIntegratedCoolantLoop(num_nodes=nn,
                                                                       hot_side_balance_param_units='inch**2',
                                                                       hot_side_balance_param_lower=1e-10,
                                                                       hot_side_balance_param_upper=100),
                           promotes_inputs=['mdot_coolant_hot', 'T_in_hot', ('T_h_set', 'refrig_T_h_set'),
                                            ('T_c_set', 'refrig_T_c_set'), 'bypass_heat_pump', 'eff_factor'])

        # Hot side components
        self.add_subsystem('hx', HXGroup(num_nodes=nn), promotes_inputs=[('mdot_hot', 'mdot_coolant_hot'),('rho_cold','fltcond|rho'),
                                                                         ('T_in_cold', 'fltcond|T'), ('rho_hot', 'rho_coolant')])
        self.add_subsystem('duct', ExplicitIncompressibleDuct(num_nodes=nn), promotes_inputs=['fltcond|*'])

        # Atmospheric model
        self.add_subsystem('atmos', ComputeAtmosphericProperties(num_nodes=nn),
                           promotes_inputs=['fltcond|h', 'fltcond|Ueas'], promotes_outputs=['fltcond|*'])

        ######### Connect cold side #########
        # Battery
        self.connect('refrig.T_out_cold', 'battery_heat_sink.T_in')
        self.connect('coolant_splitter.mdot_out_B', 'battery_heat_sink.mdot_coolant')
        self.connect('coolant_splitter.mdot_out_B', 'coolant_combiner.mdot_in_B')
        self.connect('battery_heat_sink.T_out', 'coolant_combiner.T_in_B')

        # Motor
        self.connect('refrig.T_out_cold', 'motor_heat_sink.T_in')
        self.connect('coolant_splitter.mdot_out_A', 'motor_heat_sink.mdot_coolant')
        self.connect('motor_heat_sink.T_out', 'coolant_combiner.T_in_A')
        self.connect('coolant_splitter.mdot_out_A', 'coolant_combiner.mdot_in_A')

        # Connecting back to refrigerator
        self.connect('coolant_combiner.mdot_out', 'refrig.mdot_coolant_cold')
        self.connect('coolant_combiner.T_out', 'refrig.T_in_cold')

        ######### Connect hot side #########
        self.connect('refrig.hot_side_balance_param','duct.area_nozzle')
        self.connect('refrig.T_out_hot', 'hx.T_in_hot')
        self.connect('hx.T_out_hot', 'T_in_hot')
        self.connect('hx.delta_p_cold', 'duct.delta_p_hex')
        self.connect('duct.mdot','hx.mdot_cold')

        ######### Modulate the refrig cold set temp and FlowSplitter fraction to hit battery and motor temp limits #########
        self.add_subsystem('motor_temp_bal', BalanceComp(name='splitter_fraction', units=None, eq_units='K', val=0.01*np.ones(nn),
                                                        lower=1e-4*np.ones(nn), upper=np.ones(nn)-1e-4, lhs_name='motor_T',
                                                        rhs_name='motor_T_limit'), promotes_inputs=['motor_T_limit'],
                                                        promotes_outputs=[('splitter_fraction', 'mdot_split_fraction')])
        self.connect('motor_heat_sink.T', 'motor_temp_bal.motor_T')

        self.add_subsystem('batt_temp_bal', BalanceComp(name='T_c_set', units='degC', eq_units='K', val=30.*np.ones(nn),
                                                        lower=-1e2*np.ones(nn), upper=99*np.ones(nn), lhs_name='batt_T',
                                                        rhs_name='batt_T_limit'), promotes_inputs=['batt_T_limit'],
                                                        promotes_outputs=[('T_c_set', 'refrig_T_c_set')])
        self.connect('battery_heat_sink.T', 'batt_temp_bal.batt_T')

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
