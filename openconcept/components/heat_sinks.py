import openmdao.api as om 
import numpy as np 
from openconcept.utilities.math.integrals import Integrator
import warnings

class BandolierCoolingSystem(om.ExplicitComponent):
    """
    Computes battery heat transfer for a parameteric battery
    based on Tesla's Model 3 design.

    Assumptions:
    Heat generated uniformly in the cell
    Weight per cell and thermal resistance stay constant
    even as specific energy varies parametrically
    (this means that cell count is constant with pack WEIGHT,
    not pack ENERGY as technology improves)
    Cylindrical cells attached to Tesla-style thermal ribbon
    Liquid cooling
    Heat transfer through axial direction only (not baseplate)
    2170 cells (21 mm diameter, 70mm tall)
    Battery thermal model assumes unsteady cell temperature,
    quasi-steady temperature gradients

    Inputs
    ------
    q_in : float
        Heat generation rate in the battery (vector, W)
    T_in : float
        Coolant inlet temperature (vector, K)
    T_battery : float
        Volume averaged battery temperature (vector, K)
    mdot_coolant : float
        Mass flow rate of coolant through the bandolier (vector, kg/s)
    battery_weight : float
        Weight of the battery (overall). Default 100kg (scalar)
    n_cpb : float
        Number of cells long per "bandolier" actual count is 2x (scalar, default 82, Tesla)
    t_channel : float
        Thickness (width) of the cooling channel in the bandolier
        (scalar, default 1mm)
    Outputs
    -------
    dTdt : float
        Time derivative dT/dt (Tbar in the paper) (vector, K/s)
    T_surface : float
        Surface temp of the battery (vector, K)
    T_core : float
        Center temp of the battery (vector, K)
    q : float
        Heat transfer rate from the motor to the fluid (vector, W)
    T_out : float
        Outlet fluid temperature (vector, K)

    Options
    -------
    num_nodes : float
        The number of analysis points to run
    coolant_specific_heat : float
        Specific heat of the coolant (J/kg/K) (default 3801, glycol/water)
    fluid_k : float
        Thermal conductivity of the coolant (W/m/K)
    nusselt : float
        Nusselt number of the coolant channel (default 7.54 for uniform surf temp)
    cell_kr : float
        Thermal conductivity of the cell in the radial direction (W/m/k)
    cell_diameter : float
        Battery diameter (default 21mm for 2170 cell)
    cell_height : float
        Battery height (default 70mm for 2170 cell)
    cell_mass : float
        Battery weight (default 70g for 2170 cell)
    cell_specific_heat : float
        Mass average specific heat of the battery (default 900, LiIon cylindrical cell)
    battery_weight_fraction : float
        Fraction of battery by weight that is cells (default 0.72 knocks down Tesla by a bit)
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of analysis points')
        self.options.declare('coolant_specific_heat', default=3801, desc='Coolant specific heat in J/kg/K')
        self.options.declare('fluid_k', default=0.405, desc='Thermal conductivity of the fluid in W / mK')
        self.options.declare('nusselt', default=7.54, desc='Hydraulic diameter Nusselt number')

        self.options.declare('cell_kr', default=0.3) # 0.455 for an 18650 cell, knocked down a bit
        self.options.declare('cell_diameter', default=0.021)
        self.options.declare('cell_height', default=0.070)
        self.options.declare('cell_mass', default=0.070)
        self.options.declare('cell_specific_heat', default=875.)
        self.options.declare('battery_weight_fraction', default=0.65)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('q_in', shape=(nn,), units='W', val=0.0)
        self.add_input('T_in', shape=(nn,), units='K', val=300.)
        self.add_input('T_battery', shape=(nn,), units='K', val=300.)
        self.add_input('mdot_coolant', shape=(nn,), units='kg/s', val=0.20)
        self.add_input('battery_weight', units='kg', val=478.)
        self.add_input('n_cpb', units=None, val=82.)
        self.add_input('t_channel', units='m', val=0.0005)

        self.add_output('dTdt', shape=(nn,), units='K/s', tags=['integrate', 'state_name:T_battery', 'state_units:K', 'state_val:300.0', 'state_promotes:True'])
        self.add_output('T_surface', shape=(nn,), units='K', lower=1e-10)
        self.add_output('T_core', shape=(nn,), units='K', lower=1e-10)
        self.add_output('q', shape=(nn,), units='W')
        self.add_output('T_out', shape=(nn,), units='K', val=300, lower=1e-10)

        self.declare_partials(['*'], ['*'], method='cs')

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        n_cells = inputs['battery_weight'] * self.options['battery_weight_fraction'] / self.options['cell_mass']
        n_bandoliers = n_cells / inputs['n_cpb'] / 2

        mdot_b = inputs['mdot_coolant'] / n_bandoliers
        q_cell = inputs['q_in'] / n_cells
        hconv = self.options['nusselt'] * self.options['fluid_k'] / 2 / inputs['t_channel']

        Hc = self.options['cell_height']
        Dc = self.options['cell_diameter']
        mc = self.options['cell_mass']
        krc = self.options['cell_kr']
        cpc = self.options['cell_specific_heat']
        L_bandolier = inputs['n_cpb'] * Dc

        cpf = self.options['coolant_specific_heat'] # of the coolant

        A_heat_trans = Hc * L_bandolier * 2 # two sides of the tape
        NTU = hconv * A_heat_trans / mdot_b / cpf
        Kcell = mdot_b * cpf * (1 - np.exp(-NTU)) / 2 / inputs['n_cpb'] # divide out the total bandolier convection by 2 * n_cpb cells
        # the convective heat transfer is (Ts - Tin) * Kcell
        PI = np.pi
        
        Tbar = inputs['T_battery']
        Rc = Dc / 2

        K_cyl = 8*np.pi*Hc*krc

        Ts = (K_cyl * Tbar + Kcell * inputs['T_in']) / (K_cyl + Kcell)
        
        outputs['T_surface'] = Ts

        q_conv = (Ts - inputs['T_in']) * Kcell * n_cells
        outputs['dTdt'] = (q_cell - (Ts - inputs['T_in']) * Kcell) / mc / cpc  # todo check that this quantity matches convection


        outputs['q'] = q_conv

        qcheck = (Tbar - Ts) * K_cyl
        # UAcomb = 1/(1/hconv/A_heat_trans+1/K_cyl/2/inputs['n_cpb'])
        # qcheck2 = (Tbar - inputs['T_in']) * mdot_b * cpf * (1 - np.exp(-UAcomb/mdot_b/cpf)) / 2 / inputs['n_cpb']

        # if np.sum(np.abs(qcheck - outputs['q']/n_cells)) > 1e-5:
        #     # the heat flux across the cell is not equal to the heat flux due to convection
        #     raise ValueError('The surface temperature solution appears to be wrong')

        outputs['T_out'] = inputs['T_in'] + outputs['q'] / inputs['mdot_coolant'] / cpf
        outputs['T_core'] = (Tbar - Ts) + Tbar

class LiquidCooledBattery(om.Group):
    """A battery with liquid cooling 

    Inputs
    ------
    q_in : float
        Heat produced by the operating component (vector, W)
    mdot_coolant : float
        Coolant mass flow rate (vector, kg/s)
    T_in : float
        Instantaneous coolant inflow temperature (vector, K)
    battery_weight : float
        Battery weight (scalar, kg)
    n_cpb : float
        Number of cells long per "bandolier" actual count is 2x (scalar, default 82, Tesla)
    t_channel : float
        Thickness (width) of the cooling channel in the bandolier
        (scalar, default 1mm)
    T_initial : float
        Initial temperature of the battery (only required in thermal mass mode) (scalar, K)
    duration : float
        Duration of mission segment, only required in unsteady mode

    Outputs
    -------
    T_out : float
        Instantaneous coolant outlet temperature (vector, K)
    T: float
        Battery volume averaged temperature (vector, K)
    T_core : float
        Battery core temperature (vector, K)
    T_surface : float
        Battery surface temperature (vector, K)
        
    Options
    -------
    num_nodes : int
        Number of analysis points to run
    quasi_steady : bool
        Whether or not to treat the component as having thermal mass
        num_nodes : float
        The number of analysis points to run
    coolant_specific_heat : float
        Specific heat of the coolant (J/kg/K) (default 3801, glycol/water)
    fluid_k : float
        Thermal conductivity of the coolant (W/m/K)
    nusselt : float
        Nusselt number of the coolant channel (default 7.54 for uniform surf temp)
    cell_kr : float
        Thermal conductivity of the cell in the radial direction (W/m/k)
    cell_diameter : float
        Battery diameter (default 21mm for 2170 cell)
    cell_height : float
        Battery height (default 70mm for 2170 cell)
    cell_mass : float
        Battery weight (default 70g for 2170 cell)
    cell_specific_heat : float
        Mass average specific heat of the battery (default 900, LiIon cylindrical cell)
    battery_weight_fraction : float
        Fraction of battery by weight that is cells (default 0.72 knocks down Tesla by a bit)
    """

    def initialize(self):
        self.options.declare('quasi_steady', default=False, desc='Treat the component as quasi-steady or with thermal mass')
        self.options.declare('num_nodes', default=1, desc='Number of quasi-steady points to runs')
        self.options.declare('coolant_specific_heat', default=3801, desc='Coolant specific heat in J/kg/K')
        self.options.declare('fluid_k', default=0.405, desc='Thermal conductivity of the fluid in W / mK')
        self.options.declare('nusselt', default=7.54, desc='Hydraulic diameter Nusselt number')

        self.options.declare('cell_kr', default=0.3) # 0.455 for an 18650 cell, knocked down a bit
        self.options.declare('cell_diameter', default=0.021)
        self.options.declare('cell_height', default=0.070)
        self.options.declare('cell_mass', default=0.070)
        self.options.declare('cell_specific_heat', default=875.)
        self.options.declare('battery_weight_fraction', default=0.65)
    def setup(self):
        nn = self.options['num_nodes']
        quasi_steady = self.options['quasi_steady']
        
        self.add_subsystem('hex', BandolierCoolingSystem(num_nodes=nn, 
                                                         coolant_specific_heat=self.options['coolant_specific_heat'],
                                                         fluid_k=self.options['fluid_k'],
                                                         nusselt=self.options['nusselt'],
                                                         cell_kr=self.options['cell_kr'],
                                                         cell_diameter=self.options['cell_diameter'],
                                                         cell_height=self.options['cell_height'],
                                                         cell_mass=self.options['cell_mass'],
                                                         cell_specific_heat=self.options['cell_specific_heat'],
                                                         battery_weight_fraction=self.options['battery_weight_fraction']),
                            promotes_inputs=['q_in', 'mdot_coolant', 'T_in', ('T_battery', 'T'), 'battery_weight', 'n_cpb', 't_channel'],
                            promotes_outputs=['T_core', 'T_surface', 'T_out', 'dTdt'])
        
        if not quasi_steady:
            ode_integ = self.add_subsystem('ode_integ', Integrator(num_nodes=nn, diff_units='s', method='simpson', time_setup='duration'),
                                           promotes_outputs=['*'], promotes_inputs=['*'])
            ode_integ.add_integrand('T', rate_name='dTdt', units='K', lower=1e-10)
        else:
            self.add_subsystem('thermal_bal',
                               om.BalanceComp('T', eq_units='K/s', lhs_name='dTdt', rhs_val=0.0, units='K', lower=1.0, val=299.*np.ones((nn,))),
                               promotes_inputs=['dTdt'],
                               promotes_outputs=['T'])

class MotorCoolingJacket(om.ExplicitComponent):
    """
    Computes motor winding temperature assuming 
    well-designed, high-power-density aerospace motor.
    This component is based on the following assumptions:
    - 2020 technology level
    - 200kW-1MW class inrunner PM motor
    - Liquid cooling of the stators
    - "Reasonable" coolant flow rates (component will validate this)
    - Thermal performance similiar to the Siemens SP200D motor

    The component assumes a constant heat transfer coefficient based
    on the surface area of the motor casing (not counting front and rear faces)
    The MagniX Magni 250/500 and Siemens SP200D motors were measured
    using rough photogrammetry.

    Magni250: 280kW rated power, ~0.559m OD, 0.2m case "depth" (along thrust axis)
    Magni500: 560kW rated power, ~0.652m OD, 0.4m case "depth"
    Siemens SP200D: 200kW rated power, ~0.63m OD, ~0.16 case "depth"

    Based on these dimensions I assume 650kW per square meter 
    of casing surface area. This includes only the cylindrical portion, 
    not the front and rear motor faces.

    Using a thermal FEM image of the SP200D, I estimate 
    a temperature rise of 23K from coolant inlet temperature (~85C) 
    to winding max temp (~108C) at the steady state operating point. 
    With 95% efficiency at 200kW, this is about 1373 W / m^2 casing area / K.
    We'll reduce that somewhat since this is a direct oil cooling system, 
    and assume 1100 W/m^2/K instead.

    Dividing 1.1 kW/m^2/K by 650kWrated/m^2 gives: 1.69e-3 kW / kWrated / K
    At full rated power and 95% efficiency, this is 29.5C steady state temp rise
    which the right order of magnitude. 

    Inputs
    ------
    q_in : float
        Heat production rate in the motor (vector, W)
    T_in : float
        Coolant inlet temperature (vector, K)
    T : float
        Temperature of the motor windings (vector, K)
    mdot_coolant : float
        Mass flow rate of the coolant (vector, kg/s)
    power_rating : float
        Rated steady state power of the motor (scalar, W)
    motor_weight : float
        Weight of electric motor (scalar, kg)

    Outputs
    -------
    dTdt : float
        Time derivative dT/dt (vector, K/s)
    q : float
        Heat transfer rate from the motor to the fluid (vector, W)
    T_out : float
        Outlet fluid temperature (vector, K)
    

    Options
    -------
    num_nodes : float
        The number of analysis points to run
    coolant_specific_heat : float
        Specific heat of the coolant (J/kg/K) (default 3801, glycol/water)
    case_cooling_coefficient : float
        Watts of heat transfer per square meter of case surface area per K
        temperature differential (default 1100 W/m^2/K)
    case_area_coefficient : float
        rated motor power per square meter of case surface area
        (default 650,000 W / m^2)
    motor_specific_heat : float
        Specific heat of the motor casing (J/kg/K) (default 921, alu)
    """
        
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of analysis points')
        self.options.declare('coolant_specific_heat', default=3801, desc='Specific heat in J/kg/K')
        self.options.declare('case_cooling_coefficient', default=1100.)
        self.options.declare('case_area_coefficient', default=650000.)
        self.options.declare('motor_specific_heat', default=921, desc='Specific heat in J/kg/K - default 921 for aluminum')
    
    def setup(self):
        nn = self.options['num_nodes']
        arange = np.arange(nn)
        self.add_input('q_in', shape=(nn,), units='W', val=0.0)
        self.add_input('T_in', shape=(nn,), units='K', val=330)
        self.add_input('T', shape=(nn,), units='K', val=359.546)
        self.add_input('mdot_coolant', shape=(nn,), units='kg/s', val=1.0)
        self.add_input('power_rating', units='W', val=2e5)
        self.add_input('motor_weight', units='kg', val=100)
        self.add_output('q', shape=(nn,), units='W')
        self.add_output('T_out', shape=(nn,), units='K', val=300, lower=1e-10)
        self.add_output('dTdt', shape=(nn,), units='K/s', tags=['integrate', 'state_name:T_motor', 'state_units:K', 'state_val:300.0', 'state_promotes:True'])        
        
        self.declare_partials(['T_out','q','dTdt'], ['power_rating'], rows=arange, cols=np.zeros((nn,)))
        self.declare_partials(['dTdt'], ['motor_weight'], rows=arange, cols=np.zeros((nn,)))

        self.declare_partials(['T_out','q','dTdt'], ['T_in', 'T','mdot_coolant'], rows=arange, cols=arange)
        self.declare_partials(['dTdt'], ['q_in'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        const = self.options['case_cooling_coefficient'] / self.options['case_area_coefficient']
        
        NTU = const * inputs['power_rating'] / inputs['mdot_coolant'] / self.options['coolant_specific_heat']
        effectiveness = 1 - np.exp(-NTU)
        heat_transfer = (inputs['T'] - inputs['T_in']) * effectiveness * inputs['mdot_coolant'] * self.options['coolant_specific_heat']
        outputs['q'] = heat_transfer
        outputs['T_out'] = inputs['T_in'] + heat_transfer / inputs['mdot_coolant'] / self.options['coolant_specific_heat']
        outputs['dTdt'] = (inputs['q_in'] - outputs['q']) / inputs['motor_weight'] / self.options['motor_specific_heat']
        
    def compute_partials(self, inputs, J):
        nn = self.options['num_nodes']
        cp = self.options['coolant_specific_heat']
        mdot = inputs['mdot_coolant']
        const = self.options['case_cooling_coefficient'] / self.options['case_area_coefficient']
        
        NTU = const * inputs['power_rating'] / mdot / cp
        dNTU_dP = const / mdot / cp 
        dNTU_dmdot = -const * inputs['power_rating'] / mdot **2 / cp 
        effectiveness = 1 - np.exp(-NTU)
        deff_dP = np.exp(-NTU) * dNTU_dP
        deff_dmdot = np.exp(-NTU) * dNTU_dmdot

        heat_transfer = (inputs['T'] - inputs['T_in']) * effectiveness * inputs['mdot_coolant'] * self.options['coolant_specific_heat']

        J['q', 'T'] = effectiveness * mdot * cp
        J['q', 'T_in'] = - effectiveness * mdot * cp
        J['q', 'power_rating'] = (inputs['T'] - inputs['T_in']) * deff_dP * mdot * cp
        J['q', 'mdot_coolant'] = (inputs['T'] - inputs['T_in']) * cp * (effectiveness + deff_dmdot * mdot)

        J['T_out', 'T'] = J['q','T'] / mdot / cp
        J['T_out', 'T_in'] = np.ones(nn) + J['q','T_in'] / mdot / cp
        J['T_out', 'power_rating'] = J['q', 'power_rating'] / mdot / cp
        J['T_out', 'mdot_coolant'] = (J['q', 'mdot_coolant'] * mdot - heat_transfer) / cp / mdot ** 2

        J['dTdt', 'q_in'] = 1 / inputs['motor_weight'] / self.options['motor_specific_heat']
        J['dTdt', 'T'] = -J['q', 'T'] / inputs['motor_weight'] / self.options['motor_specific_heat']
        J['dTdt', 'T_in'] = -J['q', 'T_in'] / inputs['motor_weight'] / self.options['motor_specific_heat']
        J['dTdt', 'power_rating'] = -J['q', 'power_rating'] / inputs['motor_weight'] / self.options['motor_specific_heat']
        J['dTdt', 'mdot_coolant'] = -J['q', 'mdot_coolant'] / inputs['motor_weight'] / self.options['motor_specific_heat']
        J['dTdt', 'motor_weight'] = -(inputs['q_in'] - heat_transfer) / inputs['motor_weight']**2 / self.options['motor_specific_heat']

class LiquidCooledMotor(om.Group):
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
                               om.BalanceComp('T', eq_units='K/s', lhs_name='dTdt', rhs_val=0.0, units='K', lower=1.0, val=299.*np.ones((nn,))),
                               promotes_inputs=['dTdt'],
                               promotes_outputs=['T'])

class SimplePump(om.ExplicitComponent):
    """
    A pump that circulates coolant against pressure.
    The default parameters are based on a survey of commercial 
    airplane fuel pumps of a variety of makes and models.

    Inputs
    ------
    power_rating : float
        Maximum rated electrical power (scalar, W)
    mdot_coolant : float
        Coolant mass flow rate (vector, kg/s)
    rho_coolant : float
        Coolant density (vector, kg/m3)
    delta_p : float
        Pressure rise provided by the pump (vector, kg/s)

    Outputs
    -------
    elec_load : float
        Electricity used by the pump (vector, W)
    component_weight : float
        Pump weight (scalar, kg)
    component_sizing_margin : float
        Fraction of total power rating used via elec_load (vector, dimensionless)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length; default 1)
    efficiency : float
        Pump electrical + mech efficiency. Sensible range 0.0 to 1.0 (default 0.35)
    weight_base : float
        Base weight of pump, doesn't change with power rating (default 0)
    weight_inc : float
        Incremental weight of pump, scales linearly with power rating (default 1/450 kg/W)
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')
        self.options.declare('efficiency', default=0.35, desc='Efficiency (dimensionless)')
        self.options.declare('weight_base', default=0.0, desc='Pump base weight')
        self.options.declare('weight_inc', default=1/450, desc='Incremental pump weight (kg/W)')

    def setup(self):
        nn = self.options['num_nodes']
        eta = self.options['efficiency']
        weight_inc = self.options['weight_inc']

        self.add_input('power_rating', units='W', desc='Pump electrical power rating')
        self.add_input('mdot_coolant', units='kg/s', desc='Coolant mass flow rate', val=np.ones((nn,)))
        self.add_input('delta_p', units='Pa', desc='Pump pressure rise', val=np.ones((nn,)))
        self.add_input('rho_coolant', units='kg/m**3', desc='Coolant density', val=np.ones((nn,)))

        self.add_output('elec_load', units='W', desc='Pump electrical load', val=np.ones((nn,)))
        self.add_output('component_weight', units='kg', desc='Pump weight')
        self.add_output('component_sizing_margin', units=None, val=np.ones((nn,)), desc='Comp sizing margin')

        self.declare_partials(['elec_load','component_sizing_margin'], ['rho_coolant', 'delta_p', 'mdot_coolant'], rows=np.arange(nn), cols=np.arange(nn))
        self.declare_partials(['component_sizing_margin'], ['power_rating'], rows=np.arange(nn), cols=np.zeros(nn))
        self.declare_partials(['component_weight'], ['power_rating'], val=weight_inc)



    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        eta = self.options['efficiency']
        weight_inc = self.options['weight_inc']
        weight_base = self.options['weight_base']

        outputs['component_weight'] = weight_base + weight_inc * inputs['power_rating']
        
        # compute the fluid power
        vol_flow_rate = inputs['mdot_coolant'] / inputs['rho_coolant'] # m3/s
        fluid_power = vol_flow_rate * inputs['delta_p']
        outputs['elec_load'] = fluid_power / eta 
        outputs['component_sizing_margin'] = outputs['elec_load'] / inputs['power_rating']
    

    def compute_partials(self, inputs, J):
        nn = self.options['num_nodes']
        eta = self.options['efficiency']

        J['elec_load', 'mdot_coolant'] =  inputs['delta_p'] / inputs['rho_coolant'] / eta
        J['elec_load', 'delta_p'] = inputs['mdot_coolant'] / inputs['rho_coolant'] / eta
        J['elec_load', 'rho_coolant'] = -inputs['mdot_coolant'] * inputs['delta_p'] / inputs['rho_coolant'] ** 2 / eta
        for in_var in ['mdot_coolant', 'delta_p', 'rho_coolant']:
            J['component_sizing_margin', in_var] = J['elec_load', in_var]  / inputs['power_rating']
        J['component_sizing_margin', 'power_rating'] = - inputs['mdot_coolant'] * inputs['delta_p'] / inputs['rho_coolant'] / eta / inputs['power_rating'] ** 2

class SimpleHose(om.ExplicitComponent):
    """
    A coolant hose used to track pressure drop and weight in long hose runs.

    Inputs
    ------
    hose_diameter : float
        Inner diameter of the hose (scalar, m)
    hose_length
        Length of the hose (scalar, m)
    hose_design_pressure
        Max operating pressure of the hose (scalar, Pa)
    mdot_coolant : float
        Coolant mass flow rate (vector, kg/s)
    rho_coolant : float
        Coolant density (vector, kg/m3)
    mu_coolant : float
        Coolant viscosity (scalar, kg/m/s)

    Outputs
    -------
    delta_p : float
        Pressure drop in the hose - positive is loss (vector, kg/s)
    component_weight : float
        Weight of hose AND coolant (scalar, kg)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length; default 1)
    hose_operating_stress : float
        Hoop stress at design pressure (Pa) set to 300 Psi equivalent per empirical data
    hose_density : float
        Material density of the hose (kg/m3) set to 0.049 lb/in3 equivalent per empirical data
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')
        self.options.declare('hose_operating_stress', default=2.07e6, desc='Hoop stress at max op press in Pa')
        self.options.declare('hose_density', default=1356.3, desc='Hose matl density in kg/m3')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('hose_diameter', val=0.0254, units='m')
        self.add_input('hose_length', val=1.0, units='m')
        self.add_input('hose_design_pressure', units='Pa', val=1.03e6, desc='Hose max operating pressure')


        self.add_input('mdot_coolant', units='kg/s', desc='Coolant mass flow rate', val=np.ones((nn,)))
        self.add_input('rho_coolant', units='kg/m**3', desc='Coolant density', val=1020.*np.ones((nn,)))
        self.add_input('mu_coolant', val=1.68e-3, units='kg/m/s', desc='Coolant viscosity')

        self.add_output('delta_p', units='Pa', desc='Hose pressure drop', val=np.ones((nn,)))
        self.add_output('component_weight', units='kg', desc='Pump weight')

        self.declare_partials(['delta_p'], ['rho_coolant', 'mdot_coolant'], rows=np.arange(nn), cols=np.arange(nn))
        self.declare_partials(['delta_p'], ['hose_diameter', 'hose_length', 'mu_coolant'], rows=np.arange(nn), cols=np.zeros(nn))
        self.declare_partials(['component_weight'], ['hose_design_pressure','hose_length','hose_diameter'], rows=[0], cols=[0])
        self.declare_partials(['component_weight'], ['rho_coolant'], rows=[0], cols=[0])


    def _compute_pressure_drop(self, inputs):
        xs_area = np.pi * (inputs['hose_diameter'] / 2) ** 2
        U = inputs['mdot_coolant'] / inputs['rho_coolant'] / xs_area
        Redh = inputs['rho_coolant'] * U * inputs['hose_diameter'] / inputs['mu_coolant']
        # darcy friction from the Blasius correlation
        f = 0.3164 * Redh ** (-1/4)
        dp = f * inputs['rho_coolant'] * U ** 2 * inputs['hose_length'] / 2 / inputs['hose_diameter']
        return dp

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        sigma = self.options['hose_operating_stress']
        rho_hose = self.options['hose_density']

        outputs['delta_p'] = self._compute_pressure_drop(inputs)

        thickness = inputs['hose_diameter'] * inputs['hose_design_pressure'] / 2 / sigma
        
        w_hose = (inputs['hose_diameter'] + thickness) * np.pi * thickness * rho_hose * inputs['hose_length']
        w_coolant = (inputs['hose_diameter'] / 2) ** 2 * np.pi * inputs['rho_coolant'][0] * inputs['hose_length']
        outputs['component_weight'] = w_hose + w_coolant
    

    def compute_partials(self, inputs, J):
        nn = self.options['num_nodes']
        sigma = self.options['hose_operating_stress']
        rho_hose = self.options['hose_density']
        thickness = inputs['hose_diameter'] * inputs['hose_design_pressure'] / 2 / sigma

        d_thick_d_diam = inputs['hose_design_pressure'] / 2 / sigma
        d_thick_d_press = inputs['hose_diameter'] / 2 / sigma

        J['component_weight','rho_coolant'] = (inputs['hose_diameter'] / 2) ** 2 * np.pi * inputs['hose_length']
        J['component_weight', 'hose_design_pressure'] = (inputs['hose_diameter'] + thickness) * np.pi * d_thick_d_press * \
                                                        rho_hose * inputs['hose_length'] + np.pi * thickness * rho_hose * \
                                                        inputs['hose_length'] * d_thick_d_press
        J['component_weight', 'hose_length'] = (inputs['hose_diameter'] + thickness) * np.pi * thickness * rho_hose + \
                                               (inputs['hose_diameter'] / 2) ** 2 * np.pi * inputs['rho_coolant'][0]
        J['component_weight', 'hose_diameter'] = (inputs['hose_diameter'] + thickness) * np.pi * d_thick_d_diam * rho_hose * \
                                                 inputs['hose_length'] + (1 + d_thick_d_diam) * np.pi * thickness * rho_hose * \
                                                 inputs['hose_length'] + inputs['hose_diameter'] / 2 * np.pi * \
                                                 inputs['rho_coolant'][0] * inputs['hose_length']

        # use a colored complex step approach
        cs_step = 1e-30
        dp_base = self._compute_pressure_drop(inputs)

        cs_inp_list = ['rho_coolant', 'mdot_coolant', 'hose_diameter', 'hose_length', 'mu_coolant']
        fake_inputs = dict()
        # make a perturbable, complex copy of the inputs
        for inp in cs_inp_list:
            fake_inputs[inp] = inputs[inp].astype(np.complex_, copy=True)
        
        for inp in cs_inp_list:
            arr_to_restore = fake_inputs[inp].copy()
            fake_inputs[inp] += (0.0+cs_step*1.0j)
            dp_perturbed = self._compute_pressure_drop(fake_inputs)
            fake_inputs[inp] = arr_to_restore
            J['delta_p', inp] = np.imag(dp_perturbed) / cs_step


if __name__ == "__main__":
    ivg = om.IndepVarComp()
    ivg.add_output('mdot_coolant', 6.0, units='kg/s')
    ivg.add_output('hose_diameter', 0.033, units='m')
    ivg.add_output('rho_coolant', 1020., units='kg/m**3')
    ivg.add_output('hose_length', 20., units='m')
    ivg.add_output('power_rating', 4035., units='W')

    grp = om.Group()
    grp.add_subsystem('ivg', ivg, promotes=['*'])
    grp.add_subsystem('hose', SimpleHose(num_nodes=1), promotes_inputs=['*'], promotes_outputs=['delta_p'])
    grp.add_subsystem('pump', SimplePump(num_nodes=1), promotes_inputs=['*'])
    grp.add_subsystem('motorcool', MotorCoolingJacket(num_nodes=5))
    p = om.Problem(model=grp)
    p.setup(force_alloc_complex=True)

    p['motorcool.q_in'] = 50000
    p['motorcool.power_rating'] = 1e6
    p['motorcool.motor_weight'] = 1e6/5000
    p['motorcool.mdot_coolant'] = 0.1

    p.run_model()
    p.model.list_inputs(units=True, print_arrays=True)

    p.model.list_outputs(units=True, print_arrays=True)
    p.check_partials(compact_print=True, method='cs')
    print(p.get_val('delta_p', units='psi'))