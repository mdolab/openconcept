import openmdao.api as om 
import numpy as np 
from openconcept.utilities.math.integrals import Integrator
import warnings

class BandolierCoolingSystem(om.ExplicitComponent):
    """
    Computes battery heat transfer for a parameteric battery
    based on Tesla's Model 3 design.

    Assumptions:
    - Heat generated uniformly in the cell
    - Weight per cell and thermal resistance stay constant
    even as specific energy varies parametrically
    (this means that cell count is constant with pack WEIGHT,
    not pack ENERGY as technology improves)
    - Cylindrical cells attached to Tesla-style thermal ribbon
    - Liquid cooling
    - Heat transfer through axial direction only (not baseplate)
    - 2170 cells (21 mm diameter, 70mm tall)
    - Battery thermal model assumes unsteady cell temperature,
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
        Number of cells long per "bandolier" actual count is 2x (scalar, default 21, Tesla)
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
        self.add_input('n_cpb', units=None, val=21.)
        self.add_input('t_channel', units='m', val=0.0005)

        self.add_output('dTdt', shape=(nn,), units='K/s', tags=['integrate', 'state_name:T_battery', 'state_units:K', 'state_val:300.0', 'state_promotes:True'])
        self.add_output('T_surface', shape=(nn,), units='K')
        self.add_output('T_core', shape=(nn,), units='K')
        self.add_output('q', shape=(nn,), units='W')
        self.add_output('T_out', shape=(nn,), units='K')

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

        if np.sum(np.abs(qcheck - outputs['q']/n_cells)) > 1e-5:
            # the heat flux across the cell is not equal to the heat flux due to convection
            raise ValueError('The surface temperature solution appears to be wrong')

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
        Number of cells long per "bandolier" actual count is 2x (scalar, default 21, Tesla)
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
            ode_integ.add_integrand('T', rate_name='dTdt', units='K', lower=0.0)
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
    # TODO reg tests
    
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
        self.add_output('T_out', shape=(nn,), units='K')
        self.add_output('dTdt', shape=(nn,), units='K/s', tags=['integrate', 'state_name:T_motor', 'state_units:K', 'state_val:300.0', 'state_promotes:True'])        
        
        self.declare_partials(['T_out','q','dTdt'], ['power_rating'], rows=arange, cols=np.zeros((nn,)))
        self.declare_partials(['dTdt'], ['motor_weight'], rows=arange, cols=np.zeros((nn,)))

        self.declare_partials(['T_out','q','dTdt'], ['T_in', 'T'], rows=arange, cols=arange)
        self.declare_partials(['T_out'], ['mdot_coolant'], rows=arange, cols=arange)
        self.declare_partials(['dTdt'], ['q_in'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        const = self.options['case_cooling_coefficient'] / self.options['case_area_coefficient']
        outputs['q'] = (inputs['T'] - inputs['T_in']) * const * inputs['power_rating']
        outputs['T_out'] = inputs['T_in'] + (inputs['T'] - inputs['T_in']) * const * inputs['power_rating'] / inputs['mdot_coolant'] / self.options['coolant_specific_heat']
        outputs['dTdt'] = (inputs['q_in'] - outputs['q']) / inputs['motor_weight'] / self.options['motor_specific_heat']
        
        if np.count_nonzero((inputs['T'] - outputs['T_out']) < 0):
            warnings.warn(self.msginfo + ' Motor sink coolant outlet temperature is hotter than the object itself (physically impossible).'
                'This may resolve after the solver converges, but should be double checked.', stacklevel=2)

    def compute_partials(self, inputs, J):
        nn = self.options['num_nodes']
        const = self.options['case_cooling_coefficient'] / self.options['case_area_coefficient']
        q = (inputs['T'] - inputs['T_in']) * const * inputs['power_rating']
        J['q', 'T'] = const * inputs['power_rating']
        J['q', 'T_in'] = - const * inputs['power_rating']
        J['q', 'power_rating'] = (inputs['T'] - inputs['T_in']) * const 
        
        J['T_out', 'T'] = const * inputs['power_rating'] / inputs['mdot_coolant'] / self.options['coolant_specific_heat']
        J['T_out', 'T_in'] = np.ones((nn,)) - const * inputs['power_rating'] / inputs['mdot_coolant'] / self.options['coolant_specific_heat']
        J['T_out', 'power_rating'] = (inputs['T'] - inputs['T_in']) * const / inputs['mdot_coolant'] / self.options['coolant_specific_heat']
        J['T_out', 'mdot_coolant'] = - (inputs['T'] - inputs['T_in']) * const * inputs['power_rating'] / inputs['mdot_coolant'] ** 2 / self.options['coolant_specific_heat']

        J['dTdt', 'q_in'] = 1 / inputs['motor_weight'] / self.options['motor_specific_heat']
        J['dTdt', 'T'] = - const * inputs['power_rating'] / inputs['motor_weight'] / self.options['motor_specific_heat']
        J['dTdt', 'T_in'] = const * inputs['power_rating'] / inputs['motor_weight'] / self.options['motor_specific_heat']
        J['dTdt', 'power_rating'] = - (inputs['T'] - inputs['T_in']) * const / inputs['motor_weight'] / self.options['motor_specific_heat']
        J['dTdt', 'motor_weight'] =  - (inputs['q_in'] - q) / inputs['motor_weight'] ** 2 / self.options['motor_specific_heat']

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
    """

    def initialize(self):
        self.options.declare('motor_specific_heat', default=921.0, desc='Specific heat in J/kg/K')
        self.options.declare('coolant_specific_heat', default=3801, desc='Specific heat in J/kg/K')
        self.options.declare('quasi_steady', default=False, desc='Treat the component as quasi-steady or with thermal mass')
        self.options.declare('num_nodes', default=1, desc='Number of quasi-steady points to runs')

    def setup(self):
        nn = self.options['num_nodes']
        quasi_steady = self.options['quasi_steady']
        self.add_subsystem('hex',
                           MotorCoolingJacket(num_nodes=nn, coolant_specific_heat=self.options['coolant_specific_heat'],
                                              motor_specific_heat=self.options['motor_specific_heat']),
                           promotes_inputs=['q_in','T_in', 'T','power_rating','mdot_coolant','motor_weight'],
                           promotes_outputs=['T_out', 'dTdt'])
        if not quasi_steady:
            ode_integ = self.add_subsystem('ode_integ', Integrator(num_nodes=nn, diff_units='s', method='simpson', time_setup='duration'),
                                           promotes_outputs=['*'], promotes_inputs=['*'])
            ode_integ.add_integrand('T', rate_name='dTdt', units='K', lower=0.0)
        else:
            self.add_subsystem('thermal_bal',
                               om.BalanceComp('T', eq_units='K/s', lhs_name='dTdt', rhs_val=0.0, units='K', lower=1.0, val=299.*np.ones((nn,))),
                               promotes_inputs=['dTdt'],
                               promotes_outputs=['T'])
