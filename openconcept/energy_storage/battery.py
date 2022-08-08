from __future__ import division
import numpy as np
from openmdao.api import ExplicitComponent, Group, BalanceComp
from openconcept.utilities.math.multiply_divide_comp import ElementMultiplyDivideComp
from openconcept.utilities.math.integrals import Integrator
from openconcept.utilities.dvlabel import DVLabel
from openconcept.thermal import BandolierCoolingSystem

class SOCBattery(Group):
    """
    Same as SimpleBattery but also tracks state of charge

    Inputs
    ------
    battery_weight : float
        Weight of the battery pack (scalar, kg)
    elec_load: float
        Electric power draw upstream (vector, W)
    SOC_initial : float
        Initial state of charge (default 1) (scalar, dimensionless)
    duration : float
        Length of the mission phase (corresponding to num_nodes) (scalar, s)

    Outputs
    -------
    SOC : float
        State of charge of the battery on a scale of 0 to 1 (vector, dimensionless)
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
    default_specific_energy : float
        Battery energy per unit weight **NOTE UNITS** (default 300, !!!! Wh/kg)
        Can be set using variable input 'specific_energy' as well if doing a sweep
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

        eta_b = self.options['efficiency']
        e_b = self.options['specific_energy']
        p_b = self.options['specific_power']
        cost_inc = self.options['cost_inc']
        cost_base = self.options['cost_base']

        # defaults = [['SOC_initial', 'batt_SOC_initial', 1, None]]
        # self.add_subsystem('defaults', DVLabel(defaults),
        #                    promotes_inputs=["*"], promotes_outputs=["*"])

        self.add_subsystem('batt_base',SimpleBattery(num_nodes=nn, efficiency=eta_b, specific_energy=e_b,
                                                     specific_power=p_b, cost_inc=cost_inc, cost_base=cost_base),
                           promotes_outputs=['*'],promotes_inputs=['*'])


        # change in SOC over time is (- elec_load) / max_energy

        self.add_subsystem('divider',ElementMultiplyDivideComp(output_name='dSOCdt',input_names=['elec_load','max_energy'],vec_size=[nn,1],scaling_factor=-1,divide=[False,True],input_units=['W','kJ']),
                           promotes_inputs=['*'],promotes_outputs=['*'])

        integ = self.add_subsystem('ode_integ', Integrator(num_nodes=nn, method='simpson', diff_units='s', time_setup='duration'), promotes_inputs=['*'], promotes_outputs=['*'])
        integ.add_integrand('SOC', rate_name='dSOCdt', start_name='SOC_initial', end_name='SOC_final', units=None, val=1.0, start_val=1.0)


class LiquidCooledBattery(Group):
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
                               BalanceComp('T', eq_units='K/s', lhs_name='dTdt', rhs_val=0.0, units='K', lower=1.0, val=299.*np.ones((nn,))),
                               promotes_inputs=['dTdt'],
                               promotes_outputs=['T'])


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
        Can override this with variable input during a sweep (input specific_energy)
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
        e_b = self.options['specific_energy']
        self.add_input('specific_energy', units='W * h / kg', val=e_b)
        eta_b = self.options['efficiency']
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
        self.declare_partials('max_energy', ['battery_weight','specific_energy'])

    def compute(self, inputs, outputs):
        eta_b = self.options['efficiency']
        p_b = self.options['specific_power']
        e_b = inputs['specific_energy']
        cost_inc = self.options['cost_inc']
        cost_base = self.options['cost_base']

        outputs['heat_out'] = inputs['elec_load'] * (1 - eta_b)
        outputs['component_cost'] = inputs['battery_weight'] * cost_inc + cost_base
        outputs['component_sizing_margin'] = inputs['elec_load'] / (p_b * inputs['battery_weight'])
        outputs['max_energy'] = inputs['battery_weight'] * e_b

    def compute_partials(self, inputs, J):
        eta_b = self.options['efficiency']
        p_b = self.options['specific_power']
        e_b = inputs['specific_energy']
        J['component_sizing_margin', 'elec_load'] = 1 / (p_b * inputs['battery_weight'])
        J['component_sizing_margin', 'battery_weight'] = - (inputs['elec_load'] /
                                                            (p_b * inputs['battery_weight'] ** 2))
        J['max_energy','battery_weight'] = e_b
        J['max_energy', 'specific_energy'] = inputs['battery_weight']