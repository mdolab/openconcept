import openmdao.api as om
import numpy as np

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

    .. note::
        See the ``LiquidCooledBattery`` for a group that already integrates
        this component with a battery.

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
