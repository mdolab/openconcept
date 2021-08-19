from __future__ import division
import numpy as np
import openmdao.api as om
from openconcept.utilities.math.integrals import Integrator
from openconcept.utilities.math.add_subtract_comp import AddSubtractComp

class SimpleBoilOff(om.ExplicitComponent):
    """
    Simplest possible model for boil-off. Boil-off
    mass flow rate equals Q/h where Q is heat entering
    liquid and h is latent heat of vaporization.

    Inputs
    ------
    heat_into_liquid : float
        Heat entering liquid propellant (vector, W)
    
    Outputs
    -------
    m_boil_off : float
        Mass flow rate of boil-off (vector, kg/s)
    
    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    h_vap : float
        Latent heat of vaporization of propellant, default hydrogen 446592 J/kg (scalar, J/kg)
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of design points to run')
        self.options.declare('h_vap', default=446592., desc='Latent heat of vaporization (J/kg)')
    
    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('heat_into_liquid', val=100., units='W', shape=(nn,))
        self.add_output('m_boil_off', val=0.1, units='kg/s', shape=(nn,))
        self.declare_partials('m_boil_off', 'heat_into_liquid', val=np.ones(nn)/self.options['h_vap'],
                              rows=np.arange(nn), cols=np.arange(nn))
    
    def compute(self, inputs, outputs):
        outputs['m_boil_off'] = inputs['heat_into_liquid'] / self.options['h_vap']


class LH2BoilOff(om.Group):
    """
    Models all relevant behavior of liquid and vapor in LH2 tank.

    Inputs
    ------
    radius : float
        Inner radius of the cylinder and hemispherical end caps (scalar, m)
    length : float
        Length of JUST THE CYLIDRICAL part of the tank (scalar, m)
    fill_level : float
        Fraction of tank (in range 0-1) filled with liquid propellant; assumes
        tank is oriented horizontally as shown above (vector, dimensionless)
    m_dot_liq : float
        Mass flow rate of liquid propellant being consumed;
        positive indicates vapor leaving the tank (vector, kg/s)
    Q_vap : float
        Heat into vapor from environment through walls (vector, W)
    Q_liq : float
        Heat into liquid from environment through walls (vector, W)
    design_pressure : float
        Maximum expected operating pressure (MEOP) (scalar, Pa)

    
    Outputs
    -------
    W_LH2 : float
        Mass of remaining liquid hydrogen (vector, kg)
    W_GH2 : float
        Mass of gaseous hydrogen in ullage (vector, kg)
    T_vap : float
        Temperature of vapor in ullage of tank (vector, K)
    T_liq : float
        Temperature of liquid hydrogen in tank (vector, K)
    P_vap : float
        Pressure of vapor in ullage of tank (vector, K)
    P_liq : float
        Pressure of liquid hydrogen in tank (vector, K)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    init_fill_level : float
        Initial fill level (in range 0-1) of the tank, default 0.95
        to leave space for gas expansion; this should never be higher
        than 0.99 or so since it's not enough space for gas expansion and
        the model behaves poorly with values very close to 1 (scalar, dimensionless)
    P_vap_frac_init : float
        Fraction of design pressure at initial state, default 0.7 (scalar, dimensionless)
    T_vap_init : float
        Initial temperature of vapor in tank, default 21 K just above saturation
        temperature at 1 atm (scalar, K)
    T_liq_init : float
        Initial temperature of liquid in tank, default 19 K just below saturation
        temperature at 1 atm (scalar, K)
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of design points to run')
        self.options.declare('init_fill_level', default=0.95, desc='Initial fill level')
        self.options.declare('P_vap_frac_init', default=.7, desc='Fraction of design pressure at start')
        self.options.declare('T_vap_init', default=21., desc='Initial H2 vapor temp (K)')
        self.options.declare('T_liq_init', default=19., desc='Initial LH2 temp (K)')
    
    def setup(self):
        nn = self.options['num_nodes']

        # Fluid properties in vapor, liquid, saturated and actual
        self.add_subsystem('GH2_prop', GH2Properties(num_nodes=nn))
        self.add_subsystem('LH2_prop', LH2Properties(num_nodes=nn))
        self.add_subsystem('H2_sat_prop', SaturatedH2Properties(num_nodes=nn))

        # Governing equations of boil-off
        self.add_subsystem('boil_off_ODE', LH2BoilOffODE(num_nodes=nn),
                           promotes_inputs=['radius', 'length', 'fill_level',
                                            'm_dot_liq', 'Q_vap', 'Q_liq'])

        integ = self.add_subsystem('integ', Integrator(num_nodes=nn, diff_units='s',
                                   time_setup='duration'), promotes_inputs=['m_dot_liq'], 
                                   promotes_outputs=['T_vap', 'T_liq'])
        integ.add_integrand('m_liq_to_vap', rate_name='m_liq_to_vap_dot', units='kg')
        integ.add_integrand('T_vap', rate_name='T_vap_dot', units='K', start_val=self.options['T_vap_init'])
        integ.add_integrand('T_liq', rate_name='T_liq_dot', units='K', start_val=self.options['T_liq_init'])
        integ.add_integrand('delta_V_vap', rate_name='V_vap_dot', units='m**3')
        integ.add_integrand('delta_V_liq', rate_name='V_liq_dot', units='m**3')
        integ.add_integrand('m_liq_used', rate_name='m_dot_liq', units='kg')
        integ.add_integrand('m_vap_vented', rate_name='m_dot_vent', units='kg')

        # Track weight of hydrogen in liquid and vapor
        self.add_subsystem('GH2_init', om.ExecComp('W_init = (4/3*pi*r**3 + pi*r**2*L)*(1-fill_init)*P*P_frac*MW/R/T',
                                                   W_init={'units': 'kg'},
                                                   r={'units': 'm'},
                                                   L={'units': 'm'},
                                                   fill_init={'value': self.options['init_fill_level']},
                                                   P={'units': 'Pa'},
                                                   P_frac={'value': self.options['P_vap_frac_init']},
                                                   MW={'units': 'kg/mol', 'value': 2.016e-3},  # molecular weight of H2
                                                   R={'units': 'J/(mol*K)', 'value': 8.314},  # gas constant
                                                   T={'units': 'K', 'value': self.options['T_vap_init']}),
                           promotes_inputs=[('r', 'radius'), ('L', 'length'), ('P', 'design_pressure')])
        self.add_subsystem('LH2_init', om.ExecComp('W_init = (4/3*pi*r**3 + pi*r**2*L)*fill_init*rho',
                                                   W_init={'units': 'kg'},
                                                   r={'units': 'm'},
                                                   L={'units': 'm'},
                                                   fill_init={'value': self.options['init_fill_level']},
                                                   rho={'units': 'kg/m**3', 'value': 70.85}),  # density of LH2
                           promotes_inputs=[('r', 'radius'), ('L', 'length')])
        self.add_subsystem('LH2_weight', AddSubtractComp(output_name='weight',
                                                         input_names=['W_LH2_init', 'W_LH2_used', 'W_to_vap'],
                                                          units='kg', vec_size=[1, nn, nn],
                                                          scaling_factors=[1, -1, -1]),
                           promotes_outputs=[('weight', 'W_LH2')])
        self.add_subsystem('GH2_weight', AddSubtractComp(output_name='weight',
                                                        input_names=['W_GH2_init', 'W_GH2_vented',
                                                                     'W_from_liq'],
                                                        units='kg', vec_size=[1, nn, nn],
                                                        scaling_factors=[1, -1, 1]),
                           promotes_outputs=[('weight', 'W_GH2')])
        self.connect('LH2_init.W_init', 'LH2_weight.W_LH2_init')
        self.connect('integ.m_liq_used', 'LH2_weight.W_LH2_used')
        self.connect('integ.m_liq_to_vap', 'LH2_weight.W_to_vap')
        self.connect('GH2_init.W_init', 'GH2_weight.W_GH2_init')
        self.connect('integ.m_vap_vented', 'GH2_weight.W_GH2_vented')
        self.connect('integ.m_liq_to_vap', 'GH2_weight.W_from_liq')

        # Volumes
        self.add_subsystem('liq_vol', om.ExecComp('V_liq = (4/3*pi*r**3 + pi*r**2*L)*fill_init + delta_V_liq',
                                                  V_liq={'units': 'm**3'},
                                                  r={'units': 'm'},
                                                  L={'units': 'm'},
                                                  fill_init={'value': self.options['init_fill_level']},
                                                  delta_V_liq={'units': 'm**3'}),
                           promotes_inputs=[('r', 'radius'), ('L', 'length')])
        self.add_subsystem('vap_vol', om.ExecComp('V_vap = 4/3*pi*r**3 + pi*r**2*L - V_liq',
                                                  V_vap={'units': 'm**3'},
                                                  V_liq={'units': 'm**3'},
                                                  r={'units': 'm'},
                                                  L={'units': 'm'}),
                           promotes_inputs=[('r', 'radius'), ('L', 'length')])
        self.connect('integ.delta_V_liq', 'liq_vol.delta_V_liq')
        self.connect('liq_vol.V_liq', 'vap_vol.V_liq')

        # TODO: add venting component

        # Connect ODEs to integrator
        self.connect('boil_off_ODE.m_dot', 'integ.m_liq_to_vap_dot')
        self.connect('boil_off_ODE.T_vap_dot', 'integ.T_vap_dot')
        self.connect('boil_off_ODE.T_liq_dot', 'integ.T_liq_dot')
        self.connect('boil_off_ODE.V_vap_dot', 'integ.V_vap_dot')
        self.connect('boil_off_ODE.V_liq_dot', 'integ.V_liq_dot')

        # Connect state variables back to fluid properties for ODE
        self.connect('T_vap', 'boil_off_ODE.T_vap')
        self.connect('T_liq', 'boil_off_ODE.T_liq')
        self.connect('liq_vol.V_liq', 'boil_off_ODE.V_liq')

        self.connect('W_GH2', 'GH2_prop.m_vap')
        self.connect('T_vap', 'GH2_prop.T_vap')
        self.connect('vap_vol.V_vap', 'GH2_prop.V_vap')

        self.connect('T_liq', 'LH2_prop.T_liq')

        self.connect('T_vap', 'H2_sat_prop.T_vap')
        self.connect('GH2_prop.P_vap', 'H2_sat_prop.P_vap')

        # Connect fluid property components to ODE component
        self.connect('GH2_prop.P_vap', 'boil_off_ODE.P_vap')
        self.connect('GH2_prop.c_v_vap', 'boil_off_ODE.c_v_vap')
        self.connect('GH2_prop.u_vap', 'boil_off_ODE.u_vap')
        self.connect('GH2_prop.h_vap', 'boil_off_ODE.h_vap')

        self.connect('LH2_prop.P_liq', 'boil_off_ODE.P_liq')
        self.connect('LH2_prop.rho_liq', 'boil_off_ODE.rho_liq')
        self.connect('LH2_prop.u_liq', 'boil_off_ODE.u_liq')
        self.connect('LH2_prop.h_liq', 'boil_off_ODE.h_liq')
        self.connect('LH2_prop.c_p_liq', 'boil_off_ODE.c_p_liq')

        self.connect('H2_sat_prop.T_sat', 'boil_off_ODE.T_sat')
        self.connect('H2_sat_prop.h_sat', 'boil_off_ODE.h_sat')
        self.connect('H2_sat_prop.h_vap_sat', 'boil_off_ODE.h_vap_sat')
        self.connect('H2_sat_prop.c_p_gsf', 'boil_off_ODE.c_p_gsf')
        self.connect('H2_sat_prop.mu_gsf', 'boil_off_ODE.mu_gsf')
        self.connect('H2_sat_prop.k_gsf', 'boil_off_ODE.k_gsf')
        self.connect('H2_sat_prop.beta_gsf', 'boil_off_ODE.beta_gsf')
        self.connect('H2_sat_prop.rho_gsf', 'boil_off_ODE.rho_gsf')

        # Set input default for same variable promoted from multiple sources
        self.set_input_defaults('m_dot_liq', 0, units='kg/s')


class LH2BoilOffODE(om.ExplicitComponent):
    """
    The ordinary differential equations that define the behavior
    of the liquid and vapor in the tank. These come from
    Chapter 4 of Eugina D. Mendez Ramos's dissertation, which
    can be found here: http://hdl.handle.net/1853/64797.

    Nearly all fluid properties are listed as inputs since they come
    from components that compute them via curve fits of data.

    Inputs
    ------
    # Usage parameters
    m_dot_vent : float
        Mass flow rate of vapor out of tank; used for venting;
        positive indicates vapor leaving the tank (vector, kg/s)
    m_dot_liq : float
        Mass flow rate of liquid propellant being consumed;
        positive indicates vapor leaving the tank (vector, kg/s)

    # Heat transfer from ullage to interface film
    T_vap : float
        Temperature of vapor in the ullage (vector, K)
    T_sat : float
        Saturation temperature at ullage pressure (vector, K)
    c_p_gsf : float
        Specific heat at constant pressure of vapor at mean film temperature,
        which is average of T_vap and saturation temperature (vector J/(kg-K))
    mu_gsf : float
        Viscosity of vapor at mean film temperature (vector, kg/(m-s))
    k_gsf : float
        Conductivity of the vapor at mean film temperature (vector, W/(m-K))
    beta_gsf : float
        Coefficient of thermal expansion of vapor at mean film temperature (vector, 1/K)
    rho_gsf : float
        Density of vapor at mean film temperature (vector, kg/m^3)
    radius : float
        Inner radius of the cylinder and hemispherical end caps (scalar, m)
    length : float
        Length of JUST THE CYLIDRICAL part of the tank (scalar, m)
    fill_level : float
        Fraction of tank (in range 0-1) filled with liquid propellant; assumes
        tank is oriented horizontally as shown above (vector, dimensionless)

    # Mass flow rate from interface to ullage
    T_liq : float
        Temperature of liquid (vector, K)
    c_p_liq : float
        Specific heat at constant pressure of liquid (vector, J/(kg-K))
    h_sat : float
        Enthalpy of saturated liquid (vector, J/kg)
    h_vap : float
        Enthalpy of vapor in ullage (vector, J/kg)
    h_vap_sat : float
        Enthalpy of saturated vapor (vector, J/kg)

    # Change in volume
    rho_liq : float
        Density of liquid (vector, kg/m^3)
    
    # Change in vapor temperature
    Q_vap : float
        Heat into vapor from environment through walls (vector, W)
    c_v_vap : float
        Specific heat at constant volume of vapor (vector, J/(kg-K))
    u_vap : float
        Internal energy in vapor (vector, J/kg)
    P_vap : float
        Pressure of vapor in ullage (vector, Pa)
    
    # Change in liquid temperature
    Q_liq : float
        Heat into liquid from environment through walls (vector, W)
    h_liq : float
        Enthalpy of liquid (vector, J/kg)
    V_liq : float
        Volume of liquid (vector, m^3)
    P_liq : float
        Pressure of liquid (vector, Pa)
    u_liq : float
        Internal energy in liquid (vector, J/kg)

    Outputs
    -------
    m_dot : float
        Mass flow rate from liquid to vapor (vector, kg/s)
    T_vap_dot : float
        Rate of change of vapor temperature (vector, K/s)
    T_liq_dot : float
        Rate of change of liquid temperature (vector, K/s)
    V_vap_dot : float
        Rate of change of vapor volume (vector, m^3/s)
    V_liq_dot
        Rate of change of liquid volume (vector, m^3/s)
    
    Options
    -------
    num_nodes : int
        Number of analysis points to run, default 1 (scalar, dimensionless)
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of design points to run')
    
    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('T_vap', val=50., units='K', shape=(nn,))
        self.add_input('T_sat', val=20., units='K', shape=(nn,))
        self.add_input('c_p_gsf', units='J/(kg*K)', shape=(nn,))
        self.add_input('mu_gsf', units='kg/(m*s)', shape=(nn,))
        self.add_input('k_gsf', units='W/(m*K)', shape=(nn,))
        self.add_input('beta_gsf', units='1/K', shape=(nn,))
        self.add_input('rho_gsf', units='kg/m**3', shape=(nn,))
        self.add_input('radius', units='m')
        self.add_input('length', units='m')
        self.add_input('fill_level', shape=(nn,))
        self.add_input('T_liq', val=50., units='K', shape=(nn,))
        self.add_input('c_p_liq', units='J/(kg*K)', shape=(nn,))
        self.add_input('h_sat', units='J/kg', shape=(nn,))
        self.add_input('h_vap', units='J/kg', shape=(nn,))
        self.add_input('h_vap_sat', units='J/kg', shape=(nn,))
        self.add_input('rho_liq', units='kg/m**3', shape=(nn,))
        self.add_input('Q_vap', val=100., units='W', shape=(nn,))
        self.add_input('c_v_vap', units='J/(kg*K)', shape=(nn,))
        self.add_input('u_vap', units='J/kg', shape=(nn,))
        self.add_input('P_vap', units='Pa', shape=(nn,))
        self.add_input('Q_liq', val=100., units='W', shape=(nn,))
        self.add_input('h_liq', units='J/kg', shape=(nn,))
        self.add_input('V_liq', units='m**3', shape=(nn,))
        self.add_input('P_liq', units='Pa', shape=(nn,))
        self.add_input('u_liq', units='J/kg', shape=(nn,))
        self.add_input('m_dot_vent', val=0., units='kg/s', shape=(nn,))
        self.add_input('m_dot_liq', units='kg/s', shape=(nn,))

        self.add_output('m_dot', units='kg/s', shape=(nn,))
        self.add_output('T_vap_dot', units='K/s', shape=(nn,))
        self.add_output('T_liq_dot', units='K/s', shape=(nn,))
        self.add_output('V_vap_dot', units='m**3/s', shape=(nn,))
        self.add_output('V_liq_dot', units='m**3/s', shape=(nn,))

        self.declare_partials(['*'], ['*'], method='cs')
    
    def compute(self, inputs, outputs):
        T_vap = inputs['T_vap']
        T_sat = inputs['T_sat']
        c_p_gsf = inputs['c_p_gsf']
        mu_gsf = inputs['mu_gsf']
        k_gsf = inputs['k_gsf']
        beta_gsf = inputs['beta_gsf']
        rho_gsf = inputs['rho_gsf']
        r = inputs['radius']
        L = inputs['length']
        T_liq = inputs['T_liq']
        c_p_liq = inputs['c_p_liq']
        h_sat = inputs['h_sat']
        h_vap = inputs['h_vap']
        h_vap_sat = inputs['h_vap_sat']
        rho_liq = inputs['rho_liq']
        Q_vap = inputs['Q_vap']
        Q_liq = inputs['Q_liq']
        c_v_vap = inputs['c_v_vap']
        u_vap = inputs['u_vap']
        P_vap = inputs['P_vap']
        h_liq = inputs['h_liq']
        V_liq = inputs['V_liq']
        P_liq = inputs['P_liq']
        u_liq = inputs['u_liq']
        m_dot_vent = inputs['m_dot_vent']
        m_dot_liq = inputs['m_dot_liq']
        fill_level = inputs['fill_level']

        # Compute the heat transfer from the ullage to interface film
        h_liquid = 2*r*fill_level  # linear approximation for the liquid height given fill level
        r_interface = np.sqrt(r**2 - (r - h_liquid)**2)  # radius of 2D surface of liquid
        A = 2 * r_interface * L + np.pi * r_interface**2
        L_ref = np.sqrt(A / np.pi)  # characteristic length of interface (radius assuming whole area is circular)
        g = 9.807  # local acceleration (m/s^2)
        Pr = c_p_gsf * mu_gsf / k_gsf
        Gr = g * beta_gsf * rho_gsf**2 * np.abs(T_vap - T_sat) * L_ref**3 /(mu_gsf**2)
        X = Pr * Gr
        Nu = 0.27 * X**0.25  # coefficients for top of cold horizontal surface
        h_c = k_gsf / L_ref * Nu  # heat transfer coeff, W/mˆ2-K
        Q_vs = h_c * A * (T_vap - T_sat)  # heat transfer from gas to interface

        m_dot = Q_vs / (c_p_liq * (T_sat - T_liq) + (h_vap - h_sat) + (h_vap - h_vap_sat))
        outputs['m_dot'] = m_dot

        V_vap_dot = m_dot / rho_liq
        V_liq_dot = -V_vap_dot
        outputs['V_vap_dot'] = V_vap_dot
        outputs['V_liq_dot'] = V_liq_dot

        outputs['T_vap_dot'] = (Q_vap - Q_vs + m_dot * h_vap - P_vap * V_vap_dot 
                                - m_dot_vent * u_vap) / (m_dot * c_v_vap)
        outputs['T_liq_dot'] = (Q_liq - m_dot * h_liq + P_liq * V_liq_dot - m_dot_liq * u_liq) \
                               / (rho_liq * V_liq * c_p_liq)


class GH2Properties(om.ExplicitComponent):
    """
    Computes necessary properties of gaseous hydrogen. Data for
    c_p and h come from pages 18 and 28 of
    https://www.bnl.gov/magnets/Staff/Gupta/cryogenic-data-handbook/Section3.pdf,
    respectively. The data is taken at 1 atmosphere, so we assume that the pressure
    is on the order of 1 atm, particularly when the vapor is at temperatures close
    to the saturation point. As the vapor becomes warmer, it more closely fits
    the ideal gas law assumption where its properties are a function only of
    temperature. Curve fits are done offline to fit to the data. Pressure, c_v,
    and internal energy are computed using the inputs, c_p, and
    h assuming an ideal gas.

    TODO: if we want to look into cryo-compressed hydrogen, this component
          should probably be changed for better accuracy at higher pressures;
          there are some models in the Mendez Ramos dissertation, but it is
          much more involved and I think some equations in this model would
          still rely on the ideal gas law

    Inputs
    ------
    m_vap : float
        Mass of vapor in ullage (vector, kg)
    V_vap : float
        Volume of ullage (vector, m^3)
    T_vap : float
        Temperature of vapor in ullage (vector, K)

    Outputs
    -------
    P_vap : float
        Pressure of vapor in ullage (vector, Pa)
    c_v_vap : float
        Specific heat at constant volume of vapor (vector, J/(kg-K))
    u_vap : float
        Internal energy in vapor (vector, J/kg)
    h_vap : float
        Enthalpy of vapor in ullage (vector, J/kg)
    
    Options
    -------
    num_nodes : int
        Number of analysis points to run, default 1 (scalar, dimensionless)
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of design points to run')
    
    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('m_vap', units='kg', shape=(nn,))
        self.add_input('V_vap', val=10., units='m**3', shape=(nn,))
        self.add_input('T_vap', val=100., units='K', shape=(nn,))

        self.add_output('P_vap', val=1e5, lower=0., units='Pa', shape=(nn,))
        self.add_output('c_v_vap', val=10e3, units='J/(kg*K)', shape=(nn,))
        self.add_output('u_vap', units='J/kg', shape=(nn,))
        self.add_output('h_vap', units='J/kg', shape=(nn,))

        self.declare_partials('P_vap', ['*'], rows=np.arange(nn), cols=np.arange(nn))
        self.declare_partials(['c_v_vap', 'u_vap', 'h_vap'], 'T_vap',
                              rows=np.arange(nn), cols=np.arange(nn))
    
    def compute(self, inputs, outputs):
        m = inputs['m_vap']
        V = inputs['V_vap']
        T = inputs['T_vap']

        MW_H2 = 2.016e-3  # molecular weight of hydrogen, kg/mol
        R = 8.314  # universal gas constant, J/(mol-K)
        
        P = m * T * R / (V * MW_H2)
        outputs['P_vap'] = P

        c_p = 5e-6 * T**4 - 0.0038 * T**3 + 0.9615 * T**2 - 73.365 * T + 12217
        outputs['c_v_vap'] = c_p - R/MW_H2
        outputs['h_vap'] = -0.0062 * T**3 + 11.93 * T**2 + 9311.6 * T + 534435

        outputs['u_vap'] = outputs['h_vap'] - R/MW_H2*T
    
    def compute_partials(self, inputs, J):
        m = inputs['m_vap']
        V = inputs['V_vap']
        T = inputs['T_vap']

        MW_H2 = 2.016e-3  # molecular weight of hydrogen, kg/mol
        R = 8.314  # universal gas constant, J/(mol-K)

        J['P_vap', 'm_vap'] = T * R / (V * MW_H2)
        J['P_vap', 'V_vap'] = - T * R / (V**2 * MW_H2)
        J['P_vap', 'T_vap'] = m * R / (V * MW_H2)

        J['c_v_vap', 'T_vap'] = 4*5e-6 * T**3 - 3*0.0038 * T**2 + 2*0.9615 * T - 73.365
        J['h_vap', 'T_vap'] = -3*0.0062 * T**2 + 2*11.93 * T + 9311.6
        J['u_vap', 'T_vap'] = J['h_vap', 'T_vap'] - R/MW_H2


class LH2Properties(om.ExplicitComponent):
    """
    Computes necessary properties of liquid hydrogen. Curve fits
    from http://hdl.handle.net/1853/64797, which are based on
    curve fits of NIST data done by neural networks.

    Inputs
    ------
    T_liq : float
        Temperature of liquid hydrogen (vector, K)

    Outputs
    -------
    P_liq : float
        Pressure of liquid (vector, Pa)
    rho_liq : float
        Density of liquid (vector, kg/m^3)
    u_liq : float
        Internal energy in liquid (vector, J/kg)
    h_liq : float
        Enthalpy of liquid (vector, J/kg)
    c_p_liq : float
        Specific heat at constant pressure of liquid (vector, J/(kg-K))
    
    Options
    -------
    num_nodes : int
        Number of analysis points to run, default 1 (scalar, dimensionless)
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of design points to run')
    
    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('T_liq', val=20., units='K', shape=(nn,))

        self.add_output('P_liq', val=1e5, lower=0., units='Pa', shape=(nn,))
        self.add_output('rho_liq', val=10., lower=0., units='kg/m**3', shape=(nn,))
        self.add_output('u_liq', units='J/kg', shape=(nn,))
        self.add_output('h_liq', units='J/kg', shape=(nn,))
        self.add_output('c_p_liq', val=10e3, units='J/(kg*K)', shape=(nn,))

        self.declare_partials(['*'], 'T_liq', rows=np.arange(nn), cols=np.arange(nn))
    
    def compute(self, inputs, outputs):
        T = inputs['T_liq']

        outputs['P_liq'] = 0.0138 * T**5.2644
        outputs['rho_liq'] = 115.53291 - 2.0067591*T - 0.1067411*(T-27.6691)**2 \
                             - 0.0085915*(T-27.6691)**3 - 0.0019879*(T-27.6691)**4 \
                             - 0.0003988*(T-27.6691)**5 - 2.7179e-5*(T-27.6691)**6
        outputs['u_liq'] = -334268 + 15183.043*T + 614.10133*(T-27.6691)**2 \
                           + 40.845478*(T-27.6691)**3 + 9.1394916*(T-27.6691)**4 \
                           + 1.8297788*(T-27.6691)**5 + 0.1246228*(T-27.6691)**6
        outputs['h_liq'] = -371985.2 + 16864.749*T + 893.59208*(T-27.6691)**2 \
                           + 103.63758*(T-27.6691)**3 + 7.756004*(T-27.6691)**4
        outputs['c_p_liq'] = 1/(0.0002684 - 7.6143e-6*T - 2.5759e-7*(T-27.6691)**2)
    
    def compute_partials(self, inputs, J):
        T = inputs['T_liq']

        J['P_liq', 'T_liq'] = 5.2644 * 0.0138 * T**(5.2644 - 1)
        J['rho_liq', 'T_liq'] = - 2.0067591 - 2*0.1067411*(T-27.6691) \
                                - 3*0.0085915*(T-27.6691)**2 - 4*0.0019879*(T-27.6691)**3 \
                                - 5*0.0003988*(T-27.6691)**4 - 6*2.7179e-5*(T-27.6691)**5
        J['u_liq', 'T_liq'] = 15183.043 + 2*614.10133*(T-27.6691) \
                              + 3*40.845478*(T-27.6691)**2 + 4*9.1394916*(T-27.6691)**3 \
                              + 5*1.8297788*(T-27.6691)**4 + 6*0.1246228*(T-27.6691)**5
        J['h_liq', 'T_liq'] = 16864.749 + 2*893.59208*(T-27.6691) \
                              + 3*103.63758*(T-27.6691)**2 + 4*7.756004*(T-27.6691)**3
        J['c_p_liq', 'T_liq'] = -1/(0.0002684 - 7.6143e-6*T - 2.5759e-7*(T-27.6691)**2)**2 * \
                                (7.6143e-6 - 2*2.5759e-7*(T-27.6691))


class SaturatedH2Properties(om.ExplicitComponent):
    """
    Computes necessary properties of saturated liquid and gaseous hydrogen.
    Curve fits from http://hdl.handle.net/1853/64797, which are based on
    curve fits of NIST data done by neural networks.

    Inputs
    ------
    T_vap : float
        Temperature of vapor in ullage (vector, K)
    P_vap : float
        Pressure of vapor in ullage (vector, Pa)

    Outputs
    -------
    T_sat : float
        Saturation temperature at ullage pressure (vector, K)
    h_sat : float
        Enthalpy of saturated liquid (vector, J/kg)
    h_vap_sat : float
        Enthalpy of saturated vapor (vector, J/kg)
    c_p_gsf : float
        Specific heat at constant pressure of vapor at mean film temperature,
        which is average of T_vap and saturation temperature (vector J/(kg-K))
    mu_gsf : float
        Viscosity of vapor at mean film temperature (vector, kg/(m-s))
    k_gsf : float
        Conductivity of the vapor at mean film temperature (vector, W/(m-K))
    beta_gsf : float
        Coefficient of thermal expansion of vapor at mean film temperature (vector, 1/K)
    rho_gsf : float
        Density of vapor at mean film temperature (vector, kg/m^3)

    Options
    -------
    num_nodes : int
        Number of analysis points to run, default 1 (scalar, dimensionless)
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of design points to run')
    
    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('T_vap', val=100., units='K', shape=(nn,))
        self.add_input('P_vap', val=1e5, units='Pa', shape=(nn,))

        self.add_output('T_sat', val=20., lower=0., units='K', shape=(nn,))
        self.add_output('h_sat', units='J/kg', shape=(nn,))
        self.add_output('h_vap_sat', units='J/kg', shape=(nn,))
        self.add_output('c_p_gsf', units='J/(kg*K)', shape=(nn,))
        self.add_output('mu_gsf', units='kg/(m*s)', shape=(nn,))
        self.add_output('k_gsf', units='W/(m*K)', shape=(nn,))
        self.add_output('beta_gsf', units='1/K', shape=(nn,))
        self.add_output('rho_gsf', units='kg/m**3', shape=(nn,))

        self.declare_partials(['*'], 'P_vap', rows=np.arange(nn), cols=np.arange(nn))
        self.declare_partials(['*_gsf'], 'T_vap', rows=np.arange(nn), cols=np.arange(nn))
    
    def compute(self, inputs, outputs):
        T = inputs['T_vap']
        P = inputs['P_vap']

        T_sat = 22.509518 + 9.5791e-6*P - 5.85e-12*(P-598825)**2 \
                + 3.292e-18*(P-598825)**3 - 1.246e-24*(P-598825)**4 \
                + 2.053e-29*(P-598825)**5 - 3.463e-35*(P-598825)**6
        outputs['T_sat'] = T_sat
        outputs['h_sat'] = -371985.2 + 16864.749*T_sat + 893.59208*(T_sat-27.6691)**2 \
                           + 103.63758*(T_sat-27.6691)**3 + 7.756004*(T_sat-27.6691)**4
        outputs['h_vap_sat'] = 577302.07 - 4284.432*T_sat - 1084.1238*(T_sat-27.6691)**2 \
                               - 73.011186*(T_sat-27.6691)**3 - 15.407809*(T_sat-27.6691)**4 \
                               - 2.9987887*(T_sat-27.6691)**5 - 0.2022147*(T_sat-27.6691)**6


        T = (T + T_sat) / 2  # switch temperature to mean film temperature

        outputs['rho_gsf'] = -28.97599 + 1.2864736*T + 0.1140157*(T-27.6691)**2 \
                             + 0.0086723*(T-27.6691)**3 + 0.0019006*(T-27.6691)**4 \
                             + 0.0003805*(T-27.6691)**5 + 2.5918e-5*(T-27.6691)**6
        outputs['c_p_gsf'] = np.exp(6.445199 + 0.1249361*T + 0.0125811*(T-27.6691)**2
                                    + 0.0027137*(T-27.6691)**3 + 0.0006249*(T-27.6691)**4
                                    + 4.8352e-5*(T-27.6691)**5)
        outputs['k_gsf'] = 1/(110.21937 - 2.6596443*T - 0.0153377*(T-27.6691)**2
                              - 0.0088632*(T-27.6691)**3)
        outputs['mu_gsf'] = 1/(1582670.2 - 34545.242*T - 211.73722*(T-27.6691)**2
                               - 283.70972*(T-27.6691)**3 - 18.848797*(T-27.6691)**4)
        outputs['beta_gsf'] = 1 / T
    
    def compute_partials(self, inputs, J):
        T = inputs['T_vap']
        P = inputs['P_vap']

        T_sat = 22.509518 + 9.5791e-6*P - 5.85e-12*(P-598825)**2 \
                + 3.292e-18*(P-598825)**3 - 1.246e-24*(P-598825)**4 \
                + 2.053e-29*(P-598825)**5 - 3.463e-35*(P-598825)**6

        J['T_sat', 'P_vap'] = 9.5791e-6 - 2*5.85e-12*(P-598825) \
                              + 3*3.292e-18*(P-598825)**2 - 4*1.246e-24*(P-598825)**3 \
                              + 5*2.053e-29*(P-598825)**4 - 6*3.463e-35*(P-598825)**5
        J['h_sat', 'P_vap'] = J['T_sat', 'P_vap'] * (16864.749 + 2*893.59208*(T_sat-27.6691)
                              + 3*103.63758*(T_sat-27.6691)**2 + 4*7.756004*(T_sat-27.6691)**3)
        J['h_vap_sat', 'P_vap'] = J['T_sat', 'P_vap'] * (-4284.432 - 2*1084.1238*(T_sat-27.6691)
                                  - 3*73.011186*(T_sat-27.6691)**2 - 4*15.407809*(T_sat-27.6691)**3
                                  - 5*2.9987887*(T_sat-27.6691)**4 - 6*0.2022147*(T_sat-27.6691)**5)

        d_T_d_T_vap = 1/2
        d_T_d_P_vap = 1/2 * J['T_sat', 'P_vap']

        T = (T + T_sat) / 2  # switch temperature to mean film temperature

        d_rho_d_T = 1.2864736 + 2*0.1140157*(T-27.6691) \
                    + 3*0.0086723*(T-27.6691)**2 + 4*0.0019006*(T-27.6691)**3 \
                    + 5*0.0003805*(T-27.6691)**4 + 6*2.5918e-5*(T-27.6691)**5
        d_c_p_d_T = np.exp(6.445199 + 0.1249361*T + 0.0125811*(T-27.6691)**2
                           + 0.0027137*(T-27.6691)**3 + 0.0006249*(T-27.6691)**4
                           + 4.8352e-5*(T-27.6691)**5) \
                    * (0.1249361 + 2*0.0125811*(T-27.6691) + 3*0.0027137*(T-27.6691)**2 + 
                       4*0.0006249*(T-27.6691)**3 + 5*4.8352e-5*(T-27.6691)**4)
        d_k_d_T = -1/(110.21937 - 2.6596443*T - 0.0153377*(T-27.6691)**2 - 0.0088632*(T-27.6691)**3)**2 * \
                  (-2.6596443 - 2*0.0153377*(T-27.6691) - 3*0.0088632*(T-27.6691)**2)
        d_mu_d_T = -1/(1582670.2 - 34545.242*T - 211.73722*(T-27.6691)**2
                       - 283.70972*(T-27.6691)**3 - 18.848797*(T-27.6691)**4)**2 \
                   * (-34545.242 - 2*211.73722*(T-27.6691) - 3*283.70972*(T-27.6691)**2
                      - 4*18.848797*(T-27.6691)**3)
        d_beta_d_T = -1/T**2
        
        J['rho_gsf', 'T_vap'] = d_rho_d_T * d_T_d_T_vap
        J['rho_gsf', 'P_vap'] = d_rho_d_T * d_T_d_P_vap
        J['c_p_gsf', 'T_vap'] = d_c_p_d_T * d_T_d_T_vap
        J['c_p_gsf', 'P_vap'] = d_c_p_d_T * d_T_d_P_vap
        J['k_gsf', 'T_vap'] = d_k_d_T * d_T_d_T_vap
        J['k_gsf', 'P_vap'] = d_k_d_T * d_T_d_P_vap
        J['mu_gsf', 'T_vap'] = d_mu_d_T * d_T_d_T_vap
        J['mu_gsf', 'P_vap'] = d_mu_d_T * d_T_d_P_vap
        J['beta_gsf', 'T_vap'] = d_beta_d_T * d_T_d_T_vap
        J['beta_gsf', 'P_vap'] = d_beta_d_T * d_T_d_P_vap
