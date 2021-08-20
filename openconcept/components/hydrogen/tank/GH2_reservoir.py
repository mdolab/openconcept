from __future__ import division
import numpy as np
import openmdao.api as om
from openconcept.utilities.math.integrals import Integrator
from openconcept.utilities.math.add_subtract_comp import AddSubtractComp

class GH2Reservoir(om.Group):
    """
    Reservoir of gaseous hydrogen. This could be used to model
    the ullage of an LH2 tank since it is capable of variable volume
    or it could model the entire interior of a GH2 tank.

    Inputs
    ------
    V : float
        Volume of the tank; could be scalar or vector depending
        on vector_V option, scalar by default (scalar/vector, m^3)
    V_dot : float
        Rate of change of the volume of the tank, default 0 (vector, m^3/s)
    m : float
        Mass of GH2 in tank (vector, kg)
    m_dot_out : float
        Flow rate of GH2 out of the tank; positive m_dot_out
        is H2 leaving the tank, default 0 (vector, kg/s)
    m_dot_in : float
        Flow rate of GH2 into the tank; positive m_dot_in
        is H2 entering the tank, default 0 (vector, kg/s)
    T_in : float
        Temperature of GH2 entering the tank, default 21 K (vector, K)
    Q_dot : float
        Rate of heat entering gaseous hydrogen, default 0 (vector, W)
    
    Outputs
    -------
    T : float
        Temperature of GH2 in the tank (vector, K)
    P : float
        Pressure of GH2 in the tank (vector, Pa)
    
    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    T_init : float
        Initial temperature of the gaseous hydrogen, default 25 K (scalar, K)
    vector_V : bool
        Whether to have the volume be a vector (True)
        or scalar (False) input, default False
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of design points to run')
        self.options.declare('T_init', default=25, desc='Initial temp of GH2 (scalar, K)')
        self.options.declare('vector_V', default=False, desc='If volume is vector input')
    
    def setup(self):
        nn = self.options['num_nodes']

        integ = self.add_subsystem('integ', Integrator(num_nodes=nn, diff_units='s',
                                   time_setup='duration'), promotes_inputs=['duration'],
                                   promotes_outputs=['T'])
        integ.add_integrand('T', rate_name='T_dot', units='K', start_val=self.options['T_init'])

        self.add_subsystem('GH2_ODE', GH2ReservoirODE(num_nodes=nn, vector_V=self.options['vector_V']),
                           promotes_inputs=['V', 'V_dot', 'T', 'm', 'm_dot_out',
                                            'm_dot_in', 'T_in', 'Q_dot'],
                           promotes_outputs=['P'])
        self.connect('GH2_ODE.T_dot', 'integ.T_dot')

class GH2ReservoirODE(om.ExplicitComponent):
    """
    State equation and pressure calculation for the reservoir. The
    approach is heavily based on the ullage control volume formulation
    from Eugina Mendez Ramos' dissertation (section 4.1.2.1) found here
    http://hdl.handle.net/1853/64797.

    Inputs
    ------
    V : float
        Volume of the tank; could be scalar or vector depending
        on vector_V option, scalar by default (scalar/vector, m^3)
    V_dot : float
        Rate of change of the volume of the tank, default 0 (vector, m^3/s)
    T : float
        Temperature of GH2 in the tank (vector, K)
    m : float
        Mass of GH2 in tank (vector, kg)
    m_dot_out : float
        Flow rate of GH2 out of the tank; positive m_dot_out
        is H2 leaving the tank, default 0 (vector, kg/s)
    m_dot_in : float
        Flow rate of GH2 into the tank; positive m_dot_in
        is H2 entering the tank, default 0 (vector, kg/s)
    T_in : float
        Temperature of GH2 entering the tank, default 21 K (vector, K)
    Q_dot : float
        Rate of heat entering gaseous hydrogen, default 0 (vector, W)
    
    Outputs
    -------
    T_dot : float
        Temperature of GH2 in tank (vector, K/s)
    P : float
        Pressure of GH2 in tank (vector, Pa)
    
    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    vector_V : bool
        Whether to have the volume be a vector (True)
        or scalar (False) input, default False
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of design points to run')
        self.options.declare('vector_V', default=False, desc='If volume is vector input')
    
    def setup(self):
        nn = self.options['num_nodes']

        if self.options['vector_V']:
            self.add_input('V', units='m**3', shape=(nn,))
        else:
            self.add_input('V', units='m**3')
        self.add_input('V_dot', val=0., units='m**3/s', shape=(nn,))
        self.add_input('T', val=200, units='K', shape=(nn,))
        self.add_input('m', units='kg', shape=(nn,))
        self.add_input('m_dot_out', val=0., units='kg/s', shape=(nn,))
        self.add_input('m_dot_in', val=0., units='kg/s', shape=(nn,))
        self.add_input('T_in', val=21., units='K', shape=(nn,))  # default just above boiling
                                                                  # point of H2 at 1 atm
        self.add_input('Q_dot', val=0., units='W', shape=(nn,))

        self.add_output('T_dot', units='K/s', shape=(nn,))
        self.add_output('P', units='Pa', shape=(nn,))

        if self.options['vector_V']:
            self.declare_partials('T_dot', 'V', rows=np.arange(nn), cols=np.arange(nn))
            self.declare_partials('P', 'V', rows=np.arange(nn), cols=np.arange(nn))
        else:
            self.declare_partials('T_dot', 'V', rows=np.arange(nn), cols=np.zeros(nn))
            self.declare_partials('P', 'V', rows=np.arange(nn), cols=np.zeros(nn))
        
        self.declare_partials('T_dot', ['V_dot', 'T', 'm', 'm_dot_out',
                              'm_dot_in', 'T_in', 'Q_dot'], rows=np.arange(nn), cols=np.arange(nn))
        self.declare_partials('P', ['m', 'T'], rows=np.arange(nn), cols=np.arange(nn))
    
    def compute(self, inputs, outputs):
        V = inputs['V']
        V_dot = inputs['V_dot']
        T = inputs['T']
        m = inputs['m']
        m_dot_out = inputs['m_dot_out']
        m_dot_in = inputs['m_dot_in']
        T_in = inputs['T_in']
        Q_dot = inputs['Q_dot']

        MW_H2 = 2.016e-3  # molecular weight of hydrogen, kg/mol
        R = 8.314  # universal gas constant, J/(mol-K)

        # c_p and h are from curve fits of data from pages 18 and 28 of
        # https://www.bnl.gov/magnets/Staff/Gupta/cryogenic-data-handbook/Section3.pdf
        # NOTE: they were taken with data at 1 atm so for compressed hydrogen,
        #       a better model may be necessary to get accurate results
        c_p = 5e-6 * T**4 - 0.0038 * T**3 + 0.9615 * T**2 - 73.365 * T + 12217
        c_v = c_p - R/MW_H2
        h_in = -0.0062 * T_in**3 + 11.93 * T_in**2 + 9311.6 * T_in + 534435
        h_out = -0.0062 * T**3 + 11.93 * T**2 + 9311.6 * T + 534435

        P = m * R/MW_H2 * T / V
        outputs['P'] = P

        # Derived from first law of thermodynamics (see Eugina Mendez
        # Ramos' dissertation section 4.1.2.1 for details)
        outputs['T_dot'] = (Q_dot - P * V_dot + m_dot_in * (h_in - c_v*T)
                            - m_dot_out * (h_out - c_v*T)) / (m * c_v)
    
    def compute_partials(self, inputs, J):
        V = inputs['V']
        V_dot = inputs['V_dot']
        T = inputs['T']
        m = inputs['m']
        m_dot_out = inputs['m_dot_out']
        m_dot_in = inputs['m_dot_in']
        T_in = inputs['T_in']
        Q_dot = inputs['Q_dot']

        MW_H2 = 2.016e-3  # molecular weight of hydrogen, kg/mol
        R = 8.314  # universal gas constant, J/(mol-K)

        c_p = 5e-6 * T**4 - 0.0038 * T**3 + 0.9615 * T**2 - 73.365 * T + 12217
        c_v = c_p - R/MW_H2
        d_cv_d_T = 4 * 5e-6 * T**3 - 3 * 0.0038 * T**2 + 2 * 0.9615 * T - 73.365
        h_in = -0.0062 * T_in**3 + 11.93 * T_in**2 + 9311.6 * T_in + 534435
        d_h_in_d_T_in = -3 * 0.0062 * T_in**2 + 2 * 11.93 * T_in + 9311.6
        h_out = -0.0062 * T**3 + 11.93 * T**2 + 9311.6 * T + 534435
        d_h_out_d_T = -3 * 0.0062 * T**2 + 2 * 11.93 * T + 9311.6

        P = m * R/MW_H2 * T / V

        J['P', 'V'] = -m * R/MW_H2 * T / V**2
        J['P', 'm'] = R/MW_H2 * T / V
        J['P', 'T'] = m * R/MW_H2 / V

        J['T_dot', 'V'] = -J['P', 'V'] * V_dot / (m * c_v)
        J['T_dot', 'V_dot'] = -P / (m * c_v)
        J['T_dot', 'T'] = (-J['P', 'T'] * V_dot - m_dot_in * (c_v + d_cv_d_T * T)
                           - m_dot_out * (d_h_out_d_T - c_v - d_cv_d_T * T)) / (m * c_v) \
                          - (Q_dot - P * V_dot + m_dot_in * (h_in - c_v*T) - m_dot_out * (h_out - c_v*T)) \
                          / (m * c_v)**2 * m * d_cv_d_T
        J['T_dot', 'm'] = -J['P', 'm'] * V_dot / (m * c_v) - (Q_dot - P * V_dot + m_dot_in * (h_in - c_v*T)
                            - m_dot_out * (h_out - c_v*T)) / (m * c_v)**2 * c_v
        J['T_dot', 'm_dot_out'] = -(h_out - c_v*T) / (m * c_v)
        J['T_dot', 'm_dot_in'] = (h_in - c_v*T) / (m * c_v)
        J['T_dot', 'T_in'] = m_dot_in * d_h_in_d_T_in / (m * c_v)
        J['T_dot', 'Q_dot'] = 1 / (m * c_v)