from __future__ import division
from openconcept.analysis.atmospherics.temperature_comp import TemperatureComp
from openconcept.analysis.atmospherics.pressure_comp import PressureComp
from openconcept.analysis.atmospherics.density_comp import DensityComp
from openconcept.analysis.atmospherics.dynamic_pressure_comp import DynamicPressureComp
from openconcept.analysis.atmospherics.true_airspeed import TrueAirspeedComp, EquivalentAirspeedComp
from openconcept.analysis.atmospherics.speedofsound_comp import SpeedOfSoundComp
from openconcept.analysis.atmospherics.mach_number_comp import MachNumberComp
import numpy as np
from openmdao.api import ExplicitComponent, Group, Problem, IndepVarComp


class ComputeAtmosphericProperties(Group):
    '''
    Computes pressure, density, temperature, dyn pressure, and true airspeed

    Inputs
    ------
    fltcond|h : float
        Altitude (vector, km)
    fltcond|Ueas : float
        Equivalent airspeed (vector, m/s)

    Outputs
    -------
    fltcond|p : float
        Pressure (vector, Pa)
    fltcond|rho : float
        Density (vector, kg/m3)
    fltcond|T : float
        Temperature (vector, K)
    fltcond|Utrue : float
        True airspeed (vector, m/s)
    fltcond|q : float
        Dynamic pressure (vector, Pa)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length) (default 1)
    true_airspeed_in : bool
        Flip to true if input vector is Utrue, not Ueas.
        If this is true, fltcond|Utrue will be an input and fltcond|Ueas will be an output.
    '''

    def initialize(self):
        self.options.declare('num_nodes',default=1,desc="Number of mission analysis points to run")
        self.options.declare('true_airspeed_in',default=False,desc="Number of mission analysis points to run")

    def setup(self):
        nn = self.options['num_nodes']
        tas_in = self.options['true_airspeed_in']
        self.add_subsystem('inputconv', InputConverter(num_nodes=nn),promotes_inputs=['*'])
        self.add_subsystem('temp', TemperatureComp(num_nodes=nn))
        self.add_subsystem('pressure',PressureComp(num_nodes=nn))
        self.add_subsystem('density',DensityComp(num_nodes=nn))
        self.add_subsystem('speedofsound',SpeedOfSoundComp(num_nodes=nn))
        self.add_subsystem('outputconv',OutputConverter(num_nodes=nn),promotes_outputs=['*'])
        if tas_in:
            self.add_subsystem('equivair',EquivalentAirspeedComp(num_nodes=nn),promotes_inputs=['*'],promotes_outputs=['*'])
        else:
            self.add_subsystem('trueair',TrueAirspeedComp(num_nodes=nn),promotes_inputs=['*'],promotes_outputs=['*'])
        self.add_subsystem('dynamicpressure',DynamicPressureComp(num_nodes=nn),promotes_inputs=["*"],promotes_outputs=["*"])
        self.add_subsystem('machnumber',MachNumberComp(num_nodes=nn),promotes_inputs=["*"],promotes_outputs=["*"])

        self.connect('inputconv.h_km','temp.h_km')
        self.connect('inputconv.h_km','pressure.h_km')
        self.connect('pressure.p_MPa','density.p_MPa')
        self.connect('temp.T_1e2_K',['density.T_1e2_K','speedofsound.T_1e2_K'])
        self.connect('pressure.p_MPa','outputconv.p_MPa')
        self.connect('temp.T_1e2_K','outputconv.T_1e2_K')
        self.connect('speedofsound.a_1e2_ms','outputconv.a_1e2_ms')
        self.connect('density.rho_kg_m3','outputconv.rho_kg_m3')


class InputConverter(ExplicitComponent):
    """
    This component adds a unitized interface to the Hwang and Jasa model.

    Inputs
    ------
    fltcond|h : float
        Altitude (vector, km)

    Outputs
    -------
    h_km : float
        Altitude in km to pass to the standard atmosphere modules (vector, unitless)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length) (default 1)
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')
    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('fltcond|h', units='km', desc='Flight condition altitude',shape=(nn,))
        #outputs and partials
        self.add_output('h_km', desc='Height in kilometers with no units',shape=(nn,))
        self.declare_partials('h_km','fltcond|h', rows=range(nn), cols=range(nn))

    def compute(self, inputs, outputs):
        outputs['h_km'] = inputs['fltcond|h']

    def compute_partials(self, inputs, J):
        nn = self.options['num_nodes']
        J['h_km','fltcond|h'] = np.ones(nn)


class OutputConverter(ExplicitComponent):
    """
    This component adds a unitized interface to the Hwang and Jasa model.

    Inputs
    ------
    p_MPa : float
        Pressure in megapascals from the standard atm model (vector, unitless)
    T_1e2_K : float
        Tempreature in 100K units from the std atm model (vector, unitless)
    rho_kg_m3 : float
        Density in kg / m3 from the std atm model (vector, unitless)

    Outputs
    -------
    fltcond|p : float
        Pressure with units (vector, Pa)
    fltcond|rho : float
        Density with units (vector, kg/m3)
    fltcond|T : float
        Temperature with units (vector, K)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length) (default 1)
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')
    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('p_MPa', desc='Flight condition pressures',shape=(nn,))
        self.add_input('T_1e2_K', desc='Flight condition temp',shape=(nn,))
        self.add_input('rho_kg_m3', desc='Flight condition density',shape=(nn,))
        self.add_input('a_1e2_ms', desc='Flight condition speed of sound',shape=(nn,))

        #outputs and partials
        self.add_output('fltcond|p', units='Pa', desc='Flight condition pressure with units',shape=(nn,))
        self.add_output('fltcond|rho', units='kg * m**-3', desc='Flight condition density with units',shape=(nn,))
        self.add_output('fltcond|T', units='K', desc='Flight condition temp with units',shape=(nn,))
        self.add_output('fltcond|a', units='m * s**-1', desc='Flight condition speed of sound with units',shape=(nn,))


        self.declare_partials(['fltcond|p'], ['p_MPa'], rows=range(nn), cols=range(nn), val=1e6*np.ones(nn))
        self.declare_partials(['fltcond|rho'], ['rho_kg_m3'], rows=range(nn), cols=range(nn), val=np.ones(nn))
        self.declare_partials(['fltcond|T'], ['T_1e2_K'], rows=range(nn), cols=range(nn), val=100*np.ones(nn))
        self.declare_partials(['fltcond|a'], ['a_1e2_ms'], rows=range(nn), cols=range(nn), val=100*np.ones(nn))

    def compute(self, inputs, outputs):
        outputs['fltcond|p'] = inputs['p_MPa'] * 1e6
        outputs['fltcond|rho'] = inputs['rho_kg_m3']
        outputs['fltcond|T'] = inputs['T_1e2_K'] * 100
        outputs['fltcond|a'] = inputs['a_1e2_ms'] * 100
