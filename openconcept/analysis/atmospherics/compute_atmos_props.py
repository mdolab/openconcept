from openconcept.analysis.atmospherics.temperature_comp import TemperatureComp
from openconcept.analysis.atmospherics.pressure_comp import PressureComp
from openconcept.analysis.atmospherics.density_comp import DensityComp
from openconcept.analysis.atmospherics.dynamic_pressure_comp import DynamicPressureComp
from openconcept.analysis.atmospherics.true_airspeed import TrueAirspeedComp

import numpy as np
from openmdao.api import ExplicitComponent, Group, Problem, IndepVarComp

class InputConverter(ExplicitComponent):
    """
    The differentiable standard atmosphere from Hwang and Jasa is unitless and modular.
    This model adds a unitized interface to other higher-level model interfaces.
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')
    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('fltcond|h', units='km', desc='Flight condition altitude',shape=(nn,))
        self.add_input('fltcond|Ueas', units='m/s', desc='Flight condition airspeed (equivalent)', shape=(nn,))

        #outputs and partials
        self.add_output('h_km', desc='Height in kilometers with no units',shape=(nn,))
        self.add_output('v_m_s', desc='Equivalent airspeed in m/s with no units',shape=(nn,))
        self.declare_partials('h_km','fltcond|h', rows=range(nn), cols=range(nn))
        self.declare_partials('v_m_s','fltcond|Ueas', rows=range(nn), cols=range(nn))
    def compute(self, inputs, outputs):
        outputs['h_km'] = inputs['fltcond|h']
        outputs['v_m_s'] = inputs['fltcond|Ueas']
    def compute_partials(self, inputs, J):
        nn = self.options['num_nodes']
        J['h_km','fltcond|h'] = np.ones(nn)
        J['v_m_s','fltcond|Ueas'] = np.ones(nn)

class OutputConverter(ExplicitComponent):
    """
    The differentiable standard atmosphere from Hwang and Jasa is unitless and modular.
    This model adds a unitized interface to other higher-level model interfaces.
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')
    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('p_MPa', desc='Flight condition pressures',shape=(nn,))
        self.add_input('T_1e2_K', desc='Flight condition temp',shape=(nn,))
        self.add_input('rho_kg_m3', desc='Flight condition density',shape=(nn,))
        #self.add_input('q_1e4_N_m2', desc='Flight condition dynamic pressure',shape=(nn,))

        #outputs and partials
        self.add_output('fltcond|p', units='Pa', desc='Flight condition pressure with units',shape=(nn,))
        self.add_output('fltcond|rho', units='kg * m**-3', desc='Flight condition density with units',shape=(nn,))
        self.add_output('fltcond|T', units='K', desc='Flight condition temp with units',shape=(nn,))
        #self.add_output('fltcond|q', units='Pa', desc='Flight condition dynamic pressure with units',shape=(nn,))

        self.declare_partials(['fltcond|p'], ['p_MPa'], rows=range(nn), cols=range(nn))
        self.declare_partials(['fltcond|rho'], ['rho_kg_m3'], rows=range(nn), cols=range(nn))
        self.declare_partials(['fltcond|T'], ['T_1e2_K'], rows=range(nn), cols=range(nn))
        #self.declare_partials(['fltcond|q'], ['q_1e4_N_m2'], rows=range(nn), cols=range(nn))


    def compute(self, inputs, outputs):
        outputs['fltcond|p'] = inputs['p_MPa'] * 1e6
        outputs['fltcond|rho'] = inputs['rho_kg_m3']
        outputs['fltcond|T'] = inputs['T_1e2_K'] * 100
        #outputs['fltcond|q'] = inputs['q_1e4_N_m2'] * 1e4
    def compute_partials(self, inputs, J):
        nn = self.options['num_nodes']
        J['fltcond|p','p_MPa'] = 1e6*np.ones(nn)
        J['fltcond|T','T_1e2_K'] = 100*np.ones(nn)
        J['fltcond|rho','rho_kg_m3'] = np.ones(nn)
        #J['fltcond|q','q_1e4_N_m2'] = 1e4*np.ones(nn)


class ComputeAtmosphericProperties(Group):
    """This computes pressure, temperature, and density for a given altitude at ISA condtions. Also true airspeed from equivalent ~ indicated airspeed
    """
    def initialize(self):
        self.options.declare('num_nodes',default=1,desc="Number of mission analysis points to run")

    def setup(self):
        nn = self.options['num_nodes']
        self.add_subsystem('inputconv', InputConverter(num_nodes=nn),promotes_inputs=['*'])
        self.add_subsystem('temp', TemperatureComp(num_nodes=nn))
        self.add_subsystem('pressure',PressureComp(num_nodes=nn))
        self.add_subsystem('density',DensityComp(num_nodes=nn))
        self.add_subsystem('outputconv',OutputConverter(num_nodes=nn),promotes_outputs=['*'])
        self.add_subsystem('trueairspeed',TrueAirspeedComp(num_nodes=nn),promotes_inputs=['*'],promotes_outputs=['*'])
        self.add_subsystem('dynamicpressure',DynamicPressureComp(num_nodes=nn),promotes_inputs=["*"],promotes_outputs=["*"])

        self.connect('inputconv.h_km','temp.h_km')
        self.connect('inputconv.h_km','pressure.h_km')
        self.connect('pressure.p_MPa','density.p_MPa')
        self.connect('temp.T_1e2_K','density.T_1e2_K')
        self.connect('pressure.p_MPa','outputconv.p_MPa')
        self.connect('temp.T_1e2_K','outputconv.T_1e2_K')
        self.connect('density.rho_kg_m3','outputconv.rho_kg_m3')



class TestModel(Group):
    def setup(self):
        dvs = self.add_subsystem('alts',IndepVarComp(),promotes_outputs=["*"])
        fltconds = self.add_subsystem('stdatm',ComputeAtmosphericProperties(num_nodes=10),promotes_inputs=["fltcond|*"])
        dvs.add_output('fltcond|h',np.linspace(0,28000,10), units='ft')
        dvs.add_output('fltcond|Ueas',np.ones(10)*150, units='kn')

# class TestUnitConv(ExplicitComponent):
#     def setup(self):
#         self.add_input('fltcond|Ueas',unit='m/s',desc='Equiv airspeed')
#         self.add_output('fltcond|Utrue',unit='m/s'desc='True airspeed')
#     def compute(self, inputs, outputs):
#         outputs['fltcond|Utrue'] = inputs['fltcond|Ueas']


def testfunc():
    prob = Problem()
    prob.model= TestModel()
    prob.setup()
    prob.run_model()
    print('Altitude: ' + str(prob['stdatm.inputconv.h_km']))
    print('Temp: ' + str(prob['stdatm.fltcond|T']))
    print('Pressure: ' + str(prob['stdatm.fltcond|p']))
    print('Density: ' + str(prob['stdatm.fltcond|rho']))
    print('TAS: ' + str(prob['stdatm.fltcond|Utrue']))
    print('Dynamic pressure:' + str(prob['stdatm.fltcond|q']))
    #prob.model.list_inputs()
    #prob.model.list_outputs()
    prob.check_partials(compact_print=True)

if __name__ == "__main__":
    from openconcept.analysis.atmospherics.compute_atmos_props import testfunc
    testfunc()
