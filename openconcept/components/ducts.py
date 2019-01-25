from __future__ import division
import numpy as np
from openmdao.api import ExplicitComponent, Problem, ImplicitComponent, NewtonSolver, DirectSolver, IndepVarComp
from openmdao.api import Group, ScipyOptimizeDriver

class TemperatureIsentropic(ExplicitComponent):
    """
    Compute static temperature via isentropic relation

    Inputs
    -------
    Tt : float
        Total temperature (vector, K)
    M : float
        Mach number (vector, dimensionless)

    Outputs
    -------
    T : float
        Static temperature  (vector, K)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    gamma : float
        Specific heat ratio (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')
        self.options.declare('gamma', default=1.4, desc='Specific heat ratio')

    def setup(self):
        nn = self.options['num_nodes']
        gam = self.options['gamma']
        self.add_input('Tt', shape=(nn,),  units='K')
        self.add_input('M', shape=(nn,))
        self.add_output('T', shape=(nn,),  units='K')
        self.declare_partials(['*'], ['*'], method='cs')

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        gam = self.options['gamma']
        outputs['T'] = inputs['Tt'] * (1 + (gam-1)/2 * inputs['M']**2) ** -1

class TotalTemperatureIsentropic(ExplicitComponent):
    """
    Compute total temperature via isentropic relation

    Inputs
    -------
    T : float
        Static temperature (vector, K)
    M : float
        Mach number (vector, dimensionless)

    Outputs
    -------
    Tt : float
        Static temperature  (vector, K)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    gamma : float
        Specific heat ratio (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')
        self.options.declare('gamma', default=1.4, desc='Specific heat ratio')

    def setup(self):
        nn = self.options['num_nodes']
        gam = self.options['gamma']
        self.add_input('T', shape=(nn,),  units='K')
        self.add_input('M', shape=(nn,))
        self.add_output('Tt', shape=(nn,),  units='K')
        self.declare_partials(['*'], ['*'], method='cs')

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        gam = self.options['gamma']
        outputs['Tt'] = inputs['T'] / (1 + (gam-1)/2 * inputs['M']**2) ** -1

class PressureIsentropic(ExplicitComponent):
    """
    Compute static pressure via isentropic relation

    Inputs
    -------
    pt : float
        Total pressure (vector, Pa)
    M : float
        Mach number (vector, dimensionless)

    Outputs
    -------
    p : float
        Static temperature  (vector, Pa)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    gamma : float
        Specific heat ratio (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')
        self.options.declare('gamma', default=1.4, desc='Specific heat ratio')

    def setup(self):
        nn = self.options['num_nodes']
        gam = self.options['gamma']
        self.add_input('pt', shape=(nn,),  units='Pa')
        self.add_input('M', shape=(nn,))
        self.add_output('p', shape=(nn,),  units='Pa')
        self.declare_partials(['*'], ['*'], method='cs')

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        gam = self.options['gamma']
        outputs['p'] = inputs['pt'] * (1 + (gam-1)/2 * inputs['M']**2) ** (- gam / (gam - 1))

class TotalPressureIsentropic(ExplicitComponent):
    """
    Compute total pressure via isentropic relation

    Inputs
    -------
    p : float
        Static pressure (vector, Pa)
    M : float
        Mach number (vector, dimensionless)

    Outputs
    -------
    pt : float
        Total pressure  (vector, Pa)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    gamma : float
        Specific heat ratio (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')
        self.options.declare('gamma', default=1.4, desc='Specific heat ratio')

    def setup(self):
        nn = self.options['num_nodes']
        gam = self.options['gamma']
        self.add_input('p', shape=(nn,),  units='Pa')
        self.add_input('M', shape=(nn,))
        self.add_output('pt', shape=(nn,),  units='Pa')
        self.declare_partials(['*'], ['*'], method='cs')

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        gam = self.options['gamma']
        outputs['pt'] = inputs['p'] / (1 + (gam-1)/2 * inputs['M']**2) ** (- gam / (gam - 1))

class DensityIdealGas(ExplicitComponent):
    """
    Compute density from ideal gas law

    Inputs
    -------
    p : float
        Static pressure (vector, Pa)
    T : float
        Static temperature (vector, K)

    Outputs
    -------
    rho : float
        Density  (vector, kg/m**3)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    R : float
        Gas constant (scalar, J / kg / K)

    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')
        self.options.declare('R', default=287.05, desc='Gas constant')

    def setup(self):
        nn = self.options['num_nodes']
        R = self.options['R']
        self.add_input('p', shape=(nn,),  units='Pa')
        self.add_input('T', shape=(nn,), units='K')
        self.add_output('rho', shape=(nn,),  units='kg/m**3')
        self.declare_partials(['*'], ['*'], method='cs')

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        R = self.options['R']
        outputs['rho'] = inputs['p'] / R / inputs['T']

class SpeedOfSound(ExplicitComponent):
    """
    Compute speed of sound

    Inputs
    -------
    T : float
        Static temperature (vector, K)

    Outputs
    -------
    a : float
        Speed of sound  (vector, m/s)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    R : float
        Gas constant (scalar, J / kg / K)
    gamma : float
        Specific heat ratio (scalar dimensionless)
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')
        self.options.declare('R', default=287.05, desc='Gas constant')
        self.options.declare('gamma', default=1.4, desc='Specific heat ratio')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('T', shape=(nn,), units='K')
        self.add_output('a', shape=(nn,),  units='m/s')
        self.declare_partials(['*'], ['*'], method='cs')

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        R = self.options['R']
        gam = self.options['gamma']
        outputs['a'] = np.sqrt(gam * R * inputs['T'])

class MachNumberfromSpeed(ExplicitComponent):
    """
    Compute Mach number from TAS and speed of sound

    Inputs
    -------
    Utrue : float
        True airspeed (vector, m/s)
    a : float
        Speed of sound (vector, m/s)

    Outputs
    -------
    M : float
        Mach number (vector, dimensionless)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('a', shape=(nn,), units='m/s')
        self.add_input('Utrue', shape=(nn,), units='m/s')
        self.add_output('M', shape=(nn,),  units='m/s')
        self.declare_partials(['*'], ['*'], method='cs')

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        outputs['M'] = inputs['Utrue'] / inputs['a']

class HeatAdditionPressureLoss(ExplicitComponent):
    """
    Adds / removes heat and pressure gain / loss

    Inputs
    -------
    Tt_in : float
        Total temperature in (vector, K)
    pt_in : float
        Total pressure in (vector, Pa)
    mdot : float
        Mass flow (vector, kg/s)
    delta_p : float
        Pressure gain / loss (vector, Pa)
    heat_in : float
        Heat addition (subtraction) rate (vector, W)
    cp : float
        Specific heat (scalar, J/kg/K)

    Outputs
    -------
    Tt_out : float
        Total temperature out  (vector, K)
    pt_out : float
        Total pressure out (vector, Pa)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('Tt_in', shape=(nn,), units='K')
        self.add_input('pt_in', shape=(nn,), units='Pa')
        self.add_input('mdot', shape=(nn,), units='kg/s')
        self.add_input('delta_p', shape=(nn,), units='Pa')
        self.add_input('heat_in', shape=(nn,), units='W')
        self.add_input('cp', units='J/kg/K')

        self.add_output('Tt_out', shape=(nn,), units='K')
        self.add_output('pt_out', shape=(nn,), units='Pa')

        self.declare_partials(['*'], ['*'], method='cs')

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        outputs['Tt_out'] = inputs['Tt_in'] + inputs['heat_in'] / inputs['cp'] / inputs['mdot']
        outputs['pt_out'] = inputs['pt_in'] + inputs['delta_p']

class MassFlow(ExplicitComponent):
    """
    Computes mass flow explicity from other parameters.
    Designed for use at the nozzle / min area point.

    Inputs
    ------
    M : float
        Mach number at this station (vector, dimensionless)
    rho : float
        Density at this station (vector, kg/m**3)
    area : float
        Flow cross sectional area at this station (vector, m**2)
    a : float
        Speed of sound (vector, m/s)

    Outputs
    -------
    mdot : float
        Mass flow rate (vector, kg/s)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('a', shape=(nn,), units='m/s')
        self.add_input('area', shape=(nn,), units='m**2')
        self.add_input('rho', shape=(nn,), units='kg/m**3')
        self.add_input('M', shape=(nn,))
        self.add_output('mdot', shape=(nn,), units='kg/s')
        arange = np.arange(0, nn)
        self.declare_partials(['mdot'], ['M', 'a', 'rho', 'area'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        outputs['mdot'] = inputs['M'] * inputs['a'] * inputs['area'] * inputs['rho']

    def compute_partials(self, inputs, J):
        J['mdot','M'] = inputs['a'] * inputs['area'] * inputs['rho']
        J['mdot','a'] = inputs['M'] * inputs['area'] * inputs['rho']
        J['mdot','area'] = inputs['M'] * inputs['a'] * inputs['rho']
        J['mdot','rho'] = inputs['M'] * inputs['a'] * inputs['area']

class MachNumberDuct(ImplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('mdot', shape=(nn,), units='kg/s')
        self.add_input('a', shape=(nn,), units='m/s')
        self.add_input('area', units='m**2')
        self.add_input('rho', shape=(nn,), units='kg/m**3')
        self.add_output('M', shape=(nn,), upper=1.0)
        arange = np.arange(0, nn)
        self.declare_partials(['M'], ['mdot'], rows=arange, cols=arange, val=np.ones((nn, )))
        self.declare_partials(['M'], ['M', 'a', 'rho'], rows=arange, cols=arange)
        self.declare_partials(['M'], ['area'], rows=arange, cols=np.zeros((nn, ), dtype=np.int32))

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['M'] = inputs['mdot'] - outputs['M'] * inputs['a'] * inputs['area'] * inputs['rho']

    def linearize(self, inputs, outputs, J):
        J['M','M'] = - inputs['a'] * inputs['area'] * inputs['rho']
        J['M','a'] = - outputs['M'] * inputs['area'] * inputs['rho']
        J['M','area'] = - outputs['M'] * inputs['a'] * inputs['rho']
        J['M','rho'] = - outputs['M'] * inputs['a'] * inputs['area']

class DuctExitMachNumber(ExplicitComponent):
    """
    Compute duct exit Mach number based on total pressure and ambient pressure

    Inputs
    -------
    p_exit : float
        Exit static pressure (vector, Pa)
    p_t : float
        Total pressure (vector, Pa)

    Outputs
    -------
    M : float
        Mach number (vector, Pa)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    gamma : float
        Specific heat ratio (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')
        self.options.declare('gamma', default=1.4, desc='Specific heat ratio')

    def setup(self):
        nn = self.options['num_nodes']
        gam = self.options['gamma']
        self.add_input('p_exit', shape=(nn,),  units='Pa')
        self.add_input('pt', shape=(nn,), units='Pa')
        self.add_output('M', shape=(nn,))
        self.declare_partials(['*'], ['*'], method='cs')

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        gam = self.options['gamma']
        pstar = inputs['pt'] * (2/(gam+1))**(gam/(gam-1))
        outputs['M'] = np.where(np.greater_equal(pstar, inputs['p_exit']), np.ones((nn,)), np.sqrt(((inputs['p_exit']/inputs['pt'])**((1-gam)/gam)-1)*2/(gam-1)))

class Inlet(Group):
    """This group takes in ambient flight conditions and computes total quantities for downstream use

    Inputs
    ------
    T : float
        Temperature (vector, K)
    p : float
        Ambient static pressure (vector, Pa)
    Utrue : float
        True airspeed (vector, m/s)

    Outputs
    -------
    Tt : float
        Total temperature (vector, K)
    pt : float
        Total pressure (vector, Pa)

    Options
    -------
    num_nodes : int
        Number of conditions to analyze
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_subsystem('speedsound',SpeedOfSound(num_nodes=nn), promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('mach',MachNumberfromSpeed(num_nodes=nn), promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('totaltemperature',TotalTemperatureIsentropic(num_nodes=nn), promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('totalpressure',TotalPressureIsentropic(num_nodes=nn), promotes_inputs=['*'], promotes_outputs=['*'])

class DuctStation(Group):
    """A 'normal' station in a duct flow.

    Inputs
    ------
    pt_in : float
        Upstream total pressure (vector, Pa)
    Tt_in : float
        Upstream total temperature (vector, K)
    mdot : float
        Mass flow (vector, kg/s)
    delta_p : float
        Pressure gain (loss) at this station (vector, Pa)
    heat_in : float
        Heat addition (loss) rate at this station (vector, W)

    Outputs
    -------
    pt_out : float
        Downstream total pressure (vector, Pa)
    Tt_out : float
        Downstream total temperature (vector, K)

    Options
    -------
    num_nodes : int
        Number of conditions to analyze
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_subsystem('totals', HeatAdditionPressureLoss(num_nodes=nn), promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('temp', TemperatureIsentropic(num_nodes=nn), promotes_inputs=['M'], promotes_outputs=['*'])
        self.connect('Tt_out','temp.Tt')
        self.add_subsystem('pressure', PressureIsentropic(num_nodes=nn), promotes_inputs=['M'], promotes_outputs=['*'])
        self.connect('pt_out','pressure.pt')
        self.add_subsystem('density', DensityIdealGas(num_nodes=nn), promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('speedsound', SpeedOfSound(num_nodes=nn), promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('mach', MachNumberDuct(num_nodes=nn), promotes_inputs=['*'], promotes_outputs=['*'])

class OutletNozzle(Group):
    """This group is designed to be the farthest downstream point in a ducted heat exchanger model.
       Mass flow is set based on the upstream total pressure and ambient static pressure.

    Inputs
    ------
    p_exit : float
        Exit static pressure. Normally set to ambient flight pressure (vector, Pa)
    pt : float
        Total pressure upstream of the nozzle (vector, Pa)
    Tt : float
        Total temperature upstream of the nozzle (vector, K)
    area : float
        Nozzle cross sectional area (vector, m**2)

    Outputs
    -------
    mdot : float
        Mass flow (vector, kg/s)

    Options
    -------
    num_nodes : int
        Number of conditions to analyze
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_subsystem('mach', DuctExitMachNumber(num_nodes=nn), promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('temp', TemperatureIsentropic(num_nodes=nn), promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('pressure', PressureIsentropic(num_nodes=nn), promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('density', DensityIdealGas(num_nodes=nn), promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('speedsound', SpeedOfSound(num_nodes=nn), promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('massflow', MassFlow(num_nodes=nn), promotes_inputs=['*'], promotes_outputs=['*'])

class DuctTestGroup(Group):
    """
    Test the 'pycycle_lite' functionality
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of analysis points' )

    def setup(self):
        nn = self.options['num_nodes']

        iv = self.add_subsystem('iv', IndepVarComp(), promotes_outputs=['cp'])
        iv.add_output('p_inf', val=37600.9*np.ones((nn,)), units='Pa')
        iv.add_output('T_inf', val=238.6*np.ones((nn,)), units='K')
        iv.add_output('Utrue', val=300.7*np.ones((nn,)), units='kn')
        iv.add_output('cp', val=1002.93, units='J/kg/K')

        iv.add_output('area_1', val=60, units='inch**2')
        iv.add_output('delta_p_1', val=np.zeros((nn,)), units='Pa')
        iv.add_output('heat_in_1', val=np.zeros((nn,)), units='W')

        iv.add_output('area_2', val=286.918, units='inch**2')
        iv.add_output('delta_p_2', val=np.ones((nn,))*0., units='Pa')
        iv.add_output('heat_in_2', val=np.ones((nn,))*0., units='W')

        iv.add_output('area_3', val=286.918, units='inch**2')
        iv.add_output('delta_p_3', val=np.ones((nn,))*-338.237, units='Pa')
        iv.add_output('heat_in_3', val=np.ones((nn,))*30408.8, units='W')

        iv.add_output('nozzle_area', val=58, units='inch**2')
        iv.add_output('p_exit', val=37600.9*np.ones((nn,)), units='Pa')


        self.add_subsystem('inlet', Inlet(num_nodes=nn))
        self.connect('iv.p_inf','inlet.p')
        self.connect('iv.T_inf','inlet.T')
        self.connect('iv.Utrue','inlet.Utrue')

        self.add_subsystem('sta1', DuctStation(num_nodes=nn), promotes_inputs=['mdot','cp'])
        self.connect('inlet.pt','sta1.pt_in')
        self.connect('inlet.Tt','sta1.Tt_in')
        self.connect('iv.area_1','sta1.area')
        self.connect('iv.delta_p_1','sta1.delta_p')
        self.connect('iv.heat_in_1','sta1.heat_in')

        self.add_subsystem('sta2', DuctStation(num_nodes=nn), promotes_inputs=['mdot','cp'])
        self.connect('sta1.pt_out','sta2.pt_in')
        self.connect('sta1.Tt_out','sta2.Tt_in')
        self.connect('iv.area_2','sta2.area')
        self.connect('iv.delta_p_2','sta2.delta_p')
        self.connect('iv.heat_in_2','sta2.heat_in')

        self.add_subsystem('sta3', DuctStation(num_nodes=nn), promotes_inputs=['mdot','cp'])
        self.connect('sta2.pt_out','sta3.pt_in')
        self.connect('sta2.Tt_out','sta3.Tt_in')
        self.connect('iv.area_3','sta3.area')
        self.connect('iv.delta_p_3','sta3.delta_p')
        self.connect('iv.heat_in_3','sta3.heat_in')

        self.add_subsystem('nozzle', OutletNozzle(num_nodes=nn), promotes_outputs=['mdot'])
        self.connect('iv.p_exit','nozzle.p_exit')
        self.connect('sta3.pt_out','nozzle.pt')
        self.connect('sta3.Tt_out','nozzle.Tt')
        self.connect('iv.nozzle_area','nozzle.area')

class DuctWithHx(Group):
    """
    Test the 'pycycle_lite' functionality with the heat exchanger
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of analysis points' )

    def setup(self):
        nn = self.options['num_nodes']

        iv = self.add_subsystem('iv', IndepVarComp(), promotes_outputs=['cp'])
        iv.add_output('p_inf', val=37600.9*np.ones((nn,)), units='Pa')
        iv.add_output('T_inf', val=238.6*np.ones((nn,)), units='K')
        iv.add_output('Utrue', val=np.linspace(35, 100, nn), units='kn')
        iv.add_output('cp', val=1002.93, units='J/kg/K')

        iv.add_output('area_1', val=60, units='inch**2')
        iv.add_output('delta_p_1', val=np.zeros((nn,)), units='Pa')
        iv.add_output('heat_in_1', val=np.zeros((nn,)), units='W')

        iv.add_output('area_2', val=286.918, units='inch**2')
        iv.add_output('delta_p_2', val=np.ones((nn,))*0., units='Pa')
        iv.add_output('heat_in_2', val=np.ones((nn,))*0., units='W')

        iv.add_output('area_3', val=286.918, units='inch**2')
        #iv.add_output('delta_p_3', val=np.ones((nn,))*-338.237, units='Pa')
        #iv.add_output('heat_in_3', val=np.ones((nn,))*30408.8, units='W')

        iv.add_output('nozzle_area', val=58*np.ones((nn,)), units='inch**2')
        iv.add_output('p_exit', val=37600.9*np.ones((nn,)), units='Pa')

        self.add_subsystem('inlet', Inlet(num_nodes=nn))
        self.connect('iv.p_inf','inlet.p')
        self.connect('iv.T_inf','inlet.T')
        self.connect('iv.Utrue','inlet.Utrue')

        self.add_subsystem('sta1', DuctStation(num_nodes=nn), promotes_inputs=['mdot','cp'])
        self.connect('inlet.pt','sta1.pt_in')
        self.connect('inlet.Tt','sta1.Tt_in')
        self.connect('iv.area_1','sta1.area')
        self.connect('iv.delta_p_1','sta1.delta_p')
        self.connect('iv.heat_in_1','sta1.heat_in')

        self.add_subsystem('sta2', DuctStation(num_nodes=nn), promotes_inputs=['mdot','cp'])
        self.connect('sta1.pt_out','sta2.pt_in')
        self.connect('sta1.Tt_out','sta2.Tt_in')
        self.connect('iv.area_2','sta2.area')
        self.connect('iv.delta_p_2','sta2.delta_p')
        self.connect('iv.heat_in_2','sta2.heat_in')

        self.add_subsystem('hx', HXTestGroup(num_nodes=nn))
        self.connect('mdot','hx.mdot_cold')
        self.connect('sta2.T','hx.T_in_cold')
        self.connect('sta2.rho','hx.rho_cold')
        self.connect('cp','hx.cp_cold')

        self.add_subsystem('sta3', DuctStation(num_nodes=nn), promotes_inputs=['mdot','cp'])
        self.connect('sta2.pt_out','sta3.pt_in')
        self.connect('sta2.Tt_out','sta3.Tt_in')
        self.connect('iv.area_3','sta3.area')
        self.connect('hx.delta_p_cold','sta3.delta_p')
        self.connect('hx.heat_transfer','sta3.heat_in')

        self.add_subsystem('nozzle', OutletNozzle(num_nodes=nn), promotes_outputs=['mdot'])
        self.connect('iv.p_exit','nozzle.p_exit')
        self.connect('sta3.pt_out','nozzle.pt')
        self.connect('sta3.Tt_out','nozzle.Tt')
        self.connect('iv.nozzle_area','nozzle.area')

if __name__ == '__main__':
    # run this script from the root openconcept directory like so:
    # python .\openconcept\components\ducts.py
    import sys, os
    sys.path.insert(0,os.getcwd())
    from openconcept.components.heat_exchanger import HXTestGroup
    nn=10
    prob = Problem(DuctWithHx(num_nodes=nn))
    prob.model.options['assembled_jac_type'] = 'csc'
    prob.model.nonlinear_solver=NewtonSolver()
    prob.model.linear_solver = DirectSolver(assemble_jac=True)
    prob.model.nonlinear_solver.options['solve_subsystems'] = True
    prob.model.nonlinear_solver.options['maxiter'] = 10
    prob.model.nonlinear_solver.options['atol'] = 1e-7
    prob.model.nonlinear_solver.options['rtol'] = 1e-7
    # prob.driver = ScipyOptimizeDriver()
    # prob.driver.options['tol'] = 1e-7

    # prob.model.add_design_var('channel_width_hot',lower=1,upper=20)
    # prob.model.add_constraint('T_out_hot', upper=55+273.15)
    # prob.model.add_objective('component_weight')

    prob.setup(check=True,force_alloc_complex=True)
    prob['sta1.M'] = 0.2*np.ones((nn,))
    prob['sta2.M'] = 0.2*np.ones((nn,))
    prob['sta3.M'] = 0.2*np.ones((nn,))
    prob.run_model()
    # prob.check_partials(method='cs', compact_print=True)
    # prob.run_driver()
    prob.model.list_inputs(units=True, print_arrays=True)
    prob.model.list_outputs(units=True,print_arrays=True)

#     Mach 0.5
#     p_inf = 5.454 psi
#