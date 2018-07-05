"""Analysis routines for simulating a mission profile with climb, cruise, and descent"""

from openmdao.api import Problem, Group, IndepVarComp, BalanceComp
from openmdao.api import DirectSolver, NewtonSolver, ScipyKrylov
from openmdao.api import ScipyOptimizeDriver, ExplicitComponent, ImplicitComponent
from openmdao.api import BalanceComp, ArmijoGoldsteinLS, NonlinearBlockGS
import numpy as np
import scipy.sparse as sp
from openconcept.analysis.atmospherics.compute_atmos_props import ComputeAtmosphericProperties
from openconcept.utilities.math.simpson_integration import simpson_integral, simpson_partials
from openconcept.utilities.math.simpson_integration import simpson_integral_every_node
from openconcept.utilities.math.simpson_integration import simpson_partials_every_node
from openconcept.utilities.math.sum_comp import SumComp
from openconcept.analysis.aerodynamics import PolarDrag
from openconcept.utilities.dvlabel import DVLabel


class MissionFlightConditions(ExplicitComponent):
    """
    Generates vectors of flight conditions for a mission profile

    Inputs
    ------
    mission|climb|vs : float
        Vertical speed in the climb segment (scalar, m/s)
    mission|descent|vs: float
        Vertical speed in the descent segment (should be neg; scalar, m/s)
    mission|climb|Ueas : float
        Indicated/equiv. airspeed during climb (scalar, m/s)
    mission|cruise|Ueas : float
        Indicated/equiv. airspeed in cruise (scalar, m/s)
    mission|descent|Ueas : float
        Indicated/equiv. airspeed during descent (scalar, m/s)
    mission|takeoff|h : float
        Takeoff (and landing, for now) altitude (scalar, m)
    mission|cruise|h : float
        Cruise altitude (scalar, m)

    Outputs
    -------
    fltcond|mission|vs : float
        Vertical speed vector for all mission phases / analysis points (vector, m/s)
    fltcond|mission|Ueas : float
        Equivalent airspeed vector for all mission phases / analysis points (vector, m/s)
    fltcond|mission|h : float
        Altitude at each analysis point (vector, m)
    mission|climb|time : float
        Time to ascent from end of takeoff to start of cruise (scalar, s)
    mission|descent|time : float
        Time to descend from end of cruise to landing (scalar, s)
    mission|climb|dt : float
        Timestep length during climb phase (scalar, s)
        Note: this represents the timestep for the Simpson subinterval, not the whole inteval
    mission|descent|dt : float
        Timestep length during descent phase (scalar, s)
        Note: this represents the timestep for the Simpson subinterval, not the whole inteval

    Options
    -------
    n_int_per_seg : int
        Number of Simpson's rule intervals to use per mission segment.
        The total number of points is 2 * n_int_per_seg + 1
    """

    def initialize(self):

        self.options.declare('n_int_per_seg', default=5,
                             desc="Number of Simpson intervals to use per seg")

    def setup(self):
        n_int_per_seg = self.options['n_int_per_seg']
        nn = (n_int_per_seg * 2 + 1)
        self.add_input('mission|climb|vs', val=5,
                       units='m / s', desc='Vertical speed in the climb segment')
        self.add_input('mission|descent|vs', val=-2.5,
                       units='m / s', desc='Vertical speed in the descent segment (should be neg)')
        self.add_input('mission|climb|Ueas', val=90,
                       units='m / s', desc='Indicated airspeed during climb')
        self.add_input('mission|cruise|Ueas', val=100,
                       units='m / s', desc='Cruise airspeed (indicated)')
        self.add_input('mission|descent|Ueas', val=80,
                       units='m / s', desc='Descent airspeed (indicated)')
        self.add_input('mission|takeoff|h', val=0,
                       units='m', desc='Airport altitude')
        self.add_input('mission|cruise|h', val=8000,
                       units='m', desc='Cruise altitude')

        self.add_output('fltcond|mission|Ueas', units='m / s',
                        desc='indicated airspeed at each analysis point', shape=(3 * nn,))
        self.add_output('fltcond|mission|h', units='m',
                        desc='altitude at each analysis point', shape=(3 * nn,))
        self.add_output('fltcond|mission|vs', units='m / s',
                        desc='vectorial representation of vertical speed', shape=(3 * nn,))
        self.add_output('mission|climb|time', units='s',
                        desc='Time from ground level to cruise')
        self.add_output('mission|descent|time', units='s',
                        desc='Time to descend to ground from cruise')
        self.add_output('mission|climb|dt', units='s',
                        desc='Timestep in climb phase')
        self.add_output('mission|descent|dt', units='s',
                        desc='Timestep in descent phase')

        # the climb speeds only have influence over their respective mission segments
        self.declare_partials(['fltcond|mission|Ueas'], ['mission|climb|Ueas'],
                              rows=np.arange(0, nn), cols=np.ones(nn) * 0, val=np.ones(nn))
        self.declare_partials(['fltcond|mission|Ueas'], ['mission|cruise|Ueas'],
                              rows=np.arange(nn, 2 * nn), cols=np.ones(nn) * 0, val=np.ones(nn))
        self.declare_partials(['fltcond|mission|Ueas'], ['mission|descent|Ueas'],
                              rows=np.arange(2 * nn, 3 * nn), cols=np.ones(nn) * 0, val=np.ones(nn))
        hcruisepartials = np.concatenate([np.linspace(0.0, 1.0, nn),
                                          np.ones(nn),
                                          np.linspace(1.0, 0.0, nn)])
        hgroundpartials = np.concatenate([np.linspace(1.0, 0.0, nn),
                                          np.linspace(0.0, 1.0, nn)])
        # the influence of each parameter linearly varies from 0 to 1 and vice versa on
        # climb and descent. The partials are different lengths on purpose - no influence
        # of ground on the mid-mission points,  so no partial derivative
        self.declare_partials(['fltcond|mission|h'], ['mission|cruise|h'],
                              rows=range(3 * nn), cols=np.zeros(3 * nn), val=hcruisepartials)
        self.declare_partials(['fltcond|mission|h'], ['mission|takeoff|h'],
                              rows=np.concatenate([np.arange(0, nn), np.arange(2 * nn, 3 * nn)]),
                              cols=np.zeros(2 * nn), val=hgroundpartials)
        self.declare_partials(['fltcond|mission|vs'], ['mission|climb|vs'],
                              rows=range(nn), cols=np.zeros(nn), val=np.ones(nn))
        self.declare_partials(['fltcond|mission|vs'], ['mission|descent|vs'],
                              rows=np.arange(2 * nn, 3 * nn), cols=np.zeros(nn), val=np.ones(nn))
        self.declare_partials(['mission|climb|time'],
                              ['mission|takeoff|h', 'mission|cruise|h', 'mission|climb|vs'])
        self.declare_partials(['mission|descent|time'],
                              ['mission|takeoff|h', 'mission|cruise|h', 'mission|descent|vs'])
        self.declare_partials(['mission|climb|dt'],
                              ['mission|takeoff|h', 'mission|cruise|h', 'mission|climb|vs'])
        self.declare_partials(['mission|descent|dt'],
                              ['mission|takeoff|h', 'mission|cruise|h', 'mission|descent|vs'])

    def compute(self, inputs, outputs):

        n_int_per_seg = self.options['n_int_per_seg']
        nn = n_int_per_seg*2 + 1
        hvec_climb = np.linspace(inputs['mission|takeoff|h'],inputs['mission|cruise|h'],nn)
        hvec_desc = np.linspace(inputs['mission|cruise|h'],inputs['mission|takeoff|h'],nn)
        hvec_cruise = np.ones(nn)*inputs['mission|cruise|h']
        outputs['fltcond|mission|h'] = np.concatenate([hvec_climb,hvec_cruise,hvec_desc])
        debug = np.concatenate([np.ones(nn)*inputs['mission|climb|Ueas'],np.ones(nn)*inputs['mission|cruise|Ueas'],np.ones(nn)*inputs['mission|descent|Ueas']])
        outputs['fltcond|mission|Ueas'] = np.concatenate([np.ones(nn)*inputs['mission|climb|Ueas'],np.ones(nn)*inputs['mission|cruise|Ueas'],np.ones(nn)*inputs['mission|descent|Ueas']])
        outputs['fltcond|mission|vs'] = np.concatenate([np.ones(nn)*inputs['mission|climb|vs'],np.ones(nn)*0.0,np.ones(nn)*inputs['mission|descent|vs']])
        outputs['mission|climb|time'] = (inputs['mission|cruise|h']-inputs['mission|takeoff|h'])/inputs['mission|climb|vs']
        outputs['mission|descent|time'] = (inputs['mission|takeoff|h']-inputs['mission|cruise|h'])/inputs['mission|descent|vs']
        outputs['mission|climb|dt'] = (inputs['mission|cruise|h']-inputs['mission|takeoff|h'])/inputs['mission|climb|vs']/(nn-1)
        outputs['mission|descent|dt'] =  (inputs['mission|takeoff|h']-inputs['mission|cruise|h'])/inputs['mission|descent|vs']/(nn-1)

    def compute_partials(self, inputs, J):

        n_int_per_seg = self.options['n_int_per_seg']
        nn = n_int_per_seg*2 + 1
        J['mission|climb|time','mission|cruise|h'] = 1/inputs['mission|climb|vs']
        J['mission|climb|time','mission|takeoff|h'] = -1/inputs['mission|climb|vs']
        J['mission|climb|time','mission|climb|vs'] = -(inputs['mission|cruise|h']-inputs['mission|takeoff|h'])/(inputs['mission|climb|vs']**2)
        J['mission|descent|time','mission|cruise|h'] = -1/inputs['mission|descent|vs']
        J['mission|descent|time','mission|takeoff|h'] = 1/inputs['mission|descent|vs']
        J['mission|descent|time','mission|descent|vs'] = -(inputs['mission|takeoff|h']-inputs['mission|cruise|h'])/(inputs['mission|descent|vs']**2)
        J['mission|climb|dt','mission|cruise|h'] = 1/inputs['mission|climb|vs']/(nn-1)
        J['mission|climb|dt','mission|takeoff|h'] = -1/inputs['mission|climb|vs']/(nn-1)
        J['mission|climb|dt','mission|climb|vs'] = -(inputs['mission|cruise|h']-inputs['mission|takeoff|h'])/(inputs['mission|climb|vs']**2)/(nn-1)
        J['mission|descent|dt','mission|cruise|h'] = -1/inputs['mission|descent|vs']/(nn-1)
        J['mission|descent|dt','mission|takeoff|h'] = 1/inputs['mission|descent|vs']/(nn-1)
        J['mission|descent|dt','mission|descent|vs'] = -(inputs['mission|takeoff|h']-inputs['mission|cruise|h'])/(inputs['mission|descent|vs']**2)/(nn-1)


class MissionNoReserves(Group):
    """This analysis group calculates energy/fuel consumption and feasibility for a given mission profile.

    This component should be instantiated in the top-level aircraft analysis / optimization script.
    **Suggested variable promotion list:**
    *"ac|aero|\*",  "ac|geom|\*",  "fltcond|mission|\*",  "mission|\*"*

    **Inputs List:**

    From aircraft config:
        - ac|aero|polar|CD0_cruise
        - ac|aero|polar|e
        - ac|geom|wing|S_ref
        - ac|geom|wing|AR

    From mission config:
        - mission|weight_initial

    From mission flight condition generator:
        - fltcond|mission|vs
        - mission|climb|time
        - mission|climb|dt
        - mission|descent|time
        - mission|descent|dt

    From standard atmosphere model/splitter:
        - fltcond|mission|Utrue
        - fltcond|mission|q

    From propulsion model:
        - mission|battery_load
        - mission|fuel_flow
        - mission|thrust

    Outputs
    -------
    mission|total_fuel : float
        Total fuel burn for climb, cruise, and descent (scalar, kg)
    mission|total_battery_energy : float
        Total energy consumption for climb, cruise, and descent (scalar, kJ)
    thrust_resid.thrust_residual : float
        Imbalance between thrust and drag for use with Newton solver (scalar, N)
    """

    def initialize(self):
        self.options.declare('n_int_per_seg',default=5,desc="Number of Simpson intervals to use per seg (eg. climb, cruise, descend). Number of analysis points is 2N+1")
        self.options.declare('track_battery',default=False, desc="Flip to true if you want to track battery state")
    def setup(self):
        n_int_per_seg = self.options['n_int_per_seg']
        nn_tot = (2*n_int_per_seg+1)*3 #climb, cruise, descent
        #Create holders for control and flight condition parameters
        track_battery = self.options['track_battery']

        dvlist = [['fltcond|mission|q','fltcond|q',100*np.ones(nn_tot),'Pa'],
                  ['ac|aero|polar|CD0_cruise','CD0',0.005,None],
                  ['ac|aero|polar|e','e',0.95,None]]
        self.add_subsystem('dvs',DVLabel(dvlist),promotes_inputs=["*"],promotes_outputs=["*"])

        groundspeeds = self.add_subsystem('gs',MissionGroundspeeds(n_int_per_seg=n_int_per_seg),promotes_inputs=["fltcond|*"],promotes_outputs=["mission|groundspeed","fltcond|*"])
        ranges = self.add_subsystem('ranges',MissionClimbDescentRanges(n_int_per_seg=n_int_per_seg),promotes_inputs=["mission|groundspeed","mission|*time"],promotes_outputs=["mission|climb|range","mission|descent|range"])
        timings = self.add_subsystem('timings',MissionTimings(n_int_per_seg=n_int_per_seg),promotes_inputs=["mission|*range","mission|groundspeed"],promotes_outputs=["mission|cruise|range","mission|cruise|time","mission|cruise|dt"])

        fbs = self.add_subsystem('fuelburn',MissionSegmentFuelBurns(n_int_per_seg=n_int_per_seg),promotes_inputs=["mission|fuel_flow","mission|*dt"],promotes_outputs=["mission|segment_fuel"])
        if track_battery:
            energy = self.add_subsystem('battery',MissionSegmentBatteryEnergyUsed(n_int_per_seg=n_int_per_seg),promotes_inputs=["mission|battery_load",'mission*dt'],promotes_outputs=["mission|segment_battery_energy_used"])
        wts = self.add_subsystem('weights',MissionSegmentWeights(n_int_per_seg=n_int_per_seg),promotes_inputs=["mission|segment_fuel","mission|weight_initial"],promotes_outputs=["mission|weights"])
        CLs = self.add_subsystem('CLs',MissionSegmentCL(n_int_per_seg=n_int_per_seg),promotes_inputs=["mission|weights","fltcond|mission|q",'fltcond|mission|cosgamma',"ac|geom|*"],promotes_outputs=["fltcond|mission|CL"])
        drag = self.add_subsystem('drag',PolarDrag(num_nodes=nn_tot),promotes_inputs=["fltcond|q","ac|geom|*","CD0","e"],promotes_outputs=["drag"])
        self.connect('fltcond|mission|CL','drag.fltcond|CL')
        td = self.add_subsystem('thrust_resid',ExplicitThrustResidual(n_int_per_seg=n_int_per_seg),promotes_inputs=["fltcond|mission|singamma","drag","mission|weights*","mission|thrust"])
        totals = SumComp(axis=None)
        totals.add_equation(output_name='mission|total_fuel',input_name='mission|segment_fuel', units='kg',scaling_factor=-1,vec_size=nn_tot-3)
        if track_battery:
            totals.add_equation(output_name='mission|total_battery_energy',input_name='mission|segment_battery_energy_used', units='MJ',vec_size=nn_tot-3)
        self.add_subsystem(name='totals',subsys=totals,promotes_inputs=['*'],promotes_outputs=['*'])


class ExplicitThrustResidual(ExplicitComponent):
    """
    Computes force imbalance in the aircraft x axis. Enables Newton solve for throttle at steady flight.

    Inputs
    ------
    drag : float
        Aircraft drag force at each analysis point (vector, N)
    fltcond|mission|singamma : float
        Sine of the flight path angle for all mission phases (vector, dimensionless)
    mission|weights : float
        Aircraft weight at each analysis point (vector, kg)
    mission|thrust : float
        Aircraft thrust force at each analysis point (vector, N)

    Outputs
    -------
    thrust_residual : float
        Imbalance in x-axis force at each analysis point (vector, N)

    Options
    -------
    n_int_per_seg : int
        Number of Simpson's rule intervals to use per mission segment.
        The total number of points is 2 * n_int_per_seg + 1
    """
    def initialize(self):

        self.options.declare('n_int_per_seg',default=5,desc="Number of Simpson intervals to use per seg (eg. climb, cruise, descend). Number of analysis points is 2N+1")

    def setup(self):
        n_int_per_seg = self.options['n_int_per_seg']
        nn = (n_int_per_seg*2 + 1)
        n_seg = 3
        arange = np.arange(0,n_seg*nn)
        self.add_input('drag', units='N',shape=(n_seg*nn,))
        self.add_input('fltcond|mission|singamma', shape=(n_seg*nn,))
        self.add_input('mission|weights', units='kg', shape=(n_seg*nn,))
        self.add_input('mission|thrust', units='N', shape=(n_seg*nn,))
        self.add_output('thrust_residual', shape=(n_seg*nn,), units='N')
        self.declare_partials(['thrust_residual'], ['drag'], rows=arange, cols=arange, val=-np.ones(nn*n_seg))
        self.declare_partials(['thrust_residual'], ['mission|thrust'], rows=arange, cols=arange, val=np.ones(nn*n_seg))
        self.declare_partials(['thrust_residual'], ['fltcond|mission|singamma','mission|weights'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):

        g = 9.80665 #m/s^2
        debug_nonlinear = False
        if debug_nonlinear:
            print('Thrust: ' + str(inputs['mission|thrust']))
            print('Drag: '+ str(inputs['drag']))
            print('mgsingamma: ' + str(inputs['mission|weights']*g*inputs['fltcond|mission|singamma']))
            print('Throttle: ' + str(outputs['throttle']))

        outputs['thrust_residual'] = inputs['mission|thrust'] - inputs['drag'] - inputs['mission|weights']*g*inputs['fltcond|mission|singamma']


    def compute_partials(self, inputs, J):

        g = 9.80665 #m/s^2
        J['thrust_residual','mission|weights'] = -g*inputs['fltcond|mission|singamma']
        J['thrust_residual','fltcond|mission|singamma'] = -g*inputs['mission|weights']


class ComputeDesignMissionResiduals(ExplicitComponent):
    """
    Computes weight margins to ensure feasible mission profiles

    For aircraft including battery energy, use `ComputeDesignMissionResidualsBattery` instead

    Inputs
    ------
    ac|weights|MTOW : float
        Maximum takeoff weight (scalar, kg)
    ac|weights|W_fuel_max : float
        Max fuel weight (inc. vol limits; scalar, kg)
    mission|payload : float
        Payload weight including pax (scalar, kg)
    mission|total_fuel : float
        Fuel consume during the mission profile (not including TO; scalar, kg)
    OEW : float
        Operational empty weight (scalar, kg)
    takeoff|total_fuel : float
        Fuel consumed during takeoff (only if `include_takeoff` option is `True`)

    Outputs
    -------
    mission|fuel_capacity_margin : float
        Excess fuel capacity for this mission (scalar, kg)
        Positive is good
    mission|MTOW_margin : float
        Excess takeoff weight avail. for this mission (scalar, kg)
        Positive is good
    fuel_burn : float
        Total fuel burn including takeoff and the mission (scalar, kg)
        Only when `include_takeoff` is `True`

    Options
    -------
    include_takeoff : bool
        Set to `True` to enable takeoff fuel burn input
    """

    def initialize(self):

        self.options.declare('include_takeoff', default=False,
                             desc='Flag yes if you want to include takeoff fuel')

    def setup(self):
        include_takeoff = self.options['include_takeoff']
        self.add_input('ac|weights|MTOW', val=2000, units='kg')
        self.add_input('mission|total_fuel', val=180, units='kg')
        self.add_input('OEW', val=1500, units='kg')
        self.add_input('mission|payload', val=200, units='kg')
        self.add_input('ac|weights|W_fuel_max', val=400, units='kg')
        self.add_output('mission|fuel_capacity_margin', units='kg')
        self.add_output('mission|MTOW_margin', units='kg')
        self.declare_partials('mission|fuel_capacity_margin', 'mission|total_fuel', val=-1)
        self.declare_partials('mission|fuel_capacity_margin', 'ac|weights|W_fuel_max', val=1)
        self.declare_partials('mission|MTOW_margin', 'ac|weights|MTOW', val=1)
        self.declare_partials(['mission|MTOW_margin'],
                              ['mission|total_fuel', 'OEW', 'mission|payload'],
                              val=-1)

        if include_takeoff:
            self.add_input('takeoff|total_fuel', val=1, units='kg')
            self.add_output('fuel_burn', units='kg')
            self.declare_partials('mission|MTOW_margin', 'takeoff|total_fuel', val=-1)
            self.declare_partials('fuel_burn', ['takeoff|total_fuel', 'mission|total_fuel'], val=1)
            self.declare_partials('mission|fuel_capacity_margin', 'takeoff|total_fuel', val=-1)

    def compute(self, inputs, outputs):

        include_takeoff = self.options['include_takeoff']
        if include_takeoff:
            outputs['mission|fuel_capacity_margin'] = (inputs['ac|weights|W_fuel_max'] -
                                                       inputs['mission|total_fuel'] -
                                                       inputs['takeoff|total_fuel'])
            outputs['mission|MTOW_margin'] = (inputs['ac|weights|MTOW'] -
                                              inputs['mission|total_fuel'] -
                                              inputs['takeoff|total_fuel'] -
                                              inputs['OEW'] -
                                              inputs['mission|payload'])
            outputs['fuel_burn'] = inputs['mission|total_fuel'] + inputs['takeoff|total_fuel']

            if inputs['mission|total_fuel'] < -1e-4 or inputs['takeoff|total_fuel'] < -1e-4:
                raise ValueError('You have negative total fuel flows for some flight phase')

        else:
            outputs['mission|fuel_capacity_margin'] = (inputs['ac|weights|W_fuel_max'] -
                                                       inputs['mission|total_fuel'])
            outputs['mission|MTOW_margin'] = (inputs['ac|weights|MTOW'] -
                                              inputs['mission|total_fuel'] -
                                              inputs['OEW'] -
                                              inputs['mission|payload'])


class ComputeDesignMissionResidualsBattery(ComputeDesignMissionResiduals):
    """
    Computes weight and energy margins to ensure feasible mission profiles.

    This routine is applicable to electric and hybrid architectures.
    For fuel-only designs, use `ComputeDesignMissionResiduals` instead.

    Inputs
    ------
    ac|weights|MTOW : float
        Maximum takeoff weight (scalar, kg)
    ac|weights|W_battery : float
        Battery weight (scalar, kg)
    ac|weights|W_fuel_max : float
        Max fuel weight (inc. vol limits; scalar, kg)
    battery_max_energy : float
        Maximum energy of the battery at 100% SOC (scalar, MJ)
    mission|payload : float
        Payload weight including pax (scalar, kg)
    mission|total_battery_energy : float
        Battery energy consumed during the mission profile (scalar, MJ)
    mission|total_fuel : float
        Fuel consumed during the mission profile (not including TO; scalar, kg)
    OEW : float
        Operational empty weight (scalar, kg)
    takeoff|total_battery_energy : float
        Battery energy consumed during takeoff (only if `include_takeoff` option is `True`)
    takeoff|total_fuel : float
        Fuel consumed during takeoff (only if `include_takeoff` option is `True`)

    Outputs
    -------
    mission|battery_margin : float
        Excess battery energy for this mission (scalar, kg)
    mission|fuel_capacity_margin : float
        Excess fuel capacity for this mission (scalar, kg)
        Positive is good
    mission|MTOW_margin : float
        Excess takeoff weight avail. for this mission (scalar, kg)
        Positive is good
    battery_energy_used : float
        Total battery energy used including takeoff and the mission (scalar, MJ)
        Only when `include_takeoff` is `True`
    fuel_burn : float
        Total fuel burn including takeoff and the mission (scalar, kg)
        Only when `include_takeoff` is `True`

    Options
    -------
    include_takeoff : bool
        Set to `True` to enable takeoff fuel burn input
    """

    def setup(self):
        super(ComputeDesignMissionResidualsBattery, self).setup()
        include_takeoff = self.options['include_takeoff']
        self.add_input('ac|weights|W_battery', val=0, units='kg')
        self.add_input('battery_max_energy', val=1, units='MJ')
        self.add_input('mission|total_battery_energy', val=0, units='MJ')
        self.add_output('mission|battery_margin', units='MJ')
        self.declare_partials('mission|MTOW_margin', 'ac|weights|W_battery', val=-1)
        self.declare_partials('mission|battery_margin', 'battery_max_energy', val=1)
        self.declare_partials('mission|battery_margin', 'mission|total_battery_energy', val=-1)
        if include_takeoff:
            self.add_input('takeoff|total_battery_energy', val=0, units='MJ')
            self.add_output('battery_energy_used', units='MJ')
            self.declare_partials('battery_energy_used',
                                  ['mission|total_battery_energy', 'takeoff|total_battery_energy'],
                                  val=1)
            self.declare_partials('mission|battery_margin', 'takeoff|total_battery_energy', val=-1)

    def compute(self, inputs, outputs):

        super(ComputeDesignMissionResidualsBattery, self).compute(inputs, outputs)
        include_takeoff = self.options['include_takeoff']
        if include_takeoff:
            outputs['battery_energy_used'] = (inputs['mission|total_battery_energy'] +
                                              inputs['takeoff|total_battery_energy'])
            outputs['mission|battery_margin'] = (inputs['battery_max_energy'] -
                                                 inputs['mission|total_battery_energy'] -
                                                 inputs['takeoff|total_battery_energy'])
            outputs['mission|MTOW_margin'] = (inputs['ac|weights|MTOW'] -
                                              inputs['mission|total_fuel'] -
                                              inputs['takeoff|total_fuel'] -
                                              inputs['ac|weights|W_battery'] -
                                              inputs['OEW'] -
                                              inputs['mission|payload'])

        else:
            outputs['mission|battery_margin'] = (inputs['battery_max_energy'] -
                                                 inputs['mission|total_battery_energy'])
            outputs['mission|MTOW_margin'] = (inputs['ac|weights|MTOW'] -
                                              inputs['mission|total_fuel'] -
                                              inputs['ac|weights|W_battery'] -
                                              inputs['OEW'] -
                                              inputs['mission|payload'])


class MissionGroundspeeds(ExplicitComponent):
    """
    Computes groundspeed for vectorial true airspeed and true vertical speed.

    This is a helper function for the main mission analysis routine `MissionNoReserves`
    and shouldn't be instantiated directly.

    Inputs
    ------
    fltcond|mission|vs : float
        Vertical speed for all mission phases (vector, m/s)
    fltcond|mission|Utrue : float
        True airspeed for all mission phases (vector, m/s)

    Outputs
    -------
    mission|groundspeed : float
        True groundspeed for all mission phases (vector, m/s)
    fltcond|mission|cosgamma : float
        Cosine of the flght path angle for all mission phases (vector, dimensionless)
    fltcond|mission|singamma : float
        Sine of the flight path angle for all mission phases (vector, dimensionless)

    Options
    -------
    n_int_per_seg : int
        Number of Simpson's rule intervals to use per mission segment.
        The total number of points is 2 * n_int_per_seg + 1
    """
    def initialize(self):

        self.options.declare('n_int_per_seg',default=5,desc="Number of Simpson intervals to use per seg (eg. climb, cruise, descend). Number of analysis points is 2N+1")

    def setup(self):
        n_int_per_seg = self.options['n_int_per_seg']
        nn = (n_int_per_seg*2 + 1)
        self.add_input('fltcond|mission|vs', units='m/s',shape=(3 * nn,))
        self.add_input('fltcond|mission|Utrue', units='m/s',shape=(3 * nn,))
        self.add_output('mission|groundspeed', units='m/s',shape=(3 * nn,))
        self.add_output('fltcond|mission|cosgamma', shape=(3 * nn,), desc='Cosine of the flight path angle')
        self.add_output('fltcond|mission|singamma', shape=(3 * nn,), desc='sin of the flight path angle' )
        self.declare_partials(['mission|groundspeed','fltcond|mission|cosgamma','fltcond|mission|singamma'], ['fltcond|mission|vs','fltcond|mission|Utrue'], rows=range(3 * nn), cols=range(3 * nn))

    def compute(self, inputs, outputs):

        n_int_per_seg = self.options['n_int_per_seg']
        nn = (n_int_per_seg*2 + 1)
        #compute the groundspeed on climb and desc
        groundspeed =  np.sqrt(inputs['fltcond|mission|Utrue']**2-inputs['fltcond|mission|vs']**2)
        outputs['mission|groundspeed'] = groundspeed
        outputs['fltcond|mission|singamma'] = inputs['fltcond|mission|vs'] / inputs['fltcond|mission|Utrue']
        outputs['fltcond|mission|cosgamma'] = groundspeed / inputs['fltcond|mission|Utrue']

    def compute_partials(self, inputs, J):

        groundspeed =  np.sqrt(inputs['fltcond|mission|Utrue']**2-inputs['fltcond|mission|vs']**2)
        J['mission|groundspeed','fltcond|mission|vs'] = (1/2) / np.sqrt(inputs['fltcond|mission|Utrue']**2-inputs['fltcond|mission|vs']**2) * (-2) * inputs['fltcond|mission|vs']
        J['mission|groundspeed','fltcond|mission|Utrue'] = (1/2) / np.sqrt(inputs['fltcond|mission|Utrue']**2-inputs['fltcond|mission|vs']**2) * 2 * inputs['fltcond|mission|Utrue']
        J['fltcond|mission|singamma','fltcond|mission|vs'] = 1 / inputs['fltcond|mission|Utrue']
        J['fltcond|mission|singamma','fltcond|mission|Utrue'] = - inputs['fltcond|mission|vs'] / inputs['fltcond|mission|Utrue'] ** 2
        J['fltcond|mission|cosgamma','fltcond|mission|vs'] = J['mission|groundspeed','fltcond|mission|vs'] / inputs['fltcond|mission|Utrue']
        J['fltcond|mission|cosgamma','fltcond|mission|Utrue'] = (J['mission|groundspeed','fltcond|mission|Utrue'] * inputs['fltcond|mission|Utrue'] - groundspeed) / inputs['fltcond|mission|Utrue']**2

class MissionClimbDescentRanges(ExplicitComponent):
    """
    Computes range over the ground during the climb and descent phases

    This is a helper function for the main mission analysis routine `MissionNoReserves`
    and shouldn't be instantiated directly.

    Inputs
    ------
    mission|groundspeed : float
        True groundspeed for all mission phases (vector, m/s)
    mission|climb|time : float
        Time elapsed during the climb phase (scalar, s)
    mission|descent|time : float
        Time elapsed during the descent phase (scalar, s)

    Outputs
    -------
    mission|climb|range : float
        Distance over the ground during climb phase (scalar, m)
    mission|descent|range : float
        Distance over the ground during descent phase (scalar , m)

    Options
    -------
    n_int_per_seg : int
        Number of Simpson's rule intervals to use per mission segment.
        The total number of points is 2 * n_int_per_seg + 1
    """
    def initialize(self):

        self.options.declare('n_int_per_seg',default=5,desc="Number of Simpson intervals to use per seg (eg. climb, cruise, descend). Number of analysis points is 2N+1")

    def setup(self):
        n_int_per_seg = self.options['n_int_per_seg']
        nn = (n_int_per_seg*2 + 1)
        self.add_input('mission|groundspeed', units='m/s',shape=(3 * nn,))
        self.add_input('mission|climb|time', units='s')
        self.add_input('mission|descent|time', units='s')
        self.add_output('mission|descent|range', units='m')
        self.add_output('mission|climb|range', units='m')
        self.declare_partials(['mission|climb|range'], ['mission|groundspeed'], rows=np.ones(nn)*0, cols=range(nn))
        self.declare_partials(['mission|descent|range'], ['mission|groundspeed'], rows=np.ones(nn)*0, cols=np.arange(2*nn,3 * nn))
        self.declare_partials(['mission|climb|range'], ['mission|climb|time'])
        self.declare_partials(['mission|descent|range'], ['mission|descent|time'])

    def compute(self, inputs, outputs):

        n_int_per_seg = self.options['n_int_per_seg']
        nn = (n_int_per_seg*2 + 1)

        groundspeed = inputs['mission|groundspeed']
        #compute distance traveled during climb and desc using Simpson's rule
        dt_climb = inputs['mission|climb|time'] / (nn-1)
        dt_desc = inputs['mission|descent|time'] / (nn-1)
        simpsons_vec = np.ones(nn)
        simpsons_vec[1:nn-1:2] = 4
        simpsons_vec[2:nn-1:2] = 2
        outputs['mission|climb|range'] = np.sum(simpsons_vec*groundspeed[0:nn])*dt_climb/3
        outputs['mission|descent|range'] = np.sum(simpsons_vec*groundspeed[2*nn:3 * nn])*dt_desc/3

    def compute_partials(self, inputs, J):

        n_int_per_seg = self.options['n_int_per_seg']
        nn = (n_int_per_seg*2 + 1)

        groundspeed = inputs['mission|groundspeed']
        simpsons_vec = np.ones(nn)
        simpsons_vec[1:nn-1:2] = 4
        simpsons_vec[2:nn-1:2] = 2

        J['mission|climb|range','mission|climb|time'] = np.sum(simpsons_vec*groundspeed[0:nn])/3/(nn-1)
        J['mission|descent|range','mission|descent|time'] = np.sum(simpsons_vec*groundspeed[2*nn:3 * nn])/3/(nn-1)
        J['mission|climb|range','mission|groundspeed'] = simpsons_vec * inputs['mission|climb|time'] / (nn-1) / 3
        J['mission|descent|range','mission|groundspeed'] = simpsons_vec * inputs['mission|descent|time'] / (nn-1) / 3

class MissionTimings(ExplicitComponent):
    """
    Computes cruise distance, time, and dt for a given total mission range

    This is a helper function for the main mission analysis routine `MissionNoReserves`
    and shouldn't be instantiated directly.

    Inputs
    ------
    mission|range : float
        Total specified range for the given mission (vector, m)
    mission|groundspeed : float
        True groundspeed for all mission phases (vector, m/s)
    mission|climb|range : float
        Distance over the ground during climb phase (scalar, m)
    mission|descent|range : float
        Distance over the ground during descent phase (scalar , m)

    Outputs
    -------
    mission|cruise|range : float
        Distance over the ground during the cruise phase (scalar, m)
    mission|cruise|time : float
        Time elapsed during cruise phase (scalar, s)
    mission|cruise|dt : float
        Simpson subinterval timestep during the cruise phase (scalar, s)

    Options
    -------
    n_int_per_seg : int
        Number of Simpson's rule intervals to use per mission segment.
        The total number of points is 2 * n_int_per_seg + 1
    """

    def initialize(self):

        self.options.declare('n_int_per_seg',default=5,desc="Number of Simpson intervals to use per seg (eg. climb, cruise, descend). Number of analysis points is 2N+1")
        n_int_per_seg = self.options['n_int_per_seg']
        nn = (n_int_per_seg*2 + 1)
    def setup(self):
        n_int_per_seg = self.options['n_int_per_seg']
        nn = (n_int_per_seg*2 + 1)

        self.add_input('mission|groundspeed', units='m/s',shape=(3 * nn,))
        self.add_input('mission|climb|range', units='m')
        self.add_input('mission|descent|range', units='m')
        self.add_input('mission|range', units='m')
        self.add_output('mission|cruise|range', units='m')
        self.add_output('mission|cruise|time', units='s')
        self.add_output('mission|cruise|dt', units="s")
        self.declare_partials(['mission|cruise|range'], ['mission|climb|range'], val=-1.0)
        self.declare_partials(['mission|cruise|range'], ['mission|descent|range'], val=-1.0)
        self.declare_partials(['mission|cruise|range'], ['mission|range'], val=1.0)
        self.declare_partials(['mission|cruise|time'], ['mission|groundspeed'], rows=np.zeros(nn), cols=np.arange(nn,2*nn))
        self.declare_partials(['mission|cruise|time'], ['mission|climb|range','mission|descent|range','mission|range'])
        self.declare_partials(['mission|cruise|dt'], ['mission|groundspeed'], rows=np.zeros(nn), cols=np.arange(nn,2*nn))
        self.declare_partials(['mission|cruise|dt'], ['mission|climb|range','mission|descent|range','mission|range'])

    def compute(self, inputs, outputs):

        n_int_per_seg = self.options['n_int_per_seg']
        nn = (n_int_per_seg*2 + 1)

        simpsons_vec = np.ones(nn)
        simpsons_vec[1:nn-1:2] = 4
        simpsons_vec[2:nn-1:2] = 2

        #compute the cruise distance
        r_cruise = inputs['mission|range'] - inputs['mission|climb|range'] - inputs['mission|descent|range']
        if r_cruise < 0:
            raise ValueError('Cruise calculated to be less than 0. Change climb and descent rates and airspeeds or increase range')
        dt_cruise = 3*r_cruise/np.sum(simpsons_vec*inputs['mission|groundspeed'][nn:2*nn])
        t_cruise = dt_cruise*(nn-1)

        outputs['mission|cruise|time'] = t_cruise
        outputs['mission|cruise|range'] = r_cruise
        outputs['mission|cruise|dt'] = dt_cruise

    def compute_partials(self, inputs, J):

        n_int_per_seg = self.options['n_int_per_seg']
        nn = (n_int_per_seg*2 + 1)

        simpsons_vec = np.ones(nn)
        simpsons_vec[1:nn-1:2] = 4
        simpsons_vec[2:nn-1:2] = 2

        #compute the cruise distance
        r_cruise = inputs['mission|range'] - inputs['mission|climb|range'] - inputs['mission|descent|range']
        J['mission|cruise|time','mission|groundspeed'] = -3*r_cruise/np.sum(simpsons_vec*inputs['mission|groundspeed'][nn:2*nn])**2 * (nn-1) * (simpsons_vec)
        J['mission|cruise|time','mission|climb|range'] = -3/np.sum(simpsons_vec*inputs['mission|groundspeed'][nn:2*nn])*(nn-1)
        J['mission|cruise|time','mission|descent|range'] = -3/np.sum(simpsons_vec*inputs['mission|groundspeed'][nn:2*nn])*(nn-1)
        J['mission|cruise|time','mission|range'] = 3/np.sum(simpsons_vec*inputs['mission|groundspeed'][nn:2*nn])*(nn-1)

        J['mission|cruise|dt','mission|groundspeed'] = -3*r_cruise/np.sum(simpsons_vec*inputs['mission|groundspeed'][nn:2*nn])**2  * (simpsons_vec)
        J['mission|cruise|dt','mission|climb|range'] = -3/np.sum(simpsons_vec*inputs['mission|groundspeed'][nn:2*nn])
        J['mission|cruise|dt','mission|descent|range'] = -3/np.sum(simpsons_vec*inputs['mission|groundspeed'][nn:2*nn])
        J['mission|cruise|dt','mission|range'] = 3/np.sum(simpsons_vec*inputs['mission|groundspeed'][nn:2*nn])

class MissionSegmentFuelBurns(ExplicitComponent):
    """
    Integrates delta fuel between each analysis point

    This is a helper function for the main mission analysis routine `MissionNoReserves`
    and shouldn't be instantiated directly.

    Takes 3 * nn fuel flow rates; produces 3 * (nn - 1) fuel burns

    Inputs
    ------
    mission|fuel_flow : float
        Fuel flow rate for all analysis points (vector, kg/s)
    mission|climb|dt : float
        Timestep length during climb phase (scalar, s)
        Note: this represents the timestep for the Simpson subinterval, not the whole inteval
    mission|cruise|dt : float
        Timestep length during descent phase (scalar, s)
        Note: this represents the timestep for the Simpson subinterval, not the whole inteval
    mission|descent|dt : float
        Timestep length during descent phase (scalar, s)
        Note: this represents the timestep for the Simpson subinterval, not the whole inteval

    Outputs
    -------
    mission|segment_fuel : float
        Fuel burn increment between each analysis point (vector, kg)
        Note: if the number of analysis points in one phase is `nn`, the number
        of segment fuel burns is `nn - 1`

    Options
    -------
    n_int_per_seg : int
        Number of Simpson's rule intervals to use per mission segment.
        The total number of points is 2 * n_int_per_seg + 1
    """
    def initialize(self):

        self.options.declare('n_int_per_seg',default=5,desc="Number of Simpson intervals to use per seg (eg. climb, cruise, descend). Number of analysis points is 2N+1")

    def setup(self):
        n_int_per_seg = self.options['n_int_per_seg']
        nn = (n_int_per_seg*2 + 1)

        self.add_input('mission|fuel_flow', units='kg/s',shape=(3 * nn,))
        self.add_input('mission|climb|dt', units='s')
        self.add_input('mission|descent|dt', units='s')
        self.add_input('mission|cruise|dt', units='s')

        self.add_output('mission|segment_fuel', units='kg',shape=(3*(nn-1)))
        #use dummy inputs for dt and q, just want the shapes
        wrt_q, wrt_dt = simpson_partials_every_node(np.ones(3),np.ones(3 * nn),n_segments=3,n_simpson_intervals_per_segment=n_int_per_seg)

        self.declare_partials(['mission|segment_fuel'], ['mission|fuel_flow'], rows=wrt_q[0], cols=wrt_q[1])
        self.declare_partials(['mission|segment_fuel'], ['mission|climb|dt'], rows=wrt_dt[0][0], cols=wrt_dt[1][0])
        self.declare_partials(['mission|segment_fuel'], ['mission|cruise|dt'], rows=wrt_dt[0][1], cols=wrt_dt[1][1])
        self.declare_partials(['mission|segment_fuel'], ['mission|descent|dt'], rows=wrt_dt[0][2], cols=wrt_dt[1][2])

    def compute(self, inputs, outputs):

        n_int_per_seg = self.options['n_int_per_seg']
        nn = (n_int_per_seg*2 + 1)
        ff = inputs['mission|fuel_flow']
        dts =  [inputs['mission|climb|dt'], inputs['mission|cruise|dt'],inputs['mission|descent|dt']]
        int_ff, delta_ff = simpson_integral_every_node(dts,ff,n_segments=3,n_simpson_intervals_per_segment=n_int_per_seg)

        outputs['mission|segment_fuel'] = delta_ff

    def compute_partials(self, inputs, J):

        n_int_per_seg = self.options['n_int_per_seg']
        nn = (n_int_per_seg*2 + 1)
        ff = inputs['mission|fuel_flow']
        dts =  [inputs['mission|climb|dt'], inputs['mission|cruise|dt'],inputs['mission|descent|dt']]

        wrt_q, wrt_dt = simpson_partials_every_node(dts,ff,n_segments=3,n_simpson_intervals_per_segment=n_int_per_seg)

        J['mission|segment_fuel','mission|fuel_flow'] = wrt_q[2]
        J['mission|segment_fuel','mission|climb|dt'] =  wrt_dt[2][0]
        J['mission|segment_fuel','mission|cruise|dt'] = wrt_dt[2][1]
        J['mission|segment_fuel','mission|descent|dt'] = wrt_dt[2][2]

class MissionSegmentBatteryEnergyUsed(ExplicitComponent):
    """
    Integrates battery energy used between each analysis point

    This is a helper function for the main mission analysis routine `MissionNoReserves`
    and shouldn't be instantiated directly.

    Takes 3 * nn battery loads; produces 3 * (nn - 1) energy increments

    Inputs
    ------
    mission|battery_load : float
        Battery load / power  for all analysis points (vector, kW)
    mission|climb|dt : float
        Timestep length during climb phase (scalar, s)
        Note: this represents the timestep for the Simpson subinterval, not the whole inteval
    mission|cruise|dt : float
        Timestep length during descent phase (scalar, s)
        Note: this represents the timestep for the Simpson subinterval, not the whole inteval
    mission|descent|dt : float
        Timestep length during descent phase (scalar, s)
        Note: this represents the timestep for the Simpson subinterval, not the whole inteval

    Outputs
    -------
    mission|segment_battery_energy_used : float
        Battery energy increment between each analysis point (vector, kW*s)
        Note: if the number of analysis points in one phase is `nn`, the number
        of segment energies is `nn - 1`

    Options
    -------
    n_int_per_seg : int
        Number of Simpson's rule intervals to use per mission segment.
        The total number of points is 2 * n_int_per_seg + 1
    """
    def initialize(self):

        self.options.declare('n_int_per_seg',default=5,desc="Number of Simpson intervals to use per seg (eg. climb, cruise, descend). Number of analysis points is 2N+1")

    def setup(self):
        n_int_per_seg = self.options['n_int_per_seg']
        nn = (n_int_per_seg*2 + 1)

        self.add_input('mission|battery_load', units='kW',shape=(3 * nn,))
        self.add_input('mission|climb|dt', units='s')
        self.add_input('mission|descent|dt', units='s')
        self.add_input('mission|cruise|dt', units='s')

        self.add_output('mission|segment_battery_energy_used', units='kW*s',shape=(3*(nn-1)))
        #use dummy inputs for dt and q, just want the shapes
        wrt_q, wrt_dt = simpson_partials_every_node(np.ones(3),np.ones(3 * nn),n_segments=3,n_simpson_intervals_per_segment=n_int_per_seg)

        self.declare_partials(['mission|segment_battery_energy_used'], ['mission|battery_load'], rows=wrt_q[0], cols=wrt_q[1])
        self.declare_partials(['mission|segment_battery_energy_used'], ['mission|climb|dt'], rows=wrt_dt[0][0], cols=wrt_dt[1][0])
        self.declare_partials(['mission|segment_battery_energy_used'], ['mission|cruise|dt'], rows=wrt_dt[0][1], cols=wrt_dt[1][1])
        self.declare_partials(['mission|segment_battery_energy_used'], ['mission|descent|dt'], rows=wrt_dt[0][2], cols=wrt_dt[1][2])

    def compute(self, inputs, outputs):

        n_int_per_seg = self.options['n_int_per_seg']
        nn = (n_int_per_seg*2 + 1)
        ff = inputs['mission|battery_load']
        dts =  [inputs['mission|climb|dt'], inputs['mission|cruise|dt'],inputs['mission|descent|dt']]
        int_ff, delta_ff = simpson_integral_every_node(dts,ff,n_segments=3,n_simpson_intervals_per_segment=n_int_per_seg)

        outputs['mission|segment_battery_energy_used'] = delta_ff

    def compute_partials(self, inputs, J):

        n_int_per_seg = self.options['n_int_per_seg']
        nn = (n_int_per_seg*2 + 1)
        ff = inputs['mission|battery_load']
        dts =  [inputs['mission|climb|dt'], inputs['mission|cruise|dt'],inputs['mission|descent|dt']]

        wrt_q, wrt_dt = simpson_partials_every_node(dts,ff,n_segments=3,n_simpson_intervals_per_segment=n_int_per_seg)

        J['mission|segment_battery_energy_used','mission|battery_load'] = wrt_q[2]
        J['mission|segment_battery_energy_used','mission|climb|dt'] =  wrt_dt[2][0]
        J['mission|segment_battery_energy_used','mission|cruise|dt'] = wrt_dt[2][1]
        J['mission|segment_battery_energy_used','mission|descent|dt'] = wrt_dt[2][2]


class MissionSegmentWeights(ExplicitComponent):
    """
    Computes aircraft weight at each analysis point including fuel burned

    This is a helper function for the main mission analysis routine `MissionNoReserves`
    and shouldn't be instantiated directly.

    Inputs
    ------
    mission|segment_fuel : float
        Fuel burn increment between each analysis point (vector, kg)
        Note: if the number of analysis points in one phase is `nn`, the number
        of segment fuel burns is `nn - 1`
    mission|weight_initial : float
        Weight immediately following takeoff (scalar, kg)

    Outputs
    -------
    mission|weights : float
        Aircraft weight at each analysis point (vector, kg)


    Options
    -------
    n_int_per_seg : int
        Number of Simpson's rule intervals to use per mission segment.
        The total number of points is 2 * n_int_per_seg + 1
    """
    def initialize(self):

        self.options.declare('n_int_per_seg',default=5,desc="Number of Simpson intervals to use per seg (eg. climb, cruise, descend). Number of analysis points is 2N+1")

    def setup(self):
        n_int_per_seg = self.options['n_int_per_seg']
        nn = (n_int_per_seg*2 + 1)
        self.add_input('mission|segment_fuel', units='kg',shape=(3*(nn-1),))
        self.add_input('mission|weight_initial', units='kg')
        self.add_output('mission|weights', units='kg',shape=(3 * nn,))

        n_seg = 3
        jacmat = np.tril(np.ones((n_seg*(nn-1),n_seg*(nn-1))))
        jacmat = np.insert(jacmat,0,np.zeros(n_seg*(nn-1)),axis=0)
        for i in range(1,n_seg):
            duplicate_row = jacmat[nn*i-1,:]
            jacmat = np.insert(jacmat,nn*i,duplicate_row,axis=0)

        self.declare_partials(['mission|weights'], ['mission|segment_fuel'], val=sp.csr_matrix(jacmat))
        self.declare_partials(['mission|weights'], ['mission|weight_initial'], rows=range(3 * nn), cols=np.zeros(3 * nn), val=np.ones(3 * nn))

    def compute(self, inputs, outputs):

        n_int_per_seg = self.options['n_int_per_seg']
        nn = (n_int_per_seg*2 + 1)
        #first endpoint needs to be the takeoff weight; insert a zero to make them the same length
        n_seg = 3
        segweights = np.insert(inputs['mission|segment_fuel'],0,0)
        weights = np.cumsum(segweights)
        for i in range(1,n_seg):
            duplicate_row = weights[i*nn-1]
            weights = np.insert(weights,i*nn,duplicate_row)
        outputs['mission|weights'] = np.ones(3 * nn)*inputs['mission|weight_initial'] + weights

class MissionSegmentCL(ExplicitComponent):
    """
    Computes lift coefficient at each analysis point

    This is a helper function for the main mission analysis routine `MissionNoReserves`
    and shouldn't be instantiated directly.

    Inputs
    ------
    mission|weights : float
        Aircraft weight at each analysis point (vector, kg)
    fltcond|mission|q : float
        Dynamic pressure at each analysis point (vector, Pascal)
    ac|geom|wing|S_ref : float
        Reference wing area (scalar, m**2)
    fltcond|mission|cosgamma : float
        Cosine of the flght path angle for all mission phases (vector, dimensionless)

    Outputs
    -------
    fltcond|mission|CL : float
        Lift coefficient (vector, dimensionless)

    Options
    -------
    n_int_per_seg : int
        Number of Simpson's rule intervals to use per mission segment.
        The total number of points is 2 * n_int_per_seg + 1
    """
    def initialize(self):

        self.options.declare('n_int_per_seg',default=5,desc="Number of Simpson intervals to use per seg (eg. climb, cruise, descend). Number of analysis points is 2N+1")

    def setup(self):
        n_int_per_seg = self.options['n_int_per_seg']
        nn = (n_int_per_seg*2 + 1)
        n_seg = 3
        arange = np.arange(0,n_seg*nn)
        self.add_input('mission|weights', units='kg', shape=(n_seg*nn,))
        self.add_input('fltcond|mission|q', units='N * m**-2', shape=(n_seg*nn,))
        self.add_input('ac|geom|wing|S_ref', units='m **2')
        self.add_input('fltcond|mission|cosgamma', shape=(n_seg*nn,))
        self.add_output('fltcond|mission|CL',shape=(n_seg*nn,))


        self.declare_partials(['fltcond|mission|CL'], ['mission|weights','fltcond|mission|q',"fltcond|mission|cosgamma"], rows=arange, cols=arange)
        self.declare_partials(['fltcond|mission|CL'], ['ac|geom|wing|S_ref'], rows=arange, cols=np.zeros(n_seg*nn))

    def compute(self, inputs, outputs):

        n_int_per_seg = self.options['n_int_per_seg']
        nn = (n_int_per_seg*2 + 1)
        #first endpoint needs to be the takeoff weight; insert a zero to make them the same length
        n_seg = 3

        g = 9.80665 #m/s^2
        outputs['fltcond|mission|CL'] = inputs['fltcond|mission|cosgamma']*g*inputs['mission|weights']/inputs['fltcond|mission|q']/inputs['ac|geom|wing|S_ref']

    def compute_partials(self, inputs, J):

        n_int_per_seg = self.options['n_int_per_seg']
        nn = (n_int_per_seg*2 + 1)
        #first endpoint needs to be the takeoff weight; insert a zero to make them the same length
        n_seg = 3

        g = 9.80665 #m/s^2
        J['fltcond|mission|CL','mission|weights'] = inputs['fltcond|mission|cosgamma']*g/inputs['fltcond|mission|q']/inputs['ac|geom|wing|S_ref']
        J['fltcond|mission|CL','fltcond|mission|q'] = - inputs['fltcond|mission|cosgamma']*g*inputs['mission|weights'] / inputs['fltcond|mission|q']**2 / inputs['ac|geom|wing|S_ref']
        J['fltcond|mission|CL','ac|geom|wing|S_ref'] = - inputs['fltcond|mission|cosgamma']*g*inputs['mission|weights'] / inputs['fltcond|mission|q'] / inputs['ac|geom|wing|S_ref']**2
        J['fltcond|mission|CL','fltcond|mission|cosgamma'] = g*inputs['mission|weights']/inputs['fltcond|mission|q']/inputs['ac|geom|wing|S_ref']



