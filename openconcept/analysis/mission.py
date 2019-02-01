from __future__ import division
from openmdao.api import IndepVarComp, Group, Problem, ExplicitComponent, ImplicitComponent, BalanceComp
import numpy as np
from openconcept.utilities.dvlabel import DVLabel
from openconcept.utilities.math.sum_comp import SumComp
from openconcept.analysis.aerodynamics import PolarDrag
from openconcept.analysis.atmospherics.compute_atmos_props import ComputeAtmosphericProperties
from openconcept.utilities.math.integrals import Integrator
from openconcept.utilities.math import AddSubtractComp
import scipy.sparse as sp
import warnings

class MissionFlightConditions(ExplicitComponent):
    """
    Generates vectors of flight conditions for a mission profile

    Inputs
    ------
    phasename|time : float
        Total time for segment 'phasename' (scalar, s)
    phasename|Ueas : float
        Indicated/equiv. airspeed during seg 'phasename' (scalar, m/s)
    phasename|h0 : float
        INITIAL altitude for segment 'phasename' (scalar, m)

    Outputs
    -------
    fltcond|vs : float
        Vertical speed vector for all mission phases / analysis points (vector, m/s)
    fltcond|Ueas : float
        Equivalent airspeed vector for all mission phases / analysis points (vector, m/s)
    fltcond|h : float
        Altitude at each analysis point (vector, m)

    phasename|dt : float
        Timestep length during an individual mission phase (scalar, s)
        Note: this represents the timestep for the Simpson subinterval, not the whole inteval

    Options
    -------
    mission_segments : list
        The names of each mission segment. The number of segments
        will correspond to the length of the list
    nn_seg : int
        Number of nodes per segment.
    """

    def initialize(self):

        self.options.declare('nn_seg', default=11,
                             desc="Number of analysis nodes per segment")
        self.options.declare('mission_segments', default=['climb', 'cruise', 'descent'])

    def setup(self):
        nn = self.options['nn_seg']
        self.nn = nn
        mission_segment_names = self.options['mission_segments']
        nn_tot = nn * len(mission_segment_names)
        self.nn_tot = nn_tot
        n_mission_segments = len(mission_segment_names)
        last_seg_index = len(mission_segment_names) - 1
        for i, segment_name in enumerate(mission_segment_names):
            self.add_input(segment_name+'|time', val=5, units='s')
            self.add_input(segment_name+'|Ueas', val=90, units='m / s')
            self.add_input(segment_name+'|h0', val=0, units='m')
            # last segment needs a final altitude
            if i == last_seg_index:
                self.add_input(segment_name+'|hf', val=0, units='m')

            self.add_output(segment_name+'|dt', units='s')

        self.add_output('fltcond|Ueas', units='m / s',
                        desc='indicated airspeed at each analysis point', shape=(nn_tot,))
        self.add_output('fltcond|h', units='m',
                        desc='altitude at each analysis point', shape=(nn_tot,))
        self.add_output('fltcond|vs', units='m / s',
                        desc='vectorial representation of vertical speed', shape=(nn_tot,))


        for i, segment_name in enumerate(mission_segment_names):
            # airspeeds are constant across each flight segment
            self.declare_partials(['fltcond|Ueas'], [segment_name+'|Ueas'],
                                  rows=np.arange(i*nn, (i+1)*nn), cols=np.zeros(nn), val=np.ones(nn))
            # vertical speeds depend on the starting altitude and segment time
            self.declare_partials(['fltcond|vs'], [segment_name+'|time'],
                                  rows=np.arange(i*nn, (i+1)*nn), cols=np.zeros(nn))

            if i == last_seg_index:
                self.declare_partials(['fltcond|vs'], [segment_name+'|hf'],
                                      rows=np.arange(i*nn, (i+1)*nn), cols=np.zeros(nn))
            if i == 0:
                self.declare_partials(['fltcond|vs'], [segment_name+'|h0'],
                                      rows=np.arange(i*nn, (i+1)*nn), cols=np.zeros(nn))
            else:
                self.declare_partials(['fltcond|vs'], [segment_name+'|h0'],
                                      rows=np.arange((i-1)*nn, (i+1)*nn), cols=np.zeros(2*nn))

            # segment dt depends only on the segment time
            self.declare_partials([segment_name+'|dt'], [segment_name+'|time'],
                                  val=1/(nn-1))

            # the influence of altitude varies linearly from 0 to 1 back to 0.
            if i == 0:
                # first segment
                rowrange = np.arange(i*nn, (i+1)*nn)
                cols = np.zeros(nn)
                h0partials = np.linspace(1.0, 0.0, nn)
            else:
                rowrange = np.arange((i-1)*nn, (i+1)*nn)
                cols = np.zeros(2*nn)
                h0partials = np.concatenate([np.linspace(0.0, 1.0, nn), np.linspace(1.0, 0.0, nn)])

            self.declare_partials(['fltcond|h'],[segment_name+'|h0'],
                                  rows=rowrange, cols=cols, val=h0partials)

            if i == last_seg_index:
                rowrange = np.arange(i*nn, (i+1)*nn)
                cols = np.zeros(nn)
                hfpartials = np.linspace(0.0, 1.0, nn)
                self.declare_partials(['fltcond|h'],[segment_name+'|hf'],
                                      rows=rowrange, cols=cols, val=hfpartials)


    def compute(self, inputs, outputs):
        mission_segment_names = self.options['mission_segments']
        nn = self.nn
        nn_tot = self.nn_tot
        last_seg_index = len(mission_segment_names) - 1
        altitude_vecs = []
        Ueas_vecs = []
        vs_vecs = []
        for i, segment_name in enumerate(mission_segment_names):
            if i == last_seg_index:
                h0 = inputs[segment_name+'|h0']
                hf = inputs[segment_name+'|hf']
            else:
                next_seg_name = mission_segment_names[i+1]
                h0 = inputs[segment_name+'|h0']
                hf = inputs[next_seg_name+'|h0']
            vs = (hf - h0) / inputs[segment_name+'|time']

            altitude_vecs.append(np.linspace(h0,hf,nn))
            Ueas_vecs.append(np.ones(nn)*inputs[segment_name+'|Ueas'])
            vs_vecs.append(np.ones(nn)*vs)
            outputs[segment_name+'|dt'] = inputs[segment_name+'|time'] / (nn-1)

        outputs['fltcond|h'] = np.concatenate(altitude_vecs).flatten()
        outputs['fltcond|Ueas'] = np.concatenate(Ueas_vecs).flatten()
        outputs['fltcond|vs'] = np.concatenate(vs_vecs).flatten()


    def compute_partials(self, inputs, J):
        mission_segment_names = self.options['mission_segments']
        nn = self.nn
        nn_tot = self.nn_tot
        last_seg_index = len(mission_segment_names) - 1
        for i, segment_name in enumerate(mission_segment_names):
            seg_time = inputs[segment_name+'|time']
            if i == last_seg_index:
                h0 = inputs[segment_name+'|h0']
                hf = inputs[segment_name+'|hf']
                J['fltcond|vs',segment_name+'|hf'] = np.ones(nn) * 1.0 / seg_time
            else:
                h0 = inputs[segment_name+'|h0']
                hf = inputs[mission_segment_names[i+1]+'|h0']

            if i == 0:
                J['fltcond|vs',segment_name+'|h0'] = np.ones(nn) * -1.0 / seg_time
            else:
                prev_seg_name = mission_segment_names[i-1]
                prev_seg_time = inputs[prev_seg_name+'|time']
                J['fltcond|vs',segment_name+'|h0'] = np.concatenate([np.ones(nn) * 1.0 / prev_seg_time,
                                                                                        np.ones(nn) * -1.0 / seg_time])

            J['fltcond|vs',segment_name+'|time'] = np.ones(nn) * -1.0 * (hf - h0) / seg_time ** 2

class Groundspeeds(ExplicitComponent):
    """
    Computes groundspeed for vectorial true airspeed and true vertical speed.

    This is a helper function for the main mission analysis routine `MissionNoReserves`
    and shouldn't be instantiated directly.

    Inputs
    ------
    fltcond|vs : float
        Vertical speed for all mission phases (vector, m/s)
    fltcond|Utrue : float
        True airspeed for all mission phases (vector, m/s)

    Outputs
    -------
    fltcond|groundspeed : float
        True groundspeed for all mission phases (vector, m/s)
    fltcond|cosgamma : float
        Cosine of the flght path angle for all mission phases (vector, dimensionless)
    fltcond|singamma : float
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
        self.add_input('fltcond|vs', units='m/s',shape=(3 * nn,))
        self.add_input('fltcond|Utrue', units='m/s',shape=(3 * nn,))
        self.add_output('fltcond|groundspeed', units='m/s',shape=(3 * nn,))
        self.add_output('fltcond|cosgamma', shape=(3 * nn,), desc='Cosine of the flight path angle')
        self.add_output('fltcond|singamma', shape=(3 * nn,), desc='sin of the flight path angle' )
        self.declare_partials(['fltcond|groundspeed','fltcond|cosgamma','fltcond|singamma'], ['fltcond|vs','fltcond|Utrue'], rows=range(3 * nn), cols=range(3 * nn))

    def compute(self, inputs, outputs):

        n_int_per_seg = self.options['n_int_per_seg']
        nn = (n_int_per_seg*2 + 1)
        #compute the groundspeed on climb and desc
        groundspeed =  np.sqrt(inputs['fltcond|Utrue']**2-inputs['fltcond|vs']**2)
        outputs['fltcond|groundspeed'] = groundspeed
        outputs['fltcond|singamma'] = inputs['fltcond|vs'] / inputs['fltcond|Utrue']
        outputs['fltcond|cosgamma'] = groundspeed / inputs['fltcond|Utrue']

    def compute_partials(self, inputs, J):

        groundspeed =  np.sqrt(inputs['fltcond|Utrue']**2-inputs['fltcond|vs']**2)
        J['fltcond|groundspeed','fltcond|vs'] = (1/2) / np.sqrt(inputs['fltcond|Utrue']**2-inputs['fltcond|vs']**2) * (-2) * inputs['fltcond|vs']
        J['fltcond|groundspeed','fltcond|Utrue'] = (1/2) / np.sqrt(inputs['fltcond|Utrue']**2-inputs['fltcond|vs']**2) * 2 * inputs['fltcond|Utrue']
        J['fltcond|singamma','fltcond|vs'] = 1 / inputs['fltcond|Utrue']
        J['fltcond|singamma','fltcond|Utrue'] = - inputs['fltcond|vs'] / inputs['fltcond|Utrue'] ** 2
        J['fltcond|cosgamma','fltcond|vs'] = J['fltcond|groundspeed','fltcond|vs'] / inputs['fltcond|Utrue']
        J['fltcond|cosgamma','fltcond|Utrue'] = (J['fltcond|groundspeed','fltcond|Utrue'] * inputs['fltcond|Utrue'] - groundspeed) / inputs['fltcond|Utrue']**2

class MissionSegmentWeights(ExplicitComponent):
    """
    Computes aircraft weight at each analysis point including fuel burned

    This is a helper function for the main mission analysis routine `MissionNoReserves`
    and shouldn't be instantiated directly.

    Inputs
    ------
    fuel_used : float
        The total amount of fuel used at each point in the mission (vector, kg)
    weight_initial : float
        Weight immediately following takeoff (scalar, kg)

    Outputs
    -------
    weight_vec : float
        Aircraft weight at each analysis point (vector, kg)


    Options
    -------
    n_int_per_seg : int
        Number of Simpson's rule intervals to use per mission segment.
        The total number of points is 2 * n_int_per_seg + 1
    mission_segments : list
        The list of mission segments to track

    """
    def initialize(self):
        self.options.declare('n_int_per_seg',default=5,desc="Number of Simpson intervals to use per seg (eg. climb, cruise, descend). Number of analysis points is 2N+1")
        self.options.declare('mission_segments',default=['climb','cruise','descent'])

    def setup(self):
        n_int_per_seg = self.options['n_int_per_seg']
        nn = (n_int_per_seg*2 + 1)
        mission_segment_names = self.options['mission_segments']
        n_segments = len(mission_segment_names)
        self.add_input('fuel_used', units='kg',val=np.ones((n_segments* nn,)))
        self.add_input('weight_initial', units='kg')
        self.add_output('weight_vec', units='kg',shape=(n_segments * nn,))

        self.declare_partials(['weight_vec'], ['fuel_used'], rows=range(n_segments * nn), cols=range(n_segments * nn), val=-np.ones(n_segments * nn))
        self.declare_partials(['weight_vec'], ['weight_initial'], rows=range(n_segments * nn), cols=np.zeros(n_segments * nn), val=np.ones(n_segments * nn))

    def compute(self, inputs, outputs):

        n_int_per_seg = self.options['n_int_per_seg']
        nn = (n_int_per_seg*2 + 1)
        #first endpoint needs to be the takeoff weight; insert a zero to make them the same length
        mission_segment_names = self.options['mission_segments']
        n_segments = len(mission_segment_names)
        outputs['weight_vec'] = np.ones(n_segments * nn)*inputs['weight_initial'] - inputs['fuel_used']

class MissionSegmentCL(ExplicitComponent):
    """
    Computes lift coefficient at each analysis point

    This is a helper function for the main mission analysis routine `MissionNoReserves`
    and shouldn't be instantiated directly.

    Inputs
    ------
    weight_vec : float
        Aircraft weight at each analysis point (vector, kg)
    fltcond|q : float
        Dynamic pressure at each analysis point (vector, Pascal)
    ac|geom|wing|S_ref : float
        Reference wing area (scalar, m**2)
    fltcond|cosgamma : float
        Cosine of the flght path angle for all mission phases (vector, dimensionless)

    Outputs
    -------
    fltcond|CL : float
        Lift coefficient (vector, dimensionless)

    Options
    -------
    n_int_per_seg : int
        Number of Simpson's rule intervals to use per mission segment.
        The total number of points is 2 * n_int_per_seg + 1
    mission_segments : list
        The list of mission segments to track
    """
    def initialize(self):

        self.options.declare('n_int_per_seg',default=5,desc="Number of Simpson intervals to use per seg (eg. climb, cruise, descend). Number of analysis points is 2N+1")
        self.options.declare('mission_segments',default=['climb','cruise','descent'])
    def setup(self):
        n_int_per_seg = self.options['n_int_per_seg']
        nn = (n_int_per_seg*2 + 1)
        mission_segment_names = self.options['mission_segments']
        n_seg = len(mission_segment_names)
        arange = np.arange(0,n_seg*nn)
        self.add_input('weight_vec', units='kg', shape=(n_seg*nn,))
        self.add_input('fltcond|q', units='N * m**-2', shape=(n_seg*nn,))
        self.add_input('ac|geom|wing|S_ref', units='m **2')
        self.add_input('fltcond|cosgamma', shape=(n_seg*nn,))
        self.add_output('fltcond|CL',shape=(n_seg*nn,))


        self.declare_partials(['fltcond|CL'], ['weight_vec','fltcond|q',"fltcond|cosgamma"], rows=arange, cols=arange)
        self.declare_partials(['fltcond|CL'], ['ac|geom|wing|S_ref'], rows=arange, cols=np.zeros(n_seg*nn))

    def compute(self, inputs, outputs):

        n_int_per_seg = self.options['n_int_per_seg']
        nn = (n_int_per_seg*2 + 1)
        mission_segment_names = self.options['mission_segments']
        n_seg = len(mission_segment_names)
        g = 9.80665 #m/s^2
        outputs['fltcond|CL'] = inputs['fltcond|cosgamma']*g*inputs['weight_vec']/inputs['fltcond|q']/inputs['ac|geom|wing|S_ref']

    def compute_partials(self, inputs, J):

        n_int_per_seg = self.options['n_int_per_seg']
        nn = (n_int_per_seg*2 + 1)
        #first endpoint needs to be the takeoff weight; insert a zero to make them the same length
        mission_segment_names = self.options['mission_segments']
        n_seg = len(mission_segment_names)

        g = 9.80665 #m/s^2
        J['fltcond|CL','weight_vec'] = inputs['fltcond|cosgamma']*g/inputs['fltcond|q']/inputs['ac|geom|wing|S_ref']
        J['fltcond|CL','fltcond|q'] = - inputs['fltcond|cosgamma']*g*inputs['weight_vec'] / inputs['fltcond|q']**2 / inputs['ac|geom|wing|S_ref']
        J['fltcond|CL','ac|geom|wing|S_ref'] = - inputs['fltcond|cosgamma']*g*inputs['weight_vec'] / inputs['fltcond|q'] / inputs['ac|geom|wing|S_ref']**2
        J['fltcond|CL','fltcond|cosgamma'] = g*inputs['weight_vec']/inputs['fltcond|q']/inputs['ac|geom|wing|S_ref']

class ThrottleSetting(ImplicitComponent):
    """
    Computes force imbalance in the aircraft x axis. Enables Newton solve for throttle at steady flight.

    Inputs
    ------
    drag : float
        Aircraft drag force at each analysis point (vector, N)
    fltcond|singamma : float
        Sine of the flight path angle for all mission phases (vector, dimensionless)
    weight_vec : float
        Aircraft weight at each analysis point (vector, kg)
    thrust : float
        Aircraft thrust force at each analysis point (vector, N)

    Outputs
    -------
    throttle : float
        Throttle setting to maintain steady flight (vector, dimensionless)

    Options
    -------
    n_int_per_seg : int
        Number of Simpson's rule intervals to use per mission segment.
        The total number of points is 2 * n_int_per_seg + 1
    mission_segments : list
        The list of mission segments to track

    """
    def initialize(self):
        self.options.declare('n_int_per_seg',default=5,desc="Number of Simpson intervals to use per seg (eg. climb, cruise, descend). Number of analysis points is 2N+1")
        self.options.declare('mission_segments', default=['climb','cruise','descent'])

    def setup(self):
        n_int_per_seg = self.options['n_int_per_seg']
        nn = (n_int_per_seg*2 + 1)
        mission_segment_names = self.options['mission_segments']
        n_seg = len(mission_segment_names)
        arange = np.arange(0,n_seg*nn)
        self.add_input('drag', units='N',shape=(n_seg*nn,))
        self.add_input('fltcond|singamma', shape=(n_seg*nn,))
        self.add_input('weight_vec', units='kg', shape=(n_seg*nn,))
        self.add_input('thrust', units='N', shape=(n_seg*nn,))
        self.add_output('throttle', shape=(n_seg*nn,))
        self.declare_partials(['throttle'], ['drag'], rows=arange, cols=arange, val=-np.ones(nn*n_seg))
        self.declare_partials(['throttle'], ['thrust'], rows=arange, cols=arange, val=np.ones(nn*n_seg))
        self.declare_partials(['throttle'], ['fltcond|singamma','weight_vec'], rows=arange, cols=arange)

    def apply_nonlinear(self, inputs, outputs, residuals):
        g = 9.80665 #m/s^2
        debug_nonlinear = False
        if debug_nonlinear:
            print('Thrust: ' + str(inputs['thrust']))
            print('Drag: '+ str(inputs['drag']))
            print('mgsingamma: ' + str(inputs['weight_vec']*g*inputs['fltcond|singamma']))
            print('Throttle: ' + str(outputs['throttle']))
        residuals['throttle'] = inputs['thrust'] - inputs['drag'] - inputs['weight_vec']*g*inputs['fltcond|singamma']


    def linearize(self, inputs, outputs, J):
        g = 9.80665 #m/s^2
        J['throttle','weight_vec'] = -g*inputs['fltcond|singamma']
        J['throttle','fltcond|singamma'] = -g*inputs['weight_vec']

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
    payload : float
        Payload weight including pax (scalar, kg)
    mission|total_fuel : float
        Fuel consumed during the mission profile (not including TO; scalar, kg)
    OEW : float
        Operational empty weight (scalar, kg)
    takeoff|total_fuel : float
        Fuel consumed during takeoff (only if `include_takeoff` option is `True`)

    Outputs
    -------
    fuel_capacity_margin : float
        Excess fuel capacity for this mission (scalar, kg)
        Positive is good
    MTOW_margin : float
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
        self.add_input('payload', val=200, units='kg')
        self.add_input('ac|weights|W_fuel_max', val=400, units='kg')
        self.add_output('fuel_capacity_margin', units='kg')
        self.add_output('MTOW_margin', units='kg')
        self.declare_partials('fuel_capacity_margin', 'mission|total_fuel', val=-1)
        self.declare_partials('fuel_capacity_margin', 'ac|weights|W_fuel_max', val=1)
        self.declare_partials('MTOW_margin', 'ac|weights|MTOW', val=1)
        self.declare_partials(['MTOW_margin'],
                              ['mission|total_fuel', 'OEW', 'payload'],
                              val=-1)

        if include_takeoff:
            self.add_input('takeoff|total_fuel', val=1, units='kg')
            self.add_output('fuel_burn', units='kg')
            self.declare_partials('MTOW_margin', 'takeoff|total_fuel', val=-1)
            self.declare_partials('fuel_burn', ['takeoff|total_fuel', 'mission|total_fuel'], val=1)
            self.declare_partials('fuel_capacity_margin', 'takeoff|total_fuel', val=-1)

    def compute(self, inputs, outputs):

        include_takeoff = self.options['include_takeoff']
        if include_takeoff:
            outputs['fuel_capacity_margin'] = (inputs['ac|weights|W_fuel_max'] -
                                                       inputs['mission|total_fuel'] -
                                                       inputs['takeoff|total_fuel'])
            outputs['MTOW_margin'] = (inputs['ac|weights|MTOW'] -
                                              inputs['mission|total_fuel'] -
                                              inputs['takeoff|total_fuel'] -
                                              inputs['OEW'] -
                                              inputs['payload'])
            outputs['fuel_burn'] = inputs['mission|total_fuel'] + inputs['takeoff|total_fuel']

            if inputs['mission|total_fuel'] < -1e-4 or inputs['takeoff|total_fuel'] < -1e-4:
                warnings.warn('You have negative total fuel flows for some flight phase. Mission fuel: '+str(inputs['mission|total_fuel'])+' Takeoff fuel:'+str(inputs['takeoff|total_fuel'])+'. It is OK if this happens during the implicit solve, but double check the final result is positive.')
        else:
            outputs['fuel_capacity_margin'] = (inputs['ac|weights|W_fuel_max'] -
                                                       inputs['mission|total_fuel'])
            outputs['MTOW_margin'] = (inputs['ac|weights|MTOW'] -
                                              inputs['mission|total_fuel'] -
                                              inputs['OEW'] -
                                              inputs['payload'])

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
    payload : float
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
    battery_margin : float
        Excess battery energy for this mission (scalar, kg)
    fuel_capacity_margin : float
        Excess fuel capacity for this mission (scalar, kg)
        Positive is good
    MTOW_margin : float
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
        self.add_output('battery_margin', units='MJ')
        self.declare_partials('MTOW_margin', 'ac|weights|W_battery', val=-1)
        self.declare_partials('battery_margin', 'battery_max_energy', val=1)
        self.declare_partials('battery_margin', 'mission|total_battery_energy', val=-1)
        if include_takeoff:
            self.add_input('takeoff|total_battery_energy', val=0, units='MJ')
            self.add_output('battery_energy_used', units='MJ')
            self.declare_partials('battery_energy_used',
                                  ['mission|total_battery_energy', 'takeoff|total_battery_energy'],
                                  val=1)
            self.declare_partials('battery_margin', 'takeoff|total_battery_energy', val=-1)

    def compute(self, inputs, outputs):

        super(ComputeDesignMissionResidualsBattery, self).compute(inputs, outputs)
        include_takeoff = self.options['include_takeoff']
        if include_takeoff:
            outputs['battery_energy_used'] = (inputs['mission|total_battery_energy'] +
                                              inputs['takeoff|total_battery_energy'])
            outputs['battery_margin'] = (inputs['battery_max_energy'] -
                                                 inputs['mission|total_battery_energy'] -
                                                 inputs['takeoff|total_battery_energy'])
            outputs['MTOW_margin'] = (inputs['ac|weights|MTOW'] -
                                              inputs['mission|total_fuel'] -
                                              inputs['takeoff|total_fuel'] -
                                              inputs['ac|weights|W_battery'] -
                                              inputs['OEW'] -
                                              inputs['payload'])

        else:
            outputs['battery_margin'] = (inputs['battery_max_energy'] -
                                                 inputs['mission|total_battery_energy'])
            outputs['MTOW_margin'] = (inputs['ac|weights|MTOW'] -
                                              inputs['mission|total_fuel'] -
                                              inputs['ac|weights|W_battery'] -
                                              inputs['OEW'] -
                                              inputs['payload'])

class MissionAnalysis(Group):
    """This analysis group calculates energy/fuel consumption and feasibility for a given mission profile.

    This component should be instantiated in the top-level aircraft analysis / optimization script.
    **Suggested variable promotion list:**
    *"ac|aero|\*",  "ac|geom|\*",  "fltcond|\*",  "\*"*

    **Inputs List:**

    From aircraft config:
        - ac|aero|polar|CD0_cruise
        - ac|aero|polar|e
        - ac|geom|wing|S_ref
        - ac|geom|wing|AR
        - any other parameters required for the propulsion system

    From controls:
        - any control parameters required for the propulsion system
        - 'throttle' output from this component should be captured and fed back to propulsion system as it is being driven using the Newton solver internally to achieve steady flight

    From mission config:
        - weight_initial
        - design_range

    FOR EACH MISSION PHASE
        - phase|h0
        - phase|Ueas
        - phase|time (except for cruise)
        - phase|hf (for final segment)

    Outputs
    -------
    total_fuel : float
        Total fuel burn for climb, cruise, and descent (scalar, kg)
    total_battery_energy : float
        Total energy consumption for climb, cruise, and descent (scalar, kJ)
    throttle : float
        Propulsion thrust throttle setting at each mission point (needs to be fed back to the propulsion system from outside)

    """

    def initialize(self):
        self.options.declare('n_int_per_seg',default=5,desc="Number of Simpson intervals to use per seg (eg. climb, cruise, descend). Number of analysis points is 2N+1")
        self.options.declare('track_battery',default=True, desc="Flip to true if you want to track battery state")
        self.options.declare('track_fuel', default=True)
        self.options.declare('include_takeoff', default=True)
        self.options.declare('mission_segments',default=['climb','cruise','descent'])
        self.options.declare('range_segments',default=['climb','cruise','descent'])
        self.options.declare('propulsion_system',default=None)

    def setup(self):
        n_int_per_seg = self.options['n_int_per_seg']
        mission_segment_names = self.options['mission_segments']
        range_segment_names = self.options['range_segments']
        n_segments = len(mission_segment_names)
        nn_tot = (2*n_int_per_seg+1)*n_segments
        nn = 2*n_int_per_seg+1
        track_battery = self.options['track_battery']
        track_fuel = self.options['track_fuel']
        include_takeoff = self.options['include_takeoff']

        # time of cruise phase only is solved implicitly based on desired cruise range
        self.add_subsystem('cruisetime',BalanceComp(name='cruise|time',units='s',eq_units='m',rhs_name='range',lhs_name='design_range'),
                           promotes_inputs=['range','design_range'],promotes_outputs=['cruise|time'])
        # obtain a vector of flight conditions for the given mission profile
        # mission parameters (speeds, altitudes, times) will need to be connected in from above (except cruise time)
        self.add_subsystem('conditions',MissionFlightConditions(nn_seg=nn,
                                                                mission_segments=mission_segment_names),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])
        # obtain true airspeeds, densities, dynamic pressure from the standard atmosphere model
        self.add_subsystem('atmos',
                           ComputeAtmosphericProperties(num_nodes=nn_tot),
                           promotes_inputs=["fltcond|h",
                                            "fltcond|Ueas"],
                           promotes_outputs=["fltcond|rho",
                                             "fltcond|Utrue",
                                             "fltcond|q"])
        propulsion_promotes_outputs = ['fuel_flow','thrust']
        propulsion_promotes_inputs = ["fltcond|*","ac|propulsion|*"]
        if track_battery:
            propulsion_promotes_outputs.append('battery_load')
            propulsion_promotes_inputs.append("ac|weights|*")
        self.add_subsystem('propmodel',self.options['propulsion_system'],
                           promotes_inputs=propulsion_promotes_inputs,promotes_outputs=propulsion_promotes_outputs)


        # Compute the speed over the ground based on true airspeed and flight path angle
        self.add_subsystem('gs',Groundspeeds(n_int_per_seg=n_int_per_seg),promotes_inputs=["fltcond|*"],promotes_outputs=["fltcond|groundspeed","fltcond|*"])
        # Compute the range
        self.add_subsystem('rangecomp',Integrator(num_intervals=n_int_per_seg,
                                                  quantity_units='m',
                                                  diff_units='s',
                                                  method='simpson',
                                                  segment_names=mission_segment_names,
                                                  segments_to_count=range_segment_names,
                                                  zero_start=True),
                                    promotes_inputs=[('dqdt',"fltcond|groundspeed"),"*|dt"],
                                    promotes_outputs=[('q_final',"range"),('q',"distance")])
        # integrate the fuel burn rate and battery use weight over time to get battery state and overall weights
        if track_fuel:
            self.add_subsystem('fuelburn',Integrator(num_intervals=n_int_per_seg, quantity_units='kg', diff_units='s',
                                                     method='simpson',segment_names=mission_segment_names,zero_start=True),
                                                     promotes_inputs=[('dqdt','fuel_flow'),'*|dt'],
                                                     promotes_outputs=[('q','fuel_used'),('q_final','mission_total_fuel')])
        self.add_subsystem('weights',MissionSegmentWeights(n_int_per_seg=n_int_per_seg,mission_segments=mission_segment_names),
                                     promotes_inputs=['weight_initial','fuel_used'],promotes_outputs=['weight_vec'])
        if track_battery:
            self.add_subsystem('batteryuse',Integrator(num_intervals=n_int_per_seg, quantity_units='MJ', diff_units='s',
                                                     method='simpson',segment_names=mission_segment_names,zero_start=True),
                                                     promotes_inputs=[('dqdt','battery_load'),'*|dt'],
                                                     promotes_outputs=[('q','battery_used'),('q_final','mission_total_battery')])

        self.add_subsystem('CLs',MissionSegmentCL(n_int_per_seg=n_int_per_seg,
                                                  mission_segments=mission_segment_names),
                           promotes_inputs=["weight_vec","fltcond|q",'fltcond|cosgamma',"ac|geom|*"],
                           promotes_outputs=["fltcond|CL"])
        self.add_subsystem('drag',PolarDrag(num_nodes=nn_tot),
                                            promotes_inputs=["fltcond|q","fltcond|CL","ac|geom|*",
                                                             ("CD0","ac|aero|polar|CD0_cruise"),
                                                             ("e","ac|aero|polar|e")],
                                            promotes_outputs=["drag"])
        self.add_subsystem('throttle',ThrottleSetting(n_int_per_seg=n_int_per_seg,mission_segments=mission_segment_names),
                           promotes_inputs=["fltcond|singamma","drag","weight_vec","thrust"],
                           promotes_outputs=['throttle'])

        if track_battery:
            self.add_subsystem(name='residuals', subsys=ComputeDesignMissionResidualsBattery(include_takeoff=include_takeoff),
                               promotes_inputs=['ac|weights*','payload','OEW'],
                               promotes_outputs=['fuel_burn','battery_energy_used'])
            self.connect('mission_total_battery','residuals.mission|total_battery_energy')
            # TODO residuals.takeoff|total_battery_energy needs an incoming connection
        else:
            self.add_subsystem(name='residuals', subsys=ComputeDesignMissionResiduals(include_takeoff=include_takeoff),
                               promotes_inputs=['ac|weights*','payload','OEW'],
                               promotes_outputs=['fuel_burn'])
        self.connect('mission_total_fuel','residuals.mission|total_fuel')
        # TODO residuals.takeoff|total_fuel needs an incoming connection

