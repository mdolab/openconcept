from __future__ import division
from openmdao.api import Group, ExplicitComponent, IndepVarComp, BalanceComp, ImplicitComponent
import openmdao.api as om
import openconcept.api as oc
from openconcept.analysis.atmospherics.compute_atmos_props import ComputeAtmosphericProperties
from openconcept.analysis.aerodynamics import Lift, StallSpeed
from openconcept.utilities.math import ElementMultiplyDivideComp, AddSubtractComp
from openconcept.utilities.math.integrals import Integrator
from openconcept.utilities.linearinterp import LinearInterpolator
from openconcept.utilities.math.integrals import Integrator
import numpy as np
import copy

class Groundspeeds(ExplicitComponent):
    """
    Computes groundspeed for vectorial true airspeed and true vertical speed.

    This is a helper function for the main mission analysis routines
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
    num_nodes : int
        Number of points to run
    """
    def initialize(self):

        self.options.declare('num_nodes',default=1,desc="Number of Simpson intervals to use per seg (eg. climb, cruise, descend). Number of analysis points is 2N+1")

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('fltcond|vs', units='m/s',shape=(nn,))
        self.add_input('fltcond|Utrue', units='m/s',shape=(nn,))
        self.add_output('fltcond|groundspeed', units='m/s',shape=(nn,))
        self.add_output('fltcond|cosgamma', shape=(nn,), desc='Cosine of the flight path angle')
        self.add_output('fltcond|singamma', shape=(nn,), desc='sin of the flight path angle' )
        self.declare_partials(['fltcond|groundspeed','fltcond|cosgamma','fltcond|singamma'], ['fltcond|vs','fltcond|Utrue'], rows=range(nn), cols=range(nn))

    def compute(self, inputs, outputs):

        nn = self.options['num_nodes']
        #compute the groundspeed on climb and desc
        inside = inputs['fltcond|Utrue']**2-inputs['fltcond|vs']**2
        groundspeed =  np.sqrt(inside)
        groundspeed_fixed = np.sqrt(np.where(np.less(inside, 0.0), 0.01, inside))
        #groundspeed =  np.sqrt(inputs['fltcond|Utrue']**2-inputs['fltcond|vs']**2)
        #groundspeed_fixed= np.where(np.isnan(groundspeed),0,groundspeed)
        outputs['fltcond|groundspeed'] = groundspeed_fixed
        outputs['fltcond|singamma'] = np.where(np.isnan(groundspeed),1,inputs['fltcond|vs'] / inputs['fltcond|Utrue'])
        outputs['fltcond|cosgamma'] = groundspeed_fixed / inputs['fltcond|Utrue']

    def compute_partials(self, inputs, J):
        inside = inputs['fltcond|Utrue']**2-inputs['fltcond|vs']**2
        groundspeed =  np.sqrt(inside)
        groundspeed_fixed = np.sqrt(np.where(np.less(inside, 0.0), 0.01, inside))
        J['fltcond|groundspeed','fltcond|vs'] = np.where(np.isnan(groundspeed),0,(1/2) / groundspeed_fixed * (-2) * inputs['fltcond|vs'])
        J['fltcond|groundspeed','fltcond|Utrue'] = np.where(np.isnan(groundspeed),0, (1/2) / groundspeed_fixed * 2 * inputs['fltcond|Utrue'])
        J['fltcond|singamma','fltcond|vs'] = np.where(np.isnan(groundspeed), 0, 1 / inputs['fltcond|Utrue'])
        J['fltcond|singamma','fltcond|Utrue'] = np.where(np.isnan(groundspeed), 0, - inputs['fltcond|vs'] / inputs['fltcond|Utrue'] ** 2)
        J['fltcond|cosgamma','fltcond|vs'] = J['fltcond|groundspeed','fltcond|vs'] / inputs['fltcond|Utrue']
        J['fltcond|cosgamma','fltcond|Utrue'] = (J['fltcond|groundspeed','fltcond|Utrue'] * inputs['fltcond|Utrue'] - groundspeed_fixed) / inputs['fltcond|Utrue']**2

class HorizontalAcceleration(ExplicitComponent):
    """
    Computes acceleration during takeoff run and effectively forms the T-D residual.

    Inputs
    ------
    weight : float
        Aircraft weight (scalar, kg)
    drag : float
        Aircraft drag at each analysis point (vector, N)
    lift : float
        Aircraft lift at each analysis point (vector, N)
    thrust : float
        Thrust at each TO analysis point (vector, N)
    fltcond|singamma : float
        The sine of the flight path angle gamma (vector, dimensionless)
    braking : float
        Effective rolling friction multiplier at each point (vector, dimensionless)

    Outputs
    -------
    accel_horiz : float
        Aircraft horizontal acceleration (vector, m/s**2)

    Options
    -------
    num_nodes : int
        Number of analysis points to run
    """
    def initialize(self):
        self.options.declare('num_nodes',default=1)

    def setup(self):
        nn = self.options['num_nodes']
        g = 9.80665 #m/s^2
        self.add_input('weight', units='kg', shape=(nn,))
        self.add_input('drag', units='N',shape=(nn,))
        self.add_input('lift', units='N',shape=(nn,))
        self.add_input('thrust', units='N',shape=(nn,))
        self.add_input('fltcond|singamma',shape=(nn,))
        self.add_input('braking',shape=(nn,))

        self.add_output('accel_horiz', units='m/s**2', shape=(nn,))
        arange=np.arange(nn)
        self.declare_partials(['accel_horiz'], ['weight','drag','lift','thrust','braking'], rows=arange, cols=arange)
        self.declare_partials(['accel_horiz'], ['fltcond|singamma'], rows=arange, cols=arange, val=-g*np.ones((nn,)))


    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        g = 9.80665 #m/s^2
        m = inputs['weight']
        floor_vec = np.where(np.less((g-inputs['lift']/m),0.0),0.0,1.0)
        accel = inputs['thrust']/m - inputs['drag']/m - floor_vec*inputs['braking']*(g-inputs['lift']/m) - g*inputs['fltcond|singamma']
        outputs['accel_horiz'] = accel

    def compute_partials(self, inputs, J):
        g = 9.80665 #m/s^2
        m = inputs['weight']
        floor_vec = np.where(np.less((g-inputs['lift']/m),0.0),0.0,1.0)
        J['accel_horiz','thrust'] = 1/m
        J['accel_horiz','drag'] = -1/m
        J['accel_horiz','braking'] = -floor_vec*(g-inputs['lift']/m)
        J['accel_horiz','lift'] = floor_vec*inputs['braking']/m
        J['accel_horiz','weight'] = (inputs['drag']-inputs['thrust']-floor_vec*inputs['braking']*inputs['lift'])/m**2

    """
    Computes acceleration during takeoff run in the vertical plane.
    Only used during full unsteady takeoff performance analysis due to stability issues

    Inputs
    ------
    weight : float
        Aircraft weight (scalar, kg)
    drag : float
        Aircraft drag at each analysis point (vector, N)
    lift : float
        Aircraft lift at each analysis point (vector, N)
    thrust : float
        Thrust at each TO analysis point (vector, N)
    fltcond|singamma : float
        The sine of the flight path angle gamma (vector, dimensionless)
    fltcond|cosgamma : float
        The sine of the flight path angle gamma (vector, dimensionless)

    Outputs
    -------
    accel_vert : float
        Aircraft horizontal acceleration (vector, m/s**2)

    Options
    -------
    num_nodes : int
        Number of analysis points to run
    """
    def initialize(self):
        self.options.declare('num_nodes',default=1)

    def setup(self):
        nn = self.options['num_nodes']
        g = 9.80665 #m/s^2
        self.add_input('weight', units='kg', shape=(nn,))
        self.add_input('drag', units='N',shape=(nn,))
        self.add_input('lift', units='N',shape=(nn,))
        self.add_input('thrust', units='N',shape=(nn,))
        self.add_input('fltcond|singamma',shape=(nn,))
        self.add_input('fltcond|cosgamma',shape=(nn,))

        self.add_output('accel_vert', units='m/s**2', shape=(nn,),upper=2.5*g,lower=-1*g)
        arange=np.arange(nn)
        self.declare_partials(['accel_vert'], ['weight','drag','lift','thrust','fltcond|singamma','fltcond|cosgamma'], rows=arange, cols=arange)


    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        g = 9.80665 #m/s^2
        cosg = inputs['fltcond|cosgamma']
        sing = inputs['fltcond|singamma']
        accel = (inputs['lift']*cosg + (inputs['thrust']-inputs['drag'])*sing - g*inputs['weight'])/inputs['weight']
        accel = np.clip(accel, -g, 2.5*g)
        outputs['accel_vert'] = accel

    def compute_partials(self, inputs, J):
        g = 9.80665 #m/s^2
        m = inputs['weight']
        cosg = inputs['fltcond|cosgamma']
        sing = inputs['fltcond|singamma']

        J['accel_vert','thrust'] = sing / m
        J['accel_vert','drag'] = -sing / m
        J['accel_vert','lift'] = cosg / m
        J['accel_vert','fltcond|singamma'] = (inputs['thrust']-inputs['drag']) / m
        J['accel_vert','fltcond|cosgamma'] = inputs['lift'] / m
        J['accel_vert','weight'] = -(inputs['lift']*cosg + (inputs['thrust']-inputs['drag'])*sing)/m**2

class SteadyFlightCL(ExplicitComponent):
    """
    Computes lift coefficient at each analysis point

    This is a helper function for the main mission analysis routine
    and shouldn't be instantiated directly.

    Inputs
    ------
    weight : float
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
    num_nodes : int
        Number of analysis nodes to run
    mission_segments : list
        The list of mission segments to track
    """
    def initialize(self):

        self.options.declare('num_nodes',default=5,desc="Number of Simpson intervals to use per seg (eg. climb, cruise, descend). Number of analysis points is 2N+1")
        self.options.declare('mission_segments',default=['climb','cruise','descent'])
    def setup(self):
        nn = self.options['num_nodes']
        arange = np.arange(nn)
        self.add_input('weight', units='kg', shape=(nn,))
        self.add_input('fltcond|q', units='N * m**-2', shape=(nn,))
        self.add_input('ac|geom|wing|S_ref', units='m **2')
        self.add_input('fltcond|cosgamma', val=1.0, shape=(nn,))
        self.add_output('fltcond|CL',shape=(nn,))
        self.declare_partials(['fltcond|CL'], ['weight','fltcond|q',"fltcond|cosgamma"], rows=arange, cols=arange)
        self.declare_partials(['fltcond|CL'], ['ac|geom|wing|S_ref'], rows=arange, cols=np.zeros(nn))

    def compute(self, inputs, outputs):
        g = 9.80665 #m/s^2
        outputs['fltcond|CL'] = inputs['fltcond|cosgamma']*g*inputs['weight']/inputs['fltcond|q']/inputs['ac|geom|wing|S_ref']

    def compute_partials(self, inputs, J):
        g = 9.80665 #m/s^2
        J['fltcond|CL','weight'] = inputs['fltcond|cosgamma']*g/inputs['fltcond|q']/inputs['ac|geom|wing|S_ref']
        J['fltcond|CL','fltcond|q'] = - inputs['fltcond|cosgamma']*g*inputs['weight'] / inputs['fltcond|q']**2 / inputs['ac|geom|wing|S_ref']
        J['fltcond|CL','ac|geom|wing|S_ref'] = - inputs['fltcond|cosgamma']*g*inputs['weight'] / inputs['fltcond|q'] / inputs['ac|geom|wing|S_ref']**2
        J['fltcond|CL','fltcond|cosgamma'] = g*inputs['weight']/inputs['fltcond|q']/inputs['ac|geom|wing|S_ref']

class DymosSteadyFlightODE(om.Group):
    """
    This component group models steady flight conditions.
    Settable mission parameters include:
    Airspeed (fltcond|Ueas)
    Vertical speed (fltcond|vs)
    Duration of the segment (duration)

    Throttle is set automatically to ensure steady flight

    The BaseAircraftGroup object is passed in.
    The BaseAircraftGroup should be built to accept the following inputs
    and return the following outputs.
    The outputs should be promoted to the top level in the component.

    Inputs
    ------
    range : float
        Total distance travelled (vector, m)
    fltcond|h : float
        Altitude (vector, m)
    fltcond|vs : float
        Vertical speed (vector, m/s)
    fltcond|Ueas : float
        Equivalent airspeed (vector, m/s)
    fltcond|Utrue : float
        True airspeed (vector, m/s)
    fltcond|p : float
        Pressure (vector, Pa)
    fltcond|rho : float
        Density (vector, kg/m3)
    fltcond|T : float
        Temperature (vector, K)
    fltcond|q : float
        Dynamic pressure (vector, Pa)
    fltcond|CL : float
        Lift coefficient (vector, dimensionless)
    throttle : float
        Motor / propeller throttle setting scaled from 0 to 1 or slightly more (vector, dimensionless)
    propulsor_active : float
        If a multi-propulsor airplane, a failure condition should be modeled in the propulsion model by multiplying throttle by propulsor_active.
        It will generally be 1.0 unless a failure condition is being modeled, in which case it will be 0 (vector, dimensionless)
    braking : float
        Brake friction coefficient (default 0.4 for dry runway braking, 0.03 for resistance unbraked)
        Should not be applied in the air or nonphysical effects will result (vector, dimensionless)
    lift : float
        Lift force (vector, N)

    Outputs
    -------
    thrust : float
        Total thrust force produced by all propulsors (vector, N)
    drag : float
        Total drag force in the airplane axis produced by all sources of drag (vector, N)
    weight : float
        Weight (mass, really) of the airplane at each point in time. (vector, kg)
    ac|geom|wing|S_ref
        Wing reference area (scalar, m**2)
    ac|aero|CLmax_TO
        CLmax with flaps in max takeoff position (scalar, dimensionless)
    ac|weights|MTOW
        Maximum takeoff weight (scalar, kg)
    """
    def initialize(self):
        self.options.declare('num_nodes',default=1)
        self.options.declare('flight_phase',default=None,desc='Phase of flight e.g. v0v1, cruise')
        self.options.declare('aircraft_model',default=None)

    def setup(self):
        nn = self.options['num_nodes']
        ivcomp = self.add_subsystem('const_settings', IndepVarComp(), promotes_outputs=["*"])
        ivcomp.add_output('propulsor_active', val=np.ones(nn))
        ivcomp.add_output('braking', val=np.zeros(nn))
        # TODO feet fltcond|Ueas as control param
        ivcomp.add_output('fltcond|Ueas',val=np.ones((nn,))*90, units='m/s')
        # TODO feed fltcond|vs as control param
        ivcomp.add_output('fltcond|vs',val=np.ones((nn,))*1, units='m/s')
        ivcomp.add_output('zero_accel',val=np.zeros((nn,)),units='m/s**2')
        
        # TODO take out the integrator
        integ = self.add_subsystem('ode_integ', Integrator(num_nodes=nn, diff_units='s', time_setup='duration', method='simpson'), promotes_inputs=['fltcond|vs', 'fltcond|groundspeed'], promotes_outputs=['fltcond|h', 'range'])
        integ.add_integrand('fltcond|h', rate_name='fltcond|vs', val=1.0, units='m')
        # TODO Feed fltcond|h as state

        self.add_subsystem('atmos', ComputeAtmosphericProperties(num_nodes=nn, true_airspeed_in=False), promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('gs',Groundspeeds(num_nodes=nn),promotes_inputs=['*'],promotes_outputs=['*'])
        # add the user-defined aircraft model
        # TODO Can I promote up ac| quantities?
        self.add_subsystem('acmodel',self.options['aircraft_model'](num_nodes=nn, flight_phase=self.options['flight_phase']),promotes_inputs=['*'],promotes_outputs=['*'])
        self.add_subsystem('clcomp',SteadyFlightCL(num_nodes=nn), promotes_inputs=['*'],promotes_outputs=['*'])
        self.add_subsystem('lift',Lift(num_nodes=nn), promotes_inputs=['*'],promotes_outputs=['*'])
        self.add_subsystem('haccel',HorizontalAcceleration(num_nodes=nn), promotes_inputs=['*'],promotes_outputs=['*'])
        # TODO add range as a state
        integ.add_integrand('range', rate_name='fltcond|groundspeed', val=1.0, units='m')
        self.add_subsystem('steadyflt',BalanceComp(name='throttle',val=np.ones((nn,))*0.5,lower=0.01,upper=2.0,units=None,normalize=False,eq_units='m/s**2',rhs_name='accel_horiz',lhs_name='zero_accel',rhs_val=np.zeros((nn,))),
                           promotes_inputs=['accel_horiz','zero_accel'],promotes_outputs=['throttle'])
        # TODO still needs a Newton solver