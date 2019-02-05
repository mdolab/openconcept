from __future__ import division
from openmdao.api import Group, ExplicitComponent, IndepVarComp, BalanceComp, ImplicitComponent
from openconcept.analysis.atmospherics.compute_atmos_props import ComputeAtmosphericProperties
from openconcept.analysis.aerodynamics import Lift, StallSpeed
from openconcept.utilities.math import ElementMultiplyDivideComp, AddSubtractComp
from openconcept.utilities.math.integrals import Integrator
import numpy as np
import copy

class FlipVectorComp(ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes',default=1)
        self.options.declare('negative',default=False)
        self.options.declare('units',default=None)

    def setup(self):
        nn = self.options['num_nodes']
        units = self.options['units']
        self.add_input('vec_in', units=units, shape=(nn,))
        self.add_output('vec_out', units=units, shape=(nn,))
        negative = self.options['negative']
        if negative:
            scaler = -1
        else:
            scaler = 1
        self.declare_partials(['vec_out'],['vec_in'],rows=np.arange(nn-1,-1,-1),cols=np.arange(0,nn,1),val=scaler*np.ones((nn,)))

    def compute(self, inputs, outputs):
        negative = self.options['negative']
        if negative:
            scaler = -1
        else:
            scaler = 1
        outputs['vec_out'] = scaler * np.flip(inputs['vec_in'], 0)


class BFLImplicitSolve(ImplicitComponent):
    """
    Computes a residual equation so Newton solver can set v1 to analyze balanced field length

    This residual is equal to zero if:
        - The rejected takeoff and engine-out takeoff distances are equal, or:
        - V1 is equal to VR and the engine out takeoff distance is longer than the RTO distance

    Since this is a discontinous function, the partial derivatives are written in a special way
    to 'coax' the V1 value into the right setting with a Newton step. It's kind of a hack.

    Inputs
    ------
    distance_continue : float
        Engine-out takeoff distance (scalar, m)
    distance_abort : float
        Distance to full-stop when takeoff is rejected at V1 (scalar, m)
    takeoff|vr : float
        Rotation speed (scalar, m/s)

    Outputs
    -------
    takeoff|v1 : float
        Decision speed (scalar, m/s)

    """
    def setup(self):
        self.add_input('distance_continue', units='m')
        self.add_input('distance_abort', units='m')
        self.add_input('takeoff|vr', units='m/s')
        self.add_output('takeoff|v1', units='m/s',val=20,lower=10,upper=150)
        self.declare_partials('takeoff|v1',['distance_continue','distance_abort','takeoff|v1','takeoff|vr'])

    def apply_nonlinear(self, inputs, outputs, residuals):
        speedtol = 1e-1
        disttol = 0
        #force the decision speed to zero
        if inputs['takeoff|vr'] < outputs['takeoff|v1'] + speedtol:
            residuals['takeoff|v1'] = inputs['takeoff|vr'] - outputs['takeoff|v1']
        else:
            residuals['takeoff|v1'] = inputs['distance_continue'] - inputs['distance_abort']

        #if you are within vtol on the correct side but the stopping distance bigger, use the regular mode
        if inputs['takeoff|vr'] >= outputs['takeoff|v1'] and inputs['takeoff|vr'] - outputs['takeoff|v1'] < speedtol and (inputs['distance_abort'] - inputs['distance_continue']) > disttol:
            residuals['takeoff|v1'] = inputs['distance_continue'] - inputs['distance_abort']


    def linearize(self, inputs, outputs, partials):
        speedtol = 1e-1
        disttol = 0

        if inputs['takeoff|vr'] < outputs['takeoff|v1'] + speedtol:
            partials['takeoff|v1','distance_continue'] = 0
            partials['takeoff|v1','distance_abort'] = 0
            partials['takeoff|v1','takeoff|vr'] = 1
            partials['takeoff|v1','takeoff|v1'] = -1
        else:
            partials['takeoff|v1','distance_continue'] = 1
            partials['takeoff|v1','distance_abort'] = -1
            partials['takeoff|v1','takeoff|vr'] = 0
            partials['takeoff|v1','takeoff|v1'] = 0

        if inputs['takeoff|vr'] >= outputs['takeoff|v1'] and inputs['takeoff|vr'] - outputs['takeoff|v1'] < speedtol and (inputs['distance_abort'] - inputs['distance_continue']) > disttol:
            partials['takeoff|v1','distance_continue'] = 1
            partials['takeoff|v1','distance_abort'] = -1
            partials['takeoff|v1','takeoff|vr'] = 0
            partials['takeoff|v1','takeoff|v1'] = 0

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
        groundspeed =  np.sqrt(inputs['fltcond|Utrue']**2-inputs['fltcond|vs']**2)
        groundspeed_fixed= np.where(np.isnan(groundspeed),0,groundspeed)
        outputs['fltcond|groundspeed'] = groundspeed_fixed
        outputs['fltcond|singamma'] = np.where(np.isnan(groundspeed),1,inputs['fltcond|vs'] / inputs['fltcond|Utrue'])
        outputs['fltcond|cosgamma'] = groundspeed_fixed / inputs['fltcond|Utrue']

    def compute_partials(self, inputs, J):

        groundspeed =  np.sqrt(inputs['fltcond|Utrue']**2-inputs['fltcond|vs']**2)
        groundspeed_fixed= np.where(np.isnan(groundspeed),0,groundspeed)
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

class VerticalAcceleration(ExplicitComponent):
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

        self.add_output('accel_vert', units='m/s**2', shape=(nn,))
        arange=np.arange(nn)
        self.declare_partials(['accel_vert'], ['weight','drag','lift','thrust','fltcond|singamma','fltcond|cosgamma'], rows=arange, cols=arange)


    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        g = 9.80665 #m/s^2
        cosg = inputs['fltcond|cosgamma']
        sing = inputs['fltcond|singamma']
        accel = (inputs['lift']*cosg + (inputs['thrust']-inputs['drag'])*sing - g*inputs['weight'])/inputs['weight']
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

    This is a helper function for the main mission analysis routine `MissionNoReserves`
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
        self.add_input('fltcond|cosgamma', shape=(nn,))
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

class ODEGroundRoll(Group):
    """
    This adds general mission analysis capabilities to an existing airplane model.
    The BaseAircraftGroup object is passed in. It should be built to accept the following inputs and return the following outputs.
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
    propulsor_failed : float
        If a multi-propulsor airplane, a failure condition should be modeled in the propulsion model by multiplying throttle by propulsor_failed.
        It will generally be 1.0 unless a failure condition is being modeled, in which case it will be 0 (vector, dimensionless)
    braking : float
        Percentage brakes applied, from 0 to 1. Should not be applied in the air or nonphysical effects will result (vector, dimensionless)
    lift : float
        Lift force (vector, N)

    Outputs
    -------
    thrust : float
        Total thrust force produced by all propulsors (vector, N)
    drag : float
        Total drag force in the airplane axis produced by all sources of drag (vector, N)
    weight : float
        Weight (mass, really) of the airplane at each point in time. Generally will need to be integrated by Dymos as a state with a rate source (vector, kg)
    ac|geom|wing|S_ref
        Wing reference area (scalar, m**2)
    ac|aero|CLmax_flaps30
        CLmax with flaps in max takeoff position (scalar, dimensionless)
    ac|weights|MTOW
        Maximum takeoff weight (scalar, kg)
    """
    # the ground roll allows horizontal but not vertical acceleration
    # need to provide an independent variable fltcond|h and fltcond|vs in the component

    def initialize(self):
        self.options.declare('num_nodes',default=1)
        self.options.declare('flight_phase',default=None,desc='Phase of flight e.g. v0v1, cruise')
        self.options.declare('aircraft_model',default=None)

    def setup(self):
        nn = self.options['num_nodes']
        ivcomp = self.add_subsystem('const_settings', IndepVarComp(), promotes_outputs=["*"])
        # set CL = 0.1 for the ground roll per Raymer's book
        ivcomp.add_output('fltcond|CL', val=np.ones((nn,))*0.1)
        ivcomp.add_output('v1_vstall_mult',val=1.1)
        ivcomp.add_output('fltcond|h',val=np.zeros((nn,)),units='m')
        ivcomp.add_output('fltcond|vs',val=np.zeros((nn,)),units='m/s')
        ivcomp.add_output('zero_speed',val=2,units='m/s')


        flight_phase = self.options['flight_phase']
        if flight_phase == 'v0v1' or flight_phase == 'v1vr':
            ivcomp.add_output('braking',val=np.ones((nn,))*0.03)
            ivcomp.add_output('propulsor_failed',val=np.zeros((nn,)))
            ivcomp.add_output('throttle',val=np.ones((nn,)))
            if flight_phase == 'v0v1':
                zero_start = True
            else:
                zero_start= False
        elif flight_phase == 'v1v0':
            ivcomp.add_output('braking',val=0.4*np.ones((nn,)))
            ivcomp.add_output('propulsor_failed',val=np.ones((nn,)))
            ivcomp.add_output('throttle',val=np.zeros((nn,)))
            zero_start=False

        self.add_subsystem('atmos', ComputeAtmosphericProperties(num_nodes=nn, true_airspeed_in=True), promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('gs',Groundspeeds(num_nodes=nn),promotes_inputs=['*'],promotes_outputs=['*'])
        # add the user-defined aircraft model
        self.add_subsystem('acmodel',self.options['aircraft_model'](num_nodes=nn,flight_phase=self.options['flight_phase']),promotes_inputs=['*'],promotes_outputs=['*'])

        self.add_subsystem('lift',Lift(num_nodes=nn), promotes_inputs=['*'],promotes_outputs=['*'])
        self.add_subsystem('stall',StallSpeed(),promotes_inputs=[('CLmax','ac|aero|CLmax_flaps30'),('weight','ac|weights|MTOW'),'ac|geom|wing|S_ref'],promotes_outputs=['*'])
        self.add_subsystem('vrspeed',ElementMultiplyDivideComp(output_name='takeoff|vr',input_names=['Vstall_eas','v1_vstall_mult'],input_units=['m/s',None]),promotes_inputs=['*'],promotes_outputs=['*'])


        self.add_subsystem('haccel',HorizontalAcceleration(num_nodes=nn), promotes_inputs=['*'],promotes_outputs=['*'])
        nn_simpson = int((nn-1)/2)
        if flight_phase == 'v1v0':
            #unfortunately need to shoot backwards to avoid negative airspeeds
            #reverse the order of the accelerations so the last one is first (and make them negative)
            self.add_subsystem('flipaccel', FlipVectorComp(num_nodes=nn, units='m/s**2', negative=True), promotes_inputs=[('vec_in','accel_horiz')])
            #integrate the timesteps in reverse from near zero speed.
            self.add_subsystem('intvelocity',Integrator(num_intervals=nn_simpson, method='simpson', quantity_units='m/s', diff_units='s',time_setup='duration', lower=1.5),
                                                        promotes_inputs=['duration',('q_initial','zero_speed')],promotes_outputs=[('q_final','fltcond|Utrue_initial')])
            self.connect('flipaccel.vec_out','intvelocity.dqdt')
            #flip the result of the reverse integration again so the flight condition is forward and consistent with everythign else
            self.add_subsystem('flipvel', FlipVectorComp(num_nodes=nn, units='m/s', negative=False), promotes_outputs=[('vec_out','fltcond|Utrue')])
            self.connect('intvelocity.q','flipvel.vec_in')
            # now set the time step so that backwards shooting results in the correct 'initial' segment airspeed
            self.add_subsystem('v0constraint',BalanceComp(name='duration',units='s',eq_units='m/s',rhs_name='fltcond|Utrue_initial',lhs_name='takeoff|v1',val=10.,upper=100.,lower=1.),
                                       promotes_inputs=['*'],promotes_outputs=['duration'])
        else:
            # forward shooting for these acceleration segmentes
            self.add_subsystem('intvelocity',Integrator(num_intervals=nn_simpson, method='simpson', quantity_units='m/s', diff_units='s', time_setup='duration', lower=1.5),
                                                        promotes_inputs=[('dqdt','accel_horiz'),'duration',('q_initial','fltcond|Utrue_initial')],promotes_outputs=[('q','fltcond|Utrue'),('q_final','fltcond|Utrue_final')])
            if flight_phase == 'v0v1':
                self.connect('zero_speed','fltcond|Utrue_initial')
                self.add_subsystem('v1constraint',BalanceComp(name='duration',units='s',eq_units='m/s',rhs_name='fltcond|Utrue_final',lhs_name='takeoff|v1',val=10.,upper=100.,lower=1.),
                               promotes_inputs=['*'],promotes_outputs=['duration'])
            elif flight_phase == 'v1vr':
                self.add_subsystem('vrconstraint',BalanceComp(name='duration',units='s',eq_units='m/s',rhs_name='fltcond|Utrue_final',lhs_name='takeoff|vr',val=5.,upper=100.,lower=0.0),
                               promotes_inputs=['*'],promotes_outputs=['duration'])

        if zero_start:
            self.add_subsystem('intrange',Integrator(num_intervals=nn_simpson, method='simpson', quantity_units='m', diff_units='s',zero_start=zero_start, time_setup='duration'),
                                                        promotes_inputs=[('dqdt','fltcond|groundspeed'),'duration'],promotes_outputs=[('q','range'),('q_final','range_final')])
        else:
            self.add_subsystem('intrange',Integrator(num_intervals=nn_simpson, method='simpson', quantity_units='m', diff_units='s',zero_start=zero_start, time_setup='duration'),
                                                    promotes_inputs=[('dqdt','fltcond|groundspeed'),'duration',('q_initial','range_initial')],promotes_outputs=[('q','range'),('q_final','range_final')])


class ODERotate(Group):
    """
    This adds general mission analysis capabilities to an existing airplane model.
    The BaseAircraftGroup object is passed in. It should be built to accept the following inputs and return the following outputs.
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
    propulsor_failed : float
        If a multi-propulsor airplane, a failure condition should be modeled in the propulsion model by multiplying throttle by propulsor_failed.
        It will generally be 1.0 unless a failure condition is being modeled, in which case it will be 0 (vector, dimensionless)
    braking : float
        Percentage brakes applied, from 0 to 1. Should not be applied in the air or nonphysical effects will result (vector, dimensionless)
    lift : float
        Lift force (vector, N)

    Outputs
    -------
    thrust : float
        Total thrust force produced by all propulsors (vector, N)
    drag : float
        Total drag force in the airplane axis produced by all sources of drag (vector, N)
    weight : float
        Weight (mass, really) of the airplane at each point in time. Generally will need to be integrated by Dymos as a state with a rate source (vector, kg)
    ac|geom|wing|S_ref
        Wing reference area (scalar, m**2)
    ac|aero|CLmax_flaps30
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
        ivcomp.add_output('CL_rotate_mult', val=np.ones((nn,))*0.83)
        ivcomp.add_output('h_obs', val=50, units='ft')
        flight_phase = self.options['flight_phase']
        if flight_phase == 'rotate':
            ivcomp.add_output('braking',val=np.zeros((nn,)))
            ivcomp.add_output('propulsor_failed',val=np.zeros((nn,)))
            ivcomp.add_output('throttle',val=np.ones((nn,)))

        self.add_subsystem('atmos', ComputeAtmosphericProperties(num_nodes=nn, true_airspeed_in=True), promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('gs',Groundspeeds(num_nodes=nn),promotes_inputs=['*'],promotes_outputs=['*'])
        # add the user-defined aircraft model
        clcomp = self.add_subsystem('clcomp',ElementMultiplyDivideComp(output_name='fltcond|CL', input_names=['CL_rotate_mult','ac|aero|CLmax_flaps30'],
                                                                       vec_size=[nn,1], length=1),
                                    promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('acmodel',self.options['aircraft_model'](num_nodes=nn,flight_phase=self.options['flight_phase']),promotes_inputs=['*'],promotes_outputs=['*'])


        self.add_subsystem('lift',Lift(num_nodes=nn), promotes_inputs=['*'],promotes_outputs=['*'])
        self.add_subsystem('haccel',HorizontalAcceleration(num_nodes=nn), promotes_inputs=['*'],promotes_outputs=['*'])
        self.add_subsystem('vaccel',VerticalAcceleration(num_nodes=nn), promotes_inputs=['*'],promotes_outputs=['*'])
        # fix CL to be 0.9 * CLmax during the rotation phase per Raymer's methods
        nn_simpson = int((nn-1)/2)
        self.add_subsystem('clear_obstacle',BalanceComp(name='duration',units='s',val=1,eq_units='m',rhs_name='fltcond|h_final',lhs_name='h_obs',lower=0.1),
                       promotes_inputs=['*'],promotes_outputs=['duration'])
        self.add_subsystem('intvelocity',Integrator(num_intervals=nn_simpson, method='simpson', quantity_units='m/s', diff_units='s',time_setup='duration',lower=0.1),
                                                    promotes_inputs=[('dqdt','accel_horiz'),'duration',('q_initial','fltcond|Utrue_initial')],promotes_outputs=[('q','fltcond|Utrue'),('q_final','fltcond|Utrue_final')])
        self.add_subsystem('intrange',Integrator(num_intervals=nn_simpson, method='simpson', quantity_units='m', diff_units='s', time_setup='duration'),
                                                    promotes_inputs=[('dqdt','fltcond|groundspeed'),'duration',('q_initial','range_initial')],promotes_outputs=[('q','range'),('q_final','range_final')])
        self.add_subsystem('intvs',Integrator(num_intervals=nn_simpson, method='simpson', quantity_units='m/s', diff_units='s', time_setup='duration',zero_start=True),
                                                    promotes_inputs=[('dqdt','accel_vert'),'duration'],promotes_outputs=[('q','fltcond|vs'),('q_final','fltcond|vs_final')])
        self.add_subsystem('inth',Integrator(num_intervals=nn_simpson, method='simpson', quantity_units='m', diff_units='s', time_setup='duration',zero_start=True),
                                                    promotes_inputs=[('dqdt','fltcond|vs'),'duration'],promotes_outputs=[('q','fltcond|h'),('q_final','fltcond|h_final')])


class ODESteady(Group):
    def initialize(self):
        self.options.declare('num_nodes',default=1)
        self.options.declare('flight_phase',default=None,desc='Phase of flight e.g. v0v1, cruise')
        self.options.declare('aircraft_model',default=None)

    def setup(self):
        nn = self.options['num_nodes']
        ivcomp = self.add_subsystem('const_settings', IndepVarComp(), promotes_outputs=["*"])
        ivcomp.add_output('propulsor_failed', val=np.zeros(nn))
        ivcomp.add_output('braking', val=np.zeros(nn))
        ivcomp.add_output('fltcond|Ueas',val=np.ones((nn,))*90, units='m/s')
        ivcomp.add_output('fltcond|vs',val=np.ones((nn,))*1, units='m/s')
        ivcomp.add_output('zero_accel',val=np.zeros((nn,)),units='m/s**2')
        nn_simpson = int((nn-1)/2)
        self.add_subsystem('inth',Integrator(num_intervals=nn_simpson, method='simpson', quantity_units='m', diff_units='s', time_setup='duration'),
                                                    promotes_inputs=[('dqdt','fltcond|vs'),'duration',('q_initial','fltcond|h_initial')],promotes_outputs=[('q','fltcond|h'),('q_final','fltcond|h_final')])
        self.add_subsystem('atmos', ComputeAtmosphericProperties(num_nodes=nn, true_airspeed_in=False), promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('gs',Groundspeeds(num_nodes=nn),promotes_inputs=['*'],promotes_outputs=['*'])
        # add the user-defined aircraft model
        self.add_subsystem('acmodel',self.options['aircraft_model'](num_nodes=nn, flight_phase=self.options['flight_phase']),promotes_inputs=['*'],promotes_outputs=['*'])
        self.add_subsystem('clcomp',SteadyFlightCL(num_nodes=nn), promotes_inputs=['*'],promotes_outputs=['*'])
        self.add_subsystem('lift',Lift(num_nodes=nn), promotes_inputs=['*'],promotes_outputs=['*'])
        self.add_subsystem('haccel',HorizontalAcceleration(num_nodes=nn), promotes_inputs=['*'],promotes_outputs=['*'])

        self.add_subsystem('intrange',Integrator(num_intervals=nn_simpson, method='simpson', quantity_units='m', diff_units='s', time_setup='duration'),
                                                    promotes_inputs=[('dqdt','fltcond|groundspeed'),'duration',('q_initial','range_initial')],promotes_outputs=[('q','range'),('q_final','range_final')])


        self.add_subsystem('steadyflt',BalanceComp(name='throttle',val=np.ones((nn,))*0.5,lower=0.01,upper=1.4,units=None,eq_units='m/s**2',rhs_name='accel_horiz',lhs_name='zero_accel',rhs_val=np.zeros((nn,))),
                           promotes_inputs=['accel_horiz','zero_accel'],promotes_outputs=['throttle'])
