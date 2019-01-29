"""Analysis routines for simulating the takeoff phase and determining takeoff field length"""
from __future__ import division
from openmdao.api import Problem, Group, IndepVarComp
from openmdao.api import ScipyOptimizeDriver, ExplicitComponent, ImplicitComponent

import numpy as np
import scipy.sparse as sp

from openconcept.analysis.atmospherics.compute_atmos_props import ComputeAtmosphericProperties
from openconcept.utilities.math.simpson_integration import IntegrateQuantity
from openconcept.utilities.math import AddSubtractComp, ElementMultiplyDivideComp, VectorConcatenateComp, VectorSplitComp
from openconcept.analysis.aerodynamics import PolarDrag, Lift, StallSpeed
from openconcept.utilities.dvlabel import DVLabel


class BalancedFieldLengthTakeoff(Group):
    """This analysis group calculates takeoff field length and fuel/energy consumption.

    This component should be instantiated in the top-level aircraft analysis / optimization script.

    **Suggested variable promotion list:**
    *'ac|aero\*', 'ac|weights|MTOW', 'ac|geom|\*', 'fltcond|\*', 'takeoff|battery_load',*
    *'takeoff|thrust','takeoff|fuel_flow','mission|takeoff|v\*'*

    **Inputs List:**

    From aircraft config:
        - ac|aero|polar|CD0_TO
        - ac|aero|polar|e
        - ac|geom|wing|S_ref
        - ac|geom|wing|AR
        - ac|weights|MTOW

    From Newton solver:
        - takeoff|v1

    From takeoff flight condition generator:
        - takeoff|vr

    From standard atmosphere model/splitter:
        - fltcond|q
        - fltcond|Utrue

    From propulsion model:
        - takeoff|battery_load
        - takeoff|fuel_flow
        - takeoff|thrust

    Outputs
    -------
    total_fuel : float
        Total fuel burn for takeoff (scalar, kg)
    total_battery_energy : float
        Total energy consumption for takeoff (scalar, kJ)
    distance_continue : float
        Takeoff distance with given propulsion settings (scalar, m)
    distance_abort : float
        Takeoff distance if maximum braking applied at v1 speed (scalar, m)

    Options
    -------
    n_int_per_seg : int
        Number of Simpson's rule intervals to use per mission segment.
        The total number of points is 2 * n_int_per_seg + 1
    track_battery : bool
        Set to `True` to track battery energy consumption during takeoff (default False)
    track_fuel : bool
        Set to `True` to track fuel burned during takeoff (default False)

    """
    def initialize(self):
        self.options.declare('n_int_per_seg',default=5,desc="Number of Simpson intervals to use per seg. Number of analysis points is 2N+1")
        self.options.declare('track_battery',default=False,desc='Set to true to enable battery inputs')
        self.options.declare('track_fuel',default=False,desc='Set to true to enable fuel inputs')
        self.options.declare('propulsion_system',default=None)

    def setup(self):
        n_int_per_seg = self.options['n_int_per_seg']
        track_battery = self.options['track_battery']
        track_fuel = self.options['track_fuel']
        nn = (n_int_per_seg*2 + 1)
        nn_tot = 3 * nn+2
        #===Some of the generic components take "weight" as an input. Need to re-label Takeoff Weight (TOW) as just weight
        dvlist = [['ac|weights|MTOW','weight',2000.0,'kg'],
                    ['takeoff|v1','v1',40,'m/s'],
                    ['takeoff|vr','vr',45,'m/s'],
                    ['ac|geom|wing|S_ref','S_ref',20,'m ** 2'],
                    ['ac|aero|CLmax_flaps30','CLmax_flaps30',1.5,None],
                    ['ac|aero|polar|CD0_TO','CD0',0.005,None],
                    ['ac|aero|polar|e','e',0.95,None]]

        self.add_subsystem('dvs',DVLabel(dvlist),promotes_inputs=["*"],promotes_outputs=["*"])
        self.add_subsystem('vstall', StallSpeed())
        self.connect('weight','vstall.weight')
        self.connect('S_ref','vstall.ac|geom|wing|S_ref')
        self.connect('CLmax_flaps30', 'vstall.CLmax')

        takeoff_conditions = self.add_subsystem('conditions',
                                                TakeoffFlightConditions(n_int_per_seg=n_int_per_seg),
                                                promotes_inputs =['takeoff|v1','takeoff|h'],
                                                promotes_outputs=["fltcond|*",
                                                                  "takeoff|*"])
        self.connect('vstall.Vstall_eas','conditions.Vstall_eas')

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
            propulsion_promotes_inputs.append('ac|weights|*')
        self.add_subsystem('propmodel',self.options['propulsion_system'],
                           promotes_inputs=propulsion_promotes_inputs,promotes_outputs=propulsion_promotes_outputs)


        #===We assume the takeoff starts at 1 m/s to avoid singularities at v=0
        const = self.add_subsystem('const',IndepVarComp())
        const.add_output('v0', val=1, units='m/s')
        const.add_output('reaction_time', val=2, units='s')

        #===Lift and Drag Calculations feed the EOMs for the takeoff roll===
        self.add_subsystem('CL',TakeoffCLs(n_int_per_seg=n_int_per_seg),promotes_inputs=["ac|geom|*","fltcond|*","weight"],)
        self.add_subsystem('drag',PolarDrag(num_nodes=nn_tot),promotes_inputs=['ac|geom|*','fltcond|q','CD0','e'],promotes_outputs=['drag'])
        self.add_subsystem('lift',Lift(num_nodes=nn_tot),promotes_inputs=['ac|geom|*','fltcond|q'],promotes_outputs=['lift'])
        self.connect('CL.CL_takeoff','drag.fltcond|CL')
        self.connect('CL.CL_takeoff','lift.fltcond|CL')

        #==The takeoff distance integrator numerically integrates the quantity (speed / acceleration) with respect to speed to obtain distance. Obtain this quantity:
        self.add_subsystem('accel',TakeoffAccels(n_int_per_seg=n_int_per_seg),promotes_inputs=['lift','drag','thrust','weight'],promotes_outputs=['_inverse_accel'])
        self.add_subsystem('mult',ElementMultiplyDivideComp(output_name='_rate_to_integrate',
                                                            input_names=['_inverse_accel','fltcond|Utrue'],
                                                            vec_size=nn_tot,
                                                            input_units=['s**2/m','m/s']),
                                                            promotes_inputs=['*'],promotes_outputs=['*'])
        if track_fuel:
            self.add_subsystem('fuelflowmult',ElementMultiplyDivideComp(output_name='_fuel_flow_rate_to_integrate',
                                                                        input_names=['_inverse_accel','fuel_flow'],
                                                                        vec_size=nn_tot,
                                                                        input_units=['s**2/m','kg/s']),
                                                                        promotes_inputs=['*'],promotes_outputs=['_*'])
        if track_battery:
            self.add_subsystem('batterymult',ElementMultiplyDivideComp(output_name='_battery_rate_to_integrate',
                                                                       input_names=['_inverse_accel','battery_load'],
                                                                       vec_size=nn_tot,
                                                                       input_units=['s**2/m','J / s']),
                                                                       promotes_inputs=['*'],promotes_outputs=['*'])

        #==The following boilerplate splits the input flight conditions, thrusts, and fuel flows into the takeoff segments for further analysis
        inputs_to_split = ['_rate_to_integrate','fltcond|Utrue','thrust','drag','lift']
        units = ['s','m/s','N','N','N']

        if track_battery:
            inputs_to_split.append('_battery_rate_to_integrate')
            units.append('J*s/m')
        if track_fuel:
            inputs_to_split.append('_fuel_flow_rate_to_integrate')
            units.append('kg*s/m')
        segments_to_split_into = ['v0v1','v1vr','v1v0','vtrans','v2']
        nn_each_segment = [nn,nn,nn,1,1]
        split_inst = VectorSplitComp()
        for kth, input_name in enumerate(inputs_to_split):
            output_names_list = []
            for segment in segments_to_split_into:
                output_names_list.append(input_name+'_'+segment)
            split_inst.add_relation(output_names=output_names_list, input_name=input_name, vec_sizes=nn_each_segment, units=units[kth])
        splitter = self.add_subsystem('splitter',subsys=split_inst,promotes_inputs=["*"],promotes_outputs=["*"])

        #==Now integrate the three continuous segments: 0 to v1, v1 to rotation with reduced power if applicable, and hard braking
        self.add_subsystem('v0v1_dist',IntegrateQuantity(num_intervals=n_int_per_seg,quantity_units='m',diff_units='m/s',force_signs=True))
        self.add_subsystem('v1vr_dist',IntegrateQuantity(num_intervals=n_int_per_seg,quantity_units='m',diff_units='m/s',force_signs=True))
        self.add_subsystem('v1v0_dist',IntegrateQuantity(num_intervals=n_int_per_seg,quantity_units='m',diff_units='m/s',force_signs=True))
        self.connect('_rate_to_integrate_v0v1','v0v1_dist.rate')
        self.connect('_rate_to_integrate_v1vr','v1vr_dist.rate')
        self.connect('_rate_to_integrate_v1v0','v1v0_dist.rate')
        self.connect('const.v0','v0v1_dist.lower_limit')
        self.connect('v1','v0v1_dist.upper_limit')
        self.connect('v1','v1v0_dist.lower_limit')
        self.connect('const.v0','v1v0_dist.upper_limit')
        self.connect('v1','v1vr_dist.lower_limit')
        self.connect('vr','v1vr_dist.upper_limit')

        if track_fuel:
            self.add_subsystem('v0v1_fuel',IntegrateQuantity(num_intervals=n_int_per_seg,quantity_units='kg',diff_units='m/s',force_signs=True))
            self.add_subsystem('v1vr_fuel',IntegrateQuantity(num_intervals=n_int_per_seg,quantity_units='kg',diff_units='m/s',force_signs=True))
            self.connect('_fuel_flow_rate_to_integrate_v0v1','v0v1_fuel.rate')
            self.connect('_fuel_flow_rate_to_integrate_v1vr','v1vr_fuel.rate')
            self.connect('const.v0','v0v1_fuel.lower_limit')
            self.connect('v1',['v0v1_fuel.upper_limit','v1vr_fuel.lower_limit'])
            self.connect('vr','v1vr_fuel.upper_limit')

        if track_battery:
            self.add_subsystem('v0v1_battery',IntegrateQuantity(num_intervals=n_int_per_seg,quantity_units='J',diff_units='m/s',force_signs=True))
            self.add_subsystem('v1vr_battery',IntegrateQuantity(num_intervals=n_int_per_seg,quantity_units='J',diff_units='m/s',force_signs=True))
            self.connect('_battery_rate_to_integrate_v0v1','v0v1_battery.rate')
            self.connect('_battery_rate_to_integrate_v1vr','v1vr_battery.rate')
            self.connect('const.v0','v0v1_battery.lower_limit')
            self.connect('v1',['v0v1_battery.upper_limit','v1vr_battery.lower_limit'])
            self.connect('vr','v1vr_battery.upper_limit')



        #==Next compute the transition and climb phase to the specified clearance height. First, need the steady climb-out angle at v2 speed
        self.add_subsystem('gamma',TakeoffV2ClimbAngle(),promotes_inputs=["drag_v2","thrust_v2","weight"],promotes_outputs=["climb|gamma"])
        self.add_subsystem('transition',TakeoffTransition(),promotes_inputs=["fltcond|Utrue_vtrans","climb|gamma"],promotes_outputs=["h_transition","s_transition"])
        self.add_subsystem('climb',TakeoffClimb(),promotes_inputs=["climb|gamma","h_transition"],promotes_outputs=["s_climb"])
        self.add_subsystem('reaction',ElementMultiplyDivideComp(output_name='s_reaction',
                                                                input_names=['v1','reaction_time'],
                                                                vec_size=1,
                                                                input_units=['m/s','s']),promotes_inputs=['v1'])
        self.connect('const.reaction_time','reaction.reaction_time')

        self.add_subsystem('total_to_distance_continue',AddSubtractComp(output_name='distance_continue',input_names=['s_v0v1','s_v1vr','s_reaction','s_transition','s_climb'],vec_size=1, units='m'),promotes_outputs=["*"])
        self.add_subsystem('total_to_distance_abort',AddSubtractComp(output_name='distance_abort',input_names=['s_v0v1','s_v1v0','s_reaction'],vec_size=1, units='m'),promotes_outputs=["*"])

        self.connect('reaction.s_reaction','total_to_distance_continue.s_reaction')
        self.connect('reaction.s_reaction','total_to_distance_abort.s_reaction')
        self.connect('v0v1_dist.delta_quantity',['total_to_distance_continue.s_v0v1','total_to_distance_abort.s_v0v1'])
        self.connect('v1vr_dist.delta_quantity','total_to_distance_continue.s_v1vr')
        self.connect('s_transition','total_to_distance_continue.s_transition')
        self.connect('s_climb','total_to_distance_continue.s_climb')
        self.connect('v1v0_dist.delta_quantity','total_to_distance_abort.s_v1v0')

        if track_battery:
            self.add_subsystem('total_battery',AddSubtractComp(output_name='total_battery_energy',input_names=['battery_v0v1','battery_v1vr'], units='J'),promotes_outputs=["*"])
            self.connect('v0v1_battery.delta_quantity','total_battery.battery_v0v1')
            self.connect('v1vr_battery.delta_quantity','total_battery.battery_v1vr')
        if track_fuel:
            self.add_subsystem('total_fuel',AddSubtractComp(output_name='total_fuel',input_names=['fuel_v0v1','fuel_v1vr'], units='kg',scaling_factors=[1,1]),promotes_outputs=["*"])
            self.connect('v0v1_fuel.delta_quantity','total_fuel.fuel_v0v1')
            self.connect('v1vr_fuel.delta_quantity','total_fuel.fuel_v1vr')
            self.add_subsystem('climb_weight',AddSubtractComp(output_name='weight_after_takeoff',input_names=['weight','total_fuel'], units='kg',scaling_factors=[1,-1]),promotes_inputs=["*"],promotes_outputs=["*"])
        self.add_subsystem('v1_solve', BFLImplicitSolve(), promotes_inputs=['*'], promotes_outputs=['*'])


def takeoff_check(prob):
    """
    Checks to ensure positive accelerations during each takeoff phase.

    In some cases, the numeric integration scheme used to calculate TOFL can give a spurious result
    if the airplane can't accelerate through to V1. This function detects this case and raises an error.
    It should be called following every model.run_driver or run_model call.

    Arguments
    ---------
    prob : OpenMDAO problem object
        The OpenMDAO problem object

    Inputs
    ------
    'takeoff._rate_to_integrate_v0v1' : float
    'takeoff._rate_to_integrate_v1vr' : float
    'takeoff._rate_to_integrate_v1v0' : float

    Raises
    -------
    ValueError if negative distances are produced

    """
    v0v1 = prob['takeoff._rate_to_integrate_v0v1']
    v1vr = prob['takeoff._rate_to_integrate_v1vr']
    v1v0 = prob['takeoff._rate_to_integrate_v1v0']
    if np.sum(v0v1 < 0) > 0:
        raise ValueError('The aircraft was unable to reach v1 speed at the optimized design point. '
                         'Restrict design variables to add power or reenable takeoff constraints')
    if np.sum(v1vr < 0) > 0:
        raise ValueError('The aircraft was unable to accelerate to vr from v1 (try adding power), '
                         'or the v1 speed is higher than vr')
    if np.sum(v1v0 > 0) < 0:
        raise ValueError('The aircraft continues to accelerate even after heavy braking in '
                         'the abort phase of takeoff. Check your RTO throttle settings '
                         '(should be near zero thrust)')

class ComputeBalancedFieldLengthResidual(ExplicitComponent):
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
    takeoff|v1 : float
        Decision speed (scalar, m/s)
    takeoff|vr : float
        Rotation speed (scalar, m/s)

    Outputs
    -------
    BFL_residual : float
        Difference between OEI TO distance and RTO distance for diagnostic purposes (scalar, m/s)
    v1vr_diff : float
        Difference between decision and rotation speed for diagnostic purposes (scalar, m/s)
    BFL_combined : float
        Residual equation combining both criteria with special partial derivatives.
        Should be used for the Newton solver when doing takeoff field length analysis
        (scalar, m)
    """
    def setup(self):
        self.add_input('distance_continue', units='m')
        self.add_input('distance_abort', units='m')
        self.add_input('takeoff|v1', units='m/s')
        self.add_input('takeoff|vr', units='m/s')
        self.add_output('BFL_residual', units='m')
        self.add_output('v1vr_diff', units='m/s')
        self.add_output('BFL_combined', units='m')
        self.declare_partials('BFL_residual','distance_continue', val=1)
        self.declare_partials('BFL_residual','distance_abort', val=-1)
        self.declare_partials('v1vr_diff','takeoff|vr', val=1)
        self.declare_partials('v1vr_diff','takeoff|v1', val=-1)
        self.declare_partials('BFL_combined',['distance_continue','distance_abort','takeoff|v1','takeoff|vr'])
    def compute(self, inputs, outputs):
        outputs['BFL_residual'] = inputs['distance_continue'] - inputs['distance_abort']
        outputs['v1vr_diff'] = inputs['takeoff|vr'] - inputs['takeoff|v1']
        speedtol = 1e-1
        disttol = 0
        #force the decision speed to zero
        if inputs['takeoff|vr'] < inputs['takeoff|v1'] + speedtol:
            outputs['BFL_combined'] = inputs['takeoff|vr'] - inputs['takeoff|v1']
        else:
            outputs['BFL_combined'] = inputs['distance_continue'] - inputs['distance_abort']
        #if you are within vtol on the correct side but the stopping distance bigger, use the regular mode
        if inputs['takeoff|vr'] >= inputs['takeoff|v1'] and inputs['takeoff|vr'] - inputs['takeoff|v1'] < speedtol and (inputs['distance_abort'] - inputs['distance_continue']) > disttol:
            outputs['BFL_combined'] = inputs['distance_continue'] - inputs['distance_abort']


    def compute_partials(self, inputs, partials):
        speedtol = 1e-1
        disttol = 0

        if inputs['takeoff|vr'] < inputs['takeoff|v1'] + speedtol:
            partials['takeoff|v1','distance_continue'] = 0
            partials['takeoff|v1','distance_abort'] = 0
            partials['takeoff|v1','takeoff|vr'] = 1
            partials['takeoff|v1','takeoff|v1'] = -1
        else:
            partials['takeoff|v1','distance_continue'] = 1
            partials['takeoff|v1','distance_abort'] = -1
            partials['takeoff|v1','takeoff|vr'] = 0
            partials['takeoff|v1','takeoff|v1'] = 0

        if inputs['takeoff|vr'] >= inputs['takeoff|v1'] and inputs['takeoff|vr'] - inputs['takeoff|v1'] < speedtol and (inputs['distance_abort'] - inputs['distance_continue']) > disttol:
            partials['takeoff|v1','distance_continue'] = 1
            partials['takeoff|v1','distance_abort'] = -1
            partials['takeoff|v1','takeoff|vr'] = 0
            partials['takeoff|v1','takeoff|v1'] = 0

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
        self.add_output('takeoff|v1', units='m/s')
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

class TakeoffFlightConditions(ExplicitComponent):
    """
    Generates flight condition vectors for takeoff segments

    Inputs
    ------
    takeoff|h : float
        Runway altitude (scalar, m)
    takeoff|v1 : float
        Takeoff decision speed (scalar, m/s)
    Vstall_eas : float
        Flaps down stall airspeed (scalar, m/s)

    Outputs
    -------
    takeoff|vr
        Takeoff rotation speed (set as multiple of stall speed). (scalar, m/s)
    takeoff|v2
        Takeoff safety speed (set as multiple of stall speed). (scalar, m/s)
    fltcond|Ueas
        Takeoff indicated/equiv. airspeed (vector, m/s)
    fltcond|h
        Takeoff altitude turned into a vector (vector, m/s)

    Options
    -------
    n_int_per_seg : int
        Number of Simpson's rule intervals to use per mission segment.
        The total number of points is 2 * n_int_per_seg + 1
    vr_multiple : float
        Rotation speed multiplier on top of stall speed (default 1.1)
    v2_multiple : float
        Climb out safety speed multiplier on top of stall speed (default 1.2)
    """

    def initialize(self):
        self.options.declare('n_int_per_seg',default=5,desc="Number of Simpson intervals to use per seg. Number of analysis points is 2N+1")
        self.options.declare('vr_multiple',default=1.1, desc="Rotation speed multiple of Vstall")
        self.options.declare('v2_multiple',default=1.2, desc='Climb out multipile of Vstall')

    def setup(self):
        n_int_per_seg = self.options['n_int_per_seg']
        nn = (n_int_per_seg*2 + 1)
        vrmult = self.options['vr_multiple']
        v2mult = self.options['v2_multiple']
        self.add_input('takeoff|h', val=0, units='m', desc='Takeoff runway altitude')
        self.add_input('takeoff|v1', val=75, units='m / s', desc='Takeoff decision airspeed')
        self.add_input('Vstall_eas', val=90, units='m / s', desc='Flaps down stall airspeed')
        self.add_output('takeoff|vr', units='m/s', desc='Takeoff rotation speed')
        self.add_output('takeoff|v2', units='m/s', desc='Takeoff climbout speed')

        ## the total length of the vector is 3 * nn (3 integrated segments) + 2 scalar flight conditions
        # condition 1: v0 to v1 [nn points]
        # condition 2: v1 to vr (one engine out) [nn points]
        # condition 3: v1 to v0 (with braking) [nn points]
        # condition 4: average between vr and v2 [1 point]
        # condition 5: v2 [1 point]
        self.add_output('fltcond|Ueas', units='m / s', desc='indicated airspeed at each analysis point',shape=(3 * nn+2,))
        self.add_output('fltcond|h', units='m', desc='altitude at each analysis point',shape=(3 * nn+2,))
        linear_nn = np.linspace(0,1,nn)
        linear_rev_nn = np.linspace(1,0,nn)

        #the climb speeds only have influence over their respective mission segments
        self.declare_partials(['fltcond|Ueas'], ['takeoff|v1'], val=np.concatenate([linear_nn,linear_rev_nn,linear_rev_nn,np.zeros(2)]))
        self.declare_partials(['fltcond|Ueas'], ['Vstall_eas'], val=np.concatenate([np.zeros(nn),linear_nn*vrmult,np.zeros(nn),np.array([(vrmult+v2mult)/2,v2mult])]))
        self.declare_partials(['fltcond|h'], ['takeoff|h'], val=np.ones(3 * nn+2))
        self.declare_partials(['takeoff|vr'], ['Vstall_eas'], val=vrmult)
        self.declare_partials(['takeoff|v2'], ['Vstall_eas'], val=v2mult)


    def compute(self, inputs, outputs):
        n_int_per_seg = self.options['n_int_per_seg']
        nn = n_int_per_seg*2 + 1
        vrmult = self.options['vr_multiple']
        v2mult = self.options['v2_multiple']
        v1 = inputs['takeoff|v1'][0]
        vstall = inputs['Vstall_eas'][0]
        vr = vrmult*vstall
        v2 = v2mult*vstall
        outputs['takeoff|vr'] = vr
        outputs['takeoff|v2'] = v2
        outputs['fltcond|h'] = inputs['takeoff|h']*np.ones(3 * nn+2)
        outputs['fltcond|Ueas'] = np.concatenate([np.linspace(1,v1,nn),np.linspace(v1,vr,nn),np.linspace(v1,1,nn),np.array([(vr+v2)/2, v2])])


class TakeoffCLs(ExplicitComponent):
    """
    Computes lift coefficient at every takeoff and transition analysis point.

    This is a helper function for the main TOFL analysis group `TakeoffTotalDistance`
    and shoudln't be instantiated in the top-level model directly.

    During the ground roll, CL is assumed constant.
    During rotation and transition, a 1.2g maneuver is assumed

    Inputs
    ------
    weight : float
        Takeoff weight (scalar, kg)
    fltcond|q : float
        Dynamic pressure at each analysis point (vector, Pascals)
    ac|geom|wing|S_ref : float
        Wing reference area (scalar, m**2)

    Outputs
    -------
    CL_takeoff : float
        Wing lift coefficient at each TO analysis point (vector, dimensionless)

    Options
    -------
    n_int_per_seg : int
        Number of Simpson's rule intervals to use per mission segment.
        The total number of points is 2 * n_int_per_seg + 1
    ground_CL : float
        Assumed CL during takeoff roll (default 0.1)
    """
    def initialize(self):
        self.options.declare('n_int_per_seg',default=5,desc="Number of Simpson intervals to use per seg. Number of analysis points is 2N+1")
        self.options.declare('ground_CL',default=0.1,desc="Assumed CL during the takeoff roll")
    def setup(self):
        n_int_per_seg = self.options['n_int_per_seg']
        nn = (n_int_per_seg*2 + 1)
        nn_tot = 3 * nn+2
        self.add_input('weight', units='kg')
        self.add_input('fltcond|q', units='N * m**-2',shape=(nn_tot,))
        self.add_input('ac|geom|wing|S_ref', units='m**2')
        self.add_output('CL_takeoff',shape=(nn_tot,))
        #the partials only apply for the last two entries
        self.declare_partials(['CL_takeoff'], ['weight','ac|geom|wing|S_ref'], rows=np.arange(nn_tot-2,nn_tot), cols=np.zeros(2))
        self.declare_partials(['CL_takeoff'], ['fltcond|q'], rows=np.arange(nn_tot-2,nn_tot), cols=np.arange(nn_tot-2,nn_tot))

    def compute(self, inputs, outputs):
        n_int_per_seg = self.options['n_int_per_seg']
        nn = (n_int_per_seg*2 + 1)
        nn_tot = 3 * nn+2
        g = 9.80665 #m/s^2
        CLs = np.ones(nn_tot) * self.options['ground_CL']
        #the 1.2 load factor is an assumption from Raymer. May want to revisit or make an option /default in the future
        loadfactors = np.array([1.2,1.2])
        CLs[-2:] = loadfactors*inputs['weight'][-2:] * g / inputs['fltcond|q'][-2:] / inputs['ac|geom|wing|S_ref'][-2:]
        outputs['CL_takeoff'] = CLs
    def compute_partials(self, inputs, J):
        n_int_per_seg = self.options['n_int_per_seg']
        nn = (n_int_per_seg*2 + 1)
        nn_tot = 3 * nn+2
        g = 9.80665 #m/s^2
        #the 1.2 load factor is an assumption from Raymer. May want to revisit or make an option /default in the future
        loadfactors = np.array([1.2,1.2])
        J['CL_takeoff','weight'] = loadfactors * g / inputs['fltcond|q'][-2:] / inputs['ac|geom|wing|S_ref'][-2:]
        J['CL_takeoff','fltcond|q'] = - loadfactors*inputs['weight'][-2:] * g / inputs['fltcond|q'][-2:]**2 / inputs['ac|geom|wing|S_ref'][-2:]
        J['CL_takeoff','ac|geom|wing|S_ref'] = - loadfactors*inputs['weight'][-2:] * g / inputs['fltcond|q'][-2:] / inputs['ac|geom|wing|S_ref'][-2:]**2

class TakeoffAccels(ExplicitComponent):
    """
    Computes acceleration during takeoff run and returns the inverse for the integrator.

    This is a helper function for the main TOFL analysis group `TakeoffTotalDistance`
    and shoudln't be instantiated in the top-level model directly.

    This returns the **INVERSE** of the accelerations during the takeoff run.
    Inverse acceleration is required due to integration wrt velocity:
    int( dr/dt * dt / dv) dv = int( v / a) dv

    Inputs
    ------
    weight : float
        Takeoff weight (scalar, kg)
    drag : float
        Aircraft drag at each TO analysis point (vector, N)
    lift : float
        Aircraft lift at each TO analysis point (vector, N)
    thrust : float
        Thrust at each TO analysis point (vector, N)

    Outputs
    -------
    _inverse_accel : float
        Inverse of the acceleration at ecah time point (vector, s**2/m)

    Options
    -------
    n_int_per_seg : int
        Number of Simpson's rule intervals to use per mission segment.
        The total number of points is 2 * n_int_per_seg + 1
    free_rolling_friction_coeff : float
        Rolling coefficient without brakes applied (default 0.03)
    braking_friction_coeff : float
        Rolling coefficient with max braking applied (default 0.40)
    """
    def initialize(self):
        self.options.declare('n_int_per_seg',default=5,desc="Number of Simpson intervals to use per seg. Number of analysis points is 2N+1")
        self.options.declare('free_rolling_friction_coeff',default=0.03,desc='Rolling friction coefficient (no brakes)')
        self.options.declare('braking_friction_coeff',default=0.4,desc='Coefficient of friction whilst applying max braking')
    def setup(self):
        n_int_per_seg = self.options['n_int_per_seg']
        nn = (n_int_per_seg*2 + 1)
        nn_tot = 3 * nn+2
        self.add_input('weight', units='kg')
        self.add_input('drag', units='N',shape=(nn_tot,))
        self.add_input('lift', units='N',shape=(nn_tot,))
        self.add_input('thrust', units='N',shape=(nn_tot,))
        self.add_output('_inverse_accel', units='s**2 / m',shape=(nn_tot,))
        arange=np.arange(0,nn_tot-2)
        self.declare_partials(['_inverse_accel'], ['drag','lift','thrust'], rows=arange, cols=arange)
        self.declare_partials(['_inverse_accel'], ['weight'], rows=arange, cols=np.zeros(nn_tot-2))

    def compute(self, inputs, outputs):
        n_int_per_seg = self.options['n_int_per_seg']
        nn = (n_int_per_seg*2 + 1)
        nn_tot = 3 * nn+2
        mu_free = self.options['free_rolling_friction_coeff']
        mu_brake = self.options['braking_friction_coeff']
        mu = np.concatenate([np.ones(2*nn)*mu_free,np.ones(nn)*mu_brake,np.zeros(2)])
        g = 9.80665 #m/s^2
        forcebal = inputs['thrust'] - inputs['drag'] - mu*(inputs['weight']*g-inputs['lift'])
        #print(forcebal)
        # print('Force balance:'+str(forcebal))
        inv_accel = inputs['weight'] / forcebal
        inv_accel[-2:] = np.zeros(2)
        outputs['_inverse_accel'] = inv_accel

    def compute_partials(self, inputs, J):
        n_int_per_seg = self.options['n_int_per_seg']
        nn = (n_int_per_seg*2 + 1)
        nn_tot = 3 * nn+2
        mu_free = self.options['free_rolling_friction_coeff']
        mu_brake = self.options['braking_friction_coeff']
        mu = np.concatenate([np.ones(2*nn)*mu_free,np.ones(nn)*mu_brake,np.zeros(2)])
        g = 9.80665 #m/s^2
        forcebal = inputs['thrust'] - inputs['drag'] - mu*(inputs['weight']*g-inputs['lift'])
        #(f/g)' = (f'g-fg')/g^2 where f is weight and g is forcebal
        ddweight = (1*forcebal - inputs['weight']*-mu*g)/forcebal**2
        ddweight=ddweight[:-2]
        dddrag = -inputs['weight'] / forcebal**2 * (-1)
        dddrag = dddrag[:-2]
        ddlift = -inputs['weight'] / forcebal**2 *mu
        ddlift = ddlift[:-2]
        ddthrust = - inputs['weight'] / forcebal**2
        ddthrust = ddthrust[:-2]

        J['_inverse_accel','thrust'] = ddthrust
        J['_inverse_accel','drag'] = dddrag
        J['_inverse_accel','lift'] = ddlift
        J['_inverse_accel','weight'] = ddweight

class TakeoffV2ClimbAngle(ExplicitComponent):
    """
    Computes climb out angle based on excess thrust.

    This is a helper function for the main TOFL analysis group `TakeoffTotalDistance`
    and shoudln't be instantiated in the top-level model directly.


    Inputs
    ------
    drag_v2 : float
        Aircraft drag at v2 (climb out) flight condition (scalar, N)
    weight : float
        Takeoff weight (scalar, kg)
    thrust_v2 : float
        Thrust at the v2 (climb out) flight condition (scalar, N)

    Outputs
    -------
    climb|gamma : float
        Climb out flight path angle (scalar, rad)

    Options
    -------
    """
    def setup(self):
        self.add_input('drag_v2', units='N')
        self.add_input('weight', units='kg')
        self.add_input('thrust_v2', units='N')
        self.add_output('climb|gamma', units='rad')

        self.declare_partials(['climb|gamma'], ['weight','thrust_v2','drag_v2'])

    def compute(self, inputs, outputs):
        g = 9.80665 #m/s^2
        outputs['climb|gamma'] = np.arcsin((inputs['thrust_v2']-inputs['drag_v2'])/inputs['weight']/g)

    def compute_partials(self, inputs, J):
        g = 9.80665 #m/s^2
        interior_qty = (inputs['thrust_v2']-inputs['drag_v2'])/inputs['weight']/g
        d_arcsin = 1/np.sqrt(1-interior_qty**2)
        J['climb|gamma','thrust_v2'] = d_arcsin/inputs['weight']/g
        J['climb|gamma','drag_v2'] = -d_arcsin/inputs['weight']/g
        J['climb|gamma','weight'] = -d_arcsin*(inputs['thrust_v2']-inputs['drag_v2'])/inputs['weight']**2/g

class TakeoffTransition(ExplicitComponent):
    """
    Computes distance and altitude at end of circular transition.

    This is a helper function for the main TOFL analysis group `TakeoffTotalDistance`
    and shoudln't be instantiated in the top-level model directly.

    Based on TO distance analysis method in Raymer book.
    Obstacle clearance height set for GA / Part 23 aircraft
    Override for analyzing Part 25 aircraft

    Inputs
    ------
    fltcond|Utrue_vtrans
        Transition true airspeed (generally avg of vr and v2) (scalar, m/s)
    climb|gamma : float
        Climb out flight path angle (scalar, rad)

    Outputs
    -------
    s_transition : float
        Horizontal distance during transition to v2 climb out (scalar, m)
    h_transition : float
        Altitude at transition point (scalar, m)

    Options
    -------
    h_obstacle : float
        Obstacle height to clear (in **meters**) (default 10.66, equiv. 35 ft)
    """

    def initialize(self):
        self.options.declare('h_obstacle',default=10.66,desc='Obstacle clearance height in m')

    def setup(self):
        self.add_input('fltcond|Utrue_vtrans', units='m/s')
        self.add_input('climb|gamma', units='rad')
        self.add_output('s_transition', units='m')
        self.add_output('h_transition', units='m')
        self.declare_partials(['s_transition','h_transition'], ['fltcond|Utrue_vtrans','climb|gamma'])

    def compute(self, inputs, outputs):
        hobs = self.options['h_obstacle']
        g = 9.80665 #m/s^2
        gam = inputs['climb|gamma']
        ut = inputs['fltcond|Utrue_vtrans']

        R = ut**2/0.2/g
        st = R*np.sin(gam)
        ht = R*(1-np.cos(gam))
        #alternate formula if the obstacle is cleared during transition
        if ht > hobs:
            st = np.sqrt(R**2-(R-hobs)**2)
            ht = hobs
        outputs['s_transition'] = st
        outputs['h_transition'] = ht

    def compute_partials(self, inputs, J):
        hobs = self.options['h_obstacle']
        g = 9.80665 #m/s^2
        gam = inputs['climb|gamma']
        ut = inputs['fltcond|Utrue_vtrans']
        R = ut**2/0.2/g
        dRdut =  2*ut/0.2/g
        st = R*np.sin(gam)
        ht = R*(1-np.cos(gam))
        #alternate formula if the obstacle is cleared during transition
        if ht > hobs:
            st = np.sqrt(R**2-(R-hobs)**2)
            dstdut = 1/2/np.sqrt(R**2-(R-hobs)**2) * (2*R*dRdut - 2*(R-hobs)*dRdut)
            dstdgam = 0
            dhtdut = 0
            dhtdgam = 0
        else:
            dhtdut = dRdut*(1-np.cos(gam))
            dhtdgam = R*np.sin(gam)
            dstdut = dRdut*np.sin(gam)
            dstdgam = R*np.cos(gam)
        J['s_transition','climb|gamma'] = dstdgam
        J['s_transition','fltcond|Utrue_vtrans'] = dstdut
        J['h_transition','climb|gamma'] = dhtdgam
        J['h_transition','fltcond|Utrue_vtrans'] = dhtdut


class TakeoffClimb(ExplicitComponent):
    """
    Computes ground distance from end of transition until obstacle is cleared.

    This is a helper function for the main TOFL analysis group `TakeoffTotalDistance`
    and shoudln't be instantiated in the top-level model directly.

    Analysis based on Raymer book.

    Inputs
    ------
    climb|gamma : float
        Climb out flight path angle (scalar, rad)
    h_transition : float
        Altitude at transition point (scalar, m)

    Outputs
    -------
    s_climb : float
        Horizontal distance from end of transition until obstacle is cleared (scalar, m)

    Options
    -------
    h_obstacle : float
        Obstacle height to clear (in **meters**) (default 10.66, equiv. 35 ft)
    """

    def initialize(self):
        self.options.declare('h_obstacle',default=10.66,desc='Obstacle clearance height in m')
    def setup(self):
        self.add_input('h_transition', units='m')
        self.add_input('climb|gamma', units='rad')
        self.add_output('s_climb', units='m')
        self.declare_partials(['s_climb'], ['h_transition','climb|gamma'])

    def compute(self, inputs, outputs):
        hobs = self.options['h_obstacle']
        gam = inputs['climb|gamma']
        ht = inputs['h_transition']
        sc = (hobs-ht)/np.tan(gam)
        outputs['s_climb'] = sc

    def compute_partials(self, inputs, J):
        hobs = self.options['h_obstacle']
        gam = inputs['climb|gamma']
        ht = inputs['h_transition']
        sc = (hobs-ht)/np.tan(gam)
        J['s_climb','climb|gamma'] = -(hobs-ht)/np.tan(gam)**2 * (1/np.cos(gam))**2
        J['s_climb','h_transition'] = -1/np.tan(gam)
