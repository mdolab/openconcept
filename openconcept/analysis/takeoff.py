from openmdao.api import Problem, Group, IndepVarComp
from openmdao.api import ScipyOptimizeDriver, ExplicitComponent, ImplicitComponent

import numpy as np
import scipy.sparse as sp

from openconcept.analysis.atmospherics.compute_atmos_props import ComputeAtmosphericProperties
from openconcept.utilities.math.simpson_integration import simpson_integral, simpson_partials, simpson_integral_every_node, simpson_partials_every_node, IntegrateQuantity
from openconcept.utilities.math import AddSubtractComp, ElementMultiplyDivideComp, VectorConcatenateComp, VectorSplitComp
from openconcept.analysis.aerodynamics import PolarDrag, Lift, StallSpeed
from openconcept.utilities.dvlabel import DVLabel

def takeoff_check(prob):
    """
    In some cases, the numeric integration scheme used to calculate TOFL can give a spurious result if the airplane can't accelerate through to V1. This function
    detects this case and raises an error. It should be called following every model.run_driver or run_model call.
    """
    v0v1 = prob['takeoff._rate_to_integrate_v0v1']
    v1vr = prob['takeoff._rate_to_integrate_v1vr']
    v1v0 = prob['takeoff._rate_to_integrate_v1v0']
    if np.sum(v0v1 < 0) > 0:
        raise ValueError('The aircraft was unable to reach v1 speed at the optimized design point. Restrict the design variables to add power or re-enable takeoff constraints')
    if np.sum(v1vr < 0) > 0:
        raise ValueError('The aircraft was unable to accelerate to vr from v1 (try adding power), or the v1 speed is higher than vr')
    if np.sum(v1v0 > 0) < 0:
        raise ValueError('Unusually, the aircraft continues to accelerate even after heavy braking in the abort phase of takeoff. Check your engine-out abort throttle settings (should be near zero thrust)')

class ComputeBalancedFieldLengthResidual(ExplicitComponent):
    def setup(self):
        self.add_input('takeoff|distance', units='m')
        self.add_input('takeoff|distance_abort', units='m')
        self.add_input('mission|takeoff|v1', units='m/s')
        self.add_input('mission|takeoff|vr', units='m/s')
        self.add_output('BFL_residual', units='m')
        self.add_output('v1vr_diff', units='m/s')
        self.add_output('BFL_combined', units='m')
        self.declare_partials('BFL_residual','takeoff|distance', val=1)
        self.declare_partials('BFL_residual','takeoff|distance_abort', val=-1)
        self.declare_partials('v1vr_diff','mission|takeoff|vr', val=1)
        self.declare_partials('v1vr_diff','mission|takeoff|v1', val=-1)
        self.declare_partials('BFL_combined',['takeoff|distance','takeoff|distance_abort','mission|takeoff|v1','mission|takeoff|vr'])
    def compute(self, inputs, outputs):
        outputs['BFL_residual'] = inputs['takeoff|distance'] - inputs['takeoff|distance_abort']
        outputs['v1vr_diff'] = inputs['mission|takeoff|vr'] - inputs['mission|takeoff|v1']
        speedtol = 1e-1
        disttol = 0
        #force the decision speed to zero
        if inputs['mission|takeoff|vr'] < inputs['mission|takeoff|v1'] + speedtol:
            outputs['BFL_combined'] = inputs['mission|takeoff|vr'] - inputs['mission|takeoff|v1']
        else:
            outputs['BFL_combined'] = inputs['takeoff|distance'] - inputs['takeoff|distance_abort']
        #if you are within vtol on the correct side but the stopping distance bigger, use the regular mode
        if inputs['mission|takeoff|vr'] >= inputs['mission|takeoff|v1'] and inputs['mission|takeoff|vr'] - inputs['mission|takeoff|v1'] < speedtol and (inputs['takeoff|distance_abort'] - inputs['takeoff|distance']) > disttol:
            outputs['BFL_combined'] = inputs['takeoff|distance'] - inputs['takeoff|distance_abort']

        # print('Cont: '+str(inputs['takeoff|distance'])+' Abort: '+str(inputs['takeoff|distance_abort']))
        # print('V1: '+str(inputs['mission|takeoff|v1'])+' Vr: '+str(inputs['mission|takeoff|vr']))

    def compute_partials(self, inputs, partials):
        speedtol = 1e-1
        disttol = 0

        if inputs['mission|takeoff|vr'] < inputs['mission|takeoff|v1'] + speedtol:
            partials['BFL_combined','takeoff|distance'] = 0
            partials['BFL_combined','takeoff|distance_abort'] = 0
            partials['BFL_combined','mission|takeoff|vr'] = 1
            partials['BFL_combined','mission|takeoff|v1'] = -1
        else:
            partials['BFL_combined','takeoff|distance'] = 1
            partials['BFL_combined','takeoff|distance_abort'] = -1
            partials['BFL_combined','mission|takeoff|vr'] = 0
            partials['BFL_combined','mission|takeoff|v1'] = 0

        if inputs['mission|takeoff|vr'] >= inputs['mission|takeoff|v1'] and inputs['mission|takeoff|vr'] - inputs['mission|takeoff|v1'] < speedtol and (inputs['takeoff|distance_abort'] - inputs['takeoff|distance']) > disttol:
            partials['BFL_combined','takeoff|distance'] = 1
            partials['BFL_combined','takeoff|distance_abort'] = -1
            partials['BFL_combined','mission|takeoff|vr'] = 0
            partials['BFL_combined','mission|takeoff|v1'] = 0

class TakeoffFlightConditions(ExplicitComponent):
    def initialize(self):
        self.options.declare('n_int_per_seg',default=5,desc="Number of Simpson intervals to use per seg. Number of analysis points is 2N+1")
        self.options.declare('vr_multiple',default=1.1, desc="Rotation speed multiple of Vstall")
        self.options.declare('v2_multiple',default=1.2, desc='Climb out multipile of Vstall')

    def setup(self):
        n_int_per_seg = self.options['n_int_per_seg']
        nn = (n_int_per_seg*2 + 1)
        vrmult = self.options['vr_multiple']
        v2mult = self.options['v2_multiple']
        self.add_input('mission|takeoff|h', val=0, units='m', desc='Takeoff runway altitude')
        self.add_input('mission|takeoff|v1', val=75, units='m / s', desc='Takeoff decision airspeed')
        self.add_input('Vstall_eas', val=90, units='m / s', desc='Flaps down stall airspeed')
        self.add_output('mission|takeoff|vr', units='m/s', desc='Takeoff rotation speed')
        self.add_output('mission|takeoff|v2', units='m/s', desc='Takeoff climbout speed')

        ## the total length of the vector is 3 * nn (3 integrated segments) + 2 scalar flight conditions
        # condition 1: v0 to v1 [nn points]
        # condition 2: v1 to vr (one engine out) [nn points]
        # condition 3: v1 to v0 (with braking) [nn points]
        # condition 4: average between vr and v2 [1 point]
        # condition 5: v2 [1 point]
        self.add_output('fltcond|takeoff|Ueas', units='m / s', desc='indicated airspeed at each analysis point',shape=(3 * nn+2,))
        self.add_output('fltcond|takeoff|h', units='m', desc='altitude at each analysis point',shape=(3 * nn+2,))
        linear_nn = np.linspace(0,1,nn)
        linear_rev_nn = np.linspace(1,0,nn)

        #the climb speeds only have influence over their respective mission segments
        self.declare_partials(['fltcond|takeoff|Ueas'], ['mission|takeoff|v1'], val=np.concatenate([linear_nn,linear_rev_nn,linear_rev_nn,np.zeros(2)]))
        self.declare_partials(['fltcond|takeoff|Ueas'], ['Vstall_eas'], val=np.concatenate([np.zeros(nn),linear_nn*vrmult,np.zeros(nn),np.array([(vrmult+v2mult)/2,v2mult])]))
        self.declare_partials(['fltcond|takeoff|h'], ['mission|takeoff|h'], val=np.ones(3 * nn+2))
        self.declare_partials(['mission|takeoff|vr'], ['Vstall_eas'], val=vrmult)
        self.declare_partials(['mission|takeoff|v2'], ['Vstall_eas'], val=v2mult)


    def compute(self, inputs, outputs):
        n_int_per_seg = self.options['n_int_per_seg']
        nn = n_int_per_seg*2 + 1
        vrmult = self.options['vr_multiple']
        v2mult = self.options['v2_multiple']
        v1 = inputs['mission|takeoff|v1'][0]
        vstall = inputs['Vstall_eas'][0]
        vr = vrmult*vstall
        v2 = v2mult*vstall
        outputs['mission|takeoff|vr'] = vr
        outputs['mission|takeoff|v2'] = v2
        outputs['fltcond|takeoff|h'] = inputs['mission|takeoff|h']*np.ones(3 * nn+2)
        outputs['fltcond|takeoff|Ueas'] = np.concatenate([np.linspace(1,v1,nn),np.linspace(v1,vr,nn),np.linspace(v1,1,nn),np.array([(vr+v2)/2, v2])])


class TakeoffCLs(ExplicitComponent):
    def initialize(self):
        self.options.declare('n_int_per_seg',default=5,desc="Number of Simpson intervals to use per seg. Number of analysis points is 2N+1")
        self.options.declare('ground_CL',default=0.1,desc="Assumed CL during the takeoff roll")
    def setup(self):
        n_int_per_seg = self.options['n_int_per_seg']
        nn = (n_int_per_seg*2 + 1)
        nn_tot = 3 * nn+2
        self.add_input('weight', units='kg')
        self.add_input('fltcond|takeoff|q', units='N * m**-2',shape=(nn_tot,))
        self.add_input('ac|geom|wing|S_ref', units='m**2')
        self.add_output('CL_takeoff',shape=(nn_tot,))
        #the partials only apply for the last two entries
        self.declare_partials(['CL_takeoff'], ['weight','ac|geom|wing|S_ref'], rows=np.arange(nn_tot-2,nn_tot), cols=np.zeros(2))
        self.declare_partials(['CL_takeoff'], ['fltcond|takeoff|q'], rows=np.arange(nn_tot-2,nn_tot), cols=np.arange(nn_tot-2,nn_tot))

    def compute(self, inputs, outputs):
        n_int_per_seg = self.options['n_int_per_seg']
        nn = (n_int_per_seg*2 + 1)
        nn_tot = 3 * nn+2
        g = 9.80665 #m/s^2
        CLs = np.ones(nn_tot) * self.options['ground_CL']
        #the 1.2 load factor is an assumption from Raymer. May want to revisit or make an option /default in the future
        loadfactors = np.array([1.2,1.2])
        CLs[-2:] = loadfactors*inputs['weight'][-2:] * g / inputs['fltcond|takeoff|q'][-2:] / inputs['ac|geom|wing|S_ref'][-2:]
        outputs['CL_takeoff'] = CLs
    def compute_partials(self, inputs, J):
        n_int_per_seg = self.options['n_int_per_seg']
        nn = (n_int_per_seg*2 + 1)
        nn_tot = 3 * nn+2
        g = 9.80665 #m/s^2
        #the 1.2 load factor is an assumption from Raymer. May want to revisit or make an option /default in the future
        loadfactors = np.array([1.2,1.2])
        J['CL_takeoff','weight'] = loadfactors * g / inputs['fltcond|takeoff|q'][-2:] / inputs['ac|geom|wing|S_ref'][-2:]
        J['CL_takeoff','fltcond|takeoff|q'] = - loadfactors*inputs['weight'][-2:] * g / inputs['fltcond|takeoff|q'][-2:]**2 / inputs['ac|geom|wing|S_ref'][-2:]
        J['CL_takeoff','ac|geom|wing|S_ref'] = - loadfactors*inputs['weight'][-2:] * g / inputs['fltcond|takeoff|q'][-2:] / inputs['ac|geom|wing|S_ref'][-2:]**2

class TakeoffAccels(ExplicitComponent):
    """This returns the INVERSE of the accelerations during the takeoff run
        This is due to the integration wrt velocity: int( dr/dt * dt / dv) dv = int( v / a) dv
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
        self.add_input('takeoff|thrust', units='N',shape=(nn_tot,))
        self.add_output('_inverse_accel', units='s**2 / m',shape=(nn_tot,))
        arange=np.arange(0,nn_tot-2)
        self.declare_partials(['_inverse_accel'], ['drag','lift','takeoff|thrust'], rows=arange, cols=arange)
        self.declare_partials(['_inverse_accel'], ['weight'], rows=arange, cols=np.zeros(nn_tot-2))

    def compute(self, inputs, outputs):
        n_int_per_seg = self.options['n_int_per_seg']
        nn = (n_int_per_seg*2 + 1)
        nn_tot = 3 * nn+2
        mu_free = self.options['free_rolling_friction_coeff']
        mu_brake = self.options['braking_friction_coeff']
        mu = np.concatenate([np.ones(2*nn)*mu_free,np.ones(nn)*mu_brake,np.zeros(2)])
        g = 9.80665 #m/s^2
        # print('Thrust:' +str(inputs['takeoff|thrust']))
        # print('Drag:'+str(inputs['drag']))
        # print('Rolling resistance:'+str(mu*(inputs['weight']*g-inputs['lift'])))
        # print(inputs['takeoff|thrust'])
        # print(inputs['drag'])
        forcebal = inputs['takeoff|thrust'] - inputs['drag'] - mu*(inputs['weight']*g-inputs['lift'])
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
        forcebal = inputs['takeoff|thrust'] - inputs['drag'] - mu*(inputs['weight']*g-inputs['lift'])
        #(f/g)' = (f'g-fg')/g^2 where f is weight and g is forcebal
        ddweight = (1*forcebal - inputs['weight']*-mu*g)/forcebal**2
        ddweight=ddweight[:-2]
        dddrag = -inputs['weight'] / forcebal**2 * (-1)
        dddrag = dddrag[:-2]
        ddlift = -inputs['weight'] / forcebal**2 *mu
        ddlift = ddlift[:-2]
        ddthrust = - inputs['weight'] / forcebal**2
        ddthrust = ddthrust[:-2]

        J['_inverse_accel','takeoff|thrust'] = ddthrust
        J['_inverse_accel','drag'] = dddrag
        J['_inverse_accel','lift'] = ddlift
        J['_inverse_accel','weight'] = ddweight

class TakeoffV2ClimbAngle(ExplicitComponent):
    def setup(self):
        self.add_input('drag_v2', units='N')
        self.add_input('weight', units='kg')
        self.add_input('takeoff|thrust_v2', units='N')
        self.add_output('takeoff|climb|gamma', units='rad')

        self.declare_partials(['takeoff|climb|gamma'], ['weight','takeoff|thrust_v2','drag_v2'])

    def compute(self, inputs, outputs):
        g = 9.80665 #m/s^2
        outputs['takeoff|climb|gamma'] = np.arcsin((inputs['takeoff|thrust_v2']-inputs['drag_v2'])/inputs['weight']/g)

    def compute_partials(self, inputs, J):
        g = 9.80665 #m/s^2
        interior_qty = (inputs['takeoff|thrust_v2']-inputs['drag_v2'])/inputs['weight']/g
        d_arcsin = 1/np.sqrt(1-interior_qty**2)
        J['takeoff|climb|gamma','takeoff|thrust_v2'] = d_arcsin/inputs['weight']/g
        J['takeoff|climb|gamma','drag_v2'] = -d_arcsin/inputs['weight']/g
        J['takeoff|climb|gamma','weight'] = -d_arcsin*(inputs['takeoff|thrust_v2']-inputs['drag_v2'])/inputs['weight']**2/g

class TakeoffTransition(ExplicitComponent):
    def initialize(self):
        self.options.declare('h_obstacle',default=10.66,desc='Obstacle clearance height in m')

    def setup(self):
        self.add_input('fltcond|takeoff|Utrue_vtrans', units='m/s')
        self.add_input('takeoff|climb|gamma', units='rad')
        self.add_output('s_transition', units='m')
        self.add_output('h_transition', units='m')
        self.declare_partials(['s_transition','h_transition'], ['fltcond|takeoff|Utrue_vtrans','takeoff|climb|gamma'])

    def compute(self, inputs, outputs):
        hobs = self.options['h_obstacle']
        g = 9.80665 #m/s^2
        gam = inputs['takeoff|climb|gamma']
        ut = inputs['fltcond|takeoff|Utrue_vtrans']

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
        gam = inputs['takeoff|climb|gamma']
        ut = inputs['fltcond|takeoff|Utrue_vtrans']
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
        J['s_transition','takeoff|climb|gamma'] = dstdgam
        J['s_transition','fltcond|takeoff|Utrue_vtrans'] = dstdut
        J['h_transition','takeoff|climb|gamma'] = dhtdgam
        J['h_transition','fltcond|takeoff|Utrue_vtrans'] = dhtdut


class TakeoffClimb(ExplicitComponent):
    def initialize(self):
        self.options.declare('h_obstacle',default=10.66,desc='Obstacle clearance height in m')
    def setup(self):
        self.add_input('h_transition', units='m')
        self.add_input('takeoff|climb|gamma', units='rad')
        self.add_output('s_climb', units='m')
        self.declare_partials(['s_climb'], ['h_transition','takeoff|climb|gamma'])

    def compute(self, inputs, outputs):
        hobs = self.options['h_obstacle']
        gam = inputs['takeoff|climb|gamma']
        ht = inputs['h_transition']
        sc = (hobs-ht)/np.tan(gam)
        outputs['s_climb'] = sc

    def compute_partials(self, inputs, J):
        hobs = self.options['h_obstacle']
        gam = inputs['takeoff|climb|gamma']
        ht = inputs['h_transition']
        sc = (hobs-ht)/np.tan(gam)
        J['s_climb','takeoff|climb|gamma'] = -(hobs-ht)/np.tan(gam)**2 * (1/np.cos(gam))**2
        J['s_climb','h_transition'] = -1/np.tan(gam)

class TakeoffTotalDistance(Group):
    """This high level component calculates lift, drag, rolling resistance during a takeoff roll, including one-engine-out and abort.
    """
    def initialize(self):
        self.options.declare('n_int_per_seg',default=5,desc="Number of Simpson intervals to use per seg. Number of analysis points is 2N+1")
        self.options.declare('track_battery',default=False,desc='Set to true to enable battery inputs')
        self.options.declare('track_fuel',default=False,desc='Set to true to enable fuel inputs')

    def setup(self):
        n_int_per_seg = self.options['n_int_per_seg']
        track_battery = self.options['track_battery']
        track_fuel = self.options['track_fuel']
        nn = (n_int_per_seg*2 + 1)
        nn_tot = 3 * nn+2
        #===Some of the generic components take "weight" as an input. Need to re-label Takeoff Weight (TOW) as just weight
        dvlist = [['ac|weights|MTOW','weight',2000.0,'kg'],
                    ['mission|takeoff|v1','v1',40,'m/s'],
                    ['mission|takeoff|vr','vr',45,'m/s'],
                    ['ac|aero|polar|CD0_TO','CD0',0.005,None],
                    ['ac|aero|polar|e','e',0.95,None],
                    ['fltcond|takeoff|q','fltcond|q',100*np.ones(nn_tot),'Pa']]
        self.add_subsystem('dvs',DVLabel(dvlist),promotes_inputs=["*"],promotes_outputs=["*"])

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
        self.add_subsystem('accel',TakeoffAccels(n_int_per_seg=n_int_per_seg),promotes_inputs=['lift','drag','takeoff|thrust','weight'],promotes_outputs=['_inverse_accel'])
        self.add_subsystem('mult',ElementMultiplyDivideComp(output_name='_rate_to_integrate',
                                                            input_names=['_inverse_accel','fltcond|takeoff|Utrue'],
                                                            vec_size=nn_tot,
                                                            input_units=['s**2/m','m/s']),
                                                            promotes_inputs=['*'],promotes_outputs=['_*'])
        if track_fuel:
            self.add_subsystem('fuelflowmult',ElementMultiplyDivideComp(output_name='_fuel_flow_rate_to_integrate',
                                                                        input_names=['_inverse_accel','takeoff|fuel_flow'],
                                                                        vec_size=nn_tot,
                                                                        input_units=['s**2/m','kg/s']),
                                                                        promotes_inputs=['*'],promotes_outputs=['_*'])
        if track_battery:
            self.add_subsystem('batterymult',ElementMultiplyDivideComp(output_name='_battery_rate_to_integrate',
                                                                       input_names=['_inverse_accel','takeoff|battery_load'],
                                                                       vec_size=nn_tot,
                                                                       input_units=['s**2/m','J / s']),
                                                                       promotes_inputs=['*'],promotes_outputs=['*'])




        #==The following boilerplate splits the input flight conditions, thrusts, and fuel flows into the takeoff segments for further analysis
        inputs_to_split = ['_rate_to_integrate','fltcond|takeoff|Utrue','takeoff|thrust','drag','lift']
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
        self.add_subsystem('gamma',TakeoffV2ClimbAngle(),promotes_inputs=["drag_v2","takeoff|thrust_v2","weight"],promotes_outputs=["takeoff|climb|gamma"])
        self.add_subsystem('transition',TakeoffTransition(),promotes_inputs=["fltcond|takeoff|Utrue_vtrans","takeoff|climb|gamma"],promotes_outputs=["h_transition","s_transition"])
        self.add_subsystem('climb',TakeoffClimb(),promotes_inputs=["takeoff|climb|gamma","h_transition"],promotes_outputs=["s_climb"])
        self.add_subsystem('reaction',ElementMultiplyDivideComp(output_name='s_reaction',
                                                                input_names=['v1','reaction_time'],
                                                                vec_size=1,
                                                                input_units=['m/s','s']),promotes_inputs=['v1'])
        self.connect('const.reaction_time','reaction.reaction_time')

        self.add_subsystem('total_to_distance_continue',AddSubtractComp(output_name='takeoff|distance',input_names=['s_v0v1','s_v1vr','s_reaction','s_transition','s_climb'],vec_size=1, units='m'),promotes_outputs=["*"])
        self.add_subsystem('total_to_distance_abort',AddSubtractComp(output_name='takeoff|distance_abort',input_names=['s_v0v1','s_v1v0','s_reaction'],vec_size=1, units='m'),promotes_outputs=["*"])

        self.connect('reaction.s_reaction','total_to_distance_continue.s_reaction')
        self.connect('reaction.s_reaction','total_to_distance_abort.s_reaction')
        self.connect('v0v1_dist.delta_quantity',['total_to_distance_continue.s_v0v1','total_to_distance_abort.s_v0v1'])
        self.connect('v1vr_dist.delta_quantity','total_to_distance_continue.s_v1vr')
        self.connect('s_transition','total_to_distance_continue.s_transition')
        self.connect('s_climb','total_to_distance_continue.s_climb')
        self.connect('v1v0_dist.delta_quantity','total_to_distance_abort.s_v1v0')

        if track_battery:
            self.add_subsystem('total_battery',AddSubtractComp(output_name='takeoff|total_battery_energy',input_names=['battery_v0v1','battery_v1vr'], units='J'),promotes_outputs=["*"])
            self.connect('v0v1_battery.delta_quantity','total_battery.battery_v0v1')
            self.connect('v1vr_battery.delta_quantity','total_battery.battery_v1vr')
        if track_fuel:
            self.add_subsystem('total_fuel',AddSubtractComp(output_name='takeoff|total_fuel',input_names=['fuel_v0v1','fuel_v1vr'], units='kg',scaling_factors=[-1,-1]),promotes_outputs=["*"])
            self.connect('v0v1_fuel.delta_quantity','total_fuel.fuel_v0v1')
            self.connect('v1vr_fuel.delta_quantity','total_fuel.fuel_v1vr')
            self.add_subsystem('climb_weight',AddSubtractComp(output_name='weight_after_takeoff',input_names=['weight','takeoff|total_fuel'], units='kg',scaling_factors=[1,-1]),promotes_inputs=["*"],promotes_outputs=["*"])



# class TakeoffAnalysisTest(Group):
#     """This analysis group calculates TOFL and mission fuel burn as well as many other quantities for an example airplane. Elements may be overridden or replaced as needed.
#         Should be instantiated as the top-level model

#     """

#     def initialize(self):
#         self.options.declare('n_int_per_seg',default=5,desc="Number of Simpson intervals to use per seg (eg. climb, cruise, descend). Number of analysis points is 2N+1")
#         #self.options.declare('propmodel',desc='Propulsion model to use. Pass in the class, not an instance')

#     def setup(self):
#         n_int_per_seg = self.options['n_int_per_seg']
#         nn_tot = (2*n_int_per_seg+1)*3 +2 #v0v1,v1vr,v1v0, vtr, v2
#         nn = (2*n_int_per_seg+1)

#         #Create holders for control and flight condition parameters. Add these as design variables as necessary for optimization when you define the prob

#         dv_comp = self.add_subsystem('dv_comp',IndepVarComp(),promotes_outputs=["CLmax_flapsdown","ac|geom|*","takeoff_*","dv_*"])
#         #eventually replace the following with an analysis module
#         dv_comp.add_output('CLmax_flapsdown', val=1.7)
#         dv_comp.add_output('polar_e', val=0.78)
#         dv_comp.add_output('polar_CD0', val=0.03)

#         #wing geometry variables
#         dv_comp.add_output('ac|geom|wing|S_ref', val=193.5, units='ft**2')
#         dv_comp.add_output('ac|geom|wing|AR', val=8.95)

#         #design weights
#         dv_comp.add_output('ac|weights|MTOW', val=3354, units='kg')

#         #takeoff parameters
#         dv_comp.add_output('mission|takeoff|h', val=0, units='ft')
#         dv_comp.add_output('mission|takeoff|v1', val=85, units='kn')

#         #propulsion parameters (rename this prefix at some point)
#         dv_comp.add_output('ac|propulsion|engine|rating',850, units='hp')
#         dv_comp.add_output('dv_prop1_diameter',2.3, units='m')




#         #== Compute the stall speed (necessary for takeoff analysis)
#         vstall = self.add_subsystem('vstall', StallSpeed(), promotes_inputs=["ac|geom|wing|S_ref","CLmax_flapsdown"], promotes_outputs=["Vstall_eas"])

#         #==Calculate flight conditions for the takeoff and mission segments here
#         conditions = self.add_subsystem('conds', TakeoffFlightConditions(n_int_per_seg=n_int_per_seg),promotes_inputs=["takeoff_*","*"],promotes_outputs=["fltcond|*","takeoff_*"])
#         a=VectorConcatenateComp()
#         a.add_relation(output_name='fltcond|h',input_names=['fltcond|takeoff|h'],vec_sizes=[nn_tot], units='m')
#         a.add_relation(output_name='fltcond|Ueas',input_names=['fltcond|takeoff|Ueas'],vec_sizes=[nn_tot], units='m/s')
#         combiner = self.add_subsystem('combiner',subsys=a,promotes_inputs=["*"],promotes_outputs=["*"])
#         #==Calculate atmospheric properties and true airspeeds for all mission segments
#         atmos = self.add_subsystem('atmos',ComputeAtmosphericProperties(num_nodes=nn_tot),promotes_inputs=["fltcond|h","fltcond|Ueas"],promotes_outputs=["fltcond|rho","fltcond|Utrue","fltcond|q"])

#         #==Calculate engine thrusts and fuel flows. You will need to override this module to vary number of engines, prop architecture, etc
#         # Your propulsion model must promote up a single variable called "thrust" and a single variable called "fuel_flow". You may need to sum these at a lower level in the prop model group
#         # You will probably need to add more control parameters if you use multiple engines. You may also need to add implicit solver states if, e.g. turbogenerator power setting depends on motor power setting
#         controls = self.add_subsystem('controls',IndepVarComp())
#         prop = self.add_subsystem('propmodel',TurbopropPropulsionSystem(num_nodes=nn_tot),promotes_inputs=["fltcond|*","dv_*"],promotes_outputs=["fuel_flow","thrust"])

#         #==Define control settings for the propulsion system.
#         # Recall that all flight points including takeoff roll are calculated all at once
#         # The structure of the takeoff vector should be:
#         #[ nn points (takeoff at full power, v0 to v1),
#         #  nn points (takeoff at engine-out power (if applicable), v1 to vr),
#         #  nn points (hard braking at zero power or even reverse, vr to v0),
#         # !CAUTION! 1 point (transition at OEI power (if applicable), v_trans)
#         # !CAUTION! 1 point (v2 climb at OEI power (if app), v2)
#         # ]
#         controls.add_output('prop1|rpm', val=np.ones(nn_tot)*2000, units='rpm')
#         throttle_vec = np.concatenate([np.ones(nn),np.ones(nn)*1.0,np.zeros(nn),np.ones(2)*1.0])
#         controls.add_output('motor1_throttle', val=throttle_vec)

#         #connect control settings to the various states in the propulsion model
#         self.connect('controls.prop1|rpm','propmodel.prop1.rpm')
#         self.connect('controls.motor1_throttle','propmodel.throttle')

#         #now we have flight conditions and propulsion outputs for all flight conditions. Split into our individual analysis phases
#         #== Leave this alone==#
#         inputs_to_split = ['fltcond|q','fltcond|Utrue','fuel_flow','thrust']
#         segments_to_split_into = ['takeoff']
#         units = ['N * m**-2','m/s','kg/s','N']
#         nn_each_segment = [nn_tot]
#         b = VectorSplitComp()
#         for k, input_name in enumerate(inputs_to_split):
#             output_names_list = []
#             for segment in segments_to_split_into:
#                 output_names_list.append(input_name+'_'+segment)
#             print(output_names_list)
#             b.add_relation(output_names=output_names_list,input_name=input_name,vec_sizes=nn_each_segment, units=units[k])
#         splitter = self.add_subsystem('splitter',subsys=b,promotes_inputs=["*"],promotes_outputs=["*"])

#         #==This next module calculates balanced field length, if applicable. Your optimizer or solver MUST implicitly drive the abort distance and oei takeoff distances to the same value by varying v1

#         takeoff = self.add_subsystem('takeoff',TakeoffTotalDistance(n_int_per_seg=n_int_per_seg,track_fuel=True),promotes_inputs=['ac|geom|*','fltcond|*_takeoff','takeoff|thrust','takeoff|fuel_flow','takeoff_*'],promotes_outputs=['*'])
#         self.connect('dv_comp.polar_CD0','takeoff.drag.polar_CD0')
#         self.connect('dv_comp.polar_e','takeoff.drag.polar_e')
#         self.connect('fltcond|takeoff|q','takeoff.drag.fltcond|q')
#         self.connect('fltcond|takeoff|q','takeoff.lift.fltcond|q')
#         self.connect('dv_comp.MTOW','vstall.weight')
#         self.connect('dv_comp.MTOW','takeoff.TOW')


# if __name__ == "__main__":
#     from openconcept.examples.propulsion_layouts.simple_turboprop import TurbopropPropulsionSystem
#     prob = Problem()
#     prob.model= TakeoffAnalysisTest(n_int_per_seg=4)
#     #prob.model=TestGroup()

#     prob.setup()
#     prob.run_model()
#     # print(prob['fltcond|Ueas'])
#     # print(prob['fltcond|h'])
#     # print(prob['fltcond|rho'])
#     # print(prob['fuel_flow'])
#     print('Stall speed'+str(prob['Vstall_eas']))
#     print('Rotate speed'+str(prob['mission|takeoff|vr']))

#     print('V0V1 dist: '+str(prob['takeoff.v0v1_dist.delta_quantity']))
#     print('V1VR dist: '+str(prob['takeoff.v1vr_dist.delta_quantity']))
#     print('Braking dist:'+str(prob['takeoff.v1v0_dist.delta_quantity']))
#     print('Climb angle(rad):'+str(prob['takeoff|climb|gamma']))
#     print('h_trans:'+str(prob['h_transition']))
#     print('s_trans:'+str(prob['s_transition']))
#     print('s_climb|'+str(prob['s_climb']))
#     print('TO (continue):'+str(prob['takeoff|distance']))
#     print('TO (abort):'+str(prob['takeoff|distance_abort']))


#     #prob.model.list_inputs()
#     #prob.model.list_outputs()
#     #prob.check_partials(compact_print=True)
#     #prob.check_totals(compact_print=True)
