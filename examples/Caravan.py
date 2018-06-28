from openmdao.api import Problem, Group, IndepVarComp, DirectSolver, NewtonSolver
from openmdao.api import ScipyOptimizeDriver, ExplicitComponent, ImplicitComponent
#-------These imports are generic and should be left alone
import numpy as np
import scipy.sparse as sp
from openconcept.analysis.atmospherics.compute_atmos_props import ComputeAtmosphericProperties
from openconcept.utilities.math import VectorConcatenateComp, VectorSplitComp
from openconcept.utilities.dict_indepvarcomp import DictIndepVarComp
from openconcept.analysis.aerodynamics import StallSpeed
from openconcept.analysis.takeoff import TakeoffFlightConditions, TakeoffTotalDistance, ComputeBalancedFieldLengthResidual, takeoff_check
from openconcept.analysis.mission import MissionFlightConditions, MissionNoReserves, ComputeDesignMissionResiduals

#These imports are particular to this airplane
from propulsion_layouts.simple_turboprop import TurbopropPropulsionSystem
from aircraft_data.caravan import data as acdata
from aircraft_data.caravan_mission import data as missiondata
from methods.weights_turboprop import SingleTurboPropEmptyWeight
from methods.costs_commuter import OperatingCost

class DummyPayload(ExplicitComponent):
    def setup(self):
        self.add_input('payload_DV',units='lb')
        self.add_output('payload_objective',units='lb')
        self.declare_partials(['payload_objective'],['payload_DV'],val=1)
    def compute(self,inputs,outputs):
        outputs['payload_objective'] = inputs['payload_DV']


class TotalAnalysis(Group):
    """This analysis group calculates TOFL and mission fuel burn as well as many other quantities for an example airplane. Elements may be overridden or replaced as needed.
        Should be instantiated as the top-level model
    """

    def initialize(self):
        self.options.declare('num_integration_intervals_per_seg',default=5,desc="Number of Simpson intervals to use per seg (eg. climb, cruise, descend). Number of time points is 2N+1")

    def setup(self):
        n_int_per_seg = self.options['num_integration_intervals_per_seg']
        nn_tot_to = (2*n_int_per_seg+1)*3 +2 #v0v1,v1vr,v1v0, vtr, v2
        nn_tot_m = (2*n_int_per_seg+1)*3
        nn_tot=nn_tot_to+nn_tot_m
        nn = (2*n_int_per_seg+1)

        #Define input variables
        dv_comp = self.add_subsystem('dv_comp',DictIndepVarComp(acdata),promotes_outputs=["*"])
        #eventually replace the following aerodynamic parameters with an analysis module (maybe OpenAeroStruct)
        dv_comp.add_output_from_dict('ac:aero:CLmax_flaps30')
        dv_comp.add_output_from_dict('ac:aero:polar:e')
        dv_comp.add_output_from_dict('ac:aero:polar:CD0_TO')
        dv_comp.add_output_from_dict('ac:aero:polar:CD0_cruise')

        dv_comp.add_output_from_dict('ac:geom:wing:S_ref')
        dv_comp.add_output_from_dict('ac:geom:wing:AR')
        dv_comp.add_output_from_dict('ac:geom:wing:c4sweep')
        dv_comp.add_output_from_dict('ac:geom:wing:taper')
        dv_comp.add_output_from_dict('ac:geom:wing:toverc')
        dv_comp.add_output_from_dict('ac:geom:hstab:S_ref')
        dv_comp.add_output_from_dict('ac:geom:hstab:c4_to_wing_c4')
        dv_comp.add_output_from_dict('ac:geom:vstab:S_ref')
        dv_comp.add_output_from_dict('ac:geom:fuselage:S_wet')
        dv_comp.add_output_from_dict('ac:geom:fuselage:width')
        dv_comp.add_output_from_dict('ac:geom:fuselage:length')
        dv_comp.add_output_from_dict('ac:geom:fuselage:height')
        dv_comp.add_output_from_dict('ac:geom:nosegear:length')
        dv_comp.add_output_from_dict('ac:geom:maingear:length')

        dv_comp.add_output_from_dict('ac:weights:MTOW')
        dv_comp.add_output_from_dict('ac:weights:W_fuel_max')
        dv_comp.add_output_from_dict('ac:weights:MLW')

        dv_comp.add_output_from_dict('ac:propulsion:engine:rating')
        dv_comp.add_output_from_dict('ac:propulsion:propeller:diameter')

        dv_comp.add_output_from_dict('ac:num_passengers_max')
        dv_comp.add_output_from_dict('ac:q_cruise')

        mission_data_comp = self.add_subsystem('mission_data_comp',DictIndepVarComp(missiondata),promotes_outputs=["*"])
        mission_data_comp.add_output_from_dict('mission:takeoff:h')
        mission_data_comp.add_output_from_dict('mission:climb:vs')
        mission_data_comp.add_output_from_dict('mission:climb:Ueas')
        mission_data_comp.add_output_from_dict('mission:cruise:h')
        mission_data_comp.add_output_from_dict('mission:cruise:Ueas')
        mission_data_comp.add_output_from_dict('mission:descent:vs')
        mission_data_comp.add_output_from_dict('mission:descent:Ueas')
        mission_data_comp.add_output_from_dict('mission:range')
        mission_data_comp.add_output_from_dict('mission:payload')

        #== Compute the stall speed (necessary for takeoff analysis - leave this alone)
        vstall = self.add_subsystem('vstall', StallSpeed())
        self.connect('ac:weights:MTOW','vstall.weight')
        self.connect('ac:geom:wing:S_ref','vstall.ac:geom:wing:S_ref')
        self.connect('ac:aero:CLmax_flaps30', 'vstall.CLmax')

        # ==Calculate flight conditions for the takeoff and mission segments here (leave this alone)
        mission_conditions = self.add_subsystem('mission_conditions',
                                                MissionFlightConditions(num_integration_intervals_per_seg=n_int_per_seg),
                                                promotes_inputs=["mission:*"],
                                                promotes_outputs=["mission:*", "fltcond:mission:*"])

        takeoff_conditions = self.add_subsystem('takeoff_conditions',
                                                TakeoffFlightConditions(num_integration_intervals_per_seg=n_int_per_seg),
                                                promotes_inputs=["mission:takeoff:*"],
                                                promotes_outputs=["fltcond:takeoff:*",
                                                                  "mission:takeoff:*"])
        self.connect('vstall.Vstall_eas', 'takeoff_conditions.Vstall_eas')

        fltcondcombiner = VectorConcatenateComp(output_name='fltcond:h',
                                                 input_names=['fltcond:takeoff:h',
                                                              'fltcond:mission:h'],
                                                 units='m',
                                                 vec_sizes=[nn_tot_to, nn_tot_m])
        fltcondcombiner.add_relation(output_name='fltcond:Ueas',
                                      input_names=['fltcond:takeoff:Ueas',
                                                   'fltcond:mission:Ueas'],
                                      units='m/s',
                                      vec_sizes=[nn_tot_to, nn_tot_m])
        self.add_subsystem('fltcondcombiner', subsys=fltcondcombiner,
                           promotes_inputs=["fltcond:takeoff:*",
                                            "fltcond:mission:*"],
                           promotes_outputs=["fltcond:Ueas", "fltcond:h"])

        #==Calculate atmospheric properties and true airspeeds for all mission segments
        atmos = self.add_subsystem('atmos',
                                   ComputeAtmosphericProperties(num_nodes=nn_tot),
                                   promotes_inputs=["fltcond:h",
                                                    "fltcond:Ueas"],
                                   promotes_outputs=["fltcond:rho",
                                                     "fltcond:Utrue",
                                                     "fltcond:q"])

        #==Define control settings for the propulsion system.
        # Recall that all flight points including takeoff roll are calculated all at once
        # The structure of the takeoff vector should be:
        #[ nn points (takeoff at full power, v0 to v1),
        #  nn points (takeoff at engine-out power (if applicable), v1 to vr),
        #  nn points (hard braking at zero power or even reverse, vr to v0),
        # !CAUTION! 1 point (transition at OEI power (if applicable), v_trans)
        # !CAUTION! 1 point (v2 climb at OEI power (if app), v2)
        # ]
        # The mission throttle vector should be set implicitly using the optimizer (driving T = D + sin(gamma)mg residual to 0)

        controls = self.add_subsystem('controls',IndepVarComp())
        #set the prop to 2000 rpm for all time
        controls.add_output('prop1:rpm', val=np.ones(nn_tot) * 2000, units='rpm')
        TO_throttle_vec = np.concatenate([np.ones(nn),
                                          np.ones(nn) * 1.0,
                                          np.zeros(nn),
                                          np.ones(2) * 1.0])
        controls.add_output('motor1:throttle:takeoff', val=TO_throttle_vec)

        #combine the various controls together into one vector
        throttle_combiner = VectorConcatenateComp(output_name='motor1:throttle',
                                                  input_names=['motor1:throttle:takeoff',
                                                               'motor1:throttle:mission'],
                                                  vec_sizes=[nn_tot_to, nn_tot_m])
        self.add_subsystem('throttle_combiner', subsys=throttle_combiner,
                           promotes_outputs=["motor1:throttle"])
        self.connect('controls.motor1:throttle:takeoff',
                     'throttle_combiner.motor1:throttle:takeoff')

        #==Calculate engine thrusts and fuel flows. You will need to override this module to vary number of engines, prop architecture, etc
        # Your propulsion model must promote up a single variable called "thrust" and a single variable called "fuel_flow". You may need to sum these at a lower level in the prop model group
        # You will probably need to add more control parameters if you use multiple engines. You may also need to add implicit solver states if, e.g. turbogenerator power setting depends on motor power setting

        prop = self.add_subsystem('propmodel',TurbopropPropulsionSystem(num_nodes=nn_tot),promotes_inputs=["fltcond:*","ac:propulsion:*"],promotes_outputs=["fuel_flow","thrust"])
        #connect control settings to the various states in the propulsion model
        self.connect('controls.prop1:rpm','propmodel.prop1.rpm')
        self.connect('motor1:throttle','propmodel.throttle')


        #now we have flight conditions and propulsion outputs for all flight conditions. Split into our individual analysis phases
        #== Leave this alone==#
        splitter_inst = VectorSplitComp()

        inputs_to_split = ['fltcond:q','fltcond:Utrue','fuel_flow','thrust']
        segments_to_split_into = ['takeoff','mission']
        units = ['N * m**-2','m/s','kg/s','N']
        nn_each_segment = [nn_tot_to,nn_tot_m]

        for kth, input_name in enumerate(inputs_to_split):
            output_names_list = []
            for segment in segments_to_split_into:
                inpnamesplit = input_name.split(':')
                inpnamesplit.insert(-1,segment)
                output_names_list.append(':'.join(inpnamesplit))
            splitter_inst.add_relation(output_names=output_names_list, input_name=input_name, vec_sizes=nn_each_segment, units=units[kth])

        self.add_subsystem('splitter',subsys=splitter_inst, promotes_inputs=["*"], promotes_outputs=["*"])

        #==This next module calculates balanced field length, if applicable. Your optimizer or solver MUST implicitly drive the abort distance and oei takeoff distances to the same value by varying v1

        takeoff = self.add_subsystem('takeoff',TakeoffTotalDistance(num_integration_intervals_per_seg=n_int_per_seg,track_fuel=True),promotes_inputs=['ac:aero*','ac:weights:MTOW','ac:geom:*','fltcond:takeoff:*','takeoff:thrust','takeoff:fuel_flow','mission:takeoff:v*'])

        #==This module computes fuel consumption during the entire mission
        mission = self.add_subsystem('mission',MissionNoReserves(num_integration_intervals_per_seg=n_int_per_seg),promotes_inputs=["ac:aero:*","ac:geom:*","fltcond:mission:*","mission:thrust","mission:fuel_flow","mission:*"])
        #remember that you will need to set the mission throttle implicitly using the optimizer/solver. This was done above when we mashed the control vectors all together.
        self.connect('takeoff.weight_after_takeoff','mission:weight_initial')

        #==This module is an empirical weight tool specific to a single-engine turboprop airplane. You will need to modify or replace it.
        self.add_subsystem('OEW',SingleTurboPropEmptyWeight(),promotes_inputs=["*"])
        self.connect('ac:propulsion:engine:rating','P_TO')
        #Don't forget to couple the propulsion system to the weights module like so:
        self.connect('propmodel.prop1.component_weight','W_propeller')
        self.connect('propmodel.eng1.component_weight','W_engine')

        #==Finally, we need to compute certain quantities to ensure the airplane is feasible. Compute whether enough fuel volume exists, and whether the airplane burned more fuel than it can carry
        missionmargins = self.add_subsystem('missionmargins',ComputeDesignMissionResiduals(include_takeoff=True),promotes_inputs=['ac:weights:MTOW',"mission:*","ac:weights:W_fuel_max"],promotes_outputs=['fuel_burn'])
        self.connect('OEW.OEW','missionmargins.OEW')
        self.connect('mission.mission:total_fuel','mission:total_fuel')
        self.connect('takeoff.takeoff:total_fuel','missionmargins.takeoff:total_fuel')

        #==Calculate the difference between the one-engine-out abort distance and one-engine-out takeoff distance with obstacle clearance
        bflmargins = self.add_subsystem('bflmargins',ComputeBalancedFieldLengthResidual(),promotes_inputs=['mission:takeoff:v*'])
        self.connect('takeoff.takeoff:distance','bflmargins.takeoff:distance')
        self.connect('takeoff.takeoff:distance_abort','bflmargins.takeoff:distance_abort')

        implicit_solve = self.add_subsystem('implicit_solve',SolveImplicitStates(num_integration_intervals_per_seg=n_int_per_seg))
        self.connect('mission.thrust_resid.thrust_residual','implicit_solve.thrust_residual')
        self.connect('bflmargins.BFL_combined','implicit_solve.BFL_residual')
        #dv_comp.add_output('mission:takeoff:v1',val=48,units='m/s')
        # self.connect('mission:takeoff:v1','takeoff.v0v1_dist.upper_limit')
        # self.connect('mission:takeoff:v1','takeoff.v1vr_dist.lower_limit')
        # self.connect('mission:takeoff:v1','takeoff.v1v0_dist.lower_limit')
        # self.connect('mission:takeoff:v1','takeoff.reaction.mission:takeoff:vr')
        self.connect('implicit_solve.mission:takeoff:v1','mission:takeoff:v1')
        self.connect('implicit_solve.throttle','throttle_combiner.motor1:throttle:mission')

        components_list = ['eng1']
        opcost = self.add_subsystem('operating_cost',OperatingCost(n_components=len(components_list),n_batteries=None))
        self.connect('OEW.OEW','operating_cost.OEW')
        self.connect('fuel_burn','operating_cost.fuel_burn')
        for i, component in enumerate(components_list):
            self.connect('propmodel.'+component+'.component_weight','operating_cost.component_'+str(i+1)+'_weight')
            self.connect('propmodel.'+component+'.component_cost','operating_cost.component_'+str(i+1)+'_NR_cost')

        dummy_range = self.add_subsystem('dummypayload',DummyPayload(),promotes_outputs=['payload_objective'])
        self.connect('mission:payload','dummypayload.payload_DV')


class SolveImplicitStates(ImplicitComponent):
    def initialize(self):
        self.options.declare('num_integration_intervals_per_seg',default=5,desc="Number of Simpson intervals to use per seg (eg. climb, cruise, descend). Number of time points is 2N+1")

    def setup(self):
        n_int_per_seg = self.options['num_integration_intervals_per_seg']
        nn = (n_int_per_seg*2 + 1)
        nn_tot = 6*nn+2
        self.add_input('thrust_residual', units='N',shape=(3*nn,))
        self.add_input('BFL_residual',units='m')

        #self.add_input('mission_MTOW_margin', units='kg', shape=(n_seg*nn,))
        self.add_output('throttle',shape=(3*nn,))
        self.add_output('mission:takeoff:v1',units='m/s',val=39)

        self.declare_partials(['throttle'],['thrust_residual'],val=sp.eye(3*nn))
        self.declare_partials(['mission:takeoff:v1'],['BFL_residual'],val=1)

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['throttle'] = inputs['thrust_residual']
        residuals['mission:takeoff:v1'] = inputs['BFL_residual']


def define_analysis(n_int_per_seg):
    """
    This function sets up the problem with all DVs and constraints necessary to perform analysis only (drives throttle residuals and BFL residuals to zero).
    This does NOT ensure that the airplane has enough fuel capacity or gross weight to fly the mission.
    """
    prob = Problem()
    prob.model= TotalAnalysis(num_integration_intervals_per_seg=n_int_per_seg)
    nn = n_int_per_seg*2+1
    nn_tot_m = 3*(n_int_per_seg*2+1)
    nn_tot_to = 3*(n_int_per_seg*2+1)+2
    nn_tot = 6*(n_int_per_seg*2+1)+2

    prob.model.options['assembled_jac_type'] = 'csc'
    prob.model.nonlinear_solver=NewtonSolver()
    prob.model.linear_solver = DirectSolver(assemble_jac=True)
    prob.model.nonlinear_solver.options['solve_subsystems'] = True
    prob.model.nonlinear_solver.options['maxiter'] = 10
    prob.model.nonlinear_solver.options['atol'] = 1e-6
    prob.model.nonlinear_solver.options['rtol'] = 1e-6


    prob.driver = ScipyOptimizeDriver()
    return prob, nn_tot, nn_tot_m, nn_tot_to


if __name__ == "__main__":
    n_int_per_seg = 5
    prob, nn_tot, nn_tot_m, nn_tot_to = define_analysis(n_int_per_seg)

    run_type = 'analysis'
    if run_type == 'optimization':
        print('======Performing Multidisciplinary Design Optimization===========')
        prob.model.add_design_var('ac:weights:MTOW', lower=2500, upper=3500)
        prob.model.add_design_var('ac:geom:wing:S_ref',lower=9,upper=30)
        prob.model.add_design_var('ac:propulsion:engine:rating',lower=500,upper=1400)
        prob.model.add_design_var('ac:weights:W_fuel_max',lower=800,upper=3000)
        prob.model.add_constraint('missionmargins.mission:MTOW_margin',equals=0.0)
        prob.model.add_constraint('missionmargins.mission:fuel_capacity_margin',equals=0.0)
        prob.model.add_constraint('takeoff.takeoff:distance',upper=807)
        prob.model.add_constraint('vstall.Vstall_eas',upper=41.88)
        # prob.model.add_design_var('mission_eas_climb',lower=85,upper=300)
        # prob.model.add_design_var('mission_eas_cruise',lower=150,upper=300)
    elif run_type == 'max_range':
        print('======Analyzing Design Range at Given MTOW===========')
        prob.model.add_design_var('mission:range',lower=1000,upper=2500)
        prob.model.add_constraint('missionmargins.mission:MTOW_margin',equals=0.0)
    else:
        print('======Analyzing Fuel Burn for Given Mision============')

    prob.model.add_objective('mission.mission:total_fuel')
    prob.setup(mode='fwd')
    prob['implicit_solve.mission:takeoff:v1'] = 30
    prob['implicit_solve.throttle'] = np.ones(nn_tot_m)*0.7
    prob['OEW.const.structural_fudge'] = 1.4

    prob.run_model()
    takeoff_check(prob)
    #prob.check_partials(compact_print=True)
    print('Design range: '+str(prob.get_val('mission:range',units='NM')))
    print('MTOW: '+str(prob.get_val('ac:weights:MTOW',units='lb')))
    print('OEW: '+str(prob.get_val('OEW.OEW',units='lb')))
    print('Fuel cap:'+str(prob.get_val('ac:weights:W_fuel_max',units='lb')))
    print('MTOW margin: '+str(prob.get_val('missionmargins.mission:MTOW_margin',units='lb')))

    print('Eng power:'+str(prob.get_val('ac:propulsion:engine:rating',units='hp')))
    print('Prop diam:'+str(prob.get_val('ac:propulsion:propeller:diameter',units='m')))

    print('TO (continue):'+str(prob.get_val('takeoff.takeoff:distance',units='ft')))
    print('TO (abort):'+str(prob.get_val('takeoff.takeoff:distance_abort',units='ft')))
    print('Stall speed'+str(prob.get_val('vstall.Vstall_eas',units='kn')))
    print('Rotate speed'+str(prob.get_val('mission:takeoff:vr',units='kn')))
    print('Decision speed'+str(prob.get_val('implicit_solve.mission:takeoff:v1',units='kn')))
    print('S_ref: ' +str(prob.get_val('ac:geom:wing:S_ref',units='ft**2')))

    print('Mission Fuel burn: '+ str(prob.get_val('mission.mission:total_fuel',units='lb')))
    print('TO fuel burn: '+ str(prob.get_val('takeoff.takeoff:total_fuel',units='lb')))
    print('Total fuel burn:' +str(prob.get_val('fuel_burn',units='lb')))

    print('V0V1 dist: '+str(prob['takeoff.v0v1_dist.delta_quantity']))
    print('V1VR dist: '+str(prob['takeoff.v1vr_dist.delta_quantity']))
    print('Braking dist:'+str(prob['takeoff.v1v0_dist.delta_quantity']))
    print('Climb angle(rad):'+str(prob['takeoff.takeoff:climb:gamma']))
    print('h_trans:'+str(prob['takeoff.h_transition']))
    print('s_trans:'+str(prob['takeoff.s_transition']))
    print('s_climb:'+str(prob['takeoff.s_climb']))



    #prob.model.list_inputs(print_arrays=True)
    #prob.model.list_outputs(print_arrays=True)

    # print(prob['mission_h_cruise'])

    # # # print "------Prop 1-------"
    # # print('Thrust: ' + str(prob['propmodel.prop1.thrust']))
    # # plt.plot(prob['propmodel.prop1.thrust'])
    # # plt.show()

    # # print('Weight: ' + str(prob['propmodel.prop1.component_weight']))
    # dtclimb = prob['mission:climb:dt']
    # dtcruise = prob['mission:cruise:dt']
    # dtdesc = prob['mission:descent:dt']
    # n_int = 3
    # timevec = np.concatenate([np.linspace(0,2*n_int*dtclimb,2*n_int+1),np.linspace(2*n_int*dtclimb,2*n_int*dtclimb+2*n_int*dtcruise,2*n_int+1),np.linspace(2*n_int*(dtclimb+dtcruise),2*n_int*(dtclimb+dtcruise+dtdesc),2*n_int+1)])
    # plots = True
    # if plots:
    #     print('Flight conditions')
    #     plt.figure(1)
    #     plt.plot(timevec, prob['conds.fltcond:mission:Ueas'],'b.')
    #     plt.plot(timevec, prob['atmos.trueairspeed.fltcond:Utrue_mission'],'b-')
    #     plt.plot(timevec, prob['gs.mission:groundspeed'],'g-')
    #     plt.title('Equivalent and true airspeed vs gs')

    #     print('Propulsion conditions')
    #     plt.figure(2)
    #     plt.plot(timevec, prob['thrust'])
    #     plt.title('Thrust')

    #     plt.figure(3)
    #     plt.plot(timevec, prob['mission:fuel_flow'])
    #     plt.title('Fuel flow')

    #     plt.figure(4)
    #     # plt.plot(np.delete(timevec,[0,20,41]),np.cumsum(prob['mission:segment_fuel']))
    #     plt.plot(timevec,prob['mission:weights'])
    #     plt.title('Weight')

    #     plt.figure(5)
    #     plt.plot(timevec,prob['fltcond:mission:CL'])
    #     plt.title('CL')

    #     plt.figure(6)
    #     plt.plot(timevec,prob['aero_drag'])
    #     plt.title('Drag')

    #     plt.figure(7)
    #     plt.plot(timevec,prob['propmodel.eng1.shaft_power_out'])
    #     plt.title('Shaft power')
    #     plt.show()
    # print('Total fuel flow (totalizer):' + str(prob['mission:total_fuel']))
    # print('Total fuel flow:' + str(np.sum(prob['mission:segment_fuel'])))


    # #prob.model.list_inputs()
    # #prob.model.list_outputs()
    #prob.check_partials(compact_print=True)
    # #prob.check_totals(compact_print=True)

