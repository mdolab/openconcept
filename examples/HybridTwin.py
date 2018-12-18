from __future__ import division
from openmdao.api import Problem, Group, IndepVarComp, DirectSolver, NewtonSolver, SqliteRecorder
from openmdao.api import ScipyOptimizeDriver, ExplicitComponent, ImplicitComponent
# ------This is a hack for testing scripts on openconcept source directories that haven't been installed.
# By default, a script includes its path in sys.path. We need to add the folder one level higher (where the interpreter is run from)
# The script can be run from the root git directory as 'python examples/script.py' and the latest openconcept package will be imported by default
import sys, os
sys.path.insert(0,os.getcwd())
#-------These imports are generic and should be left alone
import numpy as np
import scipy.sparse as sp
from openconcept.utilities.math import VectorConcatenateComp, VectorSplitComp
from openconcept.utilities.dict_indepvarcomp import DictIndepVarComp
from openconcept.analysis.takeoff import BalancedFieldLengthTakeoff, takeoff_check
from openconcept.analysis.mission import MissionAnalysis
from openconcept.utilities.linearinterp import LinearInterpolator
from openconcept.utilities.nodes import compute_num_nodes
#These imports are particular to this airplane
# If run from the root git  directory as 'python examples/script.py', these imports are found because the script's path is added to sys.path by default.

from methods.weights_twin_hybrid import TwinSeriesHybridEmptyWeight
from methods.costs_commuter import OperatingCost
from aircraft_data.KingAirC90GT import data as acdata
from aircraft_data.KingAirC90GT_mission import data as missiondata
from propulsion_layouts.simple_series_hybrid import TwinSeriesHybridElectricPropulsionSystem

class DummyPayload(ExplicitComponent):
    def setup(self):
        self.add_input('payload_DV', units='lb')
        self.add_output('payload_objective', units='lb')
        self.declare_partials(['payload_objective'], ['payload_DV'], val=1)
    def compute(self, inputs, outputs):
        outputs['payload_objective'] = inputs['payload_DV']

class AugmentedFBObjective(ExplicitComponent):
    def setup(self):
        self.add_input('fuel_burn', units='kg')
        self.add_input('ac|weights|MTOW', units='kg')
        self.add_output('mixed_objective', units='kg')
        self.declare_partials(['mixed_objective'], ['fuel_burn'], val=1)
        self.declare_partials(['mixed_objective'], ['ac|weights|MTOW'], val=1/100)
    def compute(self, inputs, outputs):
        outputs['mixed_objective'] = inputs['fuel_burn'] + inputs['ac|weights|MTOW']/100

class TotalAnalysis(Group):
    """This analysis group calculates TOFL and mission fuel burn as well as many other quantities for an example airplane. Elements may be overridden or replaced as needed.
        Should be instantiated as the top-level model
    """

    def initialize(self):
        self.options.declare('n_int_per_seg',default=5,desc="Number of Simpson intervals to use per seg (eg. climb, cruise, descend). Number of analysis points is 2N+1")
        self.options.declare('specific_energy',default=750,desc="Specific energy of the battery in Wh/kg")
        self.options.declare('constant_hybridization',default=True,desc="When true, uses a single hybridization factor for the entire mission. When False, segment endpoints can be varied independently")

    def setup(self):
        n_int_per_seg = self.options['n_int_per_seg']
        mission_segments = ['climb','cruise','descent']
        nn, nn_tot_to, nn_tot_m, nn_tot = compute_num_nodes(n_int_per_seg, mission_segments)

        constant_hybridization = self.options['constant_hybridization']
        specific_energy = self.options['specific_energy']

        #Define input variables
        dv_comp = self.add_subsystem('dv_comp',DictIndepVarComp(acdata),promotes_outputs=["*"])
        #eventually replace the following aerodynamic parameters with an analysis module (maybe OpenAeroStruct)
        dv_comp.add_output_from_dict('ac|aero|CLmax_flaps30')
        dv_comp.add_output_from_dict('ac|aero|polar|e')
        dv_comp.add_output_from_dict('ac|aero|polar|CD0_TO')
        dv_comp.add_output_from_dict('ac|aero|polar|CD0_cruise')

        dv_comp.add_output_from_dict('ac|geom|wing|S_ref')
        dv_comp.add_output_from_dict('ac|geom|wing|AR')
        dv_comp.add_output_from_dict('ac|geom|wing|c4sweep')
        dv_comp.add_output_from_dict('ac|geom|wing|taper')
        dv_comp.add_output_from_dict('ac|geom|wing|toverc')
        dv_comp.add_output_from_dict('ac|geom|hstab|S_ref')
        dv_comp.add_output_from_dict('ac|geom|hstab|c4_to_wing_c4')
        dv_comp.add_output_from_dict('ac|geom|vstab|S_ref')
        dv_comp.add_output_from_dict('ac|geom|fuselage|S_wet')
        dv_comp.add_output_from_dict('ac|geom|fuselage|width')
        dv_comp.add_output_from_dict('ac|geom|fuselage|length')
        dv_comp.add_output_from_dict('ac|geom|fuselage|height')
        dv_comp.add_output_from_dict('ac|geom|nosegear|length')
        dv_comp.add_output_from_dict('ac|geom|maingear|length')

        dv_comp.add_output_from_dict('ac|weights|MTOW')
        dv_comp.add_output_from_dict('ac|weights|W_fuel_max')
        dv_comp.add_output_from_dict('ac|weights|MLW')
        dv_comp.add_output_from_dict('ac|weights|W_battery')

        dv_comp.add_output_from_dict('ac|propulsion|engine|rating')
        dv_comp.add_output_from_dict('ac|propulsion|propeller|diameter')
        dv_comp.add_output_from_dict('ac|propulsion|generator|rating')
        dv_comp.add_output_from_dict('ac|propulsion|motor|rating')

        dv_comp.add_output_from_dict('ac|num_passengers_max')
        dv_comp.add_output_from_dict('ac|q_cruise')
        dv_comp.add_output_from_dict('ac|num_engines')

        mission_data_comp = self.add_subsystem('mission_data_comp',DictIndepVarComp(missiondata),promotes_outputs=["*"])
        mission_data_comp.add_output_from_dict('takeoff|h')
        mission_data_comp.add_output_from_dict('climb|h0')
        mission_data_comp.add_output_from_dict('climb|time')
        mission_data_comp.add_output_from_dict('climb|Ueas')
        mission_data_comp.add_output_from_dict('cruise|h0')
        mission_data_comp.add_output_from_dict('cruise|Ueas')
        mission_data_comp.add_output_from_dict('descent|h0')
        mission_data_comp.add_output_from_dict('descent|hf')
        mission_data_comp.add_output_from_dict('descent|time')
        mission_data_comp.add_output_from_dict('descent|Ueas')
        mission_data_comp.add_output_from_dict('design_range')
        mission_data_comp.add_output_from_dict('payload')

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
        #set the prop to 1900 rpm for all time
        controls.add_output('prop|rpm|takeoff', val=np.ones(nn_tot_to) * 1900, units='rpm')
        controls.add_output('prop|rpm|mission', val=np.ones(nn_tot_m) * 1900, units='rpm')

        motor1_TO_throttle_vec = np.concatenate([np.ones(nn),
                                          np.ones(nn) * 1.0,
                                          np.zeros(nn),
                                          np.ones(2) * 1.0])*1.1
        motor2_TO_throttle_vec = np.concatenate([np.ones(nn),
                                          np.zeros(nn) * 0.0,
                                          np.zeros(nn),
                                          np.zeros(2) * 0.0])*1.1
        controls.add_output('motor1|throttle|takeoff', val=motor1_TO_throttle_vec)
        controls.add_output('motor2|throttle|takeoff', val=motor2_TO_throttle_vec)

        controls.add_output('hybrid_split|takeoff', val=1)
        controls.add_output('eng1|throttle|takeoff', val=np.zeros(nn_tot_to))
        self.add_subsystem('hybrid_TO',LinearInterpolator(num_nodes=nn_tot_to))
        self.connect('controls.hybrid_split|takeoff',['hybrid_TO.start_val','hybrid_TO.end_val'])

        if constant_hybridization:
            controls.add_output('hybrid_split|percentage', val=0.5)
        hybrid_comps = []
        for segment_name in mission_segments:
            hybrid_comps.append('hybrid_'+segment_name)
            self.add_subsystem('hybrid_'+segment_name, LinearInterpolator(num_nodes=nn))
            if constant_hybridization:
                self.connect('controls.hybrid_split|percentage','hybrid_'+segment_name+'.start_val')
                self.connect('controls.hybrid_split|percentage','hybrid_'+segment_name+'.end_val')
            else:
                controls.add_output('hybrid_split|'+segment_name+'_0', val=0.5)
                controls.add_output('hybrid_split|'+segment_name+'_f', val=0.5)
                self.connect('hybrid_split|'+segment_name+'_0','hybrid_'+segment_name+'.start_val')
                self.connect('hybrid_split|'+segment_name+'_f','hybrid_'+segment_name+'.end_val')
        hybrid_combiner = self.add_subsystem('hybrid_combiner',
                                             VectorConcatenateComp(output_name='hybrid_split|mission',
                                             input_names=hybrid_comps,
                                             vec_sizes=[nn,nn,nn]))
        for segment_name in mission_segments:
            self.connect('hybrid_'+segment_name+'.vec','hybrid_combiner.hybrid_'+segment_name)

        #==Calculate engine thrusts and fuel flows. You will need to override this module to vary number of engines, prop architecture, etc
        # Your propulsion model must promote up a single variable called "thrust" and a single variable called "fuel_flow". You may need to sum these at a lower level in the prop model group
        # You will probably need to add more control parameters if you use multiple engines. You may also need to add implicit solver states if, e.g. turbogenerator power setting depends on motor power setting

        #connect control settings to the various states in the propulsion model

        #now we have flight conditions and propulsion outputs for all flight conditions. Split into our individual analysis phases

        #==This next module calculates balanced field length, if applicable. Your optimizer or solver MUST implicitly drive the abort distance and oei takeoff distances to the same value by varying v1

        self.add_subsystem('takeoff',BalancedFieldLengthTakeoff(n_int_per_seg=n_int_per_seg,
                                                                track_fuel=True,track_battery=True,
                                                                propulsion_system=TwinSeriesHybridElectricPropulsionSystem(num_nodes=nn_tot_to, specific_energy=specific_energy)),
                                                                promotes_inputs=['ac|aero*','ac|geom|*','ac|propulsion|*','ac|weights|*','takeoff|h'])

        self.connect('controls.prop|rpm|takeoff',['takeoff.propmodel.prop1.rpm','takeoff.propmodel.prop2.rpm'])
        self.connect('controls.motor1|throttle|takeoff', 'takeoff.propmodel.motor1.throttle')
        self.connect('controls.motor2|throttle|takeoff', 'takeoff.propmodel.motor2.throttle')
        self.connect('hybrid_TO.vec','takeoff.propmodel.hybrid_split.power_split_fraction')
        self.connect('controls.eng1|throttle|takeoff','takeoff.propmodel.eng1.throttle')

        #==This module computes fuel consumption during the entire mission
        mission_promote_inputs = ["ac|aero|*","ac|geom|*",'ac|propulsion|*','ac|weights|*','OEW','*|Ueas','*|h0','*|hf','*|time','design_range','payload']
        self.add_subsystem('design_mission',MissionAnalysis(n_int_per_seg=n_int_per_seg,track_battery=True,
                           propulsion_system=TwinSeriesHybridElectricPropulsionSystem(num_nodes=nn_tot_m, specific_energy=specific_energy)),
                           promotes_inputs=mission_promote_inputs)
        self.connect('design_mission.throttle',['design_mission.propmodel.motor1.throttle','design_mission.propmodel.motor2.throttle'])
        self.connect('takeoff.weight_after_takeoff','design_mission.weight_initial')
        self.connect('hybrid_combiner.hybrid_split|mission','design_mission.propmodel.hybrid_split.power_split_fraction')
        self.connect('controls.prop|rpm|mission',['design_mission.propmodel.prop1.rpm','design_mission.propmodel.prop2.rpm'])

        #==This module is an empirical weight tool specific to a single-engine turboprop airplane. You will need to modify or replace it.
        self.add_subsystem('OEW',TwinSeriesHybridEmptyWeight(),promotes_inputs=["*"],promotes_outputs=['OEW'])
        self.connect('ac|propulsion|engine|rating','P_TO')
        self.connect('design_mission.propmodel.propellers_weight','W_propeller')
        self.connect('design_mission.propmodel.eng1.component_weight','W_engine')
        self.connect('design_mission.propmodel.gen1.component_weight','W_generator')
        self.connect('design_mission.propmodel.motors_weight','W_motors')

        #==Finally, we need to compute certain quantities to ensure the airplane is feasible. Compute whether enough fuel volume exists, and whether the airplane burned more fuel than it can carry
        self.connect('takeoff.total_fuel','design_mission.residuals.takeoff|total_fuel')
        self.connect('takeoff.total_battery_energy','design_mission.residuals.takeoff|total_battery_energy')
        self.connect('design_mission.propmodel.batt1.max_energy','design_mission.residuals.battery_max_energy')

        #==Calculate the difference between the one-engine-out abort distance and one-engine-out takeoff distance with obstacle clearance
        self.add_subsystem('implicit_solve',SolveImplicitStates(n_int_per_seg=n_int_per_seg))
        self.connect('design_mission.propmodel.eng_gen_resid.eng_gen_residual','implicit_solve.eng_gen_residual')
        self.connect('implicit_solve.eng_throttle','design_mission.propmodel.eng1.throttle')

        components_list = ['eng1','motor1','motor2','gen1']
        opcost = self.add_subsystem('operating_cost',OperatingCost(n_components=len(components_list),n_batteries=1))
        self.connect('design_mission.propmodel.batt1.component_cost','operating_cost.battery_1_NR_cost')
        self.connect('design_mission.battery_energy_used','operating_cost.battery_1_energy_used')

        self.connect('OEW','operating_cost.OEW')
        self.connect('design_mission.fuel_burn','operating_cost.fuel_burn')
        for i, component in enumerate(components_list):
            self.connect('design_mission.propmodel.'+component+'.component_weight','operating_cost.component_'+str(i+1)+'_weight')
            self.connect('design_mission.propmodel.'+component+'.component_cost','operating_cost.component_'+str(i+1)+'_NR_cost')

        dummy_range = self.add_subsystem('dummypayload',DummyPayload(),promotes_outputs=['payload_objective'])
        self.connect('payload','dummypayload.payload_DV')
        dummy_obj = self.add_subsystem('dummyobj',AugmentedFBObjective(),promotes_inputs=['ac|weights|MTOW'],promotes_outputs=['mixed_objective'])
        self.connect('design_mission.fuel_burn','dummyobj.fuel_burn')

class SolveImplicitStates(ImplicitComponent):
    def initialize(self):
        self.options.declare('n_int_per_seg',default=5,desc="Number of Simpson intervals to use per seg (eg. climb, cruise, descend). Number of analysis points is 2N+1")

    def setup(self):
        n_int_per_seg = self.options['n_int_per_seg']
        nn = (n_int_per_seg*2 + 1)
        nn_tot = 3*nn
        self.add_input('eng_gen_residual', units='kW', shape=(nn_tot,))
        self.add_output('eng_throttle',shape=(nn_tot,))
        self.declare_partials(['eng_throttle'], ['eng_gen_residual'], val=sp.eye(nn_tot))

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['eng_throttle'] = inputs['eng_gen_residual']


def define_analysis(n_int_per_seg,specific_energy):
    """
    This function sets up the problem with all DVs and constraints necessary to perform analysis only (drives throttle residuals and BFL residuals to zero).
    This does NOT ensure that the airplane has enough fuel capacity or gross weight to fly the mission.
    """
    prob = Problem()
    prob.model= TotalAnalysis(n_int_per_seg=n_int_per_seg,specific_energy=specific_energy)
    prob.model.options['assembled_jac_type'] = 'csc'
    prob.model.nonlinear_solver=NewtonSolver()
    prob.model.linear_solver = DirectSolver(assemble_jac=True)
    prob.model.nonlinear_solver.options['solve_subsystems'] = True
    prob.model.nonlinear_solver.options['maxiter'] = 10
    prob.model.nonlinear_solver.options['atol'] = 1e-7
    prob.model.nonlinear_solver.options['rtol'] = 1e-7
    prob.driver = ScipyOptimizeDriver()
    prob.driver.options['tol'] = 1e-7
    return prob


if __name__ == "__main__":
    #design_ranges = [350,400,450,500,550,600,650,700]
    #specific_energies = [250,300,350,400,450,500,550,600,650,700,750,800]
    design_ranges = [400]
    specific_energies = [500]
	#redo spec range 450, spec energy 700, 750, 800
    for design_range in design_ranges:
        for spec_energy in specific_energies:
            n_int_per_seg = 3
            mission_segments=['climb','cruise','descent']
            prob = define_analysis(n_int_per_seg,spec_energy)
            nn, nn_tot_to, nn_tot_m, nn_tot = compute_num_nodes(n_int_per_seg, mission_segments)
            filename_to_save = 'case_'+str(spec_energy)+'_'+str(design_range)+'.sql'
            recorder = SqliteRecorder(filename_to_save)
            prob.model.add_recorder(recorder)

            run_type = 'optimization'
            if run_type == 'optimization':
                print('======Performing Multidisciplinary Design Optimization===========')
                prob.model.add_design_var('ac|weights|MTOW', lower=3000, upper=5700)
                prob.model.add_design_var('ac|geom|wing|S_ref',lower=9,upper=40)
                #prob.model.add_design_var('ac|propulsion|propeller|diameter',lower=2.2,upper=2.5)
                prob.model.add_design_var('ac|propulsion|engine|rating',lower=1,upper=3000)
                prob.model.add_design_var('ac|propulsion|motor|rating',lower=1,upper=3000)
                prob.model.add_design_var('ac|propulsion|generator|rating',lower=1,upper=3000)
                prob.model.add_design_var('ac|weights|W_battery',lower=20,upper=2250)
                prob.model.add_design_var('ac|weights|W_fuel_max',lower=500,upper=3000)
                prob.model.add_design_var('controls.hybrid_split|percentage',lower=0,upper=1)

                prob.model.add_constraint('design_mission.residuals.MTOW_margin',equals=0.0)
                prob.model.add_constraint('design_mission.residuals.fuel_capacity_margin',lower=0.0)
                prob.model.add_constraint('takeoff.distance_continue',upper=1357)
                prob.model.add_constraint('takeoff.vstall.Vstall_eas',upper=42.0)
                prob.model.add_constraint('design_mission.residuals.battery_margin',lower=0.0)
                prob.model.add_constraint('design_mission.propmodel.eng1.component_sizing_margin',upper=1.0*np.ones(nn_tot_m))
                prob.model.add_constraint('design_mission.propmodel.gen1.component_sizing_margin',upper=1.0*np.ones(nn_tot_m))
                prob.model.add_constraint('design_mission.propmodel.batt1.component_sizing_margin',upper=1.0*np.ones(nn_tot_m))
                prob.model.add_constraint('takeoff.propmodel.batt1.component_sizing_margin',upper=1.0*np.ones(nn_tot_to))
                prob.model.add_constraint('takeoff.climb|gamma',lower=0.009)
                #prob.model.add_constraint('implicit_solve.motor_throttle',upper=1.05*np.ones(nn_tot_m))
                prob.model.add_objective('mixed_objective')
            elif run_type == 'comp_sizing':
                print('======Performing Component Sizing Optimization===========')
                prob.model.add_design_var('ac|propulsion|engine|rating',lower=1,upper=3000)
                prob.model.add_design_var('ac|propulsion|motor|rating',lower=1,upper=3000)
                prob.model.add_design_var('ac|propulsion|generator|rating',lower=1,upper=3000)
                prob.model.add_design_var('ac|weights|W_battery',lower=20,upper=2250)
                prob.model.add_design_var('controls.hybrid_split|percentage',lower=0,upper=1)

                prob.model.add_constraint('missionmargins.mission|MTOW_margin',equals=0.0)
                prob.model.add_constraint('takeoff.takeoff|distance',upper=1357)
                prob.model.add_constraint('missionmargins.mission|battery_margin',lower=0.0)
                prob.model.add_constraint('propmodel.eng1.component_sizing_margin',upper=1.0*np.ones(nn_tot))
                prob.model.add_constraint('propmodel.gen1.component_sizing_margin',upper=1.0*np.ones(nn_tot))
                prob.model.add_constraint('propmodel.batt1.component_sizing_margin',upper=1.0*np.ones(nn_tot))
                prob.model.add_constraint('takeoff.takeoff|climb|gamma',lower=0.009)
                prob.model.add_constraint('design_mission.throttle',upper=1.05*np.ones(nn_tot_m))
                prob.model.add_objective('fuel_burn')



            else:
                print('======Analyzing Fuel Burn for Given Mision============')
                prob.model.add_objective('mixed_objective')

            prob.setup(mode='fwd')
            prob['takeoff.v1_solve.takeoff|v1'] = 40
            prob['design_mission.throttle'] = np.ones(nn_tot_m)*0.5
            prob['implicit_solve.eng_throttle'] = np.ones(nn_tot_m)*0.5
            prob['OEW.const.structural_fudge'] = 2.0
            prob['design_range'] = design_range
            prob['ac|propulsion|propeller|diameter'] = 2.2
            prob['ac|propulsion|engine|rating'] = 1117.2


            #prob.set_val('mission|range',1000,'NM')
            prob.run_driver()
            takeoff_check(prob)
            prob.cleanup()
            #prob.check_partials(compact_print=True)
            print('Design range: '+str(prob.get_val('design_mission.range', units='NM')))
            print('MTOW: '+str(prob.get_val('ac|weights|MTOW', units='lb')))
            print('OEW: '+str(prob.get_val('OEW', units='lb')))
            print('Battery wt: '+str(prob.get_val('ac|weights|W_battery', units='lb')))
            print('Fuel cap:'+str(prob.get_val('ac|weights|W_fuel_max', units='lb')))
            print('MTOW margin: '+str(prob.get_val('design_mission.residuals.MTOW_margin', units='lb')))
            print('Battery margin: '+str(prob.get_val('design_mission.residuals.battery_margin', units='J')))

            print('Eng power:'+str(prob.get_val('ac|propulsion|engine|rating', units='hp')))
            print('Gen power:'+str(prob.get_val('ac|propulsion|generator|rating', units='hp')))
            print('Motor power:'+str(prob.get_val('ac|propulsion|motor|rating', units='hp')))
            print('Hybrid split|'+str(prob.get_val('controls.hybrid_split|percentage', units=None)))
            print('Prop diam:'+str(prob.get_val('ac|propulsion|propeller|diameter', units='m')))

            print('TO (continue):'+str(prob.get_val('takeoff.distance_continue', units='ft')))
            print('TO (abort):'+str(prob.get_val('takeoff.distance_abort', units='ft')))
            print('Stall speed'+str(prob.get_val('takeoff.vstall.Vstall_eas', units='kn')))
            print('Rotate speed'+str(prob.get_val('takeoff.takeoff|vr', units='kn')))
            print('Decision speed'+str(prob.get_val('takeoff.takeoff|v1', units='kn')))
            print('S_ref: ' +str(prob.get_val('ac|geom|wing|S_ref', units='ft**2')))

            print('Mission Fuel burn: '+ str(prob.get_val('design_mission.mission_total_fuel', units='lb')))
            print('TO fuel burn: '+ str(prob.get_val('takeoff.total_fuel', units='lb')))
            print('Total fuel burn:' +str(prob.get_val('design_mission.fuel_burn', units='lb')))

            print('V0V1 dist: '+str(prob['takeoff.v0v1_dist.delta_quantity']))
            print('V1VR dist: '+str(prob['takeoff.v1vr_dist.delta_quantity']))
            print('Braking dist:'+str(prob['takeoff.v1v0_dist.delta_quantity']))
            print('Climb angle(rad):'+str(prob['takeoff.climb|gamma']))
            print('h_trans:'+str(prob['takeoff.h_transition']))
            print('s_trans:'+str(prob['takeoff.s_transition']))
            print('s_climb|'+str(prob['takeoff.s_climb']))
            #print('Mission throttle settings:'+str(prob['implicit_solve.throttle']))s
            #print('Fuel_flows:'+str(prob['fuel_flow']))

            #prob.model.list_outputs()

            # prob.check_partials(compact_print=True)

            # print(str(prob['propmodel.eng1.component_sizing_margin']))
            # print(str(prob['propmodel.gen1.component_sizing_margin']))
