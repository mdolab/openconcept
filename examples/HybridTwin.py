from openmdao.api import Problem, Group, IndepVarComp, DirectSolver, NewtonSolver, SqliteRecorder
from openmdao.api import ScipyOptimizeDriver, ExplicitComponent, ImplicitComponent
#-------These imports are generic and should be left alone
import numpy as np
import scipy.sparse as sp
from openconcept.analysis.atmospherics.compute_atmos_props import ComputeAtmosphericProperties
from openconcept.utilities.math import VectorConcatenateComp, VectorSplitComp
from openconcept.utilities.dict_indepvarcomp import DictIndepVarComp
from openconcept.analysis.aerodynamics import StallSpeed
from openconcept.analysis.takeoff import TakeoffFlightConditions, TakeoffTotalDistance, ComputeBalancedFieldLengthResidual, takeoff_check
from openconcept.analysis.mission import MissionFlightConditions, MissionNoReserves, ComputeDesignMissionResidualsBattery
from openconcept.utilities.linearinterp import LinearInterpolator
#These imports are particular to this airplane
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
        nn_tot_to = (2*n_int_per_seg+1)*3 +2 #v0v1,v1vr,v1v0, vtr, v2
        nn_tot_m = (2*n_int_per_seg+1)*3
        nn_tot=nn_tot_to+nn_tot_m
        nn = (2*n_int_per_seg+1)

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
        mission_data_comp.add_output_from_dict('mission|takeoff|h')
        mission_data_comp.add_output_from_dict('mission|climb|vs')
        mission_data_comp.add_output_from_dict('mission|climb|Ueas')
        mission_data_comp.add_output_from_dict('mission|cruise|h')
        mission_data_comp.add_output_from_dict('mission|cruise|Ueas')
        mission_data_comp.add_output_from_dict('mission|descent|vs')
        mission_data_comp.add_output_from_dict('mission|descent|Ueas')
        mission_data_comp.add_output_from_dict('mission|range')
        mission_data_comp.add_output_from_dict('mission|payload')

        #== Compute the stall speed (necessary for takeoff analysis - leave this alone)
        vstall = self.add_subsystem('vstall', StallSpeed())
        self.connect('ac|weights|MTOW','vstall.weight')
        self.connect('ac|geom|wing|S_ref','vstall.ac|geom|wing|S_ref')
        self.connect('ac|aero|CLmax_flaps30', 'vstall.CLmax')

        # ==Calculate flight conditions for the takeoff and mission segments here (leave this alone)
        mission_conditions = self.add_subsystem('mission_conditions',
                                                MissionFlightConditions(n_int_per_seg=n_int_per_seg),
                                                promotes_inputs=["mission|*"],
                                                promotes_outputs=["mission|*", "fltcond|mission|*"])

        takeoff_conditions = self.add_subsystem('takeoff_conditions',
                                                TakeoffFlightConditions(n_int_per_seg=n_int_per_seg),
                                                promotes_inputs=["mission|takeoff|*"],
                                                promotes_outputs=["fltcond|takeoff|*",
                                                                  "mission|takeoff|*"])
        self.connect('vstall.Vstall_eas', 'takeoff_conditions.Vstall_eas')

        fltcondcombiner = VectorConcatenateComp(output_name='fltcond|h',
                                                 input_names=['fltcond|takeoff|h',
                                                              'fltcond|mission|h'],
                                                 units='m',
                                                 vec_sizes=[nn_tot_to, nn_tot_m])
        fltcondcombiner.add_relation(output_name='fltcond|Ueas',
                                      input_names=['fltcond|takeoff|Ueas',
                                                   'fltcond|mission|Ueas'],
                                      units='m/s',
                                      vec_sizes=[nn_tot_to, nn_tot_m])
        self.add_subsystem('fltcondcombiner', subsys=fltcondcombiner,
                           promotes_inputs=["fltcond|takeoff|*",
                                            "fltcond|mission|*"],
                           promotes_outputs=["fltcond|Ueas", "fltcond|h"])

        #==Calculate atmospheric properties and true airspeeds for all mission segments
        atmos = self.add_subsystem('atmos',
                                   ComputeAtmosphericProperties(num_nodes=nn_tot),
                                   promotes_inputs=["fltcond|h",
                                                    "fltcond|Ueas"],
                                   promotes_outputs=["fltcond|rho",
                                                     "fltcond|Utrue",
                                                     "fltcond|q"])

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
        controls.add_output('prop1|rpm', val=np.ones(nn_tot) * 1900, units='rpm')
        controls.add_output('prop2|rpm', val=np.ones(nn_tot) * 1900, units='rpm')

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
        self.add_subsystem('hybrid_TO',LinearInterpolator(num_nodes=nn_tot_to))
        self.connect('controls.hybrid_split|takeoff',['hybrid_TO.start_val','hybrid_TO.end_val'])

        hybrid_climb = self.add_subsystem('hybrid_climb',LinearInterpolator(num_nodes=nn))
        hybrid_cruise = self.add_subsystem('hybrid_cruise',LinearInterpolator(num_nodes=nn))
        hybrid_desc = self.add_subsystem('hybrid_desc',LinearInterpolator(num_nodes=nn))

        if constant_hybridization:
            controls.add_output('hybrid_split|percentage', val=0.5)
            self.connect('controls.hybrid_split|percentage','hybrid_climb.start_val')
            self.connect('controls.hybrid_split|percentage','hybrid_climb.end_val')
            self.connect('controls.hybrid_split|percentage','hybrid_cruise.start_val')
            self.connect('controls.hybrid_split|percentage','hybrid_cruise.end_val')
            self.connect('controls.hybrid_split|percentage','hybrid_desc.start_val')
            self.connect('controls.hybrid_split|percentage','hybrid_desc.end_val')
        else:
            controls.add_output('hybrid_split|climb_0', val=0.5)
            controls.add_output('hybrid_split|climb_f', val=0.5)
            controls.add_output('hybrid_split|cruise_0', val=0.5)
            controls.add_output('hybrid_split|cruise_f', val=0.5)
            controls.add_output('hybrid_split|desc_0', val=0.5)
            controls.add_output('hybrid_split|desc_f', val=0.5)
            self.connect('controls.hybrid_split|climb_0','hybrid_climb.start_val')
            self.connect('controls.hybrid_split|climb_f','hybrid_climb.end_val')
            self.connect('controls.hybrid_split|cruise_0','hybrid_cruise.start_val')
            self.connect('controls.hybrid_split|cruise_f','hybrid_cruise.end_val')
            self.connect('controls.hybrid_split|desc_0','hybrid_desc.start_val')
            self.connect('controls.hybrid_split|desc_f','hybrid_desc.end_val')

        hybrid_combiner = self.add_subsystem('hybrid_combiner',
                                             VectorConcatenateComp(output_name='hybrid_split|mission',
                                             input_names=['hybrid_climb',
                                                          'hybrid_cruise',
                                                          'hybrid_desc'],
                                             vec_sizes=[nn,nn,nn]))
        self.connect('hybrid_climb.vec','hybrid_combiner.hybrid_climb')
        self.connect('hybrid_cruise.vec','hybrid_combiner.hybrid_cruise')
        self.connect('hybrid_desc.vec','hybrid_combiner.hybrid_desc')
        #combine the various controls together into one vector
        throttle_combiner = VectorConcatenateComp(output_name='motor1|throttle',
                                                  input_names=['motor1|throttle|takeoff',
                                                               'motor1|throttle|mission'],
                                                  vec_sizes=[nn_tot_to, nn_tot_m])
        throttle_combiner.add_relation(output_name='motor2|throttle',
                                       input_names=['motor2|throttle|takeoff',
                                                    'motor2|throttle|mission'],
                                       vec_sizes=[nn_tot_to, nn_tot_m])
        throttle_combiner.add_relation(output_name='hybrid_split',
                                input_names=['hybrid_split|takeoff',
                                             'hybrid_split|mission'],
                                vec_sizes=[nn_tot_to, nn_tot_m])
        self.add_subsystem('throttle_combiner', subsys=throttle_combiner,
                           promotes_outputs=["motor*|throttle"])
        self.connect('controls.motor1|throttle|takeoff',
                     'throttle_combiner.motor1|throttle|takeoff')
        self.connect('controls.motor2|throttle|takeoff',
                     'throttle_combiner.motor2|throttle|takeoff')
        self.connect('hybrid_TO.vec','throttle_combiner.hybrid_split|takeoff')
        self.connect('hybrid_combiner.hybrid_split|mission','throttle_combiner.hybrid_split|mission')
        #==Calculate engine thrusts and fuel flows. You will need to override this module to vary number of engines, prop architecture, etc
        # Your propulsion model must promote up a single variable called "thrust" and a single variable called "fuel_flow". You may need to sum these at a lower level in the prop model group
        # You will probably need to add more control parameters if you use multiple engines. You may also need to add implicit solver states if, e.g. turbogenerator power setting depends on motor power setting

        prop = self.add_subsystem('propmodel',TwinSeriesHybridElectricPropulsionSystem(num_nodes=nn_tot, specific_energy=specific_energy),promotes_inputs=["fltcond|*","ac|propulsion|*","ac|weights|*"],promotes_outputs=["fuel_flow","thrust"])
        #connect control settings to the various states in the propulsion model
        self.connect('controls.prop1|rpm','propmodel.prop1.rpm')
        self.connect('controls.prop2|rpm','propmodel.prop2.rpm')
        self.connect('motor1|throttle','propmodel.motor1.throttle')
        self.connect('motor2|throttle','propmodel.motor2.throttle')
        self.connect('throttle_combiner.hybrid_split','propmodel.hybrid_split.power_split_fraction')

        #now we have flight conditions and propulsion outputs for all flight conditions. Split into our individual analysis phases
        #== Leave this alone==#
        splitter_inst = VectorSplitComp()

        inputs_to_split = ['fltcond|q','fltcond|Utrue','fuel_flow','thrust','battery_load']
        segments_to_split_into = ['takeoff','mission']
        units = ['N * m**-2','m/s','kg/s','N','kW']
        nn_each_segment = [nn_tot_to,nn_tot_m]

        for kth, input_name in enumerate(inputs_to_split):
            output_names_list = []
            for segment in segments_to_split_into:
                inpnamesplit = input_name.split('|')
                inpnamesplit.insert(-1,segment)
                output_names_list.append('|'.join(inpnamesplit))
            splitter_inst.add_relation(output_names=output_names_list, input_name=input_name, vec_sizes=nn_each_segment, units=units[kth])

        self.add_subsystem('splitter',subsys=splitter_inst, promotes_inputs=["*"], promotes_outputs=["*"])
        self.connect('propmodel.hybrid_split.power_out_A','battery_load')


        #==This next module calculates balanced field length, if applicable. Your optimizer or solver MUST implicitly drive the abort distance and oei takeoff distances to the same value by varying v1

        takeoff = self.add_subsystem('takeoff',TakeoffTotalDistance(n_int_per_seg=n_int_per_seg,track_fuel=True,track_battery=True),promotes_inputs=['ac|aero*','ac|weights|MTOW','ac|geom|*','fltcond|takeoff|*','takeoff|battery_load','takeoff|thrust','takeoff|fuel_flow','mission|takeoff|v*'])

        #==This module computes fuel consumption during the entire mission
        mission = self.add_subsystem('mission',MissionNoReserves(n_int_per_seg=n_int_per_seg,track_battery=True),promotes_inputs=["ac|aero|*","ac|geom|*","fltcond|mission|*","mission|thrust","mission|fuel_flow",'mission|battery_load',"mission|*"])
        #remember that you will need to set the mission throttle implicitly using the optimizer/solver. This was done above when we mashed the control vectors all together.
        self.connect('takeoff.weight_after_takeoff','mission|weight_initial')

        #==This module is an empirical weight tool specific to a single-engine turboprop airplane. You will need to modify or replace it.
        self.add_subsystem('OEW',TwinSeriesHybridEmptyWeight(),promotes_inputs=["*"])
        self.connect('ac|propulsion|engine|rating','P_TO')
        #Don't forget to couple the propulsion system to the weights module like so:
        self.connect('propmodel.propellers_weight','W_propeller')
        self.connect('propmodel.eng1.component_weight','W_engine')
        self.connect('propmodel.gen1.component_weight','W_generator')
        self.connect('propmodel.motors_weight','W_motors')


        #==Finally, we need to compute certain quantities to ensure the airplane is feasible. Compute whether enough fuel volume exists, and whether the airplane burned more fuel than it can carry
        missionmargins = self.add_subsystem('missionmargins',ComputeDesignMissionResidualsBattery(include_takeoff=True),promotes_inputs=['ac|weights|MTOW','ac|weights|W_battery',"mission|*","ac|weights|W_fuel_max"],promotes_outputs=['fuel_burn','battery_energy_used'])
        self.connect('OEW.OEW','missionmargins.OEW')
        self.connect('mission.mission|total_fuel','mission|total_fuel')
        self.connect('mission.mission|total_battery_energy','mission|total_battery_energy')
        self.connect('takeoff.takeoff|total_fuel','missionmargins.takeoff|total_fuel')
        self.connect('takeoff.takeoff|total_battery_energy','missionmargins.takeoff|total_battery_energy')
        self.connect('propmodel.batt1.max_energy','missionmargins.battery_max_energy')

        #==Calculate the difference between the one-engine-out abort distance and one-engine-out takeoff distance with obstacle clearance
        bflmargins = self.add_subsystem('bflmargins',ComputeBalancedFieldLengthResidual(),promotes_inputs=['mission|takeoff|v*'])
        self.connect('takeoff.takeoff|distance','bflmargins.takeoff|distance')
        self.connect('takeoff.takeoff|distance_abort','bflmargins.takeoff|distance_abort')

        implicit_solve = self.add_subsystem('implicit_solve',SolveImplicitStates(n_int_per_seg=n_int_per_seg))
        self.connect('mission.thrust_resid.thrust_residual','implicit_solve.thrust_residual')
        self.connect('bflmargins.BFL_combined','implicit_solve.BFL_residual')
        self.connect('propmodel.eng_gen_resid.eng_gen_residual','implicit_solve.eng_gen_residual')

        self.connect('implicit_solve.mission|takeoff|v1','mission|takeoff|v1')
        self.connect('implicit_solve.eng_throttle','propmodel.eng1.throttle')
        self.connect('implicit_solve.motor_throttle',['throttle_combiner.motor1|throttle|mission',
                                                'throttle_combiner.motor2|throttle|mission'])

        components_list = ['eng1','motor1','motor2','gen1']
        opcost = self.add_subsystem('operating_cost',OperatingCost(n_components=len(components_list),n_batteries=1))
        self.connect('propmodel.batt1.component_cost','operating_cost.battery_1_NR_cost')
        self.connect('battery_energy_used','operating_cost.battery_1_energy_used')

        self.connect('OEW.OEW','operating_cost.OEW')
        self.connect('fuel_burn','operating_cost.fuel_burn')
        for i, component in enumerate(components_list):
            self.connect('propmodel.'+component+'.component_weight','operating_cost.component_'+str(i+1)+'_weight')
            self.connect('propmodel.'+component+'.component_cost','operating_cost.component_'+str(i+1)+'_NR_cost')

        dummy_range = self.add_subsystem('dummypayload',DummyPayload(),promotes_outputs=['payload_objective'])
        self.connect('mission|payload','dummypayload.payload_DV')
        dummy_obj = self.add_subsystem('dummyobj',AugmentedFBObjective(),promotes_inputs=['ac|weights|MTOW','fuel_burn'],promotes_outputs=['mixed_objective'])


class SolveImplicitStates(ImplicitComponent):
    def initialize(self):
        self.options.declare('n_int_per_seg',default=5,desc="Number of Simpson intervals to use per seg (eg. climb, cruise, descend). Number of analysis points is 2N+1")

    def setup(self):
        n_int_per_seg = self.options['n_int_per_seg']
        nn = (n_int_per_seg*2 + 1)
        nn_tot = 6*nn+2
        self.add_input('eng_gen_residual', units='kW', shape=(nn_tot,))
        self.add_input('thrust_residual', units='N',shape=(3 * nn,))
        self.add_input('BFL_residual', units='m')

        #self.add_input('mission_MTOW_margin', units='kg', shape=(n_seg*nn,))
        self.add_output('eng_throttle',shape=(nn_tot,))
        self.add_output('motor_throttle',shape=(3 * nn,))
        self.add_output('mission|takeoff|v1', units='m/s', val=39)

        self.declare_partials(['eng_throttle'], ['eng_gen_residual'], val=sp.eye(nn_tot))
        self.declare_partials(['motor_throttle'], ['thrust_residual'], val=sp.eye(3 * nn))
        self.declare_partials(['mission|takeoff|v1'], ['BFL_residual'], val=1)

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['eng_throttle'] = inputs['eng_gen_residual']
        residuals['motor_throttle'] = inputs['thrust_residual']
        residuals['mission|takeoff|v1'] = inputs['BFL_residual']


def define_analysis(n_int_per_seg,specific_energy):
    """
    This function sets up the problem with all DVs and constraints necessary to perform analysis only (drives throttle residuals and BFL residuals to zero).
    This does NOT ensure that the airplane has enough fuel capacity or gross weight to fly the mission.
    """
    prob = Problem()
    prob.model= TotalAnalysis(n_int_per_seg=n_int_per_seg,specific_energy=specific_energy)
    nn = n_int_per_seg*2+1
    nn_tot_m = 3*(n_int_per_seg*2+1)
    nn_tot_to = 3*(n_int_per_seg*2+1)+2
    nn_tot = 6*(n_int_per_seg*2+1)+2

    prob.model.options['assembled_jac_type'] = 'csc'
    prob.model.nonlinear_solver=NewtonSolver()
    prob.model.linear_solver = DirectSolver(assemble_jac=True)
    prob.model.nonlinear_solver.options['solve_subsystems'] = True
    prob.model.nonlinear_solver.options['maxiter'] = 10
    prob.model.nonlinear_solver.options['atol'] = 1e-7
    prob.model.nonlinear_solver.options['rtol'] = 1e-7

    prob.driver = ScipyOptimizeDriver()
    prob.driver.options['tol'] = 1e-7
    return prob, nn_tot, nn_tot_m, nn_tot_to


if __name__ == "__main__":
    design_ranges = [350,400,450,500,550,600,650,700]
    specific_energies = [250,300,350,400,450,500,550,600,650,700,750,800]
	#redo spec range 450, spec energy 700, 750, 800
    for design_range in design_ranges:
        for spec_energy in specific_energies:
            n_int_per_seg = 3
            prob, nn_tot, nn_tot_m, nn_tot_to = define_analysis(n_int_per_seg,spec_energy)
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

                prob.model.add_constraint('missionmargins.mission|MTOW_margin',equals=0.0)
                prob.model.add_constraint('missionmargins.mission|fuel_capacity_margin',lower=0.0)
                prob.model.add_constraint('takeoff.takeoff|distance',upper=1357)
                prob.model.add_constraint('vstall.Vstall_eas',upper=42.0)
                prob.model.add_constraint('missionmargins.mission|battery_margin',lower=0.0)
                prob.model.add_constraint('propmodel.eng1.component_sizing_margin',upper=1.0*np.ones(nn_tot))
                prob.model.add_constraint('propmodel.gen1.component_sizing_margin',upper=1.0*np.ones(nn_tot))
                prob.model.add_constraint('propmodel.batt1.component_sizing_margin',upper=1.0*np.ones(nn_tot))
                prob.model.add_constraint('takeoff.takeoff|climb|gamma',lower=0.009)
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
                prob.model.add_constraint('implicit_solve.motor_throttle',upper=1.05*np.ones(nn_tot_m))
                prob.model.add_objective('fuel_burn')



            else:
                print('======Analyzing Fuel Burn for Given Mision============')
                prob.model.add_objective('mixed_objective')

            prob.setup(mode='fwd')
            prob['implicit_solve.mission|takeoff|v1'] = 40
            prob['implicit_solve.motor_throttle'] = np.ones(nn_tot_m)*0.5
            prob['implicit_solve.eng_throttle'] = np.ones(nn_tot)*0.5
            prob['OEW.const.structural_fudge'] = 2.0
            prob['mission|range'] = design_range
            prob['ac|propulsion|propeller|diameter'] = 2.2
            prob['ac|propulsion|engine|rating'] = 1117.2


            #prob.set_val('mission|range',1000,'NM')
            prob.run_driver()
            takeoff_check(prob)
            prob.cleanup()
            #prob.check_partials(compact_print=True)
            print('Design range: '+str(prob.get_val('mission|range', units='NM')))
            print('MTOW: '+str(prob.get_val('ac|weights|MTOW', units='lb')))
            print('OEW: '+str(prob.get_val('OEW.OEW', units='lb')))
            print('Battery wt: '+str(prob.get_val('ac|weights|W_battery', units='lb')))
            print('Fuel cap:'+str(prob.get_val('ac|weights|W_fuel_max', units='lb')))
            print('MTOW margin: '+str(prob.get_val('missionmargins.mission|MTOW_margin', units='lb')))
            print('Battery margin: '+str(prob.get_val('missionmargins.mission|battery_margin', units='J')))

            print('Eng power:'+str(prob.get_val('ac|propulsion|engine|rating', units='hp')))
            print('Gen power:'+str(prob.get_val('ac|propulsion|generator|rating', units='hp')))
            print('Motor power:'+str(prob.get_val('ac|propulsion|motor|rating', units='hp')))
            print('Hybrid split|'+str(prob.get_val('controls.hybrid_split|percentage', units=None)))
            print('Prop diam:'+str(prob.get_val('ac|propulsion|propeller|diameter', units='m')))

            print('TO (continue):'+str(prob.get_val('takeoff.takeoff|distance', units='ft')))
            print('TO (abort):'+str(prob.get_val('takeoff.takeoff|distance_abort', units='ft')))
            print('Stall speed'+str(prob.get_val('vstall.Vstall_eas', units='kn')))
            print('Rotate speed'+str(prob.get_val('mission|takeoff|vr', units='kn')))
            print('Decision speed'+str(prob.get_val('implicit_solve.mission|takeoff|v1', units='kn')))
            print('S_ref: ' +str(prob.get_val('ac|geom|wing|S_ref', units='ft**2')))

            print('Mission Fuel burn: '+ str(prob.get_val('mission.mission|total_fuel', units='lb')))
            print('TO fuel burn: '+ str(prob.get_val('takeoff.takeoff|total_fuel', units='lb')))
            print('Total fuel burn:' +str(prob.get_val('fuel_burn', units='lb')))

            print('V0V1 dist: '+str(prob['takeoff.v0v1_dist.delta_quantity']))
            print('V1VR dist: '+str(prob['takeoff.v1vr_dist.delta_quantity']))
            print('Braking dist:'+str(prob['takeoff.v1v0_dist.delta_quantity']))
            print('Climb angle(rad):'+str(prob['takeoff.takeoff|climb|gamma']))
            print('h_trans:'+str(prob['takeoff.h_transition']))
            print('s_trans:'+str(prob['takeoff.s_transition']))
            print('s_climb|'+str(prob['takeoff.s_climb']))
            #print('Mission throttle settings:'+str(prob['implicit_solve.throttle']))s
            #print('Fuel_flows:'+str(prob['fuel_flow']))

            #prob.model.list_outputs()

            #prob.check_partials(compact_print=True)

            # print(str(prob['propmodel.eng1.component_sizing_margin']))
            # print(str(prob['propmodel.gen1.component_sizing_margin']))
