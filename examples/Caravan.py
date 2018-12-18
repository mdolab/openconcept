from __future__ import division
from openmdao.api import Problem, Group, IndepVarComp, DirectSolver, NewtonSolver
from openmdao.api import ScipyOptimizeDriver, ExplicitComponent, ImplicitComponent
# ------This is a hack for testing scripts on openconcept source directories that haven't been installed.
# By default, a script includes its path in sys.path. We need to add the folder one level higher (where the interpreter is run from)
# The script can be run from the root git directory as 'python examples/script.py' and the latest openconcept package will be imported by default
import sys, os
sys.path.insert(0,os.getcwd())

#-------These imports are generic and should be left alone
import numpy as np
import scipy.sparse as sp
from openconcept.utilities.dict_indepvarcomp import DictIndepVarComp
from openconcept.analysis.takeoff import BalancedFieldLengthTakeoff, takeoff_check
from openconcept.analysis.mission import MissionAnalysis
from openconcept.utilities.nodes import compute_num_nodes
# These imports are particular to this airplane
# If run from the root git  directory as 'python examples/script.py', these imports are found because the script's path is added to sys.path by default.
from examples.propulsion_layouts.simple_turboprop import TurbopropPropulsionSystem
from examples.aircraft_data.caravan import data as acdata
from examples.aircraft_data.caravan_mission import data as missiondata
from examples.methods.weights_turboprop import SingleTurboPropEmptyWeight
from examples.methods.costs_commuter import OperatingCost

class TotalAnalysis(Group):
    """This analysis group calculates TOFL and mission fuel burn as well as many other quantities for an example airplane. Elements may be overridden or replaced as needed.
        Should be instantiated as the top-level model
    """

    def initialize(self):
        self.options.declare('n_int_per_seg',default=5,desc="Number of Simpson intervals to use per seg (eg. climb, cruise, descend). Number of analysis points is 2N+1")

    def setup(self):
        n_int_per_seg = self.options['n_int_per_seg']
        mission_segments=['climb','cruise','descent']
        nn, nn_tot_to, nn_tot_m, nn_tot = compute_num_nodes(n_int_per_seg, mission_segments)

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

        dv_comp.add_output_from_dict('ac|propulsion|engine|rating')
        dv_comp.add_output_from_dict('ac|propulsion|propeller|diameter')

        dv_comp.add_output_from_dict('ac|num_passengers_max')
        dv_comp.add_output_from_dict('ac|q_cruise')

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
        #set the prop to 2000 rpm for all time
        controls.add_output('prop|rpm|takeoff', val=np.ones(nn_tot_to) * 2000, units='rpm')
        controls.add_output('prop|rpm|mission', val=np.ones(nn_tot_m) * 2000, units='rpm')

        TO_throttle_vec = np.concatenate([np.ones(nn),
                                          np.ones(nn) * 1.0,
                                          np.zeros(nn),
                                          np.ones(2) * 1.0])
        controls.add_output('motor|throttle|takeoff', val=TO_throttle_vec)

        self.add_subsystem('takeoff',BalancedFieldLengthTakeoff(n_int_per_seg=n_int_per_seg,
                                                                track_fuel=True, track_battery=False,
                                                                propulsion_system=TurbopropPropulsionSystem(num_nodes=nn_tot_to)),
                                                                promotes_inputs=['ac|aero*','ac|geom|*','ac|propulsion|*','ac|weights|*','takeoff|h'])
        self.connect('controls.prop|rpm|takeoff','takeoff.propmodel.prop1.rpm')
        self.connect('controls.motor|throttle|takeoff', 'takeoff.propmodel.throttle')

        #==This module computes fuel consumption during the entire mission
        mission_promote_inputs = ["ac|aero|*","ac|geom|*",'ac|propulsion|*','ac|weights|*','OEW','*|Ueas','*|h0','*|hf','*|time','design_range','payload']
        self.add_subsystem('design_mission',MissionAnalysis(n_int_per_seg=n_int_per_seg,track_battery=False,
                           propulsion_system=TurbopropPropulsionSystem(num_nodes=nn_tot_m)),
                           promotes_inputs=mission_promote_inputs)
        self.connect('design_mission.throttle','design_mission.propmodel.throttle')
        self.connect('takeoff.weight_after_takeoff','design_mission.weight_initial')
        self.connect('takeoff.total_fuel','design_mission.residuals.takeoff|total_fuel')
        self.connect('controls.prop|rpm|mission','design_mission.propmodel.prop1.rpm')

        #==This module is an empirical weight tool specific to a single-engine turboprop airplane. You will need to modify or replace it.
        self.add_subsystem('OEW',SingleTurboPropEmptyWeight(),promotes_inputs=["*"], promotes_outputs=['OEW'])
        self.connect('ac|propulsion|engine|rating','P_TO')
        #Don't forget to couple the propulsion system to the weights module like so:
        self.connect('design_mission.propmodel.prop1.component_weight','W_propeller')
        self.connect('design_mission.propmodel.eng1.component_weight','W_engine')

        components_list = ['eng1']
        opcost = self.add_subsystem('operating_cost',OperatingCost(n_components=len(components_list),n_batteries=None))
        self.connect('OEW','operating_cost.OEW')
        self.connect('design_mission.fuel_burn','operating_cost.fuel_burn')
        for i, component in enumerate(components_list):
            self.connect('design_mission.propmodel.'+component+'.component_weight','operating_cost.component_'+str(i+1)+'_weight')
            self.connect('design_mission.propmodel.'+component+'.component_cost','operating_cost.component_'+str(i+1)+'_NR_cost')

def define_analysis(n_int_per_seg):
    """
    This function sets up the problem with all DVs and constraints necessary to perform analysis only (drives throttle residuals and BFL residuals to zero).
    This does NOT ensure that the airplane has enough fuel capacity or gross weight to fly the mission.
    """
    prob = Problem()
    prob.model= TotalAnalysis(n_int_per_seg=n_int_per_seg)

    prob.model.options['assembled_jac_type'] = 'csc'
    prob.model.nonlinear_solver=NewtonSolver()
    prob.model.linear_solver = DirectSolver(assemble_jac=True)
    prob.model.nonlinear_solver.options['solve_subsystems'] = True
    prob.model.nonlinear_solver.options['maxiter'] = 10
    prob.model.nonlinear_solver.options['atol'] = 1e-6
    prob.model.nonlinear_solver.options['rtol'] = 1e-6


    prob.driver = ScipyOptimizeDriver()
    return prob


if __name__ == "__main__":
    n_int_per_seg = 5
    prob = define_analysis(n_int_per_seg)
    mission_segments=['climb','cruise','descent']
    nn, nn_tot_to, nn_tot_m, nn_tot = compute_num_nodes(n_int_per_seg, mission_segments)
    run_type = 'analysis'
    if run_type == 'optimization':
        print('======Performing Multidisciplinary Design Optimization===========')
        prob.model.add_design_var('ac|weights|MTOW', lower=2500, upper=3500)
        prob.model.add_design_var('ac|geom|wing|S_ref',lower=9,upper=30)
        prob.model.add_design_var('ac|propulsion|engine|rating',lower=500,upper=1400)
        prob.model.add_design_var('ac|weights|W_fuel_max',lower=800,upper=3000)

        prob.model.add_constraint('design_mission.residuals.MTOW_margin',equals=0.0)
        prob.model.add_constraint('design_mission.residuals.fuel_capacity_margin',equals=0.0)
        prob.model.add_constraint('takeoff.distance_continue',upper=807)
        prob.model.add_constraint('takeoff.vstall.Vstall_eas',upper=41.88)

        # prob.model.add_design_var('mission_eas_climb',lower=85,upper=300)
        # prob.model.add_design_var('mission_eas_cruise',lower=150,upper=300)
    elif run_type == 'max_range':
        print('======Analyzing Design Range at Given MTOW===========')
        prob.model.add_design_var('design_range',lower=1000,upper=2500)
        prob.model.add_constraint('design_mission.residuals.MTOW_margin',equals=0.0)
    else:
        print('======Analyzing Fuel Burn for Given Mision============')

    prob.model.add_objective('design_mission.fuel_burn')
    prob.setup(mode='fwd')
    prob['takeoff.v1_solve.takeoff|v1'] = 30
    prob['design_mission.throttle'] = np.ones(nn_tot_m)*0.7
    prob['OEW.const.structural_fudge'] = 1.4

    prob.run_model()
    takeoff_check(prob)
    #prob.check_partials(compact_print=True)
    print('Design range: '+str(prob.get_val('design_range', units='NM')))
    print('MTOW: '+str(prob.get_val('ac|weights|MTOW', units='lb')))
    print('OEW: '+str(prob.get_val('OEW', units='lb')))
    print('Fuel cap:'+str(prob.get_val('ac|weights|W_fuel_max', units='lb')))
    print('MTOW margin: '+str(prob.get_val('design_mission.residuals.MTOW_margin', units='lb')))

    print('Eng power:'+str(prob.get_val('ac|propulsion|engine|rating', units='hp')))
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

    #prob.model.list_outputs()

    #prob.check_partials(compact_print=True)
