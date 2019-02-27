from __future__ import division
import sys, os
sys.path.insert(0,os.getcwd())
import numpy as np
from openmdao.api import Problem, Group, ScipyOptimizeDriver, DirectSolver, SqliteRecorder,IndepVarComp,BalanceComp,NewtonSolver,BoundsEnforceLS,NonlinearBlockGS
from openconcept.analysis.aerodynamics import PolarDrag
from openconcept.utilities.math import AddSubtractComp
from openconcept.utilities.math.integrals import Integrator
from openconcept.utilities.dvlabel import DVLabel
from openconcept.analysis.solver_based.to_mission_analysis import ODEGroundRoll, ODERotate, ODESteady, BFLImplicitSolve

from examples.methods.weights_turboprop import SingleTurboPropEmptyWeight
from examples.propulsion_layouts.simple_all_electric import AllElectricPropulsionSystem, AllElectricPropulsionSystemCompressible
from examples.methods.costs_commuter import OperatingCost

from openconcept.utilities.dict_indepvarcomp import DictIndepVarComp
from examples.aircraft_data.TBM850 import data as acdata
from examples.aircraft_data.TBM850_mission import data as missiondata

class TBM850Model(Group):
    """
    A custom model specific to the TBM 850 airplane
    """
    def initialize(self):
        self.options.declare('num_nodes',default=1)
        self.options.declare('flight_phase',default=None)

    def setup(self):
        nn = self.options['num_nodes']
        flight_phase = self.options['flight_phase']
        controls = self.add_subsystem('controls',IndepVarComp(),promotes_outputs=['*'])
        controls.add_output('prop1rpm',val=np.ones((nn,))*2000,units='rpm')
        controls.add_output('fuel_used',val=np.zeros((nn,)),units='kg')

        propulsion_promotes_outputs = ['thrust']
        propulsion_promotes_inputs = ["fltcond|*","ac|propulsion|*","throttle","ac|weights|*","duration"]

        self.add_subsystem('propmodel',AllElectricPropulsionSystemCompressible(num_nodes=nn),
                           promotes_inputs=propulsion_promotes_inputs,promotes_outputs=propulsion_promotes_outputs)
        self.connect('prop1rpm','propmodel.prop1.rpm')

        if flight_phase != 'v0v1' and flight_phase != 'v1vr' and flight_phase != 'rotate':
            self.add_subsystem('drag',PolarDrag(num_nodes=nn),promotes_inputs=['fltcond|CL','ac|geom|*',('CD0','ac|aero|polar|CD0_cruise'),'fltcond|q',('e','ac|aero|polar|e')],promotes_outputs=['drag'])
        else:
            self.add_subsystem('drag',PolarDrag(num_nodes=nn),promotes_inputs=['fltcond|CL','ac|geom|*',('CD0','ac|aero|polar|CD0_TO'),'fltcond|q',('e','ac|aero|polar|e')],promotes_outputs=['drag'])

        # self.add_subsystem('OEW',SingleTurboPropEmptyWeight(),promotes_inputs=['*',('P_TO','ac|propulsion|motor|rating')], promotes_outputs=['OEW'])
        # self.connect('propmodel.prop1.component_weight','W_propeller')
        # self.connect('propmodel.eng1.component_weight','W_engine')

        self.add_subsystem('weight',AddSubtractComp(output_name='weight',input_names=['ac|weights|MTOW','fuel_used'],units='kg',vec_size=[1,nn],scaling_factors=[1,-1]),promotes_inputs=['*'],promotes_outputs=['weight'])

class TotalAnalysis(Group):
    """This is an example of a balanced field takeoff and three-phase mission analysis.
    """

    def initialize(self):
        self.options.declare('num_nodes',default=9,desc="Number of points per segment. Needs to be 2N + 1 due to simpson's rule")

    def setup(self):
        nn = self.options['num_nodes']

        dv_comp = self.add_subsystem('dv_comp',DictIndepVarComp(acdata,seperator='|'),promotes_outputs=["*"])
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
        dv_comp.add_output('ac|propulsion|motor|rating', val=850, units='hp')
        dv_comp.add_output('ac|weights|W_battery', val=2000, units='lb')

        mission_data_comp = self.add_subsystem('mission_data_comp',DictIndepVarComp(missiondata),promotes_outputs=["*"])
        mission_data_comp.add_output('takeoff|h',val=0,units='ft')
        mission_data_comp.add_output('cruise|h0',val=6000, units='m')
        mission_data_comp.add_output('design_range',val=190,units='NM')
        mission_data_comp.add_output('T_motor_initial', val=15, units='degC')
        mission_data_comp.add_output('T_res_initial', val=15.1, units='degC')
        # add the four balanced field length takeoff segments and the implicit v1 solver
        # v0v1 - from a rolling start to v1 speed
        # v1vr - from the decision speed to rotation
        # rotate - in the air following rotation in 2DOF
        # v1vr - emergency stopping from v1 to a stop.

        self.add_subsystem('bfl',BFLImplicitSolve(),promotes_outputs=['takeoff|v1'])
        self.add_subsystem('v0v1',ODEGroundRoll(num_nodes=nn, aircraft_model=TBM850Model, flight_phase='v0v1'),promotes_inputs=['ac|*','takeoff|v1'])
        self.add_subsystem('v1vr',ODEGroundRoll(num_nodes=nn, aircraft_model=TBM850Model, flight_phase='v1vr'),promotes_inputs=['ac|*'])
        self.connect('takeoff|v1','v1vr.fltcond|Utrue_initial')
        self.connect('v0v1.range_final','v1vr.range_initial')
        self.connect('v0v1.propmodel.batt1.SOC_final','v1vr.propmodel.batt1.SOC_initial')
        self.add_subsystem('rotate',ODERotate(num_nodes=nn, aircraft_model=TBM850Model, flight_phase='rotate'),promotes_inputs=['ac|*'])
        self.connect('v1vr.range_final','rotate.range_initial')
        self.connect('v1vr.fltcond|Utrue_final','rotate.fltcond|Utrue_initial')
        self.connect('v1vr.propmodel.batt1.SOC_final','rotate.propmodel.batt1.SOC_initial')
        self.connect('rotate.range_final','bfl.distance_continue')
        self.connect('v1vr.takeoff|vr','bfl.takeoff|vr')
        self.add_subsystem('v1v0',ODEGroundRoll(num_nodes=nn, aircraft_model=TBM850Model, flight_phase='v1v0'), promotes_inputs=['ac|*','takeoff|v1'])
        self.connect('v0v1.range_final','v1v0.range_initial')
        self.connect('v1v0.range_final','bfl.distance_abort')

        # add the climb, cruise, and descent segments
        self.add_subsystem('climb',ODESteady(num_nodes=nn, aircraft_model=TBM850Model, flight_phase='climb'),promotes_inputs=['ac|*'])
        # set the climb time such that the specified initial cruise altitude is exactly reached
        self.add_subsystem('climbdt',BalanceComp(name='duration',units='s',eq_units='m',val=120,lower=0,rhs_name='cruise|h0',lhs_name='fltcond|h_final'),promotes_inputs=['cruise|h0'])
        self.connect('climb.fltcond|h_final','climbdt.fltcond|h_final')
        self.connect('climbdt.duration','climb.duration')
        self.connect('T_motor_initial','v0v1.propmodel.motorheatsink.T_initial')
        self.connect('T_res_initial','v0v1.propmodel.reservoir.T_initial')

        self.add_subsystem('cruise',ODESteady(num_nodes=nn, aircraft_model=TBM850Model, flight_phase='cruise'),promotes_inputs=['ac|*'])
        # set the cruise time such that the desired design range is flown by the end of the mission
        self.add_subsystem('cruisedt',BalanceComp(name='duration',units='s',eq_units='m',val=120, lower=0,rhs_name='design_range',lhs_name='range_final'),promotes_inputs=['design_range'])
        self.connect('cruisedt.duration','cruise.duration')

        self.add_subsystem('descent',ODESteady(num_nodes=nn, aircraft_model=TBM850Model, flight_phase='descent'),promotes_inputs=['ac|*'])
        # set the descent time so that the final altitude is sea level again
        self.add_subsystem('descentdt',BalanceComp(name='duration',units='s',eq_units='m', val=120, lower=0,rhs_name='takeoff|h',lhs_name='fltcond|h_final'),promotes_inputs=['takeoff|h'])
        self.connect('descent.range_final','cruisedt.range_final')
        self.connect('descent.fltcond|h_final','descentdt.fltcond|h_final')
        self.connect('descentdt.duration','descent.duration')

        # connect range, fuel burn, and altitude from the end of each segment to the beginning of the next, in order
        connect_from = ['v0v1','v1vr','rotate','climb','cruise']
        connect_to = ['v1vr','rotate','climb','cruise','descent']
        for i, from_phase in enumerate(connect_from):
            for state in ['propmodel.motorheatsink.T','propmodel.reservoir.T']:
                to_phase = connect_to[i]
                self.connect(from_phase+'.'+state+'_final',to_phase+'.'+state+'_initial')

        connect_from = ['rotate','climb','cruise']
        connect_to = ['climb','cruise','descent']
        for i, from_phase in enumerate(connect_from):
            for state in ['range','propmodel.batt1.SOC','fltcond|h']:
                to_phase = connect_to[i]
                self.connect(from_phase+'.'+state+'_final',to_phase+'.'+state+'_initial')

if __name__ == "__main__":
    num_nodes = 9
    prob = Problem()
    prob.model= TotalAnalysis(num_nodes=num_nodes)


    prob.model.nonlinear_solver=NewtonSolver(iprint=2)
    prob.model.options['assembled_jac_type'] = 'csc'
    prob.model.linear_solver = DirectSolver(assemble_jac=True)
    prob.model.nonlinear_solver.options['solve_subsystems'] = True
    prob.model.nonlinear_solver.options['maxiter'] = 20
    prob.model.nonlinear_solver.options['atol'] = 1e-8
    prob.model.nonlinear_solver.options['rtol'] = 1e-8
    prob.model.nonlinear_solver.linesearch = BoundsEnforceLS(bound_enforcement='scalar',print_bound_enforce=False)

    # prob.model.add_design_var('climb.fltcond|Ueas', lower=50*np.ones((num_nodes,)), upper=120*np.ones((num_nodes,)))
    # prob.model.add_design_var('descent.fltcond|Ueas', lower=50*np.ones((num_nodes,)), upper=120*np.ones((num_nodes,)))
    # prob.model.add_design_var('climb.fltcond|vs', lower=0.5*np.ones((num_nodes,)), upper=15*np.ones((num_nodes,)))
    # prob.model.add_design_var('descent.fltcond|vs', lower=-15*np.ones((num_nodes,)), upper=-0.5*np.ones((num_nodes,)))
    prob.model.add_design_var('cruise|h0', lower=3000., upper=9000., scaler=1e-3)
    prob.model.add_design_var('design_range',lower=100,upper=300,scaler=1e-2)
    prob.model.add_constraint('descent.propmodel.batt1.SOC_final',lower=0.0)
    # prob.model.add_constraint('descent.throttle',lower=0.1*np.ones((num_nodes,)),upper=np.ones((num_nodes,)))
    # prob.model.add_constraint('cruise.fltcond|h',upper=9735*np.ones((num_nodes,)))
    # prob.model.add_constraint('cruise.throttle',upper=np.ones((num_nodes,)))

    prob.model.add_objective('design_range',scaler=-1.0)
    prob.driver = ScipyOptimizeDriver()
    prob.driver.options['dynamic_simul_derivs'] = False
    #prob.driver.options['tol'] = 1e-13

    prob.setup(check=True,mode='fwd')
    # set some (optional) guesses for takeoff speeds and (required) mission parameters
    prob.set_val('v0v1.fltcond|Utrue',np.ones((num_nodes))*50,units='kn')
    prob.set_val('v1vr.fltcond|Utrue',np.ones((num_nodes))*85,units='kn')
    prob.set_val('v1v0.fltcond|Utrue',np.ones((num_nodes))*85,units='kn')
    prob.set_val('rotate.fltcond|Utrue',np.ones((num_nodes))*80,units='kn')
    prob.set_val('rotate.accel_vert',np.ones((num_nodes))*0.1,units='m/s**2')
    prob.set_val('climb.fltcond|vs', np.ones((num_nodes,))*1000, units='ft/min')
    prob.set_val('climb.fltcond|Ueas', np.ones((num_nodes,))*140, units='kn')
    prob.set_val('cruise.fltcond|vs', np.ones((num_nodes,))*0.01, units='ft/min')
    prob.set_val('cruise.fltcond|Ueas', np.ones((num_nodes,))*140, units='kn')
    prob.set_val('descent.fltcond|vs', np.ones((num_nodes,))*(-600), units='ft/min')
    prob.set_val('descent.fltcond|Ueas', np.ones((num_nodes,))*140, units='kn')

    prob.run_model()

    # prob.model.climb.list_inputs(print_arrays=True,units=True)
    #prob.model.v0v1.list_outputs(print_arrays=True,units=True)
    #prob.check_partials(compact_print=True)

    # list some outputs
    units=['lb',None]
    for i, thing in enumerate(['ac|weights|MTOW','descent.propmodel.batt1.SOC_final']):
        if units[i] is not None:
            print(thing+' '+str(prob.get_val(thing,units=units[i])[0])+' '+units[i])
        else:
            print(thing+' '+str(prob.get_val(thing,units=units[i])[0]))

    # plot some stuff
    plots = True
    save_file = False
    load_file = True
    file_base = 'compressible'

    if plots:
        from matplotlib import pyplot as plt
        # x_variable = 'range'
        # x_units='ft'
        # y_variables = ['fltcond|Ueas','fltcond|h']
        # y_units = ['kn','ft']
        # phases_to_plot = ['v0v1','v1vr','rotate','v1v0']
        # val_list=[]
        # for phase in phases_to_plot:
        #     val_list.append(prob.get_val(phase+'.'+x_variable,units=x_units))
        # x_vec = np.concatenate(val_list)

        # for i, y_var in enumerate(y_variables):
        #     val_list = []
        #     for phase in phases_to_plot:
        #         val_list.append(prob.get_val(phase+'.'+y_var,units=y_units[i]))
        #     y_vec = np.concatenate(val_list)
        #     plt.figure()
        #     plt.plot(x_vec, y_vec,'o')
        #     plt.xlabel(x_variable)
        #     plt.ylabel(y_var)
        #     plt.title('takeoff / rejected takeoff')
        # plt.show()

        phases_to_plot = ['v0v1','v1vr','rotate','climb','cruise','descent']
        x_variable = 'range'
        x_units='NM'
        y_variables = ['fltcond|h','fltcond|Ueas','throttle','fltcond|vs','propmodel.batt1.SOC','propmodel.motorheatsink.T','propmodel.reservoir.T_out','propmodel.duct.mdot']
        y_units = ['ft','kn',None,'ft/min',None,'degC','degC','kg/s']

        val_list= []
        for phase in phases_to_plot:
            val_list.append(prob.get_val(phase+'.'+x_variable,units=x_units))
        x_vec = np.concatenate(val_list)
        if save_file:
            np.save(file_base+'_x',x_vec)

        for i, y_var in enumerate(y_variables):
            val_list = []
            for phase in phases_to_plot:
                val_list.append(prob.get_val(phase+'.'+y_var,units=y_units[i]))
            y_vec = np.concatenate(val_list)
            if save_file:
                filename = file_base+'_'+y_var.replace("|","")
                np.save(filename,y_vec)
            plt.figure()
            if load_file:
                x_loaded = np.load(file_base+'_x'+'.npy')
                y_loaded = np.load(file_base+'_'+y_var.replace("|","")+'.npy')
                plt.plot(x_vec,y_vec,x_loaded,y_loaded)
            else:
                plt.plot(x_vec, y_vec)
            plt.xlabel(x_variable)
            plt.ylabel(y_var)
            plt.title('mission profile')
        plt.show()
