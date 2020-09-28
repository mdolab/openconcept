from __future__ import division
import sys
import os
import numpy as np

sys.path.insert(0, os.getcwd())
import openmdao.api as om
import openconcept.api as oc
# imports for the airplane model itself
from openconcept.analysis.aerodynamics import PolarDrag
from examples.aircraft_data.HybridSingleAisle import data as acdata
from openconcept.analysis.performance.mission_profiles import MissionWithReserve, BasicMission
from openconcept.components.N3opt import N3Hybrid
from openconcept.components.motor import SimpleMotor
from openconcept.components.battery import SOCBattery
from openconcept.utilities.math.add_subtract_comp import AddSubtractComp
from openconcept.utilities.math.multiply_divide_comp import ElementMultiplyDivideComp
from openconcept.components.thermal import LiquidCooledComp
from openconcept.components.splitter import FlowSplit, FlowCombine
from openconcept.components.heat_sinks import LiquidCooledMotor
from openconcept.components.heat_sinks import LiquidCooledBattery
from openconcept.components.ducts import ImplicitCompressibleDuct_ExternalHX, ExplicitIncompressibleDuct
from openconcept.components.heat_exchanger import HXGroup

# TODO run an engine sweep at positive net shaft power offtake 

class HybridSingleAisleModel(oc.IntegratorGroup):
    """
    Model for NASA twin hybrid single aisle study

    """
    def initialize(self):
        self.options.declare('num_nodes', default=1)
        self.options.declare('flight_phase', default=None)

    def setup(self):
        nn = self.options['num_nodes']
        flight_phase = self.options['flight_phase']


        # engine model is surrogate modeled based on N+3 pycycle runs
        self.add_subsystem('engine', N3Hybrid(num_nodes=nn, plot=False),
                           promotes_inputs=["fltcond|*", "throttle"])

        doubler = ElementMultiplyDivideComp()
        doubler.add_equation(output_name='thrust', input_names=['thrust_in'], 
                             vec_size=nn, scaling_factor=2.0, input_units=['kN'])
        doubler.add_equation(output_name='fuel_flow', input_names=['fuel_flow_in'], 
                             vec_size=nn, scaling_factor=2.0, input_units=['kg/s'], 
                             tags=['integrate', 'state_name:fuel_used', 'state_units:kg', 'state_val:1.0', 'state_promotes:True']) 
        
        self.add_subsystem('doubler', doubler, promotes_outputs=['*'])
        self.connect('engine.thrust', 'doubler.thrust_in')
        self.connect('engine.fuel_flow', 'doubler.fuel_flow_in')

        # Hybrid propulsion motor (model one side only, then double the weight)
        self.add_subsystem('hybrid_motor', SimpleMotor(num_nodes=nn, efficiency=0.97), 
                           promotes_inputs=[('elec_power_rating','ac|propulsion|motor|rating')])
        self.connect('hybrid_motor.shaft_power_out', 'engine.hybrid_power')

        # Hybrid propulsion battery (model one side only, then double the weight)
        self.add_subsystem('battery', SOCBattery(num_nodes=nn, efficiency=0.95, specific_energy=400), 
                           promotes_inputs=[('battery_weight','ac|propulsion|battery|weight')])
        self.connect('hybrid_motor.elec_load', 'battery.elec_load')

        iv = self.add_subsystem('iv',om.IndepVarComp(), promotes_outputs=['*'])
        iv.add_output('mdot_coolant', val=6.0*np.ones((nn,)), units='kg/s')
        iv.add_output('rho_coolant', val=997*np.ones((nn,)),units='kg/m**3')
        iv.add_output('area_nozzle', val=58*np.ones((nn,)), units='inch**2')
        lc_promotes = [('power_rating','ac|propulsion|motor|rating')]
        self.add_subsystem('motorheatsink',
                           LiquidCooledMotor(num_nodes=nn,
                                            quasi_steady=False),
                                            promotes_inputs=lc_promotes)
        self.connect('hybrid_motor.heat_out','motorheatsink.q_in')
        self.connect('hybrid_motor.component_weight','motorheatsink.motor_weight')
        
        bc_promotes = [('battery_weight','ac|propulsion|battery|weight')]
        self.add_subsystem('batteryheatsink',
                           LiquidCooledBattery(num_nodes=nn,
                                               quasi_steady=False),
                                               promotes_inputs=bc_promotes)
        self.connect('battery.heat_out', 'batteryheatsink.q_in')

        self.add_subsystem('duct',
                           ExplicitIncompressibleDuct(num_nodes=nn),
                           promotes_inputs=['fltcond|*'])

        self.add_subsystem('hx',HXGroup(num_nodes=nn),promotes_inputs=['ac|propulsion|thermal|hx|n_wide_cold'])
        # self.connect('duct.mdot','hx.mdot_cold')
        self.connect('hx.delta_p_cold','duct.delta_p_hex')

        self.connect('motorheatsink.T_out','batteryheatsink.T_in')
        self.connect('batteryheatsink.T_out', 'hx.T_in_hot')
        self.connect('hx.T_out_hot','motorheatsink.T_in')
        self.connect('rho_coolant','hx.rho_hot')
        self.connect('mdot_coolant',['motorheatsink.mdot_coolant','hx.mdot_hot','batteryheatsink.mdot_coolant'])

        duct = self.add_subsystem('duct2',
                           ImplicitCompressibleDuct_ExternalHX(num_nodes=nn),
                           promotes_inputs=[('p_inf','fltcond|p'),('T_inf','fltcond|T'),('Utrue','fltcond|Utrue')])
        # in to HXGroup:
        self.connect('duct2.mdot', 'hx.mdot_cold')
        self.connect('duct2.sta2.T', 'hx.T_in_cold')
        self.connect('duct2.sta2.rho', 'hx.rho_cold')

        #out from HXGroup
        self.connect('hx.delta_p_cold', 'duct2.sta3.delta_p')
        self.connect('hx.heat_transfer', 'duct2.sta3.heat_in')
        self.connect('hx.frontal_area', ['duct2.area_2', 'duct2.area_3'])
        self.connect('area_nozzle', 'duct2.area_nozzle_in')

        # duct.nonlinear_solver=om.NewtonSolver(iprint=0)
        # duct.linear_solver = om.DirectSolver(assemble_jac=True)
        # duct.nonlinear_solver.options['solve_subsystems'] = True
        # duct.nonlinear_solver.options['maxiter'] = 20
        # duct.nonlinear_solver.options['atol'] = 1e-8
        # duct.nonlinear_solver.options['rtol'] = 1e-8
        # duct.nonlinear_solver.linesearch = om.BoundsEnforceLS(bound_enforcement='scalar',print_bound_enforce=False)
        # duct.nonlinear_solver.linesearch.options['print_bound_enforce'] = False

        # self.connect('motorheatsink.T_out','duct.T_in_hot')
        # self.connect('rho_coolant','duct.rho_hot')
        # self.connect('duct.T_out_hot','motorheatsink.T_in')
        # self.connect('mdot_coolant',['motorheatsink.mdot_coolant','duct.mdot_hot'])



        self.connect('area_nozzle','duct.area_nozzle')




        # NOTE do we ever need to take shaft power off for thermal when the battery is empty or something?
        # TODO add a refrigerator/passthru comp
        # TODO add a sum comp for electrical load (motor + refrigerator)
        # TODO connect the summed electrical load to the battery
        # TODO FUTURE handle fault protection components

        # use a different drag coefficient for takeoff versus cruise
        if flight_phase not in ['v0v1', 'v1v0', 'v1vr', 'rotate']:
            cd0_source = 'ac|aero|polar|CD0_cruise'
        else:
            cd0_source = 'ac|aero|polar|CD0_TO'
        
        self.add_subsystem('drag', PolarDrag(num_nodes=nn),
                           promotes_inputs=['fltcond|CL', 'ac|geom|*', ('CD0', cd0_source),
                                            'fltcond|q', ('e', 'ac|aero|polar|e')])
        
        adder = self.add_subsystem('adder', AddSubtractComp(), promotes_outputs=['drag'])
        adder.add_equation('drag', ['airframe_drag','hx_drag'], vec_size=1, length=nn, val=1.0,
                     units='N', scaling_factors=[1.0, -2.0])
        self.connect('drag.drag', 'adder.airframe_drag')
        self.connect('duct2.force.F_net', 'adder.hx_drag')
        # generally the weights module will be custom to each airplane
        # TODO add parameterized component weights to OEW
        # Motor, Battery, TMS, N+3 weight delta
        oewmodel = om.ExecComp('OEW=x',
                  x={'value': 1.0,
                     'units': 'kg'},
                  OEW={'value': 1.0,
                       'units': 'kg'})

        self.add_subsystem('OEW', oewmodel,
                           promotes_inputs=[('x', 'ac|weights|OEW')],
                           promotes_outputs=['OEW'])

        self.add_subsystem('weight', oc.AddSubtractComp(output_name='weight',
                                                     input_names=['ac|weights|MTOW', 'fuel_used'],
                                                     units='kg', vec_size=[1, nn],
                                                     scaling_factors=[1, -1]),
                           promotes_inputs=['*'],
                           promotes_outputs=['weight'])

class HybridSingleAisleAnalysisGroup(om.Group):
    def setup(self):
        # Define number of analysis points to run pers mission segment
        nn = 21

        # Define a bunch of design varaiables and airplane-specific parameters
        dv_comp = self.add_subsystem('dv_comp',  oc.DictIndepVarComp(acdata),
                                     promotes_outputs=["*"])
        dv_comp.add_output_from_dict('ac|aero|CLmax_TO')
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

        dv_comp.add_output_from_dict('ac|geom|nosegear|length')
        dv_comp.add_output_from_dict('ac|geom|maingear|length')

        dv_comp.add_output_from_dict('ac|weights|MTOW')
        dv_comp.add_output_from_dict('ac|weights|W_fuel_max')
        dv_comp.add_output_from_dict('ac|weights|MLW')
        dv_comp.add_output_from_dict('ac|weights|OEW')

        dv_comp.add_output_from_dict('ac|propulsion|engine|rating')
        dv_comp.add_output_from_dict('ac|propulsion|motor|rating')
        dv_comp.add_output_from_dict('ac|propulsion|battery|weight')
        dv_comp.add_output_from_dict('ac|propulsion|thermal|hx|n_wide_cold')

        dv_comp.add_output_from_dict('ac|num_passengers_max')
        dv_comp.add_output_from_dict('ac|q_cruise')

        # Run a full mission analysis including takeoff, reserve_, cruise,reserve_ and descereserve_nt
        analysis = self.add_subsystem('analysis',
                                      BasicMission(num_nodes=nn,
                                                   aircraft_model=HybridSingleAisleModel,
                                                   include_ground_roll=True),
                                      promotes_inputs=['*'], promotes_outputs=['*'])

def configure_problem():
    prob = om.Problem()
    prob.model = HybridSingleAisleAnalysisGroup()
    prob.model.nonlinear_solver = om.NewtonSolver(iprint=2,solve_subsystems=True)
    prob.model.linear_solver = om.DirectSolver()
    prob.model.nonlinear_solver.options['maxiter'] = 8
    prob.model.nonlinear_solver.options['atol'] = 1e-6
    prob.model.nonlinear_solver.options['rtol'] = 1e-6
    prob.model.nonlinear_solver.linesearch = om.BoundsEnforceLS(bound_enforcement='scalar', print_bound_enforce=False)
    prob.model.nonlinear_solver.linesearch.options['print_bound_enforce'] = True
    return prob

def set_values(prob, num_nodes):
    # set some (required) mission parameters. Each pahse needs a vertical and air-speed
    # the entire mission needs a cruise altitude and range
    prob.set_val('climb.fltcond|vs', np.linspace(2300.,  600.,num_nodes), units='ft/min')
    prob.set_val('climb.fltcond|Ueas', np.linspace(230, 220,num_nodes), units='kn')
    prob.set_val('cruise.fltcond|vs', np.ones((num_nodes,)) * 4., units='ft/min')
    prob.set_val('cruise.fltcond|Ueas', np.linspace(265, 258, num_nodes), units='kn')
    prob.set_val('descent.fltcond|vs', np.linspace(-1000, -150, num_nodes), units='ft/min')
    prob.set_val('descent.fltcond|Ueas', np.ones((num_nodes,)) * 250, units='kn')
    # prob.set_val('reserve_climb.fltcond|vs', np.linspace(3000.,  2300.,num_nodes), units='ft/min')
    # prob.set_val('reserve_climb.fltcond|Ueas', np.linspace(230, 230,num_nodes), units='kn')
    # prob.set_val('reserve_cruise.fltcond|vs', np.ones((num_nodes,)) * 4., units='ft/min')
    # prob.set_val('reserve_cruise.fltcond|Ueas', np.linspace(250, 250, num_nodes), units='kn')
    # prob.set_val('reserve_descent.fltcond|vs', np.linspace(-800, -800, num_nodes), units='ft/min')
    # prob.set_val('reserve_descent.fltcond|Ueas', np.ones((num_nodes,)) * 250, units='kn')
    # prob.set_val('loiter.fltcond|vs', np.linspace(0.0, 0.0, num_nodes), units='ft/min')
    # prob.set_val('loiter.fltcond|Ueas', np.ones((num_nodes,)) * 200, units='kn')
    prob.set_val('cruise|h0',33000.,units='ft')
    # prob.set_val('reserve|h0',15000.,units='ft')
    prob.set_val('mission_range',2050,units='NM')
    prob.set_val('takeoff|v2', 160., units='kn')
    # phases_list = ['climb', 'cruise', 'descent', 'reserve_climb', 
    #               'reserve_cruise', 'reserve_descent', 'loiter']
    phases_list = ['groundroll','climb', 'cruise', 'descent']          
    for phase in phases_list:
        prob.set_val(phase+'.hybrid_motor.throttle', 0.00)
        prob.set_val(phase+'.fltcond|TempIncrement', 20, units='degC')
        prob.set_val(phase+'.duct2.sta1.M', 0.8)
        prob.set_val(phase+'.duct2.sta2.M', 0.05)
        prob.set_val(phase+'.duct2.sta3.M', 0.05)
        prob.set_val(phase+'.duct2.nozzle.nozzle_pressure_ratio', 0.95)
    prob.set_val('groundroll.duct2.sta1.M', 0.2)
    prob.set_val('groundroll.duct2.nozzle.nozzle_pressure_ratio', 0.85)

    prob.set_val('groundroll.hybrid_motor.throttle', np.linspace(1.0, 1.0, num_nodes))
    prob.set_val('climb.hybrid_motor.throttle', np.linspace(0.5, 1.0, num_nodes))
    prob.set_val('cruise.hybrid_motor.throttle', np.linspace(1.0, 1.0, num_nodes))
    prob.set_val('groundroll.motorheatsink.T_initial', 30., 'degC')
    prob.set_val('groundroll.batteryheatsink.T_initial', 30., 'degC')
    prob.set_val('groundroll.fltcond|Utrue',np.ones((num_nodes))*50,units='kn')

def show_outputs(prob):
    # print some outputs
    vars_list = ['descent.fuel_used_final']
    units = ['lb']
    nice_print_names = ['Block fuel']
    print("=======================================================================")
    for i, thing in enumerate(vars_list):
        print(nice_print_names[i]+': '+str(prob.get_val(thing,units=units[i])[0])+' '+units[i])

    # plot some stuff
    plots = True
    if plots:
        x_var = 'range'
        x_unit = 'NM'
        y_vars = ['fltcond|h','fltcond|Ueas','fuel_used','throttle','fltcond|vs','fltcond|M','fltcond|CL', 'battery.SOC', 'motorheatsink.T', 'batteryheatsink.T', 'batteryheatsink.T_in', 'duct2.force.F_net', 'duct.drag']
        y_units = ['ft','kn','lbm',None,'ft/min', None, None, None, 'degC', 'degC','degC','lbf', 'lbf']
        x_label = 'Range (nmi)'
        y_labels = ['Altitude (ft)', 'Veas airspeed (knots)', 'Fuel used (lb)', 'Throttle setting', 'Vertical speed (ft/min)', 'Mach number', 'CL', 'Batt SOC', 'Motor Temp', 'Battery Temp', 'Battery Coolant Inflow Temp', 'Compressible Duct Drag', 'Incomp Duct Drag']
        # phases = ['climb', 'cruise', 'descent','reserve_climb','reserve_cruise','reserve_descent','loiter']
        phases = ['groundroll','climb', 'cruise', 'descent']
        oc.plot_trajectory(prob, x_var, x_unit, y_vars, y_units, phases,
                        x_label=x_label, y_labels=y_labels, marker='-',
                        plot_title='Hybrid SA Mission Profile')
    # prob.model.list_outputs()

def run_hybrid_sa_analysis(plots=True):
    num_nodes = 21
    prob = configure_problem()
    prob.setup(check=True, mode='fwd')
    set_values(prob, num_nodes)
    prob.run_model()
    prob.model.list_outputs(includes=['*.M','*.nozzle_pressure*'], print_arrays=True, units=True)
    # prob.model.list_inputs(includes=['*.sta*'], print_arrays=True, units=True)

    # prob.check_partials(includes=['*duct2.*'], compact_print=True)
    if plots:
        show_outputs(prob)
    return prob


if __name__ == "__main__":
    run_hybrid_sa_analysis(plots=True)    