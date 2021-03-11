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
from openconcept.utilities.linearinterp import LinearInterpolator
from openconcept.utilities.math.add_subtract_comp import AddSubtractComp
from openconcept.utilities.math.multiply_divide_comp import ElementMultiplyDivideComp
from openconcept.components.thermal import LiquidCooledComp, HeatPumpWithIntegratedCoolantLoop_FixedWdot
from openconcept.components.splitter import FlowSplit, FlowCombine
from openconcept.components.heat_sinks import LiquidCooledMotor
from openconcept.components.heat_sinks import LiquidCooledBattery
from openconcept.components.ducts import ImplicitCompressibleDuct_ExternalHX, ExplicitIncompressibleDuct
from openconcept.components.heat_exchanger import HXGroup

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

        # Hybrid propulsion motor (model one side only, then double the weight)
        self.add_subsystem('hybrid_motor', SimpleMotor(num_nodes=nn, efficiency=0.97), 
                           promotes_inputs=[('elec_power_rating','ac|propulsion|motor|rating')])
        self.connect('hybrid_motor.shaft_power_out', 'engine.hybrid_power')

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

        # Hybrid propulsion battery (model one side only, then double the weight)
        self.add_subsystem('battery', SOCBattery(num_nodes=nn, efficiency=0.95, specific_energy=400), 
                           promotes_inputs=[('battery_weight','ac|propulsion|battery|weight')])
        elecadder = self.add_subsystem('elecadder', AddSubtractComp())
        elecadder.add_equation('elec_load', ['motor_load','refrig_load'], vec_size=nn, length=1, val=1.0,
                     units='kW', scaling_factors=[1.0, 2.0])
        
        self.connect('hybrid_motor.elec_load','elecadder.motor_load')
        self.connect('elecadder.elec_load','battery.elec_load')

        iv = self.add_subsystem('iv',om.IndepVarComp(), promotes_outputs=['*'])
        iv.add_output('mdot_coolant', val=6.0*np.ones((nn,)), units='kg/s')
        iv.add_output('rho_coolant', val=1020*np.ones((nn,)),units='kg/m**3')
        iv.add_output('eff_factor', val=0.40)
        iv.add_output('bypass_heat_pump', val=np.ones((nn,)))

        iv.add_output('area_nozzle_start', val=20, units='inch**2')
        iv.add_output('area_nozzle_end', val=20, units='inch**2')

        self.add_subsystem('fluid_split', FlowSplit(num_nodes=nn))
        self.connect('mdot_coolant', 'fluid_split.mdot_in')
        self.connect('fluid_split.mdot_out_A',['motorheatsink.mdot_coolant','fluid_combine.mdot_in_A'])
        self.connect('fluid_split.mdot_out_B',['batteryheatsink.mdot_coolant','fluid_combine.mdot_in_B'])


        self.add_subsystem('fluid_combine', FlowCombine(num_nodes=nn))
        self.connect('motorheatsink.T_out','fluid_combine.T_in_A')
        self.connect('batteryheatsink.T_out','fluid_combine.T_in_B')


        li = self.add_subsystem('li',LinearInterpolator(num_nodes=nn, units='inch**2'), promotes_outputs=[('vec', 'area_nozzle')])
        self.connect('area_nozzle_start','li.start_val')
        self.connect('area_nozzle_end','li.end_val')
        
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

        # Hot side balance param will be set to the cooling duct nozzle area
        self.add_subsystem('refrig', HeatPumpWithIntegratedCoolantLoop_FixedWdot(num_nodes=nn),
                           promotes_inputs=['bypass_heat_pump', 'eff_factor',('power_rating','ac|propulsion|thermal|heatpump|power_rating')])
        self.connect('refrig.elec_load','elecadder.refrig_load')

        
        self.connect('mdot_coolant',['hx.mdot_hot','refrig.mdot_coolant_hot'])
        self.connect('hx.T_out_hot','refrig.T_in_hot')
        self.connect('refrig.T_out_hot','hx.T_in_hot')


        self.connect('fluid_combine.mdot_out','refrig.mdot_coolant_cold')
        self.connect('fluid_combine.T_out','refrig.T_in_cold')
        self.connect('refrig.T_out_cold',['motorheatsink.T_in','batteryheatsink.T_in'])

        self.add_subsystem('hx',HXGroup(num_nodes=nn),promotes_inputs=['ac|propulsion|thermal|hx|n_wide_cold'])
        self.connect('hx.delta_p_cold','duct2.sta3.delta_p')
        self.connect('hx.heat_transfer','duct2.sta3.heat_in')
        self.connect('duct2.mdot','hx.mdot_cold')        
        self.connect('rho_coolant','hx.rho_hot')

        duct = self.add_subsystem('duct2',
                           ImplicitCompressibleDuct_ExternalHX(num_nodes=nn),
                           promotes_inputs=[('p_inf','fltcond|p'),('T_inf','fltcond|T'),('Utrue','fltcond|Utrue')])
        # in to HXGroup:
        self.connect('duct2.sta2.T', 'hx.T_in_cold')
        self.connect('duct2.sta2.rho', 'hx.rho_cold')

        #out from HXGroup
        self.connect('hx.frontal_area', ['duct2.area_2', 'duct2.area_3'])
        self.connect('area_nozzle', ['duct.area_nozzle','duct2.area_nozzle_in'])

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
        adder.add_equation('drag', ['airframe_drag','hx_drag'], vec_size=nn, length=1, val=1.0,
                     units='N', scaling_factors=[1.0, -2.0])
        self.connect('drag.drag', 'adder.airframe_drag')
        self.connect('duct2.force.F_net', 'adder.hx_drag')
        # generally the weights module will be custom to each airplane

        # Motor, Battery, TMS, N+3 weight delta
        self.add_subsystem('oewcalc', oc.AddSubtractComp(output_name='OEW',
                                                     input_names=['ac|weights|OEW', 'refrig_weight'],
                                                     units='kg', vec_size=[1, 1],
                                                     scaling_factors=[1, 2]),
                           promotes_inputs=['ac|weights|OEW'],
                           promotes_outputs=['OEW'])
        self.connect('refrig.component_weight','oewcalc.refrig_weight')

        self.add_subsystem('weight', oc.AddSubtractComp(output_name='weight',
                                                     input_names=['ac|weights|MTOW', 'fuel_used'],
                                                     units='kg', vec_size=[1, nn],
                                                     scaling_factors=[1, -1]),
                           promotes_inputs=['*'],
                           promotes_outputs=['weight'])

        weightmargin = om.ExecComp('margin=MTOW-OEW-battery-fuel',
                                    margin={'value': 0.0,
                                            'units': 'kg'},
                                    MTOW={'value': 0.0,
                                          'units': 'kg'},
                                    battery={'value': 0.0,
                                             'units':'kg'},
                                    fuel={'value':0.0,
                                          'units':'kg'},
                                    OEW={'value': 0.0,
                                         'units':'kg'})
        self.add_subsystem('weightmargin', weightmargin, promotes_inputs=[('MTOW','ac|weights|MTOW'),
                                                                          'OEW',
                                                                          ('fuel', 'fuel_used_final'),
                                                                          ('battery','ac|propulsion|battery|weight')],
                                                        promotes_outputs=['*'])
            
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
        dv_comp.add_output_from_dict('ac|propulsion|thermal|heatpump|power_rating')

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
    prob.model.nonlinear_solver.options['maxiter'] = 20
    prob.model.nonlinear_solver.options['atol'] = 5e-8
    prob.model.nonlinear_solver.options['rtol'] = 5e-8
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
    prob.set_val('cruise|h0',33000.,units='ft')
    prob.set_val('mission_range',2050,units='NM')
    prob.set_val('takeoff|v2', 160., units='kn')
    nozzleprs = [0.85,0.85,0.71,0.88]
    phases_list = ['groundroll','climb', 'cruise', 'descent']          
    for i, phase in enumerate(phases_list):
        prob.set_val(phase+'.duct2.area_1', 150, units='inch**2')
        prob.set_val(phase+'.hybrid_motor.throttle', 0.00)
        prob.set_val(phase+'.fltcond|TempIncrement', 20, units='degC')
        prob.set_val(phase+'.duct2.sta1.M', 0.8)
        prob.set_val(phase+'.duct2.sta2.M', 0.05)
        prob.set_val(phase+'.duct2.sta3.M', 0.05)
        # prob.set_val(phase+'.duct2.mdot',9)
        prob.set_val(phase+'.duct2.nozzle.nozzle_pressure_ratio', 0.95)
        prob.set_val(phase+'.duct2.convergence_hack', -1.0, units='Pa')
        prob.set_val(phase+'.fluid_split.mdot_split_fraction', 0.2, units=None)
        prob.set_val(phase+'.hx.channel_height_hot', 3, units='mm')
        prob.set_val(phase+'.hx.n_long_cold', 3)
        prob.set_val(phase+'.hx.n_tall', 50, units=None)
    prob.set_val('groundroll.duct2.sta1.M', 0.2)
    prob.set_val('groundroll.duct2.nozzle.nozzle_pressure_ratio', 0.85)
    prob.set_val('groundroll.duct2.convergence_hack', -500, units='Pa')
    prob.set_val('groundroll.bypass_heat_pump', np.zeros((num_nodes,)))
    prob.set_val('climb.bypass_heat_pump', np.zeros((num_nodes,)))

    prob.set_val('groundroll.area_nozzle_start', 60, units='inch**2')
    prob.set_val('groundroll.area_nozzle_end', 60, units='inch**2')
    prob.set_val('descent.area_nozzle_start', 20, units='inch**2')
    prob.set_val('descent.area_nozzle_end', 20, units='inch**2')

    prob.set_val('groundroll.hybrid_motor.throttle', np.linspace(1.0, 1.0, num_nodes))
    prob.set_val('climb.hybrid_motor.throttle', np.linspace(0.5, 1.0, num_nodes))
    prob.set_val('cruise.hybrid_motor.throttle', np.linspace(1.0, 1.0, num_nodes))
    prob.set_val('groundroll.motorheatsink.T_initial', 30., 'degC')
    prob.set_val('groundroll.batteryheatsink.T_initial', 30., 'degC')
    prob.set_val('groundroll.fltcond|Utrue',np.ones((num_nodes))*50,units='kn')

def show_outputs(prob):
    # print some outputs
    vars_list = ['descent.fuel_used_final', 'descent.hx.xs_area_cold', 'descent.hx.frontal_area']
    units = ['lb','inch**2','inch**2']
    nice_print_names = ['Block fuel','Duct HX XS area','Duct HX Frontal Area']
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
        y_labels = ['Altitude (ft)', 'Veas airspeed (knots)', 'Fuel used (lb)', 'Throttle setting', 'Vertical speed (ft/min)', 'Mach number', 'CL', 'Batt SOC', 'Motor Temp', 'Battery Temp (C)', 'Battery Coolant Inflow Temp', 'Cooling Net Force (lb)', 'Incomp Duct Drag']
        # phases = ['climb', 'cruise', 'descent','reserve_climb','reserve_cruise','reserve_descent','loiter']
        phases = ['groundroll','climb', 'cruise', 'descent']
        oc.plot_trajectory(prob, x_var, x_unit, y_vars, y_units, phases,
                        x_label=x_label, y_labels=y_labels, marker='-',
                        plot_title='Hybrid Single Aisle Mission')
    # prob.model.list_outputs()

def run_hybrid_sa_analysis(plots=True):
    num_nodes = 21
    prob = configure_problem()
    prob.model.add_design_var('ac|propulsion|thermal|hx|n_wide_cold', 2, 1500, scaler=0.005, units=None)
    prob.model.add_design_var('cruise.hx.n_long_cold', lower=7., upper=75., scaler=0.05)
    prob.model.add_design_var('ac|propulsion|battery|weight', lower=2000, upper=15000, scaler=0.0001)
    prob.model.add_constraint('descent.battery.SOC_final', lower=0.05, scaler=10)
    prob.model.add_constraint('descent.hx.width_overall', upper=1.2)
    prob.model.add_constraint('descent.hx.xs_area_cold', lower=70, units='inch**2', scaler=0.01)
    prob.model.add_objective('descent.margin', scaler=-0.0001)
    prob.model.add_design_var('climb.refrig.control.Wdot_start', lower=0.01, upper=1.0, units=None, scaler=2.)
    prob.model.add_design_var('ac|propulsion|thermal|heatpump|power_rating', lower=5.0, upper=50., units='kW', scaler=0.1)
    # prob.model.add_design_var('climb.refrig.control.Wdot_end', lower=0.2, upper=1.0, units=None, scaler=2.)
    # prob.model.add_design_var('cruise.refrig.control.Wdot_start', lower=0.2, upper=1.0, units=None, scaler=2.)
    # prob.model.add_design_var('cruise.refrig.control.Wdot_end', lower=0.2, upper=1.0, units=None, scaler=2.)
    phases_list = ['climb','cruise']          
    for phase in phases_list:
        prob.model.add_design_var(phase+'.area_nozzle_start', lower=10., upper=150., scaler=0.1, units='inch**2')
        prob.model.add_design_var(phase+'.area_nozzle_end', lower=10., upper=150., scaler=0.1, units='inch**2')
        prob.model.add_constraint(phase+'.batteryheatsink.T',  upper=45, scaler=0.1, units='degC')
    phases_list = ['descent']          
    for phase in phases_list:
        prob.model.add_design_var(phase+'.area_nozzle_start', lower=10., upper=150., scaler=0.5, units='inch**2')
        prob.model.add_design_var(phase+'.area_nozzle_end', lower=6.5, upper=150., scaler=0.5, units='inch**2')
        prob.model.add_constraint(phase+'.batteryheatsink.T',  indices=[20], upper=35, scaler=0.1, units='degC')
    
    # prob.driver = om.pyOptSparseDriver(optimizer='IPOPT')
    # prob.driver.opt_settings['limited_memory_max_history'] = 1000
    # prob.driver.opt_settings['print_level'] = 5


    prob.driver = om.pyOptSparseDriver(optimizer='SNOPT')
    prob.driver.opt_settings['Major iterations limit'] = 100
    prob.driver.opt_settings['Function precision'] = 0.00001
    prob.driver.opt_settings['Major optimality tolerance'] = 5e-9
    prob.driver.opt_settings['Hessian frequency'] = 10
    prob.driver.opt_settings['Linesearch tolerance'] = 0.99
    prob.driver.opt_settings['Penalty parameter'] = 5.

    prob.driver.options['debug_print'] = ['desvars','objs']
    prob.setup(check=True, mode='fwd', force_alloc_complex=True)
    set_values(prob, num_nodes)
    prob.run_model()
    phases_list = ['groundroll','climb', 'cruise', 'descent']          
    print('=======================================')
    for phase in phases_list:
        prob.set_val(phase+'.hx.n_long_cold', 50)
        if phase != 'groundroll':
            prob.set_val(phase+'.duct2.nozzle.dynamic_pressure_loss_factor', 0.15)
    # myvec = np.zeros((num_nodes,))
    # prob.set_val('cruise.bypass_heat_pump', myvec)
    prob.run_model()
    prob.run_driver()   

    
    prob.model.list_inputs(includes=['*cruise.hx*'], excludes=['*duct*'], print_arrays=True)
    prob.model.list_outputs(includes=['*cruise.hx*'], units=True,  print_arrays=True)
    prob.list_problem_vars(print_arrays=True)
    # prob.check_partials(show_only_incorrect=False, compact_print=True, method='cs',excludes=['*engine*'])
    # prob.check_totals(compact_print=True, step=1e-3)
    if plots:
        show_outputs(prob)
    return prob


if __name__ == "__main__":
    run_hybrid_sa_analysis(plots=False)    