from __future__ import division
import sys
import os
import warnings
import numpy as np

sys.path.insert(0, os.getcwd())
import openmdao.api as om
import openconcept.api as oc
# imports for the airplane model itself
from openconcept.analysis.openaerostruct.aerostructural import OASAerostructDragPolar
from examples.aircraft_data.B738 import data as acdata
from openconcept.analysis.performance.mission_profiles import MissionWithReserve
from openconcept.components.cfm56 import CFM56
from openconcept.analysis.openaerostruct.aerostructural import Aerostruct
from openconcept.analysis.aerodynamics import Lift
from openconcept.analysis.atmospherics.dynamic_pressure_comp import DynamicPressureComp

NUM_X = 5
NUM_Y = 15

class B738AirplaneModel(oc.IntegratorGroup):
    """
    A custom model specific to the Boeing 737-800 airplane.
    This class will be passed in to the mission analysis code.

    """
    def initialize(self):
        self.options.declare('num_nodes', default=1)
        self.options.declare('flight_phase', default=None)

    def setup(self):
        nn = self.options['num_nodes']
        flight_phase = self.options['flight_phase']


        # a propulsion system needs to be defined in order to provide thrust
        # information for the mission analysis code
        # propulsion_promotes_outputs = ['fuel_flow', 'thrust']
        propulsion_promotes_inputs = ["fltcond|*", "throttle"]

        self.add_subsystem('propmodel', CFM56(num_nodes=nn, plot=False),
                           promotes_inputs=propulsion_promotes_inputs)

        doubler = om.ExecComp(['thrust=2*thrust_in', 'fuel_flow=2*fuel_flow_in'], 
                  thrust_in={'val': 1.0*np.ones((nn,)),
                     'units': 'kN'},
                  thrust={'val': 1.0*np.ones((nn,)),
                       'units': 'kN'},
                  fuel_flow={'val': 1.0*np.ones((nn,)),
                     'units': 'kg/s',
                     'tags': ['integrate', 'state_name:fuel_used', 'state_units:kg', 'state_val:1.0', 'state_promotes:True']},
                  fuel_flow_in={'val': 1.0*np.ones((nn,)),
                       'units': 'kg/s'})
        
        self.add_subsystem('doubler', doubler, promotes_outputs=['*'])
        self.connect('propmodel.thrust', 'doubler.thrust_in')
        self.connect('propmodel.fuel_flow', 'doubler.fuel_flow_in')

        oas_surf_dict = {}  # options for OpenAeroStruct
        # Grid size and number of spline control points (must be same as B738AnalysisGroup)
        num_x = NUM_X
        num_y = NUM_Y
        n_twist = 3
        n_toverc = 3
        n_skin = 3
        n_spar = 3
        self.add_subsystem('drag', OASAerostructDragPolar(num_nodes=nn, num_x=num_x, num_y=num_y,
                                                num_twist=n_twist, num_toverc=n_toverc,
                                                num_skin=n_skin, num_spar=n_spar,
                                                surf_options=oas_surf_dict),
                           promotes_inputs=['fltcond|CL', 'fltcond|M', 'fltcond|h', 'fltcond|q', 'ac|geom|wing|S_ref',
                                            'ac|geom|wing|AR', 'ac|geom|wing|taper', 'ac|geom|wing|c4sweep',
                                            'ac|geom|wing|twist', 'ac|geom|wing|toverc',
                                            'ac|geom|wing|skin_thickness', 'ac|geom|wing|spar_thickness',
                                            'ac|aero|CD_nonwing'],
                           promotes_outputs=['drag', 'ac|weights|W_wing', ('failure', 'ac|struct|failure')])

        # generally the weights module will be custom to each airplane
        passthru = om.ExecComp('OEW=x',
                  x={'val': 1.0,
                     'units': 'kg'},
                  OEW={'val': 1.0,
                       'units': 'kg'})
        self.add_subsystem('OEW', passthru,
                           promotes_inputs=[('x', 'ac|weights|OEW')],
                           promotes_outputs=['OEW'])

        # Use Raymer as estimate for 737 original wing weight, subtract it
        # out, then add in OpenAeroStruct wing weight estimate
        self.add_subsystem('weight', oc.AddSubtractComp(output_name='weight',
                                                     input_names=['ac|weights|MTOW', 'fuel_used',
                                                                  'ac|weights|orig_W_wing',
                                                                  'ac|weights|W_wing'],
                                                     units='kg', vec_size=[1, nn, 1, 1],
                                                     scaling_factors=[1, -1, -1, 1]),
                           promotes_inputs=['*'],
                           promotes_outputs=['weight'])

class B738AnalysisGroup(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', default=11, desc='Number of analysis points per flight segment')

    def setup(self):
        # Define number of analysis points to run pers mission segment
        nn = self.options['num_nodes']

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
        # dv_comp.add_output_from_dict('ac|geom|wing|toverc')
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

        dv_comp.add_output_from_dict('ac|num_passengers_max')
        dv_comp.add_output_from_dict('ac|q_cruise')

        # Aerostructural design parameters
        num_x = NUM_X
        num_y = NUM_Y
        n_twist = 3
        n_toverc = 3
        n_skin = 3
        n_spar = 3
        twist = np.linspace(-2, 2, n_twist)
        toverc = acdata['ac']['geom']['wing']['toverc']['value'] * np.ones(n_toverc)
        t_skin = np.array([0.005, 0.007, 0.015])
        t_spar = np.array([0.005, 0.007, 0.015])
        self.set_input_defaults('ac|geom|wing|twist', twist, units='deg')
        self.set_input_defaults('ac|geom|wing|toverc', toverc)
        self.set_input_defaults('ac|geom|wing|skin_thickness', t_skin, units='m')
        self.set_input_defaults('ac|geom|wing|spar_thickness', t_spar, units='m')
        self.set_input_defaults('ac|aero|CD_nonwing', 0.0145)  # based on matching fuel burn of B738.py example

        # Compute Raymer wing weight to know what to subtract from the MTOW before adding the OpenAeroStruct weight
        W_dg = 174.2e3  # design gross weight, lbs
        N_z = 1.5*3.  # ultimate load factor (1.5 x limit load factor of 3g)
        S_w = 1368.  # trapezoidal wing area, ft^2 (from photogrammetry)
        A = 9.44  # aspect ratio
        t_c = 0.12  # root thickness to chord ratio
        taper = 0.159  # taper ratio
        sweep = 25.  # wing sweep at 25% MAC
        S_csw = 196.8  # wing-mounted control surface area, ft^2 (from photogrammetry)
        W_wing_raymer = 0.0051 * (W_dg * N_z)**0.557 * S_w**0.649 * A**0.5 * \
                        (t_c)**(-0.4) * (1 + taper)**0.1 / np.cos(np.deg2rad(sweep)) * S_csw**0.1
        self.set_input_defaults('ac|weights|orig_W_wing', W_wing_raymer, units='lb')

        # ======================== Mission analysis ========================
        # Run a full mission analysis including takeoff, reserve_, cruise,reserve_ and descereserve_nt
        analysis = self.add_subsystem('analysis',
                                      MissionWithReserve(num_nodes=nn,
                                                          aircraft_model=B738AirplaneModel),
                                      promotes_inputs=['*'], promotes_outputs=['*'])
        
        # ======================== Aerostructural sizing at 2.5g ========================
        # Add single point aerostructural analysis at 2.5g and MTOW to size the wingbox structure
        self.add_subsystem('aerostructural_maneuver', Aerostruct(num_x=num_x, num_y=num_y, num_twist=n_twist,
                                                                    num_toverc=n_toverc, num_skin=n_skin,
                                                                    num_spar=n_spar),
                           promotes_inputs=['ac|geom|wing|S_ref', 'ac|geom|wing|AR', 'ac|geom|wing|taper',
                                            'ac|geom|wing|c4sweep', 'ac|geom|wing|toverc',
                                            'ac|geom|wing|skin_thickness', 'ac|geom|wing|spar_thickness',
                                            'load_factor'],
                           promotes_outputs=[('failure', '2_5g_KS_failure')])
        
        # Flight condition of 2.5g maneuver load case
        self.set_input_defaults('aerostructural_maneuver.fltcond|M', 0.8)
        self.set_input_defaults('aerostructural_maneuver.fltcond|h', 20e3, units='ft')
        self.set_input_defaults('load_factor', 2.5)  # multiplier on weights in structural problem

        # Find angle of attack for 2.5g sizing flight condition such that lift = 2.5 * MTOW
        self.add_subsystem('dyn_pressure', DynamicPressureComp(num_nodes=1))
        self.add_subsystem('lift', Lift(num_nodes=1), promotes_inputs=['ac|geom|wing|S_ref'])
        self.add_subsystem('kg_to_N', om.ExecComp('force = load_factor * mass * a',
                                                                      force={'units': 'N'},
                                                                      mass={'units': 'kg'},
                                                                      a={'units': 'm/s**2', 'val': 9.807}),
                           promotes_inputs=[('mass', 'ac|weights|MTOW'), 'load_factor'])
        self.add_subsystem('struct_sizing_AoA', om.BalanceComp('alpha', eq_units='N', lhs_name='MTOW', rhs_name='lift', units='deg'))
        self.connect('kg_to_N.force', 'struct_sizing_AoA.MTOW')
        self.connect('aerostructural_maneuver.density.fltcond|rho', 'dyn_pressure.fltcond|rho')
        self.connect('aerostructural_maneuver.airspeed.Utrue', 'dyn_pressure.fltcond|Utrue')
        self.connect('dyn_pressure.fltcond|q', 'lift.fltcond|q')
        self.connect('aerostructural_maneuver.fltcond|CL', 'lift.fltcond|CL')
        self.connect('lift.lift', 'struct_sizing_AoA.lift')
        self.connect('struct_sizing_AoA.alpha', 'aerostructural_maneuver.fltcond|alpha')
        

def configure_problem(num_nodes):
    prob = om.Problem()
    prob.model.add_subsystem('analysis', B738AnalysisGroup(num_nodes=num_nodes), promotes=['*'])
    prob.model.nonlinear_solver = om.NewtonSolver(iprint=2,solve_subsystems=True)
    prob.model.linear_solver = om.DirectSolver()
    prob.model.nonlinear_solver.options['maxiter'] = 10
    prob.model.nonlinear_solver.options['atol'] = 1e-6
    prob.model.nonlinear_solver.options['rtol'] = 1e-6
    prob.model.nonlinear_solver.options['err_on_non_converge'] = True
    prob.model.nonlinear_solver.linesearch = om.BoundsEnforceLS(bound_enforcement='scalar', print_bound_enforce=False)

    prob.driver = om.pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SNOPT'
    prob.driver.opt_settings['Major feasibility tolerance'] = 7e-6
    prob.driver.options['debug_print'] = ['objs', 'desvars']

    # =========================== Mission design variables/constraints ===========================
    prob.model.add_objective('descent.fuel_used_final')  # minimize block fuel burn
    # prob.model.add_design_var('cruise|h0', upper=45e3, units='ft')
    prob.model.add_constraint('climb.throttle', lower=0.01, upper=1.05)
    prob.model.add_constraint('cruise.throttle', lower=0.01, upper=1.05)
    prob.model.add_constraint('descent.throttle', lower=0.01, upper=1.05)

    # =========================== Aerostructural wing design variables/constraints ===========================
    # Find twist distribution that minimizes fuel burn; lock the twist tip in place
    # to prevent rigid rotation of the whole wing
    prob.model.add_design_var('ac|geom|wing|twist', lower=np.array([0, -10, -10]),
                              upper=np.array([0, 10, 10]), units='deg')
    prob.model.add_design_var('ac|geom|wing|AR', lower=5., upper=17.)
    prob.model.add_design_var('ac|geom|wing|c4sweep', lower=0., upper=45.)
    prob.model.add_design_var('ac|geom|wing|toverc', lower=.1, upper=0.25)
    prob.model.add_design_var("ac|geom|wing|spar_thickness", lower=0.003, upper=0.1, scaler=1e2, units='m')
    prob.model.add_design_var("ac|geom|wing|skin_thickness", lower=0.003, upper=0.1, scaler=1e2, units='m')
    prob.model.add_design_var('ac|geom|wing|taper', lower=.01, scaler=1e1)
    prob.model.add_constraint('2_5g_KS_failure', upper=0.)
    
    return prob

def set_values(prob, num_nodes):
    # set some (required) mission parameters. Each pahse needs a vertical and air-speed
    # the entire mission needs a cruise altitude and range
    prob.set_val('climb.fltcond|vs', np.linspace(2300.,  500.,num_nodes), units='ft/min')
    prob.set_val('climb.fltcond|Ueas', np.linspace(230, 210,num_nodes), units='kn')
    prob.set_val('cruise.fltcond|vs', np.ones((num_nodes,)) * 4., units='ft/min')
    prob.set_val('cruise.fltcond|Ueas', np.linspace(265, 258, num_nodes), units='kn')
    prob.set_val('descent.fltcond|vs', np.linspace(-1000, -150, num_nodes), units='ft/min')
    prob.set_val('descent.fltcond|Ueas', np.ones((num_nodes,)) * 250, units='kn')
    prob.set_val('reserve_climb.fltcond|vs', np.linspace(3000.,  2300.,num_nodes), units='ft/min')
    prob.set_val('reserve_climb.fltcond|Ueas', np.linspace(230, 230,num_nodes), units='kn')
    prob.set_val('reserve_cruise.fltcond|vs', np.ones((num_nodes,)) * 4., units='ft/min')
    prob.set_val('reserve_cruise.fltcond|Ueas', np.linspace(250, 250, num_nodes), units='kn')
    prob.set_val('reserve_descent.fltcond|vs', np.linspace(-800, -800, num_nodes), units='ft/min')
    prob.set_val('reserve_descent.fltcond|Ueas', np.ones((num_nodes,)) * 250, units='kn')
    prob.set_val('loiter.fltcond|vs', np.linspace(0.0, 0.0, num_nodes), units='ft/min')
    prob.set_val('loiter.fltcond|Ueas', np.ones((num_nodes,)) * 200, units='kn')
    prob.set_val('cruise|h0',33000.,units='ft')
    prob.set_val('reserve|h0',15000.,units='ft')
    prob.set_val('mission_range',2050,units='NM')

def show_outputs(prob, plots=True):
    # print some outputs
    vars_list = ['descent.fuel_used_final','loiter.fuel_used_final']
    units = ['lb','lb']
    nice_print_names = ['Block fuel', 'Total fuel']
    print("=======================================================================")
    for i, thing in enumerate(vars_list):
        print(nice_print_names[i]+': '+str(prob.get_val(thing,units=units[i])[0])+' '+units[i])

    # plot some stuff
    if plots:
        x_var = 'range'
        x_unit = 'NM'
        y_vars = ['fltcond|h','fltcond|Ueas','fuel_used','throttle','fltcond|vs','fltcond|M','fltcond|CL']
        y_units = ['ft','kn','lbm',None,'ft/min', None, None]
        x_label = 'Range (nmi)'
        y_labels = ['Altitude (ft)', 'Veas airspeed (knots)', 'Fuel used (lb)', 'Throttle setting', 'Vertical speed (ft/min)', 'Mach number', 'CL']
        phases = ['climb', 'cruise', 'descent','reserve_climb','reserve_cruise','reserve_descent','loiter']
        oc.plot_trajectory(prob, x_var, x_unit, y_vars, y_units, phases,
                        x_label=x_label, y_labels=y_labels, marker='-',
                        plot_title='737-800 Mission Profile')
    # prob.model.list_outputs()

def run_738_analysis(plots=False):
    num_nodes = 11
    global NUM_X, NUM_Y
    NUM_X = 3
    NUM_Y = 7
    prob = configure_problem(num_nodes)
    prob.setup(check=False, mode='fwd')
    set_values(prob, num_nodes)
    prob.run_model()
    om.n2(prob, show_browser=False)
    # prob.model.list_inputs(print_arrays=True)
    # prob.model.list_outputs(print_arrays=True)
    show_outputs(prob, plots=plots)
    print(f"Wing weight = {prob.get_val('ac|weights|W_wing', units='lb')[0]} lb")
    print(f"Raymer wing weight = {prob.get_val('ac|weights|orig_W_wing', units='lb')[0]} lb")
    print(f"2.5g failure = {prob.get_val('2_5g_KS_failure')}")
    print(f"Climb failure = {prob.get_val('climb.ac|struct|failure')}")
    print(f"Cruise failure = {prob.get_val('cruise.ac|struct|failure')}")
    print(f"Descent failure = {prob.get_val('descent.ac|struct|failure')}")
    return prob

def run_738_optimization(plots=False):
    num_nodes = 11
    global NUM_X, NUM_Y
    NUM_X = 3
    NUM_Y = 7
    prob = configure_problem(num_nodes)
    prob.setup(check=True, mode='fwd')
    set_values(prob, num_nodes)
    prob.run_driver()
    prob.list_problem_vars(driver_scaling=False)
    print(f"Wing weight = {prob.get_val('ac|weights|W_wing', units='lb')[0]} lb")
    print(f"Raymer wing weight = {prob.get_val('ac|weights|orig_W_wing', units='lb')[0]} lb")
    print(f"2.5g failure = {prob.get_val('2_5g_KS_failure')}")
    print(f"Climb failure = {prob.get_val('climb.ac|struct|failure')}")
    print(f"Cruise failure = {prob.get_val('cruise.ac|struct|failure')}")
    print(f"Descent failure = {prob.get_val('descent.ac|struct|failure')}")
    if plots:
        show_outputs(prob)
    return prob


if __name__ == "__main__":
    # run_738_analysis(plots=False)
    run_738_optimization(plots=True)
