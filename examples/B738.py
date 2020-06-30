from __future__ import division
import sys
import os
import numpy as np

sys.path.insert(0, os.getcwd())
from openmdao.api import Problem, Group, ScipyOptimizeDriver
from openmdao.api import DirectSolver, SqliteRecorder, IndepVarComp
from openmdao.api import NewtonSolver, BoundsEnforceLS
import openmdao.api as om
# imports for the airplane model itself
from openconcept.analysis.aerodynamics import PolarDrag
from openconcept.utilities.math import AddSubtractComp
from openconcept.utilities.math.integrals import Integrator
from openconcept.utilities.dict_indepvarcomp import DictIndepVarComp
from examples.aircraft_data.B738 import data as acdata
from openconcept.analysis.performance.mission_profiles import MissionWithReserve
from openconcept.utilities.visualization import plot_trajectory
from openconcept.components.cfm56 import CFM56

class B738AirplaneModel(Group):
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
                  thrust_in={'value': 1.0*np.ones((nn,)),
                     'units': 'kN'},
                  thrust={'value': 1.0*np.ones((nn,)),
                       'units': 'kN'},
                  fuel_flow={'value': 1.0*np.ones((nn,)),
                     'units': 'kg/s'},
                  fuel_flow_in={'value': 1.0*np.ones((nn,)),
                       'units': 'kg/s'})
        self.add_subsystem('doubler', doubler, promotes_outputs=['*'])
        self.connect('propmodel.thrust', 'doubler.thrust_in')
        self.connect('propmodel.fuel_flow', 'doubler.fuel_flow_in')
        # use a different drag coefficient for takeoff versus cruise
        if flight_phase not in ['v0v1', 'v1v0', 'v1vr', 'rotate']:
            cd0_source = 'ac|aero|polar|CD0_cruise'
        else:
            cd0_source = 'ac|aero|polar|CD0_TO'
        self.add_subsystem('drag', PolarDrag(num_nodes=nn),
                           promotes_inputs=['fltcond|CL', 'ac|geom|*', ('CD0', cd0_source),
                                            'fltcond|q', ('e', 'ac|aero|polar|e')],
                           promotes_outputs=['drag'])

        # generally the weights module will be custom to each airplane
        passthru = om.ExecComp('OEW=x',
                  x={'value': 1.0,
                     'units': 'kg'},
                  OEW={'value': 1.0,
                       'units': 'kg'})
        self.add_subsystem('OEW', passthru,
                           promotes_inputs=[('x', 'ac|weights|OEW')],
                           promotes_outputs=['OEW'])

        # airplanes which consume fuel will need to integrate
        # fuel usage across the mission and subtract it from TOW
        nn_simpson = int((nn - 1) / 2)
        self.add_subsystem('intfuel', Integrator(num_intervals=nn_simpson, method='simpson',
                                                 quantity_units='kg', diff_units='s',
                                                 time_setup='duration'),
                           promotes_inputs=[('dqdt', 'fuel_flow'), 'duration',
                                            ('q_initial', 'fuel_used_initial')],
                           promotes_outputs=[('q', 'fuel_used'), ('q_final', 'fuel_used_final')])
        self.add_subsystem('weight', AddSubtractComp(output_name='weight',
                                                     input_names=['ac|weights|MTOW', 'fuel_used'],
                                                     units='kg', vec_size=[1, nn],
                                                     scaling_factors=[1, -1]),
                           promotes_inputs=['*'],
                           promotes_outputs=['weight'])


class B738AnalysisGroup(Group):
    """This is an example of a three-phase mission analysis.
    """
    def setup(self):
        # Define number of analysis points to run pers mission segment
        nn = 11

        # Define a bunch of design varaiables and airplane-specific parameters
        dv_comp = self.add_subsystem('dv_comp', DictIndepVarComp(acdata, seperator='|'),
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

        dv_comp.add_output_from_dict('ac|num_passengers_max')
        dv_comp.add_output_from_dict('ac|q_cruise')

        # Ensure that any state variables are connected across the mission as intended
        connect_phases = ['climb', 'cruise', 'descent']
        connect_states = ['range', 'fuel_used', 'fltcond|h']
        extra_states_tuple = [(connect_state, connect_phases) for connect_state in connect_states]
        connect_phases = ['reserve_climb', 'reserve_cruise', 'reserve_descent']
        connect_states = ['range', 'fuel_used', 'fltcond|h']
        for connect_state in connect_states:
            extra_states_tuple.append((connect_state, connect_phases))
        extra_states_tuple.append(('fuel_used', ['descent', 'reserve_climb']))
        extra_states_tuple.append(('fuel_used', ['reserve_descent', 'loiter']))
        extra_states_tuple.append(('range', ['descent', 'reserve_climb']))
        extra_states_tuple.append(('range', ['reserve_descent', 'loiter']))

        print(extra_states_tuple)

        # Run a full mission analysis including takeoff, reserve_, cruise,reserve_ and descereserve_nt
        analysis = self.add_subsystem('analysis',
                                      MissionWithReserve(num_nodes=nn,
                                                          aircraft_model=B738AirplaneModel,
                                                          extra_states=extra_states_tuple),
                                      promotes_inputs=['*'], promotes_outputs=['*'])


if __name__ == "__main__":
    # Set up OpenMDAO to analyze the airplane
    num_nodes = 11
    prob = Problem()
    prob.model = B738AnalysisGroup()
    prob.model.nonlinear_solver = NewtonSolver(iprint=2,solve_subsystems=True)
    prob.model.options['assembled_jac_type'] = 'csc'
    prob.model.linear_solver = DirectSolver(assemble_jac=True)
    prob.model.nonlinear_solver.options['solve_subsystems'] = True
    prob.model.nonlinear_solver.options['maxiter'] = 20
    prob.model.nonlinear_solver.options['atol'] = 1e-6
    prob.model.nonlinear_solver.options['rtol'] = 1e-6
    prob.model.nonlinear_solver.linesearch = BoundsEnforceLS(bound_enforcement='scalar', print_bound_enforce=False)
    prob.setup(check=True, mode='fwd')

    # set some (required) mission parameters. Each pahse needs a vertical and air-speed
    # the entire mission needs a cruise altitude and range
    prob.set_val('climb.fltcond|vs', np.linspace(2300.,  600.,num_nodes), units='ft/min')
    prob.set_val('climb.fltcond|Ueas', np.linspace(230, 220,num_nodes), units='kn')
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

    prob.run_model()

    # print some outputs
    vars_list = ['descent.fuel_used_final','loiter.fuel_used_final']
    units = ['lb','lb']
    nice_print_names = ['Block fuel', 'Total fuel']
    print("=======================================================================")
    for i, thing in enumerate(vars_list):
        print(nice_print_names[i]+': '+str(prob.get_val(thing,units=units[i])[0])+' '+units[i])

    # plot some stuff
    plots = True
    if plots:
        x_var = 'range'
        x_unit = 'NM'
        y_vars = ['fltcond|h','fltcond|Ueas','fuel_used','throttle','fltcond|vs','fltcond|M','fltcond|CL']
        y_units = ['ft','kn','lbm',None,'ft/min', None, None]
        x_label = 'Range (nmi)'
        y_labels = ['Altitude (ft)', 'Veas airspeed (knots)', 'Fuel used (lb)', 'Throttle setting', 'Vertical speed (ft/min)', 'Mach number', 'CL']
        phases = ['climb', 'cruise', 'descent','reserve_climb','reserve_cruise','reserve_descent','loiter']
        plot_trajectory(prob, x_var, x_unit, y_vars, y_units, phases,
                        x_label=x_label, y_labels=y_labels, marker='-',
                        plot_title='737-800 Mission Profile')
    prob.model.list_outputs(residuals=True)

    #28808 block fuel
    #1440 reserve