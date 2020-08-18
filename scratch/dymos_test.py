from __future__ import division
from openmdao.api import Group, ExplicitComponent, IndepVarComp, BalanceComp, ImplicitComponent
import openmdao.api as om
import openconcept.api as oc
from openconcept.analysis.atmospherics.compute_atmos_props import ComputeAtmosphericProperties
from openconcept.analysis.aerodynamics import Lift
from openconcept.utilities.math import ElementMultiplyDivideComp, AddSubtractComp
from openconcept.utilities.math.integrals import Integrator
from openconcept.utilities.linearinterp import LinearInterpolator
from openconcept.utilities.math.integrals import Integrator
from openconcept.utilities.dict_indepvarcomp import DymosDesignParamsFromDict
from openconcept.analysis.performance.solver_phases import Groundspeeds, SteadyFlightCL, HorizontalAcceleration
import numpy as np
import copy
import dymos as dm
import matplotlib
import matplotlib.pyplot as plt
from examples.aircraft_data.B738 import data as b738data
from examples.B738 import B738AirplaneModel

# TODO make a DymosODE group
# Configure method will iterate down and find tags
# Setup_procs method will push DymosODE metadata down
# where to invoke the add_state in the call stack?

class DymosSteadyFlightODE(om.Group):
    """
    Test
    """
    def initialize(self):
        self.options.declare('num_nodes',default=1)
        self.options.declare('aircraft_model')
        self.options.declare('flight_phase', default='cruise')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_subsystem('atmos', ComputeAtmosphericProperties(num_nodes=nn, true_airspeed_in=False), promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('gs',Groundspeeds(num_nodes=nn),promotes_inputs=['*'],promotes_outputs=['*'])
        self.add_subsystem('acmodel',self.options['aircraft_model'](num_nodes=nn, flight_phase=self.options['flight_phase']),promotes_inputs=['*'],promotes_outputs=['*'])
        self.add_subsystem('clcomp',SteadyFlightCL(num_nodes=nn), promotes_inputs=['*'],promotes_outputs=['*'])
        self.add_subsystem('lift',Lift(num_nodes=nn), promotes_inputs=['*'],promotes_outputs=['*'])
        self.add_subsystem('haccel',HorizontalAcceleration(num_nodes=nn), promotes_inputs=['*'],promotes_outputs=['*'])


def extract_states_from_airplane(acmodel):
    pass

if __name__ == "__main__":

    #
    # Define the OpenMDAO problem
    #
    p = om.Problem(model=om.Group())

    #
    # Define a Trajectory object
    #
    traj = dm.Trajectory()
    p.model.add_subsystem('traj', subsys=traj)

    #
    # Define a Dymos Phase object with GaussLobatto Transcription
    #

    odekwargs = {'aircraft_model': B738AirplaneModel}
    phase0 = dm.Phase(ode_class=DymosSteadyFlightODE, ode_init_kwargs=odekwargs,
                    transcription=dm.Radau(num_segments=11, order=3, solve_segments=True))

    traj.add_phase(name='phase0', phase=phase0)
    # traj.add_phase(name='phase1', phase=phase1)
    acparams = DymosDesignParamsFromDict(b738data, traj)
    acparams.add_output_from_dict('ac|aero|polar|e')
    acparams.add_output_from_dict('ac|aero|polar|CD0_cruise')

    acparams.add_output_from_dict('ac|geom|wing|S_ref')
    acparams.add_output_from_dict('ac|geom|wing|AR')

    acparams.add_output_from_dict('ac|weights|MTOW')
    acparams.add_output_from_dict('ac|weights|OEW')

    #
    # Set the time options
    # Time has no targets in our ODE.
    # We fix the initial time so that the it is not a design variable in the optimization.
    # The duration of the phase is allowed to be optimized, but is bounded on [0.5, 10].
    #
    phase0.set_time_options(fix_initial=True, duration_bounds=(100, 10000), units='s', duration_ref=100., initial_ref0=0.0, initial_ref=1.0)
    # phase1.set_time_options(fix_initial=False, duration_bounds=(50, 10000), units='s')
    #
    # Set the time options
    # Initial values of positions and velocity are all fixed.
    # The final value of position are fixed, but the final velocity is a free variable.
    # The equations of motion are not functions of position, so 'x' and 'y' have no targets.
    # The rate source points to the output in the ODE which provides the time derivative of the
    # given state.

    
    # auto add these
    phase0.add_control(name='throttle', units=None, lower=0.0, upper=1.00, targets=['throttle'], ref=1.0)
    phase0.add_path_constraint('accel_horiz', lower=0.0, ref=1.)
    phase0.add_control(name='fltcond|vs', units='ft/min', lower=400, upper=7000, targets=['fltcond|vs'], opt=False, ref=3000.)
    phase0.add_state('fltcond|h', fix_initial=True, fix_final=False, units='km', rate_source='fltcond|vs', targets=['fltcond|h'], ref=10., defect_ref=10.)
    phase0.add_boundary_constraint('fltcond|h', loc='final', units='ft', equals=33000., ref=33000.)
    phase0.add_control(name='fltcond|Ueas', units='kn', lower=180, upper=250, targets=['fltcond|Ueas'], opt=False, ref=250.)
    phase0.add_state('range', fix_initial=True, fix_final=False, units='km', rate_source='fltcond|groundspeed', ref=100., defect_ref=100.)
    
    # add states for the temperatures
    # add states for the battery SOC
    # add a control for Tc and Th set
    # add a control for duct exit area (with limits)
    # add a path constraint Tmotor < 90C > -10C
    # add a path constraint Tbattery <70C > 0C

    # custom state
    phase0.add_state('fuel_used', fix_initial=True, fix_final=False, units='kg', rate_source='fuel_flow', targets=['fuel_used'], ref=1000., defect_ref=1000.)
    # need to know
    # rate source location
    # target location
    # initial condition
    # scaler
    # defect scaler
    # unit

    # phase1.add_control(name='throttle', units=None, lower=0.0, upper=1.5, targets=['throttle'])
    # phase1.add_path_constraint('accel_horiz', equals=0.0)
    # phase1.add_control(name='fltcond|vs', units='m/s', lower=0, upper=10, targets=['fltcond|vs'])
    # phase1.add_state('fltcond|h', fix_initial=True, fix_final=True, units='km', rate_source='fltcond|vs', targets=['fltcond|h'])
    # phase1.add_control(name='fltcond|Ueas', units='kn', lower=180, upper=250, targets=['fltcond|Ueas'])
    # phase1.add_state('range', fix_initial=False, fix_final=False, units='km', rate_source='fltcond|groundspeed')
    # phase1.add_state('fuel_used', fix_initial=False, fix_final=False, units='kg', lower=0.0, rate_source='fuel_flow', targets=['fuel_used'])

    # traj.link_phases(['phase0','phase1'])
    # traj.add_design_parameter('ac|weights|MTOW', units='kg', val=500., opt=False, targets={'phase0':['ac|weights|MTOW']}, dynamic=False)
    # Minimize final time.
    phase0.add_objective('weight', loc='final', ref=-1000.)
    # phase0.add_boundary_constraint('time', loc='final', units='s', upper=800.)


    # Set the driver.
    p.driver = om.pyOptSparseDriver()
    p.driver.options['optimizer'] = 'SNOPT'
    # Allow OpenMDAO to automatically determine our sparsity pattern.
    # Doing so can significant speed up the execution of Dymos.
    p.driver.declare_coloring()

    # p.model.promotes('traj', inputs=['phase*.rhs_all.ac|*'])

    # Setup the problem


    p.setup(check=True)
    
    # Now that the OpenMDAO problem is setup, we can set the values of the states.
    p['traj.phase0.t_initial'] = 1.0
    p['traj.phase0.t_duration'] = 900.0
    p.set_val('traj.phase0.states:fltcond|h',
            phase0.interpolate(ys=[0, 33000], nodes='state_input'),
            units='ft')

    p.set_val('traj.phase0.states:range',
            phase0.interpolate(ys=[0, 80], nodes='state_input'),
            units='km')

    p.set_val('traj.phase0.states:fuel_used',
            phase0.interpolate(ys=[0, 1000], nodes='state_input'),
            units='kg')

    p.set_val('traj.phase0.controls:fltcond|Ueas',
            phase0.interpolate(ys=[230, 220], nodes='control_input'),
            units='kn')

    p.set_val('traj.phase0.controls:fltcond|vs',
            phase0.interpolate(ys=[2300, 600], nodes='control_input'),
            units='ft/min')

    p.set_val('traj.phase0.controls:throttle', 
            phase0.interpolate(ys=[0.4, 0.8], nodes='control_input'),
            units=None)
    # p.set_val('traj.phase1.states:fltcond|h',
    #         phase1.interpolate(ys=[25000, 27000], nodes='state_input'),
    #         units='ft')

    # p.set_val('traj.phase1.states:range',
    #         phase1.interpolate(ys=[50, 60], nodes='state_input'),
    #         units='km')

    # p.set_val('traj.phase1.states:fuel_used',
    #         phase1.interpolate(ys=[500, 1000], nodes='state_input'),
    #         units='kg')

    # p.set_val('traj.phase1.controls:fltcond|Ueas',
    #         phase1.interpolate(ys=[180, 180], nodes='control_input'),
    #         units='kn')

    # p.set_val('traj.phase1.controls:fltcond|vs',
    #         phase1.interpolate(ys=[0, 0], nodes='control_input'),
    #         units='ft/s')

    # Run the driver to solve the problem
    # p['traj.phases.phase0.initial_conditions.initial_value:range'] = 100.
    p.run_driver()

    # Check the validity of our results by using scipy.integrate.solve_ivp to
    # integrate the solution.
    sim_out = traj.simulate()

    # Plot the results
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 4.5))

    axes[0].plot(p.get_val('traj.phase0.timeseries.states:range'),
                p.get_val('traj.phase0.timeseries.states:fltcond|h'),
                'ro', label='solution')

    axes[0].plot(sim_out.get_val('traj.phase0.timeseries.states:range'),
                sim_out.get_val('traj.phase0.timeseries.states:fltcond|h'),
                'b-', label='simulation')

    axes[0].set_xlabel('range (km)')
    axes[0].set_ylabel('alt (km)')
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(p.get_val('traj.phase0.timeseries.time'),
                p.get_val('traj.phase0.timeseries.controls:fltcond|Ueas', units='kn'),
                'ro', label='solution')

    axes[1].plot(sim_out.get_val('traj.phase0.timeseries.time'),
                sim_out.get_val('traj.phase0.timeseries.controls:fltcond|Ueas', units='kn'),
                'b-', label='simulation')

    axes[1].set_xlabel('time (s)')
    axes[1].set_ylabel(r'Command speed (kn)')
    axes[1].legend()
    axes[1].grid()

    axes[2].plot(p.get_val('traj.phase0.timeseries.time'),
                p.get_val('traj.phase0.timeseries.controls:fltcond|vs', units='ft/min'),
                'ro', label='solution')

    axes[2].plot(sim_out.get_val('traj.phase0.timeseries.time'),
                sim_out.get_val('traj.phase0.timeseries.controls:fltcond|vs', units='ft/min'),
                'b-', label='simulation')

    axes[2].set_xlabel('time (s)')
    axes[2].set_ylabel(r'Command climb rate (ft/min)')
    axes[2].legend()
    axes[2].grid()

    axes[3].plot(p.get_val('traj.phase0.timeseries.time'),
                p.get_val('traj.phase0.timeseries.controls:throttle', units=None),
                'ro', label='solution')

    axes[3].plot(sim_out.get_val('traj.phase0.timeseries.time'),
                sim_out.get_val('traj.phase0.timeseries.controls:throttle', units=None),
                'b-', label='simulation')

    axes[3].set_xlabel('time (s)')
    axes[3].set_ylabel(r'Throttle')
    axes[3].legend()
    axes[3].grid()

    plt.show()
    # p.check_partials(compact_print=True)
    p.model.list_inputs(print_arrays=False, units=True)