from openmdao.api import IndepVarComp, Group, BalanceComp
from openconcept.analysis.performance.solver_phases import BFLImplicitSolve, GroundRollPhase, RotationPhase, RobustRotationPhase, SteadyFlightPhase, ClimbAnglePhase

class ThreePhaseMissionOnly(Group):
    """
    This analysis group is set up to compute all the major parameters
    of a fixed wing mission, including climb, cruise, and descent.

    To use this analysis, pass in an aircraft model following OpenConcept interface.
    Namely, the model should consume the following:
    - flight conditions (fltcond|q/rho/p/T/Utrue/Ueas/...)
    - aircraft design parameters (ac|*)
    - lift coefficient (fltcond|CL; either solved from steady flight or assumed during ground roll)
    - throttle
    - propulsor_failed (value 0 when failed, 1 when not failed)

    and produce top-level outputs:
    - thrust
    - drag
    - weight

    the following parameters need to either be defined as design variables or
    given as top-level analysis outputs from the airplane model:
    - ac|geom|S_ref
    - ac|aero|CL_max_flaps30
    - ac|weights|MTOW


    Inputs
    ------
    ac|* : various
        All relevant airplane design variables to pass to the airplane model
    takeoff|h : float
        Takeoff obstacle clearance height (default 50 ft)
    cruise|h0 : float
        Initial cruise altitude (default 28000 ft)
    payload : float
        Mission payload (default 1000 lbm)
    mission_range : float
        Design range (deault 1250 NM)

    Options
    -------
    aircraft_model : class
        An aircraft model class with the standard OpenConcept interfaces promoted correctly
    num_nodes : int
        Number of analysis points per segment. Higher is more accurate but more expensive
    extra_states : tuple
        Any extra integrated states to connect across the model.
        Format is ('state_var_name', ('segments','to','connect','across'))
    """

    def initialize(self):
        self.options.declare('num_nodes', default=9, desc="Number of points per segment. Needs to be 2N + 1 due to simpson's rule")
        self.options.declare('aircraft_model', default=None, desc="OpenConcept-compliant airplane model")
        self.options.declare('extra_states', default=None, desc="Extra states to connect across mission phases")

    def setup(self):
            nn = self.options['num_nodes']
            acmodelclass = self.options['aircraft_model']

            mp = self.add_subsystem('missionparams',IndepVarComp(),promotes_outputs=['*'])
            mp.add_output('takeoff|h',val=0.,units='ft')
            mp.add_output('cruise|h0',val=28000.,units='ft')
            mp.add_output('mission_range',val=1250.,units='NM')
            mp.add_output('payload',val=1000.,units='lbm')

            # add the climb, cruise, and descent segments
            self.add_subsystem('climb',SteadyFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='climb'),promotes_inputs=['ac|*'])
            # set the climb time such that the specified initial cruise altitude is exactly reached
            self.add_subsystem('climbdt',BalanceComp(name='duration',units='s',eq_units='m',val=120,upper=2000,lower=0,rhs_name='cruise|h0',lhs_name='fltcond|h_final'),promotes_inputs=['cruise|h0'])
            self.connect('climb.fltcond|h_final','climbdt.fltcond|h_final')
            self.connect('climbdt.duration','climb.duration')

            self.add_subsystem('cruise',SteadyFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='cruise'),promotes_inputs=['ac|*'])
            # set the cruise time such that the desired design range is flown by the end of the mission
            self.add_subsystem('cruisedt',BalanceComp(name='duration',units='s',eq_units='m',val=120, upper=25000, lower=0,rhs_name='mission_range',lhs_name='range_final'),promotes_inputs=['mission_range'])
            self.connect('cruisedt.duration','cruise.duration')

            self.add_subsystem('descent',SteadyFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='descent'),promotes_inputs=['ac|*'])
            # set the descent time so that the final altitude is sea level again
            self.add_subsystem('descentdt',BalanceComp(name='duration',units='s',eq_units='m', val=120, upper=2000, lower=0,rhs_name='takeoff|h',lhs_name='fltcond|h_final'),promotes_inputs=['takeoff|h'])
            self.connect('descent.range_final','cruisedt.range_final')
            self.connect('descent.fltcond|h_final','descentdt.fltcond|h_final')
            self.connect('descentdt.duration','descent.duration')

            # connect range, fuel burn, and altitude from the end of each segment to the beginning of the next, in order

            extra_states = self.options['extra_states']
            for extra_state in extra_states:
                state_name = extra_state[0]
                phases = extra_state[1]
                for i in range(len(phases) - 1):
                    from_phase = phases[i]
                    to_phase = phases[i+1]
                    self.connect(from_phase+'.'+state_name+'_final',to_phase+'.'+state_name+'_initial')

class FullMissionAnalysis(Group):
    """
    This analysis group is set up to compute all the major parameters
    of a fixed wing mission, including balanced-field takeoff, climb, cruise, and descent.

    To use this analysis, pass in an aircraft model following OpenConcept interface.
    Namely, the model should consume the following:
    - flight conditions (fltcond|q/rho/p/T/Utrue/Ueas/...)
    - aircraft design parameters (ac|*)
    - lift coefficient (fltcond|CL; either solved from steady flight or assumed during ground roll)
    - throttle
    - propulsor_failed (value 0 when failed, 1 when not failed)

    and produce top-level outputs:
    - thrust
    - drag
    - weight

    the following parameters need to either be defined as design variables or
    given as top-level analysis outputs from the airplane model:
    - ac|geom|S_ref
    - ac|aero|CL_max_flaps30
    - ac|weights|MTOW


    Inputs
    ------
    ac|* : various
        All relevant airplane design variables to pass to the airplane model
    takeoff|h : float
        Takeoff obstacle clearance height (default 50 ft)
    cruise|h0 : float
        Initial cruise altitude (default 28000 ft)
    payload : float
        Mission payload (default 1000 lbm)
    mission_range : float
        Design range (deault 1250 NM)

    Outputs
    -------
    takeoff|v1 : float
        Decision speed

    Options
    -------
    aircraft_model : class
        An aircraft model class with the standard OpenConcept interfaces promoted correctly
    num_nodes : int
        Number of analysis points per segment. Higher is more accurate but more expensive
    extra_states : tuple
        Any extra integrated states to connect across the model.
        Format is ('state_var_name', ('segments','to','connect','across'))
    transition_method : str
        Analysis method to compute distance, altitude, and time during transition
        Default "simplified" is the Raymer circular arc method and is more robust
        Option "ode" is a 2DOF ODE integration method which is arguably just as inaccurate and less robust
    """

    def initialize(self):
        self.options.declare('num_nodes', default=9, desc="Number of points per segment. Needs to be 2N + 1 due to simpson's rule")
        self.options.declare('aircraft_model', default=None, desc="OpenConcept-compliant airplane model")
        self.options.declare('extra_states', default=None, desc="Extra states to connect across mission phases")
        self.options.declare('transition_method', default='simplified', desc="Method to use for computing transition")

    def setup(self):
            nn = self.options['num_nodes']
            acmodelclass = self.options['aircraft_model']
            transition_method = self.options['transition_method']

            # add the four balanced field length takeoff segments and the implicit v1 solver
            # v0v1 - from a rolling start to v1 speed
            # v1vr - from the decision speed to rotation
            # rotate - in the air following rotation in 2DOF
            # v1vr - emergency stopping from v1 to a stop.

            mp = self.add_subsystem('missionparams',IndepVarComp(),promotes_outputs=['*'])
            mp.add_output('takeoff|h',val=0.,units='ft')
            mp.add_output('cruise|h0',val=28000.,units='ft')
            mp.add_output('mission_range',val=1250.,units='NM')
            mp.add_output('payload',val=1000.,units='lbm')

            self.add_subsystem('bfl', BFLImplicitSolve(), promotes_outputs=['takeoff|v1'])
            self.add_subsystem('v0v1', GroundRollPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='v0v1'), promotes_inputs=['ac|*','takeoff|v1'])
            self.add_subsystem('v1vr', GroundRollPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='v1vr'), promotes_inputs=['ac|*'])
            self.connect('takeoff|v1','v1vr.fltcond|Utrue_initial')
            self.connect('v0v1.range_final','v1vr.range_initial')
            if transition_method == 'simplified':
                self.add_subsystem('rotate',RobustRotationPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='rotate'),promotes_inputs=['ac|*'])
            elif transition_method == 'ode':
                self.add_subsystem('rotate',RotationPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='rotate'),promotes_inputs=['ac|*'])
                self.connect('v1vr.fltcond|Utrue_final','rotate.fltcond|Utrue_initial')
            else:
                raise IOError('Invalid option for transition method')
            self.connect('v1vr.range_final','rotate.range_initial')
            self.connect('rotate.range_final','bfl.distance_continue')
            self.connect('v1vr.takeoff|vr','bfl.takeoff|vr')
            self.add_subsystem('v1v0',GroundRollPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='v1v0'), promotes_inputs=['ac|*','takeoff|v1'])
            self.connect('v0v1.range_final','v1v0.range_initial')
            self.connect('v1v0.range_final','bfl.distance_abort')
            self.add_subsystem('engineoutclimb',ClimbAnglePhase(num_nodes=1, aircraft_model=acmodelclass, flight_phase='EngineOutClimbAngle'), promotes_inputs=['ac|*'])

            # add the climb, cruise, and descent segments
            self.add_subsystem('climb',SteadyFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='climb'),promotes_inputs=['ac|*'])
            # set the climb time such that the specified initial cruise altitude is exactly reached
            self.add_subsystem('climbdt',BalanceComp(name='duration',units='s',eq_units='m',val=120,lower=0,upper=3000,rhs_name='cruise|h0',lhs_name='fltcond|h_final'),promotes_inputs=['cruise|h0'])
            self.connect('climb.fltcond|h_final','climbdt.fltcond|h_final')
            self.connect('climbdt.duration','climb.duration')

            self.add_subsystem('cruise',SteadyFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='cruise'),promotes_inputs=['ac|*'])
            # set the cruise time such that the desired design range is flown by the end of the mission
            self.add_subsystem('cruisedt',BalanceComp(name='duration',units='s',eq_units='km',val=120, lower=0,upper=30000,rhs_name='mission_range',lhs_name='range_final'),promotes_inputs=['mission_range'])
            self.connect('cruisedt.duration','cruise.duration')

            self.add_subsystem('descent',SteadyFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='descent'),promotes_inputs=['ac|*'])
            # set the descent time so that the final altitude is sea level again
            self.add_subsystem('descentdt',BalanceComp(name='duration',units='s',eq_units='m', val=120, lower=0,upper=3000,rhs_name='takeoff|h',lhs_name='fltcond|h_final'),promotes_inputs=['takeoff|h'])
            self.connect('descent.range_final','cruisedt.range_final')
            self.connect('descent.fltcond|h_final','descentdt.fltcond|h_final')
            self.connect('descentdt.duration','descent.duration')

            # connect range, fuel burn, and altitude from the end of each segment to the beginning of the next, in order

            extra_states = self.options['extra_states']
            for extra_state in extra_states:
                state_name = extra_state[0]
                phases = extra_state[1]
                for i in range(len(phases) - 1):
                    from_phase = phases[i]
                    to_phase = phases[i+1]
                    self.connect(from_phase+'.'+state_name+'_final',to_phase+'.'+state_name+'_initial')

class CruiseOnly(Group):
    """
    This analysis group is set up to compute all the major parameters
    of a fixed wing mission, including balanced-field takeoff, climb, cruise, and descent.

    To use this analysis, pass in an aircraft model following OpenConcept interface.
    Namely, the model should consume the following:
    - flight conditions (fltcond|q/rho/p/T/Utrue/Ueas/...)
    - aircraft design parameters (ac|*)
    - lift coefficient (fltcond|CL; either solved from steady flight or assumed during ground roll)
    - throttle
    - propulsor_failed (value 0 when failed, 1 when not failed)

    and produce top-level outputs:
    - thrust
    - drag
    - weight

    the following parameters need to either be defined as design variables or
    given as top-level analysis outputs from the airplane model:
    - ac|geom|S_ref
    - ac|aero|CL_max_flaps30
    - ac|weights|MTOW


    Inputs
    ------
    ac|* : various
        All relevant airplane design variables to pass to the airplane model
    takeoff|h : float
        Takeoff obstacle clearance height (default 50 ft)
    cruise|h0 : float
        Initial cruise altitude (default 28000 ft)
    payload : float
        Mission payload (default 1000 lbm)
    mission_range : float
        Design range (deault 1250 NM)

    Outputs
    -------
    takeoff|v1 : float
        Decision speed

    Options
    -------
    aircraft_model : class
        An aircraft model class with the standard OpenConcept interfaces promoted correctly
    num_nodes : int
        Number of analysis points per segment. Higher is more accurate but more expensive
    extra_states : tuple
        Any extra integrated states to connect across the model.
        Format is ('state_var_name', ('segments','to','connect','across'))
    """

    def initialize(self):
        self.options.declare('num_nodes', default=9, desc="Number of points per segment. Needs to be 2N + 1 due to simpson's rule")
        self.options.declare('aircraft_model', default=None, desc="OpenConcept-compliant airplane model")
        self.options.declare('extra_states', default=None, desc="Extra states to connect across mission phases")

    def setup(self):
            nn = self.options['num_nodes']
            acmodelclass = self.options['aircraft_model']

            # add the four balanced field length takeoff segments and the implicit v1 solver
            # v0v1 - from a rolling start to v1 speed
            # v1vr - from the decision speed to rotation
            # rotate - in the air following rotation in 2DOF
            # v1vr - emergency stopping from v1 to a stop.

            mp = self.add_subsystem('missionparams',IndepVarComp(),promotes_outputs=['*'])
            mp.add_output('takeoff|h',val=0.,units='ft')
            mp.add_output('cruise|h0',val=28000.,units='ft')
            mp.add_output('mission_range',val=1250.,units='NM')
            mp.add_output('payload',val=1000.,units='lbm')
            mp.add_output('cruise|duration',val=1.,units='h')

            self.add_subsystem('cruise',SteadyFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='cruise'),promotes_inputs=['ac|*'])
            # set the cruise time such that the desired design range is flown by the end of the mission
            self.connect('cruise|duration','cruise.duration')
            self.connect('cruise|h0','cruise.fltcond|h_initial')


class BalancedFieldTakeoff(Group):
    """
    This analysis group is set up to compute balanced-field takeoff only

    To use this analysis, pass in an aircraft model following OpenConcept interface.
    Namely, the model should consume the following:
    - flight conditions (fltcond|q/rho/p/T/Utrue/Ueas/...)
    - aircraft design parameters (ac|*)
    - lift coefficient (fltcond|CL; either solved from steady flight or assumed during ground roll)
    - throttle
    - propulsor_failed (value 0 when failed, 1 when not failed)

    and produce top-level outputs:
    - thrust
    - drag
    - weight

    the following parameters need to either be defined as design variables or
    given as top-level analysis outputs from the airplane model:
    - ac|geom|S_ref
    - ac|aero|CL_max_flaps30
    - ac|weights|MTOW


    Inputs
    ------
    ac|* : various
        All relevant airplane design variables to pass to the airplane model
    takeoff|h : float
        Takeoff obstacle clearance height (default 50 ft)
    cruise|h0 : float
        Initial cruise altitude (default 28000 ft)
    payload : float
        Mission payload (default 1000 lbm)
    mission_range : float
        Design range (deault 1250 NM)

    Outputs
    -------
    takeoff|v1 : float
        Decision speed

    Options
    -------
    aircraft_model : class
        An aircraft model class with the standard OpenConcept interfaces promoted correctly
    num_nodes : int
        Number of analysis points per segment. Higher is more accurate but more expensive
    extra_states : tuple
        Any extra integrated states to connect across the model.
        Format is ('state_var_name', ('segments','to','connect','across'))
    """

    def initialize(self):
        self.options.declare('num_nodes', default=9, desc="Number of points per segment. Needs to be 2N + 1 due to simpson's rule")
        self.options.declare('aircraft_model', default=None, desc="OpenConcept-compliant airplane model")
        self.options.declare('extra_states', default=None, desc="Extra states to connect across mission phases")

    def setup(self):
            nn = self.options['num_nodes']
            acmodelclass = self.options['aircraft_model']

            # add the four balanced field length takeoff segments and the implicit v1 solver
            # v0v1 - from a rolling start to v1 speed
            # v1vr - from the decision speed to rotation
            # rotate - in the air following rotation in 2DOF
            # v1vr - emergency stopping from v1 to a stop.

            mp = self.add_subsystem('missionparams',IndepVarComp(),promotes_outputs=['*'])
            mp.add_output('takeoff|h',val=0.,units='ft')

            self.add_subsystem('bfl', BFLImplicitSolve(), promotes_outputs=['takeoff|v1'])
            self.add_subsystem('v0v1', GroundRollPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='v0v1'), promotes_inputs=['ac|*','takeoff|v1'])
            self.add_subsystem('v1vr', GroundRollPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='v1vr'), promotes_inputs=['ac|*'])
            self.connect('takeoff|v1','v1vr.fltcond|Utrue_initial')
            self.connect('v0v1.range_final','v1vr.range_initial')
            self.connect('v0v1.fuel_used_final','v1vr.fuel_used_initial')
            self.add_subsystem('rotate',RotationPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='rotate'),promotes_inputs=['ac|*'])
            self.connect('v1vr.range_final','rotate.range_initial')
            self.connect('v1vr.fltcond|Utrue_final','rotate.fltcond|Utrue_initial')
            self.connect('v1vr.fuel_used_final','rotate.fuel_used_initial')
            self.connect('rotate.range_final','bfl.distance_continue')
            self.connect('v1vr.takeoff|vr','bfl.takeoff|vr')
            self.add_subsystem('v1v0',GroundRollPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='v1v0'), promotes_inputs=['ac|*','takeoff|v1'])
            self.connect('v0v1.range_final','v1v0.range_initial')
            self.connect('v1v0.range_final','bfl.distance_abort')

            # connect range, fuel burn, and altitude from the end of each segment to the beginning of the next, in order

            extra_states = self.options['extra_states']
            if extra_states is not None:
                for extra_state in extra_states:
                    state_name = extra_state[0]
                    phases = extra_state[1]
                    for i in range(len(phases) - 1):
                        from_phase = phases[i]
                        to_phase = phases[i+1]
                        self.connect(from_phase+'.'+state_name+'_final',to_phase+'.'+state_name+'_initial')