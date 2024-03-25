import openmdao.api as om
from openconcept.utilities import DVLabel
from .phases import (
    BFLImplicitSolve,
    GroundRollPhase,
    RotationPhase,
    RobustRotationPhase,
    ClimbAnglePhase,
    SteadyFlightPhase,
)
from .mission_groups import TrajectoryGroup


class FullMissionWithReserve(TrajectoryGroup):
    """
    A combination of the FullMissionAnalysis and MissionWithReserve profiles (see their documentation)
    for more details. This combination includes the balanced field length simulation from FullMissionAnalysis,
    climb, cruise, and descent phases, and the reserve phases from MissionWithReserve.

    This acts as the main group that includes all the features of FullMissionAnalysis, BasicMission, and
    MissionWithReserve profiles so that those can inherit from this without duplicating code.
    """

    def __init__(self, include_takeoff=True, include_reserve=True, **kwargs):
        self.include_takeoff = include_takeoff  # include flight phases associated with takeoff
        self.include_reserve = include_reserve  # include flight phases associated with reserve mission
        super().__init__(**kwargs)

    def initialize(self):
        self.options.declare(
            "num_nodes", default=9, desc="Number of points per phase. Needs to be 2N + 1 due to simpson's rule"
        )
        self.options.declare("aircraft_model", default=None, desc="OpenConcept-compliant airplane model")
        if self.include_takeoff:
            self.options.declare(
                "transition_method", default="simplified", desc="Method to use for computing transition"
            )
        if not self.include_reserve and not self.include_takeoff:  # BasicMission
            self.options.declare("include_ground_roll", default=False, desc="Whether to include groundroll phase")

    def setup(self):
        nn = self.options["num_nodes"]
        acmodelclass = self.options["aircraft_model"]

        mp = self.add_subsystem("missionparams", om.IndepVarComp(), promotes_outputs=["*"])
        mp.add_output("takeoff|h", val=0.0, units="ft")
        mp.add_output("cruise|h0", val=28000.0, units="ft")
        mp.add_output("mission_range", val=1250.0, units="NM")
        mp.add_output("payload", val=1000.0, units="lbm")
        if self.include_reserve:
            mp.add_output("reserve_range", val=200.0, units="NM")
            mp.add_output("reserve|h0", val=25000.0, units="ft")
            mp.add_output("loiter|h0", val=1500.0, units="ft")
            mp.add_output("loiter_duration", val=30.0 * 60.0, units="s")
        if not self.include_reserve and not self.include_takeoff:  # BasicMission
            mp.add_output("takeoff|v2", val=150.0, units="kn")

        if self.include_takeoff:
            # add the four balanced field length takeoff phases and the implicit v1 solver
            # v0v1 - from a rolling start to v1 speed
            # v1vr - from the decision speed to rotation
            # rotate - in the air following rotation in 2DOF
            # v1v0 - emergency stopping from v1 to a stop.

            self.add_subsystem("bfl", BFLImplicitSolve(), promotes_outputs=["takeoff|v1"])
            v0v1 = self.add_subsystem(
                "v0v1",
                GroundRollPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase="v0v1"),
                promotes_inputs=["ac|*", "takeoff|v1"],
            )
            v1vr = self.add_subsystem(
                "v1vr",
                GroundRollPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase="v1vr"),
                promotes_inputs=["ac|*"],
            )
            self.connect("takeoff|v1", "v1vr.fltcond|Utrue_initial")
            self.connect("v0v1.range_final", "v1vr.range_initial")
            if self.options["transition_method"] == "simplified":
                rotate = self.add_subsystem(
                    "rotate",
                    RobustRotationPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase="rotate"),
                    promotes_inputs=["ac|*"],
                )
            elif self.options["transition_method"] == "ode":
                rotate = self.add_subsystem(
                    "rotate",
                    RotationPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase="rotate"),
                    promotes_inputs=["ac|*"],
                )
                self.connect("v1vr.fltcond|Utrue_final", "rotate.fltcond|Utrue_initial")
            else:
                raise IOError("Invalid option for transition method")
            self.connect("v1vr.range_final", "rotate.range_initial")
            self.connect("rotate.range_final", "bfl.distance_continue")
            self.connect("v1vr.takeoff|vr", "bfl.takeoff|vr")
            v1v0 = self.add_subsystem(
                "v1v0",
                GroundRollPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase="v1v0"),
                promotes_inputs=["ac|*", "takeoff|v1"],
            )
            self.connect("v0v1.range_final", "v1v0.range_initial")
            self.connect("v1v0.range_final", "bfl.distance_abort")
            self.add_subsystem(
                "engineoutclimb",
                ClimbAnglePhase(num_nodes=1, aircraft_model=acmodelclass, flight_phase="EngineOutClimbAngle"),
                promotes_inputs=["ac|*"],
            )
        elif not self.include_reserve:  # BasicMission
            if self.options["include_ground_roll"]:
                mp.add_output("takeoff|v0", val=4.0, units="kn")
                ground_roll = self.add_subsystem(
                    "groundroll",
                    GroundRollPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase="v0v1"),
                    promotes_inputs=["ac|*"],
                )
                self.connect("takeoff|v2", "groundroll.takeoff|v1")

        # add the climb, cruise, and descent phases
        climb = self.add_subsystem(
            "climb",
            SteadyFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase="climb"),
            promotes_inputs=["ac|*"],
        )
        # set the climb time such that the specified initial cruise altitude is exactly reached
        climb.add_subsystem(
            "climbdt",
            om.BalanceComp(
                name="duration",
                units="s",
                eq_units="m",
                val=1200,
                upper=1e4,
                lower=0,
                rhs_name="cruise|h0",
                lhs_name="fltcond|h_final",
            ),
            promotes_outputs=["duration"],
        )
        climb.connect("ode_integ_phase.fltcond|h_final", "climbdt.fltcond|h_final")
        self.connect("cruise|h0", "climb.climbdt.cruise|h0")
        if not self.include_reserve and not self.include_takeoff:  # BasicMission
            self.connect("takeoff|h", "climb.ode_integ_phase.fltcond|h_initial")

        cruise = self.add_subsystem(
            "cruise",
            SteadyFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase="cruise"),
            promotes_inputs=["ac|*"],
        )
        # set the cruise time such that the desired design range is flown by the end of the mission
        cruise.add_subsystem(
            "cruisedt",
            om.BalanceComp(
                name="duration",
                units="s",
                eq_units="km",
                val=1e4,
                upper=1e5,
                lower=0,
                rhs_name="mission_range",
                lhs_name="range_final",
            ),
            promotes_outputs=["duration"],
        )
        self.connect("mission_range", "cruise.cruisedt.mission_range")
        descent = self.add_subsystem(
            "descent",
            SteadyFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase="descent"),
            promotes_inputs=["ac|*"],
        )
        # set the descent time so that the final altitude is sea level again
        descent.add_subsystem(
            "descentdt",
            om.BalanceComp(
                name="duration",
                units="s",
                eq_units="m",
                val=1200,
                upper=1e4,
                lower=0,
                rhs_name="takeoff|h",
                lhs_name="fltcond|h_final",
            ),
            promotes_outputs=["duration"],
        )
        self.connect("descent.ode_integ_phase.range_final", "cruise.cruisedt.range_final")
        self.connect("takeoff|h", "descent.descentdt.takeoff|h")
        descent.connect("ode_integ_phase.fltcond|h_final", "descentdt.fltcond|h_final")

        # add the phases for the reserve mission
        if self.include_reserve:
            reserve_climb = self.add_subsystem(
                "reserve_climb",
                SteadyFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase="reserve_climb"),
                promotes_inputs=["ac|*"],
            )
            # set the climb time such that the specified initial cruise altitude is exactly reached
            reserve_climb.add_subsystem(
                "reserve_climbdt",
                om.BalanceComp(
                    name="duration",
                    units="s",
                    eq_units="m",
                    val=120,
                    upper=2000,
                    lower=0,
                    rhs_name="reserve|h0",
                    lhs_name="fltcond|h_final",
                ),
                promotes_outputs=["duration"],
            )
            reserve_climb.connect("ode_integ_phase.fltcond|h_final", "reserve_climbdt.fltcond|h_final")
            self.connect("reserve|h0", "reserve_climb.reserve_climbdt.reserve|h0")

            reserve_cruise = self.add_subsystem(
                "reserve_cruise",
                SteadyFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase="reserve_cruise"),
                promotes_inputs=["ac|*"],
            )
            # set the reserve_cruise time such that the desired design range is flown by the end of the mission
            reserve_cruise.add_subsystem(
                "reserve_cruisedt",
                om.BalanceComp(
                    name="duration",
                    units="s",
                    eq_units="m",
                    val=120,
                    upper=25000,
                    lower=0,
                    rhs_name="reserve_range",
                    lhs_name="range_final",
                ),
                promotes_outputs=["duration"],
            )
            self.connect("reserve_range", "reserve_cruise.reserve_cruisedt.reserve_range")

            reserve_descent = self.add_subsystem(
                "reserve_descent",
                SteadyFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase="reserve_descent"),
                promotes_inputs=["ac|*"],
            )
            # set the reserve_descent time so that the final altitude is sea level again
            reserve_descent.add_subsystem(
                "reserve_descentdt",
                om.BalanceComp(
                    name="duration",
                    units="s",
                    eq_units="m",
                    val=120,
                    upper=8000,
                    lower=0,
                    rhs_name="takeoff|h",
                    lhs_name="fltcond|h_final",
                ),
                promotes_outputs=["duration"],
            )
            reserve_descent.connect("ode_integ_phase.fltcond|h_final", "reserve_descentdt.fltcond|h_final")
            self.connect("takeoff|h", "reserve_descent.reserve_descentdt.takeoff|h")

            reserverange = om.ExecComp(
                "reserverange=rangef-rangeo",
                reserverange={"val": 100.0, "units": "NM"},
                rangeo={"val": 0.0, "units": "NM"},
                rangef={"val": 100.0, "units": "NM"},
            )
            self.add_subsystem("resrange", reserverange)
            self.connect("descent.ode_integ_phase.range_final", "resrange.rangeo")
            self.connect("reserve_descent.ode_integ_phase.range_final", "resrange.rangef")
            self.connect("resrange.reserverange", "reserve_cruise.reserve_cruisedt.range_final")
            # self.connect('reserve_descent.range_final', 'reserve_cruisedt.range_final')

            loiter = self.add_subsystem(
                "loiter",
                SteadyFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase="loiter"),
                promotes_inputs=["ac|*"],
            )
            dvlist = [["duration_in", "duration", 300, "s"]]
            loiter.add_subsystem("loiter_dt", DVLabel(dvlist), promotes_inputs=["*"], promotes_outputs=["*"])
            self.connect("loiter|h0", "loiter.ode_integ_phase.fltcond|h_initial")
            self.connect("loiter_duration", "loiter.duration_in")

        if self.include_takeoff:
            self.link_phases(v0v1, v1vr, states_to_skip=["fltcond|Utrue", "range"])
            self.link_phases(v1vr, rotate, states_to_skip=["fltcond|Utrue", "range"])
            self.link_phases(v0v1, v1v0, states_to_skip=["fltcond|Utrue", "range"])
            self.link_phases(rotate, climb)
            self.connect("rotate.range_final", "climb.ode_integ_phase.range_initial")
            self.connect("rotate.fltcond|h_final", "climb.ode_integ_phase.fltcond|h_initial")
        elif not self.include_reserve:  # BasicMission
            if self.options["include_ground_roll"]:
                self.link_phases(ground_roll, climb, states_to_skip=["fltcond|h"])
        self.link_phases(climb, cruise)
        self.link_phases(cruise, descent)
        if self.include_reserve:
            self.link_phases(descent, reserve_climb, states_to_skip=["ode_integ_phase.fltcond|h"])
            self.link_phases(reserve_climb, reserve_cruise)
            self.link_phases(reserve_cruise, reserve_descent)
            self.link_phases(reserve_descent, loiter, states_to_skip=["ode_integ_phase.fltcond|h"])


class MissionWithReserve(FullMissionWithReserve):
    """
    This analysis group is set up to compute all the major parameters
    of a fixed wing mission, including climb, cruise, and descent as well as Part 25 reserve fuel phases.
    The 5% of block fuel is not accounted for here.

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
        Takeoff and landing altitude (default 0 ft)
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
        Number of analysis points per phase. Higher is more accurate but more expensive
    """

    def __init__(self, **kwargs):
        super().__init__(include_takeoff=False, include_reserve=True, **kwargs)


class BasicMission(FullMissionWithReserve):
    """
    This analysis group is set up to compute all the major parameters
    of a fixed wing mission, including climb, cruise, and descent but no Part 25 reserves
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
        Takeoff and landing altitude (default 0 ft). However, if the ground roll is
        included it will always occur at 0 ft unless its fltcond|h is specifically set.
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
        Number of analysis points per phase. Higher is more accurate but more expensive
    """

    def __init__(self, **kwargs):
        super().__init__(include_takeoff=False, include_reserve=False, **kwargs)


class FullMissionAnalysis(FullMissionWithReserve):
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
        WARNING: This parameter will set the landing altitude, but takeoff
        altitude will always be 0 ft unless specifically set in each takeoff
        phase's fltcond|h value. However, even if you change this value, the
        climb phase will begin at the rotation phase's obstacle height. Also,
        the rotation phase does it's own thing (see the source to understand).
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
        Number of analysis points per phase. Higher is more accurate but more expensive
    transition_method : str
        Analysis method to compute distance, altitude, and time during transition
        Default "simplified" is the Raymer circular arc method and is more robust
        Option "ode" is a 2DOF ODE integration method which is arguably just as inaccurate and less robust
    """

    def __init__(self, **kwargs):
        super().__init__(include_takeoff=True, include_reserve=False, **kwargs)
