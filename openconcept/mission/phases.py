from openmdao.api import Group, ExplicitComponent, IndepVarComp, BalanceComp, ImplicitComponent
from .mission_groups import PhaseGroup
from openconcept.atmospherics import ComputeAtmosphericProperties
from openconcept.aerodynamics import Lift, StallSpeed
from openconcept.utilities import ElementMultiplyDivideComp, AddSubtractComp, Integrator, LinearInterpolator
from openconcept.utilities.constants import GRAV_CONST
import numpy as np


class ClimbAngleComp(ExplicitComponent):
    """
    Computes steady climb angle based on excess thrust.

    This is a helper function
    and shouldn't be instantiated in the top-level model directly.

    Inputs
    ------
    drag : float
        Aircraft drag at v2 (climb out) flight condition (scalar, N)
    weight : float
        Takeoff weight (scalar, kg)
    thrust : float
        Thrust at the v2 (climb out) flight condition (scalar, N)

    Outputs
    -------
    gamma : float
        Climb out flight path angle (scalar, rad)

    Options
    -------
    num_nodes : int
        Number of points to run
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1)

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("drag", units="N", shape=(nn,))
        self.add_input("weight", units="kg", shape=(nn,))
        self.add_input("thrust", units="N", shape=(nn,))
        self.add_output("gamma", units="rad", shape=(nn,))

        self.declare_partials(["gamma"], ["weight", "thrust", "drag"], cols=np.arange(0, nn), rows=np.arange(0, nn))

    def compute(self, inputs, outputs):
        outputs["gamma"] = np.arcsin((inputs["thrust"] - inputs["drag"]) / inputs["weight"] / GRAV_CONST)

    def compute_partials(self, inputs, J):
        interior_qty = (inputs["thrust"] - inputs["drag"]) / inputs["weight"] / GRAV_CONST
        d_arcsin = 1 / np.sqrt(1 - interior_qty**2)
        J["gamma", "thrust"] = d_arcsin / inputs["weight"] / GRAV_CONST
        J["gamma", "drag"] = -d_arcsin / inputs["weight"] / GRAV_CONST
        J["gamma", "weight"] = -d_arcsin * (inputs["thrust"] - inputs["drag"]) / inputs["weight"] ** 2 / GRAV_CONST


class FlipVectorComp(ExplicitComponent):
    """
    Reverses the order of an OpenMDAO vector

    This is a helper function
    and shouldn't be instantiated in the top-level model directly.

    Inputs
    ------
    vec_in : float
        Incoming vector in forward order

    Outputs
    -------
    vec_out : float
        Reversed order version of vec_in

    Options
    -------
    num_nodes : int
        Number of points to run
    negative : boolean
        Whether to apply a negative scaler. Default False preserves vector values.
        True returns all values with negative sign.
    units : string or None
        Units for vec_in and vec_out (Default None)
        Specify as an OpenMDAO unit string (e.g. 'kg')
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1)
        self.options.declare("negative", default=False)
        self.options.declare("units", default=None)

    def setup(self):
        nn = self.options["num_nodes"]
        units = self.options["units"]
        self.add_input("vec_in", units=units, shape=(nn,))
        self.add_output("vec_out", units=units, shape=(nn,))
        negative = self.options["negative"]
        if negative:
            scaler = -1
        else:
            scaler = 1
        self.declare_partials(
            ["vec_out"],
            ["vec_in"],
            rows=np.arange(nn - 1, -1, -1),
            cols=np.arange(0, nn, 1),
            val=scaler * np.ones((nn,)),
        )

    def compute(self, inputs, outputs):
        negative = self.options["negative"]
        if negative:
            scaler = -1
        else:
            scaler = 1
        outputs["vec_out"] = scaler * np.flip(inputs["vec_in"], 0)


class BFLImplicitSolve(ImplicitComponent):
    """
    Computes a residual equation so Newton solver can set v1 to analyze balanced field length

    This residual is equal to zero if:
        - The rejected takeoff and engine-out takeoff distances are equal, or:
        - V1 is equal to VR and the engine out takeoff distance is longer than the RTO distance

    Since this is a discontinous function, the partial derivatives are written in a special way
    to 'coax' the V1 value into the right setting with a Newton step. It's kind of a hack.

    Inputs
    ------
    distance_continue : float
        Engine-out takeoff distance (scalar, m)
    distance_abort : float
        Distance to full-stop when takeoff is rejected at V1 (scalar, m)
    takeoff|vr : float
        Rotation speed (scalar, m/s)

    Outputs
    -------
    takeoff|v1 : float
        Decision speed (scalar, m/s)

    """

    def setup(self):
        self.add_input("distance_continue", units="m")
        self.add_input("distance_abort", units="m")
        self.add_input("takeoff|vr", units="m/s")
        self.add_output("takeoff|v1", units="m/s", val=20, lower=10, upper=150)
        self.declare_partials("takeoff|v1", ["distance_continue", "distance_abort", "takeoff|v1", "takeoff|vr"])

    def apply_nonlinear(self, inputs, outputs, residuals):
        speedtol = 1e-1
        disttol = 0
        # force the decision speed to zero
        if inputs["takeoff|vr"] < outputs["takeoff|v1"] + speedtol:
            residuals["takeoff|v1"] = inputs["takeoff|vr"] - outputs["takeoff|v1"]
        else:
            residuals["takeoff|v1"] = inputs["distance_continue"] - inputs["distance_abort"]

        # if you are within vtol on the correct side but the stopping distance bigger, use the regular mode
        if (
            inputs["takeoff|vr"] >= outputs["takeoff|v1"]
            and inputs["takeoff|vr"] - outputs["takeoff|v1"] < speedtol
            and (inputs["distance_abort"] - inputs["distance_continue"]) > disttol
        ):
            residuals["takeoff|v1"] = inputs["distance_continue"] - inputs["distance_abort"]

    def linearize(self, inputs, outputs, partials):
        speedtol = 1e-1
        disttol = 0

        if inputs["takeoff|vr"] < outputs["takeoff|v1"] + speedtol:
            partials["takeoff|v1", "distance_continue"] = 0
            partials["takeoff|v1", "distance_abort"] = 0
            partials["takeoff|v1", "takeoff|vr"] = 1
            partials["takeoff|v1", "takeoff|v1"] = -1
        else:
            partials["takeoff|v1", "distance_continue"] = 1
            partials["takeoff|v1", "distance_abort"] = -1
            partials["takeoff|v1", "takeoff|vr"] = 0
            partials["takeoff|v1", "takeoff|v1"] = 0

        if (
            inputs["takeoff|vr"] >= outputs["takeoff|v1"]
            and inputs["takeoff|vr"] - outputs["takeoff|v1"] < speedtol
            and (inputs["distance_abort"] - inputs["distance_continue"]) > disttol
        ):
            partials["takeoff|v1", "distance_continue"] = 1
            partials["takeoff|v1", "distance_abort"] = -1
            partials["takeoff|v1", "takeoff|vr"] = 0
            partials["takeoff|v1", "takeoff|v1"] = 0


class Groundspeeds(ExplicitComponent):
    """
    Computes groundspeed for vectorial true airspeed and true vertical speed.

    This is a helper function for the main mission analysis routines
    and shouldn't be instantiated directly.

    Inputs
    ------
    fltcond|vs : float
        Vertical speed for all mission phases (vector, m/s)
    fltcond|Utrue : float
        True airspeed for all mission phases (vector, m/s)

    Outputs
    -------
    fltcond|groundspeed : float
        True groundspeed for all mission phases (vector, m/s)
    fltcond|cosgamma : float
        Cosine of the flght path angle for all mission phases (vector, dimensionless)
    fltcond|singamma : float
        Sine of the flight path angle for all mission phases (vector, dimensionless)

    Options
    -------
    num_nodes : int
        Number of points to run
    """

    def initialize(self):
        self.options.declare(
            "num_nodes",
            default=1,
            desc="Number of Simpson intervals to use per seg (eg. climb, cruise, descend). Number of analysis points is 2N+1",
        )

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("fltcond|vs", units="m/s", shape=(nn,))
        self.add_input("fltcond|Utrue", units="m/s", shape=(nn,))
        self.add_output("fltcond|groundspeed", units="m/s", shape=(nn,))
        self.add_output("fltcond|cosgamma", shape=(nn,), desc="Cosine of the flight path angle")
        self.add_output("fltcond|singamma", shape=(nn,), desc="sin of the flight path angle")
        self.declare_partials(
            ["fltcond|groundspeed", "fltcond|cosgamma", "fltcond|singamma"],
            ["fltcond|vs", "fltcond|Utrue"],
            rows=range(nn),
            cols=range(nn),
        )

    def compute(self, inputs, outputs):
        # compute the groundspeed on climb and desc
        inside = inputs["fltcond|Utrue"] ** 2 - inputs["fltcond|vs"] ** 2
        groundspeed = np.sqrt(inside)
        groundspeed_fixed = np.sqrt(np.where(np.less(inside, 0.0), 0.01, inside))
        # groundspeed =  np.sqrt(inputs['fltcond|Utrue']**2-inputs['fltcond|vs']**2)
        # groundspeed_fixed= np.where(np.isnan(groundspeed),0,groundspeed)
        outputs["fltcond|groundspeed"] = groundspeed_fixed
        outputs["fltcond|singamma"] = np.where(np.isnan(groundspeed), 1, inputs["fltcond|vs"] / inputs["fltcond|Utrue"])
        outputs["fltcond|cosgamma"] = groundspeed_fixed / inputs["fltcond|Utrue"]

    def compute_partials(self, inputs, J):
        inside = inputs["fltcond|Utrue"] ** 2 - inputs["fltcond|vs"] ** 2
        groundspeed = np.sqrt(inside)
        groundspeed_fixed = np.sqrt(np.where(np.less(inside, 0.0), 0.01, inside))
        J["fltcond|groundspeed", "fltcond|vs"] = np.where(
            np.isnan(groundspeed), 0, (1 / 2) / groundspeed_fixed * (-2) * inputs["fltcond|vs"]
        )
        J["fltcond|groundspeed", "fltcond|Utrue"] = np.where(
            np.isnan(groundspeed), 0, (1 / 2) / groundspeed_fixed * 2 * inputs["fltcond|Utrue"]
        )
        J["fltcond|singamma", "fltcond|vs"] = np.where(np.isnan(groundspeed), 0, 1 / inputs["fltcond|Utrue"])
        J["fltcond|singamma", "fltcond|Utrue"] = np.where(
            np.isnan(groundspeed), 0, -inputs["fltcond|vs"] / inputs["fltcond|Utrue"] ** 2
        )
        J["fltcond|cosgamma", "fltcond|vs"] = J["fltcond|groundspeed", "fltcond|vs"] / inputs["fltcond|Utrue"]
        J["fltcond|cosgamma", "fltcond|Utrue"] = (
            J["fltcond|groundspeed", "fltcond|Utrue"] * inputs["fltcond|Utrue"] - groundspeed_fixed
        ) / inputs["fltcond|Utrue"] ** 2


class HorizontalAcceleration(ExplicitComponent):
    """
    Computes acceleration during takeoff run and effectively forms the T-D residual.

    Inputs
    ------
    weight : float
        Aircraft weight (scalar, kg)
    drag : float
        Aircraft drag at each analysis point (vector, N)
    lift : float
        Aircraft lift at each analysis point (vector, N)
    thrust : float
        Thrust at each TO analysis point (vector, N)
    fltcond|singamma : float
        The sine of the flight path angle gamma (vector, dimensionless)
    braking : float
        Effective rolling friction multiplier at each point (vector, dimensionless)

    Outputs
    -------
    accel_horiz : float
        Aircraft horizontal acceleration (vector, m/s**2)

    Options
    -------
    num_nodes : int
        Number of analysis points to run
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1)

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("weight", units="kg", shape=(nn,))
        self.add_input("drag", units="N", shape=(nn,))
        self.add_input("lift", units="N", shape=(nn,))
        self.add_input("thrust", units="N", shape=(nn,))
        self.add_input("fltcond|singamma", shape=(nn,))
        self.add_input("braking", shape=(nn,))

        self.add_output("accel_horiz", units="m/s**2", shape=(nn,))
        arange = np.arange(nn)
        self.declare_partials(
            ["accel_horiz"], ["weight", "drag", "lift", "thrust", "braking"], rows=arange, cols=arange
        )
        self.declare_partials(
            ["accel_horiz"], ["fltcond|singamma"], rows=arange, cols=arange, val=-GRAV_CONST * np.ones((nn,))
        )

    def compute(self, inputs, outputs):
        m = inputs["weight"]
        floor_vec = np.where(np.less((GRAV_CONST - inputs["lift"] / m), 0.0), 0.0, 1.0)
        accel = (
            inputs["thrust"] / m
            - inputs["drag"] / m
            - floor_vec * inputs["braking"] * (GRAV_CONST - inputs["lift"] / m)
            - GRAV_CONST * inputs["fltcond|singamma"]
        )
        outputs["accel_horiz"] = accel

    def compute_partials(self, inputs, J):
        m = inputs["weight"]
        floor_vec = np.where(np.less((GRAV_CONST - inputs["lift"] / m), 0.0), 0.0, 1.0)
        J["accel_horiz", "thrust"] = 1 / m
        J["accel_horiz", "drag"] = -1 / m
        J["accel_horiz", "braking"] = -floor_vec * (GRAV_CONST - inputs["lift"] / m)
        J["accel_horiz", "lift"] = floor_vec * inputs["braking"] / m
        J["accel_horiz", "weight"] = (
            inputs["drag"] - inputs["thrust"] - floor_vec * inputs["braking"] * inputs["lift"]
        ) / m**2


class VerticalAcceleration(ExplicitComponent):
    """
    Computes acceleration during takeoff run in the vertical plane.
    Only used during full unsteady takeoff performance analysis due to stability issues

    Inputs
    ------
    weight : float
        Aircraft weight (scalar, kg)
    drag : float
        Aircraft drag at each analysis point (vector, N)
    lift : float
        Aircraft lift at each analysis point (vector, N)
    thrust : float
        Thrust at each TO analysis point (vector, N)
    fltcond|singamma : float
        The sine of the flight path angle gamma (vector, dimensionless)
    fltcond|cosgamma : float
        The sine of the flight path angle gamma (vector, dimensionless)

    Outputs
    -------
    accel_vert : float
        Aircraft horizontal acceleration (vector, m/s**2)

    Options
    -------
    num_nodes : int
        Number of analysis points to run
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1)

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("weight", units="kg", shape=(nn,))
        self.add_input("drag", units="N", shape=(nn,))
        self.add_input("lift", units="N", shape=(nn,))
        self.add_input("thrust", units="N", shape=(nn,))
        self.add_input("fltcond|singamma", shape=(nn,))
        self.add_input("fltcond|cosgamma", shape=(nn,))

        self.add_output("accel_vert", units="m/s**2", shape=(nn,), upper=2.5 * GRAV_CONST, lower=-1 * GRAV_CONST)
        arange = np.arange(nn)
        self.declare_partials(
            ["accel_vert"],
            ["weight", "drag", "lift", "thrust", "fltcond|singamma", "fltcond|cosgamma"],
            rows=arange,
            cols=arange,
        )

    def compute(self, inputs, outputs):
        cosg = inputs["fltcond|cosgamma"]
        sing = inputs["fltcond|singamma"]
        accel = (
            inputs["lift"] * cosg + (inputs["thrust"] - inputs["drag"]) * sing - GRAV_CONST * inputs["weight"]
        ) / inputs["weight"]
        accel = np.clip(accel, -GRAV_CONST, 2.5 * GRAV_CONST)
        outputs["accel_vert"] = accel

    def compute_partials(self, inputs, J):
        m = inputs["weight"]
        cosg = inputs["fltcond|cosgamma"]
        sing = inputs["fltcond|singamma"]

        J["accel_vert", "thrust"] = sing / m
        J["accel_vert", "drag"] = -sing / m
        J["accel_vert", "lift"] = cosg / m
        J["accel_vert", "fltcond|singamma"] = (inputs["thrust"] - inputs["drag"]) / m
        J["accel_vert", "fltcond|cosgamma"] = inputs["lift"] / m
        J["accel_vert", "weight"] = -(inputs["lift"] * cosg + (inputs["thrust"] - inputs["drag"]) * sing) / m**2


class SteadyFlightCL(ExplicitComponent):
    """
    Computes lift coefficient at each analysis point

    This is a helper function for the main mission analysis routine
    and shouldn't be instantiated directly.

    Inputs
    ------
    weight : float
        Aircraft weight at each analysis point (vector, kg)
    fltcond|q : float
        Dynamic pressure at each analysis point (vector, Pascal)
    ac|geom|wing|S_ref : float
        Reference wing area (scalar, m**2)
    fltcond|cosgamma : float
        Cosine of the flght path angle for all mission phases (vector, dimensionless)

    Outputs
    -------
    fltcond|CL : float
        Lift coefficient (vector, dimensionless)

    Options
    -------
    num_nodes : int
        Number of analysis nodes to run
    """

    def initialize(self):
        self.options.declare(
            "num_nodes",
            default=5,
            desc="Number of Simpson intervals to use per seg (eg. climb, cruise, descend). Number of analysis points is 2N+1",
        )

    def setup(self):
        nn = self.options["num_nodes"]
        arange = np.arange(nn)
        self.add_input("weight", units="kg", shape=(nn,))
        self.add_input("fltcond|q", units="N * m**-2", shape=(nn,))
        self.add_input("ac|geom|wing|S_ref", units="m **2")
        self.add_input("fltcond|cosgamma", val=1.0, shape=(nn,))
        self.add_output("fltcond|CL", shape=(nn,))
        self.declare_partials(["fltcond|CL"], ["weight", "fltcond|q", "fltcond|cosgamma"], rows=arange, cols=arange)
        self.declare_partials(["fltcond|CL"], ["ac|geom|wing|S_ref"], rows=arange, cols=np.zeros(nn))

    def compute(self, inputs, outputs):
        outputs["fltcond|CL"] = (
            inputs["fltcond|cosgamma"]
            * GRAV_CONST
            * inputs["weight"]
            / inputs["fltcond|q"]
            / inputs["ac|geom|wing|S_ref"]
        )

    def compute_partials(self, inputs, J):
        J["fltcond|CL", "weight"] = (
            inputs["fltcond|cosgamma"] * GRAV_CONST / inputs["fltcond|q"] / inputs["ac|geom|wing|S_ref"]
        )
        J["fltcond|CL", "fltcond|q"] = (
            -inputs["fltcond|cosgamma"]
            * GRAV_CONST
            * inputs["weight"]
            / inputs["fltcond|q"] ** 2
            / inputs["ac|geom|wing|S_ref"]
        )
        J["fltcond|CL", "ac|geom|wing|S_ref"] = (
            -inputs["fltcond|cosgamma"]
            * GRAV_CONST
            * inputs["weight"]
            / inputs["fltcond|q"]
            / inputs["ac|geom|wing|S_ref"] ** 2
        )
        J["fltcond|CL", "fltcond|cosgamma"] = (
            GRAV_CONST * inputs["weight"] / inputs["fltcond|q"] / inputs["ac|geom|wing|S_ref"]
        )


class GroundRollPhase(PhaseGroup):
    """
    This component group models the ground roll phase of a takeoff (acceleration before flight)
    User-settable parameters include:
    throttle (default 100 percent)
    rolling friction coeff (default 0.03 for accelerating phases and 0.4 for braking)
    propulsor_active (default 1 for v0 to v1, 0 for v1 to vr and braking) to model engine failure
    altitude (fltcond|h)

    The BaseAircraftGroup object is passed in.
    The BaseAircraftGroup should be built to accept the following inputs
    and return the following outputs.
    The outputs should be promoted to the top level in the component.

    Inputs
    ------
    range : float
        Total distance travelled (vector, m)
    fltcond|h : float
        Altitude (vector, m)
    fltcond|vs : float
        Vertical speed (vector, m/s)
    fltcond|Ueas : float
        Equivalent airspeed (vector, m/s)
    fltcond|Utrue : float
        True airspeed (vector, m/s)
    fltcond|p : float
        Pressure (vector, Pa)
    fltcond|rho : float
        Density (vector, kg/m3)
    fltcond|T : float
        Temperature (vector, K)
    fltcond|q : float
        Dynamic pressure (vector, Pa)
    fltcond|CL : float
        Lift coefficient (vector, dimensionless)
    throttle : float
        Motor / propeller throttle setting scaled from 0 to 1 or slightly more (vector, dimensionless)
    propulsor_active : float
        If a multi-propulsor airplane, a failure condition should be modeled in the propulsion model by multiplying throttle by propulsor_active.
        It will generally be 1.0 unless a failure condition is being modeled, in which case it will be 0 (vector, dimensionless)
    braking : float
        Brake friction coefficient (default 0.4 for dry runway braking, 0.03 for resistance unbraked)
        Should not be applied in the air or nonphysical effects will result (vector, dimensionless)
    lift : float
        Lift force (vector, N)

    Outputs
    -------
    thrust : float
        Total thrust force produced by all propulsors (vector, N)
    drag : float
        Total drag force in the airplane axis produced by all sources of drag (vector, N)
    weight : float
        Weight (mass, really) of the airplane at each point in time. (vector, kg)
    ac|geom|wing|S_ref
        Wing reference area (scalar, m**2)
    ac|aero|CLmax_TO
        CLmax with flaps in max takeoff position (scalar, dimensionless)
    ac|weights|MTOW
        Maximum takeoff weight (scalar, kg)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1)
        self.options.declare("flight_phase", default=None, desc="Phase of flight e.g. v0v1, cruise")
        self.options.declare("aircraft_model", default=None)

    def setup(self):
        nn = self.options["num_nodes"]
        ivcomp = self.add_subsystem("const_settings", IndepVarComp(), promotes_outputs=["*"])
        # set CL = 0.1 for the ground roll per Raymer's book
        ivcomp.add_output("fltcond|CL", val=np.ones((nn,)) * 0.1)
        ivcomp.add_output("vr_vstall_mult", val=1.1)
        ivcomp.add_output("fltcond|h", val=np.zeros((nn,)), units="m")
        ivcomp.add_output("fltcond|vs", val=np.zeros((nn,)), units="m/s")
        ivcomp.add_output("zero_speed", val=2, units="m/s")

        flight_phase = self.options["flight_phase"]
        if flight_phase == "v0v1":
            ivcomp.add_output("braking", val=np.ones((nn,)) * 0.03)
            ivcomp.add_output("propulsor_active", val=np.ones((nn,)))
            ivcomp.add_output("throttle", val=np.ones((nn,)))
            zero_start = True
        elif flight_phase == "v1vr":
            ivcomp.add_output("braking", val=np.ones((nn,)) * 0.03)
            ivcomp.add_output("propulsor_active", val=np.zeros((nn,)))
            ivcomp.add_output("throttle", val=np.ones((nn,)))
            zero_start = False

        elif flight_phase == "v1v0":
            ivcomp.add_output("braking", val=0.4 * np.ones((nn,)))
            ivcomp.add_output("propulsor_active", val=np.zeros((nn,)))
            ivcomp.add_output("throttle", val=np.zeros((nn,)))
            zero_start = False

        self.add_subsystem(
            "atmos",
            ComputeAtmosphericProperties(num_nodes=nn, true_airspeed_in=True),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )
        self.add_subsystem("gs", Groundspeeds(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])
        # add the user-defined aircraft model
        self.add_subsystem(
            "acmodel",
            self.options["aircraft_model"](num_nodes=nn, flight_phase=self.options["flight_phase"]),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )

        self.add_subsystem("lift", Lift(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem(
            "stall",
            StallSpeed(),
            promotes_inputs=[("CLmax", "ac|aero|CLmax_TO"), ("weight", "ac|weights|MTOW"), "ac|geom|wing|S_ref"],
            promotes_outputs=["*"],
        )
        self.add_subsystem(
            "vrspeed",
            ElementMultiplyDivideComp(
                output_name="takeoff|vr", input_names=["Vstall_eas", "vr_vstall_mult"], input_units=["m/s", None]
            ),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )

        self.add_subsystem(
            "haccel", HorizontalAcceleration(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"]
        )
        if flight_phase == "v1v0":
            # unfortunately need to shoot backwards to avoid negative airspeeds
            # reverse the order of the accelerations so the last one is first (and make them negative)
            self.add_subsystem(
                "flipaccel",
                FlipVectorComp(num_nodes=nn, units="m/s**2", negative=True),
                promotes_inputs=[("vec_in", "accel_horiz")],
            )
            # integrate the timesteps in reverse from near zero speed.
            ode_integ = self.add_subsystem(
                "ode_integ_phase",
                Integrator(num_nodes=nn, method="simpson", diff_units="s", time_setup="duration"),
                promotes_inputs=["*"],
                promotes_outputs=["*"],
            )
            ode_integ.add_integrand(
                "vel_q",
                units="m/s",
                rate_name="vel_dqdt",
                start_name="zero_speed",
                end_name="fltcond|Utrue_initial",
                lower=1.5,
            )
            self.connect("flipaccel.vec_out", "vel_dqdt")
            # flip the result of the reverse integration again so the flight condition is forward and consistent with everythign else
            self.add_subsystem(
                "flipvel",
                FlipVectorComp(num_nodes=nn, units="m/s", negative=False),
                promotes_outputs=[("vec_out", "fltcond|Utrue")],
            )
            self.connect("vel_q", "flipvel.vec_in")
            # now set the time step so that backwards shooting results in the correct 'initial' segment airspeed
            self.add_subsystem(
                "v0constraint",
                BalanceComp(
                    name="duration",
                    units="s",
                    eq_units="m/s",
                    rhs_name="fltcond|Utrue_initial",
                    lhs_name="takeoff|v1",
                    val=10.0,
                    upper=100.0,
                    lower=1.0,
                ),
                promotes_inputs=["*"],
                promotes_outputs=["duration"],
            )
        else:
            # forward shooting for these acceleration phases
            ode_integ = self.add_subsystem(
                "ode_integ_phase",
                Integrator(num_nodes=nn, method="simpson", diff_units="s", time_setup="duration"),
                promotes_inputs=["*"],
                promotes_outputs=["*"],
            )
            ode_integ.add_integrand(
                "fltcond|Utrue",
                units="m/s",
                rate_name="accel_horiz",
                start_name="fltcond|Utrue_initial",
                end_name="fltcond|Utrue_final",
                lower=1.5,
            )
            if flight_phase == "v0v1":
                self.connect("zero_speed", "fltcond|Utrue_initial")
                self.add_subsystem(
                    "v1constraint",
                    BalanceComp(
                        name="duration",
                        units="s",
                        eq_units="m/s",
                        rhs_name="fltcond|Utrue_final",
                        lhs_name="takeoff|v1",
                        val=10.0,
                        upper=100.0,
                        lower=1.0,
                    ),
                    promotes_inputs=["*"],
                    promotes_outputs=["duration"],
                )
            elif flight_phase == "v1vr":
                self.add_subsystem(
                    "vrconstraint",
                    BalanceComp(
                        name="duration",
                        units="s",
                        eq_units="m/s",
                        rhs_name="fltcond|Utrue_final",
                        lhs_name="takeoff|vr",
                        val=5.0,
                        upper=12.0,
                        lower=0.0,
                    ),
                    promotes_inputs=["*"],
                    promotes_outputs=["duration"],
                )

        if zero_start:
            ode_integ.add_integrand("range", rate_name="fltcond|groundspeed", units="m", zero_start=True)
        else:
            ode_integ.add_integrand("range", rate_name="fltcond|groundspeed", units="m")


class RotationPhase(PhaseGroup):
    """
    This group models the transition from ground roll to climb out during a takeoff
    using force balance in the vertical and horizontal directions.

    User-settable parameters include:
    throttle (default 100 percent)
    rolling friction coeff (default 0.03 for accelerating phases and 0.4 for braking)
    propulsor_active (default 1 for v0 to v1, 0 for v1 to vr and braking) to model engine failure
    altitude (fltcond|h)
    obstacle clearance hight (h_obs) default 35 feet per FAR 25
    Rotation CL/CLmax ratio (default 0.83)

    The BaseAircraftGroup object is passed in.
    The BaseAircraftGroup should be built to accept the following inputs
    and return the following outputs.
    The outputs should be promoted to the top level in the component.

    Inputs
    ------
    range : float
        Total distance travelled (vector, m)
    fltcond|h : float
        Altitude (vector, m)
    fltcond|vs : float
        Vertical speed (vector, m/s)
    fltcond|Ueas : float
        Equivalent airspeed (vector, m/s)
    fltcond|Utrue : float
        True airspeed (vector, m/s)
    fltcond|p : float
        Pressure (vector, Pa)
    fltcond|rho : float
        Density (vector, kg/m3)
    fltcond|T : float
        Temperature (vector, K)
    fltcond|q : float
        Dynamic pressure (vector, Pa)
    fltcond|CL : float
        Lift coefficient (vector, dimensionless)
    throttle : float
        Motor / propeller throttle setting scaled from 0 to 1 or slightly more (vector, dimensionless)
    propulsor_active : float
        If a multi-propulsor airplane, a failure condition should be modeled in the propulsion model by multiplying throttle by propulsor_active.
        It will generally be 1.0 unless a failure condition is being modeled, in which case it will be 0 (vector, dimensionless)
    braking : float
        Percentage brakes applied, from 0 to 1. Should not be applied in the air or nonphysical effects will result (vector, dimensionless)
    lift : float
        Lift force (vector, N)

    Outputs
    -------
    thrust : float
        Total thrust force produced by all propulsors (vector, N)
    drag : float
        Total drag force in the airplane axis produced by all sources of drag (vector, N)
    weight : float
        Weight (mass, really) of the airplane at each point in time. Generally will need to be integrated by Dymos as a state with a rate source (vector, kg)
    ac|geom|wing|S_ref
        Wing reference area (scalar, m**2)
    ac|aero|CLmax_TO
        CLmax with flaps in max takeoff position (scalar, dimensionless)
    ac|weights|MTOW
        Maximum takeoff weight (scalar, kg)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1)
        self.options.declare("flight_phase", default=None)
        self.options.declare("aircraft_model", default=None)

    def setup(self):
        nn = self.options["num_nodes"]
        ivcomp = self.add_subsystem("const_settings", IndepVarComp(), promotes_outputs=["*"])
        ivcomp.add_output("CL_rotate_mult", val=np.ones((nn,)) * 0.83)
        ivcomp.add_output("h_obs", val=35, units="ft")
        flight_phase = self.options["flight_phase"]
        if flight_phase == "rotate":
            ivcomp.add_output("braking", val=np.zeros((nn,)))
            ivcomp.add_output("propulsor_active", val=np.zeros((nn,)))
            ivcomp.add_output("throttle", val=np.ones((nn,)))

        self.add_subsystem(
            "atmos",
            ComputeAtmosphericProperties(num_nodes=nn, true_airspeed_in=True),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )
        self.add_subsystem("gs", Groundspeeds(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem(
            "clcomp",
            ElementMultiplyDivideComp(
                output_name="fltcond|CL", input_names=["CL_rotate_mult", "ac|aero|CLmax_TO"], vec_size=[nn, 1], length=1
            ),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )
        self.add_subsystem(
            "acmodel",
            self.options["aircraft_model"](num_nodes=nn, flight_phase=self.options["flight_phase"]),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )

        self.add_subsystem("lift", Lift(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem(
            "haccel", HorizontalAcceleration(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"]
        )
        self.add_subsystem("vaccel", VerticalAcceleration(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])

        # TODO always starts from zero altitude
        self.add_subsystem(
            "clear_obstacle",
            BalanceComp(
                name="duration",
                units="s",
                val=1,
                eq_units="m",
                rhs_name="fltcond|h_final",
                lhs_name="h_obs",
                lower=0.1,
                upper=15,
            ),
            promotes_inputs=["*"],
            promotes_outputs=["duration"],
        )
        int1 = self.add_subsystem(
            "intvelocity",
            Integrator(num_nodes=nn, method="simpson", diff_units="s", time_setup="duration"),
            promotes_outputs=["*"],
            promotes_inputs=["*"],
        )
        int1.add_integrand("fltcond|Utrue", rate_name="accel_horiz", units="m/s", lower=0.1)
        int2 = self.add_subsystem(
            "intrange",
            Integrator(num_nodes=nn, method="simpson", diff_units="s", time_setup="duration"),
            promotes_outputs=["*"],
            promotes_inputs=["*"],
        )
        int2.add_integrand("range", rate_name="fltcond|groundspeed", units="m")
        int3 = self.add_subsystem(
            "intvs",
            Integrator(num_nodes=nn, method="simpson", diff_units="s", time_setup="duration"),
            promotes_outputs=["*"],
            promotes_inputs=["*"],
        )
        int3.add_integrand("fltcond|vs", rate_name="accel_vert", units="m/s", zero_start=True)
        int4 = self.add_subsystem(
            "inth",
            Integrator(num_nodes=nn, method="simpson", diff_units="s", time_setup="duration"),
            promotes_outputs=["*"],
            promotes_inputs=["*"],
        )
        int4.add_integrand("fltcond|h", rate_name="fltcond|vs", units="m", zero_start=True)


class SteadyFlightPhase(PhaseGroup):
    """
    This component group models steady flight conditions.
    Settable mission parameters include:
    Airspeed (fltcond|Ueas)
    Vertical speed (fltcond|vs)
    Duration of the phase (duration)

    Throttle is set automatically to ensure steady flight

    The BaseAircraftGroup object is passed in.
    The BaseAircraftGroup should be built to accept the following inputs
    and return the following outputs.
    The outputs should be promoted to the top level in the component.

    Inputs
    ------
    range : float
        Total distance travelled (vector, m)
    fltcond|h : float
        Altitude (vector, m)
    fltcond|vs : float
        Vertical speed (vector, m/s)
    fltcond|Ueas : float
        Equivalent airspeed (vector, m/s)
    fltcond|Utrue : float
        True airspeed (vector, m/s)
    fltcond|p : float
        Pressure (vector, Pa)
    fltcond|rho : float
        Density (vector, kg/m3)
    fltcond|T : float
        Temperature (vector, K)
    fltcond|q : float
        Dynamic pressure (vector, Pa)
    fltcond|CL : float
        Lift coefficient (vector, dimensionless)
    throttle : float
        Motor / propeller throttle setting scaled from 0 to 1 or slightly more (vector, dimensionless)
    propulsor_active : float
        If a multi-propulsor airplane, a failure condition should be modeled in the propulsion model by multiplying throttle by propulsor_active.
        It will generally be 1.0 unless a failure condition is being modeled, in which case it will be 0 (vector, dimensionless)
    braking : float
        Brake friction coefficient (default 0.4 for dry runway braking, 0.03 for resistance unbraked)
        Should not be applied in the air or nonphysical effects will result (vector, dimensionless)
    lift : float
        Lift force (vector, N)

    Outputs
    -------
    thrust : float
        Total thrust force produced by all propulsors (vector, N)
    drag : float
        Total drag force in the airplane axis produced by all sources of drag (vector, N)
    weight : float
        Weight (mass, really) of the airplane at each point in time. (vector, kg)
    ac|geom|wing|S_ref
        Wing reference area (scalar, m**2)
    ac|aero|CLmax_TO
        CLmax with flaps in max takeoff position (scalar, dimensionless)
    ac|weights|MTOW
        Maximum takeoff weight (scalar, kg)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1)
        self.options.declare("flight_phase", default=None, desc="Phase of flight e.g. v0v1, cruise")
        self.options.declare("aircraft_model", default=None)

    def setup(self):
        nn = self.options["num_nodes"]
        # propulsor_active only exists in some aircraft models, so set_input_defaults
        # can't be used since it throws an error when it can't find an input
        ivcomp = self.add_subsystem("const_settings", IndepVarComp(), promotes_outputs=["*"])
        ivcomp.add_output("propulsor_active", val=np.ones(nn))

        # Use set_input_defaults as opposed to independent variable component to enable
        # users to connect linear interpolators to these inputs for "trajectory optimization"
        self.set_input_defaults("braking", np.zeros(nn))
        self.set_input_defaults("fltcond|Ueas", np.ones((nn,)) * 90, units="m/s")
        self.set_input_defaults("fltcond|vs", np.ones((nn,)) * 1, units="m/s")
        self.set_input_defaults("zero_accel", np.zeros((nn,)), units="m/s**2")

        integ = self.add_subsystem(
            "ode_integ_phase",
            Integrator(num_nodes=nn, diff_units="s", time_setup="duration", method="simpson"),
            promotes_inputs=["fltcond|vs", "fltcond|groundspeed"],
            promotes_outputs=["fltcond|h", "range"],
        )
        integ.add_integrand("fltcond|h", rate_name="fltcond|vs", val=1.0, units="m")
        self.add_subsystem(
            "atmos",
            ComputeAtmosphericProperties(num_nodes=nn, true_airspeed_in=False),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )
        self.add_subsystem("gs", Groundspeeds(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])
        # add the user-defined aircraft model
        self.add_subsystem(
            "acmodel",
            self.options["aircraft_model"](num_nodes=nn, flight_phase=self.options["flight_phase"]),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )
        self.add_subsystem("clcomp", SteadyFlightCL(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("lift", Lift(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem(
            "haccel", HorizontalAcceleration(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"]
        )
        integ.add_integrand("range", rate_name="fltcond|groundspeed", val=1.0, units="m")
        self.add_subsystem(
            "steadyflt",
            BalanceComp(
                name="throttle",
                val=np.ones((nn,)) * 0.5,
                lower=0.01,
                upper=1.05,
                units=None,
                normalize=False,
                eq_units="m/s**2",
                rhs_name="accel_horiz",
                lhs_name="zero_accel",
                rhs_val=np.zeros((nn,)),
            ),
            promotes_inputs=["accel_horiz", "zero_accel"],
            promotes_outputs=["throttle"],
        )


class ClimbAnglePhase(Group):
    """
    This component checks the climb angle for a
    single flight condition at the V2 speed. No integration is performed.

    User settable parameter includes the V2/Vstall multiple (default 1.2)

    Useful for ensuring all-engine climb gradients in optimization.
    Choose flight_phase = AllEngineClimbAngle or EngineOutClimbAngle
    to set the propulsor_active property correctly.

    Inputs
    ------
    range : float
        Total distance travelled (vector, m)
    fltcond|h : float
        Altitude (vector, m)
    fltcond|vs : float
        Vertical speed (vector, m/s)
    fltcond|Ueas : float
        Equivalent airspeed (vector, m/s)
    fltcond|Utrue : float
        True airspeed (vector, m/s)
    fltcond|p : float
        Pressure (vector, Pa)
    fltcond|rho : float
        Density (vector, kg/m3)
    fltcond|T : float
        Temperature (vector, K)
    fltcond|q : float
        Dynamic pressure (vector, Pa)
    fltcond|CL : float
        Lift coefficient (vector, dimensionless)
    throttle : float
        Motor / propeller throttle setting scaled from 0 to 1 or slightly more (vector, dimensionless)
    propulsor_active : float
        If a multi-propulsor airplane, a failure condition should be modeled in the propulsion model by multiplying throttle by propulsor_active.
        It will generally be 1.0 unless a failure condition is being modeled, in which case it will be 0 (vector, dimensionless)
    lift : float
        Lift force (vector, N)

    Outputs
    -------
    thrust : float
        Total thrust force produced by all propulsors (vector, N)
    drag : float
        Total drag force in the airplane axis produced by all sources of drag (vector, N)
    weight : float
        Weight (mass, really) of the airplane at each point in time. Generally will need to be integrated by Dymos as a state with a rate source (vector, kg)
    ac|geom|wing|S_ref
        Wing reference area (scalar, m**2)
    ac|aero|CLmax_TO
        CLmax with flaps in max takeoff position (scalar, dimensionless)
    ac|weights|MTOW
        Maximum takeoff weight (scalar, kg)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1)
        self.options.declare("flight_phase", default=None, desc="Phase of flight e.g. v0v1, cruise")
        self.options.declare("aircraft_model", default=None)

    def setup(self):
        nn = self.options["num_nodes"]
        ivcomp = self.add_subsystem("const_settings", IndepVarComp(), promotes_outputs=["*"])
        ivcomp.add_output("v2_vstall_mult", val=1.2)
        ivcomp.add_output("fltcond|h", val=np.zeros((nn,)), units="m")
        ivcomp.add_output("fltcond|cosgamma", val=np.ones((nn,)))

        flight_phase = self.options["flight_phase"]
        if flight_phase == "AllEngineClimbAngle":
            ivcomp.add_output("propulsor_active", val=np.ones((nn,)))
            ivcomp.add_output("throttle", val=np.ones((nn,)))
        elif flight_phase == "EngineOutClimbAngle":
            ivcomp.add_output("propulsor_active", val=np.zeros((nn,)))
            ivcomp.add_output("throttle", val=np.ones((nn,)))
        self.add_subsystem(
            "stall",
            StallSpeed(),
            promotes_inputs=[("CLmax", "ac|aero|CLmax_TO"), ("weight", "ac|weights|MTOW"), "ac|geom|wing|S_ref"],
            promotes_outputs=["*"],
        )
        self.add_subsystem(
            "vrspeed",
            ElementMultiplyDivideComp(
                output_name="takeoff|v2", input_names=["Vstall_eas", "v2_vstall_mult"], input_units=["m/s", None]
            ),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )
        self.add_subsystem(
            "atmos",
            ComputeAtmosphericProperties(num_nodes=nn, true_airspeed_in=False),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )
        self.add_subsystem(
            "clcomp",
            SteadyFlightCL(num_nodes=nn),
            promotes_inputs=[("weight", "ac|weights|MTOW"), "fltcond|*", "ac|*"],
            promotes_outputs=["*"],
        )
        self.connect("takeoff|v2", "fltcond|Ueas")
        # the aircraft model needs to provide thrust and drag
        self.add_subsystem(
            "acmodel",
            self.options["aircraft_model"](num_nodes=nn, flight_phase=self.options["flight_phase"]),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )
        self.add_subsystem(
            "climbangle",
            ClimbAngleComp(num_nodes=nn),
            promotes_inputs=["drag", ("weight", "ac|weights|MTOW"), "thrust"],
            promotes_outputs=["gamma"],
        )


class TakeoffTransition(ExplicitComponent):
    """
    Computes distance and altitude at end of circular transition.

    Based on TO distance analysis method in Raymer book.
    Obstacle clearance height set for GA / Part 23 aircraft
    Override for analyzing Part 25 aircraft

    Inputs
    ------
    fltcond|Utrue
        Transition true airspeed (generally avg of vr and v2) (scalar, m/s)
    gamma : float
        Climb out flight path angle (scalar, rad)

    Outputs
    -------
    s_transition : float
        Horizontal distance during transition to v2 climb out (scalar, m)
    h_transition : float
        Altitude at transition point (scalar, m)
    t_transition : float
        Elapsed time in transition (scalar, s)

    Options
    -------
    h_obstacle : float
        Obstacle height to clear (in **meters**) (default 10.66, equiv. 35 ft)
    load_factor : float
        Load factor during rotation and transition (default 1.2 from Raymer book)
    """

    def initialize(self):
        self.options.declare("h_obstacle", default=10.66, desc="Obstacle clearance height in m")
        self.options.declare("load_factor", default=1.2, desc="Load factor during circular arc transition")

    def setup(self):
        self.add_input("fltcond|Utrue", units="m/s")
        self.add_input("gamma", units="rad")
        self.add_output("s_transition", units="m")
        self.add_output("h_transition", units="m")
        self.add_output("t_transition", units="s")
        self.declare_partials(["s_transition", "h_transition", "t_transition"], ["fltcond|Utrue", "gamma"])

    def compute(self, inputs, outputs):
        hobs = self.options["h_obstacle"]
        nfactor = self.options["load_factor"] - 1
        gam = inputs["gamma"]
        ut = inputs["fltcond|Utrue"]

        R = ut**2 / nfactor / GRAV_CONST
        st = R * np.sin(gam)
        ht = R * (1 - np.cos(gam))
        # alternate formula if the obstacle is cleared during transition
        if ht > hobs:
            st = np.sqrt(R**2 - (R - hobs) ** 2)
            ht = hobs
        outputs["s_transition"] = st
        outputs["h_transition"] = ht
        outputs["t_transition"] = st / ut

    def compute_partials(self, inputs, J):
        hobs = self.options["h_obstacle"]
        nfactor = self.options["load_factor"] - 1
        gam = inputs["gamma"]
        ut = inputs["fltcond|Utrue"]
        R = ut**2 / nfactor / GRAV_CONST
        dRdut = 2 * ut / nfactor / GRAV_CONST
        st = R * np.sin(gam)
        ht = R * (1 - np.cos(gam))
        # alternate formula if the obstacle is cleared during transition
        if ht > hobs:
            st = np.sqrt(R**2 - (R - hobs) ** 2)
            dstdut = 1 / 2 / np.sqrt(R**2 - (R - hobs) ** 2) * (2 * R * dRdut - 2 * (R - hobs) * dRdut)
            dstdgam = 0
            dhtdut = 0
            dhtdgam = 0
        else:
            dhtdut = dRdut * (1 - np.cos(gam))
            dhtdgam = R * np.sin(gam)
            dstdut = dRdut * np.sin(gam)
            dstdgam = R * np.cos(gam)
        J["s_transition", "gamma"] = dstdgam
        J["s_transition", "fltcond|Utrue"] = dstdut
        J["h_transition", "gamma"] = dhtdgam
        J["h_transition", "fltcond|Utrue"] = dhtdut
        J["t_transition", "gamma"] = dstdgam / ut
        J["t_transition", "fltcond|Utrue"] = (dstdut * ut - st) / ut**2


class TakeoffClimb(ExplicitComponent):
    """
    Computes ground distance from end of transition until obstacle is cleared.

    Analysis based on Raymer book.

    Inputs
    ------
    gamma : float
        Climb out flight path angle (scalar, rad)
    h_transition : float
        Altitude at transition point (scalar, m)

    Outputs
    -------
    s_climb : float
        Horizontal distance from end of transition until obstacle is cleared (scalar, m)

    Options
    -------
    h_obstacle : float
        Obstacle height to clear (in **meters**) (default 10.66, equiv. 35 ft)
    """

    def initialize(self):
        self.options.declare("h_obstacle", default=10.66, desc="Obstacle clearance height in m")

    def setup(self):
        self.add_input("h_transition", units="m")
        self.add_input("gamma", units="rad")
        self.add_input("fltcond|Utrue", units="m/s")

        self.add_output("s_climb", units="m")
        self.add_output("t_climb", units="s")
        self.declare_partials(["s_climb"], ["h_transition", "gamma"])
        self.declare_partials(["t_climb"], ["h_transition", "gamma", "fltcond|Utrue"])

    def compute(self, inputs, outputs):
        hobs = self.options["h_obstacle"]
        gam = inputs["gamma"]
        ht = inputs["h_transition"]
        ut = inputs["fltcond|Utrue"]
        sc = (hobs - ht) / np.tan(gam)
        outputs["s_climb"] = sc
        outputs["t_climb"] = sc / ut

    def compute_partials(self, inputs, J):
        hobs = self.options["h_obstacle"]
        gam = inputs["gamma"]
        ht = inputs["h_transition"]
        ut = inputs["fltcond|Utrue"]
        sc = (hobs - ht) / np.tan(gam)
        J["s_climb", "gamma"] = -(hobs - ht) / np.tan(gam) ** 2 * (1 / np.cos(gam)) ** 2
        J["s_climb", "h_transition"] = -1 / np.tan(gam)
        J["t_climb", "gamma"] = J["s_climb", "gamma"] / ut
        J["t_climb", "h_transition"] = J["s_climb", "h_transition"] / ut
        J["t_climb", "fltcond|Utrue"] = -sc / ut**2


class RobustRotationPhase(PhaseGroup):
    """
    This adds general mission analysis capabilities to an existing airplane model.
    The BaseAircraftGroup object is passed in. It should be built to accept the following inputs and return the following outputs.
    The outputs should be promoted to the top level in the component.

    Inputs
    ------
    range : float
        Total distance travelled (vector, m)
    fltcond|h : float
        Altitude (vector, m)
    fltcond|vs : float
        Vertical speed (vector, m/s)
    fltcond|Ueas : float
        Equivalent airspeed (vector, m/s)
    fltcond|Utrue : float
        True airspeed (vector, m/s)
    fltcond|p : float
        Pressure (vector, Pa)
    fltcond|rho : float
        Density (vector, kg/m3)
    fltcond|T : float
        Temperature (vector, K)
    fltcond|q : float
        Dynamic pressure (vector, Pa)
    fltcond|CL : float
        Lift coefficient (vector, dimensionless)
    throttle : float
        Motor / propeller throttle setting scaled from 0 to 1 or slightly more (vector, dimensionless)
    propulsor_active : float
        If a multi-propulsor airplane, a failure condition should be modeled in the propulsion model by multiplying throttle by propulsor_active.
        It will generally be 1.0 unless a failure condition is being modeled, in which case it will be 0 (vector, dimensionless)
    braking : float
        Percentage brakes applied, from 0 to 1. Should not be applied in the air or nonphysical effects will result (vector, dimensionless)
    lift : float
        Lift force (vector, N)

    Outputs
    -------
    thrust : float
        Total thrust force produced by all propulsors (vector, N)
    drag : float
        Total drag force in the airplane axis produced by all sources of drag (vector, N)
    weight : float
        Weight (mass, really) of the airplane at each point in time. Generally will need to be integrated by Dymos as a state with a rate source (vector, kg)
    ac|geom|wing|S_ref
        Wing reference area (scalar, m**2)
    ac|aero|CLmax_TO
        CLmax with flaps in max takeoff position (scalar, dimensionless)
    ac|weights|MTOW
        Maximum takeoff weight (scalar, kg)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1)
        self.options.declare("flight_phase", default=None, desc="Phase of flight e.g. v0v1, cruise")
        self.options.declare("aircraft_model", default=None)
        self.options.declare(
            "h_obstacle",
            default=10.66,
        )

    def setup(self):
        nn = self.options["num_nodes"]
        ivcomp = self.add_subsystem("const_settings", IndepVarComp(), promotes_outputs=["*"])
        flight_phase = self.options["flight_phase"]
        if flight_phase == "rotate":
            ivcomp.add_output("braking", val=np.zeros((nn,)))
            ivcomp.add_output("propulsor_active", val=np.zeros((nn,)))
            ivcomp.add_output("throttle", val=np.ones((nn,)))
        # flight conditions are sea level takeoff, transition speed
        # split off a single node to compute climb angle
        # compute the transition distance and add it to range_initial
        # compute the transition time as a function of the groundspeed
        # provide transition time as duration
        ivcomp.add_output("v2_vstall_mult", val=1.2)
        ivcomp.add_output("vr_vstall_mult", val=1.1)
        ivcomp.add_output("fltcond|vs", val=np.zeros((nn,)), units="m/s")
        ivcomp.add_output("fltcond|cosgamma", val=np.ones((nn,)), units=None)

        ivcomp.add_output("h_obstacle", val=35, units="ft")

        self.add_subsystem(
            "altitudes",
            LinearInterpolator(num_nodes=nn, units="m"),
            promotes_inputs=[("start_val", "h_initial")],
            promotes_outputs=[("vec", "fltcond|h")],
        )
        self.connect("h_obstacle", "altitudes.end_val")

        self.add_subsystem(
            "stall",
            StallSpeed(),
            promotes_inputs=[("CLmax", "ac|aero|CLmax_TO"), ("weight", "ac|weights|MTOW"), "ac|geom|wing|S_ref"],
            promotes_outputs=["*"],
        )
        self.add_subsystem(
            "vrspeed",
            ElementMultiplyDivideComp(
                output_name="takeoff|vr", input_names=["Vstall_eas", "vr_vstall_mult"], input_units=["m/s", None]
            ),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )
        self.add_subsystem(
            "v2speed",
            ElementMultiplyDivideComp(
                output_name="takeoff|v2", input_names=["Vstall_eas", "v2_vstall_mult"], input_units=["m/s", None]
            ),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )
        self.add_subsystem(
            "speeds",
            LinearInterpolator(num_nodes=nn, units="kn"),
            promotes_inputs=[("start_val", "takeoff|vr"), ("end_val", "takeoff|v2")],
            promotes_outputs=[("vec", "fltcond|Ueas")],
        )
        self.add_subsystem(
            "atmos",
            ComputeAtmosphericProperties(num_nodes=nn, true_airspeed_in=False),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )
        # pretty confident there's a simpler closed form multiple for CL at v2
        self.add_subsystem(
            "clcomp",
            SteadyFlightCL(num_nodes=nn),
            promotes_inputs=["weight", "fltcond|*", "ac|*"],
            promotes_outputs=["*"],
        )
        # the aircraft model needs to provide thrust and drag
        self.add_subsystem(
            "acmodel",
            self.options["aircraft_model"](num_nodes=nn, flight_phase=self.options["flight_phase"]),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )
        self.add_subsystem(
            "climbangle",
            ClimbAngleComp(num_nodes=nn),
            promotes_inputs=["drag", "weight", "thrust"],
            promotes_outputs=["gamma"],
        )
        self.add_subsystem(
            "transition", TakeoffTransition(), promotes_outputs=["h_transition", "s_transition", "t_transition"]
        )
        self.promotes("transition", inputs=["fltcond|Utrue", "gamma"], src_indices=[0], flat_src_indices=True)
        self.add_subsystem(
            "v2climb", TakeoffClimb(), promotes_inputs=["h_transition"], promotes_outputs=["s_climb", "t_climb"]
        )
        self.promotes("v2climb", inputs=["fltcond|Utrue", "gamma"], src_indices=[-1], flat_src_indices=True)
        self.add_subsystem(
            "tod_final",
            AddSubtractComp(
                output_name="range_final", input_names=["range_initial", "s_transition", "s_climb"], units="m"
            ),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )
        self.add_subsystem(
            "duration",
            AddSubtractComp(output_name="duration", input_names=["t_transition", "t_climb"], units="s"),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )
        self.add_subsystem(
            "h_final",
            AddSubtractComp(output_name="fltcond|h_final", input_names=["h_obstacle"], units="m"),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )
        self.add_subsystem(
            "ranges",
            LinearInterpolator(num_nodes=nn, units="m"),
            promotes_inputs=[("start_val", "range_initial"), ("end_val", "range_final")],
            promotes_outputs=[("vec", "range")],
        )
