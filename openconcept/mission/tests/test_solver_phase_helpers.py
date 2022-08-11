import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.api import IndepVarComp, Group, Problem
from openconcept.mission import (
    ClimbAngleComp,
    Groundspeeds,
    HorizontalAcceleration,
    VerticalAcceleration,
    SteadyFlightCL,
    FlipVectorComp,
)
from openconcept.utilities.constants import GRAV_CONST

# TESTS FOR ClimbAngleComp ===================================


class ClimbAngleCompTestGroup(Group):
    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of mission analysis points to run")

    def setup(self):
        nn = self.options["num_nodes"]
        iv = self.add_subsystem("conditions", IndepVarComp(), promotes_outputs=["*"])
        iv.add_output("thrust", val=np.ones((nn,)) * 1000, units="N")
        iv.add_output("drag", val=np.ones((nn,)) * 1000, units="N")
        iv.add_output("weight", val=np.ones((nn,)) * 1000, units="kg")

        self.add_subsystem("climbangle", ClimbAngleComp(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])


class ClimbAngleTestCase_Scalar(unittest.TestCase):
    def setUp(self):
        self.prob = Problem(ClimbAngleCompTestGroup(num_nodes=1))
        self.prob.setup(check=True, force_alloc_complex=True)
        self.prob.run_model()

    def test_level_flight(self):
        assert_near_equal(self.prob["gamma"][0], 0, tolerance=1e-10)

    def test_climb_flight(self):
        self.prob["thrust"] = np.ones((1,)) * 1200
        self.prob.run_model()
        assert_near_equal(self.prob["gamma"][0], np.arcsin(200 / 1000 / GRAV_CONST), tolerance=1e-10)

    def test_partials(self):
        partials = self.prob.check_partials(method="cs", out_stream=None)
        assert_check_partials(partials)


# TESTS FOR FlipVectorComp ===================================


class FlipVectorCompTestGroup(Group):
    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of points to run")
        self.options.declare("units", default=None)
        self.options.declare("negative", default=False)

    def setup(self):
        nn = self.options["num_nodes"]
        unit_string = self.options["units"]
        neg_flag = self.options["negative"]

        iv = self.add_subsystem("conditions", IndepVarComp(), promotes_outputs=["*"])
        iv.add_output("thrust", val=np.linspace(0, 100, nn), units="N")

        self.add_subsystem(
            "flipvector", FlipVectorComp(num_nodes=nn, units=unit_string, negative=neg_flag), promotes_outputs=["*"]
        )
        self.connect("thrust", "flipvector.vec_in")


class FlipVectorCompTestCase_Vector(unittest.TestCase):
    def setUp(self):
        self.prob = Problem(FlipVectorCompTestGroup(num_nodes=11, units="N"))
        self.prob.setup(check=True, force_alloc_complex=True)
        self.prob.run_model()

    def test_flip_vec_order(self):
        assert_near_equal(self.prob["vec_out"], np.linspace(100, 0, 11), tolerance=1e-10)

    def test_partials(self):
        partials = self.prob.check_partials(method="cs", out_stream=None)
        assert_check_partials(partials)


class FlipVectorCompTestCase_Scalar(unittest.TestCase):
    def setUp(self):
        self.prob = Problem(FlipVectorCompTestGroup(num_nodes=1, units="N"))
        self.prob.setup(check=True, force_alloc_complex=True)
        self.prob.run_model()

    def test_flip_vec_order(self):
        assert_near_equal(self.prob["vec_out"], np.zeros((1,)), tolerance=1e-10)

    def test_partials(self):
        partials = self.prob.check_partials(method="cs", out_stream=None)
        assert_check_partials(partials)


class FlipVectorCompTestCase_Negative(unittest.TestCase):
    def setUp(self):
        self.prob = Problem(FlipVectorCompTestGroup(num_nodes=11, units="N", negative=True))
        self.prob.setup(check=True, force_alloc_complex=True)
        self.prob.run_model()

    def test_flip_vec_order(self):
        assert_near_equal(self.prob["vec_out"], np.linspace(-100, 0, 11), tolerance=1e-10)

    def test_partials(self):
        partials = self.prob.check_partials(method="cs", out_stream=None)
        assert_check_partials(partials)


# TESTS FOR Groundspeeds ===================================


class GroundspeedsTestGroup(Group):
    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of mission analysis points to run")

    def setup(self):
        nn = self.options["num_nodes"]
        iv = self.add_subsystem("conditions", IndepVarComp(), promotes_outputs=["*"])
        iv.add_output("fltcond|vs", val=np.linspace(0, 3, nn), units="m/s")
        iv.add_output("fltcond|Utrue", val=np.ones((nn,)) * 50, units="m/s")
        self.add_subsystem("gs", Groundspeeds(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])


class GroundspeedsTestCase(unittest.TestCase):
    def setUp(self):
        self.prob = Problem(GroundspeedsTestGroup(num_nodes=15))
        self.prob.setup(check=True, force_alloc_complex=True)
        self.prob.run_model()

    def test_level_flight(self):
        assert_near_equal(self.prob["fltcond|groundspeed"][0], 50, tolerance=1e-10)
        assert_near_equal(self.prob["fltcond|cosgamma"][0], 1.0, tolerance=1e-10)
        assert_near_equal(self.prob["fltcond|singamma"][0], 0.0, tolerance=1e-10)

    def test_climb_flight(self):
        gs = np.sqrt(50**2 - 3**2)
        assert_near_equal(self.prob["fltcond|groundspeed"][-1], gs, tolerance=1e-10)
        assert_near_equal(self.prob["fltcond|cosgamma"][-1], gs / 50.0, tolerance=1e-10)
        assert_near_equal(self.prob["fltcond|singamma"][-1], 3.0 / 50.0, tolerance=1e-10)

    def test_partials(self):
        partials = self.prob.check_partials(method="cs", out_stream=None)
        assert_check_partials(partials)


# TESTS FOR HorizontalAcceleration ===================================


class HorizontalAccelerationTestGroup(Group):
    def initialize(self):
        self.options.declare("num_nodes", default=9, desc="Number of mission analysis points to run")

    def setup(self):
        nn = self.options["num_nodes"]
        iv = self.add_subsystem("conditions", IndepVarComp(), promotes_outputs=["*"])
        iv.add_output("weight", val=np.ones((nn,)) * 100, units="kg")
        iv.add_output("lift", val=np.ones((nn,)) * 100, units="N")
        iv.add_output("thrust", val=np.ones((nn,)) * 100, units="N")
        iv.add_output("drag", val=np.ones((nn,)) * 100, units="N")
        iv.add_output("fltcond|singamma", val=np.zeros((nn,)), units=None)
        iv.add_output("braking", val=np.zeros((nn,)), units=None)
        self.add_subsystem("ha", HorizontalAcceleration(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])


class HorizontalAccelerationTestCase_SteadyLevel(unittest.TestCase):
    def setUp(self):
        self.prob = Problem(HorizontalAccelerationTestGroup(num_nodes=9))
        self.prob.setup(check=True, force_alloc_complex=True)
        self.prob.run_model()

    def test_steady_level_flights(self):
        assert_near_equal(self.prob["accel_horiz"], np.zeros((9,)), tolerance=1e-10)

    def test_partials(self):
        partials = self.prob.check_partials(method="cs", out_stream=None)
        assert_check_partials(partials)


class HorizontalAccelerationTestCase_SteadyClimb(unittest.TestCase):
    def setUp(self):
        self.prob = Problem(HorizontalAccelerationTestGroup(num_nodes=9))
        self.prob.setup(check=True, force_alloc_complex=True)
        self.prob["thrust"] = np.ones((9,)) * (100 + 100 * GRAV_CONST * 0.02)
        self.prob["fltcond|singamma"] = np.ones((9,)) * 0.02
        self.prob.run_model()

    def test_steady_climb_flights(self):
        assert_near_equal(self.prob["accel_horiz"], np.zeros((9,)), tolerance=1e-10)

    def test_partials(self):
        partials = self.prob.check_partials(method="cs", out_stream=None)
        assert_check_partials(partials)


class HorizontalAccelerationTestCase_UnsteadyRunwayAccel(unittest.TestCase):
    def setUp(self):
        self.prob = Problem(HorizontalAccelerationTestGroup(num_nodes=9))
        self.prob.setup(check=True, force_alloc_complex=True)
        self.prob["braking"] = np.ones((9,)) * 0.03
        self.prob["lift"] = np.linspace(0, 150, 9) * GRAV_CONST
        self.prob["drag"] = np.ones((9,)) * 50
        self.prob.run_model()

    def test_accel_with_braking(self):
        drag = 50.0
        thrust = 100.0
        lift = 0.0
        mass = 100
        weight = mass * GRAV_CONST
        singamma = 0.0
        brakeforce = 0.03 * (weight - lift)
        slopeforce = weight * singamma
        accel_horz_actual = (thrust - drag - brakeforce - slopeforce) / mass
        assert_near_equal(self.prob["accel_horiz"][0], accel_horz_actual, tolerance=1e-10)

    def test_accel_with_braking_and_lift(self):
        drag = 50.0
        thrust = 100.0
        mass = 100
        weight = mass * GRAV_CONST
        singamma = 0.0
        lift = weight * 0.75
        brakeforce = 0.03 * (weight - lift)
        slopeforce = weight * singamma
        accel_horz_actual = (thrust - drag - brakeforce - slopeforce) / mass
        assert_near_equal(self.prob["accel_horiz"][4], accel_horz_actual, tolerance=1e-10)

    def test_accel_lift_exceeds_weight(self):
        drag = 50.0
        thrust = 100.0
        mass = 100
        weight = mass * GRAV_CONST
        singamma = 0.0
        # if lift exceeds weight (as it does here) no braking force is applied
        brakeforce = 0.0
        slopeforce = weight * singamma
        accel_horz_actual = (thrust - drag - brakeforce - slopeforce) / mass
        assert_near_equal(self.prob["accel_horiz"][-1], accel_horz_actual, tolerance=1e-10)

    def test_partials(self):
        partials = self.prob.check_partials(method="cs", out_stream=None)
        assert_check_partials(partials)


# TESTS FOR VerticalAcceleration ===================================


class VerticalAccelerationTestGroup(Group):
    def initialize(self):
        self.options.declare("num_nodes", default=9, desc="Number of mission analysis points to run")

    def setup(self):
        nn = self.options["num_nodes"]
        iv = self.add_subsystem("conditions", IndepVarComp(), promotes_outputs=["*"])
        iv.add_output("weight", val=np.ones((nn,)) * 100, units="kg")
        iv.add_output("lift", val=np.ones((nn,)) * 100 * GRAV_CONST, units="N")
        iv.add_output("thrust", val=np.ones((nn,)) * 100, units="N")
        iv.add_output("drag", val=np.ones((nn,)) * 100, units="N")
        iv.add_output("fltcond|singamma", val=np.zeros((nn,)), units=None)
        iv.add_output("fltcond|cosgamma", val=np.ones((nn,)), units=None)
        self.add_subsystem("va", VerticalAcceleration(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])


class VerticalAccelerationTestCase_SteadyLevel(unittest.TestCase):
    def setUp(self):
        self.prob = Problem(VerticalAccelerationTestGroup(num_nodes=9))
        self.prob.setup(check=True, force_alloc_complex=True)
        self.prob.run_model()

    def test_steady_level_flights(self):
        assert_near_equal(self.prob["accel_vert"], np.zeros((9,)), tolerance=1e-10)

    def test_partials(self):
        partials = self.prob.check_partials(method="cs", out_stream=None)
        assert_check_partials(partials)


class VerticalAccelerationTestCase_SteadyClimbing(unittest.TestCase):
    def setUp(self):
        self.prob = Problem(VerticalAccelerationTestGroup(num_nodes=9))
        self.prob.setup(check=True, force_alloc_complex=True)
        self.prob["fltcond|singamma"] = np.ones((9,)) * np.sin(0.02)
        self.prob["fltcond|cosgamma"] = np.ones((9,)) * np.cos(0.02)
        self.prob["lift"] = np.ones((9,)) * 100 * GRAV_CONST / np.cos(0.02)

        self.prob.run_model()

    def test_steady_climbing_flight(self):
        assert_near_equal(self.prob["accel_vert"], np.zeros((9,)), tolerance=1e-10)

    def test_partials(self):
        partials = self.prob.check_partials(method="cs", out_stream=None)
        assert_check_partials(partials)


class VerticalAccelerationTestCase_UnsteadyPullUp(unittest.TestCase):
    def setUp(self):
        self.prob = Problem(VerticalAccelerationTestGroup(num_nodes=9))
        self.prob.setup(check=True, force_alloc_complex=True)
        self.prob["lift"] = np.ones((9,)) * 100 * GRAV_CONST + 100
        self.prob.run_model()

    def test_unsteady_pullup(self):
        assert_near_equal(self.prob["accel_vert"], (100.0 / 100.0) * np.ones((9,)), tolerance=1e-10)

    def test_partials(self):
        partials = self.prob.check_partials(method="cs", out_stream=None)
        assert_check_partials(partials)


# TESTS FOR SteadyFlightCL ===================================


class SteadyFlightCLTestGroup(Group):
    def initialize(self):
        self.options.declare("num_nodes", default=9, desc="Number of mission analysis points to run")

    def setup(self):
        nn = self.options["num_nodes"]
        iv = self.add_subsystem("conditions", IndepVarComp(), promotes_outputs=["*"])
        iv.add_output("weight", val=np.ones((nn,)) * 100, units="kg")
        iv.add_output("fltcond|q", val=np.ones((nn,)) * 1000, units="Pa")
        iv.add_output("ac|geom|wing|S_ref", val=10, units="m**2")
        iv.add_output("fltcond|cosgamma", val=np.ones((nn,)), units=None)
        self.add_subsystem("cls", SteadyFlightCL(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])


class SteadyFlightCLTestCase_Level(unittest.TestCase):
    def setUp(self):
        self.prob = Problem(SteadyFlightCLTestGroup(num_nodes=9))
        self.prob.setup(check=True, force_alloc_complex=True)
        self.prob.run_model()

    def test_steady_level_flights(self):
        assert_near_equal(
            self.prob["fltcond|CL"], np.ones((9,)) * 100 * GRAV_CONST / 1000.0 / 10.0 / 1.0, tolerance=1e-10
        )

    def test_partials(self):
        partials = self.prob.check_partials(method="cs", out_stream=None)
        assert_check_partials(partials)


class SteadyFlightCLTestCase_Climb(unittest.TestCase):
    def setUp(self):
        self.prob = Problem(SteadyFlightCLTestGroup(num_nodes=9))
        self.prob.setup(check=True, force_alloc_complex=True)
        self.prob["fltcond|cosgamma"] = 0.98 * np.ones((9,))
        self.prob.run_model()

    def test_steady_level_flights(self):
        assert_near_equal(
            self.prob["fltcond|CL"], np.ones((9,)) * 100 * GRAV_CONST / 1000.0 / 10.0 * 0.98, tolerance=1e-10
        )

    def test_partials(self):
        partials = self.prob.check_partials(method="cs", out_stream=None)
        assert_check_partials(partials)


# TODO add TakeoffTransition and TakeoffClimb component unit tests
