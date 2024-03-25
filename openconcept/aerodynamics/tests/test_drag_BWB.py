import unittest
import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openconcept.aerodynamics import ParasiteDragCoefficient_BWB


class ParasiteDragCoefficient_BWBTestCase(unittest.TestCase):
    def test_BWB_clean(self):
        prob = om.Problem()
        prob.model = om.Group()

        nn = 2

        dvs = prob.model.add_subsystem("dvs", om.IndepVarComp(), promotes_outputs=["*"])

        dvs.add_output("ac|geom|wing|S_ref", 100, units="m**2")
        dvs.add_output("ac|propulsion|engine|rating", 74.1e3, units="lbf")
        dvs.add_output("ac|propulsion|num_engines", 2)
        dvs.add_output("ac|geom|nacelle|length", 4.3, units="m")
        dvs.add_output("ac|geom|nacelle|S_wet", 27, units="m**2")  # estimate using cylinder and nacelle diameter of 2 m

        # Flight conditions at 37k ft Mach 0.875 cruise (ISA)
        dvs.add_output("fltcond|Utrue", np.full(nn, 450), units="kn")
        dvs.add_output("fltcond|rho", np.full(nn, 0.348), units="kg/m**3")
        dvs.add_output("fltcond|T", np.full(nn, 217), units="K")

        prob.model.add_subsystem("drag", ParasiteDragCoefficient_BWB(num_nodes=nn), promotes_inputs=["*"])

        prob.setup(force_alloc_complex=True)
        prob.run_model()

        # This does not include the wing so it shouldn't be a shock that it's a low
        assert_near_equal(prob.get_val("drag.CD0"), [0.00211192, 0.00211192], tolerance=1e-5)

        partials = prob.check_partials(method="cs", compact_print=True, show_only_incorrect=False)
        assert_check_partials(partials)

    def test_BWB_takeoff(self):
        prob = om.Problem()
        prob.model = om.Group()

        nn = 2

        dvs = prob.model.add_subsystem("dvs", om.IndepVarComp(), promotes_outputs=["*"])

        dvs.add_output("ac|geom|wing|S_ref", 100, units="m**2")
        dvs.add_output("ac|geom|wing|c4sweep", 32.2, units="deg")
        dvs.add_output("ac|propulsion|engine|rating", 74.1e3, units="lbf")
        dvs.add_output("ac|propulsion|num_engines", 2)
        dvs.add_output("ac|geom|nacelle|length", 4.3, units="m")
        dvs.add_output("ac|geom|nacelle|S_wet", 27, units="m**2")  # estimate using cylinder and nacelle diameter of 2 m
        dvs.add_output("ac|aero|takeoff_flap_deg", 15, units="deg")

        # Flight conditions at 37k ft Mach 0.875 cruise (ISA)
        dvs.add_output("fltcond|Utrue", np.full(nn, 100), units="kn")
        dvs.add_output("fltcond|rho", np.full(nn, 1.225), units="kg/m**3")
        dvs.add_output("fltcond|T", np.full(nn, 288), units="K")

        prob.model.add_subsystem(
            "drag", ParasiteDragCoefficient_BWB(num_nodes=nn, configuration="takeoff"), promotes_inputs=["*"]
        )

        prob.setup(force_alloc_complex=True)
        prob.run_model()

        assert_near_equal(prob.get_val("drag.CD0"), [0.01225882, 0.01225882], tolerance=1e-5)

        partials = prob.check_partials(method="cs", compact_print=True, show_only_incorrect=False)
        assert_check_partials(partials)


if __name__ == "__main__":
    unittest.main()
