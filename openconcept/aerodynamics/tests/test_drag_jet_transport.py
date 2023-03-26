import unittest
import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openconcept.aerodynamics import ParasiteDragCoefficient_JetTransport


class ParasiteDragCoefficient_JetTransportTestCase(unittest.TestCase):
    def test_B738_clean(self):
        """
        737-800 cruise drag validation-ish case. Data from a combination of:
            - Technical site: http://www.b737.org.uk/techspecsdetailed.htm
            - Wikipedia: https://en.wikipedia.org/wiki/Boeing_737_Next_Generation#Specifications
        """
        prob = om.Problem()
        prob.model = om.Group()

        nn = 2

        dvs = prob.model.add_subsystem("dvs", om.IndepVarComp(), promotes_outputs=["*"])

        dvs.add_output("ac|geom|wing|S_ref", 124.6, units="m**2")
        dvs.add_output("ac|geom|wing|AR", 9.45)
        dvs.add_output("ac|geom|wing|taper", 0.159)
        dvs.add_output("ac|geom|wing|toverc", 0.12)  # guess

        dvs.add_output("ac|geom|hstab|S_ref", 32.78, units="m**2")
        dvs.add_output("ac|geom|hstab|AR", 6.16)
        dvs.add_output("ac|geom|hstab|taper", 0.203)
        dvs.add_output("ac|geom|hstab|toverc", 0.12)  # guess

        dvs.add_output("ac|geom|vstab|S_ref", 26.44, units="m**2")
        dvs.add_output("ac|geom|vstab|AR", 1.91)
        dvs.add_output("ac|geom|vstab|taper", 0.271)
        dvs.add_output("ac|geom|vstab|toverc", 0.12)  # guess

        dvs.add_output("ac|geom|fuselage|height", 3.76, units="m")
        dvs.add_output("ac|geom|fuselage|length", 38.08, units="m")
        dvs.add_output("ac|geom|fuselage|S_wet", 450, units="m**2")  # estimate using cylinder

        dvs.add_output("ac|geom|nacelle|length", 4.3, units="m")  # photogrammetry
        dvs.add_output("ac|geom|nacelle|S_wet", 27, units="m**2")  # estimate using cylinder and nacelle diameter of 2 m

        dvs.add_output("ac|propulsion|num_engines", 2)

        # Flight conditions at 37k ft Mach 0.875 cruise (ISA)
        dvs.add_output("fltcond|Utrue", np.full(nn, 450), units="kn")
        dvs.add_output("fltcond|rho", np.full(nn, 0.348), units="kg/m**3")
        dvs.add_output("fltcond|T", np.full(nn, 217), units="K")

        prob.model.add_subsystem("drag", ParasiteDragCoefficient_JetTransport(num_nodes=nn), promotes_inputs=["*"])

        prob.setup(force_alloc_complex=True)
        prob.run_model()

        # Check result is unchanged, actual 737-800 cruise CD0 Ben has in the B738 data file is 0.01925 (very close!)
        assert_near_equal(prob.get_val("drag.CD0"), [0.01930831, 0.01930831], tolerance=1e-6)

        partials = prob.check_partials(method="cs", compact_print=True, show_only_incorrect=False)
        assert_check_partials(partials)

    def test_B738_takeoff(self):
        """
        737-800 takeoff drag validation-ish case. Data from a combination of:
            - Technical site: http://www.b737.org.uk/techspecsdetailed.htm
            - Wikipedia: https://en.wikipedia.org/wiki/Boeing_737_Next_Generation#Specifications
        """
        prob = om.Problem()
        prob.model = om.Group()

        nn = 2

        dvs = prob.model.add_subsystem("dvs", om.IndepVarComp(), promotes_outputs=["*"])

        dvs.add_output("ac|geom|wing|S_ref", 124.6, units="m**2")
        dvs.add_output("ac|geom|wing|AR", 9.45)
        dvs.add_output("ac|geom|wing|c4sweep", 25, units="deg")
        dvs.add_output("ac|geom|wing|taper", 0.159)
        dvs.add_output("ac|geom|wing|toverc", 0.12)  # guess

        dvs.add_output("ac|geom|hstab|S_ref", 32.78, units="m**2")
        dvs.add_output("ac|geom|hstab|AR", 6.16)
        dvs.add_output("ac|geom|hstab|taper", 0.203)
        dvs.add_output("ac|geom|hstab|toverc", 0.12)  # guess

        dvs.add_output("ac|geom|vstab|S_ref", 26.44, units="m**2")
        dvs.add_output("ac|geom|vstab|AR", 1.91)
        dvs.add_output("ac|geom|vstab|taper", 0.271)
        dvs.add_output("ac|geom|vstab|toverc", 0.12)  # guess

        dvs.add_output("ac|geom|fuselage|height", 3.76, units="m")
        dvs.add_output("ac|geom|fuselage|length", 38.08, units="m")
        dvs.add_output("ac|geom|fuselage|S_wet", 450, units="m**2")  # estimate using cylinder

        dvs.add_output("ac|geom|nacelle|length", 4.3, units="m")  # photogrammetry
        dvs.add_output("ac|geom|nacelle|S_wet", 27, units="m**2")  # estimate using cylinder and nacelle diameter of 2 m

        dvs.add_output("ac|propulsion|num_engines", 2)

        dvs.add_output("ac|aero|takeoff_flap_deg", 15, units="deg")

        # Flight conditions at 37k ft Mach 0.875 cruise (ISA)
        dvs.add_output("fltcond|Utrue", np.full(nn, 100), units="kn")
        dvs.add_output("fltcond|rho", np.full(nn, 1.225), units="kg/m**3")
        dvs.add_output("fltcond|T", np.full(nn, 288), units="K")

        prob.model.add_subsystem(
            "drag", ParasiteDragCoefficient_JetTransport(num_nodes=nn, configuration="takeoff"), promotes_inputs=["*"]
        )

        prob.setup(force_alloc_complex=True)
        prob.run_model()

        # Check result is unchanged. Actual 737-800 takeoff CD0 is challenging to predict, but
        # estimates range between 0.03 (in B738 data file) and 0.08
        # (https://www.sesarju.eu/sites/default/files/documents/sid/2018/papers/SIDs_2018_paper_75.pdf),
        # so this value isn't unreasonable
        assert_near_equal(prob.get_val("drag.CD0"), [0.04531526, 0.04531526], tolerance=1e-6)

        partials = prob.check_partials(method="cs", compact_print=True, show_only_incorrect=False)
        assert_check_partials(partials)

    def test_B738_nowing(self):
        """
        737-800 cruise drag without the wing drag. Data from a combination of:
            - Technical site: http://www.b737.org.uk/techspecsdetailed.htm
            - Wikipedia: https://en.wikipedia.org/wiki/Boeing_737_Next_Generation#Specifications
        """
        prob = om.Problem()
        prob.model = om.Group()

        nn = 2

        dvs = prob.model.add_subsystem("dvs", om.IndepVarComp(), promotes_outputs=["*"])

        dvs.add_output("ac|geom|wing|S_ref", 124.6, units="m**2")

        dvs.add_output("ac|geom|hstab|S_ref", 32.78, units="m**2")
        dvs.add_output("ac|geom|hstab|AR", 6.16)
        dvs.add_output("ac|geom|hstab|taper", 0.203)
        dvs.add_output("ac|geom|hstab|toverc", 0.12)  # guess

        dvs.add_output("ac|geom|vstab|S_ref", 26.44, units="m**2")
        dvs.add_output("ac|geom|vstab|AR", 1.91)
        dvs.add_output("ac|geom|vstab|taper", 0.271)
        dvs.add_output("ac|geom|vstab|toverc", 0.12)  # guess

        dvs.add_output("ac|geom|fuselage|height", 3.76, units="m")
        dvs.add_output("ac|geom|fuselage|length", 38.08, units="m")
        dvs.add_output("ac|geom|fuselage|S_wet", 450, units="m**2")  # estimate using cylinder

        dvs.add_output("ac|geom|nacelle|length", 4.3, units="m")  # photogrammetry
        dvs.add_output("ac|geom|nacelle|S_wet", 27, units="m**2")  # estimate using cylinder and nacelle diameter of 2 m

        dvs.add_output("ac|propulsion|num_engines", 2)

        # Flight conditions at 37k ft Mach 0.875 cruise (ISA)
        dvs.add_output("fltcond|Utrue", np.full(nn, 450), units="kn")
        dvs.add_output("fltcond|rho", np.full(nn, 0.348), units="kg/m**3")
        dvs.add_output("fltcond|T", np.full(nn, 217), units="K")

        prob.model.add_subsystem(
            "drag", ParasiteDragCoefficient_JetTransport(num_nodes=nn, include_wing=False), promotes_inputs=["*"]
        )

        prob.setup(force_alloc_complex=True)
        prob.run_model()

        assert_near_equal(prob.get_val("drag.CD0"), [0.0134069, 0.0134069], tolerance=1e-6)

        partials = prob.check_partials(method="cs", compact_print=True, show_only_incorrect=False)
        assert_check_partials(partials)


if __name__ == "__main__":
    unittest.main()
