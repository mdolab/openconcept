import unittest
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openconcept.weights import JetTransportEmptyWeight


class JetTransportEmptyWeightTestCase(unittest.TestCase):
    def test_B732(self):
        """
        737-200 validation case. This is useful since Roskam has the component weight
        breakdown in the appendix. Parameter data from a combination of:
        - Technical site: http://www.b737.org.uk/techspecsdetailed.htm
        - Wikipedia: https://en.wikipedia.org/wiki/Boeing_737#Specifications
        """
        prob = om.Problem()
        prob.model = om.Group()

        dvs = prob.model.add_subsystem("dvs", om.IndepVarComp(), promotes_outputs=["*"])
        dvs.add_output("ac|num_passengers_max", 136)
        dvs.add_output("ac|num_flight_deck_crew", 2)
        dvs.add_output("ac|num_cabin_crew", 3)
        dvs.add_output("ac|cabin_pressure", 8.5, units="psi")

        dvs.add_output("ac|aero|Mach_max", 0.82)
        dvs.add_output("ac|aero|Vstall_land", 115, units="kn")  # estimate

        dvs.add_output("ac|geom|wing|S_ref", 102, units="m**2")
        dvs.add_output("ac|geom|wing|AR", 8.83)
        dvs.add_output("ac|geom|wing|c4sweep", 25, units="deg")
        dvs.add_output("ac|geom|wing|taper", 0.266)
        dvs.add_output("ac|geom|wing|toverc", 0.12)  # guess

        dvs.add_output("ac|geom|hstab|S_ref", 28.99, units="m**2")
        dvs.add_output("ac|geom|hstab|AR", 4.15)
        dvs.add_output("ac|geom|hstab|c4sweep", 30, units="deg")
        dvs.add_output("ac|geom|hstab|c4_to_wing_c4", 29.54 / 2, units="m")  # guess (half of fuselage length)

        dvs.add_output("ac|geom|vstab|S_ref", 20.81, units="m**2")
        dvs.add_output("ac|geom|vstab|AR", 1.64)
        dvs.add_output("ac|geom|vstab|c4sweep", 35, units="deg")
        dvs.add_output("ac|geom|vstab|toverc", 0.12)  # guess
        dvs.add_output("ac|geom|vstab|c4_to_wing_c4", 29.54 / 2, units="m")  # guess (half of fuselage length)

        dvs.add_output("ac|geom|fuselage|height", 3.76, units="m")
        dvs.add_output("ac|geom|fuselage|length", 29.54, units="m")
        dvs.add_output("ac|geom|fuselage|S_wet", 350, units="m**2")  # estimate using cylinder

        dvs.add_output("ac|geom|maingear|length", 1.8, units="m")
        dvs.add_output("ac|geom|maingear|num_wheels", 4)
        dvs.add_output("ac|geom|maingear|num_shock_struts", 2)

        dvs.add_output("ac|geom|nosegear|length", 1.3, units="m")
        dvs.add_output("ac|geom|nosegear|num_wheels", 2)

        dvs.add_output("ac|propulsion|engine|rating", 16e3, units="lbf")
        dvs.add_output("ac|propulsion|num_engines", 2)

        dvs.add_output("ac|weights|MTOW", 115500, units="lb")
        dvs.add_output("ac|weights|MLW", 48534, units="lb")
        dvs.add_output("ac|weights|W_fuel_max", 34718, units="lb")

        prob.model.add_subsystem("OEW", JetTransportEmptyWeight(), promotes_inputs=["*"])

        prob.setup(force_alloc_complex=True)
        prob.run_model()

        # Check that the result is unchanged, the actual 737-200 OEW is 60,210 lbs according to Roskam
        assert_near_equal(prob.get_val("OEW.OEW"), 60724.22319076, tolerance=1e-6)

        partials = prob.check_partials(method="cs", out_stream=None, compact_print=True, show_only_incorrect=False)
        assert_check_partials(partials)

    def test_B738(self):
        """
        737-800 validation case. Data from a combination of:
            - Technical site: http://www.b737.org.uk/techspecsdetailed.htm
            - Wikipedia: https://en.wikipedia.org/wiki/Boeing_737_Next_Generation#Specifications
        """
        prob = om.Problem()
        prob.model = om.Group()

        dvs = prob.model.add_subsystem("dvs", om.IndepVarComp(), promotes_outputs=["*"])
        dvs.add_output("ac|num_passengers_max", 189)
        dvs.add_output("ac|num_flight_deck_crew", 2)
        dvs.add_output("ac|num_cabin_crew", 4)
        dvs.add_output("ac|cabin_pressure", 8.95, units="psi")

        dvs.add_output("ac|aero|Mach_max", 0.82)
        dvs.add_output("ac|aero|Vstall_land", 115, units="kn")  # estimate

        dvs.add_output("ac|geom|wing|S_ref", 124.6, units="m**2")
        dvs.add_output("ac|geom|wing|AR", 9.45)
        dvs.add_output("ac|geom|wing|c4sweep", 25, units="deg")
        dvs.add_output("ac|geom|wing|taper", 0.159)
        dvs.add_output("ac|geom|wing|toverc", 0.12)  # guess

        dvs.add_output("ac|geom|hstab|S_ref", 32.78, units="m**2")
        dvs.add_output("ac|geom|hstab|AR", 6.16)
        dvs.add_output("ac|geom|hstab|c4sweep", 30, units="deg")
        dvs.add_output("ac|geom|hstab|c4_to_wing_c4", 38.08 / 2, units="m")  # guess (half of fuselage length)

        dvs.add_output("ac|geom|vstab|S_ref", 26.44, units="m**2")
        dvs.add_output("ac|geom|vstab|AR", 1.91)
        dvs.add_output("ac|geom|vstab|c4sweep", 35, units="deg")
        dvs.add_output("ac|geom|vstab|toverc", 0.12)  # guess
        dvs.add_output("ac|geom|vstab|c4_to_wing_c4", 38.08 / 2, units="m")  # guess (half of fuselage length)

        dvs.add_output("ac|geom|fuselage|height", 3.76, units="m")
        dvs.add_output("ac|geom|fuselage|length", 38.08, units="m")
        dvs.add_output("ac|geom|fuselage|S_wet", 450, units="m**2")  # estimate using cylinder

        dvs.add_output("ac|geom|maingear|length", 1.8, units="m")
        dvs.add_output("ac|geom|maingear|num_wheels", 4)
        dvs.add_output("ac|geom|maingear|num_shock_struts", 2)

        dvs.add_output("ac|geom|nosegear|length", 1.3, units="m")
        dvs.add_output("ac|geom|nosegear|num_wheels", 2)

        dvs.add_output("ac|propulsion|engine|rating", 24.2e3, units="lbf")
        dvs.add_output("ac|propulsion|num_engines", 2)

        dvs.add_output("ac|weights|MTOW", 79002, units="kg")
        dvs.add_output("ac|weights|MLW", 66349, units="kg")
        dvs.add_output("ac|weights|W_fuel_max", 21000, units="kg")

        prob.model.add_subsystem("OEW", JetTransportEmptyWeight(), promotes_inputs=["*"])

        prob.setup(force_alloc_complex=True)
        prob.run_model()

        # Check that the result is unchanged, the actual 737-800 OEW is 91,300 lbs
        assert_near_equal(prob.get_val("OEW.OEW"), 91377.02987079, tolerance=1e-6)

    def test_B789(self):
        """
        787-9 validation case. Data from https://en.wikipedia.org/wiki/Boeing_787_Dreamliner#Specifications.
        """
        prob = om.Problem()
        prob.model = om.Group()

        dvs = prob.model.add_subsystem("dvs", om.IndepVarComp(), promotes_outputs=["*"])
        dvs.add_output("ac|num_passengers_max", 406)
        dvs.add_output("ac|num_flight_deck_crew", 2)
        dvs.add_output("ac|num_cabin_crew", 9)
        dvs.add_output("ac|cabin_pressure", 11.8, units="psi")  # 6,000 ft cabin altitude

        dvs.add_output("ac|aero|Mach_max", 0.9)
        dvs.add_output("ac|aero|Vstall_land", 135, units="kn")  # estimate

        dvs.add_output("ac|geom|wing|S_ref", 377, units="m**2")
        dvs.add_output("ac|geom|wing|AR", 9.59)
        dvs.add_output("ac|geom|wing|c4sweep", 32.2, units="deg")
        dvs.add_output("ac|geom|wing|taper", 0.18)
        dvs.add_output("ac|geom|wing|toverc", 0.13)  # guess

        dvs.add_output("ac|geom|hstab|S_ref", 77.3, units="m**2")
        dvs.add_output("ac|geom|hstab|AR", 5)
        dvs.add_output("ac|geom|hstab|c4sweep", 36, units="deg")
        dvs.add_output("ac|geom|hstab|c4_to_wing_c4", 61 / 2, units="m")  # guess (half of fuselage length)

        dvs.add_output("ac|geom|vstab|S_ref", 39.7, units="m**2")
        dvs.add_output("ac|geom|vstab|AR", 1.7)
        dvs.add_output("ac|geom|vstab|c4sweep", 40, units="deg")
        dvs.add_output("ac|geom|vstab|toverc", 0.1)  # guess
        dvs.add_output("ac|geom|vstab|c4_to_wing_c4", 61 / 2, units="m")  # guess (half of fuselage length)

        dvs.add_output("ac|geom|fuselage|height", 5.9, units="m")
        dvs.add_output("ac|geom|fuselage|length", 61, units="m")
        dvs.add_output("ac|geom|fuselage|S_wet", 1131, units="m**2")  # estimate using cylinder

        dvs.add_output("ac|geom|maingear|length", 9.7, units="ft")
        dvs.add_output("ac|geom|maingear|num_wheels", 8)
        dvs.add_output("ac|geom|maingear|num_shock_struts", 2)

        dvs.add_output("ac|geom|nosegear|length", 6, units="ft")
        dvs.add_output("ac|geom|nosegear|num_wheels", 2)

        dvs.add_output("ac|propulsion|engine|rating", 74.1e3, units="lbf")
        dvs.add_output("ac|propulsion|num_engines", 2)

        dvs.add_output("ac|weights|MTOW", 561.5e3, units="lb")
        dvs.add_output("ac|weights|MLW", 425e3, units="lb")
        dvs.add_output("ac|weights|W_fuel_max", 223673, units="lb")

        prob.model.add_subsystem("OEW", JetTransportEmptyWeight(), promotes_inputs=["*"])

        prob.setup(force_alloc_complex=True)
        prob.run_model()

        # Check that the result is unchanged, the actual 787-9 OEW is 284,000 lbs
        assert_near_equal(prob.get_val("OEW.OEW"), 289119.77474878, tolerance=1e-6)


if __name__ == "__main__":
    unittest.main()
