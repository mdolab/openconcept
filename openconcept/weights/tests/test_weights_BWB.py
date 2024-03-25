import unittest
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openconcept.weights import BWBEmptyWeight


class BWBEmptyWeightTestCase(unittest.TestCase):
    def test_BWB(self):
        """
        There is very limited validation data for BWB empty weights, so this
        just ensures the output value is reasonable.
        """
        prob = om.Problem()
        prob.model = om.Group()

        dvs = prob.model.add_subsystem("dvs", om.IndepVarComp(), promotes_outputs=["*"])
        dvs.add_output("ac|num_passengers_max", 406)
        dvs.add_output("ac|num_flight_deck_crew", 2)
        dvs.add_output("ac|num_cabin_crew", 9)
        dvs.add_output("ac|cabin_pressure", 10, units="psi")

        dvs.add_output("ac|aero|Mach_max", 0.9)
        dvs.add_output("ac|aero|Vstall_land", 135, units="kn")

        dvs.add_output("ac|geom|wing|S_ref", 100, units="m**2")
        dvs.add_output("ac|geom|wing|AR", 9)
        dvs.add_output("ac|geom|wing|c4sweep", 32.2, units="deg")
        dvs.add_output("ac|geom|wing|taper", 0.3)
        dvs.add_output("ac|geom|wing|toverc", 0.12)

        dvs.add_output("ac|geom|centerbody|S_cabin", 2800, units="ft**2")
        dvs.add_output("ac|geom|centerbody|S_aftbody", 1000, units="ft**2")
        dvs.add_output("ac|geom|centerbody|taper_aftbody", 0.6)

        dvs.add_output("ac|geom|V_pressurized", 2800 * 10, units="ft**3")

        dvs.add_output("ac|geom|maingear|length", 9.7, units="ft")
        dvs.add_output("ac|geom|maingear|num_wheels", 8)
        dvs.add_output("ac|geom|maingear|num_shock_struts", 2)

        dvs.add_output("ac|geom|nosegear|length", 6, units="ft")
        dvs.add_output("ac|geom|nosegear|num_wheels", 2)

        dvs.add_output("ac|propulsion|engine|rating", 74.1e3, units="lbf")
        dvs.add_output("ac|propulsion|num_engines", 2)

        dvs.add_output("ac|weights|MTOW", 500e3, units="lb")
        dvs.add_output("ac|weights|MLW", 400e3, units="lb")
        dvs.add_output("ac|weights|W_fuel_max", 200e3, units="lb")

        prob.model.add_subsystem("OEW", BWBEmptyWeight(structural_fudge=1.2, total_fudge=1.15), promotes_inputs=["*"])

        prob.setup(force_alloc_complex=True)
        prob.run_model()

        assert_near_equal(prob.get_val("OEW.OEW"), 276337.38740904, tolerance=1e-6)

        partials = prob.check_partials(method="cs", out_stream=None, compact_print=True, show_only_incorrect=False)
        assert_check_partials(partials)


if __name__ == "__main__":
    unittest.main()
