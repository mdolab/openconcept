import unittest
import numpy
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openconcept.propulsion import RubberizedTurbofan
from parameterized import parameterized_class


@parameterized_class([{"engine": "N3"}, {"engine": "CFM56"}])
class RubberizedTurbofanTestCase(unittest.TestCase):
    def test_default(self):
        """
        Check that when N_engines is one and hydrogen is False the outputs
        are the same as the surrogate model.
        """
        p = om.Problem()
        p.model.add_subsystem("model", RubberizedTurbofan(engine=self.engine), promotes=["*"])
        p.setup()
        p.run_model()

        assert_near_equal(p.get_val("engine_deck.thrust", units="N"), p.get_val("thrust", units="N"))
        assert_near_equal(p.get_val("engine_deck.fuel_flow", units="kg/s"), p.get_val("fuel_flow", units="kg/s"))

    def test_multiple_engines(self):
        """
        Check that N_engines scales properly.
        """
        N_eng = 3.4

        p = om.Problem()
        p.model.add_subsystem("model", RubberizedTurbofan(num_nodes=3, engine=self.engine), promotes=["*"])
        p.setup()
        p.set_val("N_engines", N_eng)
        p.run_model()

        assert_near_equal(p.get_val("engine_deck.thrust", units="N") * N_eng, p.get_val("thrust", units="N"))
        assert_near_equal(
            p.get_val("engine_deck.fuel_flow", units="kg/s") * N_eng, p.get_val("fuel_flow", units="kg/s")
        )

    def test_hydrogen(self):
        """
        Check that hydrogen scales the fuel flow properly.
        """
        p = om.Problem()
        p.model.add_subsystem(
            "model", RubberizedTurbofan(num_nodes=3, engine=self.engine, hydrogen=True), promotes=["*"]
        )
        p.setup()
        p.run_model()

        fuel_ratio = 43 / 120

        assert_near_equal(p.get_val("engine_deck.thrust", units="N"), p.get_val("thrust", units="N"))
        assert_near_equal(
            p.get_val("engine_deck.fuel_flow", units="kg/s") * fuel_ratio, p.get_val("fuel_flow", units="kg/s")
        )

    def test_all_together(self):
        """
        Check hydrogen and N_engines together.
        """
        N_eng = 3.4

        p = om.Problem()
        p.model.add_subsystem(
            "model", RubberizedTurbofan(num_nodes=3, engine=self.engine, hydrogen=True), promotes=["*"]
        )
        p.setup()
        p.set_val("N_engines", N_eng)
        p.run_model()

        fuel_ratio = 43 / 120

        assert_near_equal(p.get_val("engine_deck.thrust", units="N") * N_eng, p.get_val("thrust", units="N"))
        assert_near_equal(
            p.get_val("engine_deck.fuel_flow", units="kg/s") * fuel_ratio * N_eng, p.get_val("fuel_flow", units="kg/s")
        )


if __name__ == "__main__":
    unittest.main()
