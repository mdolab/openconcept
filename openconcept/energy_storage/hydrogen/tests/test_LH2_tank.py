import unittest
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
import openmdao.api as om
from openconcept.energy_storage.hydrogen.LH2_tank import *


class LH2TankTestCase(unittest.TestCase):
    def test_simple(self):
        p = om.Problem()
        p.model = LH2Tank(ullage_P_init=101325.0, init_fill_level=0.95, ullage_T_init=25)
        p.model.linear_solver = om.DirectSolver()
        p.model.nonlinear_solver = om.NewtonSolver()
        p.model.nonlinear_solver.options["err_on_non_converge"] = True
        p.model.nonlinear_solver.options["solve_subsystems"] = True
        p.model.nonlinear_solver.options["maxiter"] = 20
        p.setup(force_alloc_complex=True)

        p.run_model()

        assert_near_equal(p.get_val("m_gas", units="kg"), 0.28298698, tolerance=1e-7)
        assert_near_equal(p.get_val("m_liq", units="kg"), 389.40636198, tolerance=1e-9)
        assert_near_equal(p.get_val("T_gas", units="K"), 25, tolerance=1e-9)
        assert_near_equal(p.get_val("T_liq", units="K"), 20, tolerance=1e-9)
        assert_near_equal(p.get_val("P", units="Pa"), 101325, tolerance=1e-9)
        assert_near_equal(p.get_val("fill_level"), 0.95, tolerance=1e-9)
        assert_near_equal(p.get_val("tank_weight", units="kg"), 252.70942027, tolerance=1e-9)
        assert_near_equal(p.get_val("total_weight", units="kg"), 642.39876923, tolerance=1e-9)
        assert_near_equal(
            p.get_val("total_weight", units="kg"),
            p.get_val("tank_weight", units="kg") + p.get_val("m_gas", units="kg") + p.get_val("m_liq", units="kg"),
            tolerance=1e-9,
        )

        partials = p.check_partials(method="cs", compact_print=True)
        assert_check_partials(partials)

    def test_time_history(self):
        duration = 15.0  # hr
        nn = 11

        p = om.Problem()
        p.model.add_subsystem("tank", LH2Tank(num_nodes=nn, init_fill_level=0.95), promotes=["*"])

        p.setup()

        p.set_val("boil_off.integ.duration", duration, units="h")
        p.set_val("radius", 2.75, units="m")
        p.set_val("length", 2.0, units="m")
        p.set_val("Q_add", np.linspace(1e3, 0.0, nn), units="W")
        p.set_val("m_dot_gas_out", -1.0, units="kg/h")
        p.set_val("m_dot_liq_out", 100.0, units="kg/h")
        p.set_val("m_dot_gas_in", 1.0, units="kg/h")
        p.set_val("m_dot_liq_in", 1.0, units="kg/h")
        p.set_val("T_env", 300, units="K")
        p.set_val("N_layers", 10)
        p.set_val("environment_design_pressure", 1, units="atm")
        p.set_val("max_expected_operating_pressure", 3, units="bar")
        p.set_val("vacuum_gap", 0.1, units="m")

        p.run_model()

        assert_near_equal(
            p.get_val("m_gas", units="kg"),
            np.array(
                [
                    11.657715010524457,
                    16.43953842688591,
                    22.08870949789011,
                    27.838243320482125,
                    33.479934932744925,
                    39.059925412833124,
                    44.5866737236875,
                    50.00668659847522,
                    55.263120440875454,
                    60.31815864032765,
                    65.14495106349747,
                ]
            ),
            tolerance=1e-7,
        )
        assert_near_equal(
            p.get_val("m_liq", units="kg"),
            np.array(
                [
                    9102.373711349434,
                    8952.091887933071,
                    8800.942716862068,
                    8649.693183039475,
                    8498.551491427213,
                    8347.471500947126,
                    8196.444752636271,
                    8045.524739761482,
                    7894.768305919082,
                    7744.2132677196305,
                    7593.886475296461,
                ]
            ),
            tolerance=1e-9,
        )
        assert_near_equal(
            p.get_val("T_gas", units="K"),
            np.array(
                [
                    21.0,
                    24.72365745623197,
                    25.23429373994755,
                    25.153574187569333,
                    25.195784410313603,
                    25.255375020792766,
                    25.235602063894643,
                    25.15035581935711,
                    25.030510035316535,
                    24.886151901225222,
                    24.714875081968668,
                ]
            ),
            tolerance=1e-9,
        )
        assert_near_equal(
            p.get_val("T_liq", units="K"),
            np.array(
                [
                    20.0,
                    20.072619663709517,
                    20.139655872466925,
                    20.201308117540623,
                    20.257741092749292,
                    20.309078797774912,
                    20.355411941321876,
                    20.396805710201278,
                    20.433305225258078,
                    20.464939176068956,
                    20.491722195096845,
                ]
            ),
            tolerance=1e-9,
        )
        assert_near_equal(
            p.get_val("P", units="Pa"),
            np.array(
                [
                    150000.0,
                    189548.4653410521,
                    209554.00473580387,
                    220455.37047851048,
                    228436.27880909137,
                    234353.7579511462,
                    238075.70480708423,
                    239886.40052419002,
                    240179.17802175425,
                    239204.54631162607,
                    237092.96600641473,
                ]
            ),
            tolerance=1e-9,
        )
        assert_near_equal(
            p.get_val("fill_level"),
            np.array(
                [
                    0.95,
                    0.9343082850873371,
                    0.9185119233636299,
                    0.9026918126596172,
                    0.886870479483351,
                    0.8710439137808385,
                    0.8552121229337393,
                    0.8393816649464999,
                    0.823559498622189,
                    0.8077506289187172,
                    0.7919589453560123,
                ]
            ),
            tolerance=1e-9,
        )
        assert_near_equal(
            p.get_val("total_weight", units="kg"),
            np.array(
                [
                    15184.82556927447,
                    15039.32556927447,
                    14893.82556927447,
                    14748.32556927447,
                    14602.82556927447,
                    14457.325569274471,
                    14311.825569274471,
                    14166.32556927447,
                    14020.82556927447,
                    13875.325569274471,
                    13729.825569274471,
                ]
            ),
            tolerance=1e-9,
        )
        assert_near_equal(p.get_val("tank_weight", units="kg"), 6070.794142914512, tolerance=1e-9)


if __name__ == "__main__":
    unittest.main()
