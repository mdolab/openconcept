import numpy as np
import openmdao.api as om
from openconcept.energy_storage.hydrogen.LH2_tank import LH2Tank

# A validation case for the Aydelott experiments (described below)
def aydelott_validation(case):
    nn = 31
    p = om.Problem()
    p.model.add_subsystem(
        "tank",
        LH2Tank(
            num_nodes=nn,
            init_fill_level=case["fill_init"],
            ullage_T_init=case["T_init"],
            ullage_P_init=1e3 * case["P_init"],
        ),
        promotes=["*"],
    )
    p.model.add_subsystem(
        "Q_dot_bal",
        om.BalanceComp(
            "T_inf",
            units="K",
            eq_units="W",
            lhs_name="Q_model",
            rhs_name="Q_exp",
            rhs_val=case["Q_dot"] * np.ones(nn),
            val=300 * np.ones(nn),
        ),
        promotes_outputs=["T_inf"],
    )
    p.model.connect("heat.Q_wall.heat_into_walls", "Q_dot_bal.Q_model")
    p.model.linear_solver = om.DirectSolver()
    p.model.nonlinear_solver = om.NewtonSolver()
    p.model.nonlinear_solver.options["solve_subsystems"] = True
    p.model.nonlinear_solver.options["maxiter"] = 100
    p.model.nonlinear_solver.options["iprint"] = 0

    p.setup()

    p.set_val("design_pressure", 3.0, units="bar")
    p.set_val("m_dot_gas", np.zeros(nn), units="kg/s")
    p.set_val("radius", 23 / 2, units="cm")
    p.set_val("length", 0.0, units="m")
    p.set_val("insulation_thickness", 0.5, units="inch")
    p.set_val("duration", case["duration"], units="s")

    p.run_model()

    print("\n                         | Experiment | Model   | Error (%)")
    print(f"Initial pressure (kPa)   | {case['P_init']}    | {p.get_val('ullage.P', units='kPa')[0]:.2f}")
    print(
        f"Final pressure (kPa)     | {case['P_final']}    | {p.get_val('ullage.P', units='kPa')[-1]:.2f}  | {(p.get_val('ullage.P', units='kPa')[-1] - case['P_final'])/case['P_final']*100}"
    )
    if case["T_final"] != 0:
        print(f"Initial temperature (K)  | {case['T_init']}      | {p.get_val('ullage.T', units='K')[0]:.2f}")
        print(
            f"Final temperature (K)    | {case['T_final']}      | {p.get_val('ullage.T', units='K')[-1]:.2f}   | {(p.get_val('ullage.T', units='K')[-1] - case['T_final'])/case['T_final']*100}"
        )


# A validation case for the MHTB experiments (described below)
def MHTB_validation(case):
    nn = 31
    p = om.Problem()
    p.model.add_subsystem(
        "tank",
        LH2Tank(
            num_nodes=nn,
            init_fill_level=case["fill_init"],
            ullage_T_init=case["T_init"],
            ullage_P_init=1e3 * case["P_init"],
        ),
        promotes=["*"],
    )
    p.model.add_subsystem(
        "Q_dot_bal",
        om.BalanceComp(
            "T_inf",
            units="K",
            eq_units="W",
            lhs_name="Q_model",
            rhs_name="Q_exp",
            rhs_val=case["Q_dot"] * np.ones(nn),
            val=200 * np.ones(nn),
        ),
        promotes_outputs=["T_inf"],
    )
    p.model.connect("heat.Q_wall.heat_into_walls", "Q_dot_bal.Q_model")
    p.model.linear_solver = om.DirectSolver()
    p.model.nonlinear_solver = om.NewtonSolver()
    p.model.nonlinear_solver.options["solve_subsystems"] = True
    p.model.nonlinear_solver.options["maxiter"] = 100
    p.model.nonlinear_solver.options["iprint"] = 0

    p.setup()

    p.set_val("design_pressure", 3.0, units="bar")
    p.set_val("m_dot_gas", np.zeros(nn), units="kg/s")
    p.set_val("radius", 3.05 / 2, units="m")
    p.set_val("length", 3.05, units="m")
    p.set_val("insulation_thickness", 1000.0, units="inch")
    p.set_val("duration", case["duration"], units="s")

    p.run_model()

    # print(f"\nHeat into tank: {p.get_val('heat.Q_wall.heat_into_walls', units='W')} W")
    # print(f"Environment temp: {p.get_val('T_inf', units='K')} K")
    print("\n                         | Experiment | Model   | Error (%)")
    print(f"Initial pressure (kPa)   | {case['P_init']}      | {p.get_val('ullage.P', units='kPa')[0]:.2f}")
    print(
        f"Final pressure (kPa)     | {case['P_final']}    | {p.get_val('ullage.P', units='kPa')[-1]:.2f}  | {(p.get_val('ullage.P', units='kPa')[-1] - case['P_final'])/case['P_final']*100}"
    )
    print(f"Initial temperature (K)  | {case['T_init']}      | {p.get_val('ullage.T', units='K')[0]:.2f}")
    print(
        f"Final temperature (K)    | {case['T_final']}      | {p.get_val('ullage.T', units='K')[-1]:.2f}   | {(p.get_val('ullage.T', units='K')[-1] - case['T_final'])/case['T_final']*100}"
    )


# A validation case of the Ball tank designed for Boeing's Phantom Eye
# https://aip.scitation.org/doi/pdf/10.1063/1.4706990 other data
# https://www.ball.com/aerospace/Aerospace/media/Aerospace/Downloads/Aero_tech-comp_cryogen-fuel-storage.pdf?ext=.pdf 295 K environment
# https://www.airforce-technology.com/projects/phantomeyeunmannedae/ tank diameter
def Ball_HALE_validation(T_inf, Q_dot, boil_off):
    nn = 31
    p = om.Problem()
    p.model.add_subsystem(
        "tank",
        LH2Tank(num_nodes=nn, init_fill_level=0.95, ullage_T_init=21.0, ullage_P_init=85 * 6895.0),  # 85 PSIG
        promotes=["*"],
    )
    p.model.linear_solver = om.DirectSolver()
    p.model.nonlinear_solver = om.NewtonSolver()
    p.model.nonlinear_solver.options["solve_subsystems"] = True
    p.model.nonlinear_solver.options["maxiter"] = 100
    p.model.nonlinear_solver.options["iprint"] = 0

    p.setup()

    p.set_val("design_pressure", 95 * 6895.0, units="Pa")
    p.set_val("T_inf", T_inf * np.ones(nn), units="K")
    p.set_val("m_dot_gas", np.zeros(nn), units="kg/s")
    p.set_val("radius", 4, units="ft")
    p.set_val("length", 0.0, units="m")
    p.set_val("insulation_thickness", 4.6, units="inch")
    p.set_val("duration", 10, units="min")

    p.run_model()

    print("\n                     | Experiment       | Model")
    print(
        f"Heat leak (W)        | {Q_dot}          | {np.mean(p.get_val('heat.Q_wall.heat_into_walls', units='W')):.2f}"
    )
    print(f"Environment temp (K) |                  | {T_inf:.2f}")
    print(f"Boil off (lb/hr)     | {boil_off}     | {np.mean(p.get_val('boil_off.m_boil_off', units='lb/h')):.2f}")
    W_tank = (p.get_val("weight", units="lb") - p.get_val("W_LH2", units="lb") - p.get_val("W_GH2", units="lb"))[0]
    print(f"Tank weight (lb)     | 615              | {W_tank:.2f}")


if __name__ == "__main__":
    # This validation is based on experimental data from the MHTB tank, reported in section 5.1 of
    # Eugina Mendez Ramos' dissertation, available here: http://hdl.handle.net/1853/64797
    # The data is from a cylindrical tank with elliptical end caps (more details in the dissertation),
    # where the ullage starts at the specified temperature and pressure. The outside temperature is
    # set automatically to match the heat flux specified in the test. Because this test setup
    # specifies a heat flux, this does not validate the wall sizing or heat transfer with the
    # environment. However, wall sizing and weights are validated separately in the structural.py file.
    #
    # We can see from these cases that the reservoir is pressurizing more quickly than
    # during the experiment. The fidelity of the heat transfer from the environment to the ullage
    # and from the ullage to the liquid is not well modelled, which may explain the discrepancy.
    # Despite this difference, having some reasonable model of the ullage that can hold GH2 is better than
    # the simpler solution of venting any boil off gas immediately.
    cases = [
        {
            "fill_init": 0.9,
            "Q_dot": 54.1,  # W
            "P_init": 111.5,  # kPa
            "P_final": 138.012,  # kPa
            "T_init": 20.71,  # K
            "T_final": 23.65,  # K
            "duration": 19591,  # sec
        },
        {
            "fill_init": 0.9,
            "Q_dot": 20.2,
            "P_init": 111.5,
            "P_final": 136.202,
            "T_init": 20.71,
            "T_final": 23.08,
            "duration": 51138,
        },
        {
            "fill_init": 0.25,
            "Q_dot": 18.8,
            "P_init": 122.0,
            "P_final": 137.993,
            "T_init": 21.01,
            "T_final": 26.32,
            "duration": 66446,
        },
        {
            "fill_init": 0.5,
            "Q_dot": 51.0,
            "P_init": 111.5,
            "P_final": 138.013,
            "T_init": 20.71,
            "T_final": 27.78,
            "duration": 49869,
        },
    ]
    print("            MHTB TESTS             ")
    print("----------------------------------")
    for case in cases:
        MHTB_validation(case)

    # This test case validates the heat transfer model and boil off rate
    # for a tank that has flown on an aircraft. The experiment was imperfect
    # in many ways, most notably because they lost the vacuum in the MLI on their
    # input and output pipes to the tank. This resulted in an extra heat leak of
    # somewhere between 88 and 169 watts by their estimation. The reported boil
    # off rate for this extra heat leak between 740 and 765 W is 18.2 lb/hr
    # with a standard deviation of 0.2 lb/hr. The environment temperature is set
    # so that the heat leak is in this range. Our model gives a reasonably close
    # estimate for the boil off rate (25% low).
    #
    # For the experiment with an environment temperature of 295 and now erroneous
    # heat leaks, the boil off rate is unknown. Our model gives a reasonable estimate.
    #
    # Finally, the weight of the tank is knownm but the tank is made with an
    # aluminum structure as opposed to the carbon fiber composite one used in this
    # model. This may explain the low tank weight estimate of our model.
    print("\n\n      BALL PHANTOM EYE TEST       ")
    print("----------------------------------")
    Ball_HALE_validation(425, "740-765", "18.2 +/- 0.2")
    Ball_HALE_validation(295, "571-677", "Unknown     ")

    # These Aydelott tests (also reported in section 5.1 of Eugina Ramos' dissertation) are
    # from a spherical tank with a 0.23 m diameter. They tend to give results that are quite far
    # off, but I suspect this is due to a combination of the relatively high heat flux (since
    # the tank is rather small) and the final pressures being somewhat high. For these reasons,
    # this test case is commented out. The other validation cases give good results.
    #
    #                                      W               kPa                 kPa                K                 K                  sec
    cases = [{'fill_init': 0.514, 'Q_dot': 9.50, 'P_init': 105.462, 'P_final': 705.396, 'T_init': 20.28, 'T_final': 0.000, 'duration': 1441},
             {'fill_init': 0.349, 'Q_dot': 31.4, 'P_init': 122.161, 'P_final': 671.488, 'T_init': 20.12, 'T_final': 35.44, 'duration': 433},
             {'fill_init': 0.489, 'Q_dot': 34.1, 'P_init': 110.633, 'P_final': 741.566, 'T_init': 20.19, 'T_final': 31.17, 'duration': 400},
             {'fill_init': 0.765, 'Q_dot': 38.1, 'P_init': 127.036, 'P_final': 658.015, 'T_init': 20.51, 'T_final': 63.85, 'duration': 272},
             {'fill_init': 0.507, 'Q_dot': 58.5, 'P_init': 105.462, 'P_final': 717.103, 'T_init': 20.28, 'T_final': 0.000, 'duration': 222}]
    print("\n\n          AYDELOTT TESTS          ")
    print("----------------------------------")
    for case in cases:
        aydelott_validation(case)
