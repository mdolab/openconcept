"""
This script compares time histories from OpenConcept's LH2 tank to experimental
data and results from other simulation tools. The validation data is presented
in Eugina Mendez Ramos's thesis (http://hdl.handle.net/1853/64797).

See more potential validation data here:
    - https://ntrs.nasa.gov/citations/19920009200
    - https://ntrs.nasa.gov/citations/19670028965 (I think Mendez Ramos covers this one)
    - https://ntrs.nasa.gov/citations/19910011011 (I think Mendez Ramos has this one too)
"""
import numpy as np
import matplotlib.pyplot as plt
import openmdao.api as om
from openconcept.energy_storage.hydrogen import LH2Tank
from openconcept.energy_storage.hydrogen.boil_off import BoilOff
from validation_data.validation_data_boiloff import MHTB_results


# A validation case for the MHTB experiments (described below) using the new boil off model
def new_MHTB_validation_model(fill_init, T_init, P_init, Q_dot, duration, T_init_liq=20.0):
    """
    Return a setup and run OpenMDAO model for the MHTB tank. Data for the tank dimensions
    is given in Table 5.1a of the Mendez Ramos thesis. In reality it is a cylinder
    with equal height and diameter and 2:1 elliptical end caps. Here, we approximate
    that as a sphere because the current OpenConcept model supports only hemispherical
    end caps. Because the heat into the tank is given, this model sets the environment
    temperature to achieve the desired heat.

    Parameters
    ----------
    fill_init : float
        Fraction of tank filled with liquid hydrogen at the start of the simulation.
    T_init : float
       Ullage temperature in K at the start of the simulation.
    P_init : float
        Ullage pressure in Pa at the start of the simulation.
    Q_dot : float
        Total heat in W entering the tank throughout the simulation.
    duration : float
        Duration of the simulation in seconds.
    T_init_liq : float, optional
        Initial temperature of the liquid (K), by default 20 K
    """
    nn = 41

    # MHTB tank geometry (approximate as a sphere)
    r = 3.05 / 2
    L = 0.0

    p = om.Problem()
    p.model.add_subsystem(
        "tank",
        BoilOff(
            num_nodes=nn,
            init_fill_level=fill_init,
            ullage_T_init=T_init,
            ullage_P_init=P_init,
            liquid_T_init=T_init_liq,
        ),
        promotes=["*"],
    )

    p.setup()

    # Set values for test case
    p.set_val("Q_dot", Q_dot, units="W")
    p.set_val("radius", r, units="m")
    p.set_val("length", L, units="m")
    p.set_val("integ.duration", duration, units="s")

    p.run_model()

    om.n2(p, show_browser=False)

    return p


# A validation case for the MHTB experiments (described below)
def MHTB_validation_model(fill_init, T_init, P_init, Q_dot, duration):
    """
    Return a setup and run OpenMDAO model for the MHTB tank. Data for the tank dimensions
    is given in Table 5.1a of the Mendez Ramos thesis. In reality it is a cylinder
    with equal height and diameter and 2:1 elliptical end caps. Here, we approximate
    that as a sphere because the current OpenConcept model supports only hemispherical
    end caps. Because the heat into the tank is given, this model sets the environment
    temperature to achieve the desired heat.

    Parameters
    ----------
    fill_init : float
        Fraction of tank filled with liquid hydrogen at the start of the simulation.
    T_init : float
       Ullage temperature in K at the start of the simulation.
    P_init : float
        Ullage pressure in Pa at the start of the simulation.
    Q_dot : float
        Total heat in W entering the tank throughout the simulation.
    duration : float
        Duration of the simulation in seconds.
    """
    nn = 101
    p = om.Problem()
    p.model.add_subsystem(
        "tank",
        LH2Tank(
            num_nodes=nn,
            init_fill_level=fill_init,
            ullage_T_init=T_init,
            ullage_P_init=P_init,
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
            rhs_val=Q_dot * np.ones(nn),
            val=200 * np.ones(nn),
        ),
        promotes_outputs=["T_inf"],
    )
    p.model.connect("heat.Q_wall.heat_into_walls", "Q_dot_bal.Q_model")
    p.model.linear_solver = om.DirectSolver()
    p.model.nonlinear_solver = om.NewtonSolver()
    p.model.nonlinear_solver.options["solve_subsystems"] = True
    p.model.nonlinear_solver.options["maxiter"] = 100
    p.model.nonlinear_solver.options["iprint"] = 2

    p.setup()

    p.set_val("design_pressure", 3.0, units="bar")
    p.set_val("m_dot_gas", np.zeros(nn), units="kg/s")
    p.set_val("radius", 3.05 / 2, units="m")
    p.set_val("length", 0.0, units="m")

    # Set the insulation to be thick enough such that it is possible
    # for the BalanceComp to find an environment temperature that meets
    # the heat entering requested
    p.set_val("insulation_thickness", 1000.0, units="inch")
    p.set_val("duration", duration, units="s")

    p.run_model()

    return p


def run_MHTB_validation(fname=None, new_model=True):
    """
    Run a validation case based on the MHTB tank.

    Parameters
    ----------
    fname : str, optional
        Specify a file name to save the plot to. If not specified, will show on screen.
    """
    # Case names and data from page 118 of Mendez Ramos's thesis
    MHTB_cases = {
        "P263981D": {
            "P_init": 111.5e3,
            "T_init": 20.71,
            "fill_init": 0.9,
            "Q_dot": 54.1,
            "duration": 19_591,
            "T_init_liq": 20.66,
        },
        "P263968E": {
            "P_init": 111.5e3,
            "T_init": 20.71,
            "fill_init": 0.9,
            "Q_dot": 20.2,
            "duration": 51_138,
            "T_init_liq": 20.62,
        },
        "P263968K": {
            "P_init": 122e3,
            "T_init": 21.01,
            "fill_init": 0.25,
            "Q_dot": 18.8,
            "duration": 66_446,
            "T_init_liq": 20.97,
        },
        "P263981T": {
            "P_init": 111.5e3,
            "T_init": 20.71,
            "fill_init": 0.5,
            "Q_dot": 51.0,
            "duration": 49_869,
            "T_init_liq": 20.70,
        },
    }

    fig, axs = plt.subplots(len(MHTB_cases), 2, figsize=(9, 3 * len(MHTB_cases)), tight_layout=True)
    axs = np.atleast_2d(axs)

    for i, case in enumerate(MHTB_cases.keys()):
        # Simulate the result
        if new_model:
            p = new_MHTB_validation_model(**MHTB_cases[case])
            pressure = p.get_val("P_gas", units="Pa")
            temp_ullage = p.get_val("T_gas", units="K")
            t = np.linspace(0, p.get_val("integ.duration", units="s"), pressure.size)
        else:
            args = MHTB_cases[case]
            del args["T_init_liq"]
            p = MHTB_validation_model(**args)
            pressure = p.get_val("ullage.P", units="Pa")
            temp_ullage = p.get_val("ullage.T", units="K")
            t = np.linspace(0, p.get_val("duration", units="s"), pressure.size)

        # Plot the result from the test
        for j, param in enumerate(["P", "Tg"]):
            scaler = 1e-3 if param == "P" else 1.0
            # Experimental result
            axs[i, j].plot(
                1 / 60**2 * MHTB_results[case]["test_t" + param],
                MHTB_results[case]["test_" + param] * scaler,
                label="Test",
            )

            # OpenConcept model
            axs[i, j].plot(1 / 60**2 * t, (pressure if param == "P" else temp_ullage) * scaler, label="OpenConcpet")

            # Eugina Mendez Ramos's model
            axs[i, j].plot(
                1 / 60**2 * MHTB_results[case]["EBM_model_t" + param],
                MHTB_results[case]["EBM_model_" + param] * scaler,
                label="EBM",
            )

            axs[i, j].set_xlim((0.0, t[-1] / 60**2))
            axs[i, j].legend()
            axs[i, j].set_ylabel("Pressure (kPa)" if param == "P" else "Temperature (K)")
            axs[i, j].set_xlabel("Time (hrs)")
            axs[i, j].set_title(f"Test ID {case}")
            axs[i, j].spines[["top", "right"]].set_visible(False)

    fig.suptitle("MHTB Validation Cases")

    if fname:
        fig.savefig(fname)
    else:
        plt.show()


if __name__ == "__main__":
    run_MHTB_validation("MHTB_validation.pdf")
