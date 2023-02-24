import openmdao.api as om
import numpy as np

from openconcept.energy_storage.hydrogen.structural import VacuumTankWeight
from openconcept.energy_storage.hydrogen.thermal import HeatTransferVacuumTank
from openconcept.energy_storage.hydrogen.boil_off import BoilOff
from openconcept.utilities.math.add_subtract_comp import AddSubtractComp


class LH2Tank(om.Group):
    """
    Model of a liquid hydrogen storage tank that is cylindrical with hemispherical
    end caps. It uses vacuum insulation with MLI and aluminum inner and outer tank
    walls. It includes thermal and boil-off models to simulate the heat entering
    the cryogenic propellant and how that heat causes the pressure in the tank
    to change.

          |--- length ---|
         . -------------- .         ---
      ,'                    `.       | radius
     /                        \      |
    |                          |    ---
     \                        /
      `.                    ,'
         ` -------------- '

    Inputs
    ------
    radius : float
        Inner radius of the cylinder and hemispherical end caps. This value
        does not include the insulation (scalar, m).
    length : float
        Length of JUST THE CYLIDRICAL part of the tank (scalar, m)
    Q_add : float
        Additional heat added to the tank other than the natural environment
        heat leak, make negative to remove heat (vector, W)
    m_dot_gas_in : float
        Mass flow rate of gaseous hydrogen into the ullage EXCLUDING any boil off (this is
        handled internally in this component); unlikely to ever be nonzero but left here
        to maintain generality (vector, kg/s)
    m_dot_gas_out : float
        Mass flow rate of gaseous hydrogen out of the ullage; could be for venting
        or gaseous hydrogen consumption (vector, kg/s)
    m_dot_liq_in : float
        Mass flow rate of liquid hydrogen into the bulk liquid; unlikely to ever be nonzero
        but left here to maintain generality (vector, kg/s)
    m_dot_liq_out : float
        Mass flow rate of liquid hydrogen out of the tank; this is where fuel being consumed
        is bookkept, assuming it is removed from the tank as a liquid (vector, kg/s)
    T_env : float
        External environment temperature (vector, K)
    N_layers : float
        Number of reflective sheild layers in the MLI, should be at least ~10 for model
        to retain reasonable accuracy (scalar, dimensionless)
    environment_design_pressure : float
        Maximum environment exterior pressure expected, probably ~1 atmosphere (scalar, Pa)
    max_expected_operating_pressure : float
        Maximum expected operating pressure of tank (scalar, Pa)
    vacuum_gap : float
        Thickness of vacuum gap, used to compute radius of outer vacuum wall, by default
        5 cm, which seems standard. This parameter only affects the radius of the outer
        shell, so it's probably ok to leave at 5 cm (scalar, m)

    Outputs
    -------
    m_gas : float
        Mass of the gaseous hydrogen in the tank ullage (vector, kg)
    m_liq : float
        Mass of liquid hydrogen in the tank (vector, kg)
    T_gas : float
        Temperature of the gaseous hydrogen in the ullage (vector, K)
    T_liq : float
        Temperature of the bulk liquid hydrogen (vector, K)
    P : float
        Pressure of the gas in the ullage (vector, Pa)
    fill_level : float
        Fraction of tank volume filled with liquid (vector, dimensionless)
    tank_weight : float
        Weight of the tank walls (scalar, kg)
    total_weight : float
        Current total weight of the liquid hydrogen, gaseous hydrogen, and tank structure (vector, kg)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    fill_level_init : float
        Initial fill level (in range 0-1) of the tank, default 0.95
        to leave space for boil off gas; 5% adopted from Millis et al. 2009 (scalar, dimensionless)
    ullage_T_init : float
        Initial temperature of gas in ullage, default 20 K (scalar, K)
    ullage_P_init : float
        Initial pressure of gas in ullage, default 150,000 Pa; ullage pressure must be higher than ambient
        to prevent air leaking in and creating a combustible mixture (scalar, Pa)
    liquid_T_init : float
        Initial temperature of bulk liquid hydrogen, default 20 K (scalar, K)
    heat_multiplier : float
        Multiplier on the output heat to account for heat through supports
        and other connections, by default 1.3 (scalar, dimensionless)
    weight_fudge_factor : float
        Multiplier on tank weight to account for supports, valves, etc., by default 1.1
    stiffening_multiplier : float
        Machining stiffeners into the inner side of the vacuum shell enhances its buckling
        performance, enabling weight reductions. The value provided in this option is a
        multiplier on the outer wall thickness. The default value of 0.8 is higher than it
        would be if it were purely empirically determined from Sullivan et al. 2006
        (https://ntrs.nasa.gov/citations/20060021606), but has been made much more
        conservative to fall more in line with ~60% gravimetric efficiency tanks
    inner_safety_factor : float
        Safety factor for sizing inner wall, by default 1.5
    inner_yield_stress : float
        Yield stress of inner wall material (Pa), by default Al 2014-T6 taken from Table IV of
        Sullivan et al. 2006 (https://ntrs.nasa.gov/citations/20060021606)
    inner_density : float
        Density of inner wall material (kg/m^3), by default Al 2014-T6 taken from Table IV of
        Sullivan et al. 2006 (https://ntrs.nasa.gov/citations/20060021606)
    outer_safety_factor : float
        Safety factor for sizing outer wall, by default 2
    outer_youngs_modulus : float
        Young's modulus of outer wall material (Pa), by default LiAl 2090 taken from Table XIII of
        Sullivan et al. 2006 (https://ntrs.nasa.gov/citations/20060021606)
    outer_density : float
        Density of outer wall material (kg/m^3), by default LiAl 2090 taken from Table XIII of
        Sullivan et al. 2006 (https://ntrs.nasa.gov/citations/20060021606)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of design points to run")
        self.options.declare("fill_level_init", default=0.95, desc="Initial fill level")
        self.options.declare("ullage_T_init", default=21.0, desc="Initial ullage temp (K)")
        self.options.declare("ullage_P_init", default=1.5e5, desc="Initial ullage pressure (Pa)")
        self.options.declare("liquid_T_init", default=20.0, desc="Initial bulk liquid temp (K)")
        self.options.declare("heat_multiplier", default=1.3, desc="Multiplier on heat leak")
        self.options.declare("weight_fudge_factor", default=1.1, desc="Weight multiplier to account for other stuff")
        self.options.declare("stiffening_multiplier", default=0.8, desc="Multiplier on wall thickness")
        self.options.declare("inner_safety_factor", default=1.5, desc="Safety factor on inner wall thickness")
        self.options.declare("inner_yield_stress", default=413.7e6, desc="Yield stress of inner wall material in Pa")
        self.options.declare("inner_density", default=2796.0, desc="Density of inner wall material in kg/m^3")
        self.options.declare("outer_safety_factor", default=2.0, desc="Safety factor on outer wall thickness")
        self.options.declare("outer_youngs_modulus", default=8.0e10, desc="Young's modulus of outer wall material, Pa")
        self.options.declare("outer_density", default=2699.0, desc="Density of outer wall material in kg/m^3")

    def setup(self):
        nn = self.options["num_nodes"]

        # Boil-off model
        self.add_subsystem(
            "boil_off",
            BoilOff(
                num_nodes=nn,
                fill_level_init=self.options["fill_level_init"],
                ullage_T_init=self.options["ullage_T_init"],
                ullage_P_init=self.options["ullage_P_init"],
                liquid_T_init=self.options["liquid_T_init"],
            ),
            promotes_inputs=["radius", "length", "m_dot_gas_in", "m_dot_gas_out", "m_dot_liq_in", "m_dot_liq_out"],
            promotes_outputs=["m_gas", "m_liq", "T_gas", "T_liq", ("P_gas", "P"), "fill_level"],
        )

        # Thermal model
        self.add_subsystem(
            "heat_leak",
            HeatTransferVacuumTank(num_nodes=nn, heat_multiplier=self.options["heat_multiplier"]),
            promotes_inputs=["T_env", "N_layers", "T_liq", "T_gas"],
        )
        self.add_subsystem(
            "total_heat",
            AddSubtractComp(output_name="Q", input_names=["heat_leak", "Q_add"], vec_size=[nn, nn], units="W"),
            promotes_inputs=["Q_add"],
        )
        self.connect("heat_leak.Q", "total_heat.heat_leak")
        self.connect("total_heat.Q", "boil_off.Q_dot")
        self.connect("boil_off.interface_params.A_wet", "heat_leak.A_wet")
        self.connect("boil_off.interface_params.A_dry", "heat_leak.A_dry")

        # Structural weight model
        self.add_subsystem(
            "structure",
            VacuumTankWeight(
                weight_fudge_factor=self.options["weight_fudge_factor"],
                stiffening_multiplier=self.options["stiffening_multiplier"],
                inner_safety_factor=self.options["inner_safety_factor"],
                inner_yield_stress=self.options["inner_yield_stress"],
                inner_density=self.options["inner_density"],
                outer_safety_factor=self.options["outer_safety_factor"],
                outer_youngs_modulus=self.options["outer_youngs_modulus"],
                outer_density=self.options["outer_density"],
            ),
            promotes_inputs=[
                "environment_design_pressure",
                "max_expected_operating_pressure",
                "vacuum_gap",
                "radius",
                "length",
                "N_layers",
            ],
            promotes_outputs=[("weight", "tank_weight")],
        )

        # Add all the weights
        self.add_subsystem(
            "sum_weight",
            AddSubtractComp(
                output_name="total_weight",
                input_names=["m_gas", "m_liq", "tank_weight"],
                vec_size=[nn, nn, 1],
                units="kg",
            ),
            promotes_inputs=["m_gas", "m_liq", "tank_weight"],
            promotes_outputs=["total_weight"],
        )

        # Set default for some inputs
        self.set_input_defaults("radius", 1.0, units="m")
        self.set_input_defaults("N_layers", 20)
        self.set_input_defaults("Q_add", np.zeros(nn), units="W")
        self.set_input_defaults("vacuum_gap", 5, units="cm")

        # Use block Gauss-Seidel solver for this component
        self.nonlinear_solver = om.NonlinearBlockGS(iprint=2, atol=1e-9, rtol=1e-9)
        self.linear_solver = om.LinearBlockGS(iprint=0)


if __name__ == "__main__":
    duration = 15.0  # hr
    nn = 51

    p = om.Problem()
    p.model.add_subsystem("tank", LH2Tank(num_nodes=nn, fill_level_init=0.95), promotes=["*"])

    p.setup()

    p.set_val("boil_off.integ.duration", duration, units="h")
    p.set_val("radius", 2.75, units="m")
    p.set_val("length", 2.0, units="m")
    p.set_val("Q_add", 1000.0, units="W")
    p.set_val("m_dot_gas_out", 0.0, units="kg/h")
    p.set_val("m_dot_liq_out", 100.0, units="kg/h")
    p.set_val("m_dot_gas_in", 0.0, units="kg/h")
    p.set_val("m_dot_liq_in", 0.0, units="kg/h")
    p.set_val("T_env", 300, units="K")
    p.set_val("N_layers", 10)
    p.set_val("environment_design_pressure", 1, units="atm")
    p.set_val("max_expected_operating_pressure", 3, units="bar")
    p.set_val("vacuum_gap", 0.1, units="m")

    p.run_model()

    om.n2(p)
    # p.model.list_outputs(print_arrays=True)

    import numpy as np
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 3, figsize=(9, 5))
    axs = axs.flatten()

    t = np.linspace(0, duration, nn)

    axs[0].plot(t, p.get_val("P", units="bar"))
    axs[0].set_xlabel("Time (hrs)")
    axs[0].set_ylabel("Ullage pressure (bar)")

    axs[1].plot(t, p.get_val("T_gas", units="K"))
    axs[1].set_xlabel("Time (hrs)")
    axs[1].set_ylabel("Ullage temperature (K)")

    axs[2].plot(t, p.get_val("T_liq", units="K"))
    axs[2].set_xlabel("Time (hrs)")
    axs[2].set_ylabel("Liquid temperature (K)")

    heat_leak = p.get_val("heat_leak.Q", units="W")
    Q_add = p.get_val("Q_add", units="W")
    axs[3].plot(t, heat_leak, label="Heat leak")
    axs[3].plot(t, Q_add, label="Additional heat")
    axs[3].set_xlabel("Time (hrs)")
    axs[3].set_ylabel("Heat leak (W)")
    axs[3].legend()

    axs[4].plot(t, 100 * p.get_val("fill_level"))
    axs[4].set_xlabel("Time (hrs)")
    axs[4].set_ylabel("Fill level (%)")

    m_liq_init = p.get_val("m_liq", units="kg")[0]
    m_dot_boil_off = (
        p.get_val("boil_off.boil_off_ode.m_dot_gas", units="kg/d")
        - p.get_val("m_dot_gas_in", units="kg/d")
        + p.get_val("m_dot_gas_out", units="kg/d")
    )
    m_dot_boil_off *= 100 / m_liq_init
    axs[5].plot(t, m_dot_boil_off)
    axs[5].set_xlabel("Time (hrs)")
    axs[5].set_ylabel("Boil off rate (% initial\nfuel weight per day)")

    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig("LH2_tank_plot.pdf")

    print(f"\n\nGravimetric efficiency: {m_liq_init / p.get_val('total_weight', units='kg')[0] * 100 :.1f}%")
