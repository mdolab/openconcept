import openmdao.api as om
import numpy as np

from openconcept.energy_storage.hydrogen.structural import VacuumTankWeight
from openconcept.utilities.math import Integrator
from openconcept.utilities.math.add_subtract_comp import AddSubtractComp


class LH2TankNoBoilOff(om.Group):
    """
    Model of a liquid hydrogen storage tank that is cylindrical with hemispherical
    end caps. It uses vacuum insulation with MLI and aluminum inner and outer tank
    walls. This model does not include the boil-off or thermal models, so it only
    estimates the weight and not pressure/temperature time histories.

    This model includes an integrator to compute the weight of liquid hydrogen
    remaining in the tank. It is the responsibility of the user to constrain
    the fill level to be greater than zero (or slightly more), since the integrator
    is perfectly happy with negative fill levels.

    .. code-block:: text

              |--- length ---|
             . -------------- .         ---
          ,'                    `.       | radius
         /                        \      |
        |                          |    ---
         \                        /
          `.                    ,'
             ` -------------- '

    WARNING: Do not modify or connect anything to the initial integrated delta state value
             "integ.delta_m_liq_initial". It must remain zero for the initial tank state to be
             the expected value. Set the initial tank condition using the fill_level_init option.

    Inputs
    ------
    radius : float
        Inner radius of the cylinder and hemispherical end caps. This value
        does not include the insulation (scalar, m).
    length : float
        Length of JUST THE CYLIDRICAL part of the tank (scalar, m)
    m_dot_liq : float
        Mass flow rate of liquid hydrogen out of the tank; positive indicates fuel leaving the tank (vector, kg/s)
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
    m_liq : float
        Mass of the liquid hydrogen in the tank (vector, kg)
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
    LH2_density : float
        Liquid hydrogen density, by default 70.85 kg/m^3 (scalar, kg/m^3)
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
        self.options.declare("LH2_density", default=70.85, desc="Liquid hydrogen density in kg/m^3")
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

        # The initial tank states are specified indirectly by the fill_level_init and LH2_density options, along
        # with the input tank radius and length. We can't connect a component directly to the integrator's
        # inputs because those initial values are linked between phases. Thus, we use a bit of a trick where
        # we actually integrate the amount of LH2 consumed since the beginning of the mission and then
        # add the correct initial values in the add_init_state_values component.
        integ = self.add_subsystem(
            "integ",
            Integrator(num_nodes=nn, diff_units="s", time_setup="duration", method="simpson"),
            promotes_inputs=["m_dot_liq"],
        )
        integ.add_integrand("delta_m_liq", rate_name="m_dot_liq", units="kg", val=0, start_val=0)

        self.add_subsystem(
            "add_init_state_values",
            InitialLH2MassModification(
                num_nodes=nn,
                fill_level_init=self.options["fill_level_init"],
                LH2_density=self.options["LH2_density"],
            ),
            promotes_inputs=["radius", "length"],
            promotes_outputs=["m_liq", "fill_level"],
        )
        self.connect("integ.delta_m_liq", "add_init_state_values.delta_m_liq")

        # Add all the weights
        self.add_subsystem(
            "sum_weight",
            AddSubtractComp(
                output_name="total_weight",
                input_names=["m_liq", "tank_weight"],
                vec_size=[nn, 1],
                units="kg",
            ),
            promotes_inputs=["m_liq", "tank_weight"],
            promotes_outputs=["total_weight"],
        )

        # Set default for some inputs
        self.set_input_defaults("radius", 1.0, units="m")
        self.set_input_defaults("N_layers", 20)
        self.set_input_defaults("vacuum_gap", 5, units="cm")


class InitialLH2MassModification(om.ExplicitComponent):
    """
    Subtract the change in LH2 mass from the initial value (computed internally).
    Also computes the fill level.

    Inputs
    ------
    radius : float
        Inner radius of the cylinder and hemispherical end caps. This value
        does not include the insulation (scalar, m).
    length : float
        Length of JUST THE CYLIDRICAL part of the tank (scalar, m)
    delta_m_liq : float
        Change in mass of liquid hydrogen in the tank since the beginning of the mission (vector, kg)

    Outputs
    -------
    m_liq : float
        Mass of liquid hydrogen in the tank (vector, kg)
    fill_level : float
        Fraction of the tank filled with LH2 (vector, dimensionless)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    fill_level_init : float
        Initial fill level (in range 0-1) of the tank, default 0.95
        to leave space for boil off gas; 5% adopted from Millis et al. 2009 (scalar, dimensionless)
    LH2_density : float
        Liquid hydrogen density, by default 70.85 kg/m^3 (scalar, kg/m^3)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of design points to run")
        self.options.declare("fill_level_init", default=0.95, desc="Initial fill level")
        self.options.declare("LH2_density", default=70.85, desc="Liquid hydrogen density in kg/m^3")

    def setup(self):
        nn = self.options["num_nodes"]

        self.add_input("radius", val=1.0, units="m")
        self.add_input("length", val=0.5, units="m")
        self.add_input("delta_m_liq", shape=(nn,), units="kg", val=0.0)
        self.add_output("m_liq", shape=(nn,), units="kg")
        self.add_output("fill_level", shape=(nn,))

        arng = np.arange(nn)
        self.declare_partials("m_liq", "delta_m_liq", rows=arng, cols=arng, val=-np.ones(nn))
        self.declare_partials("fill_level", "delta_m_liq", rows=arng, cols=arng)
        self.declare_partials(["m_liq", "fill_level"], ["radius", "length"], rows=arng, cols=np.zeros(nn))

    def compute(self, inputs, outputs):
        r = inputs["radius"]
        L = inputs["length"]
        fill_init = self.options["fill_level_init"]
        rho = self.options["LH2_density"]

        V_tank = 4 / 3 * np.pi * r**3 + np.pi * r**2 * L
        V_liq_init = V_tank * fill_init
        outputs["m_liq"] = V_liq_init * rho - inputs["delta_m_liq"]
        outputs["fill_level"] = outputs["m_liq"] / (rho * V_tank)

    def compute_partials(self, inputs, partials):
        r = inputs["radius"]
        L = inputs["length"]
        fill_init = self.options["fill_level_init"]
        rho = self.options["LH2_density"]

        V_tank = 4 / 3 * np.pi * r**3 + np.pi * r**2 * L
        Vtank_r = 4 * np.pi * r**2 + 2 * np.pi * r * L
        Vtank_L = np.pi * r**2
        V_liq_init = V_tank * fill_init
        m_liq = V_liq_init * rho - inputs["delta_m_liq"]

        partials["m_liq", "radius"] = Vtank_r * fill_init * rho
        partials["m_liq", "length"] = Vtank_L * fill_init * rho
        partials["fill_level", "delta_m_liq"] = -1 / (rho * V_tank)
        partials["fill_level", "radius"] = (
            partials["m_liq", "radius"] / (rho * V_tank) - m_liq / (rho * V_tank) ** 2 * rho * Vtank_r
        )
        partials["fill_level", "length"] = (
            partials["m_liq", "length"] / (rho * V_tank) - m_liq / (rho * V_tank) ** 2 * rho * Vtank_L
        )
