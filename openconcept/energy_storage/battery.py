import numpy as np
from openmdao.api import ExplicitComponent, Group
from openconcept.utilities import ElementMultiplyDivideComp, Integrator


class SOCBattery(Group):
    """
    Same as SimpleBattery but also tracks state of charge

    Inputs
    ------
    battery_weight : float
        Weight of the battery pack (scalar, kg)
    elec_load: float
        Electric power draw upstream (vector, W)
    SOC_initial : float
        Initial state of charge (default 1) (scalar, dimensionless)
    duration : float
        Length of the mission phase (corresponding to num_nodes) (scalar, s)

    Outputs
    -------
    SOC : float
        State of charge of the battery on a scale of 0 to 1 (vector, dimensionless)
    max_energy : float
        Total energy in the battery at 100% SOC (scalar, Wh)
    heat_out : float
        Waste heat produced (vector, W)
    component_cost : float
        Nonrecurring cost of the component (scalar, USD)
    component_sizing_margin : float
        Equal to 1 when producing full rated power (vector, dimensionless)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length; default 1)
    efficiency : float
        Shaft power efficiency. Sensible range 0.0 to 1.0 (default 1.0)
    specific_power : float
        Rated power per unit weight (default 5000, W/kg)
    default_specific_energy : float
        Battery energy per unit weight **NOTE UNITS** (default 300, !!!! Wh/kg)
        Can be set using variable input 'specific_energy' as well if doing a sweep
    cost_inc : float
        Cost per unit weight (default 50, USD/kg)
    cost_base : float
        Base cost (default 1 USD)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of flight/control conditions")
        self.options.declare("efficiency", default=1.0, desc="Efficiency (dimensionless)")
        self.options.declare("specific_power", default=5000.0, desc="Battery specific power (W/kg)")
        self.options.declare("specific_energy", default=300.0, desc="Battery spec energy")
        self.options.declare("cost_inc", default=50.0, desc="$ cost per kg")
        self.options.declare("cost_base", default=1.0, desc="$ cost base")

    def setup(self):
        nn = self.options["num_nodes"]

        eta_b = self.options["efficiency"]
        e_b = self.options["specific_energy"]
        p_b = self.options["specific_power"]
        cost_inc = self.options["cost_inc"]
        cost_base = self.options["cost_base"]

        self.add_subsystem(
            "batt_base",
            SimpleBattery(
                num_nodes=nn,
                efficiency=eta_b,
                specific_energy=e_b,
                specific_power=p_b,
                cost_inc=cost_inc,
                cost_base=cost_base,
            ),
            promotes_outputs=["*"],
            promotes_inputs=["*"],
        )

        # change in SOC over time is (- elec_load) / max_energy

        self.add_subsystem(
            "divider",
            ElementMultiplyDivideComp(
                output_name="dSOCdt",
                input_names=["elec_load", "max_energy"],
                vec_size=[nn, 1],
                scaling_factor=-1,
                divide=[False, True],
                input_units=["W", "kJ"],
            ),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )

        integ = self.add_subsystem(
            "ode_integ",
            Integrator(num_nodes=nn, method="simpson", diff_units="s", time_setup="duration"),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )
        integ.add_integrand(
            "SOC",
            rate_name="dSOCdt",
            start_name="SOC_initial",
            end_name="SOC_final",
            units=None,
            val=1.0,
            start_val=1.0,
        )


class SimpleBattery(ExplicitComponent):
    """
    A simple battery which tracks power limits and generates heat.

    Specific energy assumption INCLUDING internal losses should be used
    The efficiency parameter only generates heat

    Inputs
    ------
    battery_weight : float
        Weight of the battery pack (scalar, kg)
    elec_load: float
        Electric power draw upstream (vector, W)

    Outputs
    -------
    max_energy : float
        Total energy in the battery at 100% SOC (scalar, Wh)
    heat_out : float
        Waste heat produced (vector, W)
    component_cost : float
        Nonrecurring cost of the component (scalar, USD)
    component_sizing_margin : float
        Equal to 1 when producing full rated power (vector, dimensionless)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length; default 1)
    efficiency : float
        Shaft power efficiency. Sensible range 0.0 to 1.0 (default 1.0)
    specific_power : float
        Rated power per unit weight (default 5000, W/kg)
    specific_energy : float
        Battery energy per unit weight **NOTE UNITS** (default 300, !!!! Wh/kg)
        Can override this with variable input during a sweep (input specific_energy)
    cost_inc : float
        Cost per unit weight (default 50, USD/kg)
    cost_base : float
        Base cost (default 1 USD)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of flight/control conditions")
        self.options.declare("efficiency", default=1.0, desc="Efficiency (dimensionless)")
        self.options.declare("specific_power", default=5000.0, desc="Battery specific power (W/kg)")
        self.options.declare("specific_energy", default=300.0, desc="Battery spec energy")
        self.options.declare("cost_inc", default=50.0, desc="$ cost per kg")
        self.options.declare("cost_base", default=1.0, desc="$ cost base")

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("battery_weight", units="kg", desc="Total battery pack weight")
        self.add_input("elec_load", units="W", desc="Electrical load drawn", shape=(nn,))
        e_b = self.options["specific_energy"]
        self.add_input("specific_energy", units="W * h / kg", val=e_b)
        eta_b = self.options["efficiency"]
        cost_inc = self.options["cost_inc"]

        self.add_output("heat_out", units="W", desc="Waste heat out", shape=(nn,))
        self.add_output("component_cost", units="USD", desc="Battery cost")
        self.add_output("component_sizing_margin", desc="Load fraction of capable power", shape=(nn,))
        self.add_output("max_energy", units="W*h")

        self.declare_partials("heat_out", "elec_load", val=(1 - eta_b) * np.ones(nn), rows=range(nn), cols=range(nn))
        self.declare_partials("component_cost", "battery_weight", val=cost_inc)
        self.declare_partials("component_sizing_margin", "battery_weight")
        self.declare_partials("component_sizing_margin", "elec_load", rows=range(nn), cols=range(nn))
        self.declare_partials("max_energy", ["battery_weight", "specific_energy"])

    def compute(self, inputs, outputs):
        eta_b = self.options["efficiency"]
        p_b = self.options["specific_power"]
        e_b = inputs["specific_energy"]
        cost_inc = self.options["cost_inc"]
        cost_base = self.options["cost_base"]

        outputs["heat_out"] = inputs["elec_load"] * (1 - eta_b)
        outputs["component_cost"] = inputs["battery_weight"] * cost_inc + cost_base
        outputs["component_sizing_margin"] = inputs["elec_load"] / (p_b * inputs["battery_weight"])
        outputs["max_energy"] = inputs["battery_weight"] * e_b

    def compute_partials(self, inputs, J):
        p_b = self.options["specific_power"]
        e_b = inputs["specific_energy"]
        J["component_sizing_margin", "elec_load"] = 1 / (p_b * inputs["battery_weight"])
        J["component_sizing_margin", "battery_weight"] = -(inputs["elec_load"] / (p_b * inputs["battery_weight"] ** 2))
        J["max_energy", "battery_weight"] = e_b
        J["max_energy", "specific_energy"] = inputs["battery_weight"]
