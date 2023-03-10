import openmdao.api as om
from openconcept.propulsion import N3, CFM56
from openconcept.utilities import ElementMultiplyDivideComp


class RubberizedTurbofan(om.Group):
    """
    Optimized N+3 GTF engine deck (can optionally be switched to CFM56)
    generated as a surrogate of pyCycle data. This version adds the N_engines
    input which is a multiplier on thrust and fuel flow to enable continuous
    scaling of the engine power.

    This version of the engine can also be converted to hydrogen with the
    hydrogen option. It will scale the fuel flow by the ratio of LHV between
    jet fuel and hydrogen to deliver the same energy to the engine. This
    maintains the same thrust-specific energy consumption.

    Inputs
    ------
    throttle: float
        Engine throttle. Controls power and fuel flow.
        Produces 100% of rated power at throttle = 1.
        Should be in range 0 to 1 or slightly above 1.
        (vector, dimensionless)
    fltcond|h: float
        Altitude
        (vector, dimensionless)
    fltcond|M: float
        Mach number
        (vector, dimensionless)
    N_engines : float
        A multiplier on thrust and fuel flow outputs to
        enable changing the "size" of the engine
        (scalar, dimensionless)

    Outputs
    -------
    thrust : float
        Thrust developed by the engine (vector, lbf)
    fuel_flow : float
        Fuel flow consumed (vector, lbm/s)
    surge_margin or T4 : float
        Surge margin if engine is "N3" or T4 if engine is "CFM56" (vector, percent)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length; default 1)
    hydrogen : bool
        True to convert fuel_flow to an equivalent fuel flow of hydrogen by
        multiplying by the ratio of lower heating value between jet fuel
        and hydrogen. Otherwise it will keep the fuel flow from the jet
        fuel-powered version of the engine deck, by default False
    engine : str
        Engine deck to use, valid options are "N3" and "CFM56", by default "N3"
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, types=int, desc="Number of analysis points to run")
        self.options.declare("hydrogen", default=False, types=bool, desc="Convert fuel flow to hydrogen energy")
        self.options.declare("engine", default="N3", values=["N3", "CFM56"], desc="Engine deck to use")

    def setup(self):
        nn = self.options["num_nodes"]
        hy = self.options["hydrogen"]
        eng = self.options["engine"]

        # Scale the fuel flow by the ratio of LHV if hydrogen is specified, else leave as is
        LHV_ker = 43  # MJ/kg, Jet A-1 specific energy
        LHV_hy = 120.0  # Mj/kg, hydrogen specific energy
        scale_fac = LHV_ker / LHV_hy if hy else 1.0

        # Engine deck
        if eng == "N3":
            self.add_subsystem(
                "engine_deck",
                N3(num_nodes=nn),
                promotes_inputs=["throttle", "fltcond|h", "fltcond|M"],
                promotes_outputs=["surge_margin"],
            )
        elif eng == "CFM56":
            self.add_subsystem(
                "engine_deck",
                CFM56(num_nodes=nn),
                promotes_inputs=["throttle", "fltcond|h", "fltcond|M"],
                promotes_outputs=["T4"],
            )
        else:
            raise ValueError(f"{eng} is not a recognized engine")

        # Scale thrust and fuel flow by the number of engines
        scale = self.add_subsystem(
            "scale_engine",
            ElementMultiplyDivideComp(),
            promotes_inputs=[("N_engines_thrust", "N_engines"), ("N_engines_fuel_flow", "N_engines")],
            promotes_outputs=["thrust", "fuel_flow"],
        )
        scale.add_equation(
            output_name="thrust",
            input_names=["unit_thrust", "N_engines_thrust"],
            vec_size=nn,
            input_units=["lbf", None],
        )
        scale.add_equation(
            output_name="fuel_flow",
            input_names=["unit_fuel_flow", "N_engines_fuel_flow"],
            vec_size=nn,
            input_units=["lbm/s", None],
            scaling_factor=scale_fac,
        )
        self.connect("engine_deck.thrust", "scale_engine.unit_thrust")
        self.connect("engine_deck.fuel_flow", "scale_engine.unit_fuel_flow")
