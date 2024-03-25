import openmdao.api as om
from openconcept.propulsion import N3, CFM56
from openconcept.utilities import ElementMultiplyDivideComp


class RubberizedTurbofan(om.Group):
    """
    Optimized N+3 GTF engine deck (can optionally be switched to CFM56)
    generated as a surrogate of pyCycle data. This version adds the rated thrust
    input which adds a multiplier on thrust and fuel flow to enable continuous
    scaling of the engine power.

    This version of the engine can also be converted to hydrogen with the
    hydrogen option. It will scale the fuel flow by the ratio of LHV between
    jet fuel and hydrogen to deliver the same energy to the engine. This
    maintains the same thrust-specific energy consumption.

    NOTE: The CFM56 and N3 engine models only include data Mach 0.2 to 0.8
          and up to 35,000 ft. Outside that range, the model is unreliable.

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
    ac|propulsion|engine|rating : float
        Desired thrust rating (sea level static) of each engine; the CFM56 thrust
        and fuel flow are scaled by this value divided by 27,300, while the N+3 thrust
        and fuel flow are scaled by this value divided by 28,620 (scalar, lbf)

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
        LHV_scale_fac = LHV_ker / LHV_hy if hy else 1.0

        # Original rated SLS thrust of the engine to use in scaling factor
        if eng == "N3":
            # https://ntrs.nasa.gov/citations/20170005426
            orig_rated_thrust = 28620  # lbf
        elif eng == "CFM56":
            # https://web.archive.org/web/20161220201436/http://www.safran-aircraft-engines.com/file/download/fiche_cfm56-7b_ang.pdf
            orig_rated_thrust = 27300  # lbf

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

        # Scale thrust and fuel flow by the engine thrust rating and then divide by
        # the original sea level static rating of the engine
        scale = self.add_subsystem(
            "scale_engine",
            ElementMultiplyDivideComp(),
            promotes_inputs=[
                ("rating_thrust", "ac|propulsion|engine|rating"),
                ("rating_fuel_flow", "ac|propulsion|engine|rating"),
            ],
            promotes_outputs=["thrust", "fuel_flow"],
        )
        scale.add_equation(
            output_name="thrust",
            input_names=["unit_thrust", "rating_thrust", "orig_rating_thrust"],
            vec_size=[nn, 1, 1],
            input_units=["lbf", "lbf", "lbf"],
            divide=[False, False, True],
        )
        scale.add_equation(
            output_name="fuel_flow",
            input_names=["unit_fuel_flow", "rating_fuel_flow", "orig_rating_fuel_flow"],
            vec_size=[nn, 1, 1],
            input_units=["lbm/s", "lbf", "lbf"],
            divide=[False, False, True],
            scaling_factor=LHV_scale_fac,
        )
        self.set_input_defaults("scale_engine.orig_rating_thrust", orig_rated_thrust, units="lbf")
        self.set_input_defaults("scale_engine.orig_rating_fuel_flow", orig_rated_thrust, units="lbf")
        self.connect("engine_deck.thrust", "scale_engine.unit_thrust")
        self.connect("engine_deck.fuel_flow", "scale_engine.unit_fuel_flow")
