from openconcept.atmospherics import (
    TemperatureComp,
    PressureComp,
    DensityComp,
    DynamicPressureComp,
    TrueAirspeedComp,
    EquivalentAirspeedComp,
    SpeedOfSoundComp,
    MachNumberComp,
)
from openmdao.api import Group


class ComputeAtmosphericProperties(Group):
    """
    Computes pressure, density, temperature, dyn pressure, and true airspeed

    Inputs
    ------
    fltcond|h : float
        Altitude (vector, km)
    fltcond|Ueas : float
        Equivalent airspeed (vector, m/s)
    fltcond|TempIncrement : float
        Temperature increment for non-standard day (vector, degC)

    Outputs
    -------
    fltcond|p : float
        Pressure (vector, Pa)
    fltcond|rho : float
        Density (vector, kg/m3)
    fltcond|T : float
        Temperature (vector, K)
    fltcond|Utrue : float
        True airspeed (vector, m/s)
    fltcond|q : float
        Dynamic pressure (vector, Pa)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length) (default 1)
    true_airspeed_in : bool
        Flip to true if input vector is Utrue, not Ueas.
        If this is true, fltcond|Utrue will be an input and fltcond|Ueas will be an output.
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of mission analysis points to run")
        self.options.declare("true_airspeed_in", default=False, desc="Number of mission analysis points to run")

    def setup(self):
        nn = self.options["num_nodes"]
        tas_in = self.options["true_airspeed_in"]
        self.add_subsystem(
            "temp",
            TemperatureComp(num_nodes=nn),
            promotes_inputs=["fltcond|h", "fltcond|TempIncrement"],
            promotes_outputs=["fltcond|T"],
        )
        self.add_subsystem(
            "pressure", PressureComp(num_nodes=nn), promotes_inputs=["fltcond|h"], promotes_outputs=["fltcond|p"]
        )
        self.add_subsystem(
            "density",
            DensityComp(num_nodes=nn),
            promotes_inputs=["fltcond|p", "fltcond|T"],
            promotes_outputs=["fltcond|rho"],
        )
        self.add_subsystem(
            "speedofsound",
            SpeedOfSoundComp(num_nodes=nn),
            promotes_inputs=["fltcond|T"],
            promotes_outputs=["fltcond|a"],
        )
        if tas_in:
            self.add_subsystem(
                "equivair", EquivalentAirspeedComp(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"]
            )
        else:
            self.add_subsystem("trueair", TrueAirspeedComp(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem(
            "dynamicpressure", DynamicPressureComp(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"]
        )
        self.add_subsystem("machnumber", MachNumberComp(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])
