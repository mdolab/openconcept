# DATA FOR TBM T80
# Collected from various sources
# including SOCATA pilot manual

from openconcept.mission import IntegratorGroup
from openconcept.thermal import PerfectHeatTransferComp
from openconcept.utilities import ElementMultiplyDivideComp, AddSubtractComp

data = dict()
ac = dict()
# ==AERO==================================
aero = dict()
aero["CLmax_TO"] = {"value": 2.0}

polar = dict()
polar["e"] = {"value": 0.801}
polar["CD0_TO"] = {"value": 0.03}
polar["CD0_cruise"] = {"value": 0.01925}

aero["polar"] = polar
ac["aero"] = aero

# ==GEOMETRY==============================
geom = dict()
wing = dict()
wing["S_ref"] = {"value": 124.6, "units": "m**2"}
wing["AR"] = {"value": 9.45}
wing["c4sweep"] = {"value": 25.0, "units": "deg"}
wing["taper"] = {"value": 0.159}
wing["toverc"] = {"value": 0.12}
geom["wing"] = wing

hstab = dict()
hstab["S_ref"] = {"value": 32.78, "units": "m**2"}
hstab["c4_to_wing_c4"] = {"value": 17.9, "units": "m"}
geom["hstab"] = hstab

vstab = dict()
vstab["S_ref"] = {"value": 26.44, "units": "m**2"}
geom["vstab"] = vstab

nosegear = dict()
nosegear["length"] = {"value": 3, "units": "ft"}
geom["nosegear"] = nosegear

maingear = dict()
maingear["length"] = {"value": 4, "units": "ft"}
geom["maingear"] = maingear

thermal = dict()
thermal["hx_to_battery_length"] = {"value": 20, "units": "ft"}
thermal["hx_to_battery_diameter"] = {"value": 2, "units": "inch"}
thermal["hx_to_motor_length"] = {"value": 10, "units": "ft"}
thermal["hx_to_motor_diameter"] = {"value": 2, "units": "inch"}
geom["thermal"] = thermal

ac["geom"] = geom

# ==WEIGHTS========================
weights = dict()
weights["MTOW"] = {"value": 79002, "units": "kg"}
weights["OEW"] = {"value": 0.530 * 79002, "units": "kg"}
weights["W_fuel_max"] = {"value": 0.266 * 79002, "units": "kg"}
weights["MLW"] = {"value": 66349, "units": "kg"}

ac["weights"] = weights

# ==PROPULSION=====================
propulsion = dict()
engine = dict()
engine["rating"] = {"value": 27000, "units": "lbf"}
propulsion["engine"] = engine

motor = dict()
motor["rating"] = {"value": 1.0, "units": "MW"}
propulsion["motor"] = motor

battery = dict()
battery["weight"] = {"value": 2000, "units": "kg"}
propulsion["battery"] = battery

thermal = dict()
hx = dict()
hx["n_wide_cold"] = {"value": 750, "units": None}
hx["n_long_cold"] = {"value": 3, "units": None}
hx["n_tall"] = {"value": 50, "units": None}
hx["pump_power_rating"] = {"value": 5.0, "units": "kW"}
thermal["hx"] = hx
hx_motor = dict()
hx_motor["n_wide_cold"] = {"value": 750, "units": None}
hx_motor["n_long_cold"] = {"value": 3, "units": None}
hx_motor["n_tall"] = {"value": 10, "units": None}
hx_motor["nozzle_area"] = {"value": 40, "units": "inch**2"}
hx_motor["pump_power_rating"] = {"value": 5.0, "units": "kW"}

thermal["hx_motor"] = hx_motor
hx_fault_prot = dict()
hx_fault_prot["n_long_cold"] = {"value": 1.5, "units": None}
thermal["hx_fault_prot"] = hx_fault_prot

heatpump = dict()
heatpump["power_rating"] = {"value": 30, "units": "kW"}
thermal["heatpump"] = heatpump
propulsion["thermal"] = thermal

ac["propulsion"] = propulsion

# Some additional parameters needed by the empirical weights tools
ac["num_passengers_max"] = {"value": 180}
ac["q_cruise"] = {"value": 212.662, "units": "lb*ft**-2"}

design_mission = dict()
design_mission["TOW"] = {"value": 79002, "units": "kg"}
ac["design_mission"] = design_mission
data["ac"] = ac


class MotorFaultProtection(IntegratorGroup):
    """
    The fault protection at the motor consumes power and produces heat
    It consumes glycol/water at 3gpm and needs 40C inflow temp
    So it has to be stacked first in the motor HX duct unfortunately
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1)
        self.options.declare("efficiency", default=0.995)

    def setup(self):
        nn = self.options["num_nodes"]
        eff = self.options["efficiency"]
        self.add_subsystem(
            "coolant_temps", PerfectHeatTransferComp(num_nodes=nn), promotes=["T_in", "mdot_coolant", "T_out"]
        )
        self.add_subsystem(
            "electrical_loss",
            ElementMultiplyDivideComp(
                "elec_load", input_names=["motor_power"], vec_size=nn, scaling_factor=(1 - eff), input_units=["W"]
            ),
            promotes=["*"],
        )
        self.connect("elec_load", "coolant_temps.q")
        self.add_subsystem(
            "duct_pressure",
            AddSubtractComp(
                "delta_p_stack", input_names=["delta_p_motor_hx", "delta_p_fault_prot_hx"], vec_size=nn, units="Pa"
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "heat_transfer",
            AddSubtractComp(
                "heat_transfer",
                input_names=["heat_transfer_motor_hx", "heat_transfer_fault_prot_hx"],
                vec_size=nn,
                units="W",
            ),
            promotes=["*"],
        )
