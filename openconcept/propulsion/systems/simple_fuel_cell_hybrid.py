import numpy as np
import openmdao.api as om
from scipy.interpolate import interp1d
from openmdao.api import ExplicitComponent, Group, IndepVarComp, ExecComp
from openconcept.mission import BasicMission, IntegratorGroup
from openconcept.thermal.ducts import ExplicitIncompressibleDuctWithInletLoss, ImplicitCompressibleDuct_ExternalHX
from openconcept.thermal.hose import SimpleHose
from openconcept.thermal.pump import SimplePump
from openconcept.utilities import DVLabel
from openconcept.propulsion import SimplePropeller, SimpleMotor
from openconcept.thermal import (
    LiquidCooledComp,
    CoolantReservoir,
    HXGroup,
)
from openconcept.utilities.linearinterp import LinearInterpolator


class MassFlowFloor(ExplicitComponent):
    """Experimental safeguard developed while testing the compressible-flow model.

    Applies a lower bound to the mass-flow vector passed from the implicit
    compressible duct into the heat-exchanger model. This prototype is not
    validated for production use: review the physical floor, behavior near the
    threshold, and derivative treatment before relying on it in a simulation.
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1)
        self.options.declare("floor", default=0.05)

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("mdot_in", shape=(nn,), units="kg/s")
        self.add_output("mdot_out", shape=(nn,), units="kg/s")
        # Finite differences are used because fmax is not complex-step differentiable
        # at the floor threshold. Validate this derivative choice before production use.
        self.declare_partials("mdot_out", "mdot_in", method="fd")

    def compute(self, inputs, outputs):
        # Prevent near-zero or negative trial flows from reaching the downstream
        # HX calculations during nonlinear-solver iterations.
        outputs["mdot_out"] = np.fmax(inputs["mdot_in"], self.options["floor"])


class BoPAirSystem(ExplicitComponent):
    """
    Models the Balance of Plant (BOP) air compressor and recovery turbine 
    based on the fuel cell operating point and ambient flight conditions.

    Inputs
    ------
    air_flow : float
        Cathode air mass flow delivered to the fuel-cell stack (vector, kg/s)
    fltcond|Utrue : float
        True airspeed used to characterize inlet conditions (vector, m/s)
    fltcond|rho : float
        Ambient air density (vector, kg/m**3)
    fltcond|T : float
        Ambient/static air temperature (vector, K)
    fltcond|p : float
        Ambient/static air pressure (vector, Pa)
    compressor_power_rating : float
        Installed compressor electrical power rating for sizing weight (scalar, W)
    fltcond|Mach : float
        Flight Mach number (vector, dimensionless)

    Outputs
    -------
    compressor_power_demand : float
        Compressor shaft/electrical power demand (vector, W)
    turbine_power_recovery : float
        Turbine power recovered from cathode exhaust flow (vector, W)
    component_weight : float
        Estimated BOP compressor/turbine package mass (scalar, kg)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    """
    def initialize(self):
        self.options.declare("num_nodes", default=1)

    def setup(self):
        nn = self.options["num_nodes"]
        
        # Inputs
        self.add_input("air_flow", shape=nn, units="kg/s")
        self.add_input("fltcond|Utrue", shape=nn, units="m/s")
        self.add_input("fltcond|rho", shape=nn, units="kg/m**3")
        self.add_input("fltcond|T", shape=nn, units="K")
        self.add_input("fltcond|p", shape=nn, units="Pa")
        self.add_input("compressor_power_rating", units="W", val=300000)
        self.add_input("turbine_power_rating", units="W", val=100000)
        self.add_input("intercooler_mass", units="kg", val=0.0)
        self.add_input("humidifier_mass", units="kg", val=0.0)
        self.add_input("fltcond|Mach", shape=nn, units=None)

        # Outputs
        self.add_output("compressor_power_demand", shape=nn, units="W") # Power consumed (+)
        self.add_output("turbine_power_recovery", shape=nn, units="W")  # Power recovered (-)
        self.add_output("component_weight", units="kg")

        # Constants from the thesis methodology
        self.lambda_air = 2.0
        self.gamma_air = 1.4
        self.M_air = 28.97e-3
        self.F = 96458.0
        self.eta_comp_s = 0.76
        self.eta_comp_pc = 0.95
        self.eta_comp_elec = 0.94
        self.eta_comp_mech = 0.97
        self.eta_pr = 0.75
        self.eta_turb_mech = 0.95
        self.cp_air = 1005.0
        self.Rsp_air = 287.0
        self.delta_p_filt = 500.0
        self.delta_p_ic = 10000.0
        self.delta_p_hm_d = 5000.0
        self.delta_p_hm_w = 5000.0
        self.T_stack = 358.0
        self.p_fc = 150000.0
        self.pc_in = 160000.0
        self.specific_power_comp = 1770.0 # W/kg
        self.specific_power_turb = 12000.0 # W/kg, 

    def compute(self, inputs, outputs):
        T_2 = inputs["fltcond|T"] + (np.sqrt(self.gamma_air * self.Rsp_air * inputs["fltcond|T"])**2) / (2 * self.cp_air)
        p_2_stat = inputs["fltcond|p"] * (T_2 / inputs["fltcond|T"])**((self.gamma_air) / (self.gamma_air - 1))
        p_2 = inputs["fltcond|p"] + self.eta_pr * (p_2_stat - inputs["fltcond|p"])
        p_3 = p_2 - self.delta_p_filt
        p_4 = self.pc_in + self.delta_p_ic + self.delta_p_hm_d
        p_5 = self.p_fc
        p_6 = inputs["fltcond|p"] + self.delta_p_hm_w

        # 1. Enthalpy change for compression
        delta_h = 1 / self.eta_comp_s * self.cp_air * T_2 * ((p_4 / p_3)**(self.Rsp_air / self.cp_air) - 1)
        
        # 2. Compressor power demand
        outputs["compressor_power_demand"] = (delta_h * inputs["air_flow"]) / (self.eta_comp_elec * self.eta_comp_mech)
        
        # 3. Turbine power recovery
        T_turb_out = self.eta_turb_mech * self.T_stack * ((p_6 / p_5)**((self.gamma_air - 1) / self.gamma_air) - 1)
        outputs["turbine_power_recovery"] = self.cp_air * T_turb_out * inputs["air_flow"] * self.eta_turb_mech
        
        # 4. Weight estimation
        compressor_weight = inputs["compressor_power_rating"] / self.specific_power_comp
        turbine_weight = inputs["turbine_power_rating"] / self.specific_power_turb

        outputs["component_weight"] = (
            compressor_weight
            + turbine_weight
            + inputs["intercooler_mass"]
            + inputs["humidifier_mass"]
        )

class LH2Evaporator(ExplicitComponent):
    """
    Models the LH2 stream as a cold sink for the fuel-cell waste heat.
    The hydrogen is stored at very low temperature and warms up while absorbing heat.

    Inputs
    ------
    heat_output : float
        Total thermal power/heat generated by the fuel cell stack (vector, W)
    fuel_flow : float
        Mass flow rate of liquid hydrogen supplied to the system (vector, kg/s)

    Outputs
    -------
    heat_removed : float
        Thermal power absorbed by the hydrogen flow through latent heat and 
        sensible heating up to the stack limit temperature (vector, W)
    effective_heat_load : float
        Remaining unabsorbed thermal load that must be rejected by secondary 
        cooling systems (vector, W)

    Options
    -------
    num_nodes : int
        Number of analysis evaluation points to compute in parallel (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1)

    def setup(self):
        nn = self.options["num_nodes"]

        self.add_input("heat_output", shape=nn, units="W")
        self.add_input("fuel_flow", shape=nn, units="kg/s")

        self.add_output("heat_removed", shape=nn, units="W")
        self.add_output("effective_heat_load", shape=nn, units="W")

    def compute(self, inputs, outputs):
        heat_output = inputs["heat_output"]
        fuel_flow = inputs["fuel_flow"]


        self.cp_h2_latent = 446.300  # kJ/(kg K)
        self.cp_h2_gas = 14.300  # k J/(kg K)
        self.temp_fc_in = 350.0  # K, upper temperature limit before the stream is no longer treated as cryogenic
        self.temp_evap = 20.0  # K, LH2 boiling point

        q_h2_heating = fuel_flow * (self.cp_h2_latent * (self.temp_fc_in - self.temp_evap) * self.cp_h2_gas)

        outputs["heat_removed"] = q_h2_heating
        outputs["effective_heat_load"] = heat_output - q_h2_heating

class LH2TankSizing(om.ExplicitComponent):
    """
    Calculates the required hydrogen tank mass and total storage system mass 
    based on the total mission fuel consumption and a target gravimetric index.

    The Gravimetric Index (GI) is defined as:
        GI = mass_fuel / (mass_fuel + mass_tank_structure)
    
    Rearranging to find the structural tank mass:
        mass_tank_structure = mass_fuel * (1 - GI) / GI

    Inputs
    ------
    fuel_consumed : float
        Total mass of liquid hydrogen consumed during the mission (scalar, kg)

    Outputs
    -------
    tank_structural_mass : float
        Dry mass of the hydrogen tank structure, insulation, and system (scalar, kg)
    total_tank_system_mass : float
        Combined mass of the fuel and the dry tank structure (scalar, kg)

    Options
    -------
    gravimetric_index : float
        Ratio of stored hydrogen mass to total storage system mass (default: 0.20)
    """

    def initialize(self):
        self.options.declare(
            "gravimetric_index", 
            default=0.2, 
            desc="Gravimetric index of the tank (mass_fuel / mass_total)"
        )

    def setup(self):
        self.add_input("fuel_consumed", val=0.0, units="kg")

        self.add_output("tank_structural_mass", val=0.0, units="kg")
        self.add_output("total_tank_system_mass", val=0.0, units="kg")

        self.declare_partials(of="*", wrt="fuel_consumed", method="exact")

    def compute(self, inputs, outputs):
        fuel_consumed = inputs["fuel_consumed"]
        gi = self.options["gravimetric_index"]

        # Structural dry mass calculation: m_struct = m_fuel * (1 - GI) / GI
        tank_dry_mass = fuel_consumed * (1.0 - gi) / gi
        
        outputs["tank_structural_mass"] = tank_dry_mass
        outputs["total_tank_system_mass"] = fuel_consumed + tank_dry_mass

    def compute_partials(self, inputs, partials):
        gi = self.options["gravimetric_index"]

        partials["tank_structural_mass", "fuel_consumed"] = (1.0 - gi) / gi
        partials["total_tank_system_mass", "fuel_consumed"] = 1.0 / gi


class AdvancedFuelCell(ExplicitComponent):
    """
    Fuel Cell model based on electrochemical principles.
    Calculates fuel flow, air flow, heat output, and overall power demands.

    Inputs
    ------
    elec_power_in : float
        Electrical propulsion demand from the motor/inverter side (vector, W)
    compressor_power_demand : float
        BOP compressor power draw (vector, W)
    turbine_power_recovery : float
        BOP turbine recovered power (vector, W)
    fuel_cell_rating : float
        Installed fuel-cell maximum power for sizing calculations (scalar, W)

    Outputs
    -------
    elec_power_req : float
        Net electrical power requested from the fuel-cell stack (vector, W)
    net_bop_power : float
        Net BOP parasitic power (compressor minus turbine) (vector, W)
    fuel_flow : float
        Hydrogen mass flow rate consumed by the stack (vector, kg/s)
    air_flow : float
        Cathode air mass flow required by stack operation (vector, kg/s)
    heat_output : float
        Waste heat generated by the stack (vector, W)
    component_weight : float
        Estimated fuel-cell stack mass from specific-power assumption (scalar, kg)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    """
    def initialize(self):
        self.options.declare("num_nodes", default=1)
        
    def setup(self):
        nn = self.options["num_nodes"]
        
        # Inputs
        self.add_input("elec_power_in", shape=nn, units="W") # Motor power demand
        self.add_input("compressor_power_demand", shape=nn, units="W") # Power consumed by BOP compressor
        self.add_input("turbine_power_recovery", shape=nn, units="W") # Power recovered by BOP turbine
        self.add_input("fuel_cell_rating", units="W", val=1200000)  # Installed fuel-cell maximum power for sizing calculations
        self.add_input("stack_power_density", units="W/kg", val=3000.0)
        
        # Outputs
        self.add_output("elec_power_req", shape=nn, units="W") # Total power required from fuel cell
        self.add_output("net_bop_power", shape=nn, units="W")  # Net BOP power demand (Comp - Turb)
        self.add_output("fuel_flow", shape=nn, units="kg/s") 
        self.add_output("air_flow", shape=nn, units="kg/s")  
        self.add_output("heat_output", shape=nn, units="W") 
        self.add_output("component_weight", units="kg")

        # Constants
        self.F = 96458
        self.M_H2 = 2.016e-3
        self.lambda_H2 = 1.05
        self.lambda_air = 2.0
        self.U_thermo = 1.48
        self.P_hotel = 15000.0 # W, hotel load for the fuel cell system
        self.P_thermo = 35000.0 # W, thermal load for the fuel cell system

    def compute(self, inputs, outputs):
        P_prop = inputs["elec_power_in"]
        P_comp = inputs["compressor_power_demand"]
        P_turb = inputs["turbine_power_recovery"]
        P_fc_max = inputs["fuel_cell_rating"]

        # Power balance calculations
        P_fc_req = P_prop + P_comp + P_turb + self.P_hotel + self.P_thermo
        outputs["elec_power_req"] = P_fc_req
        outputs["net_bop_power"] = P_comp + P_turb
        
        # Cell sizing
        P_cell_fc_max = 1.55 # W, maximum power per fuel cell stack
        n_cells = np.round(P_fc_max / P_cell_fc_max)
        
        P_cell_fc = P_fc_req / n_cells
        
        p_cell_data = np.array([
            0.075778025833261, 0.127633437999547, 0.203357159178834, 0.278520252814293,
            0.371826309890374, 0.434224665575521, 0.518515996512557, 0.580480027625228,
            0.627573232165265, 0.724324294879773, 0.848070230927215, 1.001232636671638,
            1.168830492112718, 1.304620860325906, 1.424534630251711, 1.545098248882887,
            1.644592654313040, 1.693303562612317, 1.715763017855819, 1.729015730256389
        ])

        u_cell_data = np.array([
            0.880093168218912, 0.869245123202384, 0.852973082909549, 0.836701206008447,
            0.816656482041490, 0.803922585363104, 0.787180266744875, 0.774682598507411,
            0.765459139549104, 0.746240436716547, 0.723387030927215, 0.695340798137220,
            0.664699171638789, 0.639252051798263, 0.614843881279524, 0.586280873277772,
            0.553561588086800, 0.531228676750259, 0.512010693770805, 0.494350714859723
        ])

        get_cell_voltage = interp1d(
            p_cell_data, 
            u_cell_data, 
            kind='linear', 
            bounds_error=False, 
            fill_value="extrapolate"
        )

        U_cell_fc = np.clip(get_cell_voltage(P_cell_fc), 0.5, 1.2)
        i_cell_fc = P_cell_fc / U_cell_fc
        
        # Flows & Heat
        outputs["fuel_flow"] = n_cells * 1.0447e-8 * (P_cell_fc / U_cell_fc) * self.lambda_H2
        outputs["air_flow"] = n_cells * 3.58e-7 * (P_cell_fc / U_cell_fc) * self.lambda_air
        outputs["heat_output"] = n_cells * (self.U_thermo - U_cell_fc) * i_cell_fc
        outputs["component_weight"] = P_fc_max / inputs["stack_power_density"]



class FuelCellPropulsionSystemIncompressible(Group):
    """
    Assemble the PC-12 hydrogen-electric propulsion and thermal-management systems.

    The group couples the motor and propeller to the fuel cell and its balance of
    plant (BOP), then connects fuel-cell waste heat to the evaporator, coolant
    loop, heat exchanger, and incompressible cooling duct. Aircraft-level inputs
    can provide the design values; the HX geometry options below provide fallback
    defaults when no upstream value is connected.
    """

    def initialize(self):
        # Configure the mission vector size, hydrogen energy assumption, and fallback HX dimensions.
        self.options.declare("num_nodes", default=1, desc="Number of mission analysis points to run")
        self.options.declare("specific_energy", default=10000, desc="Hydrogen spec energy in Wh/kg")
        self.options.declare("hx_n_wide_cold", default=1500, desc="HX geometry: number of channels wide on cold side")
        self.options.declare("hx_n_long_cold", default=40, desc="HX geometry: number of fins long on cold side")
        self.options.declare("hx_n_tall", default=40, desc="HX geometry: stack count")

    def setup(self):
        nn = self.options["num_nodes"]

        # Provide fallback values for the promoted aircraft-level HX geometry inputs.
        # These defaults do not create additional design variables: connected values
        # supplied by the containing aircraft model remain authoritative.
        self.set_input_defaults("ac|propulsion|thermal|hx|n_wide_cold", val=self.options["hx_n_wide_cold"])
        self.set_input_defaults("ac|propulsion|thermal|hx|n_long_cold", val=self.options["hx_n_long_cold"])
        self.set_input_defaults("ac|propulsion|thermal|hx|n_tall", val=self.options["hx_n_tall"])

        # Define aircraft-level propulsion and fuel-cell system sizing inputs.
        # DVLabel exposes these names to the containing model and converts units
        # where needed; the HX geometry values above are separate promoted inputs.
        dvlist = [
            ["ac|propulsion|motor|rating", "motor_rating", 1200, "hp"],
            ["ac|propulsion|propeller|diameter", "prop_diameter", 2.67, "m"],
            ["ac|weights|W_LH2_tank", "lh2_tank_weight", 264.08, "kg"],
            ["ac|propulsion|fuelcell|stack_power_density", "stack_power_density", 3000.0, "W/kg"],
            ["ac|propulsion|fuelcell|bop_factor", "bop_factor", 0.2, None],
            # Component masses contribute to the BOP package weight estimate.
            ["ac|propulsion|fuelcell|intercooler_mass", "intercooler_mass", 20.0, "kg"],
            ["ac|propulsion|fuelcell|humidifier_mass", "humidifier_mass", 35, "kg"],
        ]

        self.add_subsystem("dvs", DVLabel(dvlist), promotes_inputs=["*"], promotes_outputs=["*"])

        # Convert throttle demand into motor electrical load and propeller shaft power.
        self.add_subsystem("motor", SimpleMotor(num_nodes=nn, weight_inc=1/5000, efficiency=0.97), 
                           promotes_inputs=["throttle"])
        
        self.add_subsystem("propeller", SimplePropeller(num_nodes=nn, num_blades=5), 
                           promotes_inputs=["fltcond|*"], 
                           promotes_outputs=["thrust"])
        
        # Calculate stack power, hydrogen and cathode-air flow, and waste heat.
        # The BOP model uses the resulting air flow and flight conditions to
        # calculate compressor demand, turbine recovery, and package mass.
        self.add_subsystem("fuelcell", AdvancedFuelCell(num_nodes=nn), promotes_outputs=["fuel_flow"])
        self.add_subsystem("bop", BoPAirSystem(num_nodes=nn), promotes_inputs=["fltcond|*"])
        self.connect("stack_power_density", "fuelcell.stack_power_density")
        self.connect("intercooler_mass", "bop.intercooler_mass")
        self.connect("humidifier_mass", "bop.humidifier_mass")

        # Estimate total installed fuel-cell-system mass, including the BOP
        # package and its multiplicative installation factor accounting for air and hydrogen 
        # hoses, valves, hydrogen recirculation fans, filters and sensors.
        self.add_subsystem(
            "fuel_cell_system_weight",
            ExecComp(
                "system_weight = (stack_weight + bop_weight) * (1.0 + bop_factor)",
                stack_weight={"units": "kg"},
                bop_weight={"units": "kg"},
                bop_factor={"val": 0.0},
                system_weight={"units": "kg"},
            ),
            promotes_outputs=[("system_weight", "fuel_cell_system_weight")],
        )
        self.connect("fuelcell.component_weight", "fuel_cell_system_weight.stack_weight")
        self.connect("bop.component_weight", "fuel_cell_system_weight.bop_weight")
        self.connect("bop_factor", "fuel_cell_system_weight.bop_factor")

        # Couple the motor and propeller through shaft power; feed the fuel cell
        # from motor electrical load and close the compressor/turbine power loop.
        self.connect("motor_rating", "motor.elec_power_rating")
        self.connect("motor_rating", "propeller.power_rating")
        self.connect("prop_diameter", "propeller.diameter")

        self.connect("motor.shaft_power_out", "propeller.shaft_power_in")
        self.connect("motor.elec_load", "fuelcell.elec_power_in")  

        self.connect("fuelcell.air_flow", "bop.air_flow")
        self.connect("bop.compressor_power_demand", "fuelcell.compressor_power_demand")
        self.connect("bop.turbine_power_recovery", "fuelcell.turbine_power_recovery")   

        # Set coolant-loop and HX-channel conditions. These independent outputs
        # are the active values for coolant flow, coolant properties, reservoir
        # mass, and channel geometry in this incompressible configuration.
        iv = self.add_subsystem("iv", IndepVarComp(), promotes_outputs=["*"])
        iv.add_output("mdot_coolant", val=20.0 * np.ones((nn,)), units="kg/s")
        iv.add_output("rho_coolant", val=997 * np.ones((nn,)), units="kg/m**3")
        iv.add_output("coolant_mass", val=30.0, units="kg")
        iv.add_output("channel_width", val=1.5, units="mm")
        iv.add_output("channel_height", val=40, units="mm")
        iv.add_output("channel_length", val=0.5, units="m")
        iv.add_output("n_parallel", val=1000000000) # Number of parallel channels set to infinite for assuming ideal heat transfer in LiquidCooledComp
                                                    # from the fuel cell stack to the coolant loop.

        lc_promotes = ["duration", "channel_*", "n_parallel"]

        # The evaporator uses hydrogen flow to absorb part of the stack waste
        # heat; the remaining heat load enters the transient fuel-cell coolant model.
        self.add_subsystem("evaporator", LH2Evaporator(num_nodes=nn))
        self.connect("fuelcell.heat_output", "evaporator.heat_output")
        self.connect("fuel_flow", "evaporator.fuel_flow")

        self.add_subsystem("fc_thermal", LiquidCooledComp(num_nodes=nn, quasi_steady=False), promotes_inputs=lc_promotes)
        self.connect("evaporator.effective_heat_load", "fc_thermal.q_in")
        self.connect("fuelcell.component_weight", "fc_thermal.mass")

        # Model the incompressible cooling-air path and the HX. Ambient air
        # conditions enter the duct, while the promoted HX geometry inputs and
        # coolant properties determine heat transfer and cold-side pressure loss.
        self.add_subsystem("cooling_duct", ExplicitIncompressibleDuctWithInletLoss(num_nodes=nn), promotes_inputs=["fltcond|*"])
        
        self.add_subsystem(
            "hx",
            HXGroup(num_nodes=nn),
            promotes_inputs=[
                ("T_in_cold", "fltcond|T"),
                ("rho_cold", "fltcond|rho"),
                ("ac|propulsion|thermal|hx|n_wide_cold", "ac|propulsion|thermal|hx|n_wide_cold"),
                ("ac|propulsion|thermal|hx|n_long_cold", "ac|propulsion|thermal|hx|n_long_cold"),
                ("ac|propulsion|thermal|hx|n_tall", "ac|propulsion|thermal|hx|n_tall"),
            ],
            promotes_outputs=[("heat_transfer", "heat_transfer"), ("frontal_area", "hx_frontal_area")],
        )
        
        self.connect("cooling_duct.mdot", "hx.mdot_cold")
        self.connect("hx.delta_p_cold", "cooling_duct.delta_p_hex")
        self.connect("fc_thermal.T_out", "hx.T_in_hot")
        self.connect("rho_coolant", "hx.rho_hot")

        # Calculate the duct inlet area from the HX frontal area and a
        # per-node inlet_area_ratio. This keeps the capture area tied to the
        # current HX geometry and exposes the ratio for caller-side tuning.
        self.add_subsystem(
            "inlet_area_from_hx",
            ExecComp(
                "area_inlet = hx_frontal_area * inlet_area_ratio",
                hx_frontal_area={"val": 1.0, "units": "m**2"},
                inlet_area_ratio={"val": 0.4 * np.ones((nn,))},
                area_inlet={"val": 0.4 * np.ones((nn,)), "units": "m**2"},
            ),
        )
        self.connect("hx_frontal_area", "inlet_area_from_hx.hx_frontal_area")
        self.connect("inlet_area_from_hx.area_inlet", "cooling_duct.area_inlet")

        # Calculate nozzle area from HX frontal area and a per-node
        # nozzle_area_ratio. The ratio is an independent input that can be
        # adjusted by the mission-level second optimization.
        self.add_subsystem(
            "nozzle_area_from_hx",
            ExecComp(
                "area_nozzle = nozzle_area_ratio * hx_frontal_area",
                hx_frontal_area={"val": 1.0, "units": "m**2"},
                nozzle_area_ratio={"val": 0.4 * np.ones((nn,))},
                area_nozzle={"val": np.ones((nn,)), "units": "m**2"},
            ),
        )
        self.connect("hx_frontal_area", "nozzle_area_from_hx.hx_frontal_area")
        self.connect("nozzle_area_from_hx.area_nozzle", "cooling_duct.area_nozzle")

        # Store coolant leaving the HX and return it to the fuel-cell thermal
        # model. The same coolant mass-flow and density values feed the loop
        # components to keep their operating conditions consistent.
        self.add_subsystem("coolant_reservoir", CoolantReservoir(num_nodes=nn), promotes_inputs=["duration", ("mass", "coolant_mass")])

        self.connect("hx.T_out_hot", "coolant_reservoir.T_in")
        self.connect("coolant_reservoir.T_out", "fc_thermal.T_in")
        self.connect("mdot_coolant", ["fc_thermal.mdot_coolant", "hx.mdot_hot", "coolant_reservoir.mdot_coolant"])

class FuelCellPropulsionSystemCompressible(IntegratorGroup):
    """Experimental propulsion-system assembly for testing compressible-flow logic.

    This prototype replaces the incompressible cooling duct with an implicit
    compressible duct and connects its station conditions to the heat exchanger.
    It also includes a trial coolant-loop model with a hose and pump. The setup,
    default parameters, pressure-drop coupling, solver behavior, and mass-flow
    floor have not been fully validated; review and test this group before using
    it for production analyses or design decisions.
    """

    def initialize(self):
        # Configure the number of mission analysis nodes used by each subsystem.
        self.options.declare("num_nodes", default=1)

    def setup(self):
        nn = self.options["num_nodes"]

        # Expose propulsion ratings, tank weight, and auxiliary-system masses
        # to the containing model through the shared design-variable component.
        dvlist = [
            ["ac|propulsion|motor|rating", "motor_rating", 1200, "hp"],
            ["ac|propulsion|propeller|diameter", "prop_diameter", 2.67, "m"],
            ["ac|weights|W_LH2_tank", "lh2_tank_weight", 264.08, "kg"],
            ["ac|propulsion|fuelcell|intercooler_mass", "intercooler_mass", 20.0, "kg"],
            ["ac|propulsion|fuelcell|humidifier_mass", "humidifier_mass", 35.0, "kg"],
        ]
        self.add_subsystem(
            "dvs",
            DVLabel(dvlist),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )

        # Assemble the motor, propeller, fuel cell, and balance-of-plant models.
        # The fuel cell supplies hydrogen and cathode-air flow; the BOP consumes
        # compressor power and returns turbine power to the fuel-cell balance.
        self.add_subsystem(
            "motor",
            SimpleMotor(num_nodes=nn, efficiency=0.97),
            promotes_inputs=["throttle"],
        )

        self.add_subsystem(
            "propeller",
            SimplePropeller(num_nodes=nn, num_blades=5),
            promotes_inputs=["fltcond|*"],
            promotes_outputs=["thrust"],
        )

        self.add_subsystem(
            "fuelcell",
            AdvancedFuelCell(num_nodes=nn),
            promotes_outputs=["fuel_flow"],
        )

        self.add_subsystem(
            "bop",
            BoPAirSystem(num_nodes=nn),
            promotes_inputs=["fltcond|*"],
        )

        self.connect("motor_rating", "motor.elec_power_rating")
        self.connect("motor_rating", "propeller.power_rating")
        self.connect("prop_diameter", "propeller.diameter")
        self.connect("motor.shaft_power_out", "propeller.shaft_power_in")
        self.connect("motor.elec_load", "fuelcell.elec_power_in")
        self.connect("intercooler_mass", "bop.intercooler_mass")
        self.connect("humidifier_mass", "bop.humidifier_mass")
        self.connect("fuelcell.air_flow", "bop.air_flow")
        self.connect("bop.compressor_power_demand", "fuelcell.compressor_power_demand")
        self.connect("bop.turbine_power_recovery", "fuelcell.turbine_power_recovery")

        # Experimental coolant, nozzle, and HX-channel values for this prototype.
        # These hard-coded values need engineering review and validation before
        # they are treated as representative design inputs.
        thermal_params = self.add_subsystem("thermal_params", om.IndepVarComp(), promotes_outputs=["*"])
        thermal_params.add_output("rho_coolant", val=1020.0 * np.ones((nn,)), units="kg/m**3")
        thermal_params.add_output("mdot_coolant_fc", val=4.8 * np.ones((nn,)), units="kg/s")
        thermal_params.add_output("variable_duct_nozzle_area", val=60.0 * np.ones((nn,)), units="inch**2")
        thermal_params.add_output("channel_width", val=1.5, units="mm")
        thermal_params.add_output("channel_height", val=1.5, units="mm")
        thermal_params.add_output("channel_length", val=0.5, units="m")
        thermal_params.add_output("n_parallel", val=300)

        hx_design_vars = [
            "ac|propulsion|thermal|hx|n_wide_cold",
            "ac|propulsion|thermal|hx|n_long_cold",
            "ac|propulsion|thermal|hx|n_tall",
        ]

        # Route stack waste heat through the LH2 evaporator, then pass the
        # remaining heat load into the transient liquid-cooled fuel-cell model.
        self.add_subsystem("evaporator", LH2Evaporator(num_nodes=nn))
        self.connect("fuelcell.heat_output", "evaporator.heat_output")
        self.connect("fuel_flow", "evaporator.fuel_flow")

        lc_promotes = ["duration", "channel_*", "n_parallel"]
        self.add_subsystem(
            "fc_thermal",
            LiquidCooledComp(num_nodes=nn, quasi_steady=False),
            promotes_inputs=lc_promotes,
        )
        self.connect("evaporator.effective_heat_load", "fc_thermal.q_in")
        self.connect("fuelcell.component_weight", "fc_thermal.mass")

        # Add the compressible cooling duct and the HX. The duct's ambient
        # boundary conditions are supplied from the flight-condition inputs;
        # its station-2 state becomes the HX cold-side inlet condition.
        self.add_subsystem(
            "hx_fc",
            HXGroup(num_nodes=nn),
            promotes_inputs=[(a, a) for a in hx_design_vars],
        )
        self.connect("rho_coolant", "hx_fc.rho_hot")

        self.add_subsystem(
            "cooling_duct",
            ImplicitCompressibleDuct_ExternalHX(num_nodes=nn, cfg=0.90),
            promotes_inputs=[
                ("p_inf", "fltcond|p"),
                ("T_inf", "fltcond|T"),
                ("Utrue", "fltcond|Utrue"),
            ],
        )
        self.connect("variable_duct_nozzle_area", "cooling_duct.area_nozzle_in")

        # Apply the experimental mass-flow lower bound before passing duct flow
        # to the HX. Check that this treatment remains appropriate across the
        # expected operating envelope and nonlinear-solver trial states.
        self.add_subsystem("mdot_cold_floor", MassFlowFloor(num_nodes=nn, floor=0.05))
        self.connect("cooling_duct.mdot", "mdot_cold_floor.mdot_in")

        self.add_subsystem(
            "fc_hose",
            SimpleHose(num_nodes=nn),
            promotes_inputs=[
                ("hose_length", "ac|geom|thermal|hx_to_fc_length"),
                ("hose_diameter", "ac|geom|thermal|hx_to_fc_diameter"),
            ],
        )
        self.add_subsystem(
            "fc_coolant_pump",
            SimplePump(num_nodes=nn),
            promotes_inputs=[("power_rating", "ac|propulsion|thermal|hx_fc|pump_power_rating")],
        )

        self.promote_add(
            sources=["fc_hose.delta_p", "hx_fc.delta_p_hot"],
            prom_name="pressure_drop_fc_loop",
            factors=[1.0, -1.0],
            vec_size=nn,
            units="Pa",
        )
        self.connect("pressure_drop_fc_loop", "fc_coolant_pump.delta_p")
        self.connect("rho_coolant", ["fc_hose.rho_coolant", "fc_coolant_pump.rho_coolant"])

        self.connect("cooling_duct.sta2.T", "hx_fc.T_in_cold")
        self.connect("cooling_duct.sta2.rho", "hx_fc.rho_cold")
        self.connect("mdot_cold_floor.mdot_out", "hx_fc.mdot_cold")

        # Model the coolant return path with a hose and pump. The combined
        # pressure-drop signal and fixed coolant-flow inputs are part of this
        # experimental configuration and require validation as a coupled loop.
        self.connect(
            "mdot_coolant_fc",
            [
                "fc_thermal.mdot_coolant",
                "hx_fc.mdot_hot",
                "fc_hose.mdot_coolant",
                "fc_coolant_pump.mdot_coolant",
            ],
        )

        # Couple HX heat transfer, cold-side pressure loss, and hot-side
        # temperatures back to the duct and fuel-cell thermal model, closing
        # the compressible air and coolant heat-exchange paths.
        self.connect("hx_fc.frontal_area", ["cooling_duct.area_2", "cooling_duct.area_3"])
        self.connect("hx_fc.delta_p_cold", "cooling_duct.sta3.delta_p")
        self.connect("hx_fc.heat_transfer", "cooling_duct.sta3.heat_in")
        self.connect("fc_thermal.T_out", "hx_fc.T_in_hot")
        self.connect("hx_fc.T_out_hot", "fc_thermal.T_in") 