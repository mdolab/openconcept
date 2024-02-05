import numpy as np
from openmdao.api import ExplicitComponent, ImplicitComponent, IndepVarComp, ExecComp, Group

from .heat_exchanger import HXGroup
from openconcept.utilities import AddSubtractComp, DVLabel


class ExplicitIncompressibleDuct(ExplicitComponent):
    """
    This is a very approximate model of a duct at incompressible speeds.
    Its advantage is low computational cost and improved convergence even at low speed.
    It CANNOT model flow with heat addition so it will generally be a conservative estimate on the cooling drag.
    Assumes the static pressure at the outlet duct = p_inf which may or may not be a good assumption.

    Inputs
    ------
    fltcond|Utrue : float
        True airspeed in the freestream (vector, m/s)
    fltcond|rho : float
        Density in the freestream (vector, kg/m**3)
    area_nozzle : float
        Cross-sectional area of the outlet nozzle (vector, m**2)
        Generally must be the narrowest portion of the duct for analysis to be valid
    delta_p_hex : float
        Pressure drop across the heat exchanger (vector, Pa)

    Outputs
    -------
    mdot : float
        Mass flow rate through the duct (vector, kg/s)
    drag : float
        Drag force on the duct positive rearwards (vector, N)

    Options
    -------
    num_nodes : float
        Number of analysis points to run
    static_pressure_loss_factor : float
        Delta p / local dynamic pressure to model inlet and nozzle losses (vector, dimensionless)
        Default 0.00
    gross_thrust_factor : float
        Fraction of total gross thrust to recover (aka Cfg)
        Accounts for duct losses, in particular nozzle losses
        Default 0.98
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1)
        self.options.declare("static_pressure_loss_factor", default=0.15)
        self.options.declare("gross_thrust_factor", default=0.98)

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("fltcond|Utrue", shape=(nn,), units="m/s")
        self.add_input("fltcond|rho", shape=(nn,), units="kg/m**3")
        self.add_input("area_nozzle", shape=(nn,), units="m**2")
        self.add_input("delta_p_hex", shape=(nn,), units="Pa")

        self.add_output("mdot", shape=(nn,), lower=0.01, units="kg/s")
        self.add_output("drag", shape=(nn,), units="N")

        # self.declare_partials(['drag','mdot'],['fltcond|Utrue','fltcond|rho','delta_p_hex'],rows=np.arange(nn),cols=np.arange(nn))
        # self.declare_partials(['drag','mdot'],['area_nozzle'],rows=np.arange(nn),cols=np.zeros((nn,)))

        self.declare_partials(["drag", "mdot"], ["fltcond|Utrue", "fltcond|rho", "delta_p_hex"], method="cs")
        self.declare_partials(["drag", "mdot"], ["area_nozzle"], method="cs")

    def compute(self, inputs, outputs):
        static_pressure_loss_factor = self.options["static_pressure_loss_factor"]
        cfg = self.options["gross_thrust_factor"]
        # this is an absolute hack to prevent spurious NaNs during the Newton solve
        interior = (
            inputs["fltcond|rho"] ** 2 * inputs["fltcond|Utrue"] ** 2
            + 2 * inputs["fltcond|rho"] * inputs["delta_p_hex"]
        ) / (1 + static_pressure_loss_factor)
        interior[np.where(interior < 0.0)] = 1e-10
        mdot = inputs["area_nozzle"] * np.sqrt(interior)
        mdot[np.where(mdot < 0.0)] = 1e-10
        # if self.pathname.split('.')[1] == 'climb':
        #     print('Nozzle area:'+str(inputs['area_nozzle']))
        #     print('mdot:'+str(mdot))
        #     print('delta_p_hex:'+str(inputs['delta_p_hex']))

        outputs["mdot"] = mdot
        outputs["drag"] = mdot * (inputs["fltcond|Utrue"] - cfg * mdot / inputs["area_nozzle"] / inputs["fltcond|rho"])

    # def compute_partials(self, inputs, J):
    #     static_pressure_loss_factor = self.options['static_pressure_loss_factor']
    #     # delta p is defined as NEGATIVE from the hex
    #     interior = (inputs['fltcond|rho']**2 * inputs['fltcond|Utrue']**2 + 2*inputs['fltcond|rho']*inputs['delta_p_hex'])/(1+static_pressure_loss_factor)
    #     interior[np.where(interior<0.0)]=1e-10
    #     mdot = inputs['area_nozzle'] * np.sqrt(interior)
    #     dmdotdrho = inputs['area_nozzle'] * interior ** (-1/2) * (inputs['fltcond|rho']*inputs['fltcond|U']**2+inputs['delta_p_hex'])
    #     dmdotdU =
    #     dmdotddeltap =
    #     dmdotdA =

    #     J['mdot','fltcond|rho'] = dmdotdrho
    #     J['mdot','fltcond|Utrue'] = dmdotdU
    #     J['mdot','delta_p_hex'] = dmdotddeltap
    #     J['mdot','area_nozzle'] = dmdotdA
    #     J['drag','fltcond|rho'] = (inputs['fltcond|Utrue'] - 2*mdot/inputs['area_nozzle']/inputs['fltcond|rho']) * dmdotdrho + mdot * (mdot/inputs['area_nozzle']/inputs['fltcond|rho']**2)
    #     J['drag','fltcond|Utrue'] = (inputs['fltcond|Utrue'] - 2*mdot/inputs['area_nozzle']/inputs['fltcond|rho']) * dmdotdU + mdot
    #     J['drag','delta_p_hex'] = (inputs['fltcond|Utrue'] - 2*mdot/inputs['area_nozzle']/inputs['fltcond|rho']) * dmdotddeltap
    #     J['drag','area_nozzle'] = (inputs['fltcond|Utrue'] - 2*mdot/inputs['area_nozzle']/inputs['fltcond|rho']) * dmdotdA + mdot * (mdot/inputs['area_nozzle']**2/inputs['fltcond|rho'])


class TemperatureIsentropic(ExplicitComponent):
    """
    Compute static temperature via isentropic relation

    Inputs
    -------
    Tt : float
        Total temperature (vector, K)
    M : float
        Mach number (vector, dimensionless)

    Outputs
    -------
    T : float
        Static temperature  (vector, K)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    gamma : float
        Specific heat ratio (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of flight/control conditions")
        self.options.declare("gamma", default=1.4, desc="Specific heat ratio")

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("Tt", shape=(nn,), units="K")
        self.add_input("M", shape=(nn,))
        self.add_output("T", shape=(nn,), units="K")
        arange = np.arange(nn)
        self.declare_partials(["T"], ["M", "Tt"], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        gam = self.options["gamma"]
        outputs["T"] = inputs["Tt"] * (1 + (gam - 1) / 2 * inputs["M"] ** 2) ** -1

    def compute_partials(self, inputs, J):
        gam = self.options["gamma"]
        J["T", "Tt"] = (1 + (gam - 1) / 2 * inputs["M"] ** 2) ** -1
        J["T", "M"] = -inputs["Tt"] * (1 + (gam - 1) / 2 * inputs["M"] ** 2) ** -2 * (gam - 1) * inputs["M"]


class TotalTemperatureIsentropic(ExplicitComponent):
    """
    Compute total temperature via isentropic relation

    Inputs
    -------
    T : float
        Static temperature (vector, K)
    M : float
        Mach number (vector, dimensionless)

    Outputs
    -------
    Tt : float
        Static temperature  (vector, K)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    gamma : float
        Specific heat ratio (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of flight/control conditions")
        self.options.declare("gamma", default=1.4, desc="Specific heat ratio")

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("T", shape=(nn,), units="K")
        self.add_input("M", shape=(nn,))
        self.add_output("Tt", shape=(nn,), units="K")
        arange = np.arange(nn)
        self.declare_partials(["Tt"], ["T", "M"], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        gam = self.options["gamma"]
        outputs["Tt"] = inputs["T"] * (1 + (gam - 1) / 2 * inputs["M"] ** 2)

    def compute_partials(self, inputs, J):
        gam = self.options["gamma"]
        J["Tt", "T"] = 1 + (gam - 1) / 2 * inputs["M"] ** 2
        J["Tt", "M"] = inputs["T"] * (gam - 1) * inputs["M"]


class PressureIsentropic(ExplicitComponent):
    """
    Compute static pressure via isentropic relation

    Inputs
    -------
    pt : float
        Total pressure (vector, Pa)
    M : float
        Mach number (vector, dimensionless)

    Outputs
    -------
    p : float
        Static temperature  (vector, Pa)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    gamma : float
        Specific heat ratio (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of flight/control conditions")
        self.options.declare("gamma", default=1.4, desc="Specific heat ratio")

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("pt", shape=(nn,), units="Pa")
        self.add_input("M", shape=(nn,))
        self.add_output("p", shape=(nn,), units="Pa")
        self.declare_partials(["*"], ["*"], method="cs")

    def compute(self, inputs, outputs):
        gam = self.options["gamma"]
        outputs["p"] = inputs["pt"] * (1 + (gam - 1) / 2 * inputs["M"] ** 2) ** (-gam / (gam - 1))


class TotalPressureIsentropic(ExplicitComponent):
    """
    Compute total pressure via isentropic relation

    Inputs
    -------
    p : float
        Static pressure (vector, Pa)
    M : float
        Mach number (vector, dimensionless)

    Outputs
    -------
    pt : float
        Total pressure  (vector, Pa)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    gamma : float
        Specific heat ratio (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of flight/control conditions")
        self.options.declare("gamma", default=1.4, desc="Specific heat ratio")

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("p", shape=(nn,), units="Pa")
        self.add_input("M", shape=(nn,))
        self.add_output("pt", shape=(nn,), units="Pa")
        arange = np.arange(nn)
        self.declare_partials(["pt"], ["p", "M"], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        gam = self.options["gamma"]
        outputs["pt"] = inputs["p"] * (1 + (gam - 1) / 2 * inputs["M"] ** 2) ** (gam / (gam - 1))

    def compute_partials(self, inputs, J):
        gam = self.options["gamma"]
        J["pt", "p"] = (1 + (gam - 1) / 2 * inputs["M"] ** 2) ** (gam / (gam - 1))
        J["pt", "M"] = (
            inputs["p"]
            * (gam - 1)
            * inputs["M"]
            * (gam / (gam - 1))
            * (1 + (gam - 1) / 2 * inputs["M"] ** 2) ** (gam / (gam - 1) - 1)
        )


class DensityIdealGas(ExplicitComponent):
    """
    Compute density from ideal gas law

    Inputs
    -------
    p : float
        Static pressure (vector, Pa)
    T : float
        Static temperature (vector, K)

    Outputs
    -------
    rho : float
        Density  (vector, kg/m**3)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    R : float
        Gas constant (scalar, J / kg / K)

    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of flight/control conditions")
        self.options.declare("R", default=287.05, desc="Gas constant")

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("p", shape=(nn,), units="Pa")
        self.add_input("T", shape=(nn,), units="K")
        self.add_output("rho", shape=(nn,), units="kg/m**3")
        self.declare_partials(["*"], ["*"], method="cs")

    def compute(self, inputs, outputs):
        R = self.options["R"]
        outputs["rho"] = inputs["p"] / R / inputs["T"]


class SpeedOfSound(ExplicitComponent):
    """
    Compute speed of sound

    Inputs
    -------
    T : float
        Static temperature (vector, K)

    Outputs
    -------
    a : float
        Speed of sound  (vector, m/s)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    R : float
        Gas constant (scalar, J / kg / K)
    gamma : float
        Specific heat ratio (scalar dimensionless)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of flight/control conditions")
        self.options.declare("R", default=287.05, desc="Gas constant")
        self.options.declare("gamma", default=1.4, desc="Specific heat ratio")

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("T", shape=(nn,), units="K")
        self.add_output("a", shape=(nn,), units="m/s", lower=1e0)
        arange = np.arange(nn)
        self.declare_partials(["a"], ["T"], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        R = self.options["R"]
        gam = self.options["gamma"]
        T = inputs["T"].copy()
        T[np.where(T < 0.0)] = 100
        outputs["a"] = np.sqrt(gam * R * T)

    def compute_partials(self, inputs, J):
        R = self.options["R"]
        gam = self.options["gamma"]
        T = inputs["T"].copy()
        T[np.where(T < 0.0)] = 100
        J["a", "T"] = 0.5 * np.sqrt(gam * R) / np.sqrt(T)


class MachNumberfromSpeed(ExplicitComponent):
    """
    Compute Mach number from TAS and speed of sound

    Inputs
    -------
    Utrue : float
        True airspeed (vector, m/s)
    a : float
        Speed of sound (vector, m/s)

    Outputs
    -------
    M : float
        Mach number (vector, dimensionless)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of flight/control conditions")

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("a", shape=(nn,), units="m/s")
        self.add_input("Utrue", shape=(nn,), units="m/s")
        self.add_output("M", shape=(nn,))
        self.declare_partials(["*"], ["*"], method="cs")

    def compute(self, inputs, outputs):
        outputs["M"] = inputs["Utrue"] / inputs["a"]


class HeatAdditionPressureLoss(ExplicitComponent):
    """
    Adds / removes heat and pressure gain / loss

    Inputs
    -------
    Tt_in : float
        Total temperature in (vector, K)
    pt_in : float
        Total pressure in (vector, Pa)
    mdot : float
        Mass flow (vector, kg/s)
    delta_p : float
        Pressure gain / loss (vector, Pa)
    pressure_recovery : float
        Total pressure gain / loss as a multiple (vector, dimensionless)
    heat_in : float
        Heat addition (subtraction) rate (vector, W)
    cp : float
        Specific heat (scalar, J/kg/K)

    Outputs
    -------
    Tt_out : float
        Total temperature out  (vector, K)
    pt_out : float
        Total pressure out (vector, Pa)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of flight/control conditions")

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("Tt_in", shape=(nn,), units="K")
        self.add_input("pt_in", shape=(nn,), units="Pa")
        self.add_input("mdot", shape=(nn,), units="kg/s")
        self.add_input("rho", shape=(nn,), units="kg/m**3")
        self.add_input("area", units="m**2")
        self.add_input("delta_p", shape=(nn,), val=0.0, units="Pa")
        self.add_input("dynamic_pressure_loss_factor", val=0.0)
        self.add_input("pressure_recovery", shape=(nn,), val=np.ones((nn,)))
        self.add_input("heat_in", shape=(nn,), val=0.0, units="W")
        self.add_input("cp", units="J/kg/K")

        self.add_output("Tt_out", shape=(nn,), units="K")
        self.add_output("pt_out", shape=(nn,), units="Pa")

        arange = np.arange(nn)

        self.declare_partials(["Tt_out"], ["Tt_in", "heat_in", "mdot"], rows=arange, cols=arange)
        self.declare_partials(["Tt_out"], ["cp"], rows=arange, cols=np.zeros((nn,)))
        self.declare_partials(
            ["pt_out"], ["pt_in", "pressure_recovery", "delta_p", "rho", "mdot"], rows=arange, cols=arange
        )
        self.declare_partials(["pt_out"], ["area", "dynamic_pressure_loss_factor"], rows=arange, cols=np.zeros((nn,)))

    def compute(self, inputs, outputs):
        dynamic_pressure = 0.5 * inputs["mdot"] ** 2 / inputs["rho"] / inputs["area"] ** 2

        if np.min(inputs["mdot"]) <= 0.0:
            raise ValueError(self.msginfo, inputs["mdot"])
        tt_out = inputs["Tt_in"] + inputs["heat_in"] / inputs["cp"] / inputs["mdot"]
        pt_out = (
            inputs["pt_in"] * inputs["pressure_recovery"]
            - dynamic_pressure * inputs["dynamic_pressure_loss_factor"]
            + inputs["delta_p"]
        )
        # outputs['Tt_out'] = inputs['Tt_in'] + inputs['heat_in'] / inputs['cp'] / inputs['mdot']
        # outputs['Tt_out'] = np.where(tt_out <= 0.0, inputs['Tt_in'], tt_out)
        # outputs['pt_out'] = np.where(pt_out <= 0.0, inputs['pt_in'] * inputs['pressure_recovery'], pt_out)
        outputs["pt_out"] = pt_out
        if np.max(pt_out) > 1e8:
            raise ValueError(self.msginfo, inputs["pt_in"], inputs["rho"], inputs["delta_p"])
        outputs["Tt_out"] = tt_out

    def compute_partials(self, inputs, J):
        nn = self.options["num_nodes"]

        J["Tt_out", "Tt_in"] = np.ones((nn,))
        J["Tt_out", "heat_in"] = 1 / inputs["cp"] / inputs["mdot"]
        J["Tt_out", "cp"] = -inputs["heat_in"] / inputs["cp"] ** 2 / inputs["mdot"]
        J["Tt_out", "mdot"] = -inputs["heat_in"] / inputs["cp"] / inputs["mdot"] ** 2

        J["pt_out", "pt_in"] = inputs["pressure_recovery"]
        J["pt_out", "pressure_recovery"] = inputs["pt_in"]
        J["pt_out", "delta_p"] = np.ones((nn,))
        J["pt_out", "mdot"] = (
            -inputs["dynamic_pressure_loss_factor"] * inputs["mdot"] / inputs["rho"] / inputs["area"] ** 2
        )
        J["pt_out", "rho"] = (
            0.5
            * inputs["dynamic_pressure_loss_factor"]
            * inputs["mdot"] ** 2
            / inputs["rho"] ** 2
            / inputs["area"] ** 2
        )
        J["pt_out", "area"] = (
            inputs["dynamic_pressure_loss_factor"] * inputs["mdot"] ** 2 / inputs["rho"] / inputs["area"] ** 3
        )
        J["pt_out", "dynamic_pressure_loss_factor"] = -0.5 * inputs["mdot"] ** 2 / inputs["rho"] / inputs["area"] ** 2


class MassFlow(ExplicitComponent):
    """
    Computes mass flow explicity from other parameters.
    Designed for use at the nozzle / min area point.

    Inputs
    ------
    M : float
        Mach number at this station (vector, dimensionless)
    rho : float
        Density at this station (vector, kg/m**3)
    area : float
        Flow cross sectional area at this station (vector, m**2)
    a : float
        Speed of sound (vector, m/s)

    Outputs
    -------
    mdot : float
        Mass flow rate (vector, kg/s)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of flight/control conditions")

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("a", shape=(nn,), units="m/s")
        self.add_input("area", shape=(nn,), units="m**2")
        self.add_input("rho", shape=(nn,), units="kg/m**3")
        self.add_input("M", shape=(nn,))
        self.add_output("mdot", shape=(nn,), units="kg/s", lower=1e-5)
        arange = np.arange(0, nn)
        self.declare_partials(["mdot"], ["M", "a", "rho", "area"], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        outputs["mdot"] = inputs["M"] * inputs["a"] * inputs["area"] * inputs["rho"]

    def compute_partials(self, inputs, J):
        J["mdot", "M"] = inputs["a"] * inputs["area"] * inputs["rho"]
        J["mdot", "a"] = inputs["M"] * inputs["area"] * inputs["rho"]
        J["mdot", "area"] = inputs["M"] * inputs["a"] * inputs["rho"]
        J["mdot", "rho"] = inputs["M"] * inputs["a"] * inputs["area"]


class MachNumberDuct(ImplicitComponent):
    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of flight/control conditions")

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("mdot", shape=(nn,), units="kg/s")
        self.add_input("a", shape=(nn,), units="m/s")
        self.add_input("area", units="m**2")
        self.add_input("rho", shape=(nn,), units="kg/m**3")
        # self.add_output('M', shape=(nn,), lower=0.0, upper=1.0)
        self.add_output("M", shape=(nn,), val=np.ones((nn,)) * 0.6, lower=0.00001, upper=0.99999)
        arange = np.arange(0, nn)
        self.declare_partials(["M"], ["mdot"], rows=arange, cols=arange, val=np.ones((nn,)))
        self.declare_partials(["M"], ["M", "a", "rho"], rows=arange, cols=arange)
        self.declare_partials(["M"], ["area"], rows=arange, cols=np.zeros((nn,), dtype=np.int32))

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals["M"] = inputs["mdot"] - outputs["M"] * inputs["a"] * inputs["area"] * inputs["rho"]

    def linearize(self, inputs, outputs, J):
        J["M", "M"] = -inputs["a"] * inputs["area"] * inputs["rho"]
        J["M", "a"] = -outputs["M"] * inputs["area"] * inputs["rho"]
        J["M", "area"] = -outputs["M"] * inputs["a"] * inputs["rho"]
        J["M", "rho"] = -outputs["M"] * inputs["a"] * inputs["area"]


class DuctExitPressureRatioImplicit(ImplicitComponent):
    """
    Compute duct exit pressure ratio based on total pressure and ambient pressure

    Inputs
    -------
    p_exit : float
        Exit static pressure (vector, Pa)
    pt : float
        Total pressure (vector, Pa)

    Outputs
    -------
    nozzle_pressure_ratio : float
        Computed nozzle pressure ratio (vector, dimensionless)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of flight/control conditions")

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("p_exit", shape=(nn,), units="Pa")
        self.add_input("pt", shape=(nn,), units="Pa")
        self.add_output("nozzle_pressure_ratio", shape=(nn,), val=np.ones((nn,)) * 0.9, lower=0.01, upper=0.99999999999)
        arange = np.arange(0, nn)
        self.declare_partials(
            ["nozzle_pressure_ratio"], ["nozzle_pressure_ratio"], rows=arange, cols=arange, val=np.ones((nn,))
        )
        self.declare_partials(["nozzle_pressure_ratio"], ["p_exit", "pt"], rows=arange, cols=arange)

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals["nozzle_pressure_ratio"] = outputs["nozzle_pressure_ratio"] - inputs["p_exit"] / inputs["pt"]

    def linearize(self, inputs, outputs, J):
        J["nozzle_pressure_ratio", "p_exit"] = -1 / inputs["pt"]
        J["nozzle_pressure_ratio", "pt"] = inputs["p_exit"] / inputs["pt"] ** 2


class DuctExitMachNumber(ExplicitComponent):
    """
    Compute duct exit Mach number based on nozzle pressure ratio

    Inputs
    -------
    nozzle_pressure_ratio : float
        Computed nozzle pressure ratio (vector, dimensionless)

    Outputs
    -------
    M : float
        Computed nozzle Mach number(vector, dimensionless)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    gamma : float
        Specific heat ratio (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of flight/control conditions")
        self.options.declare("gamma", default=1.4, desc="Specific heat ratio")

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("nozzle_pressure_ratio", shape=(nn,))
        self.add_output("M", shape=(nn,))
        self.declare_partials(["*"], ["*"], method="cs")

    def compute(self, inputs, outputs):
        nn = self.options["num_nodes"]
        gam = self.options["gamma"]
        critical_pressure_ratio = (2 / (gam + 1)) ** (gam / (gam - 1))
        if np.min(inputs["nozzle_pressure_ratio"]) <= 0.0:
            raise ValueError(self.msginfo)
        outputs["M"] = np.where(
            np.less_equal(inputs["nozzle_pressure_ratio"], critical_pressure_ratio),
            np.ones((nn,)),
            np.sqrt(((inputs["nozzle_pressure_ratio"]) ** ((1 - gam) / gam) - 1) * 2 / (gam - 1)),
        )


class NetForce(ExplicitComponent):
    """
    Compute net force based on inlet and outlet pressures and velocities

    Inputs
    -------
    mdot : float
        Mass flow rate (vector, kg/s)
    Utrue_inf : float
        Freestream true airspeed (vector, m/s)
    p_inf : float
        Static pressure in the free stream. (vector, Pa)
    area_nozzle : float
        Nozzle cross sectional area (vector, m**2)
    p_nozzle : float
        Static pressure at the nozzle. Equal to p_inf unless choked (vector, Pa)
    rho_nozzle : float
        Density at the nozzle (vector, kg/m**3)

    Outputs
    -------
    F_net : float
        Overall net force (positive is forward thrust) (vector, N)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of flight/control conditions")
        self.options.declare("cfg", default=0.98, desc="Factor on gross thrust (accounts for some duct losses)")

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("mdot", shape=(nn,), units="kg/s")
        self.add_input("Utrue_inf", shape=(nn,), units="m/s")
        self.add_input("p_inf", shape=(nn,), units="Pa")
        self.add_input("area_nozzle", shape=(nn,), units="m**2")
        self.add_input("p_nozzle", shape=(nn,), units="Pa")
        self.add_input("rho_nozzle", shape=(nn,), units="kg/m**3")

        self.add_output("F_net", shape=(nn,), units="N")
        self.declare_partials(["*"], ["*"], method="cs")

    def compute(self, inputs, outputs):
        cfg = self.options["cfg"]
        outputs["F_net"] = inputs["mdot"] * (
            inputs["mdot"] / inputs["area_nozzle"] / inputs["rho_nozzle"] * cfg - inputs["Utrue_inf"]
        ) + inputs["area_nozzle"] * cfg * (inputs["p_nozzle"] - inputs["p_inf"])


class Inlet(Group):
    """This group takes in ambient flight conditions and computes total quantities for downstream use

    Inputs
    ------
    T : float
        Temperature (vector, K)
    p : float
        Ambient static pressure (vector, Pa)
    Utrue : float
        True airspeed (vector, m/s)

    Outputs
    -------
    Tt : float
        Total temperature (vector, K)
    pt : float
        Total pressure (vector, Pa)

    Options
    -------
    num_nodes : int
        Number of conditions to analyze
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of flight/control conditions")

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_subsystem("speedsound", SpeedOfSound(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("mach", MachNumberfromSpeed(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem(
            "freestreamtotaltemperature",
            TotalTemperatureIsentropic(num_nodes=nn),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )
        self.add_subsystem("freestreamtotalpressure", TotalPressureIsentropic(num_nodes=nn), promotes_inputs=["*"])
        # self.add_subsystem('inlet_recovery', ExecComp('eta_ram=1.0 - 0.00*tanh(10*M)', has_diag_partials=True, eta_ram=np.ones((nn,)), M=0.1*np.ones((nn,))), promotes_inputs=['M'])
        self.add_subsystem(
            "totalpressure",
            ExecComp(
                "pt=pt_in * eta_ram",
                pt={"units": "Pa", "val": np.ones((nn,)), "lower": 1.0},
                pt_in={"units": "Pa", "val": np.ones((nn,))},
                eta_ram=np.ones((nn,)),
                has_diag_partials=True,
            ),
            promotes_outputs=["pt"],
        )
        self.connect("freestreamtotalpressure.pt", "totalpressure.pt_in")
        # self.connect('inlet_recovery.eta_ram','totalpressure.eta_ram')


class DuctStation(Group):
    """A 'normal' station in a duct flow.

    Inputs
    ------
    pt_in : float
        Upstream total pressure (vector, Pa)
    Tt_in : float
        Upstream total temperature (vector, K)
    mdot : float
        Mass flow (vector, kg/s)
    delta_p : float
        Pressure gain (loss) at this station (vector, Pa)
    heat_in : float
        Heat addition (loss) rate at this station (vector, W)

    Outputs
    -------
    pt_out : float
        Downstream total pressure (vector, Pa)
    Tt_out : float
        Downstream total temperature (vector, K)

    Options
    -------
    num_nodes : int
        Number of conditions to analyze
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of flight/control conditions")

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_subsystem(
            "totals", HeatAdditionPressureLoss(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"]
        )
        self.add_subsystem("temp", TemperatureIsentropic(num_nodes=nn), promotes_inputs=["M"], promotes_outputs=["*"])
        self.connect("Tt_out", "temp.Tt")
        self.add_subsystem("pressure", PressureIsentropic(num_nodes=nn), promotes_inputs=["M"], promotes_outputs=["*"])
        self.connect("pt_out", "pressure.pt")
        self.add_subsystem("density", DensityIdealGas(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("speedsound", SpeedOfSound(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("mach", MachNumberDuct(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])


class NozzlePressureLoss(ExplicitComponent):
    """This group adds proportional pressure loss to the nozzle component

    Inputs
    ------
    pt_in : float
        Total pressure upstream of the nozzle (vector, Pa)
    mdot : float
        Mass flow (vector, kg/s)
    area : float
        Nozzle cross sectional area (vector, m**2)
    dynamic_pressure_loss_factor : float
        Total pressure loss as a fraction of dynamic pressure

    Outputs
    -------
    pt : float
        Total pressure downstream of the nozzle (vector, Pa)

    Options
    -------
    num_nodes : int
        Number of conditions to analyze
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of flight/control conditions")

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("pt_in", shape=(nn,), units="Pa")
        self.add_input("mdot", shape=(nn,), units="kg/s")
        self.add_input("rho", shape=(nn,), units="kg/m**3")
        self.add_input("area", shape=(nn,), units="m**2")
        self.add_input("dynamic_pressure_loss_factor", val=0.0)

        self.add_output("pt", shape=(nn,), units="Pa", lower=1.0)

        arange = np.arange(nn)

        self.declare_partials(["pt"], ["pt_in", "rho", "mdot", "area"], rows=arange, cols=arange)
        self.declare_partials(["pt"], ["dynamic_pressure_loss_factor"], rows=arange, cols=np.zeros((nn,)))

    def compute(self, inputs, outputs):
        dynamic_pressure = 0.5 * inputs["mdot"] ** 2 / inputs["rho"] / inputs["area"] ** 2

        pt_out = inputs["pt_in"] - dynamic_pressure * inputs["dynamic_pressure_loss_factor"]
        outputs["pt"] = pt_out

    def compute_partials(self, inputs, J):
        nn = self.options["num_nodes"]

        J["pt", "pt_in"] = np.ones((nn,))
        J["pt", "mdot"] = -inputs["dynamic_pressure_loss_factor"] * inputs["mdot"] / inputs["rho"] / inputs["area"] ** 2
        J["pt", "rho"] = (
            0.5
            * inputs["dynamic_pressure_loss_factor"]
            * inputs["mdot"] ** 2
            / inputs["rho"] ** 2
            / inputs["area"] ** 2
        )
        J["pt", "area"] = (
            inputs["dynamic_pressure_loss_factor"] * inputs["mdot"] ** 2 / inputs["rho"] / inputs["area"] ** 3
        )
        J["pt", "dynamic_pressure_loss_factor"] = -0.5 * inputs["mdot"] ** 2 / inputs["rho"] / inputs["area"] ** 2


class OutletNozzle(Group):
    """This group is designed to be the farthest downstream point in a ducted heat exchanger model.
       Mass flow is set based on the upstream total pressure and ambient static pressure.

    Inputs
    ------
    p_exit : float
        Exit static pressure. Normally set to ambient flight pressure (vector, Pa)
    pt : float
        Total pressure upstream of the nozzle (vector, Pa)
    Tt : float
        Total temperature upstream of the nozzle (vector, K)
    area : float
        Nozzle cross sectional area (vector, m**2)

    Outputs
    -------
    mdot : float
        Mass flow (vector, kg/s)

    Options
    -------
    num_nodes : int
        Number of conditions to analyze
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of flight/control conditions")

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_subsystem(
            "pressureloss", NozzlePressureLoss(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"]
        )
        self.add_subsystem(
            "pressureratio", DuctExitPressureRatioImplicit(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"]
        )
        self.add_subsystem(
            "machimplicit", DuctExitMachNumber(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"]
        )
        self.add_subsystem("temp", TemperatureIsentropic(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("pressure", PressureIsentropic(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("density", DensityIdealGas(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("speedsound", SpeedOfSound(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem(
            "massflow", MassFlow(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=[("mdot", "mdot_actual")]
        )


class ImplicitCompressibleDuct(Group):
    """
    Ducted heat exchanger with compressible flow assumptions
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of analysis points")
        self.options.declare("cfg", default=0.98, desc="Gross thrust coefficient")

    def setup(self):
        nn = self.options["num_nodes"]

        iv = self.add_subsystem(
            "dv", IndepVarComp(), promotes_outputs=["cp", "*_1", "*_2", "*_3", "*_nozzle", "convergence_hack"]
        )
        iv.add_output("cp", val=1002.93, units="J/kg/K")

        iv.add_output("area_1", val=60, units="inch**2")
        iv.add_output("delta_p_1", val=np.zeros((nn,)), units="Pa")
        iv.add_output("heat_in_1", val=np.zeros((nn,)), units="W")
        iv.add_output("pressure_recovery_1", val=np.ones((nn,)))
        iv.add_output("loss_factor_1", val=0.0)

        iv.add_output("delta_p_2", val=np.ones((nn,)) * 0.0, units="Pa")
        iv.add_output("heat_in_2", val=np.ones((nn,)) * 0.0, units="W")
        iv.add_output("pressure_recovery_2", val=np.ones((nn,)))

        iv.add_output("pressure_recovery_3", val=np.ones((nn,)))

        iv.add_output("area_nozzle", val=58 * np.ones((nn,)), units="inch**2")
        iv.add_output("convergence_hack", val=-20, units="Pa")

        self.add_subsystem("mdotguess", FlowMatcher(num_nodes=nn), promotes=["*"])

        self.add_subsystem("inlet", Inlet(num_nodes=nn), promotes_inputs=[("p", "p_inf"), ("T", "T_inf"), "Utrue"])

        self.add_subsystem(
            "sta1",
            DuctStation(num_nodes=nn),
            promotes_inputs=[
                "mdot",
                "cp",
                ("area", "area_1"),
                ("delta_p", "delta_p_1"),
                ("heat_in", "heat_in_1"),
                ("pressure_recovery", "pressure_recovery_1"),
            ],
        )
        self.connect("inlet.pt", "sta1.pt_in")
        self.connect("inlet.Tt", "sta1.Tt_in")
        self.connect("loss_factor_1", "sta1.dynamic_pressure_loss_factor")

        self.add_subsystem(
            "sta2",
            DuctStation(num_nodes=nn),
            promotes_inputs=[
                "mdot",
                "cp",
                ("area", "area_2"),
                ("delta_p", "delta_p_2"),
                ("heat_in", "heat_in_2"),
                ("pressure_recovery", "pressure_recovery_2"),
            ],
        )
        self.connect("sta1.pt_out", "sta2.pt_in")
        self.connect("sta1.Tt_out", "sta2.Tt_in")

        self.add_subsystem(
            "hx",
            HXGroup(num_nodes=nn),
            promotes_inputs=[
                ("mdot_cold", "mdot"),
                "mdot_hot",
                "T_in_hot",
                "rho_hot",
                "ac|propulsion|thermal|hx|n_wide_cold",
            ],
            promotes_outputs=["T_out_hot"],
        )
        self.connect("sta2.T", "hx.T_in_cold")
        self.connect("sta2.rho", "hx.rho_cold")

        self.add_subsystem(
            "sta3",
            DuctStation(num_nodes=nn),
            promotes_inputs=["mdot", "cp", ("pressure_recovery", "pressure_recovery_3"), ("area", "area_3")],
        )
        self.connect("sta2.pt_out", "sta3.pt_in")
        self.connect("sta2.Tt_out", "sta3.Tt_in")
        self.connect("hx.delta_p_cold", "sta3.delta_p")
        self.connect("hx.heat_transfer", "sta3.heat_in")
        self.connect("hx.frontal_area", ["area_2", "area_3"])
        self.add_subsystem(
            "pexit",
            AddSubtractComp(
                output_name="p_exit", input_names=["p_inf", "convergence_hack"], vec_size=[nn, 1], units="Pa"
            ),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )
        self.add_subsystem(
            "nozzle",
            OutletNozzle(num_nodes=nn),
            promotes_inputs=["mdot", "p_exit", ("area", "area_nozzle")],
            promotes_outputs=["mdot_actual"],
        )
        self.connect("sta3.pt_out", "nozzle.pt_in")
        self.connect("sta3.Tt_out", "nozzle.Tt")

        self.add_subsystem(
            "force",
            NetForce(num_nodes=nn, cfg=self.options["cfg"]),
            promotes_inputs=["mdot", "p_inf", ("Utrue_inf", "Utrue"), "area_nozzle"],
        )
        self.connect("nozzle.p", "force.p_nozzle")
        self.connect("nozzle.rho", "force.rho_nozzle")


class FlowMatcher(ImplicitComponent):
    def initialize(self):
        self.options.declare("num_nodes", default=1)

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("mdot_actual", shape=(nn,), units="kg/s")
        self.add_output("mdot", shape=(nn,), units="kg/s", lower=0.005, upper=15.0, val=9.0)
        arange = np.arange(0, nn)
        self.declare_partials(["mdot"], ["mdot_actual"], rows=arange, cols=arange, val=np.ones((nn,)))
        self.declare_partials(["mdot"], ["mdot"], rows=arange, cols=arange, val=-np.ones((nn,)))

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals["mdot"] = inputs["mdot_actual"] - outputs["mdot"]


class ImplicitCompressibleDuct_ExternalHX(Group):
    """
    Ducted heat exchanger with compressible flow assumptions
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of analysis points")
        self.options.declare("cfg", default=0.98, desc="Gross thrust coefficient")

    def setup(self):
        nn = self.options["num_nodes"]

        iv = self.add_subsystem("dv", IndepVarComp(), promotes_outputs=["cp", "*_1", "*_2", "*_3", "convergence_hack"])
        iv.add_output("cp", val=1002.93, units="J/kg/K")

        iv.add_output("area_1", val=60, units="inch**2")
        iv.add_output("delta_p_1", val=np.zeros((nn,)), units="Pa")
        iv.add_output("heat_in_1", val=np.zeros((nn,)), units="W")
        iv.add_output("pressure_recovery_1", val=np.ones((nn,)))
        iv.add_output("loss_factor_1", val=0.0)

        iv.add_output("delta_p_2", val=np.ones((nn,)) * 0.0, units="Pa")
        iv.add_output("heat_in_2", val=np.ones((nn,)) * 0.0, units="W")
        iv.add_output("pressure_recovery_2", val=np.ones((nn,)))

        iv.add_output("pressure_recovery_3", val=np.ones((nn,)))

        # iv.add_output('area_nozzle', val=58*np.ones((nn,)), units='inch**2')
        iv.add_output("convergence_hack", val=-40, units="Pa")
        dvlist = [["area_nozzle_in", "area_nozzle", 58 * np.ones((nn,)), "inch**2"]]
        self.add_subsystem("dvpassthru", DVLabel(dvlist), promotes_inputs=["*"], promotes_outputs=["*"])

        self.add_subsystem("mdotguess", FlowMatcher(num_nodes=nn), promotes=["*"])

        self.add_subsystem("inlet", Inlet(num_nodes=nn), promotes_inputs=[("p", "p_inf"), ("T", "T_inf"), "Utrue"])

        self.add_subsystem(
            "sta1",
            DuctStation(num_nodes=nn),
            promotes_inputs=[
                "mdot",
                "cp",
                ("area", "area_1"),
                ("delta_p", "delta_p_1"),
                ("heat_in", "heat_in_1"),
                ("pressure_recovery", "pressure_recovery_1"),
            ],
        )
        self.connect("inlet.pt", "sta1.pt_in")
        self.connect("inlet.Tt", "sta1.Tt_in")
        self.connect("loss_factor_1", "sta1.dynamic_pressure_loss_factor")

        self.add_subsystem(
            "sta2",
            DuctStation(num_nodes=nn),
            promotes_inputs=[
                "mdot",
                "cp",
                ("area", "area_2"),
                ("delta_p", "delta_p_2"),
                ("heat_in", "heat_in_2"),
                ("pressure_recovery", "pressure_recovery_2"),
            ],
        )
        self.connect("sta1.pt_out", "sta2.pt_in")
        self.connect("sta1.Tt_out", "sta2.Tt_in")

        # in to HXGroup:
        # duct.mdot -> mdot_cold
        # mdot_hot
        # T_in_hot
        # rho_hot
        # duct.sta2.T -> T_in_cold
        # duct.sta2.rho -> rho_cold

        # out from HXGroup
        # T_out_hot
        # delta_p_cold ->sta3.delta_p
        # heat_transfer -> sta3.heat_in
        # frontal_area -> 'area_2', 'area_3'

        self.add_subsystem(
            "sta3",
            DuctStation(num_nodes=nn),
            promotes_inputs=["mdot", "cp", ("pressure_recovery", "pressure_recovery_3"), ("area", "area_3")],
        )
        self.connect("sta2.pt_out", "sta3.pt_in")
        self.connect("sta2.Tt_out", "sta3.Tt_in")

        self.add_subsystem(
            "pexit",
            AddSubtractComp(
                output_name="p_exit", input_names=["p_inf", "convergence_hack"], vec_size=[nn, 1], units="Pa"
            ),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )
        self.add_subsystem(
            "nozzle",
            OutletNozzle(num_nodes=nn),
            promotes_inputs=["mdot", "p_exit", ("area", "area_nozzle")],
            promotes_outputs=["mdot_actual"],
        )
        self.connect("sta3.pt_out", "nozzle.pt_in")
        self.connect("sta3.Tt_out", "nozzle.Tt")

        self.add_subsystem(
            "force",
            NetForce(num_nodes=nn, cfg=self.options["cfg"]),
            promotes_inputs=["mdot", "p_inf", ("Utrue_inf", "Utrue"), "area_nozzle"],
        )
        self.connect("nozzle.p", "force.p_nozzle")
        self.connect("nozzle.rho", "force.rho_nozzle")
