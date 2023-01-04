from openconcept.propulsion import SimpleTurboshaft, SimplePropeller
from openconcept.utilities import DVLabel, AddSubtractComp, ElementMultiplyDivideComp
from openmdao.api import Group


class TurbopropPropulsionSystem(Group):
    """
    This is an example model of the simplest possible propulsion system
    consisting of a constant-speed prop and a turboshaft.

    This is the Pratt and Whitney Canada PT6A-66D with 4-bladed
    propeller used by the SOCATA-DAHER TBM-850.

    Inputs
    ------
    ac|propulsion|engine|rating : float
        The maximum rated shaft power of the engine (scalar, default 850 hp)
    ac|propulsion|propeller|diameter : float
        Diameter of the propeller (scalar, default 2.3 m)
    throttle : float
        Throttle for the turboshaft (vector)
    fltcond|rho : float
        Air density (vector, kg/m**3)
    fltcond|Utrue : float
        True airspeed (vector, m/s)

    Outputs
    -------
    thrust : float
        Thrust force (vector, N)
    fuel_flow : float
        Fuel mass flow rate (vector, kg/s)

    Options
    -------
    num_nodes : float
        Number of analysis points to run (default 1)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of mission analysis points to run")

    def setup(self):
        nn = self.options["num_nodes"]

        # rename incoming design variables
        dvlist = [
            ["ac|propulsion|engine|rating", "eng1_rating", 850, "hp"],
            ["ac|propulsion|propeller|diameter", "prop1_diameter", 2.3, "m"],
        ]
        self.add_subsystem("dvs", DVLabel(dvlist), promotes_inputs=["*"], promotes_outputs=["*"])

        # introduce model components
        self.add_subsystem(
            "eng1",
            SimpleTurboshaft(num_nodes=nn, weight_inc=0.14 / 1000, weight_base=104),
            promotes_inputs=["throttle", ("shaft_power_rating", "ac|propulsion|engine|rating")],
            promotes_outputs=["fuel_flow"],
        )
        self.add_subsystem(
            "prop1",
            SimplePropeller(num_nodes=nn, num_blades=4, design_J=2.2, design_cp=0.55),
            promotes_inputs=[
                "fltcond|*",
                ("power_rating", "ac|propulsion|engine|rating"),
                ("diameter", "ac|propulsion|propeller|diameter"),
            ],
            promotes_outputs=["thrust"],
        )

        # Set default values for the engine rating and prop diameter
        self.set_input_defaults("ac|propulsion|engine|rating", 850.0, units="hp")
        self.set_input_defaults("ac|propulsion|propeller|diameter", 2.3, units="m")

        # Connect shaft power from turboshaft to propeller
        self.connect("eng1.shaft_power_out", "prop1.shaft_power_in")


class TwinTurbopropPropulsionSystem(Group):
    """
    This is an example model multiple constant-speed props and turboshafts.
    These are two P&W Canada PT6A-135A with 4-bladed Hartzell propellers used by the Beechcraft King Air C90GT
    https://www.easa.europa.eu/sites/default/files/dfu/TCDS_EASA-IM-A-503_C90-Series%20issue%206.pdf

    Inputs
    ------
    ac|propulsion|engine|rating : float
        The maximum rated shaft power of the engine (scalar, default 850 hp)
    ac|propulsion|propeller|diameter : float
        Diameter of the propeller (scalar, default 2.3 m)
    throttle : float
        Throttle for the turboshaft (vector)
    fltcond|rho : float
        Air density (vector, kg/m**3)
    fltcond|Utrue : float
        True airspeed (vector, m/s)
    propulsor_active : float
        1 if second propulsor is active or 0 if not (vector)

    Outputs
    -------
    thrust : float
        Thrust force (vector, N)
    fuel_flow : float
        Fuel mass flow rate (vector, kg/s)

    Options
    -------
    num_nodes : float
        Number of analysis points to run (default 1)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of mission analysis points to run")

    def setup(self):
        nn = self.options["num_nodes"]

        # Introduce turboshaft and propeller components (one for each side)
        self.add_subsystem(
            "eng1",
            SimpleTurboshaft(num_nodes=nn, weight_inc=0.14 / 1000, weight_base=104),
            promotes_inputs=["throttle", ("shaft_power_rating", "ac|propulsion|engine|rating")],
        )
        self.add_subsystem(
            "prop1",
            SimplePropeller(num_nodes=nn, num_blades=4, design_J=2.2, design_cp=0.55),
            promotes_inputs=[
                "fltcond|*",
                ("power_rating", "ac|propulsion|engine|rating"),
                ("diameter", "ac|propulsion|propeller|diameter"),
            ],
        )
        self.add_subsystem(
            "eng2",
            SimpleTurboshaft(num_nodes=nn, weight_inc=0.14 / 1000, weight_base=104),
            promotes_inputs=[("shaft_power_rating", "ac|propulsion|engine|rating")],
        )
        self.add_subsystem(
            "prop2",
            SimplePropeller(num_nodes=nn, num_blades=4, design_J=2.2, design_cp=0.55),
            promotes_inputs=[
                "fltcond|*",
                ("power_rating", "ac|propulsion|engine|rating"),
                ("diameter", "ac|propulsion|propeller|diameter"),
            ],
        )

        # Set default values for the engine rating and prop diameter
        self.set_input_defaults("ac|propulsion|engine|rating", 750.0, units="hp")
        self.set_input_defaults("ac|propulsion|propeller|diameter", 2.28, units="m")

        # Propulsion models expect a high-level 'throttle' parameter and a 'propulsor_active' flag to set individual throttles
        failedengine = ElementMultiplyDivideComp()
        failedengine.add_equation("eng2throttle", input_names=["throttle", "propulsor_active"], vec_size=nn)
        self.add_subsystem("failedengine", failedengine, promotes_inputs=["throttle", "propulsor_active"])
        self.connect("failedengine.eng2throttle", "eng2.throttle")

        # Connect components to each other
        self.connect("eng1.shaft_power_out", "prop1.shaft_power_in")
        self.connect("eng2.shaft_power_out", "prop2.shaft_power_in")

        # Add up the weights, thrusts and fuel flows
        add1 = AddSubtractComp(
            output_name="fuel_flow", input_names=["eng1_fuel_flow", "eng2_fuel_flow"], vec_size=nn, units="kg/s"
        )
        add1.add_equation(output_name="thrust", input_names=["prop1_thrust", "prop2_thrust"], vec_size=nn, units="N")
        add1.add_equation(output_name="engines_weight", input_names=["eng1_weight", "eng2_weight"], units="kg")
        add1.add_equation(output_name="propellers_weight", input_names=["prop1_weight", "prop2_weight"], units="kg")
        self.add_subsystem("adder", subsys=add1, promotes_inputs=["*"], promotes_outputs=["*"])
        self.connect("prop1.thrust", "prop1_thrust")
        self.connect("prop2.thrust", "prop2_thrust")
        self.connect("eng1.fuel_flow", "eng1_fuel_flow")
        self.connect("eng2.fuel_flow", "eng2_fuel_flow")
        self.connect("prop1.component_weight", "prop1_weight")
        self.connect("prop2.component_weight", "prop2_weight")
        self.connect("eng1.component_weight", "eng1_weight")
        self.connect("eng2.component_weight", "eng2_weight")
