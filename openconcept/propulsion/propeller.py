import numpy as np
from openmdao.api import ExplicitComponent
from openmdao.api import Group
from .empirical_data.prop_maps import (
    propeller_map_Raymer,
    propeller_map_highpower,
    static_propeller_map_Raymer,
    static_propeller_map_highpower,
)


class SimplePropeller(Group):
    """This propeller is representative of a constant-speed prop.

    The technology may be old.
    A general, empirical efficiency map for a constant speed turboprop is used for most of the flight regime.
    A static thrust coefficient map (from Raymer) is used for advance ratio < 0.2 (low speed).
    Linear interpolation from static thrust to dynamic thrust tables at J = 0.1 to 0.2.

    Inputs
    ------
    shaft_power_in : float
        Shaft power driving the prop (vector, W)
    diameter: float
        Prop diameter (scalar, m)
    rpm : float
        Prop RPM (vector, RPM)
    fltcond|rho : float
        Air density (vector, kg/m**3)
    fltcond|Utrue : float
        True airspeed (vector, m/s)

    Outputs
    -------
    thrust : float
        Propeller thrust (vector, N)
    component_weight : float
        Prop weight (scalar, kg)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length; default 1)
    num_blades : int
        Number of propeller blades (default 4)
    design_cp : float
        Design cruise power coefficient (cp)
    design_J : float
        Design advance ratio (J)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of flight/control conditions")
        self.options.declare("num_blades", default=4, desc="Number of prop blades")
        self.options.declare("design_cp", default=0.2, desc="Design cruise power coefficient cp")
        self.options.declare("design_J", default=2.2, desc="Design advance ratio J=V/n/D")

    def setup(self):
        nn = self.options["num_nodes"]
        n_blades = self.options["num_blades"]
        if n_blades == 3:
            propmap = propeller_map_Raymer(nn)
            staticpropmap = static_propeller_map_Raymer(nn)
        if n_blades == 4:
            propmap = propeller_map_highpower(nn)
            staticpropmap = static_propeller_map_highpower(nn)
        else:
            raise NotImplementedError("You will need to define a propeller map valid for this number of blades")
        self.add_subsystem("propcalcs", PropCoefficients(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("propmap", propmap, promotes_outputs=["*"])
        self.add_subsystem("staticpropmap", staticpropmap, promotes_outputs=["*"])
        self.add_subsystem("thrustcalc", ThrustCalc(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem(
            "propweights", WeightCalc(num_blades=n_blades), promotes_inputs=["*"], promotes_outputs=["*"]
        )
        self.connect("cp", "propmap.cp")
        self.connect("cp", "staticpropmap.cp")
        self.connect("J", "propmap.J")


class WeightCalc(ExplicitComponent):
    def initialize(self):
        self.options.declare("num_blades", default=4, desc="Number of prop blades")

    def setup(self):
        self.add_input("power_rating", units="hp", desc="Propulsor power rating")
        self.add_input("diameter", units="ft", desc="Prop diameter in feet")
        self.add_output("component_weight", units="lb", desc="Propeller weight")
        self.declare_partials("component_weight", ["power_rating", "diameter"])

    def compute(self, inputs, outputs):
        # Method from Roskam SVC6p90eq6.14
        Kprop2 = 0.108  # for turboprops
        n_blades = self.options["num_blades"]
        W_prop = Kprop2 * (inputs["diameter"] * inputs["power_rating"] * n_blades**0.5) ** 0.782
        outputs["component_weight"] = W_prop

    def compute_partials(self, inputs, J):
        Kprop2 = 0.108  # for turboprops
        n_blades = self.options["num_blades"]
        J["component_weight", "power_rating"] = (
            0.782
            * Kprop2
            * (inputs["diameter"] * inputs["power_rating"] * n_blades**0.5) ** (0.782 - 1)
            * (inputs["diameter"] * n_blades**0.5)
        )
        J["component_weight", "diameter"] = (
            0.782
            * Kprop2
            * (inputs["diameter"] * inputs["power_rating"] * n_blades**0.5) ** (0.782 - 1)
            * (inputs["power_rating"] * n_blades**0.5)
        )


class ThrustCalc(ExplicitComponent):
    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of flight/control conditions")

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("cp", desc="power coefficient", shape=(nn,))
        self.add_input("eta_prop", desc="propulsive efficiency factor", shape=(nn,))
        self.add_input("J", desc="advance ratio", shape=(nn,))
        self.add_input("fltcond|rho", units="kg / m ** 3", desc="Air density", shape=(nn,))
        self.add_input("rpm", units="rpm", val=2500 * np.ones(nn), desc="Prop speed in rpm")
        self.add_input("diameter", units="m", val=2.5, desc="Prop diameter in m")
        self.add_input("ct_over_cp", val=1.5 * np.ones(nn), desc="ct/cp from raymer for static condition")

        self.add_output("thrust", desc="Propeller thrust", units="N", shape=(nn,))

        self.declare_partials(
            "thrust", ["cp", "eta_prop", "J", "fltcond|rho", "rpm", "ct_over_cp"], rows=range(nn), cols=range(nn)
        )
        self.declare_partials("thrust", "diameter")

    def compute(self, inputs, outputs):
        # for advance ratio j between 0.10 and 0.20, linearly interpolate the thrust coefficient from the two surrogate models
        jinterp_min = 0.10
        jinterp_max = 0.20
        j = inputs["J"]
        # print(inputs['eta_prop'])
        static_idx = np.where(j <= jinterp_min)
        dynamic_idx = np.where(j >= jinterp_max)
        tmp = np.logical_and(j > jinterp_min, j < jinterp_max)
        interp_idx = np.where(tmp)

        cp = inputs["cp"]
        nn = self.options["num_nodes"]
        ct = np.zeros(nn)
        ct1 = inputs["ct_over_cp"] * cp
        ct2 = cp * inputs["eta_prop"] / j
        # if j <= jinterp_min:
        ct[static_idx] = ct1[static_idx]
        # if j > jinterp_min and < jinterp_max:
        interv = np.ones(nn) * jinterp_max - np.ones(nn) * jinterp_min
        interp1 = (np.ones(nn) * jinterp_max - j) / interv
        interp2 = (j - np.ones(nn) * jinterp_min) / interv
        ct[interp_idx] = interp1[interp_idx] * ct1[interp_idx] + interp2[interp_idx] * ct2[interp_idx]
        # else if j >= jinterp_max
        ct[dynamic_idx] = ct2[dynamic_idx]

        outputs["thrust"] = ct * inputs["fltcond|rho"] * (inputs["rpm"] / 60.0) ** 2 * inputs["diameter"] ** 4

    def compute_partials(self, inputs, J):
        # for advance ratio j between 0.10 and 0.20, linearly interpolate between the two surrogate models
        jinterp_min = 0.10
        jinterp_max = 0.20
        j = inputs["J"]
        cp = inputs["cp"]
        nn = self.options["num_nodes"]

        static_idx = np.where(j <= jinterp_min)
        dynamic_idx = np.where(j >= jinterp_max)
        tmp = np.logical_and(j > jinterp_min, j < jinterp_max)
        interp_idx = np.where(tmp)

        dctdcp = np.zeros(nn)
        ct1 = inputs["ct_over_cp"] * cp
        ct2 = cp * inputs["eta_prop"] / j
        dct1dcp = inputs["ct_over_cp"]
        dct2dcp = inputs["eta_prop"] / j

        # if j <= jinterp_min:
        dctdcp[static_idx] = dct1dcp[static_idx]
        # if j > jinterp_min and < jinterp_max:
        interv = np.ones(nn) * jinterp_max - np.ones(nn) * jinterp_min
        interp1 = (np.ones(nn) * jinterp_max - j) / interv
        interp2 = (j - np.ones(nn) * jinterp_min) / interv
        dctdcp[interp_idx] = interp1[interp_idx] * dct1dcp[interp_idx] + interp2[interp_idx] * dct2dcp[interp_idx]
        # else if j >= jinterp_max
        dctdcp[dynamic_idx] = dct2dcp[dynamic_idx]

        ct = cp * dctdcp

        j_thrust_ct_over_cp = np.zeros(nn)
        j_thrust_eta_prop = np.zeros(nn)
        j_thrust_j = np.zeros(nn)

        thrust_over_ct = inputs["fltcond|rho"] * (inputs["rpm"] / 60.0) ** 2 * inputs["diameter"] ** 4
        thrust = ct * inputs["fltcond|rho"] * (inputs["rpm"] / 60.0) ** 2 * inputs["diameter"] ** 4

        J["thrust", "fltcond|rho"] = thrust / inputs["fltcond|rho"]
        J["thrust", "rpm"] = 2 * thrust / inputs["rpm"]
        J["thrust", "diameter"] = 4 * thrust / inputs["diameter"]
        J["thrust", "cp"] = dctdcp * inputs["fltcond|rho"] * (inputs["rpm"] / 60.0) ** 2 * inputs["diameter"] ** 4

        # if j <= jinterp_min:
        j_thrust_ct_over_cp[static_idx] = thrust[static_idx] / inputs["ct_over_cp"][static_idx]
        j_thrust_eta_prop[static_idx] = np.zeros(static_idx[0].shape)
        j_thrust_j[static_idx] = np.zeros(static_idx[0].shape)

        # j < jinterp_max:
        j_thrust_ct_over_cp[interp_idx] = (
            thrust_over_ct[interp_idx] * interp1[interp_idx] * (ct1[interp_idx] / inputs["ct_over_cp"][interp_idx])
        )
        j_thrust_eta_prop[interp_idx] = (
            thrust_over_ct[interp_idx] * interp2[interp_idx] * (ct2[interp_idx] / inputs["eta_prop"][interp_idx])
        )
        j_thrust_j[interp_idx] = thrust_over_ct[interp_idx] * (
            -ct1[interp_idx] / interv[interp_idx]
            + ct2[interp_idx] / interv[interp_idx]
            - interp2[interp_idx] * ct2[interp_idx] / j[interp_idx]
        )

        # else:
        j_thrust_ct_over_cp[dynamic_idx] = np.zeros(dynamic_idx[0].shape)
        j_thrust_eta_prop[dynamic_idx] = thrust[dynamic_idx] / inputs["eta_prop"][dynamic_idx]
        j_thrust_j[dynamic_idx] = -thrust[dynamic_idx] / j[dynamic_idx]

        J["thrust", "ct_over_cp"] = j_thrust_ct_over_cp
        J["thrust", "eta_prop"] = j_thrust_eta_prop
        J["thrust", "J"] = j_thrust_j


class PropCoefficients(ExplicitComponent):
    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of flight/control conditions")

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("shaft_power_in", units="W", desc="Input shaft power", shape=(nn,), val=5 * np.ones(nn))
        self.add_input("diameter", units="m", val=2.5, desc="Prop diameter")
        self.add_input("rpm", units="rpm", val=2500 * np.ones(nn), desc="Propeller shaft speed")
        self.add_input("fltcond|rho", units="kg / m ** 3", desc="Air density", shape=(nn,))
        self.add_input("fltcond|Utrue", units="m/s", desc="Flight speed", shape=(nn,))

        # outputs and partials
        self.add_output(
            "cp", desc="Power coefficient", val=0.1 * np.ones(nn), lower=np.zeros(nn), upper=np.ones(nn) * 2.4
        )
        self.add_output(
            "J", desc="Advance ratio", val=0.2 ** np.ones(nn), lower=1e-4 * np.ones(nn), upper=4.0 * np.ones(nn)
        )
        self.add_output("prop_Vtip", desc="Propeller tip speed", shape=(nn,))

        self.declare_partials("cp", "diameter")
        self.declare_partials("cp", "shaft_power_in", rows=range(nn), cols=range(nn))
        self.declare_partials("cp", ["fltcond|rho", "rpm"], rows=range(nn), cols=range(nn))
        self.declare_partials("J", "diameter")
        self.declare_partials("J", ["fltcond|Utrue", "rpm"], rows=range(nn), cols=range(nn))
        self.declare_partials("prop_Vtip", "rpm", rows=range(nn), cols=range(nn))
        self.declare_partials("prop_Vtip", "diameter")

    def compute(self, inputs, outputs):
        # print('Prop shaft power input: ' + str(inputs['shaft_power_in']))
        outputs["cp"] = (
            inputs["shaft_power_in"] / inputs["fltcond|rho"] / (inputs["rpm"] / 60) ** 3 / inputs["diameter"] ** 5
        )
        # print('cp: '+str(outputs['cp']))
        outputs["J"] = 60.0 * inputs["fltcond|Utrue"] / inputs["rpm"] / inputs["diameter"]
        # print('U:'+str(inputs['fltcond|Utrue']))
        # print('J: '+str(outputs['J']))
        outputs["prop_Vtip"] = inputs["rpm"] / 60 * np.pi * inputs["diameter"]

    def compute_partials(self, inputs, J):
        nn = self.options["num_nodes"]
        cpval = inputs["shaft_power_in"] / inputs["fltcond|rho"] / (inputs["rpm"] / 60) ** 3 / inputs["diameter"] ** 5
        jval = 60.0 * inputs["fltcond|Utrue"] / inputs["rpm"] / inputs["diameter"]
        J["cp", "shaft_power_in"] = 1 / inputs["fltcond|rho"] / (inputs["rpm"] / 60) ** 3 / inputs["diameter"] ** 5
        J["cp", "fltcond|rho"] = -cpval / inputs["fltcond|rho"]
        J["cp", "rpm"] = -3.0 * cpval / inputs["rpm"]
        J["cp", "diameter"] = -5.0 * cpval / inputs["diameter"]
        J["J", "fltcond|Utrue"] = jval / inputs["fltcond|Utrue"]
        J["J", "rpm"] = -jval / inputs["rpm"]
        J["J", "diameter"] = -jval / inputs["diameter"]
        J["prop_Vtip", "rpm"] = 1 / 60 * np.pi * inputs["diameter"] * np.ones(nn)
        J["prop_Vtip", "diameter"] = inputs["rpm"] / 60 * np.pi
