"""Aerodynamic analysis routines usable for multiple purposes / flight phases"""
from openmdao.api import ExplicitComponent
import numpy as np
from openconcept.utilities.constants import GRAV_CONST


class PolarDrag(ExplicitComponent):
    """
    Calculates drag force based on drag polar formulation

    Inputs
    ------
    fltcond|CL : float
        Lift coefficient (vector, dimensionless)
    fltcond|q : float
        Dynamic pressure (vector, Pascals)
    ac|geom|wing|S_ref : float
        Reference wing area (scalar, m**2)
    ac|geom|wing|AR : float
        Wing aspect ratio (scalar, dimensionless)
    CD0 : float
        Zero-lift drag coefficient (scalar, dimensionless)
    e : float
        Wing Oswald efficiency (scalar, dimensionless)

    Outputs
    -------
    drag : float
        Drag force (vector, Newtons)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length) (default 1)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of nodes to compute")

    def setup(self):
        nn = self.options["num_nodes"]
        arange = np.arange(0, nn)
        self.add_input("fltcond|CL", shape=(nn,))
        self.add_input("fltcond|q", units="N * m**-2", shape=(nn,))
        self.add_input("ac|geom|wing|S_ref", units="m **2")
        self.add_input("CD0")
        self.add_input("e")
        self.add_input("ac|geom|wing|AR")
        self.add_output("drag", units="N", shape=(nn,))

        self.declare_partials(["drag"], ["fltcond|CL", "fltcond|q"], rows=arange, cols=arange)
        self.declare_partials(
            ["drag"], ["ac|geom|wing|S_ref", "ac|geom|wing|AR", "CD0", "e"], rows=arange, cols=np.zeros(nn)
        )

    def compute(self, inputs, outputs):
        outputs["drag"] = (
            inputs["fltcond|q"]
            * inputs["ac|geom|wing|S_ref"]
            * (inputs["CD0"] + inputs["fltcond|CL"] ** 2 / np.pi / inputs["e"] / inputs["ac|geom|wing|AR"])
        )

    def compute_partials(self, inputs, J):
        J["drag", "fltcond|q"] = inputs["ac|geom|wing|S_ref"] * (
            inputs["CD0"] + inputs["fltcond|CL"] ** 2 / np.pi / inputs["e"] / inputs["ac|geom|wing|AR"]
        )
        J["drag", "fltcond|CL"] = (
            inputs["fltcond|q"]
            * inputs["ac|geom|wing|S_ref"]
            * (2 * inputs["fltcond|CL"] / np.pi / inputs["e"] / inputs["ac|geom|wing|AR"])
        )
        J["drag", "CD0"] = inputs["fltcond|q"] * inputs["ac|geom|wing|S_ref"]
        J["drag", "e"] = -(
            inputs["fltcond|q"]
            * inputs["ac|geom|wing|S_ref"]
            * inputs["fltcond|CL"] ** 2
            / np.pi
            / inputs["e"] ** 2
            / inputs["ac|geom|wing|AR"]
        )
        J["drag", "ac|geom|wing|S_ref"] = inputs["fltcond|q"] * (
            inputs["CD0"] + inputs["fltcond|CL"] ** 2 / np.pi / inputs["e"] / inputs["ac|geom|wing|AR"]
        )
        J["drag", "ac|geom|wing|AR"] = -(
            inputs["fltcond|q"]
            * inputs["ac|geom|wing|S_ref"]
            * inputs["fltcond|CL"] ** 2
            / np.pi
            / inputs["e"]
            / inputs["ac|geom|wing|AR"] ** 2
        )


class Lift(ExplicitComponent):
    """
    Calculates lift force based on CL, dynamic pressure, and wing area

    Inputs
    ------
    fltcond|CL : float
        Lift coefficient (vector, dimensionless)
    fltcond|q : float
        Dynamic pressure (vector, Pascals)
    ac|geom|wing|S_ref : float
        Reference wing area (scalar, m**2)

    Outputs
    -------
    lift : float
        Lift force (vector, Newtons)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length) (default 1)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of nodes to compute")

    def setup(self):
        nn = self.options["num_nodes"]
        arange = np.arange(0, nn)
        self.add_input("fltcond|CL", shape=(nn,))
        self.add_input("fltcond|q", units="N * m**-2", shape=(nn,))
        self.add_input("ac|geom|wing|S_ref", units="m **2")

        self.add_output("lift", units="N", shape=(nn,))
        self.declare_partials(["lift"], ["fltcond|CL", "fltcond|q"], rows=arange, cols=arange)
        self.declare_partials(["lift"], ["ac|geom|wing|S_ref"], rows=arange, cols=np.zeros(nn))

    def compute(self, inputs, outputs):
        outputs["lift"] = inputs["fltcond|q"] * inputs["ac|geom|wing|S_ref"] * inputs["fltcond|CL"]

    def compute_partials(self, inputs, J):
        J["lift", "fltcond|q"] = inputs["ac|geom|wing|S_ref"] * inputs["fltcond|CL"]
        J["lift", "fltcond|CL"] = inputs["fltcond|q"] * inputs["ac|geom|wing|S_ref"]
        J["lift", "ac|geom|wing|S_ref"] = inputs["fltcond|q"] * inputs["fltcond|CL"]


class StallSpeed(ExplicitComponent):
    """
    Calculates stall speed based on CLmax, wing area, and weight

    Inputs
    ------
    CLmax : float
        Maximum lift coefficient (scalar, dimensionless)
    weight : float
        Aircraft weight (scalar, kg)
    ac|geom|wing|S_ref : float
        Reference wing area (scalar, m**2)

    Outputs
    -------
    Vstall_eas : float
        Stall speed (scalar, m/s)
    """

    def setup(self):
        self.add_input("weight", units="kg")
        self.add_input("ac|geom|wing|S_ref", units="m**2")
        self.add_input("CLmax")
        self.add_output("Vstall_eas", units="m/s")
        self.declare_partials(["Vstall_eas"], ["weight", "ac|geom|wing|S_ref", "CLmax"])

    def compute(self, inputs, outputs):
        rho = 1.225  # kg/m3
        outputs["Vstall_eas"] = np.sqrt(
            2 * inputs["weight"] * GRAV_CONST / rho / inputs["ac|geom|wing|S_ref"] / inputs["CLmax"]
        )

    def compute_partials(self, inputs, J):
        rho = 1.225  # kg/m3
        J["Vstall_eas", "weight"] = (
            1
            / np.sqrt(2 * inputs["weight"] * GRAV_CONST / rho / inputs["ac|geom|wing|S_ref"] / inputs["CLmax"])
            * GRAV_CONST
            / rho
            / inputs["ac|geom|wing|S_ref"]
            / inputs["CLmax"]
        )
        J["Vstall_eas", "ac|geom|wing|S_ref"] = -(
            1
            / np.sqrt(2 * inputs["weight"] * GRAV_CONST / rho / inputs["ac|geom|wing|S_ref"] / inputs["CLmax"])
            * inputs["weight"]
            * GRAV_CONST
            / rho
            / inputs["ac|geom|wing|S_ref"] ** 2
            / inputs["CLmax"]
        )
        J["Vstall_eas", "CLmax"] = -(
            1
            / np.sqrt(2 * inputs["weight"] * GRAV_CONST / rho / inputs["ac|geom|wing|S_ref"] / inputs["CLmax"])
            * inputs["weight"]
            * GRAV_CONST
            / rho
            / inputs["ac|geom|wing|S_ref"]
            / inputs["CLmax"] ** 2
        )
