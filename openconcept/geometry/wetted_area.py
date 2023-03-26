import numpy as np
import openmdao.api as om


class CylinderSurfaceArea(om.ExplicitComponent):
    """
    Compute the surface area of a cylinder. This can be
    used to estimate the wetted area of a fuselage or
    engine nacelle.

    Inputs
    ------
    L : float
        Cylinder length (scalar, m)
    D : float
        Cylinder diameter (scalar, m)

    Outputs
    -------
    A : float
        Cylinder surface area (scalar, sq m)
    """

    def setup(self):
        self.add_input("L", units="m")
        self.add_input("D", units="m")
        self.add_output("A", units="m**2")
        self.declare_partials("A", ["L", "D"])

    def compute(self, inputs, outputs):
        outputs["A"] = np.pi * inputs["D"] * inputs["L"]

    def compute_partials(self, inputs, J):
        J["A", "L"] = np.pi * inputs["D"]
        J["A", "D"] = np.pi * inputs["L"]
