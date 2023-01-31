import numpy as np
import openmdao.api as om
from openconcept.utilities import AddSubtractComp


class VacuumTankStructure(om.Group):
    """
    Sizes the structure and computes the weight of the tank's vacuum walls. For now this uses
    the same pressure vessel computation, sized by hoop stress, for both the inner and outer
    tank walls. In reality, the outer wall is sized by other constraints, likely buckling,
    so a very high safety factor is used for the outer wall.

    TODO: Switch to using ASME BPVC Section VIII UG-28 to compute outer wall thickness

          |--- length ---|
         . -------------- .         ---
      ,'                    `.       | radius
     /                        \      |
    |                          |    ---
     \                        /
      `.                    ,'
         ` -------------- '

    Inputs
    ------
    environment_design_pressure : float
        Maximum environment exterior pressure expected, probably ~1 atmosphere (scalar, Pa)
    max_expected_operating_pressure : float
        Maximum expected operating pressure of tank (scalar, Pa)
    vacuum_gap : float
        Thickness of vacuum gap, used to compute radius of outer vacuum wall (scalar, m)
    radius : float
        Tank inner radius of the cylinder and hemispherical end caps (scalar, m)
    length : float
        Length of JUST THE CYLIDRICAL part of the tank (scalar, m)

    Outputs
    -------
    weight : float
        Weight of the tank walls (scalar, kg)

    Options
    -------
    weight_fudge_factor : float
        Multiplier on tank weight to account for supports, valves, etc., by default 1.3
    inner_wall_safety_factor : float
        Safety factor for sizing inner wall, by default 1.5
    inner_yield_stress : float
        Yield stress of inner wall material (Pa), by default Al 2014-T6 taken from Table IV of
        Sullivan et al. 2006 (https://ntrs.nasa.gov/citations/20060021606)
    inner_density : float
        Density of inner wall material (kg/m^3), by default Al 2014-T6 taken from Table IV of
        Sullivan et al. 2006 (https://ntrs.nasa.gov/citations/20060021606)
    outer_wall_safety_factor : float
        Safety factor for sizing outer wall, by default 10
    outer_yield_stress : float
        Yield stress of outer wall material (Pa), by default LiAl 2090 taken from Table XIII of
        Sullivan et al. 2006 (https://ntrs.nasa.gov/citations/20060021606)
    outer_density : float
        Density of outer wall material (kg/m^3), by default LiAl 2090 taken from Table XIII of
        Sullivan et al. 2006 (https://ntrs.nasa.gov/citations/20060021606)
    """

    def initialize(self):
        self.options.declare("weight_fudge_factor", default=1.3, desc="Weight multiplier to account for other stuff")
        self.options.declare("inner_safety_factor", default=1.5, desc="Safety factor on inner wall thickness")
        self.options.declare("inner_yield_stress", default=413.7e6, desc="Yield stress of inner wall material in Pa")
        self.options.declare("inner_density", default=2796.0, desc="Density of inner wall material in kg/m^3")
        # TODO: very large outer safety factor is necessary because buckling is probably what actually sizes the outer
        #       tank wall, switch to using ASME BPVC Section VIII UG-28 (I think this is appropriate) for more realism
        self.options.declare("outer_safety_factor", default=10.0, desc="Safety factor on outer wall thickness")
        self.options.declare("outer_yield_stress", default=470.2e6, desc="Yield stress of outer wall material in Pa")
        self.options.declare("outer_density", default=2699.0, desc="Density of outer wall material in kg/m^3")

    def setup(self):
        # Inner tank wall thickness and weight computation
        self.add_subsystem(
            "inner_wall",
            PressureVesselWallThickness(
                safety_factor=self.options["inner_safety_factor"],
                yield_stress=self.options["inner_yield_stress"],
                density=self.options["inner_density"],
            ),
            promotes_inputs=[("design_pressure_differential", "max_expected_operating_pressure"), "radius", "length"],
        )

        # Compute radius of outer tank wall
        self.add_subsystem(
            "outer_radius",
            AddSubtractComp(
                output_name="outer_radius",
                input_names=["radius", "vacuum_gap"],
                scaling_factors=[1, 1],
                lower=0.0,
                units="m",
            ),
            promotes_inputs=["radius", "vacuum_gap"],
        )

        # Outer tank wall thickness and weight computation
        self.add_subsystem(
            "outer_wall",
            PressureVesselWallThickness(
                safety_factor=self.options["outer_safety_factor"],
                yield_stress=self.options["outer_yield_stress"],
                density=self.options["outer_density"],
            ),
            promotes_inputs=[("design_pressure_differential", "environment_design_pressure"), "length"],
        )
        self.connect("outer_radius.outer_radius", "outer_wall.radius")

        # Compute total weight multiplied by fudge factor
        W_mult = self.options["weight_fudge_factor"]
        self.add_subsystem(
            "total_weight",
            AddSubtractComp(
                output_name="weight",
                input_names=["W_outer", "W_inner"],
                scaling_factors=[W_mult, W_mult],
                lower=0.0,
                units="kg",
            ),
            promotes_outputs=["weight"],
        )
        self.connect("inner_wall.weight", "total_weight.W_inner")
        self.connect("outer_wall.weight", "total_weight.W_outer")

        # Set defaults for inputs promoted from multiple sources
        self.set_input_defaults("radius", 0.5, units="m")


class PressureVesselWallThickness(om.ExplicitComponent):
    """
    Compute the wall thickness of a metallic pressure vessel to support a specified
    pressure load. The model assumes an isotropic wall material, hence the metallic
    constraint. This uses a simple equation to compute the hoop stress (also referred
    to as Barlow's formula) to size the wall thickness.

    Theoretically, it applies for the cases where the internal pressure is greater than
    external and when internal pressure is less. However, the case where internal pressure
    is less than external usually is sized by some buckling constraint. A greater safety
    factor for this case should be used if this component is expected to size the wall.

    This component assumes that the wall is thin enough relative to the radius such that
    it is valid to compute the weight as the product of the surface area, wall thickness,
    and material density.

          |--- length ---|
         . -------------- .         ---
      ,'                    `.       | radius
     /                        \      |
    |                          |    ---
     \                        /
      `.                    ,'
         ` -------------- '

    Inputs
    ------
    design_pressure_differential : float
        The maximum pressure differential between the interior and exterior of the
        pressure vessel that is used to size the wall thickness; should ALWAYS
        be positive, otherwise wall thickness and weight will be negative (scalar, Pa)
    radius : float
        Inner radius of the cylinder and hemispherical end caps (scalar, m)
    length : float
        Length of JUST THE CYLIDRICAL part of the tank (scalar, m)

    Outputs
    -------
    thickness : float
        Pressure vessel wall thickness (scalar, m)
    weight : float
        Weight of the wall (scalar, kg)

    Options
    -------
    safety_factor : float
        Safety factor for sizing wall, by default 2
    yield_stress : float
        Yield stress of wall material (Pa), by default LiAl 2090 taken from Table XIII of
        Sullivan et al. 2006 (https://ntrs.nasa.gov/citations/20060021606)
    density : float
        Density of wall material (kg/m^3), by default LiAl 2090 taken from Table XIII of
        Sullivan et al. 2006 (https://ntrs.nasa.gov/citations/20060021606)
    """

    def initialize(self):
        self.options.declare("safety_factor", default=2.0, desc="Safety factor on wall thickness")
        self.options.declare("yield_stress", default=470.2e6, desc="Yield stress of wall material in Pa")
        self.options.declare("density", default=2699.0, desc="Density of wall material in kg/m^3")

    def setup(self):
        self.add_input("design_pressure_differential", val=3e5, units="Pa")
        self.add_input("radius", val=0.5, units="m")
        self.add_input("length", val=2.0, units="m")

        self.add_output("thickness", lower=0.0, units="m")
        self.add_output("weight", lower=0.0, units="kg")

        self.declare_partials("thickness", ["design_pressure_differential", "radius"])
        self.declare_partials("weight", ["design_pressure_differential", "radius", "length"])

    def compute(self, inputs, outputs):
        p = inputs["design_pressure_differential"]
        r = inputs["radius"]
        L = inputs["length"]
        SF = self.options["safety_factor"]
        yield_stress = self.options["yield_stress"]
        density = self.options["density"]

        outputs["thickness"] = p * r * SF / yield_stress

        surface_area = 4 * np.pi * r**2 + 2 * np.pi * r * L
        outputs["weight"] = surface_area * outputs["thickness"] * density

    def compute_partials(self, inputs, J):
        p = inputs["design_pressure_differential"]
        r = inputs["radius"]
        L = inputs["length"]
        SF = self.options["safety_factor"]
        yield_stress = self.options["yield_stress"]
        density = self.options["density"]

        t = p * r * SF / yield_stress

        J["thickness", "design_pressure_differential"] = r * SF / yield_stress
        J["thickness", "radius"] = p * SF / yield_stress

        A = 4 * np.pi * r**2 + 2 * np.pi * r * L
        dAdr = 8 * np.pi * r + 2 * np.pi * L
        dAdL = 2 * np.pi * r
        J["weight", "design_pressure_differential"] = A * J["thickness", "design_pressure_differential"] * density
        J["weight", "radius"] = (dAdr * t + A * J["thickness", "radius"]) * density
        J["weight", "length"] = dAdL * t * density


if __name__ == "__main__":
    p = om.Problem()
    p.model.add_subsystem("model", VacuumTankStructure(), promotes=["*"])
    p.setup()

    p.set_val("environment_design_pressure", 1, units="atm")
    p.set_val("max_expected_operating_pressure", 3, units="bar")
    p.set_val("vacuum_gap", 5, units="inch")
    p.set_val("radius", 8.5 / 2, units="ft")
    p.set_val("length", 4, units="ft")

    p.run_model()

    p.model.list_outputs(units=True)

    r = p.get_val("radius", units="m").item()
    L = p.get_val("length", units="m").item()
    W_LH2 = (4 / 3 * np.pi * r**3 + np.pi * r**2 * L) * 70 * 0.95
    W_tank = p.get_val("weight", units="kg").item()
    print(f"-------- Approximate gravimetric efficiency: {W_LH2 / (W_LH2 + W_tank) * 100:.1f}% --------")
