import numpy as np
import openmdao.api as om
from openconcept.utilities import AddSubtractComp


class VacuumTankWeight(om.Group):
    """
    Sizes the structure and computes the weight of the tank's vacuum walls.
    This includes the weight of MLI.

    .. code-block:: text

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
    N_layers : float
        Number of reflective sheild layers in the MLI, should be at least ~10 for model
        to retain reasonable accuracy (scalar, dimensionless)

    Outputs
    -------
    weight : float
        Weight of the tank walls (scalar, kg)

    Options
    -------
    weight_fudge_factor : float
        Multiplier on tank weight to account for supports, valves, etc., by default 1.1
    stiffening_multiplier : float
        Machining stiffeners into the inner side of the vacuum shell enhances its buckling
        performance, enabling weight reductions. The value provided in this option is a
        multiplier on the outer wall thickness. The default value of 0.8 is higher than it
        would be if it were purely empirically determined from Sullivan et al. 2006
        (https://ntrs.nasa.gov/citations/20060021606), but has been made much more
        conservative to fall more in line with ~60% gravimetric efficiency tanks
    inner_safety_factor : float
        Safety factor for sizing inner wall, by default 1.5
    inner_yield_stress : float
        Yield stress of inner wall material (Pa), by default Al 2014-T6 taken from Table IV of
        Sullivan et al. 2006 (https://ntrs.nasa.gov/citations/20060021606)
    inner_density : float
        Density of inner wall material (kg/m^3), by default Al 2014-T6 taken from Table IV of
        Sullivan et al. 2006 (https://ntrs.nasa.gov/citations/20060021606)
    outer_safety_factor : float
        Safety factor for sizing outer wall, by default 2
    outer_youngs_modulus : float
        Young's modulus of outer wall material (Pa), by default LiAl 2090 taken from Table XIII of
        Sullivan et al. 2006 (https://ntrs.nasa.gov/citations/20060021606)
    outer_density : float
        Density of outer wall material (kg/m^3), by default LiAl 2090 taken from Table XIII of
        Sullivan et al. 2006 (https://ntrs.nasa.gov/citations/20060021606)
    """

    def initialize(self):
        self.options.declare("weight_fudge_factor", default=1.1, desc="Weight multiplier to account for other stuff")
        self.options.declare("stiffening_multiplier", default=0.8, desc="Multiplier on wall thickness")
        self.options.declare("inner_safety_factor", default=1.5, desc="Safety factor on inner wall thickness")
        self.options.declare("inner_yield_stress", default=413.7e6, desc="Yield stress of inner wall material in Pa")
        self.options.declare("inner_density", default=2796.0, desc="Density of inner wall material in kg/m^3")
        self.options.declare("outer_safety_factor", default=2.0, desc="Safety factor on outer wall thickness")
        self.options.declare("outer_youngs_modulus", default=8.0e10, desc="Young's modulus of outer wall material, Pa")
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
            VacuumWallThickness(
                safety_factor=self.options["outer_safety_factor"],
                stiffening_multiplier=self.options["stiffening_multiplier"],
                youngs_modulus=self.options["outer_youngs_modulus"],
                density=self.options["outer_density"],
            ),
            promotes_inputs=[("design_pressure_differential", "environment_design_pressure"), "length"],
        )
        self.connect("outer_radius.outer_radius", "outer_wall.radius")

        # Compute the weight of the MLI
        self.add_subsystem("MLI", MLIWeight(), promotes_inputs=["radius", "length", "N_layers"])

        # Compute total weight multiplied by fudge factor
        W_mult = self.options["weight_fudge_factor"]
        self.add_subsystem(
            "total_weight",
            AddSubtractComp(
                output_name="weight",
                input_names=["W_outer", "W_inner", "W_MLI"],
                scaling_factors=[W_mult, W_mult, W_mult],
                lower=0.0,
                units="kg",
            ),
            promotes_outputs=["weight"],
        )
        self.connect("inner_wall.weight", "total_weight.W_inner")
        self.connect("outer_wall.weight", "total_weight.W_outer")
        self.connect("MLI.weight", "total_weight.W_MLI")

        # Set defaults for inputs promoted from multiple sources
        self.set_input_defaults("radius", 1.0, units="m")
        self.set_input_defaults("length", 0.5, units="m")


class PressureVesselWallThickness(om.ExplicitComponent):
    """
    Compute the wall thickness of a metallic pressure vessel to support a specified
    pressure load. The model assumes an isotropic wall material, hence the metallic
    constraint. This uses a simple equation to compute the hoop stress (also referred
    to as Barlow's formula) to size the wall thickness.

    This component assumes that the wall is thin enough relative to the radius such that
    it is valid to compute the weight as the product of the surface area, wall thickness,
    and material density.

    .. code-block:: text

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


class VacuumWallThickness(om.ExplicitComponent):
    """
    Compute the wall thickness when the exterior pressure is greater than the interior
    one. This applies to the outer wall of a vacuum-insulated tank. It does this by
    computing the necessary wall thickness for a cylindrical shell under uniform compression
    and sphere under uniform compression and taking the maximum thickness of the two.

    The equations are from Table 15.2 of Roark's Formulas for Stress and Strain, 9th
    Edition by Budynas and Sadegh.

    This component assumes that the wall is thin relative to the radius.

    .. code-block:: text

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
        be positive (scalar, Pa)
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
        Safety factor for sizing wall applied to design pressure, by default 2
    stiffening_multiplier : float
        Machining stiffeners into the inner side of the vacuum shell enhances its buckling
        performance, enabling weight reductions. The value provided in this option is a
        multiplier on the outer wall thickness. The default value of 0.8 is higher than it
        would be if it were purely empirically determined from Sullivan et al. 2006
        (https://ntrs.nasa.gov/citations/20060021606), but has been made much more
        conservative to fall more in line with ~60% gravimetric efficiency tanks
    youngs_modulus : float
        Young's modulus of wall material (Pa), by default LiAl 2090 taken from Table XIII of
        Sullivan et al. 2006 (https://ntrs.nasa.gov/citations/20060021606)
    density : float
        Density of wall material (kg/m^3), by default LiAl 2090 taken from Table XIII of
        Sullivan et al. 2006 (https://ntrs.nasa.gov/citations/20060021606)
    """

    def initialize(self):
        self.options.declare("safety_factor", default=2.0, desc="Safety factor on wall thickness")
        self.options.declare("stiffening_multiplier", default=0.8, desc="Multiplier on wall thickness")
        self.options.declare("youngs_modulus", default=8.0e10, desc="Young's modulus of wall material in Pa")
        self.options.declare("density", default=2699.0, desc="Density of wall material in kg/m^3")

    def setup(self):
        self.add_input("design_pressure_differential", val=101325.0, units="Pa")
        self.add_input("radius", val=0.5, units="m")
        self.add_input("length", val=2.0, units="m")

        self.add_output("thickness", lower=0.0, units="m")
        self.add_output("weight", lower=0.0, units="kg")

        self.declare_partials(["thickness", "weight"], ["design_pressure_differential", "radius", "length"])

    def compute(self, inputs, outputs):
        p = inputs["design_pressure_differential"]
        r = inputs["radius"]
        L = inputs["length"]
        SF = self.options["safety_factor"]
        E = self.options["youngs_modulus"]
        density = self.options["density"]
        stiff_mult = self.options["stiffening_multiplier"]

        # Compute the thickness necessary for the cylindrical portion
        t_cyl = (p * SF * L * r**1.5 / (0.92 * E)) ** (1 / 2.5)

        # Compute the thickness necessary for the spherical portion
        t_sph = r * np.sqrt(p * SF / (0.365 * E))

        # Take the maximum of the two, when r and L are small the KS
        # isn't a great approximation and the weighting parameter needs
        # to be very high, so just let it be C1 discontinuous
        outputs["thickness"] = stiff_mult * np.maximum(t_cyl, t_sph)

        surface_area = 4 * np.pi * r**2 + 2 * np.pi * r * L
        outputs["weight"] = surface_area * outputs["thickness"] * density

    def compute_partials(self, inputs, J):
        p = inputs["design_pressure_differential"]
        r = inputs["radius"]
        L = inputs["length"]
        SF = self.options["safety_factor"]
        E = self.options["youngs_modulus"]
        density = self.options["density"]
        stiff_mult = self.options["stiffening_multiplier"]

        # Compute the thickness necessary for the cylindrical portion
        t_cyl = (p * SF * L * r**1.5 / (0.92 * E)) ** (1 / 2.5)
        if L < 1e-6:
            dtcyl_dp = 0.0
            dtcyl_dr = 0.0
            dtcyl_dL = 0.0
        else:
            first_term = (p * SF * L * r**1.5 / (0.92 * E)) ** (1 / 2.5 - 1) / 2.5
            dtcyl_dp = first_term * SF * L * r**1.5 / (0.92 * E)
            dtcyl_dr = first_term * p * SF * L * r**0.5 / (0.92 * E) * 1.5
            dtcyl_dL = first_term * p * SF * r**1.5 / (0.92 * E)

        # Compute the thickness necessary for the spherical portion
        t_sph = r * np.sqrt(p * SF / (0.365 * E))
        dtsph_dp = 0.5 * r * (p * SF / (0.365 * E)) ** (-0.5) * SF / (0.365 * E)
        dtsph_dr = t_sph / r
        dtsph_dL = 0.0

        # Derivative is from whichever thickness is greater
        use_cyl = t_cyl.item() > t_sph.item()
        J["thickness", "design_pressure_differential"] = (dtcyl_dp if use_cyl else dtsph_dp) * stiff_mult
        J["thickness", "radius"] = (dtcyl_dr if use_cyl else dtsph_dr) * stiff_mult
        J["thickness", "length"] = (dtcyl_dL if use_cyl else dtsph_dL) * stiff_mult

        t = stiff_mult * np.maximum(t_cyl, t_sph)
        A = 4 * np.pi * r**2 + 2 * np.pi * r * L
        dAdr = 8 * np.pi * r + 2 * np.pi * L
        dAdL = 2 * np.pi * r
        J["weight", "design_pressure_differential"] = A * J["thickness", "design_pressure_differential"] * density
        J["weight", "radius"] = (dAdr * t + A * J["thickness", "radius"]) * density
        J["weight", "length"] = (dAdL * t + A * J["thickness", "length"]) * density


class MLIWeight(om.ExplicitComponent):
    """
    Compute the weight of the MLI given the tank geometry and number of MLI layers.
    Foil and spacer areal density per layer estimated from here:
    https://frakoterm.com/cryogenics/multi-layer-insulation-mli/

    Inputs
    ------
    radius : float
        Inner radius of the cylinder and hemispherical end caps. This value
        does not include the insulation (scalar, m).
    length : float
        Length of JUST THE CYLIDRICAL part of the tank (scalar, m)
    N_layers : float
        Number of reflective sheild layers in the MLI, should be at least ~10 for model
        to retain reasonable accuracy (scalar, dimensionless)

    Outputs
    -------
    weight : float
        Total weight of the MLI insulation (scalar, kg)

    Options
    -------
    foil_layer_areal_weight : float
        Areal weight of a single foil layer, by default 18e-3 (scalar, kg/m^2)
    spacer_layer_areal_weight : float
        Areal weight of a single spacer layer, by default 12e-3 (scalar, kg/m^2)
    """

    def initialize(self):
        self.options.declare("foil_layer_areal_weight", default=18e-3, desc="Areal weight of foil layer in kg/m^2")
        self.options.declare("spacer_layer_areal_weight", default=12e-3, desc="Areal weight of spacer layer in kg/m^2")

    def setup(self):
        self.add_input("radius", units="m")
        self.add_input("length", units="m")
        self.add_input("N_layers")

        self.add_output("weight", units="kg")

        self.declare_partials("weight", ["radius", "length", "N_layers"])

    def compute(self, inputs, outputs):
        r = inputs["radius"]
        L = inputs["length"]
        N = inputs["N_layers"]
        W_foil = self.options["foil_layer_areal_weight"]
        W_spacer = self.options["spacer_layer_areal_weight"]

        # Compute surface area
        A = 4 * np.pi * r**2 + 2 * np.pi * r * L

        outputs["weight"] = (W_foil + W_spacer) * N * A

    def compute_partials(self, inputs, J):
        r = inputs["radius"]
        L = inputs["length"]
        N = inputs["N_layers"]
        W_foil = self.options["foil_layer_areal_weight"]
        W_spacer = self.options["spacer_layer_areal_weight"]

        # Compute surface area
        A = 4 * np.pi * r**2 + 2 * np.pi * r * L

        J["weight", "N_layers"] = (W_foil + W_spacer) * A
        J["weight", "radius"] = (W_foil + W_spacer) * N * (8 * np.pi * r + 2 * np.pi * L)
        J["weight", "length"] = (W_foil + W_spacer) * N * (2 * np.pi * r)


if __name__ == "__main__":
    p = om.Problem()
    p.model.add_subsystem("model", VacuumTankWeight(), promotes=["*"])
    p.setup(force_alloc_complex=True)

    p.set_val("environment_design_pressure", 1.0, units="atm")
    p.set_val("max_expected_operating_pressure", 2.5, units="bar")
    p.set_val("vacuum_gap", 4, units="inch")
    p.set_val("radius", 8.5 / 2, units="ft")
    p.set_val("length", 0.0, units="ft")

    p.run_model()

    # p.check_partials(method="cs", compact_print=True)

    p.model.list_outputs(units=True)

    r = p.get_val("radius", units="m").item()
    L = p.get_val("length", units="m").item()
    W_LH2 = (4 / 3 * np.pi * r**3 + np.pi * r**2 * L) * 70 * 0.95
    W_tank = p.get_val("weight", units="kg").item()
    print(f"\n-------- Approximate gravimetric efficiency: {W_LH2 / (W_LH2 + W_tank) * 100:.1f}% --------")
