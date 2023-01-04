from __future__ import division
import numpy as np
import openmdao.api as om

class CompositeOverwrap(om.ExplicitComponent):
    """
    Computes the wall thickness and composite weight of a cylindrical composite overwrap
    pressure vessel (COPV) with hemispherical end caps given a design pressure,
    geometry, safety factor, and fiber direction parameters.

    The pressure vessel is assumed to have a windings in the hoop direction and
    helical windings at +/- theta. The layer thicknesses are sized using a netting
    analysis, which ignores the contribution of the resin to the yielding. With
    this procedure, the angle of the helical winding ends up dropping out of the
    total composite thickness equation, which is why it is not an input.

    The liner is assumed to bear no loads and is not considered in determining
    the thickness of the composite layers.

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
    design_pressure : float
        Maximum expected operating pressure (MEOP) (scalar, Pa)
    radius : float
        Inner radius of the cylinder and hemispherical end caps (scalar, m)
    length : float
        Length of JUST THE CYLIDRICAL part of the tank (scalar, m)
    
    Outputs
    -------
    thickness : float
        Total thickness of composite layers (scalar, m)
    weight : float
        Weight of composite layers (scalar, kg)
    
    Options
    -------
    safety_factor : float
        Safety factor for sizing composite thicknesses, applied to MEOP; default 3
    yield_stress : float
        Tensile yield stress of composite in fiber direction (Pa); default 3.896 GPa
        from Toray T1100G UD carbon fiber
        https://www.toraycma.com/wp-content/uploads/3960-PREPREG-SYSTEM.pdf
    density : float
        Density of composite (kg/m^3); default 1.58 g/cm^3 computed from Toray 3960 material
        system with T1100G fibers assuming fiber density of 1.79 g/cm^3, resin density
        of 1.274 g/cm^3, and resin mass fraction of 33.5%
        https://www.toraycma.com/wp-content/uploads/3960-PREPREG-SYSTEM.pdf
    fiber_volume_fraction : float
        Fraction of volume of composite taken up by fibers; default 0.586 computed using
        same values as density above
    """
    def initialize(self):
        self.options.declare('safety_factor', default=3., desc='Safety factor on composite thickness')
        self.options.declare('yield_stress', default=3.896e9, desc='Tensile yield stress in fiber direction in Pa')
        self.options.declare('density', default=1580., desc='Density of composite in kg/m^3')
        self.options.declare('fiber_volume_fraction', default=0.586, desc='Fraction of volume taken up by fibers')
    
    def setup(self):
        self.add_input('design_pressure', val=70e6, units='Pa')
        self.add_input('radius', val=0.5, units='m')
        self.add_input('length', val=2., units='m')

        self.add_output('thickness', lower=0., units='m')
        self.add_output('weight', lower=0., units='kg')

        self.declare_partials('thickness', ['design_pressure', 'radius'])
        self.declare_partials('weight', ['design_pressure', 'radius', 'length'])

    def compute(self, inputs, outputs):
        p = inputs['design_pressure']
        r = inputs['radius']
        L = inputs['length']
        SF = self.options['safety_factor']
        yield_stress = self.options['yield_stress']
        density = self.options['density']
        vol_frac = self.options['fiber_volume_fraction']

        fiber_thickness = 3*p*r*SF / (2*yield_stress)
        outputs['thickness'] = fiber_thickness / vol_frac

        composite_volume = (4/3*np.pi*(r + outputs['thickness'])**3 + np.pi*(r + outputs['thickness'])**2*L) - \
                           (4/3*np.pi*r**3 + np.pi*r**2*L)
        outputs['weight'] = composite_volume * density
    
    def compute_partials(self, inputs, J):
        p = inputs['design_pressure']
        r = inputs['radius']
        L = inputs['length']
        SF = self.options['safety_factor']
        yield_stress = self.options['yield_stress']
        density = self.options['density']
        vol_frac = self.options['fiber_volume_fraction']

        fiber_thickness = 3*p*r*SF / (2*yield_stress)
        thickness = fiber_thickness / vol_frac

        J['thickness', 'design_pressure'] = 3*r*SF / (2*yield_stress*vol_frac)
        J['thickness', 'radius'] = 3*p*SF / (2*yield_stress*vol_frac)
        J['weight', 'design_pressure'] = J['thickness', 'design_pressure'] * (4*np.pi*(r + thickness)**2 + 2*np.pi*(r + thickness)*L) * density
        J['weight', 'radius'] = density * ((1 + J['thickness', 'radius']) * (4*np.pi*(r + thickness)**2 + 2*np.pi*(r + thickness)*L) - (4*np.pi*r**2 + 2*np.pi*r*L))
        J['weight', 'length'] = density * (np.pi*(r + thickness)**2 - np.pi*r**2)


class COPVLinerWeight(om.ExplicitComponent):
    """
    Computes the weight of the metallic pressure vessel liner used
    to prevent leakage through the pressure vessel. This model assumes
    the liner is not load-bearing, so it has no effect on the sizing
    of the composite overwrap.

    This component uses a simple surface area calculation of a
    cylindrical pressure vessel with hemispherical end caps. That
    surface area is multiplied by the thickness of the liner and
    its density to find the weight.

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
    radius : float
        Inner radius of the cylinder and hemispherical end caps (scalar, m)
    length : float
        Length of JUST THE CYLIDRICAL part of the tank (scalar, m)
    
    Outputs
    -------
    weight : float
        Weight of the liner (scalar, kg)
    
    Options
    -------
    density : float
        Density of the liner material (kg/m^3); default 2700 kg/m^3 aluminum 6061
    thickness : float
        Liner thickness (m); default 0.5 mm
    """
    def initialize(self):
        self.options.declare('density', default=2700., desc='Liner material density (kg/m^3)')
        self.options.declare('thickness', default=0.5e-3, desc='Liner thickness (m))')
    
    def setup(self):
        self.add_input('radius', val=0.5, units='m')
        self.add_input('length', val=2., units='m')
        self.add_output('weight', lower=0., units='kg')
        self.declare_partials('weight', ['radius', 'length'])

    def compute(self, inputs, outputs):
        r = inputs['radius']
        L = inputs['length']
        outputs['weight'] = (2*np.pi*r*L + 4*np.pi*r**2) * self.options['density'] * self.options['thickness']
    
    def compute_partials(self, inputs, J):
        r = inputs['radius']
        L = inputs['length']
        J['weight', 'radius'] = (2*np.pi*L + 8*np.pi*r) * self.options['density'] * self.options['thickness']
        J['weight', 'length'] = 2*np.pi*r * self.options['density'] * self.options['thickness']


class COPVInsulationWeight(om.ExplicitComponent):
    """
    Computes the weight of the insulation outside the composite overwrap.
    Unlike the liner weight this calculation does not assume the
    insulation is thin, since it computes the total volume and
    subtracts out the inner volume.

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
    radius : float
        Inner radius of insulation in the cylinder and hemispherical
        end caps; usually inner tank radius + composite thickness (scalar, m)
    length : float
        Length of JUST THE CYLIDRICAL part of the tank (scalar, m)
    thickness : float
        Thickness of the insulation layer (scalar, m)
    
    Outputs
    -------
    weight : float
        Weight of the liner (scalar, kg)
    
    Options
    -------
    density : float
        Density of the insulation material (kg/m^3); default 32.1 kg/m^3 rigid open cell
        polyurethane, other options listed on page 16 of
        https://ntrs.nasa.gov/api/citations/20020085127/downloads/20020085127.pdf
    fairing_areal_density : float
        If insulation is included, a fairing must be used to prevent damage to the insulation
        layer, default 1.304 kg/m^2 from https://www.mdpi.com/1996-1073/11/1/105 (scalar, kg/m^2)
    """
    def initialize(self):
        self.options.declare('density', default=32.1, desc='Insulation material density (kg/m^3)')
        self.options.declare('fairing_areal_density', default=1.304, desc='Mass of fairing per area (kg/m^2)')
    
    def setup(self):
        self.add_input('radius', val=0.5, units='m')
        self.add_input('length', val=2., units='m')
        self.add_input('thickness', val=0.05, units='m')
        self.add_output('weight', lower=0., units='kg')
        self.declare_partials('weight', ['radius', 'length', 'thickness'])

    def compute(self, inputs, outputs):
        r = inputs['radius']
        L = inputs['length']
        t = inputs['thickness']
        volume = (np.pi*L*(r + t)**2 + 4/3*np.pi*(r + t)**3) - (np.pi*L*r**2 + 4/3*np.pi*r**3)
        surf_area = 2*np.pi*(r + t)*L + 4*np.pi*(r + t)**2
        outputs['weight'] = volume * self.options['density'] + surf_area * self.options['fairing_areal_density']
    
    def compute_partials(self, inputs, J):
        r = inputs['radius']
        L = inputs['length']
        t = inputs['thickness']
        J['weight', 'radius'] = ((2*np.pi*L*(r + t) + 4*np.pi*(r + t)**2) - (2*np.pi*L*r + 4*np.pi*r**2)) * self.options['density'] + \
                                (2*np.pi*L + 8*np.pi*(r + t)) * self.options['fairing_areal_density']
        J['weight', 'length'] = (np.pi*(r + t)**2 - np.pi*r**2) * self.options['density'] + \
                                2*np.pi*(r + t)*self.options['fairing_areal_density']
        J['weight', 'thickness'] = (2*np.pi*L*(r + t) + 4*np.pi*(r + t)**2) * self.options['density'] + \
                                   (2*np.pi*L + 8*np.pi*(r + t)) * self.options['fairing_areal_density']


if __name__ == "__main__":
    # Validation from Argonne National Lab 149 L, 700 bar hydrogen tank
    # https://www1.eere.energy.gov/hydrogenandfuelcells/pdfs/compressedtank_storage.pdf
    from openconcept.utilities.math.add_subtract_comp import AddSubtractComp
    p = om.Problem()
    p.model.add_subsystem('composite', CompositeOverwrap(safety_factor=2.25, yield_stress=2.55e9, fiber_volume_fraction=0.6),
                          promotes_inputs=['design_pressure', 'radius', 'length'], promotes_outputs=[('weight', 'w_composite')])
    p.model.add_subsystem('liner', COPVLinerWeight(density=970, thickness=0.005),  # 5 mm HDPE liner
                          promotes_inputs=['radius', 'length'], promotes_outputs=[('weight', 'w_liner')])
    p.model.add_subsystem('insulation', COPVInsulationWeight(fairing_areal_density=0), promotes_inputs=['radius', 'length'], promotes_outputs=[('weight', 'w_insulation')])
    add = AddSubtractComp()
    add.add_equation('weight', ['w_composite', 'w_liner', 'w_insulation'], units='kg')
    p.model.add_subsystem('total', add, promotes=['w_composite', 'w_liner', 'w_insulation', 'weight'])

    p.setup()
    
    # Radius and length computed based on 149 L volume and length-to-diameter ratio of 3
    p.set_val('radius', 0.20718, units='m')
    p.set_val('length', 0.828716, units='m')
    p.set_val('insulation.thickness', 0., units='inch')  # no insulation (gaseous hydrogen)
    p.set_val('design_pressure', 700., units='bar')

    p.run_model()
    p.model.list_outputs(units=True)

    # The true tank weight is 108.6 kg; this estimation is 9.5% low,
    # which is good enough for our purposes
