from __future__ import division
import numpy as np
from openmdao.api import ExplicitComponent, Problem, NewtonSolver, Group, MetaModelStructuredComp, ExecComp, n2
from openconcept.utilities.math.multiply_divide_comp import ElementMultiplyDivideComp
from math import sin, cos, radians

class TankCompositeThickness(ExplicitComponent):
    """
    Computes the wall thickness and composite weight of a cylindrical composite overwrap
    pressure vessel (COPV) with hemispherical end caps given a design pressure,
    geometry, safety factor, and fiber direction parameters.

    The pressure vessel is assumed to have a windings in the hoop direction and in
    the +/- alpha direction where the angle is defined from the hoop winding
    direction. The layer thicknesses are sized using a netting analysis, which
    ignores the contribution of the resin to the yielding.

    The liner is assumed to bear no loads and is not considered in determining
    the thickness of the composite layers.

    Inputs
    ------
    alpha : float
        Angle from vertical of the helical windings (scalar, radians)
    design_pressure : float
        Maximum expected operating pressure (MEOP) (scalar, Pa)
    radius : float
        Radius of the cylinder and hemispherical end caps (scalar, m)
    length : float
        Length of JUST THE CYLIDRICAL part of the tank (scalar, m)
    
    Outputs
    -------
    thickness : float
        Total thickness of composite layers (scalar, m)
    composite_weight : float
        Weight of composite layers (scalar, kg)
    
    Options
    -------
    safety_factor : float
        Safety factor for sizing composite thicknesses, applied to MEOP; default 3
    yield_stress : float
        Tensile yield stress of composite filament fiber direction (Pa); default 7 GPa
        from Toray T1100G carbon fiber
        https://www.toraycma.com/wp-content/uploads/T1100G-Technical-Data-Sheet-1.pdf.pdf
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
        self.options.declare('yield_stress', default=7e9, desc='Tensile yield stress of fibers in Pa')
        self.options.declare('density', default=1580., desc='Density of composite in kg/m^3')
        self.options.declare('fiber_volume_fraction', default=0.586, desc='Fraction of volume taken up by fibers')
    
    def setup(self):
        self.add_input('alpha', val=45.*np.pi/180., units='rad')
        self.add_input('design_pressure', val=70e6, units='Pa')
        self.add_input('radius', val=0.5, units='m')
        self.add_input('length', val=2., units='m')

        self.add_output('thickness', lower=0., units='m')
        self.add_output('composite_weight', lower=0., units='kg')

        self.declare_partials('thickness', ['alpha', 'design_pressure', 'radius'])
        self.declare_partials('composite_weight', ['alpha', 'design_pressure', 'radius', 'length'])

    def compute(self, inputs, outputs):
        alpha = inputs['alpha']
        p = inputs['design_pressure']
        r = inputs['radius']
        L = inputs['length']
        SF = self.options['safety_factor']
        yield_stress = self.options['yield_stress']
        density = self.options['density']
        vol_frac = self.options['fiber_volume_fraction']

        helical_fiber_thickness = p*r*SF / (2*yield_stress*np.sin(alpha)**2)
        hoop_fiber_thickness = p*r*SF / yield_stress - helical_fiber_thickness*np.cos(alpha)**2
        outputs['thickness'] = (helical_fiber_thickness + hoop_fiber_thickness) / vol_frac

        composite_volume = outputs['thickness'] * (2*np.pi*r*L + 4*np.pi*r**2)
        outputs['composite_weight'] = composite_volume * density
    
    def compute_partials(self, inputs, J):
        alpha = inputs['alpha']
        p = inputs['design_pressure']
        r = inputs['radius']
        L = inputs['length']
        SF = self.options['safety_factor']
        yield_stress = self.options['yield_stress']
        density = self.options['density']
        vol_frac = self.options['fiber_volume_fraction']

        sin_a = np.sin(alpha)
        cos_a = np.cos(alpha)
        surf_area = 2*np.pi*r*L + 4*np.pi*r**2
        helical_fiber_thickness = p*r*SF / (2*yield_stress*sin_a**2)
        hoop_fiber_thickness = p*r*SF / yield_stress - helical_fiber_thickness*cos_a**2
        thickness = (helical_fiber_thickness + hoop_fiber_thickness) / vol_frac

        J['thickness', 'alpha'] = p*r*SF / (2*yield_stress*vol_frac) * (-2*cos_a / sin_a**3 + 2*cos_a/sin_a**3)
        J['thickness', 'design_pressure'] = (r*SF / (2*yield_stress*sin_a**2) + r*SF / yield_stress - r*SF / (2*yield_stress*sin_a**2)*cos_a**2) / vol_frac
        J['thickness', 'radius'] = (p*SF / (2*yield_stress*sin_a**2) + p*SF / yield_stress - p*SF / (2*yield_stress*sin_a**2)*cos_a**2) / vol_frac
        J['composite_weight', 'alpha'] = J['thickness', 'alpha'] * surf_area * density
        J['composite_weight', 'design_pressure'] = J['thickness', 'design_pressure'] * surf_area * density
        J['composite_weight', 'radius'] = (J['thickness', 'radius'] * surf_area + thickness * (2*np.pi*L + 8*np.pi*r)) * density
        J['composite_weight', 'length'] = thickness * 2*np.pi*r * density
