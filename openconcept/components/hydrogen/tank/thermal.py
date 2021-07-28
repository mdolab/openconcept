from __future__ import division
import numpy as np
import openmdao.api as om

class COPVThermalResistance(om.ExplicitComponent):
    """
    Computes thermal resistance for heat going through walls
    of composite overwrap pressure vessel. Steady heat flow
    analysis (assumes thermal mass of tank is negligible compared
    to contents). This resistance does not include the transfer of heat
    to and from the walls due to the convection of fluids on either side.
    In other words, it only includes the thermal resistance from conduction
    through the three layers of the wall in the tank.

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
        Radius inside of tank for the cylinder and hemispherical end caps (scalar, m)
    length : float
        Length of JUST THE CYLINDRICAL part of the tank
    composite_thickness : float
        Thickness of the composite overwrap (scalar, m)
    insulation_thickness : float
        Thickness of tank insulation (scalar, m)
    
    Outputs
    -------
    R_cylinder : float
        Effective thermal resistance of the cylindrical portion of the tank (scalar, K/W)
    R_sphere : float
        Effective total thermal resistance of the two end caps of the tank (scalar, K/W)
    
    Options
    -------
    liner_thickness : float
        Thickness of liner, default 0.5 mm (scalar, m)
    liner_cond : float
        Thermal conductivity of liner material, default aluminum 6061 (scalar, W/(m-K))
    composite_cond : float
        Thermal conductivity of composite overwrap used for tank walls in transverse direction
        (perpendicular to fiber direction), default of 0.7 for CFRP laminate estimated from p. 51 of
        https://www.sciencedirect.com/science/article/pii/0266353895000364 (scalar, W/(m-K))
    insulation_cond : float
        Thermal conductivity of insulation material, default rigid open cell polyuerthane listed
        on p. 16 of https://ntrs.nasa.gov/api/citations/20020085127/downloads/20020085127.pdf (scalar, W/(m-K))
    """
    def initialize(self):
        self.options.declare('liner_thickness', default=0.5e-3, desc='Liner thickness (m)')
        self.options.declare('liner_cond', default=167., desc='Thermal conductivity of liner W/(m-K)')
        self.options.declare('composite_cond', default=0.7, desc='Transverse thermal conductivity of composite W/(m-K)')
        self.options.declare('insulation_cond', default=0.0112, desc='Thermal conductivity of insulation W/(m-K)')
    
    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('radius', val=0.5, units='m')
        self.add_input('length', val=2., units='m')
        self.add_input('composite_thickness', val=0.05, units='m')
        self.add_input('insulation_thickness', val=0.1, units='m')

        self.add_output('R_cylinder', lower=0., units='K/W')
        self.add_output('R_sphere', lower=0., units='K/W')

        self.declare_partials('R_cylinder' ['radius', 'length',
                                            'composite_thickness',
                                            'insulation_thickness'])
        self.declare_partials('R_sphere' ['radius', 'composite_thickness',
                                          'insulation_thickness'])
    
    def compute(self, inputs, outputs):
        # Unpack variables for easier use
        r_inner = inputs['radius']
        L = inputs['length']
        t_liner = self.options['liner_thickness']
        k_liner = self.options['liner_cond']
        t_com = inputs['composite_thickness']
        k_com = self.options['composite_cond']
        t_ins = inputs['insulation_thickness']
        k_ins = self.options['insulation_cond']

        # Radii of interfaces between layers
        r_com_liner = r_inner + t_liner
        r_ins_com = r_com_liner + t_com
        r_outer = r_ins_com + t_ins

        # Thermal resistance of cylindrical portion
        R_liner = np.log(r_com_liner/r_inner) / (2*np.pi*L*k_liner)
        R_com = np.log(r_ins_com/r_com_liner) / (2*np.pi*L*k_com)
        R_ins = np.log(r_outer/r_ins_com) / (2*np.pi*L*k_ins)
        outputs['R_cylinder'] = R_liner + R_com + R_ins

        # Thermal resistance of spherical portion (two end caps)
        R_liner = (1/r_inner - 1/r_com_liner) / (4*np.pi*k_liner)
        R_com = (1/r_com_liner - 1/r_ins_com) / (4*np.pi*k_com)
        R_ins = (1/r_ins_com - 1/r_outer) / (4*np.pi*k_ins)
        outputs['R_sphere'] = R_liner + R_com + R_ins

    
    def compute_partials(self, inputs, J):
        # Unpack variables for easier use
        r = inputs['radius']
        L = inputs['length']
        t_liner = self.options['liner_thickness']
        k_liner = self.options['liner_cond']
        t_comp = inputs['composite_thickness']
        k_comp = self.options['composite_cond']
        t_ins = inputs['insulation_thickness']
        k_ins = self.options['insulation_cond']


class COPVHeatFromEnvironmentIntoTankWalls(om.ExplicitComponent):
    """
    Computes the amount of heat that enters the tank walls from
    natural convection around the surface of the tank. Since
    this component takes in the surface temperature of the tank,
    which is an unknown, a BalanceComp is used outside to pick
    the tank surface temperatures such that the heat entering
    the tank walls equals the heat entering the ullage/propellant
    (to meet the steady heat flow assumption).

    Assumes the surface temperature of the tank is constant. Computes
    heat transfer via natural convection through cylinder and sphere
    independently and then adds the results.

          |--- length ---| 
         . -------------- .         ---
      ,'                    `.       | radius
     /                        \      |
    |                          |    ---
     \                        /
      `.                    ,'
         ` -------------- '

    Approach adapted from
    https://ntrs.nasa.gov/api/citations/20020085127/downloads/20020085127.pdf

    Inputs
    ------
    T_surface : float
        Temperature of the tank's outer surface (vector, K)
    T_inf : float
        Temperature of the "freestream" (but stationary) air in
        which the tank is sitting (vector, K)
    radius : float
        Radius inside of tank for the cylinder and hemispherical end caps (scalar, m)
    length : float
        Length of JUST THE CYLINDRICAL part of the tank
    composite_thickness : float
        Thickness of the composite overwrap (scalar, m)
    insulation_thickness : float
        Thickness of tank insulation (scalar, m)
    
    Outputs
    -------
    heat_into_walls : float
        Heat from surrounding still air into tank walls; positive is
        heat entering tank walls (vector, W)
    
    Options
    -------
    num_nodes : float
        Number of analysis points to run (scalar, dimensionless)
    air_cond : float
        Thermal conductivity of air surrounding tank, default 0.0245 W/(m-K) (scalar, W/(m-K))
        estimated from https://www.engineeringtoolbox.com/air-properties-viscosity-conductivity-heat-capacity-d_1509.html
    liner_thickness : float
        Thickness of liner, default 0.5 mm (scalar, m)
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of design points to run')
        self.options.declare('air_cond', default=0.0245, desc='Thermal conductivity of air W/(m-K)')
        self.options.declare('liner_thickness', default=0.5e-3, desc='Liner thickness (m)')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('T_surface', val=100., units='K', shape=(nn,))
        self.add_input('T_inf', val=300., units='K', shape=(nn,))
        self.add_input('radius', val=0.5, units='m')
        self.add_input('length', val=2., units='m')
        self.add_input('composite_thickness', val=0.05, units='m')
        self.add_input('insulation_thickness', val=0.1, units='m')

        self.add_output('heat_into_walls', units='W', shape=(nn,))

        self.declare_partials('heat_into_walls', ['T_inf', 'T_surface'],
                              rows=np.arange(nn), cols=np.arange(nn))
        self.declare_partials('heat_into_walls', ['radius', 'length',
                                                  'composite_thickness',
                                                  'insulation_thickness'],
                              rows=np.arange(nn), cols=np.zeros(nn))
    
    def compute(self, inputs, outputs):
        # Unpack variables for easier use
        r_inner = inputs['radius']
        L = inputs['length']
        t_liner = self.options['liner_thickness']
        t_com = inputs['composite_thickness']
        t_ins = inputs['insulation_thickness']
        T_surf = inputs['T_surface']
        T_inf = inputs['T_inf']
        k_air = self.options['air_cond']

        # Compute outer radius
        r_outer = r_inner + t_liner + t_com + t_ins
        D = 2*r_outer  # diameter of tank

        # Rayleigh and Prandtl numbers
        alpha = -3.119e-6 + 3.541e-8*T_inf + 1.679e-10*T_inf**2  # air diffusivity
        nu = -2.079e-6 + 2.777e-8*T_inf + 1.077e-10*T_inf**2  # air viscosity
        Pr = nu/alpha  # Prandtl number
        R_ad = 9.807 * (T_inf - T_surf) / T_inf * D**3 / (nu * alpha)

        # Nusselt numbers for cylinder and sphere
        Nu_cyl = (0.6 + 0.387 * R_ad**(1/6) / (1 + (0.559/Pr)**(9/16))**(8/27))**2
        Nu_sph = 2 + 0.589 * R_ad**(1/4) / (1 + (0.469/Pr)**(9/16))**(4/9)

        h_cyl = Nu_cyl * k_air / D
        h_sph = Nu_sph * k_air / D

        A_cyl = np.pi * D * L
        A_sph = 4 * np.pi * r_outer**2

        outputs['heat_into_walls'] = (T_inf - T_surf) * (h_cyl * A_cyl + h_sph * A_sph)
    
    def compute_partials(self, inputs, J):
        # Unpack variables for easier use
        r_inner = inputs['radius']
        L = inputs['length']
        t_liner = self.options['liner_thickness']
        t_com = inputs['composite_thickness']
        t_ins = inputs['insulation_thickness']
        T_surf = inputs['T_surface']
        T_inf = inputs['T_inf']
        k_air = self.options['air_cond']
