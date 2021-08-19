from __future__ import division
import numpy as np
import openmdao.api as om

class HeatTransfer(om.Group):
    """
    Computes the heat transfer into the hydrogen tank.

    Inputs
    ------
    radius : float
        Radius inside of tank for the cylinder and hemispherical end caps (scalar, m)
    length : float
        Length of JUST THE CYLINDRICAL part of the tank (scalar, m)
    T_liquid : float
        Temperature of the liquid propellant in the tank (vector, K)
    T_inf : float
        Temperature of the "freestream" (but stationary) air in
        which the tank is sitting (vector, K)
    composite_thickness : float
        Thickness of the composite overwrap (scalar, m)
    insulation_thickness : float
        Thickness of tank insulation (scalar, m)
    fill_level : float
        Fraction of tank (in range 0-1) filled with liquid propellant; assumes
        tank is oriented horizontally as shown above (vector, dimensionless)
    
    Outputs
    -------
    heat_into_liquid : float
        Heat entering the liquid propellant; positive is heat going
        INTO liquid (vector, W)
    heat_into_vapor : float
        Heat entering the vapor in the ullage; positive is heat
        going INTO vapor (vector, W)
    
    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    T_surf_guess : float
        Guess for surface temperature of tank, default 150 K (scalar, K)
        If convergence problems, set this parameter to a few degrees below the
        lowest expected T_inf value to give the solver a good initial guess!
        If no convergence problems, no need to touch this (it won't affect
        the solution).
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of design points to run')
        self.options.declare('T_surf_guess', default=150., desc='Guess for tank surface temperature (K)')
    
    def setup(self):
        nn = self.options['num_nodes']

        # Model heat entering tank
        self.add_subsystem('calc_resist', COPVThermalResistance(),
                           promotes_inputs=['radius', 'length', 'composite_thickness',
                                            'insulation_thickness'])
        self.add_subsystem('Q_wall', COPVHeatFromEnvironmentIntoTankWalls(num_nodes=nn),
                           promotes_inputs=['radius', 'length', 'insulation_thickness', 'T_inf',
                                            'composite_thickness'])
        self.add_subsystem('Q_LH2', COPVHeatFromWallsIntoPropellant(num_nodes=nn),
                           promotes_inputs=['radius', 'length', 'fill_level', 'T_liquid'],
                           promotes_outputs=['heat_into_vapor', 'heat_into_liquid'])
        self.connect('calc_resist.thermal_resistance', 'Q_LH2.thermal_resistance')

        # Assume liquid hydrogen is stored at saturation temperature (boiling point)
        self.set_input_defaults('T_liquid', val=20.28*np.ones(nn), units='K')

        # Find the temperature of the surface of the tank so that the heat entering the surface
        # is equal to the heat entering the contents (make it satisfy steady problem)
        self.add_subsystem('calc_T_surf', om.BalanceComp('T_surface', eq_units='W', lhs_name='Q_wall', \
                                                            rhs_name='Q_contents',
                                                            val=np.ones(nn)*self.options['T_surf_guess'],
                                                            units='K'))
        self.connect('calc_T_surf.T_surface', ['Q_wall.T_surface', 'Q_LH2.T_surface'])
        self.connect('Q_wall.heat_into_walls', 'calc_T_surf.Q_wall')
        self.connect('Q_LH2.heat_total', 'calc_T_surf.Q_contents')


class FillLevelCalc(om.ExplicitComponent):
    """
    Computes the fill level in the tank given the
    weight of the liquid.

    Inputs
    ------
    W_liquid : float
        Weight of the liquid (vector, kg)
    radius : float
        Radius inside of tank for the cylinder and hemispherical end caps (scalar, m)
    length : float
        Length of JUST THE CYLINDRICAL part of the tank (scalar, m)
    density : float
        Density of the liquid, default 70.85 kg/m^3 hydrogen (vector, kg/m^3)
    
    Outputs
    -------
    fill_level : float
        Fraction of tank (in range 0-1) filled with liquid propellant; assumes
        tank is oriented horizontally as shown above (vector, dimensionless)
    
    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of design points to run')
    
    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('W_liquid', val=np.ones(nn)*1000, units='kg', shape=(nn,))
        self.add_input('radius', val=2., units='m')
        self.add_input('length', val=.5, units='m')
        self.add_input('density', val=70.85, units='kg/m**3', shape=(nn,))
        self.add_output('fill_level', val=0.5, shape=(nn,), lower=0.01, upper=0.99)

        self.declare_partials('fill_level', ['W_liquid', 'density'], rows=np.arange(nn), cols=np.arange(nn))
        self.declare_partials('fill_level', ['radius', 'length'],
                              rows=np.arange(nn), cols=np.zeros(nn))
    
    def compute(self, inputs, outputs):
        r = inputs['radius']
        L = inputs['length']
        V = inputs['W_liquid'] / inputs['density']
        V_tank = 4/3*np.pi*r**3 + np.pi*r**2*L
        outputs['fill_level'] = V / V_tank
    
    def compute_partials(self, inputs, J):
        rho = inputs['density']
        r = inputs['radius']
        L = inputs['length']
        V = inputs['W_liquid'] / rho
        V_tank = 4/3*np.pi*r**3 + np.pi*r**2*L

        J['fill_level', 'W_liquid'] = (rho * V_tank)**(-1)
        J['fill_level', 'density'] = -inputs['W_liquid'] / (rho * V_tank)**2 * V_tank
        J['fill_level', 'radius'] = -V / V_tank**2 * (4*np.pi*r**2 + 2*np.pi*r*L)
        J['fill_level', 'length'] = -V / V_tank**2 * np.pi*r**2


class COPVThermalResistance(om.ExplicitComponent):
    """
    Computes thermal resistance for heat going through walls
    of composite overwrap pressure vessel. Steady heat flow
    analysis (assumes thermal mass of tank is negligible compared
    to contents). This resistance does not include the transfer of heat
    to and from the walls due to the convection of fluids on either side.
    In other words, it only includes the thermal resistance from conduction
    through the three layers of the wall in the tank. It also ignores the
    fairing/skin on the outside of the tank to protect the insulation.

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
        Length of JUST THE CYLINDRICAL part of the tank (scalar, m)
    composite_thickness : float
        Thickness of the composite overwrap (scalar, m)
    insulation_thickness : float
        Thickness of tank insulation (scalar, m)
    
    Outputs
    -------
    thermal_resistance : float
        Effective thermal resistance of the tank (scalar, K/W)
    
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
        self.add_input('radius', val=0.5, units='m')
        self.add_input('length', val=2., units='m')
        self.add_input('composite_thickness', val=0.05, units='m')
        self.add_input('insulation_thickness', val=0.1, units='m')

        self.add_output('thermal_resistance', lower=0., units='K/W')

        self.declare_partials('thermal_resistance', ['radius', 'length',
                                             'composite_thickness',
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
        R_cyl = R_liner + R_com + R_ins

        # Thermal resistance of spherical portion (two end caps)
        R_liner = (1/r_inner - 1/r_com_liner) / (4*np.pi*k_liner)
        R_com = (1/r_com_liner - 1/r_ins_com) / (4*np.pi*k_com)
        R_ins = (1/r_ins_com - 1/r_outer) / (4*np.pi*k_ins)
        R_sph = R_liner + R_com + R_ins

        outputs['thermal_resistance'] = 1 / (1/R_cyl + 1/R_sph)

    
    def compute_partials(self, inputs, J):
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
        R_cyl = R_liner + R_com + R_ins

        # Thermal resistance of spherical portion (two end caps)
        R_liner = (1/r_inner - 1/r_com_liner) / (4*np.pi*k_liner)
        R_com = (1/r_com_liner - 1/r_ins_com) / (4*np.pi*k_com)
        R_ins = (1/r_ins_com - 1/r_outer) / (4*np.pi*k_ins)
        R_sph = R_liner + R_com + R_ins

        d_R_cyl_d_r = (-t_liner/r_inner**2) / (r_com_liner/r_inner) / (2*np.pi*L*k_liner) + \
                      (-t_com/r_com_liner**2) / (r_ins_com/r_com_liner) / (2*np.pi*L*k_com) + \
                      (-t_ins/r_ins_com**2) / (r_outer/r_ins_com) / (2*np.pi*L*k_ins)
        d_R_cyl_d_L = -np.log(r_com_liner/r_inner) / (2*np.pi*L**2*k_liner) - \
                       np.log(r_ins_com/r_com_liner) / (2*np.pi*L**2*k_com) - \
                       np.log(r_outer/r_ins_com) / (2*np.pi*L**2*k_ins)
        d_R_cyl_d_t_com = (1/r_com_liner) / (r_ins_com/r_com_liner) / (2*np.pi*L*k_com) + \
                          (-t_ins/r_ins_com**2) / (r_outer/r_ins_com) / (2*np.pi*L*k_ins)
        d_R_cyl_d_t_ins = (1/r_ins_com) / (r_outer/r_ins_com) / (2*np.pi*L*k_ins)

        d_R_sph_d_r = (-1/r_inner**2 + 1/r_com_liner**2) / (4*np.pi*k_liner) + \
                      (-1/r_com_liner**2 + 1/r_ins_com**2) / (4*np.pi*k_com) + \
                      (-1/r_ins_com**2 + 1/r_outer**2) / (4*np.pi*k_ins)
        d_R_sph_d_t_com = 1/r_ins_com**2 / (4*np.pi*k_com) + \
                          (-1/r_ins_com**2 + 1/r_outer**2) / (4*np.pi*k_ins)
        d_R_sph_d_t_ins = 1/r_outer**2 / (4*np.pi*k_ins)

        J['thermal_resistance', 'radius'] = (R_cyl**(-1) + R_sph**(-1))**(-2) * \
                                            (R_cyl**(-2)*d_R_cyl_d_r + R_sph**(-2)*d_R_sph_d_r)
        J['thermal_resistance', 'length'] = (R_cyl**(-1) + R_sph**(-1))**(-2) * (R_cyl**(-2)*d_R_cyl_d_L)
        J['thermal_resistance', 'composite_thickness'] = (R_cyl**(-1) + R_sph**(-1))**(-2) * \
                                                         (R_cyl**(-2)*d_R_cyl_d_t_com + R_sph**(-2)*d_R_sph_d_t_com)
        J['thermal_resistance', 'insulation_thickness'] = (R_cyl**(-1) + R_sph**(-1))**(-2) * \
                                                          (R_cyl**(-2)*d_R_cyl_d_t_ins + R_sph**(-2)*d_R_sph_d_t_ins)


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
        Length of JUST THE CYLINDRICAL part of the tank (scalar, m)
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
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    air_cond : float
        Thermal conductivity of air surrounding tank, default 0.0245 W/(m-K) (scalar, W/(m-K))
        estimated from https://www.engineeringtoolbox.com/air-properties-viscosity-conductivity-heat-capacity-d_1509.html
    liner_thickness : float
        Thickness of liner, default 0.5 mm (scalar, m)
    surface_emissivity : float
        Surface emissivity of outside layer of tank; rough guess for thermal emissivity of
        foam-like material or plastic is probably 0.6-0.9, default 0.6 (scalar, dimensionless)
        https://www.thermoworks.com/emissivity-table
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of design points to run')
        self.options.declare('air_cond', default=0.0245, desc='Thermal conductivity of air W/(m-K)')
        self.options.declare('liner_thickness', default=0.5e-3, desc='Liner thickness (m)')
        self.options.declare('surface_emissivity', default=0.6, desc='Thermal radiation emissivity of tank surface')

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

        # self.declare_partials(['*'], ['*'], method='cs')
    
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
        eps_surface = self.options['surface_emissivity']

        # Compute outer radius
        r_outer = r_inner + t_liner + t_com + t_ins
        D = 2*r_outer  # diameter of tank

        # Rayleigh and Prandtl numbers
        alpha = -3.119e-6 + 3.541e-8*T_inf + 1.679e-10*T_inf**2  # air diffusivity
        nu = -2.079e-6 + 2.777e-8*T_inf + 1.077e-10*T_inf**2  # air viscosity
        Pr = nu/alpha
        R_ad = 9.807 * (T_inf - T_surf) / T_inf * D**3 / (nu * alpha)
        
        # Take absolute value of physical (positive) constants
        # in a way that plays well with complex step for derivative check
        Pr[np.where(np.real(Pr) < 0)] = -Pr[np.where(np.real(Pr) < 0)]
        R_ad[np.where(np.real(R_ad) < 0)] = -R_ad[np.where(np.real(R_ad) < 0)]

        # Nusselt numbers for cylinder and sphere
        Nu_cyl = (0.6 + 0.387 * R_ad**(1/6) / (1 + (0.559/Pr)**(9/16))**(8/27))**2
        Nu_sph = 2 + 0.589 * R_ad**(1/4) / (1 + (0.469/Pr)**(9/16))**(4/9)

        h_cyl = Nu_cyl * k_air / D
        h_sph = Nu_sph * k_air / D

        A_cyl = np.pi * D * L
        A_sph = 4 * np.pi * r_outer**2

        Q_convection = (T_inf - T_surf) * (h_cyl * A_cyl + h_sph * A_sph)

        # Radiation effects
        sig = 5.67e-8  # Stefan-Boltzmann constant (W/(m^2 K^4))
        Q_radiation = sig*eps_surface*(A_cyl + A_sph)*(T_inf**4 - T_surf**4)

        outputs['heat_into_walls'] = Q_convection + Q_radiation
    
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
        eps_surface = self.options['surface_emissivity']
        sig = 5.67e-8  # Stefan-Boltzmann constant (W/(m^2 K^4))

        # Compute outer radius
        r_outer = r_inner + t_liner + t_com + t_ins
        D = 2*r_outer  # diameter of tank

        # Rayleigh and Prandtl numbers
        alpha = -3.119e-6 + 3.541e-8*T_inf + 1.679e-10*T_inf**2  # air diffusivity
        nu = -2.079e-6 + 2.777e-8*T_inf + 1.077e-10*T_inf**2  # air viscosity
        Pr = nu/alpha
        R_ad = 9.807 * (T_inf - T_surf) / T_inf * D**3 / (nu * alpha)
        
        # Take absolute value of physical (positive) constants
        # in a way that plays well with complex step for derivative check
        Pr_flip = np.where(np.real(Pr) < 0)  # indicies to multiply Pr by -1
        R_ad_flip = np.where(np.real(R_ad) < 0)   # indicies to multiply R_ad by -1
        Pr[Pr_flip] = -Pr[Pr_flip]
        R_ad[R_ad_flip] = -R_ad[R_ad_flip]

        # Nusselt numbers for cylinder and sphere
        Nu_cyl_sqrt = 0.6 + 0.387 * R_ad**(1/6) / (1 + (0.559/Pr)**(9/16))**(8/27)
        Nu_cyl = Nu_cyl_sqrt**2
        Nu_sph = 2 + 0.589 * R_ad**(1/4) / (1 + (0.469/Pr)**(9/16))**(4/9)

        h_cyl = Nu_cyl * k_air / D
        h_sph = Nu_sph * k_air / D

        A_cyl = np.pi * D * L
        A_sph = 4 * np.pi * r_outer**2

        # Use reverse-AD style approach
        d_Q_rad = 1
        d_Q_conv = 1
        d_A_sph = d_Q_conv * (T_inf - T_surf) * h_sph + d_Q_rad * sig*eps_surface*(T_inf**4 - T_surf**4)
        d_A_cyl = d_Q_conv * (T_inf - T_surf) * h_cyl + d_Q_rad * sig*eps_surface*(T_inf**4 - T_surf**4)
        d_h_sph = d_Q_conv * (T_inf - T_surf) * A_sph
        d_h_cyl = d_Q_conv * (T_inf - T_surf) * A_cyl
        d_Nu_sph = d_h_sph * k_air / D
        d_Nu_cyl = d_h_cyl * k_air / D
        d_R_ad = d_Nu_sph * 0.589 * 1/4 * R_ad**(-3/4) / (1 + (0.469/Pr)**(9/16))**(4/9) + \
                 d_Nu_cyl * 2*Nu_cyl_sqrt * 0.387 * 1/6 * R_ad**(-5/6) / (1 + (0.559/Pr)**(9/16))**(8/27)
        d_Pr = d_Nu_cyl * 2*Nu_cyl_sqrt * 0.387 * R_ad**(1/6) * (-8/27) * (1 + (Pr/0.559)**(-9/16))**(-35/27) * \
                                          (-9/16)*(Pr/0.559)**(-25/16) / 0.559 + \
               d_Nu_sph * (-4/9) * 0.589 * R_ad**(1/4) * (1 + (Pr/0.469)**(-9/16))**(-13/9) * \
                          (-9/16) * (Pr/0.469)**(-25/16) / 0.469

        # Some were flipped and derivatives need to reflect that
        d_alpha_R_ad_part = (-9.807) * (T_inf - T_surf) / T_inf * D**3 / (nu * alpha**2)
        d_alpha_R_ad_part[R_ad_flip] = -d_alpha_R_ad_part[R_ad_flip]
        d_alpha_Pr_part = (-nu) / alpha**2
        d_alpha_Pr_part[Pr_flip] = -d_alpha_Pr_part[Pr_flip]
        d_alpha = d_R_ad * d_alpha_R_ad_part + d_Pr * d_alpha_Pr_part

        d_nu_R_ad_part = (-9.807) * (T_inf - T_surf) / T_inf * D**3 / (nu**2 * alpha)
        d_nu_R_ad_part[R_ad_flip] = -d_nu_R_ad_part[R_ad_flip]
        d_nu_Pr_part = 1 / alpha
        d_nu_Pr_part[Pr_flip] = -d_nu_Pr_part[Pr_flip]
        d_nu = d_R_ad * d_nu_R_ad_part + d_Pr * d_nu_Pr_part

        d_D_R_ad_part = 9.807 * (T_inf - T_surf) / T_inf * 3*D**2 / (nu * alpha)
        d_D_R_ad_part[R_ad_flip] = -d_D_R_ad_part[R_ad_flip]
        d_D = d_R_ad * d_D_R_ad_part + d_A_cyl * np.pi * L - \
              d_h_cyl * Nu_cyl * k_air / D**2 - d_h_sph * Nu_sph * k_air / D**2

        d_r_outer = d_A_sph * 8 * np.pi * r_outer + d_D * 2

        d_T_inf_R_ad_part = 9.807 * T_surf / T_inf**2 * D**3 / (nu * alpha)
        d_T_inf_R_ad_part[R_ad_flip] = -d_T_inf_R_ad_part[R_ad_flip]

        d_T_surf_R_ad_part = 9.807 / T_inf * D**3 / (nu * alpha)
        d_T_surf_R_ad_part[R_ad_flip] = -d_T_surf_R_ad_part[R_ad_flip]

        J['heat_into_walls', 'T_inf'] = d_Q_rad * sig*eps_surface*(A_cyl + A_sph)*4*T_inf**3 + \
                                        d_Q_conv * (h_cyl * A_cyl + h_sph * A_sph) + \
                                        d_R_ad * d_T_inf_R_ad_part + \
                                        d_alpha * (3.541e-8 + 2*1.679e-10*T_inf) + \
                                        d_nu * (2.777e-8 + 2*1.077e-10*T_inf)
        J['heat_into_walls', 'T_surface'] = d_Q_rad * sig*eps_surface*(A_cyl + A_sph)*(-4)*T_surf**3 - \
                                            d_Q_conv * (h_cyl * A_cyl + h_sph * A_sph) - \
                                            d_R_ad * d_T_surf_R_ad_part
        J['heat_into_walls', 'radius'] = d_r_outer
        J['heat_into_walls', 'length'] = d_A_cyl * np.pi * D
        J['heat_into_walls', 'composite_thickness'] = d_r_outer
        J['heat_into_walls', 'insulation_thickness'] = d_r_outer


class COPVHeatFromWallsIntoPropellant(om.ExplicitComponent):
    """
    Computes the amount of heat entering the propellant from
    the tank walls. Calculates the heat transfer to just the
    liquid assuming the entire inside wall is at the temperature
    of the liquid and then multiply by the fraction of the
    internal surface area covered by liquid. It then
    uses a rough curve fit based on data from
    https://arc.aiaa.org/doi/10.2514/6.1992-818 to estimate
    the heat to the vapor for the given fill level.

    Note that this approach only works when there is some liquid
    in the tank (if there is no liquid, it will vastly overestimate
    the amount of heat entering the tank).

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
    T_surface : float
        Temperature of the tank's outer surface (vector, K)
    T_liquid : float
        Temperature of the liquid propellant in the tank (vector, K)
    fill_level : float
        Fraction of tank (in range 0-1) filled with liquid propellant; assumes
        tank is oriented horizontally as shown above (vector, dimensionless)
    radius : float
        Radius inside of tank for the cylinder and hemispherical end caps (scalar, m)
    length : float
        Length of JUST THE CYLINDRICAL part of the tank (scalar, m)
    thermal_resistance : float
        Thermal resistance of the tank walls (scalar, K/W)
    
    Outputs
    -------
    heat_into_liquid : float
        Heat entering the liquid propellant; positive is heat going
        INTO liquid (vector, W)
    heat_into_vapor : float
        Heat entering the vapor in the ullage; positive is heat
        going INTO vapor (vector, W)
    heat_total : float
        Total heat entering the contents of the tank; positive
        is heat entering vapor and liquid (vector, W)
    
    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of design points to run')
    
    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('T_surface', val=100., units='K', shape=(nn,))
        self.add_input('T_liquid', val=20., units='K', shape=(nn,))
        self.add_input('fill_level', val=0.5, shape=(nn,))
        self.add_input('radius', val=0.5, units='m')
        self.add_input('length', val=2., units='m')
        self.add_input('thermal_resistance', val=1., units='K/W')

        self.add_output('heat_into_liquid', units='W', shape=(nn,))
        self.add_output('heat_into_vapor', units='W', shape=(nn,))
        self.add_output('heat_total', units='W', shape=(nn,))

        self.declare_partials('heat_into_liquid', ['T_liquid', 'T_surface', 'fill_level'],
                              rows=np.arange(nn), cols=np.arange(nn))
        self.declare_partials('heat_into_liquid', ['radius', 'length',
                                                  'thermal_resistance'],
                              rows=np.arange(nn), cols=np.zeros(nn))
        self.declare_partials('heat_into_vapor', ['T_liquid', 'T_surface', 'fill_level'],
                              rows=np.arange(nn), cols=np.arange(nn))
        self.declare_partials('heat_into_vapor', ['radius', 'length',
                                                  'thermal_resistance'],
                              rows=np.arange(nn), cols=np.zeros(nn))
        self.declare_partials('heat_total', ['T_liquid', 'T_surface', 'fill_level'],
                              rows=np.arange(nn), cols=np.arange(nn))
        self.declare_partials('heat_total', ['radius', 'length',
                                             'thermal_resistance'],
                              rows=np.arange(nn), cols=np.zeros(nn))
    
    def compute(self, inputs, outputs):
        T_surf = inputs['T_surface']
        T_liq = inputs['T_liquid']
        R = inputs['thermal_resistance']
        r = inputs['radius']
        L = inputs['length']
        fill_level = inputs['fill_level']

        Q_if_all_liquid = (T_surf - T_liq) / R
        h_liquid = 2*r*fill_level  # volume assuming the fluid height in the tank is linear
                                   # with fill level is never off by more than 10%
        cylinder_central_angle = 2*np.arccos(1 - h_liquid/r)
        A_liquid = 2*np.pi*r*h_liquid + cylinder_central_angle*r*L  # spherical cap + sector * length
        A_total = 4*np.pi*r**2 + 2*np.pi*r*L
        Q_liquid = Q_if_all_liquid * A_liquid / A_total
        outputs['heat_into_liquid'] = Q_liquid

        heat_liquid_frac = -fill_level**2 + 2*fill_level  # rough curve fit that follows trend for
                                                          # fraction of total heating going to liquid in
                                                          # https://arc.aiaa.org/doi/10.2514/6.1992-818
        heat_vapor_frac = 1 - heat_liquid_frac
        Q_total = Q_liquid / heat_liquid_frac
        outputs['heat_total'] = Q_total
        outputs['heat_into_vapor'] = Q_total * heat_vapor_frac

    def compute_partials(self, inputs, J):
        T_surf = inputs['T_surface']
        T_liq = inputs['T_liquid']
        R = inputs['thermal_resistance']
        r = inputs['radius']
        L = inputs['length']
        fill_level = inputs['fill_level']

        Q_if_all_liquid = (T_surf - T_liq) / R
        h_liquid = 2*r*fill_level  # volume assuming the fluid height in the tank is linear
                                   # with fill level is never off by more than 10%
        cylinder_central_angle = 2*np.arccos(1 - h_liquid/r)
        d_angle_d_r = -2/np.sqrt(1 - (1 - h_liquid/r)**2) * (-2*fill_level/r + h_liquid/r**2)
        A_liquid = 2*np.pi*r*h_liquid + cylinder_central_angle*r*L  # spherical cap + sector * length
        A_total = 4*np.pi*r**2 + 2*np.pi*r*L
        Q_liquid = Q_if_all_liquid * A_liquid / A_total

        heat_liquid_frac = -fill_level**2 + 2*fill_level  # rough curve fit that follows trend for
                                                          # fraction of total heating going to liquid in
                                                          # https://arc.aiaa.org/doi/10.2514/6.1992-818
        heat_vapor_frac = 1 - heat_liquid_frac

        J['heat_into_liquid', 'T_surface'] = A_liquid / A_total / R
        J['heat_into_vapor', 'T_surface'] = heat_vapor_frac / heat_liquid_frac * J['heat_into_liquid', 'T_surface']
        J['heat_total', 'T_surface'] = J['heat_into_liquid', 'T_surface'] / heat_liquid_frac
        J['heat_into_liquid', 'T_liquid'] = -A_liquid / A_total / R
        J['heat_into_vapor', 'T_liquid'] = heat_vapor_frac / heat_liquid_frac * J['heat_into_liquid', 'T_liquid']
        J['heat_total', 'T_liquid'] = J['heat_into_liquid', 'T_liquid'] / heat_liquid_frac
        J['heat_into_liquid', 'thermal_resistance'] = (T_liq - T_surf) / R**2 * A_liquid / A_total
        J['heat_into_vapor', 'thermal_resistance'] = heat_vapor_frac / heat_liquid_frac * \
                                                     J['heat_into_liquid', 'thermal_resistance']
        J['heat_total', 'thermal_resistance'] = J['heat_into_liquid', 'thermal_resistance'] / heat_liquid_frac
        J['heat_into_liquid', 'radius'] = Q_if_all_liquid / A_total * (2*np.pi*h_liquid + 2*np.pi*r*2*fill_level + \
                                                                       cylinder_central_angle*L + r*L*d_angle_d_r) - \
                                          Q_if_all_liquid * A_liquid / A_total**2 * (8*np.pi*r + 2*np.pi*L)
        J['heat_into_vapor', 'radius'] = heat_vapor_frac / heat_liquid_frac * J['heat_into_liquid', 'radius']
        J['heat_total', 'radius'] = J['heat_into_liquid', 'radius'] / heat_liquid_frac
        J['heat_into_liquid', 'length'] = Q_if_all_liquid / A_total * cylinder_central_angle*r - \
                                          Q_if_all_liquid * A_liquid / A_total**2 * 2*np.pi*r
        J['heat_into_vapor', 'length'] = heat_vapor_frac / heat_liquid_frac * J['heat_into_liquid', 'length']
        J['heat_total', 'length'] = J['heat_into_liquid', 'length'] / heat_liquid_frac
        J['heat_into_liquid', 'fill_level'] = Q_if_all_liquid / A_total * (4*np.pi*r**2 + \
                                              r*L*2/np.sqrt(1 - (1 - h_liquid/r)**2)*2)
        J['heat_into_vapor', 'fill_level'] = J['heat_into_liquid', 'fill_level'] * heat_vapor_frac / heat_liquid_frac - \
                                             Q_liquid / heat_liquid_frac**2 * (-2*fill_level + 2)
        J['heat_total', 'fill_level'] = J['heat_into_liquid', 'fill_level'] / heat_liquid_frac - \
                                             Q_liquid / heat_liquid_frac**2 * (-2*fill_level + 2)
