from __future__ import division
import numpy as np
from openconcept.components.hydrogen.tank.GH2_reservoir import GH2ReservoirODE
import openmdao.api as om

from openconcept.utilities.math.add_subtract_comp import AddSubtractComp
from openconcept.utilities.math.integrals import Integrator
from openconcept.components.hydrogen.tank.structural import CompositeOverwrap, \
                                                            COPVInsulationWeight, \
                                                            COPVLinerWeight
from openconcept.components.hydrogen.tank.thermal import HeatTransfer, FillLevelCalc
from openconcept.components.hydrogen.tank.boil_off import SimpleBoilOff
from openconcept.components.hydrogen.tank.GH2_reservoir import GH2Reservoir

class LH2Tank(om.Group):
    """
    Model of a liquid hydrogen storage tank that is
    cylindrical with hemispherical end caps. It uses a thin
    metallic liner on the inside, covered in carbon fiber for
    structural stiffness, followed by a layer of foam insulation
    to slow down boil-off. This model captures the boil off vapor
    by containing it in a tank and tracking its properties, venting
    as necessary to remain under design pressure.

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
    insulation_thickness : float
        Thickness of the insulation layer (scalar, m)
    T_inf : float
        Temperature of the "freestream" (but stationary) air in
        which the tank is sitting (vector, K)
    m_dot : float
        Mass flow usage rate of gaseous hydrogen,
        i.e. for propulsion, etc.; positive m_dot indicates
        hydrogen LEAVING the tank (vector, kg/s)
    
    Outputs
    -------
    W_LH2 : float
        Mass of remaining liquid hydrogen (vector, kg)
    weight : float
        Total weight of the tank and its contents (vector, kg)
    
    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    safety_factor : float
        Safety factor of composite overwrap, default 3. (scalar, dimensionless)
    init_fill_level : float
        Initial fill level (in range 0-1) of the tank, default 0.95
        to leave space for gas expansion; this should never be higher
        than 0.99 or so since it's not enough space for gas expansion and
        the model behaves poorly with values very close to 1 (scalar, dimensionless)
    T_surf_guess : float
        If convergence problems, set this parameter to a few degrees below the
        lowest expected T_inf value to give the solver a good initial guess!
        If no convergence problems, no need to touch this (it won't affect
        the solution).
        Guess for surface temperature of tank, default 150 K (scalar, K)
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of design points to run')
        self.options.declare('safety_factor', default=3., desc='Safety factor on composite thickness')
        self.options.declare('init_fill_level', default=0.95, desc='Initial fill level')
        self.options.declare('T_surf_guess', default=150., desc='Guess for tank surface temperature (K)')
    
    def setup(self):
        nn = self.options['num_nodes']

        # Size the structure of the tank and compute weights
        self.add_subsystem('composite', CompositeOverwrap(safety_factor=self.options['safety_factor']),
                           promotes_inputs=['radius', 'length', 'design_pressure'])
        self.add_subsystem('insulation', COPVInsulationWeight(),
                           promotes_inputs=['radius', 'length', ('thickness', 'insulation_thickness')])
        self.add_subsystem('liner', COPVLinerWeight(), promotes_inputs=['radius', 'length'])

        # Model heat entering tank
        self.add_subsystem('heat', HeatTransfer(num_nodes=nn, T_surf_guess=self.options['T_surf_guess']),
                           promotes_inputs=['radius', 'length', 'T_inf', 'insulation_thickness'])
        self.connect('composite.thickness', 'heat.composite_thickness')
        
        # Assume liquid hydrogen is stored at saturation temperature (boiling point)
        self.set_input_defaults('heat.T_liquid', val=20.28*np.ones(nn), units='K')

        # Boil-off model
        self.add_subsystem('boil_off', SimpleBoilOff(num_nodes=nn))
        self.connect('heat.heat_into_liquid', 'boil_off.heat_into_liquid')

        # Ullage property tracking
        self.add_subsystem('ullage', GH2Reservoir(num_nodes=nn, vector_V=True))
        self.connect('boil_off.m_boil_off', 'ullage.m_dot_in')
        self.connect('heat.heat_into_vapor', 'ullage.Q_dot')

        self.add_subsystem('mass_flow', AddSubtractComp(output_name='m_dot_total',
                                                        input_names=['m_dot_usage', 'm_dot_vent'],
                                                        units='kg/s', vec_size=[nn, nn],
                                                        scaling_factors=[1, 1]),
                           promotes_inputs=[('m_dot_usage', 'm_dot')])
        self.connect('mass_flow.m_dot_total', 'ullage.m_dot_out')

        self.add_subsystem('ullage_volume', om.ExecComp('V = 4/3*pi*r**3 + pi*r**2*L - W_LH2/rho',
                                                        V={'units': 'm**3', 'shape': (nn,)},
                                                        r={'units': 'm'},
                                                        L={'units': 'm'},
                                                        W_LH2={'units': 'kg', 'shape': (nn,)},
                                                        rho={'units': 'kg/m**3', 'value': 70.85}),  # density of LH2
                           promotes_inputs=[('r', 'radius'), ('L', 'length'), 'W_LH2'])
        self.add_subsystem('ullage_V_dot', om.ExecComp('V_dot = m_dot_boil_off / rho',
                                                       V_dot={'units': 'm**3/s', 'shape': (nn,)},
                                                       m_dot_boil_off={'units': 'kg/s', 'shape': (nn,)},
                                                       rho={'units': 'kg/m**3', 'value': 70.85}))  # density of LH2
        self.connect('ullage_volume.V', 'ullage.V')
        self.connect('boil_off.m_boil_off', 'ullage_V_dot.m_dot_boil_off')
        self.connect('ullage_V_dot.V_dot', 'ullage.V_dot')

        # Integrate total hydrogen usage
        integ = self.add_subsystem('LH2_mass_integrator', Integrator(num_nodes=nn,
                                    diff_units='s', time_setup='duration'))
        integ.add_integrand('LH2_boil_off', rate_name='LH2_flow', units='kg')
        self.connect('boil_off.m_boil_off', 'LH2_mass_integrator.LH2_flow')

        # Total LH2 weight and fill level
        self.add_subsystem('LH2_init', om.ExecComp('W_init = (4/3*pi*r**3 + pi*r**2*L)*fill_init*rho',
                                                   W_init={'units': 'kg'},
                                                   r={'units': 'm'},
                                                   L={'units': 'm'},
                                                   fill_init={'value': self.options['init_fill_level']},
                                                   rho={'units': 'kg/m**3', 'value': 70.85}),  # density of LH2
                           promotes_inputs=[('r', 'radius'), ('L', 'length')])
        self.add_subsystem('LH2_weight', AddSubtractComp(output_name='weight',
                                                         input_names=['W_LH2_init', 'W_LH2_boil_off'],
                                                          units='kg', vec_size=[1, nn],
                                                          scaling_factors=[1, -1]),
                           promotes_outputs=[('weight', 'W_LH2')])
        self.add_subsystem('level_calc', FillLevelCalc(num_nodes=nn),
                           promotes_inputs=['radius', 'length'])
        self.connect('LH2_init.W_init', 'LH2_weight.W_LH2_init')
        self.connect('LH2_mass_integrator.LH2_used', 'LH2_weight.W_LH2_used')
        self.connect('W_LH2', 'level_calc.W_liquid')
        self.connect('level_calc.fill_level', 'heat.fill_level')

        # Add weights
        self.add_subsystem('tank_weight', AddSubtractComp(output_name='weight',
                                                          input_names=['W_composite', 'W_insulation', \
                                                                       'W_liner', 'W_LH2'],
                                                          units='kg', vec_size=[1, 1, 1, nn],
                                                          scaling_factors=[1, 1, 1, 1]),
                           promotes_outputs=['weight'])
        self.connect('composite.weight', 'tank_weight.W_composite')
        self.connect('insulation.weight', 'tank_weight.W_insulation')
        self.connect('liner.weight', 'tank_weight.W_liner')
        self.connect('W_LH2', 'tank_weight.W_LH2')

        # Set defaults for common promoted names
        self.set_input_defaults('radius', 2., units='m')
        self.set_input_defaults('length', .5, units='m')
        self.set_input_defaults('insulation_thickness', 5., units='inch')
        self.set_input_defaults('mass_flow.m_dot_vent', 0.*np.ones(nn), units='kg/s')
