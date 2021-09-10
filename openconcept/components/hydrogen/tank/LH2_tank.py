from __future__ import division
import numpy as np
import openmdao.api as om

from openconcept.utilities.math.add_subtract_comp import AddSubtractComp
from openconcept.utilities.math.integrals import Integrator
from openconcept.utilities.linearinterp import LinearInterpolator
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
    by containing it in a tank and tracking its properties, with
    venting capabilities to remain under design pressure.

    NOTE: For the best result, the venting mass flow rate control
          points must be set by an optimizer to keep the pressure
          in the ullage under the design pressure. Additionally,
          heat may need to be added to the tank to maintain sufficient
          gaseous hydrogen. To do this, drive the model with a optimizer
          and include something like this:

            prob.model.add_design_var('m_dot_vent_start', lower=0.)
            prob.model.add_design_var('m_dot_vent_end', lower=0.)
            prob.model.add_design_var('LH2_heat_added_start', lower=0.)
            prob.model.add_design_var('LH2_heat_added_end', lower=0.)
            prob.model.add_constraint('ullage_P_residual', lower=0., scaler=1e-7)
            prob.model.add_constraint('W_GH2', lower=0.)


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
    m_dot_gas : float
        Mass flow usage rate of gaseous hydrogen,
        i.e. for propulsion, etc.; positive m_dot_gas indicates
        hydrogen LEAVING the tank (vector, kg/s)
    m_dot_liq : float
        Mass flow usage rate of liquid hydrogen,
        i.e. for propulsion, etc.; positive m_dot_liq indicates
        hydrogen LEAVING the tank (vector, kg/s)
    m_dot_vent_start : float
        Initial GH2 venting mass flow rate (scalar, kg/s)
        See note in description above!
    m_dot_vent_end : float
        Final GH2 venting mass flow rate (scalar, kg/s)
        See note in description above!
    LH2_heat_added_start : float
        Initial extra heat added to LH2 to boil into GH2 (vector, W)
        See note in description above!
    LH2_heat_added_start : float
        Final extra heat added to LH2 to boil into GH2 (vector, W)
        See note in description above!
    
    Outputs
    -------
    W_LH2 : float
        Mass of remaining liquid hydrogen (vector, kg)
    W_GH2 : float
        Mass of remaining gaseous hydrogen (vector, kg)
    weight : float
        Total weight of the tank and its contents (vector, kg)
    ullage_P_residual : float
        Ullage pressure residual: design_pressure - ullage_pressure;
        Should always be positive! See note above for how to 
        use an optimizer to enforce this (vector, Pa)
    
    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    safety_factor : float
        Safety factor of composite overwrap, default 3. (scalar, dimensionless)
    init_fill_level : float
        Initial fill level (in range 0-1) of the tank, default 0.97
        to leave space for boil off gas; 3% adopted from Cryoplane study (scalar, dimensionless)
    T_surf_guess : float
        If convergence problems, set this parameter to a few degrees below the
        lowest expected T_inf value to give the solver a good initial guess!
        If no convergence problems, no need to touch this (it won't affect
        the solution).
        Guess for surface temperature of tank, default 150 K (scalar, K)
    rho_LH2 : float
        Density of liquid hydrogen, default 70.85 kg/m^3 at boiling point and 1 atm (scalar, kg/m^3)
    ullage_T_init : float
        Initial temperature of gas in ullage, default 90 K (scalar, K)
    ullage_P_init : float
        Initial pressure of gas in ullage, default 1.2 atm; ullage pressure must be higher than ambient
        to prevent air leaking in and creating a combustible mixture (scalar, Pa)
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of design points to run')
        self.options.declare('safety_factor', default=3., desc='Safety factor on composite thickness')
        self.options.declare('init_fill_level', default=0.97, desc='Initial fill level')
        self.options.declare('T_surf_guess', default=150., desc='Guess for tank surface temperature (K)')
        self.options.declare('rho_LH2', default=70.85, desc='Liquid hydrogen density (kg/m^3)')
        self.options.declare('ullage_T_init', default=90, desc='Initial ullage temp (K)')
        self.options.declare('ullage_P_init', default=101325*1.2, desc='Initial ullage pressure (Pa)')
    
    def setup(self):
        nn = self.options['num_nodes']

        # Size the structure of the tank and compute weights
        self.add_subsystem('composite', CompositeOverwrap(safety_factor=self.options['safety_factor']),
                           promotes_inputs=['radius', 'length', 'design_pressure'])
        self.add_subsystem('insulation', COPVInsulationWeight(),
                           promotes_inputs=['radius', 'length', ('thickness', 'insulation_thickness')])
        self.add_subsystem('liner', COPVLinerWeight(), promotes_inputs=['radius', 'length'])

        # Compute the LH2 weight and how much of the tank it fills up (fill level)
        self.add_subsystem('LH2_init', om.ExecComp('W = (4/3*pi*r**3 + pi*r**2*L)*fill_init*rho',
                                                   W={'units': 'kg'},
                                                   r={'units': 'm'},
                                                   L={'units': 'm'},
                                                   fill_init={'val': self.options['init_fill_level']},
                                                   rho={'units': 'kg/m**3', 'val': self.options['rho_LH2']}),
                           promotes_inputs=[('r', 'radius'), ('L', 'length')])
        self.add_subsystem('LH2_weight', AddSubtractComp(output_name='weight',
                                                         input_names=['W_LH2_init', 'W_LH2_boil_off', 'W_LH2_used'],
                                                          units='kg', vec_size=[1, nn, nn],
                                                          scaling_factors=[1, -1, -1]),
                           promotes_outputs=[('weight', 'W_LH2')])
        self.add_subsystem('level_calc', FillLevelCalc(num_nodes=nn),
                           promotes_inputs=['radius', 'length'])
        self.connect('LH2_init.W', 'LH2_weight.W_LH2_init')
        self.connect('W_LH2', 'level_calc.W_liquid')

        # Model heat entering tank
        self.add_subsystem('heat', HeatTransfer(num_nodes=nn, T_surf_guess=self.options['T_surf_guess']),
                           promotes_inputs=['radius', 'length', 'T_inf', 'insulation_thickness'])
        self.connect('composite.thickness', 'heat.composite_thickness')
        self.connect('level_calc.fill_level', 'heat.fill_level')
        
        # Assume liquid hydrogen is stored at saturation temperature (boiling point)
        self.set_input_defaults('heat.T_liquid', val=20.28*np.ones(nn), units='K')

        # Boil-off model (boiling liquid goes into ullage)
        self.add_subsystem('boil_off', SimpleBoilOff(num_nodes=nn))
        self.connect('heat.heat_into_liquid', 'boil_off.heat_into_liquid')

        # Linear interpolators for venting and heat addition
        self.add_subsystem('vent_interp', LinearInterpolator(units='kg/s', num_nodes=nn),
                           promotes_inputs=[('start_val', 'm_dot_vent_start'),
                                            ('end_val', 'm_dot_vent_end')])
        self.add_subsystem('heat_add_interp', LinearInterpolator(units='W', num_nodes=nn),
                           promotes_inputs=[('start_val', 'LH2_heat_added_start'),
                                            ('end_val', 'LH2_heat_added_end')])
        self.connect('heat_add_interp.vec', 'boil_off.LH2_heat_added')
        self.set_input_defaults('m_dot_vent_start', 0., units='kg/s')
        self.set_input_defaults('m_dot_vent_end', 0., units='kg/s')
        self.set_input_defaults('LH2_heat_added_start', 0., units='W')
        self.set_input_defaults('LH2_heat_added_end', 0., units='W')

        # Integrate total gaseous and liquid hydrogen flows
        integ = self.add_subsystem('mass_integ', Integrator(num_nodes=nn,
                                   diff_units='s', time_setup='duration'),
                                   promotes_inputs=[('GH2_use_rate', 'm_dot_gas'), ('LH2_use_rate', 'm_dot_liq'),
                                                    'duration'])
        integ.add_integrand('LH2_boil_off', rate_name='LH2_flow', units='kg', lower=0.)
        integ.add_integrand('GH2_used', rate_name='GH2_use_rate', units='kg', lower=0.)
        integ.add_integrand('LH2_used', rate_name='LH2_use_rate', units='kg', lower=0.)
        integ.add_integrand('GH2_vent', rate_name='GH2_vent_rate', units='kg', lower=0.)
        self.connect('boil_off.m_boil_off', 'mass_integ.LH2_flow')
        self.connect('mass_integ.LH2_boil_off', 'LH2_weight.W_LH2_boil_off')
        self.connect('vent_interp.vec', 'mass_integ.GH2_vent_rate')
        self.connect('mass_integ.LH2_used', 'LH2_weight.W_LH2_used')

        # Sum output mass flows from ullage
        self.add_subsystem('mass_flow', AddSubtractComp(output_name='m_dot_total',
                                                        input_names=['m_dot_usage', 'm_dot_vent'],
                                                        units='kg/s', vec_size=[nn, nn],
                                                        scaling_factors=[1, 1]),
                           promotes_inputs=[('m_dot_usage', 'm_dot_gas')])
        self.connect('vent_interp.vec', 'mass_flow.m_dot_vent')

        # Ullage volume and rate of change of volume
        self.add_subsystem('ullage_volume', om.ExecComp('V = 4/3*pi*r**3 + pi*r**2*L - W_LH2/rho',
                                                        V={'units': 'm**3', 'shape': (nn,)},
                                                        r={'units': 'm'},
                                                        L={'units': 'm'},
                                                        W_LH2={'units': 'kg', 'shape': (nn,)},
                                                        rho={'units': 'kg/m**3', 'val': self.options['rho_LH2']}),
                           promotes_inputs=[('r', 'radius'), ('L', 'length'), 'W_LH2'])
        self.add_subsystem('ullage_V_dot', om.ExecComp('V_dot = (m_dot_boil_off + m_dot_liq) / rho',
                                                       V_dot={'units': 'm**3/s', 'shape': (nn,)},
                                                       m_dot_boil_off={'units': 'kg/s', 'shape': (nn,)},
                                                       m_dot_liq={'units': 'kg/s', 'shape': (nn,)},
                                                       rho={'units': 'kg/m**3', 'val': self.options['rho_LH2']}),
                           promotes_inputs=['m_dot_liq'])
        self.connect('boil_off.m_boil_off', 'ullage_V_dot.m_dot_boil_off')

        # Ullage gas property tracking
        self.add_subsystem('ullage', GH2Reservoir(num_nodes=nn, T_init=self.options['ullage_T_init'],
                                                  vector_V=True),
                           promotes_inputs=['duration'])
        self.connect('mass_flow.m_dot_total', 'ullage.m_dot_out')
        self.connect('boil_off.m_boil_off', 'ullage.m_dot_in')
        self.connect('heat.heat_into_vapor', 'ullage.Q_dot')
        self.connect('ullage_volume.V', 'ullage.V')
        self.connect('ullage_V_dot.V_dot', 'ullage.V_dot')

        # GH2 weight in ullage
        self.add_subsystem('GH2_init', om.ExecComp('W = (4/3*pi*r**3 + pi*r**2*L)*(1-fill_init)*P/T/R_H2',
                                                   W={'units': 'kg'},
                                                   r={'units': 'm'},
                                                   L={'units': 'm'},
                                                   fill_init={'val': self.options['init_fill_level']},
                                                   P={'val': self.options['ullage_P_init'], 'units': 'Pa'},
                                                   T={'val': self.options['ullage_T_init'], 'units': 'K'},
                                                   R_H2={'val': 8.314/2.016e-3, 'units': 'J/(kg*K)'}),
                           promotes_inputs=[('r', 'radius'), ('L', 'length')])
        self.add_subsystem('GH2_weight', AddSubtractComp(output_name='weight',
                                                         input_names=['W_GH2_init', 'W_LH2_boil_off',
                                                                      'W_used', 'W_vent'],
                                                         units='kg', vec_size=[1, nn, nn, nn],
                                                         scaling_factors=[1, 1, -1, -1]),
                           promotes_outputs=[('weight', 'W_GH2')])
        self.connect('mass_integ.LH2_boil_off', 'GH2_weight.W_LH2_boil_off')
        self.connect('mass_integ.GH2_used', 'GH2_weight.W_used')
        self.connect('mass_integ.GH2_vent', 'GH2_weight.W_vent')
        self.connect('GH2_init.W', 'GH2_weight.W_GH2_init')
        self.connect('W_GH2', 'ullage.m')

        # Ullage pressure residual
        self.add_subsystem('ullage_constraint', AddSubtractComp(output_name='residual',
                                                          input_names=['design_pressure',
                                                                       'ullage_pressure'],
                                                          units='Pa', vec_size=[1, nn],
                                                          scaling_factors=[1, -1]),
                           promotes_inputs=['design_pressure'],
                           promotes_outputs=[('residual', 'ullage_P_residual')])
        self.connect('ullage.P', 'ullage_constraint.ullage_pressure')

        # Sum components of tank weight
        self.add_subsystem('tank_weight', AddSubtractComp(output_name='weight',
                                                          input_names=['W_composite', 'W_insulation', \
                                                                       'W_liner', 'W_LH2', 'W_GH2'],
                                                          units='kg', vec_size=[1, 1, 1, nn, nn],
                                                          scaling_factors=[1, 1, 1, 1, 1]),
                           promotes_inputs=['W_LH2', 'W_GH2'],
                           promotes_outputs=['weight'])
        self.connect('composite.weight', 'tank_weight.W_composite')
        self.connect('insulation.weight', 'tank_weight.W_insulation')
        self.connect('liner.weight', 'tank_weight.W_liner')

        # Set defaults for common promoted names
        self.set_input_defaults('radius', 2., units='m')
        self.set_input_defaults('length', .5, units='m')
        self.set_input_defaults('insulation_thickness', 5., units='inch')
        self.set_input_defaults('design_pressure', 3., units='bar')
        self.set_input_defaults('m_dot_gas', np.zeros(nn), units='kg/s')
        self.set_input_defaults('m_dot_liq', np.zeros(nn), units='kg/s')
