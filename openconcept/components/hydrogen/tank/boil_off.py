from __future__ import division
import numpy as np
import openmdao.api as om

class SimpleBoilOff(om.ExplicitComponent):
    """
    Simplest possible model for boil-off. Boil-off
    mass flow rate equals Q/h where Q is heat entering
    liquid and h is latent heat of vaporization.

    Inputs
    ------
    heat_into_liquid : float
        Heat entering liquid propellant (vector, W)
    
    Outputs
    -------
    m_boil_off : float
        Mass flow rate of boil-off (vector, kg/s)
    
    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    h_vap : float
        Latent heat of vaporization of propellant, default hydrogen 446592 J/kg (scalar, J/kg)
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of design points to run')
        self.options.declare('h_vap', default=446592., desc='Latent heat of vaporization (J/kg)')
    
    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('heat_into_liquid', val=100., units='W', shape=(nn,))
        self.add_output('m_boil_off', val=0.1, units='kg/s', shape=(nn,))
        self.declare_partials('m_boil_off', 'heat_into_liquid', val=np.ones(nn)/self.options['h_vap'],
                              rows=np.arange(nn), cols=np.zeros(nn))
    
    def compute(self, inputs, outputs):
        outputs['m_boil_off'] = inputs['heat_into_liquid'] / self.options['h_vap']