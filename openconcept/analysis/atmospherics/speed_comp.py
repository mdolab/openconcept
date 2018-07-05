from __future__ import division
import numpy as np

from openmdao.api import ExplicitComponent


class SpeedComp(ExplicitComponent):
    '''
    This component computes airspeed from Mach number and speed of sound.

    Adapted from:
    J.P. Jasa, J.T. Hwang, and J.R.R.A. Martins: Design and Trajectory Optimization of a Morphing Wing Aircraft
    2018 AIAA/ASCE/AHS/ASC Structures, Structural Dynamics, and Materials Conference; AIAA SciTech Forum, January 2018
    '''


    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        num_points = self.options['num_nodes']

        self.add_input('M', shape=num_points)
        self.add_input('a_1e2_ms', shape=num_points)
        self.add_output('v_1e2_ms', shape=num_points)
        self.add_output('v_m_s', shape=num_points)

        arange = np.arange(num_points)
        self.declare_partials('v_1e2_ms', 'a_1e2_ms', rows=arange, cols=arange)
        self.declare_partials('v_1e2_ms', 'M', rows=arange, cols=arange)
        self.declare_partials('v_m_s', 'a_1e2_ms', rows=arange, cols=arange)
        self.declare_partials('v_m_s', 'M', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        outputs['v_1e2_ms'] = inputs['a_1e2_ms'] * inputs['M']
        outputs['v_m_s'] = inputs['a_1e2_ms'] * inputs['M'] * 1e2

    def compute_partials(self, inputs, partials):
        partials['v_1e2_ms', 'a_1e2_ms'] = inputs['M']
        partials['v_1e2_ms', 'M'] = inputs['a_1e2_ms']

        partials['v_m_s', 'a_1e2_ms'] = inputs['M'] * 1e2
        partials['v_m_s', 'M'] = inputs['a_1e2_ms'] * 1e2
