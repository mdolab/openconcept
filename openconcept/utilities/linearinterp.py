from openmdao.api import ExplicitComponent
import numpy as np

class LinearInterpolator(ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes',default=2,desc="Number of nodes")
        self.options.declare('units',default=None,desc='Units')
    def setup(self):
        nn = self.options['num_nodes']
        units = self.options['units']
        self.add_input('start_val', units=units)
        self.add_input('end_val', units=units)
        self.add_output('vec', units=units,shape=(nn,))
        arange = np.arange(0,nn)
        self.declare_partials('vec','start_val', rows=arange, cols=np.zeros(nn), val=np.linspace(1,0,nn))
        self.declare_partials('vec','end_val', rows=arange, cols=np.zeros(nn), val=np.linspace(0,1,nn))

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        outputs['vec'] = np.linspace(inputs['start_val'],inputs['end_val'],nn)