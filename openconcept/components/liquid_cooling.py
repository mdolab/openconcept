import openmdao.api as om 
import numpy as np

class SimplePump(om.ExplicitComponent):
    """
    A pump that circulates coolant against pressure.
    The default parameters are based on a survey of commercial 
    airplane fuel pumps of a variety of makes and models.

    Inputs
    ------
    power_rating : float
        Maximum rated electrical power (scalar, W)
    mdot_coolant : float
        Coolant mass flow rate (vector, kg/s)
    rho_coolant : float
        Coolant density (vector, kg/m3)
    delta_p : float
        Pressure rise provided by the pump (vector, kg/s)

    Outputs
    -------
    elec_load : float
        Electricity used by the pump (vector, W)
    component_weight : float
        Pump weight (scalar, kg)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length; default 1)
    efficiency : float
        Pump electrical + mech efficiency. Sensible range 0.0 to 1.0 (default 0.35)
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')
        self.options.declare('efficiency', default=0.35, desc='Efficiency (dimensionless)')
        self.options.declare('weight_base', default=0.0, desc='Pump base weight')
        self.options.declare('weight_inc', default=1/150*0.35, desc='Pump base weight (kg/W)')

    def setup(self):
        nn = self.options['num_nodes']
        eta = self.options['efficiency']
        weight_inc = self.options['weight_inc']

        self.add_input('power_rating', units='W', desc='Pump electrical power rating')
        self.add_input('mdot_coolant', units='kg/s', desc='Coolant mass flow rate', val=np.ones((nn,)))
        self.add_input('delta_p', units='Pa', desc='Pump pressure rise', val=np.ones((nn,)))
        self.add_input('rho_coolant', units='kg/m**3', desc='Coolant density', val=np.ones((nn,)))

        self.add_output('elec_load', units='W', desc='Pump electrical load', val=np.ones((nn,)))
        self.add_output('component_weight', units='kg', desc='Pump weight')
        self.add_output('component_sizing_margin', units=None, val=np.ones((nn,)), desc='Comp sizing margin')

        self.declare_partials(['elec_load','component_sizing_margin'], ['rho_coolant', 'delta_p', 'mdot_coolant'], rows=np.arange(nn), cols=np.arange(nn))
        self.declare_partials(['component_sizing_margin'], ['power_rating'], rows=np.arange(nn), cols=np.zeros(nn))
        self.declare_partials(['component_weight'], ['power_rating'], val=weight_inc)



    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        eta = self.options['efficiency']
        weight_inc = self.options['weight_inc']
        weight_base = self.options['weight_base']

        outputs['component_weight'] = weight_base + weight_inc * inputs['power_rating']
        
        # compute the fluid power
        vol_flow_rate = inputs['mdot_coolant'] / inputs['rho_coolant'] # m3/s
        fluid_power = vol_flow_rate * inputs['delta_p']
        outputs['elec_load'] = fluid_power / eta 
        outputs['component_sizing_margin'] = outputs['elec_load'] / inputs['power_rating']
    

    def compute_partials(self, inputs, J):
        nn = self.options['num_nodes']
        eta = self.options['efficiency']

        J['elec_load', 'mdot_coolant'] =  inputs['delta_p'] / inputs['rho_coolant'] / eta
        J['elec_load', 'delta_p'] = inputs['mdot_coolant'] / inputs['rho_coolant'] / eta
        J['elec_load', 'rho_coolant'] = -inputs['mdot_coolant'] * inputs['delta_p'] / inputs['rho_coolant'] ** 2 / eta
        for in_var in ['mdot_coolant', 'delta_p', 'rho_coolant']:
            J['component_sizing_margin', in_var] = J['elec_load', in_var]  / inputs['power_rating']
        J['component_sizing_margin', 'power_rating'] = - inputs['mdot_coolant'] * inputs['delta_p'] / inputs['rho_coolant'] / eta / inputs['power_rating'] ** 2

class SimpleHose(om.ExplicitComponent):
    """
    A coolant hose used to track pressure drop and weight in long hose runs.

    Inputs
    ------
    hose_diameter : float
        Inner diameter of the hose (scalar, m)
    hose_length
        Length of the hose (scalar, m)
    hose_design_pressure
        Max operating pressure of the hose (scalar, Pa)
    mdot_coolant : float
        Coolant mass flow rate (vector, kg/s)
    rho_coolant : float
        Coolant density (vector, kg/m3)
    mu_coolant : float
        Coolant viscosity (scalar, kg/m/s)

    Outputs
    -------
    delta_p : float
        Pressure drop in the hose - positive is loss (vector, kg/s)
    component_weight : float
        Weight of hose AND coolant (scalar, kg)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length; default 1)
    hose_operating_stress : float
        Hoop stress at design pressure (Pa) set to 300 Psi equivalent per empirical data
    hose_density : float
        Material density of the hose (kg/m3) set to 0.049 lb/in3 equivalent per empirical data
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')
        self.options.declare('hose_operating_stress', default=2.07e6, desc='Hoop stress at max op press in Pa')
        self.options.declare('hose_density', default=1356.3, desc='Hose matl density in kg/m3')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('hose_diameter', val=0.0254, units='m')
        self.add_input('hose_length', val=1.0, units='m')
        self.add_input('hose_design_pressure', units='Pa', val=1.03e6, desc='Hose max operating pressure')


        self.add_input('mdot_coolant', units='kg/s', desc='Coolant mass flow rate', val=np.ones((nn,)))
        self.add_input('rho_coolant', units='kg/m**3', desc='Coolant density', val=1020.*np.ones((nn,)))
        self.add_input('mu_coolant', val=1.68e-3, units='kg/m**3', desc='Coolant viscosity')

        self.add_output('delta_p', units='Pa', desc='Hose pressure drop', val=np.ones((nn,)))
        self.add_output('component_weight', units='kg', desc='Pump weight')

        self.declare_partials(['delta_p'], ['rho_coolant', 'mdot_coolant'], rows=np.arange(nn), cols=np.arange(nn))
        self.declare_partials(['delta_p'], ['hose_diameter', 'hose_length', 'mu_coolant'], rows=np.arange(nn), cols=np.zeros(nn))
        self.declare_partials(['component_weight'], ['hose_design_pressure','hose_length','hose_diameter'], method='cs')
        self.declare_partials(['component_weight'], ['rho_coolant'], rows=[0], cols=[0])


    def _compute_pressure_drop(self, inputs):
        xs_area = np.pi * (inputs['hose_diameter'] / 2) ** 2
        U = inputs['mdot_coolant'] / inputs['rho_coolant'] / xs_area
        Redh = inputs['rho_coolant'] * U * inputs['hose_diameter'] / inputs['mu_coolant']
        # darcy friction from the Blasius correlation
        f = 0.3164 * Redh ** (-1/4)
        dp = f * inputs['rho_coolant'] * U ** 2 * inputs['hose_length'] / 2 / inputs['hose_diameter']
        return dp

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        sigma = self.options['hose_operating_stress']
        rho_hose = self.options['hose_density']

        outputs['delta_p'] = self._compute_pressure_drop(inputs)

        thickness = inputs['hose_diameter'] * inputs['hose_design_pressure'] / 2 / sigma
        
        w_hose = (inputs['hose_diameter'] + thickness) * np.pi * thickness * rho_hose * inputs['hose_length']
        w_coolant = (inputs['hose_diameter'] / 2) ** 2 * np.pi * inputs['rho_coolant'][0] * inputs['hose_length']
        outputs['component_weight'] = w_hose + w_coolant
    

    def compute_partials(self, inputs, J):
        nn = self.options['num_nodes']
        sigma = self.options['hose_operating_stress']
        rho_hose = self.options['hose_density']

        J['component_weight','rho_coolant'] = (inputs['hose_diameter'] / 2) ** 2 * np.pi * inputs['hose_length']

        # use a colored complex step approach
        cs_step = 1e-30
        dp_base = self._compute_pressure_drop(inputs)

        cs_inp_list = ['rho_coolant', 'mdot_coolant', 'hose_diameter', 'hose_length', 'mu_coolant']
        fake_inputs = dict()
        # make a perturbable, complex copy of the inputs
        for inp in cs_inp_list:
            fake_inputs[inp] = inputs[inp].astype(np.complex_, copy=True)
        
        for inp in cs_inp_list:
            arr_to_restore = fake_inputs[inp].copy()
            fake_inputs[inp] += (0.0+cs_step*1.0j)
            dp_perturbed = self._compute_pressure_drop(fake_inputs)
            fake_inputs[inp] = arr_to_restore
            J['delta_p', inp] = np.imag(dp_perturbed) / cs_step


if __name__ == "__main__":
    ivg = om.IndepVarComp()
    ivg.add_output('mdot_coolant', 6.0, units='kg/s')
    ivg.add_output('hose_diameter', 0.033, units='m')
    ivg.add_output('rho_coolant', 1020., units='kg/m**3')
    ivg.add_output('hose_length', 20., units='m')
    ivg.add_output('power_rating', 4035., units='W')

    grp = om.Group()
    grp.add_subsystem('ivg', ivg, promotes=['*'])
    grp.add_subsystem('hose', SimpleHose(num_nodes=1), promotes_inputs=['*'], promotes_outputs=['delta_p'])
    grp.add_subsystem('pump', SimplePump(num_nodes=1), promotes_inputs=['*'])
    p = om.Problem(model=grp)
    p.setup(force_alloc_complex=True)
    p.run_model()
    p.model.list_inputs(units=True)

    p.model.list_outputs(units=True)
    p.check_partials(compact_print=True, method='cs')
    print(p.get_val('delta_p', units='psi'))