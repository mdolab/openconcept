import openmdao.api as om 
import numpy as np 
from openconcept.utilities.math.integrals import Integrator
import warnings











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
    grp.add_subsystem('motorcool', MotorCoolingJacket(num_nodes=5))
    p = om.Problem(model=grp)
    p.setup(force_alloc_complex=True)

    p['motorcool.q_in'] = 50000
    p['motorcool.power_rating'] = 1e6
    p['motorcool.motor_weight'] = 1e6/5000
    p['motorcool.mdot_coolant'] = 0.1

    p.run_model()
    p.model.list_inputs(units=True, print_arrays=True)

    p.model.list_outputs(units=True, print_arrays=True)
    p.check_partials(compact_print=True, method='cs')
    print(p.get_val('delta_p', units='psi'))