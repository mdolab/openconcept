import openmdao.api as om 
import numpy as np 
from openconcept.components.thermal import SimpleEngine

if __name__ == "__main__":
    # create a problem
    nn = 5
    prob = om.Problem(SimpleEngine(num_nodes=nn))
    prob.setup(force_alloc_complex=True)
    prob.run_model()
    prob.model.list_inputs(print_arrays=True)
    prob.model.list_outputs(print_arrays=True)
    prob.check_partials(compact_print=True, method='cs')