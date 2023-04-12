"""
@File    :   test_CL_max_critical_section.py
@Date    :   2023/04/11
@Author  :   Eytan Adler
@Description : Test the critical section lift coefficient utility.
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
import unittest
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

# ==============================================================================
# Extension modules
# ==============================================================================
from openconcept.aerodynamics.openaerostruct import CLmaxCriticalSectionVLM, TrapezoidalPlanformMesh


class CLmaxCriticalSectionVLMTestCase(unittest.TestCase):
    def test(self):
        nx = 2
        ny = 5
        p = om.Problem()

        p.model.add_subsystem(
            "mesh",
            TrapezoidalPlanformMesh(num_x=nx, num_y=ny),
            promotes_inputs=["*"],
            promotes_outputs=[("mesh", "ac|geom|wing|OAS_mesh")],
        )
        p.model.add_subsystem("CL_max_comp", CLmaxCriticalSectionVLM(num_x=nx, num_y=ny), promotes=["*"])

        p.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, iprint=2, maxiter=10)
        p.model.linear_solver = om.DirectSolver()

        p.setup()

        p.set_val("S", 125, units="m**2")
        p.set_val("AR", 10)
        p.set_val("taper", 0.15)
        p.set_val("sweep", 25, units="deg")

        Cl_max_foil = 1.5
        p.set_val("ac|aero|airfoil_Cl_max", Cl_max_foil)
        p.set_val("fltcond|M", 0.2)
        p.set_val("fltcond|h", 0, units="ft")
        p.set_val("ac|geom|wing|toverc", np.full(ny, 0.12))

        p.run_model()

        assert_near_equal(np.max(p.get_val("VLM.sectional_CL")), Cl_max_foil, tolerance=1e-6)
        assert_near_equal(p.get_val("max_sectional_CL.KS").item(), Cl_max_foil, tolerance=1e-13)


if __name__ == "__main__":
    unittest.main()
