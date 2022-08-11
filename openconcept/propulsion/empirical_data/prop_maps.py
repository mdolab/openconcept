import numpy as np
from openmdao.api import ExplicitComponent
from openmdao.components.meta_model_structured_comp import MetaModelStructuredComp


def propeller_map_Raymer(vec_size=1):
    # Data from Raymer, Aircraft Design A Conceptual Approach, 4th Ed pg 498 fig 13.12 extrapolated in low cp range
    # For a 3 bladed constant-speed propeller
    J = np.linspace(0.2, 2.8, 14)
    cp = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    # raymer_data = np.ones((9,14))*0.75
    raymer_data = np.array(
        [
            [0.45, 0.6, 0.72, 0.75, 0.70, 0.65, 0.6, 0.55, 0.5, 0.45, 0.40, 0.35, 0.3, 0.25],
            [0.35, 0.6, 0.74, 0.83, 0.86, 0.88, 0.9, 0.9, 0.88, 0.85, 0.83, 0.8, 0.75, 0.7],
            [0.2, 0.35, 0.55, 0.7, 0.8, 0.85, 0.87, 0.9, 0.91, 0.92, 0.9, 0.9, 0.88, 0.87],
            [0.12, 0.22, 0.36, 0.51, 0.66, 0.75, 0.8, 0.85, 0.87, 0.88, 0.91, 0.905, 0.902, 0.9],
            [0.07, 0.15, 0.29, 0.36, 0.45, 0.65, 0.73, 0.77, 0.83, 0.85, 0.87, 0.875, 0.88, 0.895],
            [0.05, 0.12, 0.25, 0.32, 0.38, 0.50, 0.61, 0.72, 0.77, 0.79, 0.83, 0.85, 0.86, 0.865],
            [0.04, 0.11, 0.19, 0.26, 0.33, 0.40, 0.51, 0.61, 0.71, 0.74, 0.78, 0.815, 0.83, 0.85],
            [0.035, 0.085, 0.16, 0.22, 0.28, 0.35, 0.41, 0.52, 0.605, 0.69, 0.74, 0.775, 0.8, 0.82],
            [0.03, 0.06, 0.13, 0.19, 0.24, 0.31, 0.35, 0.46, 0.52, 0.63, 0.71, 0.75, 0.78, 0.8],
        ]
    )
    # Create regular grid interpolator instance
    interp = MetaModelStructuredComp(method="scipy_cubic", extrapolate=True, vec_size=vec_size)
    interp.add_input("cp", 0.3, cp)
    interp.add_input("J", 1, J)
    interp.add_output("eta_prop", 0.8, raymer_data)
    return interp


def propeller_map_scaled(vec_size=1, design_J=2.2, design_cp=0.2):
    # Data from Raymer, Aircraft Design A Conceptual Approach, 4th Ed pg 498 fig 13.12 extrapolated in low cp range
    # For a 3 bladed constant-speed propeller, scaled for higher design Cp
    J = np.linspace(0.2, 2.8 * design_J / 2.2, 14)
    cp = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5]) * design_cp / 0.2

    # raymer_data = np.ones((9,14))*0.75
    raymer_data = np.array(
        [
            [0.45, 0.6, 0.72, 0.75, 0.70, 0.65, 0.6, 0.55, 0.5, 0.45, 0.40, 0.35, 0.3, 0.25],
            [0.35, 0.6, 0.74, 0.83, 0.86, 0.88, 0.9, 0.9, 0.88, 0.85, 0.83, 0.8, 0.75, 0.7],
            [0.2, 0.35, 0.55, 0.7, 0.8, 0.85, 0.87, 0.9, 0.91, 0.92, 0.9, 0.9, 0.88, 0.87],
            [0.12, 0.22, 0.36, 0.51, 0.66, 0.75, 0.8, 0.85, 0.87, 0.88, 0.91, 0.905, 0.902, 0.9],
            [0.07, 0.15, 0.29, 0.36, 0.45, 0.65, 0.73, 0.77, 0.83, 0.85, 0.87, 0.875, 0.88, 0.895],
            [0.05, 0.12, 0.25, 0.32, 0.38, 0.50, 0.61, 0.72, 0.77, 0.79, 0.83, 0.85, 0.86, 0.865],
        ]
    )
    # Create regular grid interpolator instance
    interp = MetaModelStructuredComp(method="scipy_cubic", extrapolate=True, vec_size=vec_size)
    interp.add_input("cp", 0.3, cp)
    interp.add_input("J", 1, J)
    interp.add_output("eta_prop", 0.8, raymer_data)
    return interp


def propeller_map_highpower(vec_size=1):
    # Data from https://frautech.wordpress.com/2011/01/28/design-fridays-thats-a-big-prop/
    J = np.linspace(0.0, 4.0, 9)
    cp = np.linspace(0.0, 2.5, 13)

    # data = np.array([[0.28,0.51,0.65,0.66,0.65,0.64,0.63,0.62,0.61],
    #                         [0.27,0.50,0.71,0.82,0.81,0.70,0.68,0.67,0.66],
    #                         [0.26,0.49,0.72,0.83,0.86,0.85,0.75,0.70,0.69],
    #                         [0.25,0.45,0.71,0.82,0.865,0.875,0.84,0.79,0.72],
    #                         [0.24,0.42,0.69,0.815,0.87,0.885,0.878,0.84,0.80],
    #                         [0.23,0.40,0.65,0.81,0.865,0.89,0.903,0.873,0.83],
    #                         [0.22,0.38,0.61,0.78,0.85,0.88,0.91,0.90,0.86],
    #                         [0.21,0.34,0.58,0.73,0.83,0.876,0.904,0.91,0.88],
    #                         [0.20,0.31,0.53,0.71,0.81,0.87,0.895,0.91,0.882]])
    data = np.array(
        [
            [0.28, 0.51, 0.65, 0.66, 0.65, 0.64, 0.63, 0.62, 0.61],
            [0.20, 0.50, 0.71, 0.82, 0.81, 0.70, 0.68, 0.67, 0.66],
            [0.19, 0.49, 0.72, 0.83, 0.86, 0.85, 0.75, 0.70, 0.69],
            [0.18, 0.45, 0.71, 0.82, 0.865, 0.875, 0.84, 0.79, 0.72],
            [0.17, 0.42, 0.69, 0.815, 0.87, 0.885, 0.878, 0.84, 0.80],
            [0.155, 0.40, 0.65, 0.81, 0.865, 0.89, 0.903, 0.873, 0.83],
            [0.13, 0.38, 0.61, 0.78, 0.85, 0.88, 0.91, 0.90, 0.86],
            [0.12, 0.34, 0.58, 0.73, 0.83, 0.876, 0.904, 0.91, 0.88],
            [0.10, 0.31, 0.53, 0.71, 0.81, 0.87, 0.895, 0.91, 0.882],
            [0.08, 0.25, 0.44, 0.62, 0.75, 0.84, 0.88, 0.89, 0.87],
            [0.06, 0.18, 0.35, 0.50, 0.68, 0.79, 0.86, 0.86, 0.85],
            [0.05, 0.14, 0.25, 0.40, 0.55, 0.70, 0.79, 0.80, 0.72],
            [0.04, 0.12, 0.19, 0.29, 0.40, 0.50, 0.60, 0.60, 0.50],
        ]
    )

    data[:, 0] = np.zeros(13)
    # Create regular grid interpolator instance
    interp = MetaModelStructuredComp(method="scipy_cubic", extrapolate=True, vec_size=vec_size)
    interp.add_input("cp", 0.3, cp)
    interp.add_input("J", 1, J)
    interp.add_output("eta_prop", 0.8, data)
    return interp


class ConstantPropEfficiency(ExplicitComponent):
    def initialize(self):
        # define technology factors
        self.options.declare("vec_size", default=1, desc="Number of flight/control conditions")

    def setup(self):
        nn = self.options["vec_size"]
        self.add_input("cp", desc="Power coefficient", shape=(nn,))
        self.add_input("J", desc="Advance ratio", shape=(nn,))
        self.add_output("eta_prop", desc="Propulsive efficiency", shape=(nn,))
        self.declare_partials(["eta_prop"], ["*"], rows=range(nn), cols=range(nn), val=np.zeros(nn))

    def compute(self, inputs, outputs):
        outputs["eta_prop"] = 0.85


def propeller_map_constant_prop_efficiency(vec_size=1):
    interp = ConstantPropEfficiency(vec_size=vec_size)
    return interp


def static_propeller_map_Raymer(vec_size=1):
    # Data from Raymer for static thrust of 3-bladed propeller
    cp = np.linspace(0.0, 0.60, 25)
    raymer_static_data = np.array(
        [
            2.5,
            3.0,
            2.55,
            2.0,
            1.85,
            1.5,
            1.25,
            1.05,
            0.95,
            0.86,
            0.79,
            0.70,
            0.62,
            0.53,
            0.45,
            0.38,
            0.32,
            0.28,
            0.24,
            0.21,
            0.18,
            0.16,
            0.14,
            0.12,
            0.10,
        ]
    )
    interp = MetaModelStructuredComp(method="scipy_cubic", extrapolate=True, vec_size=vec_size)
    interp.add_input("cp", 0.15, cp)
    interp.add_output("ct_over_cp", 1.5, raymer_static_data)
    return interp


def static_propeller_map_highpower(vec_size=1):
    # Factoring up the thrust of the Raymer static thrust data to match the high power data
    cp = np.linspace(0.0, 1.0, 41)
    factored_raymer_static_data = np.array(
        [
            2.5,
            3.0,
            2.55,
            2.0,
            1.85,
            1.5,
            1.25,
            1.05,
            0.95,
            0.86,
            0.79,
            0.70,
            0.62,
            0.53,
            0.45,
            0.38,
            0.32,
            0.28,
            0.24,
            0.21,
            0.18,
            0.16,
            0.14,
            0.12,
            0.10,
            0.09,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
        ]
    )
    factored_raymer_static_data[6:] = factored_raymer_static_data[6:] * 1.2
    interp = MetaModelStructuredComp(method="scipy_cubic", extrapolate=True, vec_size=vec_size)
    interp.add_input("cp", 0.15, cp)
    interp.add_output("ct_over_cp", 1.5, factored_raymer_static_data)
    return interp
