"""
@File    :   B738_sizing.py
@Date    :   2023/03/25
@Author  :   Eytan Adler
@Description : Data needed B738_sizing OpenConcept example. The data is from a combination of:
- Technical site: http://www.b737.org.uk/techspecsdetailed.htm
- Wikipedia: https://en.wikipedia.org/wiki/Boeing_737#Specifications
- OpenConcept B738 model
"""

data = {
    "ac": {
        # ==============================================================================
        # Aerodynamics
        # ==============================================================================
        "aero": {
            "polar": {
                "e": {"value": 0.801},  # estimate from B738 example
            },
            "Mach_max": {"value": 0.82},
            "Vstall_land": {"value": 115, "units": "kn"},  # estimate
            "airfoil_Cl_max": {"value": 1.75},  # estimate for supercritical airfoil
            "takeoff_flap_deg": {"value": 15, "units": "deg"},
        },
        # ==============================================================================
        # Propulsion
        # ==============================================================================
        "propulsion": {
            "engine": {
                "rating": {"value": 27e3, "units": "lbf"},
            },
            "num_engines": {"value": 2},
        },
        # ==============================================================================
        # Geometry
        # ==============================================================================
        "geom": {
            # -------------- Wing --------------
            "wing": {
                "S_ref": {"value": 124.6, "units": "m**2"},
                "AR": {"value": 9.45},
                "c4sweep": {"value": 25, "units": "deg"},
                "taper": {"value": 0.159},
                "toverc": {"value": 0.12},  # estimate
            },
            # -------------- Horizontal stabilizer --------------
            "hstab": {
                # "S_ref": {"value": 32.78, "units": "m**2"},  # not needed since tail volume coefficients are used
                "AR": {"value": 6.16},
                "c4sweep": {"value": 30, "units": "deg"},
                "taper": {"value": 0.203},
                "toverc": {"value": 0.12},  # guess
            },
            # -------------- Vertical stabilizer --------------
            "vstab": {
                # "S_ref": {"value": 26.44, "units": "m**2"},  # not needed since tail volume coefficients are used
                "AR": {"value": 1.91},
                "c4sweep": {"value": 35, "units": "deg"},
                "taper": {"value": 0.271},
                "toverc": {"value": 0.12},  # guess
            },
            # -------------- Fuselage --------------
            "fuselage": {
                "length": {"value": 38.08, "units": "m"},
                "height": {"value": 3.76, "units": "m"},
            },
            # -------------- Nacelle --------------
            "nacelle": {
                "length": {"value": 4.3, "units": "m"},  # photogrammetry estimate
                "diameter": {"value": 2, "units": "m"},  # photogrammetry estimate
            },
            # -------------- Main landing gear --------------
            "maingear": {
                "length": {"value": 1.8, "units": "m"},
                "num_wheels": {"value": 4},
                "num_shock_struts": {"value": 2},
            },
            # -------------- Nose landing gear --------------
            "nosegear": {
                "length": {"value": 1.3, "units": "m"},
                "num_wheels": {"value": 2},
            },
        },
        # ==============================================================================
        # Weights
        # ==============================================================================
        "weights": {
            "W_payload": {"value": 18e3, "units": "kg"},
        },
        # ==============================================================================
        # Miscellaneous
        # ==============================================================================
        "num_passengers_max": {"value": 189},
        "num_flight_deck_crew": {"value": 2},
        "num_cabin_crew": {"value": 4},
        "cabin_pressure": {"value": 8.95, "units": "psi"},
    },
}
