from openmdao.api import IndepVarComp, Group
from openconcept.utilities import AddSubtractComp, ElementMultiplyDivideComp
from .weights_turboprop import (
    WingWeight_SmallTurboprop,
    EmpennageWeight_SmallTurboprop,
    FuselageWeight_SmallTurboprop,
    NacelleWeight_SmallSingleTurboprop,
    LandingGearWeight_SmallTurboprop,
    FuelSystemWeight_SmallTurboprop,
    EquipmentWeight_SmallTurboprop,
)


class TwinSeriesHybridEmptyWeight(Group):
    def setup(self):
        const = self.add_subsystem("const", IndepVarComp(), promotes_outputs=["*"])
        const.add_output("W_fluids", val=20, units="kg")
        const.add_output("structural_fudge", val=1.6, units="m/m")
        self.add_subsystem("wing", WingWeight_SmallTurboprop(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("empennage", EmpennageWeight_SmallTurboprop(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("fuselage", FuselageWeight_SmallTurboprop(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem(
            "nacelle", NacelleWeight_SmallSingleTurboprop(), promotes_inputs=["*"], promotes_outputs=["*"]
        )
        self.add_subsystem("gear", LandingGearWeight_SmallTurboprop(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem(
            "fuelsystem", FuelSystemWeight_SmallTurboprop(), promotes_inputs=["*"], promotes_outputs=["*"]
        )
        self.add_subsystem("equipment", EquipmentWeight_SmallTurboprop(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem(
            "structural",
            AddSubtractComp(
                output_name="W_structure",
                input_names=["W_wing", "W_fuselage", "W_nacelle", "W_empennage", "W_gear"],
                units="lb",
            ),
            promotes_outputs=["*"],
            promotes_inputs=["*"],
        )
        self.add_subsystem(
            "structural_fudge",
            ElementMultiplyDivideComp(
                output_name="W_structure_adjusted",
                input_names=["W_structure", "structural_fudge"],
                input_units=["lb", "m/m"],
            ),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )
        self.add_subsystem(
            "totalempty",
            AddSubtractComp(
                output_name="OEW",
                input_names=[
                    "W_structure_adjusted",
                    "W_fuelsystem",
                    "W_equipment",
                    "W_engine",
                    "W_motors",
                    "W_generator",
                    "W_propeller",
                    "W_fluids",
                ],
                units="lb",
            ),
            promotes_outputs=["*"],
            promotes_inputs=["*"],
        )
