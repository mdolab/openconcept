# Components
from .cfm56 import CFM56
from .generator import SimpleGenerator
from .motor import SimpleMotor
from .N3 import N3, N3Hybrid
from .propeller import SimplePropeller, WeightCalc, ThrustCalc, PropCoefficients
from .splitter import PowerSplit
from .turboshaft import SimpleTurboshaft

# Pre-made propulsion systems
from .systems import (
    AllElectricSinglePropulsionSystemWithThermal_Compressible,
    AllElectricSinglePropulsionSystemWithThermal_Incompressible,
    SeriesHybridElectricPropulsionSystem,
    SingleSeriesHybridElectricPropulsionSystem,
    TwinSeriesHybridElectricPropulsionSystem,
    TurbopropPropulsionSystem,
    TwinTurbopropPropulsionSystem,
    TwinSeriesHybridElectricThermalPropulsionSystem,
    TwinSeriesHybridElectricThermalPropulsionRefrigerated,
)
