from .aerodynamics import PolarDrag, StallSpeed, Lift
from .drag_jet_transport import ParasiteDragCoefficient_JetTransport
from .drag_BWB import ParasiteDragCoefficient_BWB
from .CL_max_estimation import CleanCLmax, FlapCLmax

try:
    from .openaerostruct import VLMDragPolar, AerostructDragPolar
except ImportError:
    pass
