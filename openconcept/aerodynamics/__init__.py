from .aerodynamics import PolarDrag, StallSpeed, Lift

try:
    from .openaerostruct import VLMDragPolar, AerostructDragPolar
except ImportError:
    pass
