.. _DevRoadmap:

*******************
Development Roadmap
*******************

Known issues to be addressed include:
    - No distinction right now between calibrated, equivalent, indicated airspeeds (compressibility effects) in the standard atmosphere
    - Limited validation of the takeoff performance code (it is hard to find actual CLmax and drag polar data!)

Future ideas include:
    - Unifying the ODE integration math with NASA's Dymos toolkit
    - Adding locations to weights to be able to include stability constraints and trim OpenAeroStruct aerodynamic models
    - Incorporate OpenVSP for visualizations of the configuration
