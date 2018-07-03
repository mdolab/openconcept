.. _DevRoadmap:

*******************
Development Roadmap
*******************

OpenConcept is in its infancy and is basically the product of one conference paper and a few months of work by one person.

Known issues to be addressed include:
    - No support for compressibility in the standard atmosphere / airspeed calculations
    - No support for additional mission phases (especially a diversion/reserve mission)
    - Spotty automated testing coverage
    - Spotty documentation coverage
    - Difficulty accessing / plotting optimized aircraft results (I hacked together some custom OpenMDAO/matplotlib code for this)

Upcoming major features will include:
    - Heat exchanger / coolant loop components

Future ideas include:
    - OpenAeroStruct integration (once OAS is upgraded to OpenMDAO 2.x)

I can use feedback on the design of the API; in particular, the airplane data structure.
