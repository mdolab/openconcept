# Changes to make before merging

- [ ] Merge Shugo's work
- [x] Add docs:
  - [x] Examples
    - [x] Do simple example
    - [x] Show how aircraft model is made and how variables are passed
    - [x] Show analysis group
    - [x]  Show running the thing
  - [x] Describe features
    - [x] Aerodynamics
    - [x] Atmospherics
    - [x] Costs
    - [x] Mission analysis
    - [x] Propulsion
    - [x] Thermal
    - [x] Utilities
    - [x] Weights
  - [x] Maybe a list of papers that use OpenConcept like OpenMDAO does?
- [x] Add linting and formatting
- [ ] Fix TODO on first line of SteadyFlightPhase in solver_phases.py (ehh, it's one of the fun features of OpenConcept)
- [x] Remove old docs (old "features")

# Refactoring directory tree

- [x] Refactoring is done!

- openconcept
  - aerodynamics
  - atmospherics
  - costs
  - mission analysis
  - propulsion
    - thrust?
    - energy storage
      - batteries, (hydrogen in the future), etc.
    - layouts/architectures
  - thermal
  - utilities
    - math
  - weights

## Other refactoring

- [x] move examples folder to within openconcept directory! Will need to change the paths in the tutorial docs
- [ ] remove old code? For example, `OldIntegrator`
