# Changes to make before merging

- [ ] Merge Shugo's work
- [ ] Add docs:
  - [ ] Introduction
    - [ ] Describe generally what it does
    - [ ] What an aircraft model requires (throttle and CL in, thrust lift and drag out)
  - [ ] Examples
    - [x] Do simple example
    - [x] Show how aircraft model is made and how variables are passed
    - [x] Show analysis group
    - [x]  Show running the thing
  - [ ] Theory guide
    - [ ] Integration
  - [x] Maybe a list of papers that use OpenConcept like OpenMDAO does?
- [ ] Add linting and formatting
- [ ] Fix TODO on first line of SteadyFlightPhase in solver_phases.py

# Refactoring directory tree

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
