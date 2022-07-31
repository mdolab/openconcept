# Changes to make before merging

- Add docs:
  - Introduction
    - Describe generally what it does
    - What an aircraft model requires (throttle and CL in, thrust lift and drag out)
  - Examples
    - Do simple example
    - Show how aircraft model is made and how variables are passed
    - Show analysis group
    - Show running the thing
  - Theory guide
    - Integration
  - Maybe a list of papers that use OpenConcept like OpenMDAO does?
- Add linting and formatting

# Refactoring directory tree

### Note: talk to Neil about the best way to expose this structure to the user. Is it just by using the dot syntax in the import statements directly or using some api file (like OpenMDAO) or something else?

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