# Unit Test Architecture

Rules to follow
- Fixtures
    - Put the functions that initialize fixtures in mock_utilities
    - Use fixtures when needing default/valid base class, use functions
    when needing to mutate base class
    - When using fixtures, use type hints 
- Abstract Classes
    - For relevant functions, test the subclasses for required behavior (set_ready, store required attributes, etc.)
        - Use instantiate with mocks for building subclasses. Get the subclasses from registries
    - Split verification tests into multiple tests that describe what the test is checking for
- 

## Main

## Hardware

### Hardware Types

## Environment

### Environment File
Fixtures
    HardwareMetadata
    EnvironmentMetdata
    EnvironmentInstructions
    EnvironmentQueues
    Environment
    

## Process

## User Interface

## Examples