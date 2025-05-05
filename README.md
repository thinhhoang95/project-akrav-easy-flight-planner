# Flight ETA Solver

This package provides a solution for calculating Estimated Time of Arrival (ETA) for aircraft along a defined route, considering various factors like climb/descent profiles, cruise speed, cruise altitude, and wind forecasts.

## Overview

The ETA solver implements the "Variant of the Shortest Path" approach for Top of Descent (TOD) calculation and handles the ETA-wind dependency through a simplified iteration method. The implementation calculates ETAs for each fix (waypoint) along a flight route and determines the flight phase at each point.

## Features

- Calculate ETA for every waypoint (fix) in a flight route
- Determine flight phases (TakeOff, Climb, Cruise, Descent, Arrival)
- Account for wind effects using a provided wind model
- Estimate Top of Climb (TOC) and Top of Descent (TOD) points
- Support for simplified aircraft performance profiles

## Installation

No special installation is required. Simply clone this repository and ensure you have Python 3.6+ installed with the required dependencies (only standard libraries are used).

## Usage

### Basic Example

```python
from datetime import datetime
from eta_solver import Fix, WindVector, AircraftPerformance, calculate_eta

# Define a simple wind model
def simple_wind_model(location, altitude, time):
    return WindVector(speed=20.0, direction=270.0)  # 20 knots from the west

# Create a performance model
performance = AircraftPerformance()

# Define a route
fixes = [
    Fix("KSFO", 37.619, -122.375),  # San Francisco
    Fix("KLAS", 36.080, -115.153)   # Las Vegas
]

# Calculate ETA
result = calculate_eta(
    fixes=fixes,
    takeoff_time=datetime(2023, 6, 1, 10, 0, 0),
    cruise_alt=35000.0,  # feet
    cruise_tas=450.0,    # knots
    final_alt=1500.0,    # feet
    performance=performance,
    wind_model=simple_wind_model
)

# Access results
print(f"Final ETA: {result['final_eta']}")
print(f"TOC at fix index: {result['toc_fix_index']}")
print(f"TOD at fix index: {result['tod_fix_index']}")
```

### Running the Example

The repository includes a working example that can be run as follows:

```
python example_eta.py
```

This will calculate and display ETAs for a route from San Francisco to Las Vegas.

### Running Tests

Unit tests are provided to verify the implementation:

```
python -m unittest test_eta_solver.py
```

## Implementation Details

The ETA solver follows a sequential calculation approach:

1. **Initialization**: Set up the route with takeoff time and performance parameters
2. **Climb Phase**: Find the Top of Climb fix and calculate ETAs during climb
3. **Cruise Phase**: Calculate ETAs for fixes during cruise
4. **Top of Descent Determination**: At each cruise fix, check if TOD should be initiated based on:
   - Remaining route distance
   - Required descent distance (based on altitude to lose and descent profile)
5. **Descent Phase**: Calculate ETAs for fixes during descent
6. **Result Compilation**: Provide a comprehensive result with ETAs, phases, and TOC/TOD information

### Wind Modeling

The ETA solver handles wind through a provided wind model function that should return a `WindVector` given a location, altitude, and time. This allows for flexibility in implementing different wind forecast sources.

To handle the co-dependency between ETA and wind (wind depends on ETA, which depends on wind), a simplified iteration approach is used:
1. Estimate an initial segment time with zero wind
2. Look up wind at the midpoint of the estimated time
3. Calculate ground speed with the looked-up wind
4. Recalculate segment time with the new ground speed

## Limitations

This implementation has several limitations compared to a full Flight Management System:

- No aircraft weight & fuel burn consideration
- Fixed cruise speed (no Cost Index optimization)
- Simplified climb/descent profiles
- No step climbs
- No detailed SID/STAR/Approach procedures
- No ATC constraints/vectors/rerouting
- Simplified wind model
- No contingency planning

## Error Estimation

The implementation accuracy is estimated to be:
- Short Haul (1-2 hours): ±10-15 minutes
- Medium Haul (3-5 hours): ±15-25 minutes
- Long Haul (8+ hours): ±30 minutes or more

The largest sources of error are typically ATC interventions, wind forecast accuracy, and simplified performance modeling. 