# Airport Matching for Flight Routes

## Overview

This project provides code to match flight routes to airport-to-airport pairs by:

1. **For each flight route**, examine the first and last waypoints
2. **If a waypoint has 4 characters**, it's already an airport (ICAO code)
3. **If a waypoint is not an airport**, look up its coordinates in a navigation graph
4. **Find the nearest airport** that is closest to being 3.5 nautical miles from that waypoint
5. **Generate airport-to-airport route pairs** for flight planning analysis

## Files Created

### Main Scripts

1. **`airport_matching_final.py`** - The complete, optimized script
   - Uses the comprehensive `ats_fra_nodes_only.gml` graph (33,763 waypoints)
   - Processes all flight routes or a specified subset
   - Outputs detailed results with statistics

2. **`test_airport_matching.py`** - Test version for smaller datasets
   - Processes only 100 routes for quick testing
   - Good for verification and debugging

3. **`airport_matching_optimized.py`** - Optimized version with vectorized operations
   - Includes batched processing for large datasets
   - Better memory management

### Output Files

- **`airport_matched_routes_complete.csv`** - Complete results for all processed routes
- **`airport_matched_routes_test.csv`** - Test results for verification

## Input Data

### Required Files

1. **`data/routes_wps/cs_2023-04-01.routes.csv`** - Flight routes data
   - Contains columns: `flight_id`, `real_waypoints`, etc.
   - `real_waypoints` is a space-separated string of waypoint names

2. **`data/airac/airports.csv`** - Airport database
   - Contains columns: `icao`, `name`, `latitude`, `longitude`, `elevation`
   - Used for matching waypoints to nearby airports

3. **Navigation Graph** (automatically selected):
   - **Primary**: `data/graphs/ats_fra_nodes_only.gml` (33,763 waypoints)
   - **Fallback**: `route_graph_compute/LEMD_EGLL.gml` (566 waypoints, regional)

## Algorithm Details

### Waypoint Processing Logic

```python
for each flight route:
    first_waypoint = route.split()[0]
    last_waypoint = route.split()[-1]
    
    for waypoint in [first_waypoint, last_waypoint]:
        if len(waypoint) == 4:
            # Already an airport (ICAO code)
            matched_airport = waypoint
            distance = 0.0
        else:
            # Look up coordinates in navigation graph
            lat, lon = graph.nodes[waypoint]['lat'], graph.nodes[waypoint]['lon']
            
            # Find airport closest to being 3.5 NM away
            distances = calculate_distances_to_all_airports(lat, lon)
            target_diffs = abs(distances - 3.5)
            matched_airport = airports[argmin(target_diffs)]
```

### Distance Calculation

- Uses vectorized Haversine distance calculation for performance
- Target distance: **3.5 nautical miles**
- Finds airport closest to being exactly 3.5 NM from the waypoint

## Results Summary

### Test Results (1000 routes)

- **Waypoint Coverage**: 100% (all waypoints found in comprehensive graph)
- **Airport Matching**: 100% (all routes matched to airport pairs)
- **Already Airports**: ~33% of waypoints are already airport codes

### Distance Statistics

- **Mean Distance**: ~43 NM (waypoints to matched airports)
- **Median Distance**: ~30-35 NM  
- **Within 1 NM of Target (3.5 NM)**: ~5% of matches
- **Range**: 0.1 - 240 NM

## Output Format

The output CSV contains these columns:

| Column | Description |
|--------|-------------|
| `flight_id` | Original flight identifier |
| `original_route` | Original waypoint string |
| `first_waypoint` | First waypoint in route |
| `last_waypoint` | Last waypoint in route |
| `first_waypoint_is_airport` | Boolean: is first waypoint already an airport? |
| `last_waypoint_is_airport` | Boolean: is last waypoint already an airport? |
| `first_waypoint_lat/lon` | Coordinates from navigation graph |
| `last_waypoint_lat/lon` | Coordinates from navigation graph |
| `first_matched_airport` | Matched airport ICAO code for first waypoint |
| `last_matched_airport` | Matched airport ICAO code for last waypoint |
| `first_airport_distance` | Distance from first waypoint to matched airport (NM) |
| `last_airport_distance` | Distance from last waypoint to matched airport (NM) |
| `first_distance_diff_from_target` | Difference from 3.5 NM target |
| `last_distance_diff_from_target` | Difference from 3.5 NM target |

## Usage

### Run Complete Processing

```bash
python airport_matching_final.py
```

### Run Test with Small Subset

```bash
python test_airport_matching.py
```

### Configuration Options

Edit the configuration section in the script:

```python
# Configuration
routes_file = 'data/routes_wps/cs_2023-04-01.routes.csv'
airports_file = 'data/airac/airports.csv'
output_file = 'airport_matched_routes_complete.csv'
target_distance = 3.5  # nautical miles
max_routes = None  # Set to number for partial processing, None for all
```

## Performance

- **Processing Speed**: ~1,000 routes/minute
- **Memory Usage**: Batched processing for large datasets
- **Graph Loading**: ~10-30 seconds for large navigation graphs
- **Vectorized Operations**: Optimized distance calculations

## Key Features

1. **Automatic Graph Selection**: Chooses best available navigation graph
2. **Flexible File Format Support**: Handles both .gml and .graphml files
3. **Comprehensive Waypoint Coverage**: Uses large aviation navigation database
4. **Vectorized Distance Calculations**: Fast processing of large datasets
5. **Detailed Statistics**: Comprehensive reporting of matching results
6. **Batch Processing**: Memory-efficient handling of large route datasets

## Dependencies

```python
import pandas as pd
import numpy as np
import networkx as nx
from utils.haversine import haversine_distance
from tqdm import tqdm
import os
import glob
```

## Example Use Cases

1. **Route Analysis**: Convert waypoint-based routes to airport pairs
2. **Flight Planning**: Identify origin/destination airports for route optimization  
3. **Traffic Flow Analysis**: Study airport-to-airport traffic patterns
4. **Navigation Research**: Analyze relationship between waypoints and airports
5. **Data Quality**: Assess coverage of navigation databases

## Notes

- The algorithm prioritizes finding airports close to 3.5 NM rather than simply the nearest airport
- Waypoints with 4 characters are assumed to be airport ICAO codes
- The comprehensive navigation graph provides excellent waypoint coverage
- Results include both direct airport matches and computed matches via waypoint coordinates 