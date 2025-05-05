import math
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Callable, Optional, Union

class Fix:
    """Represents a navigation fix (waypoint) with coordinates and other properties."""
    def __init__(self, name: str, lat: float, lon: float, altitude_constraint: Optional[float] = None):
        self.name = name
        self.lat = lat
        self.lon = lon
        self.altitude_constraint = altitude_constraint
        self.location = (lat, lon)  # For convenience

class WindVector:
    """Represents a wind vector with speed and direction."""
    def __init__(self, speed: float, direction: float):
        """
        Args:
            speed: Wind speed in knots
            direction: Wind direction in degrees (meteorological convention - FROM direction)
        """
        self.speed = speed
        self.direction = direction

class AircraftPerformance:
    """Simplified aircraft performance parameters."""
    def __init__(self, 
                 standard_climb_rate: float = 2000.0,  # ft/min
                 standard_descent_rate: float = 1800.0,  # ft/min
                 descent_angle: float = 3.0,  # degrees
                 climb_speeds: Dict[str, float] = None,
                 descent_speeds: Dict[str, float] = None):
        """
        Args:
            standard_climb_rate: Standard rate of climb in ft/min
            standard_descent_rate: Standard rate of descent in ft/min
            descent_angle: Standard descent angle in degrees
            climb_speeds: Dictionary of altitude bands to climb speeds (TAS in knots)
            descent_speeds: Dictionary of altitude bands to descent speeds (TAS in knots)
        """
        self.standard_climb_rate = standard_climb_rate
        self.standard_descent_rate = standard_descent_rate
        self.descent_angle = descent_angle
        
        # Default climb speed profile if none provided
        self.climb_speeds = climb_speeds or {
            0: 180,      # Ground level
            10000: 250,  # Below 10,000 ft
            30000: 280,  # Below 30,000 ft
            99999: 300   # Above 30,000 ft
        }
        
        # Default descent speed profile if none provided
        self.descent_speeds = descent_speeds or {
            0: 180,      # Ground level
            10000: 250,  # Below 10,000 ft
            30000: 280,  # Below 30,000 ft
            99999: 300   # Above 30,000 ft
        }

def calculate_distance(fix1: Fix, fix2: Fix) -> float:
    """
    Calculate the great circle distance between two fixes in nautical miles.
    
    Args:
        fix1: First fix
        fix2: Second fix
        
    Returns:
        Distance in nautical miles
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = math.radians(fix1.lat), math.radians(fix1.lon)
    lat2, lon2 = math.radians(fix2.lat), math.radians(fix2.lon)
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    # Radius of earth in nautical miles
    radius = 3440.065  # Earth radius in nautical miles
    distance = radius * c
    
    return distance

def calculate_track(fix1: Fix, fix2: Fix) -> float:
    """
    Calculate the initial true track (course) from fix1 to fix2 in degrees.
    
    Args:
        fix1: First fix
        fix2: Second fix
        
    Returns:
        True track in degrees [0-360)
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = math.radians(fix1.lat), math.radians(fix1.lon)
    lat2, lon2 = math.radians(fix2.lat), math.radians(fix2.lon)
    
    # Calculate initial bearing
    x = math.sin(lon2 - lon1) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
    initial_bearing = math.atan2(x, y)
    
    # Convert from radians to degrees
    initial_bearing = math.degrees(initial_bearing)
    
    # Normalize to [0, 360)
    track = (initial_bearing + 360) % 360
    
    return track

def calculate_head_tailwind(wind_vector: WindVector, track: float) -> float:
    """
    Calculate the headwind/tailwind component along a given track.
    Positive values indicate a tailwind, negative values indicate a headwind.
    
    Args:
        wind_vector: Wind vector (speed and direction)
        track: Track in degrees
        
    Returns:
        Headwind/tailwind component in knots
    """
    # Convert wind direction from meteorological (FROM) to mathematical convention
    wind_dir_math = (wind_vector.direction + 180) % 360
    
    # Calculate the angle between the wind direction and the track
    angle_diff = math.radians(wind_dir_math - track)
    
    # Calculate the headwind/tailwind component
    wind_component = wind_vector.speed * math.cos(angle_diff)
    
    return wind_component

def calculate_remaining_route_distance(fixes: List[Fix], start_idx: int, end_idx: int) -> float:
    """
    Calculate the total distance remaining from a specific fix index to the end of the route.
    
    Args:
        fixes: List of fixes
        start_idx: Starting fix index
        end_idx: Ending fix index
        
    Returns:
        Total distance in nautical miles
    """
    total_distance = 0.0
    for i in range(start_idx, end_idx):
        total_distance += calculate_distance(fixes[i], fixes[i+1])
    return total_distance

def get_climb_tas(altitude: float, performance: AircraftPerformance) -> float:
    """
    Get the True Air Speed (TAS) for climbing at the given altitude.
    
    Args:
        altitude: Current altitude in feet
        performance: Aircraft performance parameters
        
    Returns:
        True Air Speed in knots
    """
    # Find the appropriate speed for the current altitude
    for alt_threshold in sorted(performance.climb_speeds.keys()):
        if altitude <= alt_threshold:
            return performance.climb_speeds[alt_threshold]
    
    # If we get here, use the highest altitude speed
    return performance.climb_speeds[max(performance.climb_speeds.keys())]

def get_descent_tas(altitude: float, performance: AircraftPerformance) -> float:
    """
    Get the True Air Speed (TAS) for descending at the given altitude.
    
    Args:
        altitude: Current altitude in feet
        performance: Aircraft performance parameters
        
    Returns:
        True Air Speed in knots
    """
    # Find the appropriate speed for the current altitude
    for alt_threshold in sorted(performance.descent_speeds.keys()):
        if altitude <= alt_threshold:
            return performance.descent_speeds[alt_threshold]
    
    # If we get here, use the highest altitude speed
    return performance.descent_speeds[max(performance.descent_speeds.keys())]

def estimate_required_descent_distance(cruise_alt: float, final_alt: float, 
                                       end_location: Tuple[float, float], eta: datetime,
                                       wind_model: Callable, performance: AircraftPerformance,
                                       track: float) -> float:
    """
    Estimate the required ground distance for descent from cruise altitude to final altitude.
    
    Args:
        cruise_alt: Cruise altitude in feet
        final_alt: Final altitude in feet
        end_location: Final location (lat, lon), arrival airport
        eta: Estimated time of arrival at current location
        wind_model: Function that returns wind vector given location, altitude, and time
        performance: Aircraft performance parameters
        
    Returns:
        Required descent distance in nautical miles
    """
    # Calculate altitude to lose
    delta_alt = cruise_alt - final_alt
    
    # Estimate average descent altitude (halfway between cruise and final)
    avg_altitude = (cruise_alt + final_alt) / 2 # Old simple average altitude
    
    # --- Calculate Weighted Average Descent TAS ---
    # Sort altitude thresholds and speeds for descent
    descent_alt_thresholds = sorted(performance.descent_speeds.keys())
    weighted_sum_tas = 0.0
    total_altitude_dropped = 0.0
    
    # Iterate through altitude bands from cruise altitude down to final altitude
    current_alt = cruise_alt
    for i in range(len(descent_alt_thresholds) - 1, -1, -1): # Iterate downwards
        band_lower_alt = descent_alt_thresholds[i]
        # Upper limit of the current band (is the lower limit of the next higher band, or cruise_alt if it's the highest band involved)
        band_upper_alt = descent_alt_thresholds[i+1] if i + 1 < len(descent_alt_thresholds) else cruise_alt

        # Determine the actual altitude range *within* this band during the descent
        effective_upper = min(current_alt, band_upper_alt)
        effective_lower = max(final_alt, band_lower_alt)

        if effective_upper > effective_lower:
            # Altitude dropped within this specific band
            band_alt_drop = effective_upper - effective_lower
            
            # Get TAS for this altitude band (using the lower threshold as key)
            band_tas = performance.descent_speeds[band_lower_alt]
            
            # Add weighted contribution of this band to the average TAS
            weighted_sum_tas += band_tas * band_alt_drop
            total_altitude_dropped += band_alt_drop
            
            # Update the current altitude for the next lower band
            current_alt = effective_lower

        # Stop if we've processed the band containing the final altitude
        if current_alt <= final_alt:
            break

    # Calculate the weighted average TAS
    if total_altitude_dropped > 0:
        avg_tas = weighted_sum_tas / total_altitude_dropped
    else:
        # Fallback if no altitude dropped or single point descent
        avg_tas = get_descent_tas( (cruise_alt + final_alt) / 2, performance) 
    # --- End Weighted Average Descent TAS Calculation ---

    # Get average descent TAS
    # avg_tas = get_descent_tas(avg_altitude, performance) # Old simple TAS calculation
    
    # Estimate wind at the average descent altitude
    # This is a simplification as we don't know the exact path yet
    wind_vector = wind_model(end_location, avg_altitude, eta) # avg_altitude is the average altitude of the descent
    
    # Calculate headwind/tailwind component
    wind_comp = calculate_head_tailwind(wind_vector, track)
    
    # Calculate average ground speed during descent
    avg_gs = avg_tas + wind_comp
    
    # Calculate descent time based on performance
    # Option 1: Using standard descent rate
    # time_desc = delta_alt / performance.standard_descent_rate  # in minutes
    
    # Option 2: Using descent angle (more common in practice)
    # Convert descent angle to radians
    descent_angle_rad = math.radians(performance.descent_angle)
    
    # Calculate descent rate based on ground speed and angle
    # RoD (ft/min) = GS (nm/min) * 6076 (ft/nm) * sin(angle)
    gs_fpm = avg_gs / 60  # Convert knots to nautical miles per minute
    rod = gs_fpm * 6076.12 * math.sin(descent_angle_rad)  # Rate of Descent in ft/min
    
    # Calculate descent time in minutes
    time_desc = delta_alt / rod
    
    # Calculate required descent distance
    dist_desc_req = (avg_gs / 60) * time_desc  # Convert from minutes to hours for GS calculation
    
    return dist_desc_req

def calculate_current_descent_altitude(current_eta: datetime, tod_eta: datetime, 
                                      cruise_alt: float, performance: AircraftPerformance) -> float:
    """
    Calculate the current altitude during descent phase based on time since TOD.
    
    Args:
        current_eta: Current ETA at the fix
        tod_eta: ETA at the Top of Descent
        cruise_alt: Cruise altitude in feet
        performance: Aircraft performance parameters
        
    Returns:
        Current altitude in feet
    """
    # Calculate time elapsed since TOD in minutes
    time_since_tod = (current_eta - tod_eta).total_seconds() / 60.0
    
    # Calculate altitude lost based on standard descent rate
    altitude_lost = time_since_tod * performance.standard_descent_rate
    
    # Calculate current altitude
    current_alt = cruise_alt - altitude_lost
    
    # Ensure altitude is not negative
    return max(0, current_alt)

def find_top_of_climb_fix(fixes: List[Fix], takeoff_time: datetime, cruise_alt: float,
                         performance: AircraftPerformance, wind_model: Callable) -> Tuple[int, Tuple[float, float], datetime]:
    """
    Find the exact location and time of Top of Climb (TOC).

    Args:
        fixes: List of original fixes
        takeoff_time: Takeoff time
        cruise_alt: Cruise altitude in feet
        performance: Aircraft performance parameters
        wind_model: Function that returns wind vector given location, altitude, and time

    Returns:
        Tuple of (index of fix before TOC, TOC coordinates (lat, lon), ETA at TOC)
    """
    if cruise_alt <= 0: # Handle cases where cruise altitude is at or below ground
         return 0, (fixes[0].lat, fixes[0].lon), takeoff_time

    # --- Simplified Climb Calculation (as before) ---
    time_to_climb = cruise_alt / performance.standard_climb_rate  # minutes, ~ CRZ_ALT/2,000
    
    # Calculate a more realistic average climb TAS by considering all altitude bands
    # Sort the altitude thresholds and create bands for calculation
    alt_thresholds = sorted(performance.climb_speeds.keys())
    weighted_sum_tas = 0.0
    total_altitude = 0.0
    
    # Process each altitude band up to cruise altitude
    for i in range(len(alt_thresholds) - 1):
        lower_alt = alt_thresholds[i]
        upper_alt = alt_thresholds[i + 1]
        
        if lower_alt >= cruise_alt:
            # We've reached the cruise altitude, no more climb needed
            break
            
        # Calculate the actual upper limit for this band (either band limit or cruise alt)
        effective_upper_alt = min(upper_alt, cruise_alt)
        
        if effective_upper_alt <= lower_alt:
            # Skip this band if it's above cruise altitude
            continue
            
        # Calculate altitude span in this band
        band_span = effective_upper_alt - lower_alt
        
        # Get TAS for this band
        band_tas = performance.climb_speeds[lower_alt]
        
        # Add weighted contribution of this band to average TAS
        weighted_sum_tas += band_tas * band_span
        total_altitude += band_span
    
    # If cruise altitude is above the highest defined threshold
    if cruise_alt > alt_thresholds[-1]:
        highest_band_tas = performance.climb_speeds[alt_thresholds[-1]]
        remaining_altitude = cruise_alt - alt_thresholds[-1]
        weighted_sum_tas += highest_band_tas * remaining_altitude
        total_altitude += remaining_altitude
    
    # Calculate weighted average TAS for climb
    avg_climb_tas = weighted_sum_tas / total_altitude if total_altitude > 0 else get_climb_tas(cruise_alt, performance)
    
    
    surface_wind = wind_model(fixes[0].location, 0, takeoff_time)
    # Estimate time at cruise for wind lookup - approximate
    approx_toc_eta = takeoff_time + timedelta(minutes=time_to_climb)
    cruise_wind = wind_model(fixes[0].location, cruise_alt, approx_toc_eta) # Use estimated TOC time

    # The average wind during climb is the average of the surface wind and the cruise wind
    avg_wind_comp = 0
    if len(fixes) > 1:
        track = calculate_track(fixes[0], fixes[1])
        surface_wind_comp = calculate_head_tailwind(surface_wind, track)
        cruise_wind_comp = calculate_head_tailwind(cruise_wind, track)
        avg_wind_comp = (surface_wind_comp + cruise_wind_comp) / 2
    else:
        # If only one fix, TOC is at the first fix.
         return 0, (fixes[0].lat, fixes[0].lon), takeoff_time


    avg_climb_gs = avg_climb_tas + avg_wind_comp
    # Prevent non-positive ground speed during climb
    if avg_climb_gs <= 0:
        print(f"Warning: Non-positive average ground speed calculated during climb ({avg_climb_gs:.1f} kts). Adjusting to 5 kts.")
        avg_climb_gs = 5.0

    climb_distance = (avg_climb_gs / 60.0) * time_to_climb
    toc_eta = takeoff_time + timedelta(minutes=time_to_climb)
    # --- End Simplified Climb Calculation ---

    # Find the segment where TOC occurs
    cumulative_distance = 0.0
    toc_segment_start_idx = -1 # store the index of the fix before the TOC

    for i in range(len(fixes) - 1):
        segment_distance = calculate_distance(fixes[i], fixes[i+1])
        if cumulative_distance + segment_distance >= climb_distance:
            # TOC falls within this segment (i -> i+1)
            distance_on_segment = climb_distance - cumulative_distance
            toc_coords = point_along_path(fixes[i], fixes[i+1], distance_on_segment)
            toc_segment_start_idx = i
            return toc_segment_start_idx, toc_coords, toc_eta
        cumulative_distance += segment_distance
        toc_segment_start_idx = i # Keep track of last segment start index

    # If climb distance exceeds route distance, TOC is effectively at the last fix
    print(f"Warning: Calculated climb distance ({climb_distance:.1f} NM) exceeds total route distance ({cumulative_distance:.1f} NM). Setting TOC at the last fix.")
    last_fix = fixes[-1]
    return len(fixes) - 2, (last_fix.lat, last_fix.lon), toc_eta # Index before last fix

def calculate_eta(fixes: List[Fix], takeoff_time: datetime, cruise_alt: float,
                 cruise_tas: float, final_alt: float, performance: AircraftPerformance,
                 wind_model: Callable, taxi_out_time: float = 15.0) -> Dict:
    """
    Calculate Estimated Time of Arrival (ETA) for each fix along the route,
    inserting distinct TOC and TOD points into the route.

    Args:
        fixes: List of original fixes defining the route structure.
        takeoff_time: Scheduled takeoff time.
        cruise_alt: Cruise altitude in feet.
        cruise_tas: Cruise True Air Speed in knots.
        final_alt: Final altitude at destination in feet.
        performance: Aircraft performance parameters.
        wind_model: Function returning wind vector (location, alt, time).
        taxi_out_time: Taxi out time in minutes (default 15.0).

    Returns:
        Dictionary with results including the modified route list, ETAs,
        phases, and original indices of TOC/TOD insertion points.
    """
    if len(fixes) < 2:
        return {"error": "At least two fixes are required"}

    # --- 1. Initialization and TOC Calculation ---
    route_fixes = list(fixes) # Create a mutable copy to insert TOC/TOD
    actual_takeoff_time = takeoff_time + timedelta(minutes=taxi_out_time)

    # Find TOC location, the index of the fix *before* it, and ETA
    toc_segment_start_idx, toc_coords, toc_eta = find_top_of_climb_fix(
        route_fixes, actual_takeoff_time, cruise_alt, performance, wind_model
    )

    # Create and insert TOC fix
    toc_fix = Fix("TOC", toc_coords[0], toc_coords[1], altitude_constraint=cruise_alt)
    # Insert *after* the starting fix of the segment where TOC occurs
    toc_inserted_idx = toc_segment_start_idx + 1
    route_fixes.insert(toc_inserted_idx, toc_fix)

    # --- 2. TOD Calculation ---
    tod_inserted_idx = -1 # Initialize TOD index

    # Estimate descent distance required once using final fix location and estimated final ETA
    # --- Refined Initial Final ETA Estimation ---
    # 1. Estimate descent time
    delta_alt_desc = cruise_alt - final_alt
    if performance.standard_descent_rate > 0:
        estimated_desc_time_min = delta_alt_desc / performance.standard_descent_rate
        estimated_desc_time_hrs = estimated_desc_time_min / 60.0
    else:
        estimated_desc_time_hrs = 0 # Avoid division by zero

    # 2. Estimate average descent TAS (simplified for initial estimate)
    avg_desc_alt_est = (cruise_alt + final_alt) / 2
    avg_desc_tas_est = get_descent_tas(avg_desc_alt_est, performance)

    # 3. Estimate distance covered during descent
    estimated_desc_dist = avg_desc_tas_est * estimated_desc_time_hrs

    # 4. Calculate total distance from TOC to destination
    total_dist_toc_to_dest = calculate_remaining_route_distance(route_fixes, toc_inserted_idx, len(route_fixes) - 1)

    # 5. Estimate cruise distance and time
    estimated_cruise_dist = max(0, total_dist_toc_to_dest - estimated_desc_dist)
    estimated_cruise_time_hrs = estimated_cruise_dist / cruise_tas if cruise_tas > 0 else 0

    # 6. Calculate improved estimated final ETA
    estimated_final_eta = toc_eta + timedelta(hours=estimated_cruise_time_hrs) + timedelta(hours=estimated_desc_time_hrs)
    # --- End Refined Initial Final ETA Estimation ---


    # A more accurate approach would refine this iteratively, but let's start simple for now!
    # Estimate final ETA roughly based on cruise speed for initial descent calc:
    # approx_cruise_dist = calculate_remaining_route_distance(route_fixes, toc_inserted_idx, len(route_fixes) - 1)
    # approx_cruise_time_hrs = approx_cruise_dist / cruise_tas if cruise_tas > 0 else 0
    # estimated_final_eta = toc_eta + timedelta(hours=approx_cruise_time_hrs) # Very rough estimate - Replaced by above logic

    # The arrival track is the track from the origin to the destination
    track = calculate_track(route_fixes[0], route_fixes[-1])

    dist_desc_req = estimate_required_descent_distance(
        cruise_alt, final_alt, route_fixes[-1].location, estimated_final_eta, wind_model, performance, track
    )

    # Iterate backwards from destination to find TOD segment
    cumulative_distance_from_end = 0.0
    tod_fix = None

    # Iterate backwards over segments of the route *including* the inserted TOC
    for i in range(len(route_fixes) - 2, toc_inserted_idx -1, -1): # Go down to the segment starting at TOC
        segment_start_fix = route_fixes[i]
        segment_end_fix = route_fixes[i+1]
        segment_distance = calculate_distance(segment_start_fix, segment_end_fix)

        if cumulative_distance_from_end + segment_distance >= dist_desc_req:
            # TOD falls within this segment (i -> i+1)
            distance_needed_on_segment = dist_desc_req - cumulative_distance_from_end
            # Distance from segment start (fix i) to TOD point
            distance_from_segment_start = segment_distance - distance_needed_on_segment

            tod_coords = point_along_path(segment_start_fix, segment_end_fix, distance_from_segment_start)
            # TOD inherits cruise altitude as its "constraint" signifying start of descent
            tod_fix = Fix("TOD", tod_coords[0], tod_coords[1], altitude_constraint=cruise_alt)
            tod_inserted_idx = i + 1 # Insert *after* fix i
            route_fixes.insert(tod_inserted_idx, tod_fix)
            break # Found TOD, exit loop

        cumulative_distance_from_end += segment_distance

    # Handle case where TOD wasn't found (e.g., route too short for cruise/descent)
    if tod_fix is None:
       # This might happen if the required descent distance is larger than the distance
       # between TOC and the destination. In this scenario, descent starts immediately at TOC.
       print("Warning: Route segment after TOC might be too short for cruise phase. Setting TOD at TOC.")
       # Find the TOC fix we inserted earlier and mark it as TOD as well.
       # This requires careful handling as we might have inserted TOC already.
       # Let's assume for now that if TOD isn't found, descent starts at TOC.
       # We will use toc_inserted_idx as the effective start of descent.
       tod_inserted_idx = toc_inserted_idx # Effectively TOD=TOC
       # Update the name/properties of the existing TOC fix if needed, or just use the index
       route_fixes[toc_inserted_idx].name = "TOC/TOD" # Indicate it serves both roles


    # --- 3. Sequential ETA Calculation over the NEW route ---
    eta = {}
    phase = {}
    altitude = {} # Store altitude at each fix for better calculations

    # Initialize first fix
    eta[0] = actual_takeoff_time
    phase[0] = "TakeOff"
    altitude[0] = 0.0 # Assuming starting at sea level or airport elevation

    # Loop through the segments of the modified route
    for i in range(len(route_fixes) - 1):
        fix_start = route_fixes[i]
        fix_end = route_fixes[i+1]

        # Determine phase for the *start* of the segment
        current_phase = ""
        if i < toc_inserted_idx:
            current_phase = "Climb"
        elif tod_inserted_idx != -1 and i < tod_inserted_idx:
             current_phase = "Cruise"
        elif tod_inserted_idx != -1: # i >= tod_inserted_idx
             current_phase = "Descent"
        else: # Should not happen if TOD logic is correct, fallback to cruise
             current_phase = "Cruise"


        # Determine altitude and TAS for the segment
        # Note: Altitude/TAS should ideally be calculated *at* the fix or averaged over the segment
        alt_start = altitude[i]
        tas_segment = 0.0

        if current_phase == "Climb":
            tas_segment = get_climb_tas(alt_start, performance)
            # Estimate altitude at end of segment (simplistic linear climb based on time)
            # A better way would integrate climb rate over time/distance
        elif current_phase == "Cruise":
            alt_start = cruise_alt # Ensure altitude is cruise
            tas_segment = cruise_tas
        elif current_phase == "Descent":
             # Use altitude from previous step if available, otherwise estimate based on time from TOD
             if i == tod_inserted_idx: # First descent segment
                 alt_start = cruise_alt
             # else alt_start is altitude[i] calculated previously
             tas_segment = get_descent_tas(alt_start, performance)

        # Simplified segment calculation (as in previous versions)
        distance_segment = calculate_distance(fix_start, fix_end)
        if distance_segment < 1e-6: # Skip zero-length segments if fixes coincided
            eta[i+1] = eta[i]
            altitude[i+1] = altitude[i] # Carry over altitude
            # Determine phase for the end fix
            if route_fixes[i+1].name in ["TOC", "TOC/TOD"]: phase[i+1] = "ClimbEnd" # Or just TOC
            elif route_fixes[i+1].name == "TOD": phase[i+1] = "DescentStart" # Or just TOD
            elif i+1 == len(route_fixes) - 1: phase[i+1] = "Arrival"
            else: phase[i+1] = phase[i] # Carry over phase
            continue

        # Mid-segment time/location for wind (approximation)
        delta_t_guess = distance_segment / tas_segment if tas_segment > 0 else 0
        t_mid = eta[i] + timedelta(hours=0.5 * delta_t_guess)
        # Use start fix location for wind lookup (simplification)
        wind_vector = wind_model(fix_start.location, alt_start, t_mid)
        track = calculate_track(fix_start, fix_end)
        wind_comp = calculate_head_tailwind(wind_vector, track)

        gs_segment = tas_segment + wind_comp
        if gs_segment <= 0:
            print(f"Warning: Non-positive GS calculated for segment {fix_start.name} -> {fix_end.name}. Using 5 kts.")
            gs_segment = 5.0

        delta_t_hours = distance_segment / gs_segment
        eta[i+1] = eta[i] + timedelta(hours=delta_t_hours)

        # Estimate altitude at the end of the segment
        if current_phase == "Climb":
             # Estimate altitude gained based on time and climb rate
             alt_gain = performance.standard_climb_rate * (delta_t_hours * 60.0)
             altitude[i+1] = min(alt_start + alt_gain, cruise_alt) # Cap at cruise alt
        elif current_phase == "Cruise":
             altitude[i+1] = cruise_alt
        elif current_phase == "Descent":
             # Estimate altitude lost based on time and descent rate
             alt_lost = performance.standard_descent_rate * (delta_t_hours * 60.0)
             altitude[i+1] = max(alt_start - alt_lost, final_alt if i + 1 == len(route_fixes) - 1 else 0) # Ensure non-negative, aim for final_alt at destination

        # Set phase for the *end* fix (i+1)
        if route_fixes[i+1].name in ["TOC", "TOC/TOD"]: phase[i+1] = "TOC"
        elif route_fixes[i+1].name == "TOD": phase[i+1] = "TOD"
        elif i+1 == len(route_fixes) - 1: phase[i+1] = "Arrival"
        else: # Assign phase based on position relative to inserted points
             if i+1 < toc_inserted_idx: phase[i+1] = "Climb"
             elif tod_inserted_idx != -1 and i+1 < tod_inserted_idx: phase[i+1] = "Cruise"
             elif tod_inserted_idx != -1: phase[i+1] = "Descent"
             else: phase[i+1] = "Cruise" # Fallback

    # --- 4. Compile Results ---
    # Map results back to fix names for clarity
    eta_by_name = {fix.name: eta[idx] for idx, fix in enumerate(route_fixes)}
    phase_by_name = {fix.name: phase[idx] for idx, fix in enumerate(route_fixes)}
    altitude_by_name = {fix.name: altitude[idx] for idx, fix in enumerate(route_fixes)}


    result = {
        "route_fixes": route_fixes, # The modified list of Fix objects
        "eta_by_name": eta_by_name,
        "phase_by_name": phase_by_name,
        "altitude_by_name": altitude_by_name, # Include calculated altitudes
        "toc_inserted_idx": toc_inserted_idx,
        "tod_inserted_idx": tod_inserted_idx,
        "final_eta": eta[len(route_fixes)-1]
    }

    return result

def point_along_path(fix1: Fix, fix2: Fix, distance_from_fix1: float) -> Tuple[float, float]:
    """
    Calculates the coordinates of a point along the great circle path between two fixes.

    Args:
        fix1: Starting fix.
        fix2: Ending fix.
        distance_from_fix1: Distance from fix1 along the path in nautical miles.

    Returns:
        Tuple of (latitude, longitude) in degrees for the point.
    """
    R = 3440.065  # Earth radius in nautical miles

    lat1 = math.radians(fix1.lat)
    lon1 = math.radians(fix1.lon)

    # Check for zero distance
    if distance_from_fix1 == 0:
        return fix1.lat, fix1.lon

    # Calculate bearing (track) from fix1 to fix2
    bearing = math.radians(calculate_track(fix1, fix2))
    
    # Check if fixes are identical
    if calculate_distance(fix1, fix2) < 1e-6: # Use a small threshold for floating point comparison
        # Cannot determine bearing if points are the same, return fix1 coords
        print(f"Warning: Attempting to find point along path between identical fixes: {fix1.name} and {fix2.name}. Returning fix1 coordinates.")
        return fix1.lat, fix1.lon

    d = distance_from_fix1 / R  # Angular distance

    lat2 = math.asin(math.sin(lat1) * math.cos(d) +
                     math.cos(lat1) * math.sin(d) * math.cos(bearing))

    lon2 = lon1 + math.atan2(math.sin(bearing) * math.sin(d) * math.cos(lat1),
                             math.cos(d) - math.sin(lat1) * math.sin(lat2))

    return math.degrees(lat2), math.degrees(lon2)

# Example usage:
"""
def example_wind_model(location, altitude, time):
    # Simplified wind model that returns a constant wind
    # In a real implementation, this would query a weather forecasting model
    return WindVector(speed=20.0, direction=270.0)  # 20 knots from the west

# Create performance model
performance = AircraftPerformance()

# Create route
fixes = [
    Fix("KSFO", 37.619, -122.375),           # San Francisco
    Fix("KSCK", 37.894, -121.238),           # Stockton
    Fix("KFAT", 36.776, -119.718),           # Fresno
    Fix("KLAS", 36.080, -115.153)            # Las Vegas
]

# Calculate ETA
takeoff_time = datetime(2023, 6, 1, 10, 0, 0)  # 10:00 AM
cruise_alt = 35000.0  # feet
cruise_tas = 450.0  # knots
final_alt = 1500.0  # feet

result = calculate_eta(
    fixes=fixes,
    takeoff_time=takeoff_time,
    cruise_alt=cruise_alt,
    cruise_tas=cruise_tas,
    final_alt=final_alt,
    performance=performance,
    wind_model=example_wind_model
)

# Print results
for i in range(len(fixes)):
    print(f"Fix {i} ({fixes[i].name}): ETA = {result['eta_by_name'][fixes[i].name]}, Phase = {result['phase_by_name'][fixes[i].name]}")

print(f"Final ETA at {fixes[-1].name}: {result['final_eta']}")
print(f"TOC at fix index: {result['toc_inserted_idx']}")
print(f"TOD at fix index: {result['tod_inserted_idx']}")
""" 