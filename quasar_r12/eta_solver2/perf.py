import math
from typing import Dict, Optional

# Define the AircraftPerformance class as provided by the user
class AircraftPerformance:
    """Simplified aircraft performance parameters."""
    def __init__(self, 
                 standard_climb_rate: float = 2000.0,  # ft/min
                 standard_descent_rate: float = 1800.0,  # ft/min
                 descent_angle: float = 3.0,  # degrees
                 climb_speeds: Optional[Dict[float, float]] = None, # TAS in knots, keys are altitudes in ft
                 descent_speeds: Optional[Dict[float, float]] = None): # TAS in knots, keys are altitudes in ft
        """
        Args:
            standard_climb_rate: Standard rate of climb in ft/min
            standard_descent_rate: Standard rate of descent in ft/min 
                                   (Note: this is not directly used by descent_time_estimator unless 
                                    passed as fixed_vs, as per problem statement).
            descent_angle: Standard descent angle in degrees.
            climb_speeds: Dictionary of altitude (ft) to climb speeds (TAS in knots). 
                          Key is the lower bound of the altitude band for that speed.
            descent_speeds: Dictionary of altitude (ft) to descent speeds (TAS in knots).
                            Key is the lower bound of the altitude band for that speed.
        """
        self.standard_climb_rate = standard_climb_rate
        self.standard_descent_rate = standard_descent_rate 
        self.descent_angle = descent_angle
        
        self.climb_speeds = climb_speeds if climb_speeds is not None else {
            0: 180,      # From 0 ft up to (but not including) 10000 ft
            10000: 250,  # From 10000 ft up to (but not including) 30000 ft
            30000: 280,  # From 30000 ft up to (but not including) 99999 ft
            99999: 300   # From 99999 ft upwards
        }
        
        self.descent_speeds = descent_speeds if descent_speeds is not None else {
            0: 180,    
            10000: 250, 
            30000: 280, 
            99999: 300  
        }

# Constants
KNOTS_TO_FT_PER_MIN = 6076.11548556 / 60.0  # 1 NM = 6076.11548556 ft (standard international)

def _get_speed_at_altitude(altitude: float, speed_profile: Dict[float, float]) -> float:
    """
    Gets the TAS for the given altitude based on the speed profile.
    The keys in speed_profile are the lower bounds of altitude bands (ft).
    The speed associated with a key `k` applies for altitudes `h` where `k <= h < next_k`.
    If altitude is higher than the highest key, the speed for the highest key is used.
    If altitude is below the lowest key, it uses the speed of the lowest key.
    """
    if not speed_profile:
        # This case should ideally be prevented by AircraftPerformance defaults
        raise ValueError("Speed profile cannot be empty.")
    
    sorted_altitudes = sorted(speed_profile.keys())
    
    # If altitude is below the lowest defined threshold (e.g. profile starts at 0, alt is -50)
    # use the speed of the lowest threshold.
    if altitude < sorted_altitudes[0]:
        return speed_profile[sorted_altitudes[0]]

    applicable_speed = speed_profile[sorted_altitudes[0]] # Default to speed of lowest band
    for threshold_alt in sorted_altitudes:
        if altitude >= threshold_alt:
            applicable_speed = speed_profile[threshold_alt]
        else:
            # Altitude is below current threshold, so the previously set speed is correct.
            break
    return applicable_speed

def climb_time_estimator(performance: AircraftPerformance, 
                         cruise_altitude: float, 
                         origin_elevation: float) -> float:
    """
    Estimates the climbing time from origin_elevation to cruise_altitude.

    Args:
        performance: AircraftPerformance object.
        cruise_altitude: The target cruise altitude in feet.
        origin_elevation: The starting elevation of the climb in feet.

    Returns:
        Estimated climb time in minutes. Returns 0.0 if cruise_altitude <= origin_elevation.
        Returns float('inf') if climb rate is zero or negative.
    """
    if cruise_altitude <= origin_elevation:
        return 0.0

    if performance.standard_climb_rate <= 0:
        return float('inf')

    total_climb_time_minutes = 0.0
    current_altitude = origin_elevation
    
    sorted_band_lower_bounds = sorted(performance.climb_speeds.keys())

    while current_altitude < cruise_altitude:
        # TAS for the current altitude is determined by _get_speed_at_altitude.
        # This speed is applicable from current_altitude up to the start of the next speed band
        # or cruise_altitude, whichever is lower.
        # Note: TAS itself doesn't directly influence climb time if ROC is constant,
        # but the problem states "Use the correct climbing TAS given in the performance".
        # This implies it might be used for something else later (e.g. distance) or is just for completeness.
        # For time calculation with a fixed ROC, TAS is not used.
        # However, if ROC were dependent on TAS or power (which depends on TAS and altitude), 
        # then it would be crucial. The problem implies fixed ROC segments.
        # For this implementation, TAS is fetched but not used in the formula as ROC is given.

        # _tas_knots = _get_speed_at_altitude(current_altitude, performance.climb_speeds) # Fetched but not directly used for time

        next_band_start_alt = float('inf')
        for band_alt in sorted_band_lower_bounds:
            if band_alt > current_altitude: # Find the next altitude where speed *might* change
                next_band_start_alt = band_alt
                break
        
        climb_to_altitude_in_segment = min(cruise_altitude, next_band_start_alt)
        
        altitude_delta_in_segment = climb_to_altitude_in_segment - current_altitude
        
        if altitude_delta_in_segment <= 0: 
            # Safeguard: should not happen if current_altitude < cruise_altitude initially
            break 
            
        time_in_segment_minutes = altitude_delta_in_segment / performance.standard_climb_rate
        total_climb_time_minutes += time_in_segment_minutes
        
        current_altitude = climb_to_altitude_in_segment
        
    return total_climb_time_minutes

def descent_time_estimator(performance: AircraftPerformance,
                           cruise_altitude: float,
                           destination_elevation: float,
                           fixed_vs: Optional[float] = None) -> float:
    """
    Estimates the descending time from cruise_altitude to destination_elevation.

    Args:
        performance: AircraftPerformance object.
        cruise_altitude: The starting altitude of the descent in feet.
        destination_elevation: The target elevation of the descent in feet.
        fixed_vs: Optional. If provided, this fixed vertical speed (ft/min, positive value expected) 
                  is used. If None, a glide path angle from performance object is used.

    Returns:
        Estimated descent time in minutes. Returns 0.0 if cruise_altitude <= destination_elevation.
        Returns float('inf') if rate of descent is effectively zero or negative.
    """
    if cruise_altitude <= destination_elevation:
        return 0.0

    total_descent_time_minutes = 0.0
    current_altitude = cruise_altitude

    sorted_band_lower_bounds = sorted(performance.descent_speeds.keys())
    if not sorted_band_lower_bounds:
        raise ValueError("Descent speed profile is empty.") # Should be handled by AircraftPerformance defaults

    while current_altitude > destination_elevation:
        # For descent, the speed to use is for the band we are *entering* from above.
        # If current_altitude is exactly on a band boundary (e.g., 10000ft), 
        # _get_speed_at_altitude(10000, ...) would give speed for the band [10000, next_higher_band).
        # We need the speed for the band *below* 10000ft. So, we lookup speed for (current_altitude - epsilon).
        # The epsilon helps select the band corresponding to the segment we are *about to traverse downwards*.
        effective_altitude_for_speed_lookup = current_altitude - 0.001 
        
        # Ensure lookup altitude doesn't go below destination if we are very close
        if effective_altitude_for_speed_lookup < destination_elevation:
             effective_altitude_for_speed_lookup = destination_elevation

        tas_knots = _get_speed_at_altitude(effective_altitude_for_speed_lookup, performance.descent_speeds)

        # Determine the lower boundary (floor) of the altitude band for which 'tas_knots' is applicable.
        # 'tas_knots' is speed_profile[k] where 'k' is the largest threshold <= effective_altitude_for_speed_lookup.
        # This 'k' is our true_band_floor_for_tas.
        true_band_floor_for_tas = -1.0 # Initialize, should always be updated
        if effective_altitude_for_speed_lookup < sorted_band_lower_bounds[0]:
            true_band_floor_for_tas = sorted_band_lower_bounds[0]
        else:
            for k_alt in sorted_band_lower_bounds: 
                if effective_altitude_for_speed_lookup >= k_alt:
                    true_band_floor_for_tas = k_alt
                else:
                    break 
        
        # Sanity check for true_band_floor_for_tas (should be set if sorted_band_lower_bounds is not empty)
        if true_band_floor_for_tas == -1.0 and sorted_band_lower_bounds: # Should not happen with current logic
             true_band_floor_for_tas = sorted_band_lower_bounds[0]


        segment_top_alt = current_altitude
        # We descend using 'tas_knots' down to 'true_band_floor_for_tas', 
        # or 'destination_elevation' if it's higher than 'true_band_floor_for_tas'.
        segment_bottom_alt = max(destination_elevation, true_band_floor_for_tas)
        
        # If segment_top_alt is already at or below segment_bottom_alt, further descent in this manner isn't needed.
        # This can happen if current_altitude is already at/below destination_elevation (handled by loop)
        # or if segment logic makes current_altitude equal to segment_bottom_alt (e.g. rounding, or already in lowest band).
        if segment_top_alt <= segment_bottom_alt :
             break 

        altitude_delta_in_segment = segment_top_alt - segment_bottom_alt
        
        if altitude_delta_in_segment <= 0: # Should be largely caught by above, but acts as a final safeguard.
            break

        rate_of_descent_fpm = 0.0
        if fixed_vs is not None:
            if fixed_vs <= 0: # Must be positive
                return float('inf') 
            rate_of_descent_fpm = fixed_vs
        else: # Use glide path angle
            if performance.descent_angle <= 0: # Angle must be positive for descent
                return float('inf')
            if tas_knots <= 0: # TAS must be positive for ROD calculation from angle
                 return float('inf')

            tas_fpm = tas_knots * KNOTS_TO_FT_PER_MIN
            rate_of_descent_fpm = tas_fpm * math.sin(math.radians(performance.descent_angle))
            
            if rate_of_descent_fpm <= 0: # Calculated ROD must be positive
                return float('inf')

        time_in_segment_minutes = altitude_delta_in_segment / rate_of_descent_fpm
        total_descent_time_minutes += time_in_segment_minutes
        
        current_altitude = segment_bottom_alt
        
    return total_descent_time_minutes