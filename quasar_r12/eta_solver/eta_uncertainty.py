# path: quasar_r12/eta_solver/eta_uncertainty.py
import math
from datetime import datetime, timedelta
from typing import Dict, Callable, Optional

# Assuming eta_solver.py is in the same directory or accessible in the Python path
from eta_solver import (
    Fix, AircraftPerformance, WindVector,
    calculate_distance, calculate_track, calculate_head_tailwind,
    get_climb_tas, get_descent_tas
)

def _calculate_segment_ground_speed(
    fix_start: Fix,
    fix_end: Fix,
    eta_start: datetime,
    alt_start: float,
    phase: str,
    cruise_tas: float,
    performance: AircraftPerformance,
    wind_model: Callable,
) -> float:
    """
    Calculates the estimated ground speed for a single segment.
    Adapted from logic within calculate_eta.
    """
    tas_segment = 0.0
    if phase == "Climb":
        tas_segment = get_climb_tas(alt_start, performance)
    elif phase == "Cruise":
        tas_segment = cruise_tas
    elif phase == "Descent":
        tas_segment = get_descent_tas(alt_start, performance)
    else: # TakeOff, Arrival, TOC, TOD - determine based on next phase or use previous logic
        # This helper might be called for segments ending in TOC/TOD,
        # the phase indicates the segment *type*.
        # Use the TAS appropriate for the segment's *action*.
        # For simplicity, reuse phase logic: Climb uses climb TAS, Cruise cruise TAS etc.
        # A more robust approach might be needed if phases are complex.
        if alt_start < performance.climb_speeds[max(performance.climb_speeds.keys())]: # Simplistic check if still climbing
             tas_segment = get_climb_tas(alt_start, performance)
        else: # Assume cruise or descent logic applies based on altitude/context
             tas_segment = cruise_tas # Fallback, adjust if needed


    distance_segment = calculate_distance(fix_start, fix_end)
    if distance_segment < 1e-6:
        # Cannot calculate GS for zero-length segment, return a default?
        # The calling function should handle zero-length segments.
        return 0.0 # Or raise an error

    # Estimate mid-segment time for wind lookup
    # Approximate segment duration based on TAS only for wind lookup time
    delta_t_guess_hrs = distance_segment / tas_segment if tas_segment > 0 else 0
    t_mid = eta_start + timedelta(hours=0.5 * delta_t_guess_hrs)

    # Use start fix location and altitude for wind lookup (simplification)
    wind_vector = wind_model(fix_start.location, alt_start, t_mid)
    track = calculate_track(fix_start, fix_end)
    wind_comp = calculate_head_tailwind(wind_vector, track)

    gs_segment = tas_segment + wind_comp
    if gs_segment <= 0:
        # Use a small positive value to avoid division by zero later
        gs_segment = 5.0

    return gs_segment


def calculate_eta_uncertainty(
    eta_results: Dict,
    performance: AircraftPerformance,
    wind_model: Callable,
    cruise_tas: float,
    sigma_tas_climb_sq: float = 10.0**2,  # Example: std dev = 10 kts
    sigma_tas_cruise_sq: float = 15.0**2, # Example: std dev = 15 kts
    sigma_tas_descent_sq: float = 12.0**2,# Example: std dev = 12 kts
    k_wind: float = 5.0**2 / 1.0       # Example: wind vector std dev grows ~5 kts per hour -> variance rate ~ 25 kts^2/hr
) -> Dict[str, float]:
    """
    Calculates the standard deviation of the ETA error for each fix based on
    summing the variance contributions from each flight segment. This ensures
    monotonic increase in uncertainty.

    Args:
        eta_results: The dictionary returned by calculate_eta.
        performance: The AircraftPerformance object used.
        wind_model: The wind model callable used.
        cruise_tas: The cruise TAS used in the original calculation.
        sigma_tas_climb_sq: Variance of TAS error during climb (kts^2).
        sigma_tas_cruise_sq: Variance of TAS error during cruise (kts^2).
        sigma_tas_descent_sq: Variance of TAS error during descent (kts^2).
        k_wind: Rate of increase of wind component error variance (kts^2 / hour).

    Returns:
        Dictionary mapping fix names to the standard deviation of ETA error
        in minutes.
    """
    route_fixes = eta_results["route_fixes"]
    eta_by_name = eta_results["eta_by_name"]
    altitude_by_name = eta_results["altitude_by_name"]
    # phase_by_name = eta_results["phase_by_name"] # Phase at fix, less useful now
    toc_inserted_idx = eta_results["toc_inserted_idx"]
    tod_inserted_idx = eta_results["tod_inserted_idx"]

    eta_std_dev_minutes_by_name: Dict[str, float] = {}
    cumulative_variance_hours_sq = 0.0 # Initialize cumulative variance

    if not route_fixes:
        return eta_std_dev_minutes_by_name

    # Takeoff time and fix name
    takeoff_fix_name = route_fixes[0].name
    takeoff_time = eta_by_name[takeoff_fix_name]
    eta_std_dev_minutes_by_name[takeoff_fix_name] = 0.0 # No uncertainty at the start

    # Iterate through flight segments (from fix k-1 to fix k)
    for k in range(1, len(route_fixes)):
        fix_k = route_fixes[k]
        fix_k_name = fix_k.name
        fix_prev = route_fixes[k-1]
        fix_prev_name = fix_prev.name

        # Get times and altitudes for the segment endpoints
        eta_k_datetime = eta_by_name[fix_k_name]
        eta_prev_datetime = eta_by_name[fix_prev_name]
        alt_prev = altitude_by_name[fix_prev_name]

        # Calculate estimated segment duration (Delta ETA) in hours
        delta_eta_segment_hours = (eta_k_datetime - eta_prev_datetime).total_seconds() / 3600.0

        segment_time_variance_hours_sq = 0.0 # Variance contribution from this segment

        if delta_eta_segment_hours > 1e-9: # Only calculate variance if time progresses
            # Determine the phase of the segment (k-1 -> k)
            segment_phase = "Unknown"
            if k <= toc_inserted_idx:
                 segment_phase = "Climb"
            # Check if tod_inserted_idx valid and distinct before checking cruise
            elif tod_inserted_idx != -1 and tod_inserted_idx != toc_inserted_idx and k <= tod_inserted_idx:
                 segment_phase = "Cruise"
            elif tod_inserted_idx != -1: # After TOD (or at TOD if distinct)
                 segment_phase = "Descent"
            elif toc_inserted_idx != -1 and k > toc_inserted_idx:
                 # Fallback if TOD wasn't distinct
                 if tod_inserted_idx == toc_inserted_idx: # TOD=TOC case
                     segment_phase = "Descent"
                 else: # Should likely be cruise if TOD is later
                     segment_phase = "Cruise" # Default assumption

            # Get estimated ground speed for the segment
            gs_segment = _calculate_segment_ground_speed(
                fix_prev, fix_k, eta_prev_datetime, alt_prev, segment_phase,
                cruise_tas, performance, wind_model
            )

            if gs_segment > 1e-6: # Avoid division by zero if GS is negligible
                # Determine TAS variance component based on phase
                sigma_tas_sq = 0.0
                if segment_phase == "Climb":
                    sigma_tas_sq = sigma_tas_climb_sq
                elif segment_phase == "Cruise":
                    sigma_tas_sq = sigma_tas_cruise_sq
                elif segment_phase == "Descent":
                    sigma_tas_sq = sigma_tas_descent_sq
                # else: Use average or default? For now, assume 0 if phase unknown

                # Calculate midpoint time relative to takeoff (in hours)
                t_prev_hours = (eta_prev_datetime - takeoff_time).total_seconds() / 3600.0
                t_k_hours = (eta_k_datetime - takeoff_time).total_seconds() / 3600.0
                t_mid_hours = (t_prev_hours + t_k_hours) / 2.0

                # Calculate ground speed error variance for the segment
                sigma_gs_sq_segment = sigma_tas_sq + k_wind * t_mid_hours

                # Calculate variance added by this segment's time error
                # Var(Delta Error) approx (Delta ETA / GS)^2 * Var(Delta GS)
                segment_time_variance_hours_sq = (delta_eta_segment_hours / gs_segment)**2 * sigma_gs_sq_segment

        # Add segment variance to cumulative total
        cumulative_variance_hours_sq += segment_time_variance_hours_sq

        # Calculate cumulative standard deviation in hours, ensuring non-negative
        std_dev_k_hours = math.sqrt(max(0, cumulative_variance_hours_sq))

        # Convert to minutes
        std_dev_k_minutes = std_dev_k_hours * 60.0

        eta_std_dev_minutes_by_name[fix_k_name] = std_dev_k_minutes

    return eta_std_dev_minutes_by_name

# Example Usage (assuming you have results from calculate_eta)
"""
# --- This part would be in your main script ---
# from quasar_r12.eta_solver.eta_solver import calculate_eta, Fix, AircraftPerformance, WindVector, example_wind_model
# from quasar_r12.eta_solver.eta_uncertainty import calculate_eta_uncertainty
# from datetime import datetime

# # Setup from eta_solver example
# performance = AircraftPerformance()
# fixes = [
#     Fix("KSFO", 37.619, -122.375), Fix("KSCK", 37.894, -121.238),
#     Fix("KFAT", 36.776, -119.718), Fix("KLAS", 36.080, -115.153)
# ]
# takeoff_time = datetime(2023, 6, 1, 10, 0, 0)
# cruise_alt = 35000.0
# cruise_tas = 450.0
# final_alt = 1500.0

# # 1. Calculate base ETAs
# eta_results = calculate_eta(
#     fixes=fixes, takeoff_time=takeoff_time, cruise_alt=cruise_alt,
#     cruise_tas=cruise_tas, final_alt=final_alt,
#     performance=performance, wind_model=example_wind_model, taxi_out_time=0 # Set taxi to 0 for consistency with derivation
# )

# if "error" not in eta_results:
#     # 2. Calculate ETA uncertainty
#     uncertainty_params = {
#         "sigma_tas_climb_sq": 10.0**2,
#         "sigma_tas_cruise_sq": 15.0**2,
#         "sigma_tas_descent_sq": 12.0**2,
#         "k_wind": 25.0 # 5.0**2 / 1.0
#     }
#     eta_std_devs = calculate_eta_uncertainty(
#         eta_results=eta_results,
#         performance=performance,
#         wind_model=example_wind_model, # Need wind model again for GS calc
#         cruise_tas=cruise_tas,
#         **uncertainty_params
#     )

#     # 3. Print results
#     print("\nETA Results with Uncertainty:")
#     for fix in eta_results["route_fixes"]:
#         name = fix.name
#         eta = eta_results["eta_by_name"].get(name)
#         phase = eta_results["phase_by_name"].get(name)
#         altitude = eta_results["altitude_by_name"].get(name)
#         std_dev_min = eta_std_devs.get(name)

#         print(f"Fix: {name:<10} Phase: {phase:<12} Altitude: {altitude:>6.0f} ft   ETA: {eta}   StdDev: {std_dev_min:.2f} min")

# else:
#     print(f"Error calculating ETAs: {eta_results['error']}")

"""