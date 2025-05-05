from datetime import datetime
from eta_solver import Fix, WindVector, AircraftPerformance, calculate_eta
from eta_solver import (
    calculate_eta, Fix, WindVector, AircraftPerformance, # Make sure calculate_eta is imported
    # Import other necessary functions if they are not in the same file
    # e.g., calculate_distance, calculate_track, point_along_path etc. 
    # OR ensure they are defined within eta_solver.py
)
from datetime import datetime

# Example Wind Model (keep as is or use your actual model)
def example_wind_model(location, altitude, time):
    """Simplified wind model."""
    # Example: Wind from the west (270 degrees) at 30 knots, decreasing with altitude
    speed = max(0, 30.0 * (1 - altitude / 50000.0)) 
    direction = 270.0 
    # Add some time variability if desired
    # speed += 5 * math.sin(time.hour * math.pi / 12) 
    return WindVector(speed=speed, direction=direction)

# Example Usage
if __name__ == "__main__":
    # Create performance model
    performance = AircraftPerformance(
        standard_climb_rate=1800,  # ft/min
        standard_descent_rate=1500, # ft/min
        descent_angle=3.0
        # Add custom speeds if needed
    )

    # Create original route
    original_fixes = [
        Fix("KSFO", 37.619, -122.375),           # San Francisco
        Fix("PYE", 37.779, -121.819),            # Point Reyes VOR (example intermediate)
        # Fix("KSCK", 37.894, -121.238),           # Stockton (commented out for a shorter route example)
        Fix("KFAT", 36.776, -119.718),           # Fresno
        Fix("BTY", 36.300, -117.500),            # Beatty VOR (example intermediate)
        Fix("KLAS", 36.080, -115.153)            # Las Vegas
    ]

    # Define flight parameters
    takeoff_time = datetime(2024, 7, 26, 10, 0, 0)  # Example: 10:00 AM UTC
    cruise_alt = 33000.0  # feet
    cruise_tas = 440.0  # knots
    final_alt = 2200.0  # feet (approx. KLAS field elevation)
    taxi_out_time = 10.0 # minutes

    # Calculate ETA using the revised function
    result = calculate_eta(
        fixes=original_fixes, # Pass the original fixes list
        takeoff_time=takeoff_time,
        cruise_alt=cruise_alt,
        cruise_tas=cruise_tas,
        final_alt=final_alt,
        performance=performance,
        wind_model=example_wind_model,
        taxi_out_time=taxi_out_time
    )

    # Check for errors first
    if "error" in result:
        print(f"Error calculating ETA: {result['error']}")
    else:
        print("===== ETA RESULTS =====")
        print(f"{'Index':<8} {'Fix':<8} {'ETA (UTC)':<20} {'Phase':<12} {'Altitude (ft)':<15} {'Marker'}")
        print("-" * 75)
        
        # Use the modified route_fixes from the result
        route_fixes = result['route_fixes'] 
        eta_by_name = result['eta_by_name']
        phase_by_name = result['phase_by_name']
        altitude_by_name = result['altitude_by_name']

        for i, fix in enumerate(route_fixes):
            fix_name = fix.name
            # Handle potential missing keys if calculation failed for a fix
            eta_str = eta_by_name.get(fix_name, None)
            phase = phase_by_name.get(fix_name, "N/A")
            altitude = altitude_by_name.get(fix_name, float('nan'))

            eta_formatted = eta_str.strftime("%Y-%m-%d %H:%M:%S") if eta_str else "N/A"
            altitude_formatted = f"{altitude:.0f}"

            # Identify TOC/TOD by name
            marker = ""
            if fix_name == "TOC": marker = "(TOC)"
            elif fix_name == "TOD": marker = "(TOD)"
            elif fix_name == "TOC/TOD": marker = "(TOC/TOD)"
            
            print(f"{i:<8} {fix_name:<8} {eta_formatted:<20} {phase:<12} {altitude_formatted:<15} {marker}")

        print("\n===== ROUTE DETAILS =====")
        for i in range(len(route_fixes) - 1):
            from_fix = route_fixes[i]
            to_fix = route_fixes[i+1]
            
            from_fix_name = from_fix.name
            to_fix_name = to_fix.name

            # Calculate segment time using eta_by_name
            eta1 = eta_by_name.get(from_fix_name)
            eta2 = eta_by_name.get(to_fix_name)

            if eta1 and eta2:
                segment_time_sec = (eta2 - eta1).total_seconds()
                segment_time_min = segment_time_sec / 60.0
                # Calculate distance for the segment if needed (requires calculate_distance)
                # segment_dist = calculate_distance(from_fix, to_fix) 
                print(f"Segment {i} ({from_fix_name} to {to_fix_name}): {segment_time_min:.1f} minutes") # Add distance: {segment_dist:.1f} NM
            else:
                 print(f"Segment {i} ({from_fix_name} to {to_fix_name}): Time N/A")

        print(f"\nFinal ETA at {route_fixes[-1].name}: {result['final_eta'].strftime('%Y-%m-%d %H:%M:%S')}")
        # Optionally print the indices where TOC/TOD were inserted
        # print(f"TOC inserted at index: {result.get('toc_inserted_idx', 'N/A')}")
        # print(f"TOD inserted at index: {result.get('tod_inserted_idx', 'N/A')}")
