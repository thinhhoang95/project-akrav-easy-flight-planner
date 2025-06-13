from eta_solver import calculate_eta, Fix, AircraftPerformance, WindVector
from example_eta import example_wind_model
from eta_uncertainty import calculate_eta_uncertainty
from datetime import datetime

# Setup from eta_solver example
performance = AircraftPerformance()
fixes = [
    Fix("KSFO", 37.619, -122.375), Fix("KSCK", 37.894, -121.238),
    Fix("KFAT", 36.776, -119.718), Fix("KLAS", 36.080, -115.153)
]
takeoff_time = datetime(2023, 6, 1, 10, 0, 0)
cruise_alt = 35000.0
cruise_tas = 450.0
final_alt = 1500.0

# 1. Calculate base ETAs
eta_results = calculate_eta(
    fixes=fixes, takeoff_time=takeoff_time, cruise_alt=cruise_alt,
    cruise_tas=cruise_tas, final_alt=final_alt,
    performance=performance, wind_model=example_wind_model, taxi_out_time=0 # Set taxi to 0 for consistency with derivation
)

if "error" not in eta_results:
    # 2. Calculate ETA uncertainty
    uncertainty_params = {
        "sigma_tas_climb_sq": 12.0**2,
        "sigma_tas_cruise_sq": 18.0**2,
        "sigma_tas_descent_sq": 15.0**2,
        "k_wind": 90.0 # 5.0**2 / 1.0
    }
    eta_std_devs = calculate_eta_uncertainty(
        eta_results=eta_results,
        performance=performance,
        wind_model=example_wind_model, # Need wind model again for GS calc
        cruise_tas=cruise_tas,
        **uncertainty_params
    )

    # 3. Print results
    print("\nETA Results with Uncertainty:")
    for fix in eta_results["route_fixes"]:
        name = fix.name
        eta = eta_results["eta_by_name"].get(name)
        phase = eta_results["phase_by_name"].get(name)
        altitude = eta_results["altitude_by_name"].get(name)
        std_dev_min = eta_std_devs.get(name)

        print(f"Fix: {name:<10} Phase: {phase:<12} Altitude: {altitude:>6.0f} ft   ETA: {eta}   StdDev: {std_dev_min:.2f} min")

else:
    print(f"Error calculating ETAs: {eta_results['error']}")