import unittest
from datetime import datetime, timedelta
from eta_solver import (
    Fix, WindVector, AircraftPerformance, calculate_distance, calculate_track,
    calculate_head_tailwind, calculate_remaining_route_distance,
    get_climb_tas, get_descent_tas, estimate_required_descent_distance,
    calculate_current_descent_altitude, find_top_of_climb_fix, calculate_eta
)

class TestETASolver(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        # Create some test fixes
        self.fixes = [
            Fix("KSFO", 37.619, -122.375),          # San Francisco
            Fix("KSCK", 37.894, -121.238),          # Stockton
            Fix("KFAT", 36.776, -119.718),          # Fresno
            Fix("KLAS", 36.080, -115.153)           # Las Vegas
        ]
        
        # Create a test performance model
        self.performance = AircraftPerformance()
        
        # Test parameters
        self.takeoff_time = datetime(2023, 6, 1, 10, 0, 0)  # 10:00 AM
        self.cruise_alt = 35000.0  # feet
        self.cruise_tas = 450.0  # knots
        self.final_alt = 1500.0  # feet
        
        # Simple constant wind model for testing
        def test_wind_model(location, altitude, time):
            if altitude < 10000:
                # Light wind at lower altitudes
                return WindVector(speed=10.0, direction=270.0)
            else:
                # Stronger wind at higher altitudes
                return WindVector(speed=40.0, direction=270.0)
                
        self.wind_model = test_wind_model
    
    def test_calculate_distance(self):
        """Test the great circle distance calculation."""
        # SFO to LAS direct distance is approximately 414 nm
        distance = calculate_distance(self.fixes[0], self.fixes[3])
        self.assertAlmostEqual(distance, 414.0, delta=10.0)
        
        # Test known short distance
        fix1 = Fix("A", 37.0, -122.0)
        fix2 = Fix("B", 37.0, -122.1)  # 0.1 degree longitude difference at this latitude
        distance = calculate_distance(fix1, fix2)
        self.assertAlmostEqual(distance, 5.5, delta=0.5)  # Approx 5.5 nm
    
    def test_calculate_track(self):
        """Test the track calculation."""
        # Track from SFO to LAS is approximately east-northeast
        track = calculate_track(self.fixes[0], self.fixes[3])
        self.assertTrue(70.0 <= track <= 110.0)
        
        # Test known track (due east)
        fix1 = Fix("A", 37.0, -122.0)
        fix2 = Fix("B", 37.0, -121.0)  # Due east
        track = calculate_track(fix1, fix2)
        self.assertAlmostEqual(track, 90.0, delta=1.0)
    
    def test_calculate_head_tailwind(self):
        """Test the headwind/tailwind component calculation."""
        # Wind from west (270°), track to east (90°) = full tailwind
        wind = WindVector(speed=20.0, direction=270.0)
        track = 90.0
        wind_component = calculate_head_tailwind(wind, track)
        self.assertAlmostEqual(wind_component, 20.0, delta=0.1)
        
        # Wind from east (90°), track to east (90°) = full headwind
        wind = WindVector(speed=20.0, direction=90.0)
        wind_component = calculate_head_tailwind(wind, track)
        self.assertAlmostEqual(wind_component, -20.0, delta=0.1)
        
        # Wind from north (0°), track to east (90°) = no head/tailwind
        wind = WindVector(speed=20.0, direction=0.0)
        wind_component = calculate_head_tailwind(wind, track)
        self.assertAlmostEqual(wind_component, 0.0, delta=0.1)
    
    def test_calculate_remaining_route_distance(self):
        """Test calculation of remaining route distance."""
        # Total route distance from fixes[0] to fixes[3]
        total_distance = calculate_remaining_route_distance(self.fixes, 0, 3)
        
        # Manually calculate segment distances
        segment1 = calculate_distance(self.fixes[0], self.fixes[1])
        segment2 = calculate_distance(self.fixes[1], self.fixes[2])
        segment3 = calculate_distance(self.fixes[2], self.fixes[3])
        manual_total = segment1 + segment2 + segment3
        
        # Compare
        self.assertAlmostEqual(total_distance, manual_total, delta=0.1)
    
    def test_get_climb_descent_tas(self):
        """Test the true airspeed lookup for climb and descent."""
        # Test climb speeds at different altitudes
        self.assertEqual(get_climb_tas(0, self.performance), 180)
        self.assertEqual(get_climb_tas(5000, self.performance), 250)
        self.assertEqual(get_climb_tas(20000, self.performance), 280)
        self.assertEqual(get_climb_tas(40000, self.performance), 300)
        
        # Test descent speeds at different altitudes
        self.assertEqual(get_descent_tas(0, self.performance), 180)
        self.assertEqual(get_descent_tas(5000, self.performance), 250)
        self.assertEqual(get_descent_tas(20000, self.performance), 280)
        self.assertEqual(get_descent_tas(40000, self.performance), 300)
    
    def test_estimate_required_descent_distance(self):
        """Test the estimation of required descent distance."""
        # Calculate for a simple case
        cruise_alt = 30000.0
        final_alt = 0.0
        location = self.fixes[2].location
        time = self.takeoff_time + timedelta(hours=1)
        
        dist_req = estimate_required_descent_distance(
            cruise_alt, final_alt, location, time, self.wind_model, self.performance
        )
        
        # The formula is complex, but we can at least check it's reasonable
        # At 3 degrees descent angle, a 30000 ft descent should take around 100 nm
        self.assertTrue(80.0 <= dist_req <= 150.0)
    
    def test_calculate_current_descent_altitude(self):
        """Test calculation of current altitude during descent."""
        cruise_alt = 30000.0
        tod_time = self.takeoff_time + timedelta(hours=1)
        
        # Test at TOD
        current_time = tod_time
        altitude = calculate_current_descent_altitude(
            current_time, tod_time, cruise_alt, self.performance
        )
        self.assertAlmostEqual(altitude, cruise_alt, delta=1.0)
        
        # Test 10 minutes into descent
        current_time = tod_time + timedelta(minutes=10)
        altitude = calculate_current_descent_altitude(
            current_time, tod_time, cruise_alt, self.performance
        )
        expected_alt = cruise_alt - (10 * self.performance.standard_descent_rate)
        self.assertAlmostEqual(altitude, expected_alt, delta=1.0)
    
    def test_find_top_of_climb_fix(self):
        """Test finding the top of climb fix."""
        toc_idx, toc_eta = find_top_of_climb_fix(
            self.fixes, self.takeoff_time, self.cruise_alt, self.performance, self.wind_model
        )
        
        # Check index is valid
        self.assertTrue(0 <= toc_idx < len(self.fixes))
        
        # Check ETA is reasonable (17.5 minutes to climb to 35000 ft at 2000 ft/min)
        expected_min_time = self.takeoff_time + timedelta(minutes=self.cruise_alt / self.performance.standard_climb_rate)
        self.assertTrue(toc_eta >= expected_min_time)
    
    def test_calculate_eta(self):
        """Test the main ETA calculation function."""
        result = calculate_eta(
            fixes=self.fixes,
            takeoff_time=self.takeoff_time,
            cruise_alt=self.cruise_alt,
            cruise_tas=self.cruise_tas,
            final_alt=self.final_alt,
            performance=self.performance,
            wind_model=self.wind_model
        )
        
        # Check that the function returns expected keys
        self.assertIn("eta", result)
        self.assertIn("phase", result)
        self.assertIn("toc_fix_index", result)
        self.assertIn("tod_fix_index", result)
        self.assertIn("final_eta", result)
        
        # Check that we have an ETA for each fix
        self.assertEqual(len(result["eta"]), len(self.fixes))
        
        # Check that we have a phase for each fix
        self.assertEqual(len(result["phase"]), len(self.fixes))
        
        # Check that the phases are valid
        valid_phases = {"TakeOff", "Climb", "Cruise", "Descent", "Arrival"}
        for phase in result["phase"].values():
            self.assertIn(phase, valid_phases)
        
        # Check that the ETAs are sequential
        for i in range(1, len(self.fixes)):
            self.assertTrue(result["eta"][i] > result["eta"][i-1])
        
        # Check that the final ETA matches the ETA of the last fix
        self.assertEqual(result["final_eta"], result["eta"][len(self.fixes)-1])
        
        # Check TOC and TOD are within range
        self.assertTrue(0 <= result["toc_fix_index"] < len(self.fixes))
        if result["tod_fix_index"] != -1:  # If TOD was determined
            self.assertTrue(result["toc_fix_index"] <= result["tod_fix_index"] < len(self.fixes))

if __name__ == "__main__":
    unittest.main() 