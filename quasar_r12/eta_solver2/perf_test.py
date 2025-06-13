from perf import AircraftPerformance, climb_time_estimator, descent_time_estimator

a320 = AircraftPerformance()

print(climb_time_estimator(a320, 33000, 0))
print(descent_time_estimator(a320, 33000, 0))
