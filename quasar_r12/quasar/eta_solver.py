import math
from typing import List, Tuple, Optional, Dict

# Constants
EARTH_RADIUS_NM = 3440.065  # nautical miles
FT_PER_NM = 6076.12        # feet per nautical mile
P0 = 101325.0               # sea level standard pressure, Pa
T0 = 288.15                 # sea level standard temperature, K
L = 0.0065                  # temperature lapse rate, K/m
R = 287.05                  # specific gas constant for dry air, J/(kg·K)
G = 9.80665                 # gravitational acceleration, m/s^2
GAMMA = 1.4                 # heat capacity ratio


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in nautical miles."""
    φ1, λ1, φ2, λ2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dφ = φ2 - φ1
    dλ = λ2 - λ1
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return EARTH_RADIUS_NM * 2 * math.asin(math.sqrt(a))


def initial_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial course from point 1 to 2 in degrees."""
    φ1, λ1, φ2, λ2 = map(math.radians, (lat1, lon1, lat2, lon2))
    y = math.sin(λ2-λ1) * math.cos(φ2)
    x = math.cos(φ1)*math.sin(φ2) - math.sin(φ1)*math.cos(φ2)*math.cos(λ2-λ1)
    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360) % 360


def isa_temperature(alt_ft: float, isa_dev: float = 0.0) -> float:
    """Standard atmosphere temperature plus ISA deviation (K)."""
    alt_m = alt_ft * 0.3048
    T_std = T0 - L * alt_m
    return T_std + isa_dev


def isa_pressure(alt_ft: float) -> float:
    """Standard atmosphere pressure at altitude (Pa)."""
    alt_m = alt_ft * 0.3048
    T_std = T0 - L * alt_m
    return P0 * (T_std / T0) ** (G * 0.3048 / (R * L))


def tas_from_mach(mach: float, alt_ft: float, isa_dev: float = 0.0) -> float:
    """Convert Mach to TAS in knots."""
    T = isa_temperature(alt_ft, isa_dev)
    a = math.sqrt(GAMMA * R * T)  # speed of sound in m/s
    tas_ms = mach * a
    return tas_ms * 1.94384       # to knots


def tas_from_cas(cas_kt: float, alt_ft: float, isa_dev: float = 0.0) -> float:
    """Approximate conversion CAS->TAS in knots using ISA model."""
    p = isa_pressure(alt_ft)
    T = isa_temperature(alt_ft, isa_dev)
    rho = p / (R * T)
    rho0 = P0 / (R * T0)
    return cas_kt * math.sqrt(rho0 / rho)


def cruise_mach(ci: float) -> float:
    """Cruise Mach as linear function of Cost Index."""
    m = 0.76 + 0.00028 * ci
    return max(0.75, min(0.82, m))


def climb_speed_cas(ci: float) -> float:
    """Climb speed in CAS knots from Cost Index."""
    return min(330.0, 280.0 + 0.4 * (ci ** 0.3))


def descent_speed_cas(ci: float) -> float:
    """Descent speed in CAS knots from Cost Index."""
    return min(340.0, 300.0 + 0.3 * (ci ** 0.3))


def climb_distance_nm(alt_ft: float, gradient: float = 0.03) -> float:
    """Horizontal climb distance in nautical miles."""
    return (alt_ft / gradient) / FT_PER_NM


def descent_distance_nm(alt_ft: float, gradient: float = 0.04) -> float:
    """Horizontal descent distance in nautical miles."""
    return (alt_ft / gradient) / FT_PER_NM


def compute_waypoint_times(
    waypoints: List[Tuple[str, float, float]],
    ci: float,
    cruise_fl: int,
    wind_along: Optional[List[float]] = None,
    isa_dev: float = 0.0
) -> List[Dict[str, float]]:
    """
    Compute passing times (sec) over each waypoint.

    :param waypoints: list of (name, lat, lon)
    :param ci: Cost Index
    :param cruise_fl: Cruise flight level (e.g. 350 for FL350)
    :param wind_along: optional wind component along each leg in knots
    :param isa_dev: ISA temperature deviation (°C)
    :return: list of dicts {'name':..., 'time_s':..., 'phase':...}
    """
    n = len(waypoints)
    # Compute geometry
    dists = []
    bears = []
    for i in range(n-1):
        _, lat1, lon1 = waypoints[i]
        _, lat2, lon2 = waypoints[i+1]
        d = haversine(lat1, lon1, lat2, lon2)
        b = initial_bearing(lat1, lon1, lat2, lon2)
        dists.append(d)
        bears.append(b)
    # Wind default
    if wind_along is None:
        wind_along = [0.0] * (n-1)

    # Altitudes
    alt_cruise_ft = cruise_fl * 100.0

    # Distances for climb/descent
    d_climb = climb_distance_nm(alt_cruise_ft)
    d_descent = descent_distance_nm(alt_cruise_ft)
    total_route = sum(dists)
    if d_climb + d_descent > total_route:
        # simple single-segment climb-descent
        d_climb = total_route * (d_climb / (d_climb + d_descent))
        d_descent = total_route - d_climb

    # Find TOC and TOD indices and fractional positions
    cum = 0.0
    toc_idx = 0
    for i, d in enumerate(dists):
        if cum + d >= d_climb:
            toc_idx = i
            toc_frac = (d_climb - cum) / d
            break
        cum += d
    cum = 0.0
    tod_idx = n-2
    for j in range(n-2, -1, -1):
        d = dists[j]
        if cum + d >= d_descent:
            tod_idx = j
            tod_frac = 1.0 - (d_descent - cum) / d
            break
        cum += d

    # March through legs
    results = []
    t = 0.0  # seconds
    for i in range(n):
        name, lat, lon = waypoints[i]
        # Determine phase
        if i < toc_idx:
            phase = 'climb'
        elif i == toc_idx:
            # fractional check
            phase = 'climb' if toc_frac < 1.0 else 'cruise'
        elif i <= tod_idx:
            phase = 'cruise'
        else:
            phase = 'descent'
        # Record time at this wp
        results.append({'name': name, 'time_s': t, 'phase': phase})
        # Add next leg time
        if i < n-1:
            d_nm = dists[i]
            wind = wind_along[i]
            if phase == 'climb':
                cas = climb_speed_cas(ci)
                tas = tas_from_cas(cas, alt_cruise_ft * (i / n), isa_dev)
            elif phase == 'cruise':
                mach = cruise_mach(ci)
                tas = tas_from_mach(mach, alt_cruise_ft, isa_dev)
            else:  # descent
                cas = descent_speed_cas(ci)
                tas = tas_from_cas(cas, alt_cruise_ft * (i / n), isa_dev)
            vg = tas + wind
            # time for full leg
            dt_h = d_nm / vg
            t += dt_h * 3600.0
    return results


if __name__ == '__main__':
    # Example usage
    wpts = [
        ('DEP', 50.033333, 8.570556),  # e.g. FRA
        ('WPT1', 51.0, 9.0),
        ('WPT2', 52.0, 10.0),
        ('ARR', 53.421389, 10.394722)  # e.g. HAM
    ]
    times = compute_waypoint_times(wpts, ci=25, cruise_fl=370)
    for rec in times:
        print(f"{rec['name']}: {rec['time_s']/3600:.2f} h, phase={rec['phase']}")
