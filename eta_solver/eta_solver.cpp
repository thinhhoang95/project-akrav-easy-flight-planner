#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <vector>

namespace py = pybind11;

struct Waypoint {
    double lat; // degrees
    double lon; // degrees
    double alt; // feet (cruise flight level * 100)
};

// Constants
static constexpr double R_EARTH_NM = 3440.065; // Earth radius in nautical miles
static constexpr double DEG2RAD = M_PI/180.0;

// Convert degrees to radians
inline double deg2rad(double deg) {
    return deg * DEG2RAD;
}

// Great-circle distance (NM)
inline double haversine_nm(double lat1, double lon1, double lat2, double lon2) {
    double φ1 = deg2rad(lat1), λ1 = deg2rad(lon1);
    double φ2 = deg2rad(lat2), λ2 = deg2rad(lon2);
    double dφ = φ2 - φ1;
    double dλ = λ2 - λ1;
    double a = std::sin(dφ/2)*std::sin(dφ/2)
             + std::cos(φ1)*std::cos(φ2)
             * std::sin(dλ/2)*std::sin(dλ/2);
    return 2 * R_EARTH_NM * std::atan2(std::sqrt(a), std::sqrt(1-a));
}

// Initial bearing (degrees)
inline double initial_bearing(double lat1, double lon1, double lat2, double lon2) {
    double φ1 = deg2rad(lat1), φ2 = deg2rad(lat2);
    double Δλ = deg2rad(lon2 - lon1);
    double y = std::sin(Δλ) * std::cos(φ2);
    double x = std::cos(φ1)*std::sin(φ2)
             - std::sin(φ1)*std::cos(φ2)*std::cos(Δλ);
    double θ = std::atan2(y, x);
    double bearing = std::fmod((θ*180.0/M_PI) + 360.0, 360.0);
    return bearing;
}

// Convert Mach and altitude to TAS (knots), simple ISA model
inline double mach_to_tas(double mach, double alt_ft) {
    // Speed of sound a = sqrt(gamma*R*T). Approx a = 38.94*sqrt(T) (knots)
    // ISA: T = 288.15 - 0.0019812 * alt_ft (approx)
    double T = 288.15 - 0.0019812 * alt_ft;
    double a = 38.94 * std::sqrt(T/288.15);
    return mach * a;
}

// IAS to TAS conversion at altitude
inline double ias_to_tas(double ias_kt, double alt_ft) {
    // approximate: TAS ≈ IAS * sqrt(rho0 / rho)
    // rho0/rho ≈ (T/288.15)^{-4.255}
    double T = 288.15 - 0.0019812 * alt_ft;
    double factor = std::pow(T/288.15, -4.255);
    return ias_kt * std::sqrt(factor);
}

// Cost Index based speed schedules
inline double cruise_mach(double ci) {
    double m = 0.76 + 0.00028 * ci;
    return std::min(std::max(m, 0.75), 0.82);
}
inline double climb_ias(double ci) {
    double v = 280.0 + 0.4 * std::pow(ci, 0.3);
    return std::min(v, 330.0);
}
inline double descent_ias(double ci) {
    double v = 300.0 + 0.3 * std::pow(ci, 0.3);
    return std::min(v, 340.0);
}

// Main ETA solver
auto eta_solver(const std::vector<Waypoint>& wpts,
                double ci,
                int cruise_fl,
                const std::vector<double>& wind) {
    size_t n = wpts.size();
    std::vector<double> dist(n-1), brg(n-1);
    
    // Precompute leg distances and bearings
    for (size_t i = 0; i < n-1; ++i) {
        dist[i] = haversine_nm(wpts[i].lat, wpts[i].lon,
                               wpts[i+1].lat, wpts[i+1].lon);
        brg[i]  = initial_bearing(wpts[i].lat, wpts[i].lon,
                                  wpts[i+1].lat, wpts[i+1].lon);
    }

    // Climb/Descent distances
    double alt_ft = cruise_fl * 100.0;
    const double climb_grad = 0.03; // 3%
    const double desc_grad  = std::tan(4.0 * M_PI/180.0);
    double d_climb  = alt_ft / (climb_grad * 6076.12); // NM
    double d_descent= alt_ft / (desc_grad * 6076.12);

    // Locate TOC and TOD indices
    size_t toc_idx = 0;
    double acc = 0;
    for (; toc_idx < dist.size(); ++toc_idx) {
        acc += dist[toc_idx];
        if (acc >= d_climb) break;
    }
    size_t tod_idx = dist.size();
    acc = 0;
    for (size_t i = dist.size(); i-- > 0;) {
        acc += dist[i];
        if (acc >= d_descent) { tod_idx = i; break; }
    }

    // Iterate legs and compute times
    std::vector<double> etas(n);
    double t = 0.0;
    for (size_t i = 0; i < n-1; ++i) {
        double phase_speed_tas;
        if (i < toc_idx) {
            double ias = climb_ias(ci);
            phase_speed_tas = ias_to_tas(ias, wpts[i].alt);
        } else if (i >= tod_idx) {
            double ias = descent_ias(ci);
            phase_speed_tas = ias_to_tas(ias, wpts[i].alt);
        } else {
            double mach = cruise_mach(ci);
            phase_speed_tas = mach_to_tas(mach, alt_ft);
        }
        // wind component
        double w = (i < wind.size() ? wind[i] : 0.0);
        double ground_speed = phase_speed_tas + w;
        t += dist[i] / ground_speed * 3600.0; // seconds
        etas[i+1] = t;
    }
    return etas;
}

PYBIND11_MODULE(eta_solver, m) {
    py::class_<Waypoint>(m, "Waypoint")
        .def(py::init<double,double,double>())
        .def_readwrite("lat", &Waypoint::lat)
        .def_readwrite("lon", &Waypoint::lon)
        .def_readwrite("alt", &Waypoint::alt);

    m.doc() = "ETA solver with Cost Index for fast estimates";
    m.def("eta_solver", &eta_solver,
          py::arg("waypoints"),
          py::arg("ci"),
          py::arg("cruise_fl"),
          py::arg("wind") = std::vector<double>(),
          "Compute cumulative ETA (seconds) over each waypoint");
}
