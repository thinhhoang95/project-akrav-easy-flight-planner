"""
Multi-Phase Vertical Profile Optimal Control Problem
------------------------------------------------------

This script sets up and solves an OCP for a full flight profile with
the following phases:
  1. Take-off
  2. Climb (split into three segments with prescribed speeds)
  3. Cruise (level flight at cruise altitude)
  4. Descent (split into three segments, reverse of climb)
  5. Landing

The prescribed climb profile is:
   - Climb 1: 250 knots until 10,000 ft,
   - Climb 2: 300 knots until 20,000 ft,
   - Climb 3: Mach 0.78 until cruise altitude (35,000 ft).

The descent profile is the reverse: Mach 0.78, then 300 knots, then 250 knots.
Landing brings the aircraft to a low touchdown speed.
"""

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from openap import prop

# -------------------------
# 0. Load aircraft and engine data
# -------------------------
fiche_ac = prop.aircraft("A320")
fiche_engine = prop.engine("CFM56-5-A1")

# -------------------------
# 1. Conversion factors and target values
# -------------------------
knots_to_ms = 0.514444
ft_to_m     = 0.3048

# Speeds (m/s)
v_250 = 250 * knots_to_ms        # ~128.61 m/s
v_300 = 300 * knots_to_ms        # ~154.33 m/s
v_mach78 = 0.78 * 340            # ~265.2 m/s (approx.)

# Altitudes (m)
h_takeoff_end   = 1000 * ft_to_m    # 1,000 ft
h_climb1_end    = 10000 * ft_to_m     # 10,000 ft
h_climb2_end    = 20000 * ft_to_m     # 20,000 ft
h_cruise        = 35000 * ft_to_m     # 35,000 ft
h_descent3_end  = 100 * ft_to_m       # ~100 ft for final descent start
h_landing_end   = 0.0                 # ground

# -------------------------
# 2. Define the aircraft dynamics function (same as before)
# -------------------------
def aircraft_dynamics(x, u, wind_s, phi, phase, fiche_engine, fiche_ac, num_engines = 2):
    """
    Computes the state derivative for the vertical dynamics.
    x = [v, gamma, h, s, m]
    u = [T, n_z, beta]
    """
    g = 9.81  # gravitational acceleration (m/s^2)
    
    # Unpack state and control
    v, gamma, h, s, m = x[0], x[1], x[2], x[3], x[4] # speed, flight path angle, altitude, along-path distance, mass
    T, nz, beta = u[0], u[1], u[2] # thrust, load factor (L/W), configuration parameter
    
    # --- Drag Model ---
    rho0 = 1.225  # sea-level density [kg/m^3]
    H    = 8500.0 # scale height [m]
    rho  = rho0 * ca.exp(-h/H) # air density [kg/m^3] - exponential model

    # --- Lift/Drag Calculations ---
    # Required lift force to maintain load factor nz
    L = nz * m * g

    # To avoid division by zero when computing Cl, add a small epsilon to v**2.
    epsilon = 1e-6
    v_sq = v**2 + epsilon


    S_area = fiche_ac['wing']['area']  # wing area (example value) [m^2]

    # Compute the lift coefficient from L = 0.5 * rho * v^2 * S_area * Cl.
    Cl = L / (0.5 * rho * v_sq * S_area)
    
    k = fiche_ac['drag']['k']
    Cd = fiche_ac['drag']['cd0'] + beta + k * Cl**2  # parabolic drag model
    D = 0.5 * rho * v**2 * S_area * Cd  # Drag force [N]

    if phase == 'cruise':
        # --- Revised Fuel Flow Model ---
        # Convert engine cruise SFC from kg/(N·hr) to kg/(N·s)
        cruise_sfc = fiche_engine['cruise_sfc'] / 3600.0  # [kg/(N·s)]
        fuel_coef = fiche_ac['fuel']['fuel_coef']      # scaling coefficient
        fuel_flow = fuel_coef * cruise_sfc * T  # fuel flow [kg/s]
    elif phase == 'takeoff':
        fuel_flow = fiche_engine['ff_to'] * num_engines
    elif phase == 'climb1' or phase == 'climb2' or phase == 'climb3':
        fuel_flow = fiche_engine['ff_co'] * num_engines
    elif phase == 'descent1' or phase == 'descent2' or phase == 'descent3':
        fuel_flow = fiche_engine['ff_app'] * num_engines
    elif phase == 'landing':
        fuel_flow = fiche_engine['ff_idl'] * num_engines
    
    # Dynamics equations
    v_dot     = (T - D) / m - g * ca.sin(gamma)
    gamma_dot = (g / v) * (nz * ca.cos(phi) - ca.cos(gamma))
    h_dot     = v * ca.sin(gamma)
    s_dot     = v * ca.cos(gamma) + wind_s  # along-path distance, plus wind component
    m_dot     = -fuel_flow
    
    return ca.vertcat(v_dot, gamma_dot, h_dot, s_dot, m_dot)

# -------------------------
# 3. Phase definitions
# -------------------------
# Each phase is defined by:
#   - name: label for the phase
#   - N: number of discretization intervals in this phase
#   - h0: desired initial altitude (m) for the phase
#   - hF: desired final altitude (m) for the phase
#   - v_target: prescribed speed (m/s) if the phase has a fixed speed profile, or None.
#       (For phases with a constant altitude cruise, we can also enforce h constant.)
#   - Additional boundary conditions on v (if desired) can be set in the takeoff/landing phases.
phases = [
    {"name": "takeoff",   "N": 10, "h0": 0.0,          "hF": h_takeoff_end,  "v_target": None},   # v free; later we set v(0)=0 and v(end)=v_250
    {"name": "climb1",    "N": 10, "h0": h_takeoff_end, "hF": h_climb1_end,   "v_target": v_250},
    {"name": "climb2",    "N": 10, "h0": h_climb1_end,  "hF": h_climb2_end,   "v_target": v_300},
    {"name": "climb3",    "N": 10, "h0": h_climb2_end,  "hF": h_cruise,       "v_target": v_mach78},
    {"name": "cruise",    "N": 20, "h0": h_cruise,      "hF": h_cruise,       "v_target": v_mach78},  # cruise: constant altitude and speed
    {"name": "descent1",  "N": 10, "h0": h_cruise,      "hF": h_climb2_end,   "v_target": v_mach78},
    {"name": "descent2",  "N": 10, "h0": h_climb2_end,  "hF": h_climb1_end,   "v_target": v_300},
    {"name": "descent3",  "N": 10, "h0": h_climb1_end,  "hF": h_descent3_end, "v_target": v_250},
    {"name": "landing",   "N": 10, "h0": h_descent3_end,"hF": h_landing_end,  "v_target": None}    # landing: allow deceleration to a low touchdown speed
]

# -------------------------
# 4. Set up the Opti problem and common parameters
# -------------------------
opti = ca.Opti()

# Common parameters for all phases
wind_s = opti.parameter();  opti.set_value(wind_s, 0.0)  # assume zero wind for simplicity
phi    = opti.parameter();  opti.set_value(phi, 0.0)     # bank angle zero in vertical dynamics
CI     = opti.parameter();  opti.set_value(CI, 1.0)        # cost index weight

# We'll store the variables for each phase in lists:
X_list = []  # states for each phase
U_list = []  # controls for each phase
T_list = []  # phase durations (free final time for each phase)
N_list = []  # discretization intervals for each phase
phase_names = []

# -------------------------
# 5. Create decision variables for each phase
# -------------------------
# Here the state vector is: [v, gamma, h, s, m]
# We impose some boundary conditions within each phase (e.g., desired altitude at phase start/end)
for phase in phases:
    name = phase["name"]
    phase_names.append(name)
    N_phase = phase["N"]
    N_list.append(N_phase)
    
    # Phase duration is free (initial guess provided later)
    T_phase = opti.variable()
    opti.set_initial(T_phase, 100.0)  # initial guess (seconds)
    T_list.append(T_phase)
    dt = T_phase / N_phase  # time step for this phase
    
    # States: dimension 5 x (N_phase+1)
    X = opti.variable(5, N_phase+1)
    # Controls: dimension 3 x (N_phase)
    U = opti.variable(3, N_phase)
    
    # Save to lists
    X_list.append(X)
    U_list.append(U)
    
    # Dynamics (multiple shooting using explicit Euler)
    for k in range(N_phase):
        xk = X[:, k]
        uk = U[:, k]
        x_next = X[:, k+1]
        # Euler integration:
        x_next_euler = xk + dt * aircraft_dynamics(xk, uk, wind_s, phi, name, fiche_engine, fiche_ac)
        opti.subject_to(x_next == x_next_euler)
    
    # Impose altitude boundary conditions for the phase:
    #   h (state index 2) at start and end
    opti.subject_to(X[2, 0] == phase["h0"])
    opti.subject_to(X[2, -1] == phase["hF"])
    
    # If a phase is meant to be flown at a prescribed speed, add path constraints:
    if phase["v_target"] is not None:
        # For all nodes in this phase, enforce v = v_target (state index 0)
        for k in range(N_phase+1):
            opti.subject_to(X[0, k] == phase["v_target"])
    
    # For the cruise phase, we also fix altitude (h) at every node.
    if name == "cruise":
        for k in range(N_phase+1):
            opti.subject_to(X[2, k] == h_cruise)
    
    # (Optionally, one might add bounds on gamma, s, and mass.)
    # Here we add some reasonable bounds (same for every phase):
    for k in range(N_phase+1):
        # Speed bounds (m/s)
        opti.subject_to(X[0, k] >= 0)
        opti.subject_to(X[0, k] <= 350)
        # Flight path angle bounds (rad)
        opti.subject_to(X[1, k] >= -0.5)
        opti.subject_to(X[1, k] <= 0.5)
        # Altitude bounds (m)
        opti.subject_to(X[2, k] >= 0)
        opti.subject_to(X[2, k] <= 12000)
        # Along–path distance s is free (>=0)
        opti.subject_to(X[3, k] >= 0)
        # Mass bounds (kg)
        opti.subject_to(X[4, k] >= 50000)
        opti.subject_to(X[4, k] <= 90000)
    
    # Also add bounds on the controls for each interval:
    for k in range(N_phase):
        # Thrust bounds [N]
        opti.subject_to(U[0, k] >= 0)
        opti.subject_to(U[0, k] <= 100000)
        # Load factor bounds
        opti.subject_to(U[1, k] >= 1.0)
        opti.subject_to(U[1, k] <= 2.5)
        # Beta (e.g., speedbrake parameter) bounds
        opti.subject_to(U[2, k] >= 0.0)
        opti.subject_to(U[2, k] <= 0.5)

# -------------------------
# 6. Link the phases (continuity conditions)
# -------------------------
# For phases i=2,...,num_phases, impose that the initial state of phase i equals
# the final state of phase i-1.
num_phases = len(phases)
for i in range(1, num_phases):
    # The entire state vector is continuous.
    opti.subject_to(X_list[i][:, 0] == X_list[i-1][:, -1])

# -------------------------
# 7. Impose boundary conditions on takeoff and landing phases
# -------------------------
# For take-off: initial state: v=0, gamma=0, h=0, s=0, m = initial mass (say 70000 kg)
x0 = np.array([0.0, 0.0, 0.0, 0.0, 70000.0])
opti.set_initial(X_list[0][:, 0], x0)
opti.subject_to(X_list[0][0, 0] == 0.0)   # speed at takeoff start = 0
opti.subject_to(X_list[0][3, 0] == 0.0)   # along-path distance starts at 0

# For take-off, we want the final speed to be the target for climb1 (v_250):
# (Even though we did not force v in takeoff phase, we can impose its terminal value.)
opti.subject_to(X_list[0][0, -1] == v_250)

# For landing: impose that the final state has zero altitude and a low touchdown speed.
# We want, for instance, v_final ≈ 30 m/s (about 60 knots) and h = 0.
opti.subject_to(X_list[-1][2, -1] == 0.0)  # altitude at landing = 0
opti.subject_to(X_list[-1][0, -1] == 30.0) # landing speed 30 m/s

# -------------------------
# 8. Define the overall cost function
# -------------------------
# The running cost in each phase is:
#   J_phase = sum_{k=0}^{N-1} (fuel_flow + CI)*dt
# where fuel_flow is computed from thrust T as in the dynamics.
total_cost = 0
for i, phase in enumerate(phases):
    N_phase = phase["N"]
    T_phase = T_list[i]
    dt = T_phase / N_phase
    X = X_list[i]
    U = U_list[i]
    # Sum cost over intervals in phase i
    J_phase = 0
    for k in range(N_phase):
        T_k = U[0, k]
        # Fuel flow as in the dynamics:
        fuel_flow_k = 0.0001 * T_k  # [kg/s]
        J_phase += (fuel_flow_k + CI) * dt
    total_cost += J_phase
opti.minimize(total_cost)

# -------------------------
# 9. Initial guesses for the states in each phase (for smoother convergence)
# -------------------------
# We linearly interpolate altitude between the phase boundaries.
for i, phase in enumerate(phases):
    N_phase = phase["N"]
    X = X_list[i]
    # For each node, set altitude to a linear interpolation between h0 and hF.
    for k in range(N_phase+1):
        h_guess = phase["h0"] + (phase["hF"] - phase["h0"]) * k / N_phase
        opti.set_initial(X[2, k], h_guess)
    # For the speed, if a target is prescribed, that will be enforced; otherwise,
    # we set a linear ramp. For takeoff, ramp from 0 to v_250.
    if phase["v_target"] is None:
        if phase["name"] == "takeoff":
            for k in range(N_phase+1):
                v_guess = 0 + (v_250 - 0) * k / N_phase
                opti.set_initial(X[0, k], v_guess)
        elif phase["name"] == "landing":
            for k in range(N_phase+1):
                # For landing, decelerate from v_250 to 30 m/s.
                v_guess = v_250 + (30.0 - v_250) * k / N_phase
                opti.set_initial(X[0, k], v_guess)
    # For other states (gamma, s, m) we provide rough guesses.
    for k in range(N_phase+1):
        opti.set_initial(X[1, k], 0.0)  # flight path angle near zero
        opti.set_initial(X[3, k], 1000.0 * (i + k/float(N_phase)))  # along-path distance increasing
        opti.set_initial(X[4, k], 70000.0 - 1000*i)  # mass decreasing gradually

    # For controls, use moderate values.
    U = U_list[i]
    for k in range(N_phase):
        opti.set_initial(U[0, k], 50000.0)  # thrust guess
        opti.set_initial(U[1, k], 1.0)       # load factor guess
        opti.set_initial(U[2, k], 0.0)       # beta guess

# -------------------------
# 10. Solve the NLP
# -------------------------
p_opts = {"expand": True}
s_opts = {"max_iter": 2000, "print_level": 5}
opti.solver('ipopt', p_opts, s_opts)

try:
    sol = opti.debug.solve()
except RuntimeError as e:
    print("Solver error:", e)
    sol = opti.debug
    pass

# -------------------------
# 11. Retrieve and plot the results
# -------------------------
# We can plot the altitude and speed profiles phase by phase.
phase_time = []
phase_altitude = []
phase_speed = []
phase_s = []

# For plotting, accumulate time along phases.
t_total = 0
time_all = []
v_all = []
h_all = []
s_all = []
phase_boundaries = [0]

for i, phase in enumerate(phases):
    N_phase = phase["N"]
    T_phase_val = sol.value(T_list[i])
    dt = T_phase_val / N_phase
    time_phase = np.linspace(0, T_phase_val, N_phase+1) + t_total
    X_phase = sol.value(X_list[i])
    t_total += T_phase_val
    phase_boundaries.append(t_total)
    
    time_all.append(time_phase)
    v_all.append(X_phase[0, :])
    h_all.append(X_phase[2, :])
    s_all.append(X_phase[3, :])

time_all = np.concatenate(time_all)
v_all = np.concatenate(v_all)
h_all = np.concatenate(h_all)
s_all = np.concatenate(s_all)

plt.figure(figsize=(12,8))
plt.subplot(3,1,1)
plt.plot(time_all, h_all, 'b-', linewidth=2)
plt.xlabel("Time [s]")
plt.ylabel("Altitude [m]")
plt.title("Altitude Profile")

plt.subplot(3,1,2)
plt.plot(time_all, v_all, 'r-', linewidth=2)
plt.xlabel("Time [s]")
plt.ylabel("Speed [m/s]")
plt.title("Speed Profile")

plt.subplot(3,1,3)
plt.plot(time_all, s_all, 'g-', linewidth=2)
plt.xlabel("Time [s]")
plt.ylabel("Along-path distance [m]")
plt.title("Distance Profile")

plt.tight_layout()
plt.show()

# Optionally, print phase durations:
for i, phase in enumerate(phases):
    print(f"Phase {i+1} ({phase['name']}): Duration = {sol.value(T_list[i]):.2f} s")
