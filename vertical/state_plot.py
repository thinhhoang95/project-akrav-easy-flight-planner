def plot_states(v, fpa, h, s, m, T, gamma_ref, beta):
    import matplotlib.pyplot as plt

    # Create a figure for the states
    plt.figure(figsize=(12, 8))

    # Create a 3x3 subplot grid for comprehensive state visualization
    plt.subplot(3, 3, 1)
    plt.plot(v, label='Velocity')
    plt.title('Velocity')
    plt.xlabel('Timestep')
    plt.ylabel('m/s')
    plt.legend()

    plt.subplot(3, 3, 2)
    plt.plot(fpa, label='Flight Path Angle')
    plt.title('Flight Path Angle')
    plt.xlabel('Timestep')
    plt.ylabel('radians')
    plt.legend()

    plt.subplot(3, 3, 3)
    plt.plot(h, label='Altitude')
    plt.title('Altitude')
    plt.xlabel('Timestep')
    plt.ylabel('m')
    plt.legend()

    plt.subplot(3, 3, 4)
    plt.plot(s, label='Along-Path Distance')
    plt.title('Along-Path Distance')
    plt.xlabel('Timestep')
    plt.ylabel('m')
    plt.legend()

    plt.subplot(3, 3, 5)
    plt.plot(m, label='Mass')
    plt.title('Mass')
    plt.xlabel('Timestep')
    plt.ylabel('kg')
    plt.legend()

    plt.subplot(3, 3, 6)
    plt.plot(T, label='Thrust')
    plt.title('Thrust')
    plt.xlabel('Timestep')
    plt.ylabel('N')
    plt.legend()

    plt.subplot(3, 3, 7)
    plt.plot(gamma_ref, label='Reference Flight Path Angle')
    plt.title('Reference Flight Path Angle')
    plt.xlabel('Timestep')
    plt.ylabel('radians')
    plt.legend()

    plt.subplot(3, 3, 8)
    plt.plot(beta, label='Speedbrake')
    plt.title('Speedbrake')
    plt.xlabel('Timestep')
    plt.ylabel('Parameter')
    plt.legend()

    plt.tight_layout()
    plt.show()

import numpy as np

def plot_states_with_phases(Xclb1, Xclb2, Xcrz, Xdes1, Uclb1, Uclb2, Ucrz, Udes1, Tphases):
    import matplotlib.pyplot as plt

    # Create a figure for the states
    plt.figure(figsize=(12, 8))

    # Extract individual states from each phase
    v1, gamma1, h1, s1, m1= Xclb1
    v2, gamma2, h2, s2, m2 = Xclb2 
    v3, gamma3, h3, s3, m3 = Xcrz
    v4, gamma4, h4, s4, m4 = Xdes1
    T1, gamma_ref1, beta1 = Uclb1
    T2, gamma_ref2, beta2 = Uclb2
    T3, gamma_ref3, beta3 = Ucrz
    T4, gamma_ref4, beta4 = Udes1

    # Calculate time vectors for each phase
    dt1 = Tphases[0]/len(v1)
    dt2 = Tphases[1]/len(v2)
    dt3 = Tphases[2]/len(v3)
    dt4 = Tphases[3]/len(v4)


    t1 = np.linspace(0, Tphases[0], len(v1))
    t2 = np.linspace(Tphases[0], Tphases[0] + Tphases[1], len(v2))
    t3 = np.linspace(Tphases[0] + Tphases[1], Tphases[0] + Tphases[1] + Tphases[2], len(v3))
    t4 = np.linspace(Tphases[0] + Tphases[1] + Tphases[2], Tphases[0] + Tphases[1] + Tphases[2] + Tphases[3], len(v4))

    # Create a 3x3 subplot grid
    plt.subplot(3, 3, 1)
    plt.plot(t1, v1, label='Climb 1')
    plt.plot(t2, v2, label='Climb 2')
    plt.plot(t3, v3, label='Cruise')
    plt.plot(t4, v4, label='Descent')
    plt.axvline(x=Tphases[0], color='k', linestyle=':')
    plt.axvline(x=Tphases[0] + Tphases[1], color='k', linestyle=':')
    plt.axvline(x=Tphases[0] + Tphases[1] + Tphases[2], color='k', linestyle=':')
    plt.title('Velocity')
    plt.xlabel('Time (s)')
    plt.ylabel('m/s')
    plt.legend()

    plt.subplot(3, 3, 2)
    plt.plot(t1, gamma1, label='Climb 1')
    plt.plot(t2, gamma2, label='Climb 2')
    plt.plot(t3, gamma3, label='Cruise')
    plt.plot(t4, gamma4, label='Descent')
    plt.axvline(x=Tphases[0], color='k', linestyle=':')
    plt.axvline(x=Tphases[0] + Tphases[1], color='k', linestyle=':')
    plt.axvline(x=Tphases[0] + Tphases[1] + Tphases[2], color='k', linestyle=':')
    plt.title('Flight Path Angle')
    plt.xlabel('Time (s)')
    plt.ylabel('radians')
    plt.legend()

    plt.subplot(3, 3, 3)
    plt.plot(t1, h1, label='Climb 1')
    plt.plot(t2, h2, label='Climb 2')
    plt.plot(t3, h3, label='Cruise')
    plt.plot(t4, h4, label='Descent')
    plt.axvline(x=Tphases[0], color='k', linestyle=':')
    plt.axvline(x=Tphases[0] + Tphases[1], color='k', linestyle=':')
    plt.axvline(x=Tphases[0] + Tphases[1] + Tphases[2], color='k', linestyle=':')
    plt.title('Altitude')
    plt.xlabel('Time (s)')
    plt.ylabel('m')
    plt.legend()

    plt.subplot(3, 3, 4)
    plt.plot(t1, s1, label='Climb 1')
    plt.plot(t2, s2, label='Climb 2')
    plt.plot(t3, s3, label='Cruise')
    plt.plot(t4, s4, label='Descent')
    plt.axvline(x=Tphases[0], color='k', linestyle=':')
    plt.axvline(x=Tphases[0] + Tphases[1], color='k', linestyle=':')
    plt.axvline(x=Tphases[0] + Tphases[1] + Tphases[2], color='k', linestyle=':')
    plt.title('Along-Path Distance')
    plt.xlabel('Time (s)')
    plt.ylabel('m')
    plt.legend()

    plt.subplot(3, 3, 5)
    plt.plot(t1, m1, label='Climb 1')
    plt.plot(t2, m2, label='Climb 2')
    plt.plot(t3, m3, label='Cruise')
    plt.plot(t4, m4, label='Descent')
    plt.axvline(x=Tphases[0], color='k', linestyle=':')
    plt.axvline(x=Tphases[0] + Tphases[1], color='k', linestyle=':')
    plt.axvline(x=Tphases[0] + Tphases[1] + Tphases[2], color='k', linestyle=':')
    plt.title('Mass')
    plt.xlabel('Time (s)')
    plt.ylabel('kg')
    plt.legend()

    # ------------------------------------------------------------
    # CONTROL INPUTS
    # ------------------------------------------------------------

    plt.subplot(3, 3, 6)
    plt.plot(t1[:-1], T1, label='Climb 1')
    plt.plot(t2[:-1], T2, label='Climb 2')
    plt.plot(t3[:-1], T3, label='Cruise')
    plt.plot(t4[:-1], T4, label='Descent')
    plt.axvline(x=Tphases[0], color='k', linestyle=':')
    plt.axvline(x=Tphases[0] + Tphases[1], color='k', linestyle=':')
    plt.axvline(x=Tphases[0] + Tphases[1] + Tphases[2], color='k', linestyle=':')
    plt.title('Thrust')
    plt.xlabel('Time (s)')
    plt.ylabel('N')
    plt.legend()

    plt.subplot(3, 3, 7)
    plt.plot(t1[:-1], gamma_ref1, label='Climb 1')
    plt.plot(t2[:-1], gamma_ref2, label='Climb 2')
    plt.plot(t3[:-1], gamma_ref3, label='Cruise')
    plt.plot(t4[:-1], gamma_ref4, label='Descent')
    plt.axvline(x=Tphases[0], color='k', linestyle=':')
    plt.axvline(x=Tphases[0] + Tphases[1], color='k', linestyle=':')
    plt.axvline(x=Tphases[0] + Tphases[1] + Tphases[2], color='k', linestyle=':')
    plt.title('Reference Flight Path Angle')
    plt.xlabel('Time (s)')
    plt.ylabel('radians')
    plt.legend()

    plt.subplot(3, 3, 8)
    plt.plot(t1[:-1], beta1, label='Climb 1')
    plt.plot(t2[:-1], beta2, label='Climb 2')
    plt.plot(t3[:-1], beta3, label='Cruise')
    plt.plot(t4[:-1], beta4, label='Descent')
    plt.axvline(x=Tphases[0], color='k', linestyle=':')
    plt.axvline(x=Tphases[0] + Tphases[1], color='k', linestyle=':')
    plt.axvline(x=Tphases[0] + Tphases[1] + Tphases[2], color='k', linestyle=':')
    plt.title('Speedbrake')
    plt.xlabel('Time (s)')
    plt.ylabel('Parameter')
    plt.legend()

    plt.tight_layout()
    plt.show()


    

def plot_states_ext(L, D, nz, Cl, Cd, rho, fuel_flow = None):
    import matplotlib.pyplot as plt

    # Create a figure for the states
    plt.figure(figsize=(12, 8))

    # Create a 3x3 subplot grid for comprehensive state visualization
    plt.subplot(3, 3, 1)
    plt.plot(L, label='Lift')
    plt.title('Lift')
    plt.xlabel('Timestep')
    plt.ylabel('N')
    plt.legend()

    plt.subplot(3, 3, 2)
    plt.plot(D, label='Drag')
    plt.title('Drag')
    plt.xlabel('Timestep')
    plt.ylabel('N')
    plt.legend()

    plt.subplot(3, 3, 3)
    plt.plot(nz, label='Load Factor')
    plt.title('Load Factor')
    plt.xlabel('Timestep')
    plt.ylabel('N')
    plt.legend()

    plt.subplot(3, 3, 4)
    plt.plot(Cl, label='Lift Coefficient')
    plt.title('Lift Coefficient')
    plt.xlabel('Timestep')
    plt.ylabel('N')
    plt.legend()

    plt.subplot(3, 3, 5)
    plt.plot(Cd, label='Drag Coefficient')
    plt.title('Drag Coefficient')
    plt.xlabel('Timestep')
    plt.ylabel('N')
    plt.legend()

    plt.subplot(3, 3, 6)
    plt.plot(rho, label='Density')
    plt.title('Density')
    plt.xlabel('Timestep')
    plt.ylabel('kg/m^3')
    plt.legend()

    if fuel_flow is not None:
        plt.subplot(3, 3, 7)
        plt.plot(fuel_flow, label='Fuel Flow')
        plt.title('Fuel Flow')
        plt.xlabel('Timestep')
        plt.ylabel('kg/s')
        plt.legend()

    plt.tight_layout()
    plt.show()
    
    
    
    
    
    
