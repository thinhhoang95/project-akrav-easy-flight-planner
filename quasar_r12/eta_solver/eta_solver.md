# ETA Calculation Process

## Initialization

The process starts by taking the scheduled takeoff time and adding a taxi-out time to determine the actual departure time. It creates a working copy of the provided route fixes list.

## Top of Climb (TOC) Calculation

The function `find_top_of_climb_fix` is called to determine where and when the aircraft reaches its cruise altitude. This function calculates the time to climb based on aircraft performance parameters and estimates the distance covered during the climb, accounting for wind. A new "TOC" fix is created at the calculated coordinates and inserted into the route at the appropriate point.

## Top of Descent (TOD) Calculation

The required distance for descent from cruise altitude to the final altitude at the destination is estimated, using aircraft performance and an estimated average wind during descent. A refined initial estimate of the final ETA is used for the wind lookup. The process iterates backward from the destination fix to find the point along the route where descent should begin to meet the required descent distance. A new "TOD" fix is created at the calculated coordinates and inserted into the route. The process also includes logic to handle scenarios where the route after TOC is too short for a distinct cruise phase, effectively placing the TOD at the TOC location.

## Sequential ETA Calculation

The function then iterates through the modified list of fixes (which now includes the inserted TOC and potentially TOD). For each segment between fixes, it determines the flight phase (Climb, Cruise, or Descent) based on the fix's position relative to the inserted TOC and TOD points. Based on the current altitude and phase, it selects the appropriate True Air Speed (TAS) using the provided aircraft performance parameters. It estimates the wind for the segment (currently using the wind at the start fix's location and estimated mid-segment time/altitude) and calculates the headwind/tailwind component. Ground speed (GS) is calculated as TAS plus the wind component. A minimum GS is enforced to avoid issues with negative values. The time required to traverse the segment is calculated based on the segment distance and ground speed. The ETA at the end fix of the segment is determined by adding the segment time to the ETA of the start fix. The estimated altitude at the end of the segment is also calculated, accounting for climb rate (capped at cruise altitude) or descent rate (ensuring non-negative altitude at the final destination).

## Result Compilation

Finally, the calculated ETAs, phases, and altitudes for each fix (keyed by their name) are compiled into a dictionary along with the modified list of fixes and the indices where TOC and TOD were inserted.

---

# Sources of Uncertainty in ETA Calculation

Excluding ATC vectoring, several factors contribute to uncertainty in the Estimated Time of Arrival (ETA). These uncertainties can contribute to a variance that grows over time, potentially like a Brownian process.

## Factors Contributing to Uncertainty

* **Wind Forecast Accuracy:**
    * **How it enters calculation:** Wind speed and direction directly affect ground speed ($gs_{segment} = tas_{segment} + wind_{comp}$). The wind model is queried at estimated mid-segment times/locations (or simplified locations) using estimated altitudes. Wind also influences the calculation of climb distance and required descent distance.
    * **Why reality differs:** Wind forecasts are inherently uncertain. Both speed and direction can deviate significantly from the forecast, especially further out in time or in areas with complex weather patterns. The model also simplifies by potentially using average winds or winds at specific points/altitudes.
    * **Impact on ETA:** Errors in headwind/tailwind component directly translate to errors in ground speed. Errors in wind during climb/descent also affect the calculated TOC/TOD locations and times.
    * **Variance Accumulation:** Wind errors on successive segments contribute to the cumulative time error. The variance of the total time error likely increases with the distance flown or time elapsed, similar to a Brownian motion process.

* **Temperature Mismatch (Actual vs. Forecast/Standard):**
    * **How it enters calculation:** Temperature affects True Air Speed (TAS) for a given Mach number. While the code uses predefined TAS values based on altitude bands, real aircraft often fly a constant Mach number in cruise, where actual TAS varies with temperature.
    * **Why reality differs:** Atmospheric temperature rarely matches standard conditions or forecasts perfectly.
    * **Impact on ETA:** If an aircraft flies a constant Mach number and the air is warmer than assumed, TAS will be higher, leading to higher ground speed and earlier ETAs (and vice versa). This effect is most pronounced at high altitudes.
    * **Variance Accumulation:** Temperature deviations can persist over larger areas, causing a drift in ETA over many segments. Fluctuations in predicted temperature contribute to accumulating variance.

* **Climb Rate / Performance Variation:**
    * **How it enters calculation:** The code uses a standard climb rate and altitude-banded climb TAS to determine time to climb and distance covered during climb, setting the TOC location and time.
    * **Why reality differs:** Actual climb rate depends heavily on aircraft weight, thrust setting, temperature, and specific aircraft performance.
    * **Impact on ETA:** A lower-than-assumed climb rate means it takes longer to reach cruise altitude, delaying TOC ETA. Errors in TOC time shift all subsequent ETAs. Errors in TOC location change the length of the first cruise segment.
    * **Variance Accumulation:** This primarily affects the initial phase transition (TOC). The error introduced here creates an offset for all subsequent calculations.

* **Cruise Speed Variation (Cost Index / Actual TAS):**
    * **How it enters calculation:** A fixed cruise TAS is provided as input.
    * **Why reality differs:** Airlines use a "Cost Index" which determines the balance between fuel cost and time, leading to variations in actual cruise speed. Aircraft typically fly a constant Mach, so actual TAS varies with temperature.
    * **Impact on ETA:** Flying a TAS different from the input directly impacts the duration of cruise segments.
    * **Variance Accumulation:** If the aircraft consistently flies at a different speed, it creates a systematic drift in ETA that grows linearly with distance covered in cruise. Random fluctuations around the target speed contribute to variance accumulation.

* **Descent Rate / Profile Variation:**
    * **How it enters calculation:** The required descent distance is estimated based on average TAS, wind, and a standard descent rate or angle, determining the TOD location. Time calculations during descent use descent TAS and implicitly assume the standard descent rate.
    * **Why reality differs:** Actual descent profiles vary based on weight, ATC clearances, desired arrival time, and descent strategy.
    * **Impact on ETA:** Errors in calculating required descent distance shift the TOD location. Using an incorrect descent rate/TAS profile during descent calculation introduces errors in segment times.
    * **Variance Accumulation:** Errors in TOD placement cause a shift in timing for the final phase. Errors in predicting descent time add further uncertainty towards the end of the flight.

* **Aircraft Weight:**
    * **How it enters calculation:** It doesn't, directly. Performance figures are assumed fixed.
    * **Why reality differs:** Aircraft weight changes significantly based on payload and fuel burn.
    * **Impact on ETA:** Weight primarily affects climb performance (heavier = slower climb, later TOC) and potentially optimal cruise speed/altitude. Lack of weight consideration is a major simplification impacting climb phase and TOC placement accuracy.
    * **Variance Accumulation:** Initial weight uncertainty significantly impacts the climb phase and TOC, creating an initial offset.

* **Taxi Time Variation:**
    * **How it enters calculation:** A fixed taxi-out time is added to scheduled takeoff time.
    * **Why reality differs:** Taxi times depend on airport layout, traffic, runway configuration, and gate location.
    * **Impact on ETA:** Any deviation from the assumed taxi-out time directly shifts the entire ETA profile.
    * **Variance Accumulation:** This introduces a single, initial offset error.

## Summary for Brownian Motion Modeling

Wind forecast errors and random fluctuations in actual cruise speed/TAS are best suited for a Brownian motion model, where cumulative time error variance grows with distance/time. Errors in climb and descent calculations (affecting TOC/TOD placement) primarily cause shifts in the timing of subsequent phases, acting more like initial offsets for those phases. Taxi time variance adds another initial variance component. Total ETA variance is a combination of initial offset variances and accumulated variance from flight segments.

## Non-Monotonic Variance Behavior

The standard deviation of the cumulative ETA error should be monotonically increasing (or stay the same). If the standard deviation decreases at certain points, it indicates an issue in how the uncertainty calculation is formulated or implemented, specifically in the step that converts accumulated position error variance into time error variance.

Let's consider the key formula used for this conversion:
$$\sigma_{\epsilon_n} \approx \frac{\sqrt{Var(e_x(ETA_n))}}{gs_n}$$
where $Var(\epsilon_n)$ is the variance of the ETA error at fix $n$, $Var(e_x(ETA_n))$ is the variance of the along-track position error at the time of arriving at fix $n$, and $gs_n$ is the estimated ground speed leading into fix $n$.

The numerator, $\sqrt{Var(e_x(ETA_n))}$, represents the accumulating uncertainty in the aircraft's along-track position and is monotonically increasing. The issue arises with the denominator, $gs_n$. This term is approximated using the ground speed of the segment *leading into* fix $n$. If the estimated ground speed $gs_n$ increases significantly from one segment to the next (e.g., due to a change in flight phase like entering a higher-speed climb segment), the denominator can increase proportionally more than the numerator increases over that short segment. This causes the overall ratio, $\sigma_{\epsilon_n}$, to decrease, leading to the non-monotonic behavior observed.

## Proposed Fix

A more robust approach that guarantees monotonicity is to calculate and sum the variance of the *time duration* of each segment directly. The variance of the time error for a single segment $i$ can be approximated as:
$$Var(\Delta Error_i) \approx \left(\frac{\Delta ETA_i}{gs_i}\right)^2 \sigma_{gs,i}^2$$
where $\Delta ETA_i$ is the segment duration, $gs_i$ is the estimated ground speed during segment $i$, and $\sigma_{gs,i}^2$ is the variance of the ground speed error during segment $i$.

The total variance of the ETA error at fix $n$ would then be the sum of the variances of the time errors of all preceding segments (assuming errors are roughly independent segment-to-segment):
$$Var(\epsilon_n) = \sum_{i=0}^{n-1} Var(\Delta Error_i) \approx \sum_{i=0}^{n-1} \left(\frac{\Delta ETA_i}{gs_i}\right)^2 \sigma_{gs,i}^2$$
This summation approach calculates the variance added by each segment and adds it to the previous total. Since we are always adding non-negative variance contributions, the total variance $Var(\epsilon_n)$ (and thus its square root, the standard deviation) will be monotonically increasing.