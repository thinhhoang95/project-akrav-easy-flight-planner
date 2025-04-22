import bisect
import numpy as np
from math import radians, sin, cos, atan2, sqrt, degrees
import xarray as xr # Needed for interpolation

class GraphRouteContext:
    """
    Context for storing and querying time- and altitude-dependent properties
    on edges of a networkx graph (e.g., routes like Madrid-London).
    """

    def __init__(self, graph):
        """
        Initialize with a networkx graph.
        """
        self.graph = graph
        # Structure: { (u, v): { property_name: { 'times': array, 'alts': array, 'values': 2D array } } }
        self._property_data = {}

    def add_property_data(self, u, v, property_name, times, altitudes, values):
        """
        Add a 2D grid of data for a given edge and property.
 
        Parameters:
        - u, v: nodes defining the edge (ordered as in the graph).
        - property_name: string, one of 'tail_wind', 'cross_wind', 'CAPE', 'CIN', etc.
        - times: 1D sorted array-like of time points (floats or datetime64).
        - altitudes: 1D sorted array-like of altitude values.
        - values: 2D array of shape (len(times), len(altitudes)), where values[i,j]
                  corresponds to (times[i], altitudes[j]).
        """
        key = (u, v)
        self._property_data.setdefault(key, {})[property_name] = {
            'times': np.array(times),
            'alts': np.array(altitudes),
            'values': np.array(values)
        }

    def upsert_property_data(self, u, v, property_name, times, altitudes, values):
        """
        Append data to an existing property or create a new one if it doesn't exist.
        
        Parameters:
        - u, v: nodes defining the edge (ordered as in the graph).
        - property_name: string, name of the property.
        - times: 1D sorted array-like of time points (floats or datetime64).
        - altitudes: 1D sorted array-like of altitude values.
        - values: 2D array of shape (len(times), len(altitudes)), where values[i,j]
                  corresponds to (times[i], altitudes[j]).
        """
        key = (u, v)
        edge_data = self._property_data.setdefault(key, {})
        
        if property_name in edge_data:
            # Append to existing property
            existing = edge_data[property_name]
            
            # Concatenate and sort times
            new_times = np.concatenate([existing['times'], np.array(times)])
            time_indices = np.argsort(new_times)
            new_times = new_times[time_indices]
            
            # Concatenate and sort altitudes
            new_alts = np.concatenate([existing['alts'], np.array(altitudes)])
            alt_indices = np.argsort(new_alts)
            new_alts = new_alts[alt_indices]
            
            # Create new values grid with combined dimensions
            old_values = existing['values']
            new_values = np.zeros((len(new_times), len(new_alts)))
            
            # Map old values to the new grid
            for i, t in enumerate(existing['times']):
                for j, a in enumerate(existing['alts']):
                    ti = np.searchsorted(new_times, t)
                    aj = np.searchsorted(new_alts, a)
                    new_values[ti, aj] = old_values[i, j]
            
            # Map new values to the new grid
            for i, t in enumerate(times):
                for j, a in enumerate(altitudes):
                    ti = np.searchsorted(new_times, t)
                    aj = np.searchsorted(new_alts, a)
                    new_values[ti, aj] = values[i, j]
            
            # Update the property data
            edge_data[property_name] = {
                'times': new_times,
                'alts': new_alts,
                'values': new_values
            }
        else:
            # Create new property (same as add_property_data)
            edge_data[property_name] = {
                'times': np.array(times),
                'alts': np.array(altitudes),
                'values': np.array(values)
            }

    def get_property(self, u, v, property_name, time, altitude, method='linear'):
        """
        Retrieve the property value at a specific time and altitude.
 
        Parameters:
        - u, v: nodes defining the edge.
        - property_name: string key for the stored property.
        - time: float or datetime64, the query time.
        - altitude: float, the query altitude.
        - method: 'linear' (bilinear interp), 'min', or 'max' over the 4 nearest grid points.
 
        Returns:
        - Interpolated or aggregated float value.
 
        Raises:
        - KeyError if no data is found.
        - ValueError for unsupported methods.
        """
        data = self._property_data.get((u, v), {}).get(property_name)
        if data is None:
            raise KeyError(f"No data for property '{property_name}' on edge ({u}, {v})")

        times = data['times']
        alts = data['alts']
        vals = data['values']

        # Locate insertion points
        i = bisect.bisect_left(times, time)
        j = bisect.bisect_left(alts, altitude)
        # Determine neighboring indices, clamped to array bounds
        i0 = max(min(i - 1, len(times) - 1), 0)
        i1 = max(min(i,     len(times) - 1), 0)
        j0 = max(min(j - 1, len(alts)  - 1), 0)
        j1 = max(min(j,     len(alts)  - 1), 0)

        # Corner values
        v00 = vals[i0, j0]
        v01 = vals[i0, j1]
        v10 = vals[i1, j0]
        v11 = vals[i1, j1]

        if method == 'min':
            return min(v00, v01, v10, v11)
        elif method == 'max':
            return max(v00, v01, v10, v11)
        elif method == 'linear':
            t0, t1 = times[i0], times[i1]
            a0, a1 = alts[j0], alts[j1]
            # Degenerate cases
            if t0 == t1 and a0 == a1:
                return v00
            elif t0 == t1:
                # Linear in altitude only
                if a1 == a0:
                    return v00
                wa = (altitude - a0) / (a1 - a0)
                return v00 * (1 - wa) + v01 * wa
            elif a0 == a1:
                # Linear in time only
                wt = (time - t0) / (t1 - t0)
                return v00 * (1 - wt) + v10 * wt
            else:
                # Bilinear interpolation
                wt = (time - t0) / (t1 - t0)
                wa = (altitude - a0) / (a1 - a0)
                return (
                    v00 * (1 - wt) * (1 - wa) +
                    v10 * wt       * (1 - wa) +
                    v01 * (1 - wt) * wa       +
                    v11 * wt       * wa
                )
        else:
            raise ValueError(f"Unsupported method '{method}'")
        
    def __str__(self):
        """Return a summary of this context's property data."""
        if not self._property_data:
            return "GraphRouteContext: no property data"
        lines = [f"GraphRouteContext: {len(self._property_data)} edge(s) with data"]
        for (u, v), props in self._property_data.items():
            lines.append(f" Edge ({u}, {v}):")
            for prop, data in props.items():
                times = data['times']
                alts = data['alts']
                lines.append(
                    f"  - {prop}: times {times[0]} to {times[-1]} ({len(times)} pts), "
                    f"alts {alts[0]} to {alts[-1]} ({len(alts)} pts)"
                )
        return "\n".join(lines)
    
    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path):
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)
        
def compute_bearing(lat1, lon1, lat2, lon2):
    """
    Compute the bearing from (lat1, lon1) to (lat2, lon2) in radians,
    where 0 points to true north and positive is clockwise.
    """
    lat1r, lat2r = radians(lat1), radians(lat2)
    dlon = radians(lon2 - lon1)
    x = sin(dlon) * cos(lat2r)
    y = cos(lat1r) * sin(lat2r) - sin(lat1r) * cos(lat2r) * cos(dlon)
    return atan2(x, y)

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance in kilometers between two points on the earth."""
    R = 6371  # Radius of Earth in kilometers
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

def sample_great_circle(lat1, lon1, lat2, lon2, n_points):
    """
    Calculate intermediate points along a great circle path.
    Returns (lats, lons) numpy arrays in degrees. Includes start and end points.
    """
    lat1_deg, lon1_deg = lat1, lon1 # Keep original degrees for return if needed
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(radians, [lat1, lon1, lat2, lon2])

    # Calculate angular distance d using haversine formula components
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    d = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Handle zero distance case
    if n_points <= 1 or np.isclose(d, 0):
        return np.array([lat1_deg]), np.array([lon1_deg]) 

    lats_rad = np.zeros(n_points)
    lons_rad = np.zeros(n_points)
    
    fractions = np.linspace(0, 1, n_points)
    
    sin_d = sin(d)
    # Handle antipodal case where sin(d) is near zero
    if np.isclose(sin_d, 0):
        # Linear interpolation for antipodal points as a fallback
        # print(f"Warning: Antipodal or near-antipodal points ({lat1},{lon1}) to ({lat2},{lon2}), using linear path.")
        return np.linspace(lat1_deg, lat2, n_points), np.linspace(lon1_deg, lon2, n_points)

    for i, f in enumerate(fractions):
        A = sin((1 - f) * d) / sin_d
        B = sin(f * d) / sin_d

        x = A * cos(lat1_rad) * cos(lon1_rad) + B * cos(lat2_rad) * cos(lon2_rad)
        y = A * cos(lat1_rad) * sin(lon1_rad) + B * cos(lat2_rad) * sin(lon2_rad)
        z = A * sin(lat1_rad) + B * sin(lat2_rad)

        lats_rad[i] = atan2(z, sqrt(x**2 + y**2))
        lons_rad[i] = atan2(y, x)

    return np.degrees(lats_rad), np.degrees(lons_rad)

def prepare_wind_property_args(Gm, ds, timestamp, sampling_resolution_km=25):
    """
    Prepare (u, v, property_name, times, alts, values) for each edge in Gm
    based on the xarray Dataset `ds` and a single UNIX‐time `timestamp`.

    Samples winds along the great-circle path of the edge for better accuracy.
    
    Parameters
    ----------
    Gm : networkx.Graph
        Route graph with node attributes 'lat', 'lon'.
    ds : xarray.Dataset
        Dataset containing 'u_wind', 'v_wind' variables with 'latitude', 'longitude' coords.
    timestamp : float or np.datetime64
        The single time point for the data.
    sampling_resolution_km : float, optional
        Approximate distance between sampling points along the edge, by default 25 km.

    Returns
    -------
    List of tuples:
      (u, v, property_name, times, altitudes, values)
      where 'values' is the average wind component along the path, shaped (1, n_alts).
    """
    # time dim (one shot)
    times = np.array([timestamp], dtype=float)
    # altitude dim (if ds has it, otherwise use provided values or dummy zero)
    if 'altitude' in ds.coords:
        # Ensure alts is at least 1D
        alts = np.atleast_1d(ds['altitude'].values) 
    else:
        alts = np.array([0.], dtype=float)
    num_alts = len(alts)

    args_list = []
    from tqdm import tqdm
    for u, v in tqdm(Gm.edges(), desc='Processing edges', total=Gm.number_of_edges()):
        lat_u, lon_u = Gm.nodes[u]['lat'], Gm.nodes[u]['lon']
        lat_v, lon_v = Gm.nodes[v]['lat'], Gm.nodes[v]['lon']
        
        # Calculate bearing (radians)
        bearing = compute_bearing(lat_u, lon_u, lat_v, lon_v)
        
        # Calculate distance and determine number of sample points
        distance_km = haversine(lat_u, lon_u, lat_v, lon_v)
        n_points = max(2, int(np.ceil(distance_km / sampling_resolution_km)))
        
        # Get sample points along the great circle path
        sample_lats, sample_lons = sample_great_circle(lat_u, lon_u, lat_v, lon_v, n_points)

        # Create xarray DataArrays for interpolation coordinates
        # Ensure unique coordinate values if needed by interpolation method, 
        # though 'linear' usually handles duplicates.
        xr_sample_lats = xr.DataArray(sample_lats, dims="sample_points")
        xr_sample_lons = xr.DataArray(sample_lons, dims="sample_points")

        try:
            # Interpolate u and v wind components using xarray
            # Assuming 'u_wind', 'v_wind' depend only on lat/lon as per wind_extract.md
            interpolated_winds = ds[['u_wind', 'v_wind']].interp(
                latitude=xr_sample_lats,
                longitude=xr_sample_lons,
                method='linear',
                kwargs={"fill_value": None} # Use None to get NaN outside domain
            )
            u_samples = interpolated_winds['u_wind'].values
            v_samples = interpolated_winds['v_wind'].values

            # Check for NaNs which indicate points outside the data domain
            valid_mask = ~np.isnan(u_samples) & ~np.isnan(v_samples)
            if not np.any(valid_mask):
                # All sample points were outside the domain, cannot calculate average
                # print(f"Warning: Edge ({u}, {v}) is outside the wind data domain. Setting winds to 0.")
                avg_tail = 0.0
                avg_cross = 0.0
            else:
                 # Project valid samples into tail / cross components
                sin_bearing = sin(bearing)
                cos_bearing = cos(bearing)
                
                tail_samples = u_samples[valid_mask] * sin_bearing + v_samples[valid_mask] * cos_bearing
                cross_samples = u_samples[valid_mask] * cos_bearing - v_samples[valid_mask] * sin_bearing

                # Calculate the average tail and cross wind components
                avg_tail = np.mean(tail_samples)
                avg_cross = np.mean(cross_samples)

        except Exception as e:
            print(f"Error interpolating wind for edge ({u}, {v}): {e}")
            print(f" Sample lats: {sample_lats}")
            print(f" Sample lons: {sample_lons}")
            # Fallback or error handling: set to 0 or re-raise
            avg_tail = 0.0
            avg_cross = 0.0


        # Create value arrays matching the shape (len(times), len(alts)) = (1, num_alts)
        # Replicate the average value across all altitudes
        tail_vals  = np.full((1, num_alts), avg_tail, dtype=float)
        cross_vals = np.full((1, num_alts), avg_cross, dtype=float)

        args_list.append((u, v, 'tail_wind',  times, alts, tail_vals))
        args_list.append((u, v, 'cross_wind', times, alts, cross_vals))

    return args_list