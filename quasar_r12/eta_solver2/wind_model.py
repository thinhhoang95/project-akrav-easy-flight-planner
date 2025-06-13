import xarray as xr
import os
from datetime import datetime
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

class WindModel:
    """
    Manages wind forecast data from multiple xarray Datasets, each associated with
    a specific forecast time and a minimum altitude layer.
    """

    def __init__(self):
        """
        Initializes the WindModel with an empty list to store data layers.
        """
        self.data_layers: List[Dict[str, Any]] = []
        # Each entry in data_layers is a dict:
        # {'timestamp': datetime, 'min_alt': float, 'dataset': xr.Dataset, 'filepath': str}

    def load_forecast_data(self, file_path: str, min_altitude: float):
        """
        Loads a single weather forecast NetCDF file (as an xarray.Dataset)
        and associates it with a forecast timestamp (from filename) and a minimum altitude.

        The filename is assumed to be a UNIX timestamp (e.g., "1700827200.nc").

        Args:
            file_path (str): The full path to the NetCDF file.
            min_altitude (float): The minimum altitude for which this dataset is valid.
        """
        try:
            if not os.path.exists(file_path):
                print(f"Error loading {file_path}: File does not exist.")
                return

            filename = os.path.basename(file_path)
            timestamp_str = os.path.splitext(filename)[0]
            forecast_timestamp_unix = int(timestamp_str)
            # Assuming UTC for forecast timestamps from filenames
            forecast_datetime = datetime.utcfromtimestamp(forecast_timestamp_unix)

            ds = xr.open_dataset(file_path)

            # Validate required variables are present in the dataset
            required_vars = ['u_wind', 'v_wind', 'latitude', 'longitude']
            for var in required_vars:
                if var not in ds:
                    ds.close() # Close the file if it's invalid
                    raise ValueError(f"Variable '{var}' not found in dataset: {file_path}")

            self.data_layers.append({
                'timestamp': forecast_datetime,
                'min_alt': min_altitude,
                'dataset': ds,
                'filepath': file_path
            })
            # Sort primarily by timestamp, then by min_alt for consistent layer ordering and selection
            self.data_layers.sort(key=lambda x: (x['timestamp'], x['min_alt']))
            # print(f"Successfully loaded: {filename} for min_alt {min_altitude} at {forecast_datetime}")

        except ValueError as ve: # Handles int conversion error for timestamp or missing variable
            print(f"Error processing file {file_path}: {ve}")
        except FileNotFoundError: # Should be caught by os.path.exists, but as a fallback
            print(f"Error loading {file_path}: File not found (should have been caught earlier).")
        except Exception as e:
            print(f"Failed to load or process {file_path}: {e}")
            # If a dataset was partially loaded and an error occurred, try to close it.
            if 'ds' in locals() and ds is not None:
                ds.close()


    @property
    def trenches(self) -> List[Tuple[datetime, float]]:
        """
        Returns a sorted list of unique (timestamp, min_alt) tuples,
        representing the start of each loaded altitude trench for each forecast time.
        """
        if not self.data_layers:
            return []
        
        # Collect all (timestamp, min_alt) pairs
        trench_tuples = [(layer['timestamp'], layer['min_alt']) for layer in self.data_layers]
        
        # Return unique pairs, sorted. Sorting is naturally handled by how data_layers is maintained.
        # To ensure uniqueness and specific sorting for this property:
        return sorted(list(set(trench_tuples)))

    def get_wind_property(self, altitude: float, latitude: float, longitude: float, query_timestamp: datetime) -> Tuple[Optional[float], Optional[float]]:
        """
        Retrieves u_wind and v_wind components for the given altitude, location, and timestamp.

        Args:
            altitude (float): The target altitude in meters.
            latitude (float): The target latitude in degrees.
            longitude (float): The target longitude in degrees.
            query_timestamp (datetime): The specific timestamp for which wind data is requested.

        Returns:
            Tuple[Optional[float], Optional[float]]: A tuple containing (u_wind, v_wind).
                                                     Returns (None, None) if data cannot be retrieved.
        """
        if not self.data_layers:
            # print("No data loaded into WindModel.")
            return None, None

        available_forecast_times = sorted(list(set(layer['timestamp'] for layer in self.data_layers)))
        if not available_forecast_times:
            # print("No forecast times available from loaded data.")
            return None, None

        # --- 1. Select the most appropriate forecast time ---
        # Prefer forecasts at or before the query_timestamp.
        past_or_current_forecasts = [t for t in available_forecast_times if t <= query_timestamp]
        
        selected_forecast_time: Optional[datetime] = None
        if past_or_current_forecasts:
            selected_forecast_time = max(past_or_current_forecasts)  # Latest forecast not after query time
        elif available_forecast_times: # All forecasts are in the future relative to query_timestamp
            selected_forecast_time = min(available_forecast_times) # Use the earliest available
            # print(f"Warning: Query timestamp {query_timestamp} is before the earliest forecast. Using earliest: {selected_forecast_time}")
        
        if selected_forecast_time is None:
            # print(f"Could not determine a suitable forecast time for query: {query_timestamp}")
            return None, None
        
        # print(f"Query timestamp: {query_timestamp}, Selected forecast time for data: {selected_forecast_time}")

        # --- 2. Filter layers for this specific forecast time ---
        # These layers are already sorted by min_alt due to the sort in load_forecast_data
        relevant_layers_for_time = [
            layer for layer in self.data_layers if layer['timestamp'] == selected_forecast_time
        ]

        if not relevant_layers_for_time:
            # This case should ideally not happen if selected_forecast_time was derived from available_forecast_times
            # print(f"No data layers found for the selected forecast time: {selected_forecast_time}")
            return None, None

        # --- 3. Select the correct dataset based on altitude ---
        selected_ds: Optional[xr.Dataset] = None
        # print(f"Searching for altitude {altitude} within layers for time {selected_forecast_time}:")
        # for l_idx, l in enumerate(relevant_layers_for_time):
            # print(f"  Layer {l_idx}: min_alt={l['min_alt']}")

        for i, current_layer in enumerate(relevant_layers_for_time):
            current_min_alt = current_layer['min_alt']
            
            # Determine the upper bound of this layer's validity
            next_min_alt_for_this_timestamp = np.inf
            if i + 1 < len(relevant_layers_for_time):
                # The next layer is guaranteed to be for the same timestamp because
                # relevant_layers_for_time is filtered by selected_forecast_time
                next_min_alt_for_this_timestamp = relevant_layers_for_time[i+1]['min_alt']

            if current_min_alt <= altitude < next_min_alt_for_this_timestamp:
                selected_ds = current_layer['dataset']
                # print(f"Selected layer for altitude {altitude}: min_alt {current_min_alt} (valid up to {next_min_alt_for_this_timestamp}) from {os.path.basename(current_layer['filepath'])}")
                break
        
        # If altitude is at or above the highest min_alt for the selected time, use the topmost layer
        if selected_ds is None and relevant_layers_for_time:
            last_layer_for_time = relevant_layers_for_time[-1]
            if altitude >= last_layer_for_time['min_alt']:
                 selected_ds = last_layer_for_time['dataset']
                 # print(f"Selected layer for altitude {altitude} (at or above highest): min_alt {last_layer_for_time['min_alt']} from {os.path.basename(last_layer_for_time['filepath'])}")

        if selected_ds is None:
            # print(f"No suitable altitude layer found for altitude {altitude} at forecast time {selected_forecast_time}.")
            return None, None

        # --- 4. Extract wind components using interpolation ---
        try:
            # Check data bounds (optional, as interp might handle it with NaNs)
            min_lat, max_lat = selected_ds['latitude'].min().item(), selected_ds['latitude'].max().item()
            min_lon, max_lon = selected_ds['longitude'].min().item(), selected_ds['longitude'].max().item()

            if not (min_lat <= latitude <= max_lat and min_lon <= longitude <= max_lon):
                # print(f"Warning: Lat/Lon ({latitude:.2f}, {longitude:.2f}) is outside the bounds of the selected dataset "
                #       f"([{min_lat:.2f}-{max_lat:.2f}], [{min_lon:.2f}-{max_lon:.2f}]). Interpolation may yield NaN.")
                pass # Let xarray's interp handle it; it will likely produce NaNs.

            # Use .interp() for robustness.
            # For 'nearest' or specific tolerance, kwargs can be added: method="linear", kwargs={"fill_value": "extrapolate"}
            data_at_point = selected_ds.interp(latitude=latitude, longitude=longitude)
            
            u_wind = data_at_point['u_wind'].item()
            v_wind = data_at_point['v_wind'].item()
            
            # Handle NaN results from interpolation (e.g., point is outside data coverage convex hull)
            if np.isnan(u_wind) or np.isnan(v_wind):
                # print(f"Interpolation resulted in NaN for lat={latitude:.2f}, lon={longitude:.2f}. "
                #       "Point might be outside effective data coverage for the selected dataset.")
                return None, None

            return u_wind, v_wind
        except Exception as e:
            # print(f"Error during data extraction/interpolation from dataset: {e}")
            return None, None

    def __str__(self) -> str:
        """
        Returns a human-readable string representation of the WindModel's current state.
        """
        if not self.data_layers:
            return "WindModel (No data loaded)"

        s = "WindModel State:\n"
        s += f"  Total data layers loaded: {len(self.data_layers)}\n"
        
        # Group layers by timestamp for structured display
        grouped_by_ts: Dict[datetime, List[Dict[str, Any]]] = {}
        for layer in self.data_layers:
            ts = layer['timestamp']
            if ts not in grouped_by_ts:
                grouped_by_ts[ts] = []
            grouped_by_ts[ts].append(layer) # Layers are already sorted by min_alt

        sorted_timestamps = sorted(grouped_by_ts.keys())

        s += "  Forecasts by Timestamp:\n"
        for ts in sorted_timestamps:
            s += f"    Timestamp: {ts.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            if not grouped_by_ts[ts]:
                s += "      - No layers for this timestamp (should not happen if data_layers is populated for this ts)\n"
            for layer in grouped_by_ts[ts]: # These are already sorted by min_alt within their original loading order
                s += f"      - Min Alt: {layer['min_alt']:<8} File: {os.path.basename(layer['filepath'])}\n"
        
        trenches_prop = self.trenches
        s += f"  Available Trenches (Unique Timestamp, Min Altitude):\n"
        if trenches_prop:
            for ts, m_alt in trenches_prop:
                s += f"    - ({ts.strftime('%Y-%m-%d %H:%M:%S UTC')}, {m_alt})\n"
        else:
            s += "    - None defined.\n"
            
        return s

    def close_all_datasets(self):
        """
        Closes all xarray datasets stored in the model.
        Useful for releasing file handles, especially before exiting or re-loading.
        """
        # print("Closing all datasets...")
        for layer in self.data_layers:
            if layer.get('dataset'):
                try:
                    layer['dataset'].close()
                    # print(f"Closed dataset from: {os.path.basename(layer['filepath'])}")
                except Exception as e:
                    print(f"Error closing dataset from {os.path.basename(layer['filepath'])}: {e}")
        # print("All datasets closed attempt complete.")

# Example Usage (can be removed or kept for testing):
if __name__ == '__main__':
    print("WindModel class defined. Example usage would be:")
    # Create dummy netCDF files for testing
    # This requires a directory structure and actual .nc files.
    # For a self-contained example, one might mock xr.open_dataset or create minimal files.

    # wind_model = WindModel()
    
    # Create dummy data directory if it doesn't exist
    # data_dir = 'project_root/data/wx/cdfs_dummy' # Relative to where script is run from
    # if not os.path.exists(data_dir):
    #     os.makedirs(data_dir, exist_ok=True)

    # Create some dummy nc files (simplified)
    # lats = np.arange(30, 70, 1.0)
    # lons = np.arange(-10, 40, 1.0)
    # time_now_unix = int(datetime.utcnow().timestamp())

    # def create_dummy_nc(filepath, lats, lons):
    #     data_vars = {
    #         'u_wind': (('latitude', 'longitude'), np.random.rand(len(lats), len(lons)) * 10),
    #         'v_wind': (('latitude', 'longitude'), np.random.rand(len(lats), len(lons)) * 10 - 5),
    #         'cape': (('latitude', 'longitude'), np.random.rand(len(lats), len(lons)) * 100),
    #         'cin': (('latitude', 'longitude'), np.random.rand(len(lats), len(lons)) * 50)
    #     }
    #     coords = {'latitude': lats, 'longitude': lons}
    #     ds = xr.Dataset(data_vars, coords=coords)
    #     ds.to_netcdf(filepath)
    #     ds.close()

    # # Example:
    # # forecast_time_unix_1 = time_now_unix
    # # forecast_time_unix_2 = time_now_unix + 3 * 3600
    
    # # path_fc1_alt0 = os.path.join(data_dir, f"{forecast_time_unix_1}_alt0.nc")
    # # path_fc1_alt5k = os.path.join(data_dir, f"{forecast_time_unix_1}_alt5000.nc")
    # # path_fc2_alt0 = os.path.join(data_dir, f"{forecast_time_unix_2}_alt0.nc")

    # # create_dummy_nc(path_fc1_alt0, lats, lons)
    # # create_dummy_nc(path_fc1_alt5k, lats, lons)
    # # create_dummy_nc(path_fc2_alt0, lats, lons)
    
    # # wind_model.load_forecast_data(path_fc1_alt0, 0)
    # # wind_model.load_forecast_data(path_fc1_alt5k, 5000)
    # # wind_model.load_forecast_data(path_fc2_alt0, 0)
    
    # # print(str(wind_model))
    
    # # query_dt = datetime.utcfromtimestamp(forecast_time_unix_1 + 1.5 * 3600) # 1.5 hours into the first forecast
    # # u, v = wind_model.get_wind_property(altitude=2500, latitude=50.5, longitude=10.5, query_timestamp=query_dt)
    # # if u is not None and v is not None:
    # #     print(f"Wind at alt 2500m, lat 50.5, lon 10.5, time {query_dt}: u={u:.2f} m/s, v={v:.2f} m/s")
    # # else:
    # #     print(f"Could not retrieve wind data for the query.")

    # # print(f"Trenches property: {wind_model.trenches}")

    # # wind_model.close_all_datasets() # Important to release file handles
    pass # End of example main block
