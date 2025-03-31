import os
import pandas as pd
import multiprocessing

# Add PROJECT_ROOT to the Python path
import sys

import networkx as nx
import math 
from infer_route5 import get_cell

def get_all_csv_files(folder_path):
    """
    Get all CSV files recursively inside a folder.
    
    Parameters:
        folder_path: Path to the folder to search for CSV files
    
    Returns:
        list: List of paths to all CSV files found in the folder and its subfolders
    """
    import os
    
    csv_files = []
    
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(folder_path):
        # Filter for CSV files
        for file in files:
            if file.lower().endswith('.csv'):
                # Append the full path to the list
                csv_files.append(os.path.join(root, file))
    
    return csv_files

def init_worker(graph_path):
    global G, spatial_index, cell_size
    import networkx as nx
    G = nx.read_gml(graph_path)
    
    cell_size = 0.5 
    spatial_index = {}

    # Build spatial index from existing graph nodes
    for node, data in G.nodes(data=True):
        if "lat" in data and "lon" in data:
            cell = get_cell(data["lat"], data["lon"], cell_size)
            if cell not in spatial_index:
                spatial_index[cell] = []
            spatial_index[cell].append((node, data["lat"], data["lon"]))

def process_one_csv_file(args):
    try:
        csv_file_path, output_folder = args
        
        # Check if output files already exist, and skip processing if they do
        base_name = os.path.basename(csv_file_path)
        name_without_ext = os.path.splitext(base_name)[0]
        routes_output_file = os.path.join(output_folder, f"{name_without_ext}.routes.csv")
        wps_output_file = os.path.join(output_folder, f"{name_without_ext}.wps.csv")
        
        # If both output files exist, skip processing this CSV file
        if os.path.exists(routes_output_file):
            print(f"Skipping {csv_file_path} - output files already exist")
            return f"Skipped {csv_file_path} (outputs already exist)"
        
        routes_df = pd.read_csv(csv_file_path)
        from infer_route51 import find_route, haversine_distance # should be from infer_route5-1
        from tqdm import tqdm

        # Get unique flight IDs from the CSV
        flight_ids = routes_df['id'].unique()
        
        df_all_routes = pd.DataFrame(columns=['flight_id', 'real_waypoints', 'pass_times', 'speeds', 'real_full_waypoints', 'full_pass_times', 'full_speeds'])
        df_synth_wps = pd.DataFrame(columns=['id', 'lat', 'lon'])

        for flight_id in tqdm(flight_ids[:10], desc=f"Processing {os.path.basename(csv_file_path)}"):
            selected_flight_df = routes_df[routes_df['id'] == flight_id]
            # Skip route inference if the flight has only one segment
            if len(selected_flight_df) <= 1:
                continue
        
            # Check route length and coordinates
            first_point = selected_flight_df.iloc[0]
            last_point = selected_flight_df.iloc[-1]
            
            # Calculate total route distance
            route_distance = haversine_distance(
                first_point['from_lat'], first_point['from_lon'], 
                last_point['to_lat'], last_point['to_lon']
            )
            
            # Skip route if total distance is less than 30 nautical miles
            if route_distance < 30:
                continue
            real_waypoints, real_full_waypoints, new_nodes = find_route(G, selected_flight_df, error_threshold=25,
                                                                                distance_threshold_for_segment_skipping=25, max_wp_search_radius=12, min_wp_search_radius=3,
                                                                                spatial_index=spatial_index, cell_size=cell_size)
            df_all_routes = pd.concat([df_all_routes, pd.DataFrame({'flight_id': [flight_id],
                                                                'real_waypoints': real_waypoints[0],
                                                                'pass_times': real_waypoints[1],
                                                                'speeds': real_waypoints[2],
                                                                'real_full_waypoints': real_full_waypoints[0],
                                                                'full_pass_times': real_full_waypoints[1],
                                                                'full_speeds': real_full_waypoints[2],
                                                                })], ignore_index=True)

            # Add synthetic waypoints to df_synth_wps
            for node_id, node_data in new_nodes.items():
                df_synth_wps = pd.concat([df_synth_wps, pd.DataFrame({
                    'id': [node_id],
                    'lat': [node_data['lat']],
                    'lon': [node_data['lon']]
                })], ignore_index=True)

        # Write the outputs to CSV files in the output folder
        base_name = os.path.basename(csv_file_path)
        name_without_ext = os.path.splitext(base_name)[0]
        routes_output_file = os.path.join(output_folder, f"{name_without_ext}.routes.csv")
        wps_output_file = os.path.join(output_folder, f"{name_without_ext}.wps.csv")

        df_all_routes.to_csv(routes_output_file, index=False)
        df_synth_wps.to_csv(wps_output_file, index=False)

        return f"Processed {csv_file_path}"
    except Exception as e:
        return f"Error processing {csv_file_path}: {e}"

if __name__ == '__main__':

    import sys
    import os
    
    # Determine input and output folders
    if len(sys.argv) > 1:
        input_folder = sys.argv[1]
    else:
        input_folder = os.path.join('data', 'routes')
    
    output_folder = os.path.join('output')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get all CSV files recursively from the input folder
    csv_files = get_all_csv_files(input_folder)

    # Sort CSV files alphabetically
    csv_files.sort()
    
    # Prepare tasks as (csv_file_path, output_folder) tuples
    tasks = [(csv_file, output_folder) for csv_file in csv_files]

    # Remove files starting with '._' from tasks
    tasks = [task for task in tasks if not '._' in task[0]]

    # Remove the files with checkpoint in the filename
    tasks = [task for task in tasks if not 'checkpoint' in task[0]]

    # Pick only one first file for testing
    tasks = tasks[:1]
    user_input = input(f'WARNING: Only processing one file for testing. Do you want to continue? (y/n): ')
    if user_input.lower() != 'y':
        print("Exiting program.")
        sys.exit(0)
    
    # Define graph path for worker initialization
    graph_path = os.path.join('data', 'graphs', 'ats_fra_nodes_only.gml')

    # Process CSV files using multiprocessing
    pool = multiprocessing.Pool(initializer=init_worker, initargs=(graph_path,))
    results = pool.map(process_one_csv_file, tasks)
    pool.close()
    pool.join()
    
    for res in results:
        print(res)
