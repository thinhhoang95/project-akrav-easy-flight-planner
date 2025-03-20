import os
import pandas as pd
import multiprocessing

# Add PROJECT_ROOT to the Python path
import sys

import networkx as nx

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
    global G
    import networkx as nx
    G = nx.read_gml(graph_path)

def process_one_csv_file(args):
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
    from infer_route4 import find_route
    from tqdm import tqdm

    # Get unique flight IDs from the CSV
    flight_ids = routes_df['id'].unique()
    
    df_all_routes = pd.DataFrame(columns=['flight_id', 'real_waypoints', 'real_full_waypoints'])
    # df_synth_wps = pd.DataFrame(columns=['id', 'lat', 'lon'])

    for flight_id in tqdm(flight_ids, desc=f"Processing {os.path.basename(csv_file_path)}"):
        selected_flight_df = routes_df[routes_df['id'] == flight_id]
        real_waypoints, final_route, new_nodes = find_route(G, selected_flight_df, 
                                                                                error_threshold=15,
                                                                                distance_threshold_for_segment_skipping=20, 
                                                                                max_wp_search_radius=12, 
                                                                                min_wp_search_radius=2)
        df_all_routes = pd.concat([
            df_all_routes, 
            pd.DataFrame({
                'flight_id': [flight_id],
                'real_waypoints': [' '.join(real_waypoints)],
                'real_full_waypoints': ''
            })
        ], ignore_index=True)

        # # Add synthetic waypoints to df_synth_wps
        # for node_id, node_data in new_nodes.items():
        #     df_synth_wps = pd.concat([
        #         df_synth_wps, 
        #         pd.DataFrame({
        #             'id': [node_id],
        #             'lat': [node_data['lat']],
        #             'lon': [node_data['lon']]
        #         })
        #     ], ignore_index=True)

    # Write the outputs to CSV files in the output folder
    base_name = os.path.basename(csv_file_path)
    name_without_ext = os.path.splitext(base_name)[0]
    routes_output_file = os.path.join(output_folder, f"{name_without_ext}.routes.csv")
    wps_output_file = os.path.join(output_folder, f"{name_without_ext}.wps.csv")

    df_all_routes.to_csv(routes_output_file, index=False)
    # df_synth_wps.to_csv(wps_output_file, index=False)

    return f"Processed {csv_file_path}"

if __name__ == '__main__':

    import sys
    import os
    
    # Determine input and output folders
    if len(sys.argv) > 1:
        input_folder = sys.argv[1]
    else:
        input_folder = os.path.join('routes')
    
    output_folder = os.path.join('output')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get all CSV files recursively from the input folder
    csv_files = get_all_csv_files(input_folder)
    
    # Prepare tasks as (csv_file_path, output_folder) tuples
    tasks = [(csv_file, output_folder) for csv_file in csv_files]
    
    # Define graph path for worker initialization
    graph_path = os.path.join('ats_fra_nodes_only.gml')
    
    # Process CSV files using multiprocessing
    pool = multiprocessing.Pool(initializer=init_worker, initargs=(graph_path,))
    results = pool.map(process_one_csv_file, tasks)
    pool.close()
    pool.join()
    
    for res in results:
        print(res)
