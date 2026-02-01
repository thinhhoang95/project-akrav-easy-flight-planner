"""
Batch route inference server using multiprocessing.

This module processes folders of CSV files containing flight route segments,
infers full routes on an ATS (Air Traffic Services) graph, and writes per-file
outputs: one CSV of inferred routes (waypoints, times, speeds, altitudes) and
one CSV of synthetic waypoint coordinates. It uses a process pool so each worker
loads the graph once and then processes one or more input CSVs.

**What it does**
- Discovers all CSV files recursively under an input folder.
- For each CSV: reads segment rows (from_lat, from_lon, to_lat, to_lon, id, etc.),
  groups by flight id, and for each flight (up to ``--max-ids`` per file, default 500) calls
  find_route() from infer_route52 to get real/full waypoints and synthetic nodes.
- Writes two outputs per input file: ``<basename>.routes.csv`` and ``<basename>.wps.csv``.
- Skips a file if its ``.routes.csv`` already exists.
- Flights with a single segment or with total route distance < 30 NM are skipped.

**Input CSV**
- Expected columns include at least: ``id`` (flight id), ``from_lat``, ``from_lon``,
  ``to_lat``, ``to_lon`` (segment endpoints). One row per segment; multiple rows
  per flight for multi-segment routes.

**Output files**
- ``<basename>.routes.csv``: columns such as ``flight_id``, ``real_waypoints``,
  ``pass_times``, ``speeds``, ``alts``, ``real_full_waypoints``, ``full_pass_times``,
  ``full_speeds``, ``full_alts`` (list-like values from find_route).
- ``<basename>.wps.csv``: columns ``id``, ``lat``, ``lon`` for synthetic waypoints
  created during route inference.

**Parameters (main script)**
- ``input_folder``: Directory to search for CSV files (default: ``data/routes``).
- ``output_folder``: Directory for ``.routes.csv`` and ``.wps.csv`` (default: ``output``).
- Graph path is configurable via ``--graph`` (default: ``data/graphs/ats_fra_nodes_only.gml``).

**Example (command line)**

  python infer_route_auto_server.py --input data/routes --output output
  python infer_route_auto_server.py -i /path/to/csv/folder -o /path/to/out --max-ids 1000
  python infer_route_auto_server.py --input data/routes --output output --limit-files 2  # process only first 2 CSVs

**Example (input CSV snippet)**

  id,from_lat,from_lon,to_lat,to_lon,...
  FLT001,50.1,8.6,50.2,8.7,...
  FLT001,50.2,8.7,50.3,8.8,...
  FLT002,51.0,7.0,51.1,7.1,...

**Example (output .routes.csv columns)**

  flight_id,real_waypoints,pass_times,speeds,alts,real_full_waypoints,full_pass_times,full_speeds,full_alts

**Example (output .wps.csv columns)**

  id,lat,lon
  SYNTH_WP_1,50.15,8.65
  SYNTH_WP_2,50.25,8.75
"""

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
    Return paths to all CSV files under a directory, recursively.

    Parameters
    ----------
    folder_path : str or path-like
        Root directory to search for CSV files (searches all subdirectories).

    Returns
    -------
    list of str
        Absolute or relative paths to every file whose name ends with ``.csv``
        (case-insensitive), in no guaranteed order.
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
    """
    Initialize a worker process with a shared graph and spatial index.

    Called once per worker when using multiprocessing.Pool(initializer=init_worker, ...).
    Loads the GML graph from graph_path, sets cell_size = 0.5, and builds a
    spatial index mapping (cell_x, cell_y) to lists of (node_id, lat, lon) for
    graph nodes that have "lat" and "lon" attributes. The graph and index are
    stored in global variables used by process_one_csv_file.

    Parameters
    ----------
    graph_path : str or path-like
        Path to the GML file (e.g. ats_fra_nodes_only.gml) to load as a NetworkX graph.
    """
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
    """
    Process a single input CSV: infer routes for each flight and write output CSVs.

    Expects the process to have been initialized with init_worker so that the
    global graph G, spatial_index, and cell_size are set. Reads the CSV (with
    columns including id, from_lat, from_lon, to_lat, to_lon), groups by flight id,
    and for each flight (up to max_flight_ids per file) with more than one segment
    and total distance >= 30 NM calls find_route from infer_route52. Writes
    <basename>.routes.csv and <basename>.wps.csv into output_folder. If
    <basename>.routes.csv already exists, the file is skipped and a skip message
    is returned.

    Parameters
    ----------
    args : tuple of (str, str, int)
        (csv_file_path, output_folder, max_flight_ids): path to the input CSV,
        directory for the two output files, and maximum number of flight IDs to
        process per file (None = no limit).

    Returns
    -------
    str
        Status message: "Processed <path>", "Skipped <path> (outputs already exist)",
        or "Error processing <path>: <exception>".
    """
    try:
        csv_file_path, output_folder, max_flight_ids = args
        
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
        from infer_route52 import find_route, haversine_distance # should be from infer_route5-1
        from tqdm import tqdm

        # Get unique flight IDs from the CSV
        flight_ids = routes_df['id'].unique()
        if max_flight_ids is not None:
            flight_ids = flight_ids[:max_flight_ids]

        df_all_routes = pd.DataFrame(columns=['flight_id', 'real_waypoints', 'pass_times', 'speeds', 'alts', 'real_full_waypoints', 'full_pass_times', 'full_speeds', 'full_alts'])
        df_synth_wps = pd.DataFrame(columns=['id', 'lat', 'lon'])

        for flight_id in tqdm(flight_ids, desc=f"Processing {os.path.basename(csv_file_path)}"):
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
                                                                'alts': real_waypoints[3],
                                                                'real_full_waypoints': real_full_waypoints[0],
                                                                'full_pass_times': real_full_waypoints[1],
                                                                'full_speeds': real_full_waypoints[2],
                                                                'full_alts': real_full_waypoints[3]
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

def parse_args():
    """Parse command-line arguments for the route inference batch server."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Batch route inference: process folders of CSV route segments and write inferred routes and waypoints.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        default=os.path.join('data', 'routes'),
        help='Input directory to search recursively for CSV files.',
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='output',
        help='Output directory for .routes.csv and .wps.csv files.',
    )
    parser.add_argument(
        '--max-ids',
        type=int,
        default=500,
        metavar='N',
        help='Maximum number of flight IDs to process per CSV file (use 0 for no limit).',
    )
    parser.add_argument(
        '--graph',
        type=str,
        default=os.path.join('data', 'graphs', 'ats_fra_nodes_only.gml'),
        help='Path to the GML graph file (ATS waypoints).',
    )
    parser.add_argument(
        '--limit-files',
        type=int,
        default=None,
        metavar='N',
        help='Process only the first N CSV files (for testing). Default: process all.',
    )
    parser.add_argument(
        '-j', '--jobs',
        type=int,
        default=None,
        metavar='N',
        help='Number of worker processes. Default: use CPU count.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    input_folder = args.input
    output_folder = args.output
    max_flight_ids = args.max_ids if args.max_ids else None  # 0 means no limit
    graph_path = args.graph
    limit_files = args.limit_files
    n_workers = args.jobs

    if not os.path.isdir(input_folder):
        print(f"Error: input directory does not exist: {input_folder}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_folder, exist_ok=True)

    csv_files = get_all_csv_files(input_folder)
    csv_files.sort()

    tasks = [(path, output_folder, max_flight_ids) for path in csv_files]
    tasks = [t for t in tasks if '._' not in t[0] and 'checkpoint' not in t[0]]

    if limit_files is not None:
        tasks = tasks[:limit_files]

    if not tasks:
        print("No CSV files to process.")
        sys.exit(0)

    print(f"Input: {input_folder}, Output: {output_folder}, Graph: {graph_path}")
    print(f"Max flight IDs per file: {max_flight_ids or 'no limit'}, Files: {len(tasks)}, Workers: {n_workers or 'auto'}")

    pool = multiprocessing.Pool(initializer=init_worker, initargs=(graph_path,), processes=n_workers)
    results = pool.map(process_one_csv_file, tasks)
    pool.close()
    pool.join()

    for res in results:
        print(res)
