import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import networkx as nx
from utils.haversine import haversine_distance
from tqdm import tqdm
import os
import glob
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial




def find_best_graph_file():
    """Find the best available graph file, prioritizing comprehensive ones"""
    graph_candidates = [
        'data/graphs/ats_fra_nodes_only.gml',
        # 'data/graphs/waypoints_graph.graphml',
        # 'data/graphs/ats_graph.graphml',
        # 'route_graph_compute/LEMD_EGLL.gml'
    ]
    
    for graph_file in graph_candidates:
        if os.path.exists(graph_file):
            return graph_file
    
    # Fallback: find any available graph file
    graph_patterns = [
        'data/graphs/*.gml',
        'data/graphs/*.graphml',
        'route_graph_compute/*.gml',
        '*.gml'
    ]
    
    for pattern in graph_patterns:
        files = glob.glob(pattern)
        if files:
            return files[0]
    
    return None

def load_graph(graph_file_path):
    """Load the route graph from GML or GraphML file"""
    if not os.path.exists(graph_file_path):
        raise FileNotFoundError(f"Graph file not found: {graph_file_path}")
    
    if graph_file_path.endswith('.gml'):
        return nx.read_gml(graph_file_path)
    elif graph_file_path.endswith('.graphml'):
        return nx.read_graphml(graph_file_path)
    else:
        raise ValueError(f"Unsupported graph format: {graph_file_path}")

def haversine_vectorized(lat1, lon1, lat2, lon2):
    """
    Vectorized haversine distance calculation.
    
    Args:
        lat1, lon1: Arrays of latitudes and longitudes for first set of points
        lat2, lon2: Arrays of latitudes and longitudes for second set of points
        
    Returns:
        Array of distances in nautical miles
    """
    R = 3440.065  # Earth's radius in nautical miles
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

def find_nearest_airport_to_target_distance_vectorized(waypoint_lat, waypoint_lon, airports_df, target_distance=3.5):
    """
    Find the airport that is closest to being target_distance nautical miles from the waypoint.
    Uses vectorized operations for better performance.
    
    Args:
        waypoint_lat: Latitude of the waypoint
        waypoint_lon: Longitude of the waypoint  
        airports_df: DataFrame containing airport data (icao, latitude, longitude)
        target_distance: Target distance in nautical miles (default 3.5)
    
    Returns:
        tuple: (closest_airport_icao, actual_distance, distance_diff_from_target)
    """
    if airports_df.empty:
        return None, None, None
    
    # Calculate distances using vectorized operation
    distances = haversine_vectorized(
        waypoint_lat, waypoint_lon,
        airports_df['latitude'].values, airports_df['longitude'].values
    )
    
    # Find the airport closest to the target distance
    distance_diffs = np.abs(distances - target_distance)
    closest_idx = np.argmin(distance_diffs)
    
    closest_airport = airports_df.iloc[closest_idx]
    actual_distance = distances[closest_idx]
    distance_diff = distance_diffs[closest_idx]
    
    return closest_airport['icao'], actual_distance, distance_diff

def get_waypoint_coordinates(waypoint_name, graph):
    """
    Get coordinates of a waypoint from the graph.
    
    Args:
        waypoint_name: Name of the waypoint
        graph: NetworkX graph containing waypoint nodes with lat/lon attributes
        
    Returns:
        tuple: (lat, lon) or (None, None) if waypoint not found
    """
    if waypoint_name in graph.nodes:
        node_data = graph.nodes[waypoint_name]
        
        # Try different possible coordinate attribute names
        lat_attrs = ['lat', 'latitude', 'y']
        lon_attrs = ['lon', 'longitude', 'x']
        
        lat = None
        lon = None
        
        for lat_attr in lat_attrs:
            if lat_attr in node_data:
                lat = float(node_data[lat_attr])
                break
        
        for lon_attr in lon_attrs:
            if lon_attr in node_data:
                lon = float(node_data[lon_attr])
                break
        
        if lat is not None and lon is not None:
            return lat, lon
    
    return None, None

def process_single_route(route_data, graph, airports_df, target_distance=3.5):
    """
    Process a single flight route to replace first and last waypoints with airport matches.
    
    Args:
        route_data: Dictionary containing route information
        graph: NetworkX graph containing waypoint coordinates
        airports_df: DataFrame with airport data
        target_distance: Target distance in nautical miles for airport matching
        
    Returns:
        Dictionary with original data and modified waypoints
    """
    result = route_data.copy()
    
    waypoints_str = route_data['real_waypoints']
    waypoints = waypoints_str.strip().split()
    
    if len(waypoints) < 2:
        return result
    
    first_waypoint = waypoints[0]
    last_waypoint = waypoints[-1]
    modified_waypoints = waypoints.copy()
    
    # Process first waypoint
    if len(first_waypoint) != 4:  # Not already an airport
        first_lat, first_lon = get_waypoint_coordinates(first_waypoint, graph)
        if first_lat is not None and first_lon is not None:
            airport_icao, distance, diff = find_nearest_airport_to_target_distance_vectorized(
                first_lat, first_lon, airports_df, target_distance)
            if airport_icao is not None:
                modified_waypoints[0] = airport_icao
    
    # Process last waypoint
    if len(last_waypoint) != 4:  # Not already an airport
        last_lat, last_lon = get_waypoint_coordinates(last_waypoint, graph)
        if last_lat is not None and last_lon is not None:
            airport_icao, distance, diff = find_nearest_airport_to_target_distance_vectorized(
                last_lat, last_lon, airports_df, target_distance)
            if airport_icao is not None:
                modified_waypoints[-1] = airport_icao
    
    # Update the real_waypoints with modified waypoints
    result['real_waypoints'] = ' '.join(modified_waypoints)
    
    return result

def process_csv_file(file_path, graph, airports_df, target_distance=3.5):
    """
    Process a single CSV file to replace waypoints with airport matches.
    
    Args:
        file_path: Path to the CSV file to process
        graph: NetworkX graph containing waypoint coordinates
        airports_df: DataFrame with airport data
        target_distance: Target distance in nautical miles for airport matching
        
    Returns:
        String: Output file path
    """
    try:
        print(f"Processing: {file_path}")
        
        # Load the CSV file
        df = pd.read_csv(file_path)
        
        if 'real_waypoints' not in df.columns:
            print(f"Warning: 'real_waypoints' column not found in {file_path}")
            return None
        
        # Process each route
        processed_routes = []
        for _, row in df.iterrows():
            route_dict = row.to_dict()
            processed_route = process_single_route(route_dict, graph, airports_df, target_distance)
            processed_routes.append(processed_route)
        
        # Create output DataFrame
        output_df = pd.DataFrame(processed_routes)
        
        # Generate output file path
        input_filename = os.path.basename(file_path)
        output_filename = f"airport_matched_{input_filename}"
        output_path = os.path.join('filtered_data', output_filename)
        
        # Save the processed file
        output_df.to_csv(output_path, index=False)
        
        print(f"Completed: {file_path} -> {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def process_file_wrapper(args):
    """Wrapper function for multiprocessing"""
    file_path, graph, airports_df, target_distance = args
    return process_csv_file(file_path, graph, airports_df, target_distance)

def main():
    """Main function to process all CSV files with multiprocessing"""
    
    # Configuration
    input_directory = 'filtered_data'
    input_pattern = os.path.join(input_directory, '*.csv')
    airports_file = 'data/airac/airports_high.csv'
    target_distance = 3.5  # nautical miles
    
    print("=== MULTIPROCESSING AIRPORT MATCHING FOR FLIGHT ROUTES ===")
    
    # Find all CSV files to process
    csv_files = glob.glob(input_pattern)
    if not csv_files:
        print(f"No CSV files found in {input_directory}")
        return
    
    # Filter out already processed files
    csv_files = [f for f in csv_files if not os.path.basename(f).startswith('airport_matched_')]
    
    if not csv_files:
        print(f"No unprocessed CSV files found in {input_directory}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process:")
    for file_path in csv_files:
        print(f"  {file_path}")
    
    # Load airports data
    print(f"\nLoading airports from: {airports_file}")
    if not os.path.exists(airports_file):
        print(f"Error: Airports file not found: {airports_file}")
        return
    airports_df = pd.read_csv(airports_file)
    print(f"Loaded {len(airports_df)} airports")
    
    # Find and load graph
    graph_file = find_best_graph_file()
    if graph_file is None:
        print("Error: No graph files found. Please ensure a .gml or .graphml graph file is available.")
        return
    
    print(f"Using graph file: {graph_file}")
    
    try:
        print("Loading graph (this may take a moment for large files)...")
        graph = load_graph(graph_file)
        print(f"Loaded graph with {len(graph.nodes)} nodes")
    except Exception as e:
        print(f"Error loading graph: {e}")
        return
    
    # Determine number of processes
    num_processes = min(mp.cpu_count(), len(csv_files))
    print(f"\nUsing {num_processes} processes for parallel processing")
    
    # Prepare arguments for multiprocessing
    process_args = [(file_path, graph, airports_df, target_distance) for file_path in csv_files]
    
    # Process files in parallel
    print("\nStarting multiprocessing...")
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_file_wrapper, process_args),
            total=len(csv_files),
            desc="Processing files"
        ))
    
    # Count successful processing
    successful_files = [r for r in results if r is not None]
    failed_files = len(results) - len(successful_files)
    
    print(f"\n=== PROCESSING COMPLETE ===")
    print(f"Successfully processed: {len(successful_files)} files")
    if failed_files > 0:
        print(f"Failed to process: {failed_files} files")
    
    print(f"\nOutput files saved in: {input_directory}")
    print("Output files have the same format as input files but with airport-matched waypoints.")
    print("First and last waypoints are replaced with nearest airport ICAO codes when matches are found.")

if __name__ == "__main__":
    main() 