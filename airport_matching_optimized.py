import pandas as pd
import numpy as np
import networkx as nx
from utils.haversine import haversine_distance
from tqdm import tqdm
import os
import glob
from scipy.spatial.distance import cdist

def find_available_graph_files():
    """Find available graph files in the project"""
    graph_patterns = [
        'route_graph_compute/*.gml',
        '*.gml',
        'data/**/*.gml'
    ]
    
    graph_files = []
    for pattern in graph_patterns:
        graph_files.extend(glob.glob(pattern, recursive=True))
    
    return graph_files

def load_graph(graph_file_path):
    """Load the route graph from GML file"""
    if not os.path.exists(graph_file_path):
        raise FileNotFoundError(f"Graph file not found: {graph_file_path}")
    return nx.read_gml(graph_file_path)

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
        if 'lat' in node_data and 'lon' in node_data:
            return float(node_data['lat']), float(node_data['lon'])
    return None, None

def process_flight_routes_for_airport_matching(routes_df, graph, airports_df, target_distance=3.5, batch_size=1000):
    """
    Process flight routes to match first and last waypoints to nearest airports.
    Uses batched processing for better performance.
    
    Args:
        routes_df: DataFrame with flight routes (must have 'flight_id' and 'real_waypoints' columns)
        graph: NetworkX graph containing waypoint coordinates
        airports_df: DataFrame with airport data
        target_distance: Target distance in nautical miles for airport matching
        batch_size: Number of routes to process in each batch
        
    Returns:
        DataFrame with original data plus airport matching results
    """
    results = []
    
    print("Processing flight routes for airport matching...")
    
    # Process in batches for better memory management
    for start_idx in tqdm(range(0, len(routes_df), batch_size), desc="Processing batches"):
        end_idx = min(start_idx + batch_size, len(routes_df))
        batch_df = routes_df.iloc[start_idx:end_idx]
        
        batch_results = []
        
        for idx, row in batch_df.iterrows():
            flight_id = row['flight_id']
            waypoints_str = row['real_waypoints']
            
            # Parse waypoints
            waypoints = waypoints_str.strip().split()
            if len(waypoints) < 2:
                # Skip routes with less than 2 waypoints
                continue
                
            first_waypoint = waypoints[0]
            last_waypoint = waypoints[-1]
            
            # Initialize result dictionary
            result = {
                'flight_id': flight_id,
                'first_waypoint': first_waypoint,
                'last_waypoint': last_waypoint,
                'first_waypoint_is_airport': len(first_waypoint) == 4,
                'last_waypoint_is_airport': len(last_waypoint) == 4,
                'first_waypoint_lat': None,
                'first_waypoint_lon': None,
                'last_waypoint_lat': None,
                'last_waypoint_lon': None,
                'first_matched_airport': None,
                'first_airport_distance': None,
                'first_distance_diff_from_target': None,
                'last_matched_airport': None,
                'last_airport_distance': None,
                'last_distance_diff_from_target': None
            }
            
            # Process first waypoint
            if len(first_waypoint) == 4:
                # First waypoint is already an airport
                result['first_matched_airport'] = first_waypoint
                result['first_airport_distance'] = 0.0
                result['first_distance_diff_from_target'] = target_distance
                
                # Still get coordinates if available in graph
                first_lat, first_lon = get_waypoint_coordinates(first_waypoint, graph)
                result['first_waypoint_lat'] = first_lat
                result['first_waypoint_lon'] = first_lon
            else:
                # Look up coordinates in graph
                first_lat, first_lon = get_waypoint_coordinates(first_waypoint, graph)
                result['first_waypoint_lat'] = first_lat
                result['first_waypoint_lon'] = first_lon
                
                if first_lat is not None and first_lon is not None:
                    # Find nearest airport to target distance
                    airport_icao, distance, diff = find_nearest_airport_to_target_distance_vectorized(
                        first_lat, first_lon, airports_df, target_distance)
                    result['first_matched_airport'] = airport_icao
                    result['first_airport_distance'] = distance
                    result['first_distance_diff_from_target'] = diff
            
            # Process last waypoint
            if len(last_waypoint) == 4:
                # Last waypoint is already an airport
                result['last_matched_airport'] = last_waypoint
                result['last_airport_distance'] = 0.0
                result['last_distance_diff_from_target'] = target_distance
                
                # Still get coordinates if available in graph
                last_lat, last_lon = get_waypoint_coordinates(last_waypoint, graph)
                result['last_waypoint_lat'] = last_lat
                result['last_waypoint_lon'] = last_lon
            else:
                # Look up coordinates in graph
                last_lat, last_lon = get_waypoint_coordinates(last_waypoint, graph)
                result['last_waypoint_lat'] = last_lat
                result['last_waypoint_lon'] = last_lon
                
                if last_lat is not None and last_lon is not None:
                    # Find nearest airport to target distance
                    airport_icao, distance, diff = find_nearest_airport_to_target_distance_vectorized(
                        last_lat, last_lon, airports_df, target_distance)
                    result['last_matched_airport'] = airport_icao
                    result['last_airport_distance'] = distance
                    result['last_distance_diff_from_target'] = diff
            
            batch_results.append(result)
        
        results.extend(batch_results)
    
    return pd.DataFrame(results)

def main():
    """Main function to process flight routes and match airports"""
    
    # Configuration
    routes_file = 'data/routes_wps/cs_2023-04-01.routes.csv'
    airports_file = 'data/airac/airports.csv'
    output_file = 'airport_matched_routes.csv'
    target_distance = 3.5  # nautical miles
    max_routes = None  # Set to a number like 1000 for testing, None for all routes
    
    print("=== AIRPORT MATCHING FOR FLIGHT ROUTES ===")
    print("Loading data...")
    
    # Load routes data
    print(f"Loading routes from: {routes_file}")
    if not os.path.exists(routes_file):
        print(f"Error: Routes file not found: {routes_file}")
        return
    routes_df = pd.read_csv(routes_file)
    
    # Limit routes for testing if specified
    if max_routes is not None:
        routes_df = routes_df.head(max_routes)
        print(f"Limited to first {max_routes} routes for testing")
    
    print(f"Processing {len(routes_df)} flight routes")
    
    # Load airports data
    print(f"Loading airports from: {airports_file}")
    if not os.path.exists(airports_file):
        print(f"Error: Airports file not found: {airports_file}")
        return
    airports_df = pd.read_csv(airports_file)
    print(f"Loaded {len(airports_df)} airports")
    
    # Find and select graph file
    available_graphs = find_available_graph_files()
    if not available_graphs:
        print("Error: No graph files found. Please ensure a .gml graph file is available.")
        return
    
    # Use the first available graph file (you can modify this logic)
    graph_file = available_graphs[0]
    print(f"Available graph files: {available_graphs}")
    print(f"Using graph file: {graph_file}")
    
    # Load graph
    try:
        graph = load_graph(graph_file)
        print(f"Loaded graph with {len(graph.nodes)} nodes")
    except Exception as e:
        print(f"Error loading graph: {e}")
        return
    
    # Process routes
    results_df = process_flight_routes_for_airport_matching(
        routes_df, graph, airports_df, target_distance)
    
    # Save results
    print(f"Saving results to: {output_file}")
    results_df.to_csv(output_file, index=False)
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Total flights processed: {len(results_df)}")
    
    first_airport_matches = results_df['first_matched_airport'].notna().sum()
    last_airport_matches = results_df['last_matched_airport'].notna().sum()
    
    print(f"First waypoint airport matches: {first_airport_matches}/{len(results_df)} ({first_airport_matches/len(results_df)*100:.1f}%)")
    print(f"Last waypoint airport matches: {last_airport_matches}/{len(results_df)} ({last_airport_matches/len(results_df)*100:.1f}%)")
    
    # Airport-to-airport routes
    both_matched = (results_df['first_matched_airport'].notna() & 
                   results_df['last_matched_airport'].notna()).sum()
    print(f"Complete airport-to-airport routes: {both_matched}/{len(results_df)} ({both_matched/len(results_df)*100:.1f}%)")
    
    # Waypoints that are already airports
    first_already_airports = results_df['first_waypoint_is_airport'].sum()
    last_already_airports = results_df['last_waypoint_is_airport'].sum()
    print(f"First waypoints that are already airports: {first_already_airports}/{len(results_df)} ({first_already_airports/len(results_df)*100:.1f}%)")
    print(f"Last waypoints that are already airports: {last_already_airports}/{len(results_df)} ({last_already_airports/len(results_df)*100:.1f}%)")
    
    # Distance statistics for matched airports (excluding already-airport waypoints)
    first_distances = results_df[(results_df['first_airport_distance'].notna()) & 
                                (results_df['first_airport_distance'] > 0)]['first_airport_distance']
    last_distances = results_df[(results_df['last_airport_distance'].notna()) & 
                               (results_df['last_airport_distance'] > 0)]['last_airport_distance']
    
    if len(first_distances) > 0:
        print(f"\nFirst waypoint airport distances (NM) - excluding already-airport waypoints:")
        print(f"  Mean: {first_distances.mean():.2f}")
        print(f"  Median: {first_distances.median():.2f}")
        print(f"  Min: {first_distances.min():.2f}")
        print(f"  Max: {first_distances.max():.2f}")
        
        # How many are close to target distance?
        within_1nm = (np.abs(first_distances - target_distance) <= 1.0).sum()
        print(f"  Within 1 NM of target ({target_distance} NM): {within_1nm}/{len(first_distances)} ({within_1nm/len(first_distances)*100:.1f}%)")
    
    if len(last_distances) > 0:
        print(f"\nLast waypoint airport distances (NM) - excluding already-airport waypoints:")
        print(f"  Mean: {last_distances.mean():.2f}")
        print(f"  Median: {last_distances.median():.2f}")
        print(f"  Min: {last_distances.min():.2f}")
        print(f"  Max: {last_distances.max():.2f}")
        
        # How many are close to target distance?
        within_1nm = (np.abs(last_distances - target_distance) <= 1.0).sum()
        print(f"  Within 1 NM of target ({target_distance} NM): {within_1nm}/{len(last_distances)} ({within_1nm/len(last_distances)*100:.1f}%)")
    
    print(f"\nResults saved to: {output_file}")
    print("\nOutput file contains the following columns:")
    print("- flight_id: Original flight identifier")
    print("- first_waypoint, last_waypoint: First and last waypoints from route")
    print("- first_waypoint_is_airport, last_waypoint_is_airport: Boolean indicating if waypoint is already an airport (4 chars)")
    print("- first_waypoint_lat/lon, last_waypoint_lat/lon: Coordinates from graph")
    print("- first_matched_airport, last_matched_airport: Matched airport ICAO codes")
    print("- first_airport_distance, last_airport_distance: Distance to matched airport (NM)")
    print("- first_distance_diff_from_target, last_distance_diff_from_target: Difference from target distance")

if __name__ == "__main__":
    main() 