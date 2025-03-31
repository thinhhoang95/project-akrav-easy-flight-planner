import networkx as nx
import pandas as pd
import random
import string
from math import radians, sin, cos, sqrt, atan2, inf, asin

import math

# Helper function for spatial indexing
def get_cell(lat, lon, cell_size):
    """Return a tuple (cell_x, cell_y) for the given latitude and longitude based on cell_size in degrees."""
    return (int(math.floor(lat / cell_size)), int(math.floor(lon / cell_size)))

# Constant: Earth's radius in nautical miles.
EARTH_RADIUS_NM = 3440.065

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on Earth.
    
    Parameters:
        lat1, lon1: Latitude and longitude of the first point in decimal degrees
        lat2, lon2: Latitude and longitude of the second point in decimal degrees
    
    Returns:
        float: Distance between points in nautical miles
    """
    # Convert decimal degrees to radians.
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1 
    dlon = lon2 - lon1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return EARTH_RADIUS_NM * c

def generate_random_node_name(num_chars=8):
    """
    Generate a random node name with a specified number of characters.
    
    Parameters:
        num_chars: Integer specifying the length of the random string (default: 8)
    
    Returns:
        str: A random string prefixed with underscore
    """
    return '_' + ''.join(random.choices(string.ascii_uppercase + string.digits + string.ascii_lowercase, k=num_chars))

def find_best_candidate(point, radius, spatial_index, cell_size):
    """
    Find the closest node in the graph to a given point within a specified radius using a spatial index.
    
    Parameters:
        point: Tuple (lat, lon) of the point to find candidates for
        graph: NetworkX graph containing nodes with lat/lon attributes
        radius: Maximum search radius in nautical miles
        spatial_index: Dictionary mapping cell coordinates to lists of nodes [(node, lat, lon), ...]
        cell_size: The size of each cell in degrees
    
    Returns:
        tuple: (node_name, error_distance) where node_name is the closest node within radius,
               and error_distance is the distance in nautical miles. If no nodes are found within radius,
               returns (None, inf).
    """
    lat, lon = point
    best_node = None
    best_error = inf
    
    # Convert search radius (NM) to degrees (approximation: 1 deg latitude ~ 60 NM)
    degrees_radius = radius / 60.0
    
    # Get the base cell of the point
    base_cell = get_cell(lat, lon, cell_size)
    
    # Determine how many cells to search in each direction (be generous by adding 1 extra cell)
    cell_range = int(math.ceil(degrees_radius / cell_size)) + 1
    
    # Search in adjacent cells
    for i in range(base_cell[0] - cell_range, base_cell[0] + cell_range + 1):
        for j in range(base_cell[1] - cell_range, base_cell[1] + cell_range + 1):
            cell = (i, j)
            if cell in spatial_index:
                for node, node_lat, node_lon in spatial_index[cell]:
                    dist = haversine_distance(lat, lon, node_lat, node_lon)
                    # print(f'Distance from {node} to {lat}, {lon}: {dist}')
                    if dist <= radius and dist < best_error:
                        best_error = dist
                        best_node = node
    return best_node, best_error

def process_flight_segments(graph, segments_df, error_threshold=7.5, max_radius=12, min_radius=2,
                            spatial_index=None, cell_size=0.5):
    """
    Process flight segments to create a route by matching segments to existing waypoints or
    creating new nodes when necessary, using spatial indexing to improve performance.
    
    Parameters:
        graph: NetworkX graph containing existing waypoint nodes
        segments_df: DataFrame containing flight segments with columns:
                     from_lat, from_lon, to_lat, to_lon, from_time, to_time, from_speed, to_speed
        error_threshold: Maximum acceptable total error (in NM) for using existing waypoints
        max_radius: Maximum search radius (in NM) for finding candidate waypoints
        min_radius: Minimum search radius (in NM) for finding candidate waypoints
    
    Returns:
        tuple: (route, new_nodes) where:
               - route is a list of tuples (from_node, to_node, distance)
               - new_nodes is a dictionary of newly created nodes {node_name: {'lat': lat, 'lon': lon}}
    """
    # Directly use the input graph as the new graph.
    new_graph = graph
    full_route = []
    new_nodes = {}
    pass_times = []
    speeds = []

    # start_time = time.time()
    # Process each flight segment (each row in the dataframe).
    for idx, row in segments_df.iterrows():
        # Flight endpoints (using latitude, longitude)
        point_A = (row["from_lat"], row["from_lon"])
        point_B = (row["to_lat"], row["to_lon"])
        time_A = row["from_time"]
        time_B = row["to_time"]
        spd_A = row["from_speed"]
        spd_B = row["to_speed"]
        pass_times.append((time_A, time_B))
        speeds.append((spd_A, spd_B))

        # Compute the segment length (great circle distance)
        seg_length = haversine_distance(*point_A, *point_B)
        
        # Search radius: 10% of segment length, capped between min_radius and max_radius.
        search_radius = max(min(0.1 * seg_length, max_radius), min_radius)
        
        # Find best candidate for each endpoint using the spatial index
        cand_A, error_A = find_best_candidate(point_A, search_radius, spatial_index, cell_size)
        cand_B, error_B = find_best_candidate(point_B, search_radius, spatial_index, cell_size)
        
        # If no candidate is found, treat that endpoint as requiring a new node.
        if cand_A is None:
            error_A = 0  # new node placed exactly at the flight endpoint
        if cand_B is None:
            error_B = 0
            
        total_error = error_A + error_B
        number_of_nodes_added = 0
        
        # If the error is within threshold, we use the candidate pair (or new node if candidate missing)
        if total_error <= error_threshold:
            node_A = cand_A if cand_A is not None else generate_and_add_node(new_nodes, point_A, new_graph, spatial_index, cell_size)
            node_B = cand_B if cand_B is not None else generate_and_add_node(new_nodes, point_B, new_graph, spatial_index, cell_size)
            number_of_nodes_added = 0
        else:
            # Try the two options of adding one new node:
            # Option 1: New node at point_A, keep candidate for B.
            opt1_error = 0 + error_B if cand_B is not None else inf
            # Option 2: New node at point_B, keep candidate for A.
            opt2_error = error_A + 0 if cand_A is not None else inf
            
            if min(opt1_error, opt2_error) < error_threshold:
                if opt1_error <= opt2_error:
                    # Choose Option 1: new node for A.
                    node_A = generate_and_add_node(new_nodes, point_A, new_graph, spatial_index, cell_size)
                    node_B = cand_B
                    total_error = 0 + error_B
                    number_of_nodes_added = 1
                else:
                    # Option 2: new node for B.
                    node_A = cand_A
                    node_B = generate_and_add_node(new_nodes, point_B, new_graph, spatial_index, cell_size)
                    total_error = error_A + 0
                    number_of_nodes_added = 1
            else:
                # If neither option brings the error below threshold, add new nodes for both endpoints.
                node_A = generate_and_add_node(new_nodes, point_A, new_graph, spatial_index, cell_size)
                node_B = generate_and_add_node(new_nodes, point_B, new_graph, spatial_index, cell_size)
                total_error = 0  # error is now 0 as both are exact.
                number_of_nodes_added = 2
        
        # Add an edge between the chosen nodes if it doesn't exist.
        if not new_graph.has_edge(node_A, node_B):
            edge_attrs = { "distance": seg_length }
            new_graph.add_edge(node_A, node_B, **edge_attrs)
        
        # Calculate the actual distance between the chosen nodes
        distance_AB = haversine_distance(
            new_graph.nodes[node_A].get("lat", new_nodes.get(node_A, {}).get("lat")),
            new_graph.nodes[node_A].get("lon", new_nodes.get(node_A, {}).get("lon")),
            new_graph.nodes[node_B].get("lat", new_nodes.get(node_B, {}).get("lat")),
            new_graph.nodes[node_B].get("lon", new_nodes.get(node_B, {}).get("lon"))
        )
        full_route.append((node_A, node_B, distance_AB))
    # print(f'Route processed in {time.time() - start_time} seconds')
    return full_route, new_nodes, pass_times, speeds

def consolidate_nodes(route, nodes, pass_times, speeds):
    """
    Consolidate nodes by removing duplicates with the same lat/lon coordinates.
    
    Parameters:
        route: List of tuples (node_A, node_B, distance)
        nodes: Dictionary of nodes with their coordinates {node_name: {'lat': lat, 'lon': lon}}
        pass_times: List of tuples (time_A, time_B)
        speeds: List of tuples (speed_A, speed_B)
    
    Returns:
        tuple: (new_route, new_nodes, new_pass_times, new_speeds) where:
               - new_route is the updated route with consolidated node names
               - new_nodes is the dictionary with duplicate nodes removed
               - new_pass_times is the list of pass times
               - new_speeds is the list of speeds
    """
    # Find duplicates by coordinates
    coord_to_node = {}
    duplicates = {}
    
    for node_name, attrs in nodes.items():
        coord = (attrs['lat'], attrs['lon'])
        if coord in coord_to_node:
            # This is a duplicate, map it to the first node with these coordinates
            duplicates[node_name] = coord_to_node[coord]
        else:
            # First time seeing these coordinates
            coord_to_node[coord] = node_name
    
    # Create new nodes dictionary without duplicates
    new_nodes = {node: attrs for node, attrs in nodes.items() if node not in duplicates}
    
    # Update route to use the consolidated node names
    new_route = []
    for node_A, node_B, distance in route:
        # Replace node names if they are duplicates
        new_node_A = duplicates.get(node_A, node_A)
        new_node_B = duplicates.get(node_B, node_B)
        new_route.append((new_node_A, new_node_B, distance))
    
    # Simply pass through the pass_times and speeds without modification
    new_pass_times = pass_times
    new_speeds = speeds
    
    return new_route, new_nodes, new_pass_times, new_speeds
    

def generate_and_add_node(new_nodes, point, graph, spatial_index, cell_size):
    """
    Generate a new node with a random name, add it to the new_nodes dict and the graph,
    and update the spatial index accordingly.
    
    Parameters:
        new_nodes: Dictionary to store the new node information
        point: Tuple (lat, lon) specifying the location for the new node
        graph: The NetworkX graph where the node should be added
        spatial_index: The spatial index dictionary to update
        cell_size: The size of each cell in degrees
    
    Returns:
        str: Name of the newly created node
    """
    new_name = generate_random_node_name()
    lat, lon = point
    new_nodes[new_name] = {"lat": lat, "lon": lon}
    # Add new node to the graph so it can be found in subsequent queries
    # graph.add_node(new_name, lat=lat, lon=lon)
    # cell = get_cell(lat, lon, cell_size)
    # if cell not in spatial_index:
    #     spatial_index[cell] = []
    # spatial_index[cell].append((new_name, lat, lon))
    
    return new_name

def initial_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate the initial bearing (forward azimuth) from point 1 to point 2.
    
    Parameters:
        lat1, lon1: Latitude and longitude of the first point in decimal degrees
        lat2, lon2: Latitude and longitude of the second point in decimal degrees
    
    Returns:
        float: Initial bearing in radians
    """
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = sin(dlon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    return atan2(x, y)

def cross_track_distance(point, point_A, point_B):
    """
    Calculate the perpendicular distance from a point to the great circle path defined by points A and B.
    
    Parameters:
        point: Tuple (lat, lon) of the point to measure from
        point_A: Tuple (lat, lon) of the first point defining the great circle path
        point_B: Tuple (lat, lon) of the second point defining the great circle path
    
    Returns:
        float: Cross-track distance in nautical miles
    """
    # Compute the angular distance from point_A to the point
    d13 = haversine_distance(point_A[0], point_A[1], point[0], point[1]) / EARTH_RADIUS_NM
    theta13 = initial_bearing(point_A[0], point_A[1], point[0], point[1])
    theta12 = initial_bearing(point_A[0], point_A[1], point_B[0], point_B[1])
    # Cross-track distance formula
    d_xt = abs(asin(sin(d13) * sin(theta13 - theta12))) * EARTH_RADIUS_NM
    return d_xt

def find_best_waypoint_for_data_capture(graph, point_A, point_B, speed_A, speed_B, passtime_A, passtime_B, prefer_endpoint=None):
    """
    Find the optimal waypoint in the graph near a flight segment for data capture purposes.
    
    Parameters:
        graph: NetworkX graph containing nodes with lat/lon attributes
        point_A: Tuple (lat, lon) for the first endpoint of the flight segment
        point_B: Tuple (lat, lon) for the second endpoint of the flight segment
        speed_A: Speed of aircraft at point_A (in nautical miles per second)
        speed_B: Speed of aircraft at point_B (in nautical miles per second)
        passtime_A: Time when aircraft is at point_A
        passtime_B: Time when aircraft is at point_B
        prefer_endpoint: Optional; 'A' or 'B' to prefer nodes in the direction away from the 
                         opposite endpoint, or None to not consider direction (default: None)
    
    Returns:
        tuple: (best_node, best_score, best_pass_time) where best_node is the name of the optimal waypoint,
               best_score is its combined distance score, and best_pass_time is the estimated time
               the aircraft would pass closest to the waypoint
    """
    best_node = None
    best_score = inf
    best_pass_time = None
    best_extension_time = None
    best_along_track_distance = 0
    
    # Calculate total distance between A and B
    total_distance = haversine_distance(point_A[0], point_A[1], point_B[0], point_B[1])
    
    # Calculate the travel speed: if we are at the beginning of the route, use the speed at A,
    # if we are at the end of the route, use the speed at B, otherwise use the average of the two.
    if prefer_endpoint == 'A':
        avg_speed = speed_A
    elif prefer_endpoint == 'B':
        avg_speed = speed_B
    else:
        avg_speed = (speed_A + speed_B) / 2
        
    # Total time to travel the distance between A and B using the average speed
    if avg_speed > 0:
        total_time = total_distance / avg_speed # in seconds
    else:
        total_time = 0
    
    for node, data in graph.nodes(data=True):
        if "lat" not in data or "lon" not in data:
            continue
        node_point = (data["lat"], data["lon"])
        # Compute the distance from node to each endpoint
        dist_to_A = haversine_distance(node_point[0], node_point[1], point_A[0], point_A[1])
        dist_to_B = haversine_distance(node_point[0], node_point[1], point_B[0], point_B[1])

        if prefer_endpoint == 'A':
            # Ensure the candidate is in the direction away from B relative to A
            # Compute the dot product of vector (node - A) with vector (A - B)
            dot = (node_point[0] - point_A[0]) * (point_A[0] - point_B[0]) + (node_point[1] - point_A[1]) * (point_A[1] - point_B[1])
            if dot <= 0:
                score = inf
            else:
                score = dist_to_A
        elif prefer_endpoint == 'B':
            # Ensure the candidate is in the direction away from A relative to B
            dot = (node_point[0] - point_B[0]) * (point_B[0] - point_A[0]) + (node_point[1] - point_B[1]) * (point_B[1] - point_A[1])
            if dot <= 0:
                score = inf
            else:
                score = dist_to_B
        else:
            # Compute the cross-track distance from node to the extended flight segment
            d_xt = cross_track_distance(node_point, point_A, point_B)
            # Use combined score: cross-track distance plus the smaller distance to an endpoint
            score = d_xt + min(dist_to_A, dist_to_B)

        if score < best_score:
            best_score = score
            best_node = node
            
            # Calculate bearing from A to B and from A to the node
            bearing_AB = initial_bearing(point_A[0], point_A[1], point_B[0], point_B[1])
            bearing_AN = initial_bearing(point_A[0], point_A[1], node_point[0], node_point[1])
            
            # Convert distances to angular measure (in radians)
            dist_A_to_node_rad = dist_to_A / EARTH_RADIUS_NM
            
            # Calculate the along-track distance using the spherical law of cosines
            along_track_dist = asin(sin(dist_A_to_node_rad) * sin(bearing_AN - bearing_AB)) * EARTH_RADIUS_NM
            best_along_track_distance = along_track_dist
            
            # Use along-track distance to calculate the pass time
            if avg_speed >= 0:
                # Calculate the proportional position along the path (0 at A, 1 at B)
                path_position = along_track_dist / total_distance
                # Interpolate the time
                extension_time = path_position * total_time
                if prefer_endpoint == 'A':
                    pass_time_seconds = passtime_A + extension_time
                elif prefer_endpoint == 'B':
                    pass_time_seconds = passtime_B + extension_time
                else:
                    raise ValueError("Invalid prefer_endpoint value. Must be 'A', 'B', but got: " + str(prefer_endpoint))
                best_pass_time = pass_time_seconds
                best_extension_time = extension_time
                
    return best_node, best_score, best_pass_time, best_extension_time, avg_speed, abs(best_along_track_distance)

def extract_real_waypoints(draft_route, distance_threshold=20, skip_synthetic_waypoints=True):
    """
    Extract waypoints from a route, optionally filtering out synthetic (auto-generated) waypoints.
    
    Parameters:
        draft_route: List of tuples (from_node, to_node, distance)
        distance_threshold: Include synthetic waypoints when segment distance exceeds this value (in NM)
        skip_synthetic_waypoints: If True, exclude all synthetic waypoints (those starting with '_')
                                 unless distance_threshold is exceeded
    
    Returns:
        list: Ordered list of unique waypoint names that meet the filtering criteria
    """
    waypoints = []
    
    # Iterate through each segment in the route
    for from_node, to_node, distance_AB in draft_route:
        # Add from_node if it's a real waypoint or if distance exceeds threshold
        if not from_node.startswith('_') or (not skip_synthetic_waypoints and distance_AB > distance_threshold):
            waypoints.append(from_node)
            
        # Add to_node if it's a real waypoint or if distance exceeds threshold
        if not to_node.startswith('_') or (not skip_synthetic_waypoints and distance_AB > distance_threshold):
            waypoints.append(to_node)
    
    # Remove duplicates while preserving order
    unique_waypoints = []
    for waypoint in waypoints:
        if waypoint not in unique_waypoints:
            unique_waypoints.append(waypoint)
    
    return unique_waypoints

def extract_waypoints_from_augmented_route(augmented_route_object, skip_synthetic_waypoints=False):
    """
    Extract waypoints, pass times, and speeds from an augmented route object.
    
    Parameters:
        augmented_route_object: List of tuples in the format
            (from_node, to_node, distance, passover_time_from, passover_time_to, 
             speed_from, speed_to)
        skip_synthetic_waypoints: If True, exclude synthetic waypoints (those starting with '_')
    
    Returns:
        tuple: (route_str, pass_times_str, speeds_str) where:
               - route_str: String of waypoints separated by spaces
               - pass_times_str: String of passover times corresponding to each waypoint
               - speeds_str: String of speeds at each waypoint
    """
    waypoints = []
    pass_times = []
    speeds = []
    
    # Process each segment to extract unique waypoints
    for i, segment in enumerate(augmented_route_object):
        from_node, to_node, _, pass_time_from, pass_time_to, speed_from, speed_to = segment
        
        # Add from_node if it's not already the last waypoint added and not a synthetic waypoint to be skipped
        if (not waypoints or from_node != waypoints[-1]) and (not skip_synthetic_waypoints or not from_node.startswith('_')):
            waypoints.append(from_node)
            pass_times.append(str(pass_time_from))
            speeds.append(str(speed_from))
        
        # Add to_node if not a synthetic waypoint to be skipped
        # Only process the last segment's to_node
        if i == len(augmented_route_object) - 1 and (not skip_synthetic_waypoints or not to_node.startswith('_')):
            waypoints.append(to_node)
            pass_times.append(str(pass_time_to))
            speeds.append(str(speed_to))
    
    # Convert lists to space-separated strings
    route_str = ' '.join(waypoints)
    pass_times_str = ' '.join(pass_times)
    speeds_str = ' '.join(speeds)
    
    return route_str, pass_times_str, speeds_str

def find_route(graph, segments_df, error_threshold=7.5, distance_threshold_for_segment_skipping=20, max_wp_search_radius=7, min_wp_search_radius=2,
               spatial_index=None, cell_size=0.5):
    """
    Find a complete route through the flight segments, including suitable start and end waypoints for data capture.
    
    Parameters:
        graph: NetworkX graph containing existing waypoint nodes
        segments_df: DataFrame containing flight segments with columns:
                     from_lat, from_lon, to_lat, to_lon
        error_threshold: Maximum acceptable error when matching segments to existing waypoints (in NM)
        distance_threshold_for_segment_skipping: Distance threshold for including synthetic waypoints (in NM)
        max_wp_search_radius: Maximum search radius for finding candidate waypoints (in NM)
        min_wp_search_radius: Minimum search radius for finding candidate waypoints (in NM)
    
    Returns:
        tuple: (real_waypoints, real_full_waypoints, final_route, new_nodes) where:
               - real_waypoints: List of non-synthetic waypoints in route order
               - real_full_waypoints: List including both real and necessary synthetic waypoints
               - final_route: Complete route as list of segment tuples (from_node, to_node, distance)
               - new_nodes: Dictionary of all newly created nodes
    """
    # import time
    # start_time = time.time()
    final_route, new_nodes, pass_times, speeds = process_flight_segments(graph, segments_df, error_threshold=error_threshold, max_radius=max_wp_search_radius,
                                                     min_radius=min_wp_search_radius, spatial_index=spatial_index, cell_size=cell_size)
    final_route, new_nodes, pass_times, speeds = consolidate_nodes(final_route, new_nodes, pass_times, speeds)
    
    augmented_route_object = [] # (node_A, node_B, distance, pass_time_A, pass_time_B, speed_A, speed_B)
    for i in range(len(final_route)):
        node_A, node_B, distance = final_route[i]
        pass_time_A, pass_time_B = pass_times[i]
        speed_A, speed_B = speeds[i]
        augmented_route_object.append((node_A, node_B, distance, pass_time_A, pass_time_B, speed_A, speed_B))
    
    # print(f'Route processed in {time.time() - start_time} seconds')
    
    # Find the best waypoint for data capture at the start of the route
    best_starting_point, best_starting_score, best_starting_pass_time, best_starting_extension_time, best_starting_avg_speed, best_starting_along_track_distance = find_best_waypoint_for_data_capture(graph, (segments_df.iloc[0]['from_lat'], segments_df.iloc[0]['from_lon']), (segments_df.iloc[0]['to_lat'], segments_df.iloc[0]['to_lon']), passtime_A=segments_df.iloc[0]['from_time'], passtime_B=segments_df.iloc[0]['to_time'],
                                                                                                            speed_A=segments_df.iloc[0]['from_speed'], speed_B=segments_df.iloc[0]['to_speed'], prefer_endpoint='A')
    # Add the best starting point to the route
    first_node = augmented_route_object[0][0]
    first_node_pass_time = augmented_route_object[0][3]
    first_node_speed = augmented_route_object[0][5]
    augmented_route_object.insert(0, (
        best_starting_point,
        first_node,
        best_starting_along_track_distance,  # distance
        best_starting_pass_time,
        first_node_pass_time,
        best_starting_avg_speed,
        first_node_speed
    ))
    
    
    # Find the best waypoint for data capture at the end of the route
    best_ending_point, best_ending_score, best_ending_pass_time, best_ending_extension_time, best_ending_avg_speed, best_ending_along_track_distance = find_best_waypoint_for_data_capture(graph, (segments_df.iloc[-1]['from_lat'], segments_df.iloc[-1]['from_lon']), (segments_df.iloc[-1]['to_lat'], segments_df.iloc[-1]['to_lon']), passtime_A=segments_df.iloc[-1]['from_time'], passtime_B=segments_df.iloc[-1]['to_time'],
                                                                                                            speed_A=segments_df.iloc[-1]['from_speed'], speed_B=segments_df.iloc[-1]['to_speed'], prefer_endpoint='B')
    # Add the best endpoint for data capture to the route
    # Get the last node from the current route
    last_node = augmented_route_object[-1][1]
    last_node_pass_time = augmented_route_object[-1][4]
    last_node_speed = augmented_route_object[-1][6]
    
    # Add the best ending point to the route
    augmented_route_object.append((
        last_node,
        best_ending_point,
        best_ending_along_track_distance,  # distance
        last_node_pass_time,
        best_ending_pass_time,
        last_node_speed,
        best_ending_avg_speed
    ))
    
    real_waypoints = extract_waypoints_from_augmented_route(augmented_route_object, skip_synthetic_waypoints=True)
    # print(f'Real waypoints extracted in {time.time() - start_time} seconds')
    # start_time = time.time()
    real_full_waypoints = extract_waypoints_from_augmented_route(augmented_route_object, skip_synthetic_waypoints=False)
    # print(f'Real full waypoints extracted in {time.time() - start_time} seconds')
    # return real_waypoints, final_route, new_nodes
    return real_waypoints, real_full_waypoints, new_nodes
