import math

# Constants
EARTH_RADIUS = 6371000  # meters

# -----------------------------
# Helper Functions
# -----------------------------

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Compute the great-circle distance between two points on Earth (in meters).
    """
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad)*math.cos(lat2_rad)*math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return EARTH_RADIUS * c

def compute_bearing(lat1, lon1, lat2, lon2):
    """
    Compute the initial bearing (in degrees) from point A to point B.
    """
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlon_rad = math.radians(lon2 - lon1)
    
    x = math.sin(dlon_rad) * math.cos(lat2_rad)
    y = math.cos(lat1_rad)*math.sin(lat2_rad) - math.sin(lat1_rad)*math.cos(lat2_rad)*math.cos(dlon_rad)
    
    initial_bearing = math.atan2(x, y)
    bearing_deg = (math.degrees(initial_bearing) + 360) % 360
    return bearing_deg

def convert_to_xy(lat, lon, ref_lat, ref_lon):
    """
    Convert latitude and longitude to local Cartesian (x, y) coordinates in meters,
    using an equirectangular approximation around a reference point.
    """
    # Convert degrees to radians
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    ref_lat_rad = math.radians(ref_lat)
    ref_lon_rad = math.radians(ref_lon)
    
    x = (lon_rad - ref_lon_rad) * math.cos(ref_lat_rad) * EARTH_RADIUS
    y = (lat_rad - ref_lat_rad) * EARTH_RADIUS
    return (x, y)

import math
import networkx as nx

# Assume compute_bearing, haversine_distance, and convert_to_xy are defined elsewhere

# -----------------------------
# Projection Fitness Function
# -----------------------------
def projection_fitness(flight_segment, edge, graph, is_endpoint=False):
    """
    Compute a fitness score for how well a flight segment (straight line between 
    recorded coordinates) matches a given edge in the networkx graph.
    
    The score is a weighted sum of:
      - Orientation (bearing) similarity.
      - Overlap (projected along the edge) between the flight segment and the edge.
    
    For the first and last segments (is_endpoint=True) the overlap is downweighted.
    """
    # Retrieve flight segment endpoints.
    f_from_lat = flight_segment['from_lat']
    f_from_lon = flight_segment['from_lon']
    f_to_lat = flight_segment['to_lat']
    f_to_lon = flight_segment['to_lon']
    
    # Compute flight segment bearing and length.
    flight_bearing = compute_bearing(f_from_lat, f_from_lon, f_to_lat, f_to_lon)
    flight_length = haversine_distance(f_from_lat, f_from_lon, f_to_lat, f_to_lon)
    
    # In a networkx graph, each edge is a tuple (u, v, data).
    u, v, _ = edge
    # Retrieve edge endpoints using node attributes.
    e_from_lat = graph.nodes[u]['lat']
    e_from_lon = graph.nodes[u]['lon']
    e_to_lat = graph.nodes[v]['lat']
    e_to_lon = graph.nodes[v]['lon']
    
    # Compute edge bearing and length.
    edge_bearing = compute_bearing(e_from_lat, e_from_lon, e_to_lat, e_to_lon)
    edge_length = haversine_distance(e_from_lat, e_from_lon, e_to_lat, e_to_lon)
    
    # 1. Orientation Score:
    bearing_diff = abs(flight_bearing - edge_bearing)
    if bearing_diff > 180:
        bearing_diff = 360 - bearing_diff
    orientation_score = max(0, 1 - (bearing_diff / 90))
    
    # 2. Overlap Score:
    # Convert endpoints to local x,y coordinates (using the edge's start as reference).
    E1 = convert_to_xy(e_from_lat, e_from_lon, e_from_lat, e_from_lon)
    E2 = convert_to_xy(e_to_lat, e_to_lon, e_from_lat, e_from_lon)
    F1 = convert_to_xy(f_from_lat, f_from_lon, e_from_lat, e_from_lon)
    F2 = convert_to_xy(f_to_lat, f_to_lon, e_from_lat, e_from_lon)
    
    edge_vec = (E2[0] - E1[0], E2[1] - E1[1])
    L = math.hypot(edge_vec[0], edge_vec[1])
    if L == 0:
        return 0.0  # Degenerate edge.
    u_vec = (edge_vec[0] / L, edge_vec[1] / L)
    
    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1]
    
    t1 = dot((F1[0] - E1[0], F1[1] - E1[1]), u_vec)
    t2 = dot((F2[0] - E1[0], F2[1] - E1[1]), u_vec)
    
    seg_start = min(t1, t2)
    seg_end = max(t1, t2)
    
    # Calculate overlap along the edge.
    overlap_start = max(0, seg_start)
    overlap_end = min(L, seg_end)
    overlap_length = max(0, overlap_end - overlap_start)
    
    normalized_overlap = overlap_length / (flight_length + 1e-6)
    normalized_overlap = min(normalized_overlap, 1.0)
    
    # Weight the scores based on whether it's an endpoint segment.
    if is_endpoint:
        weight_orientation = 0.8
        weight_overlap = 0.2
    else:
        weight_orientation = 0.5
        weight_overlap = 0.5
        
    fitness = weight_orientation * orientation_score + weight_overlap * normalized_overlap
    return fitness

# -----------------------------
# Subgraph Extraction Function
# -----------------------------
def get_relevant_subgraph(graph, segments, margin=0.5):
    """
    Extract the subgraph covering the area spanned by the flight segments,
    with an added margin (in degrees).
    """
    # Compute bounding box from all segment endpoints.
    all_lats = []
    all_lons = []
    for seg in segments:
        all_lats.extend([seg['from_lat'], seg['to_lat']])
        all_lons.extend([seg['from_lon'], seg['to_lon']])
    min_lat, max_lat = min(all_lats), max(all_lats)
    min_lon, max_lon = min(all_lons), max(all_lons)
    
    # Apply margin.
    min_lat -= margin
    max_lat += margin
    min_lon -= margin
    max_lon += margin
    
    # Filter nodes based on the bounding box.
    filtered_nodes = [
        n for n, data in graph.nodes(data=True)
        if min_lat <= data['lat'] <= max_lat and min_lon <= data['lon'] <= max_lon
    ]
    
    # Create a subgraph with only the filtered nodes.
    return graph.subgraph(filtered_nodes).copy()

# -----------------------------
# DFS Search for Route Inference
# -----------------------------
def dfs_search_routes(segment_index, current_route, current_fitness, segments, graph, candidate_routes, max_routes, verbose=True):
    """
    Recursively search for routes matching the flight segments.
    
    - segment_index: index of the current flight segment.
    - current_route: list of edges (tuples) chosen so far.
    - current_fitness: cumulative fitness score.
    - segments: list of flight segments.
    - graph: a networkx (sub)graph.
    - candidate_routes: list to collect complete routes (each as (route, fitness)).
    - max_routes: maximum candidate routes to collect.
    """
    if segment_index >= len(segments):
        candidate_routes.append((list(current_route), current_fitness))
        return
    
    current_segment = segments[segment_index]
    candidate_edges = []
    
    if not current_route:
        # For the first segment, consider all edges in the graph.
        for edge in graph.edges(data=True):
            fitness = projection_fitness(current_segment, edge, graph, is_endpoint=True)
            if fitness > 0.3:
                candidate_edges.append((edge, fitness))
        candidate_edges.sort(key=lambda x: x[1], reverse=True)
        if verbose:
            # Print top 5 candidate edges
            print('--------------- FIRST SEGMENT ---------------')
            print(f'Top 5 candidate edges for segment {segment_index}:')
            for edge, fitness in candidate_edges[:5]:
                print(f'Edge: {edge}, Fitness: {fitness}')
            print('------- END OF FIRST SEGMENT -------')
    else:
        # For subsequent segments, get outgoing edges from the last node.
        last_edge = current_route[-1]
        last_to = last_edge[1]  # Second element of the tuple is the "to" node.
        for edge in graph.out_edges(last_to, data=True):
            fitness = projection_fitness(current_segment, edge, graph, is_endpoint=False)
            if fitness > 0.3:
                candidate_edges.append((edge, fitness))
        # Also allow the possibility to remain on the same edge.
        same_edge = last_edge
        fitness = projection_fitness(current_segment, same_edge, graph, is_endpoint=False)
        if fitness > 0.3:
            candidate_edges.append((same_edge, fitness))
        candidate_edges.sort(key=lambda x: x[1], reverse=True)
        
    if verbose:
        print('--------------------------------')
        print(f'Current route: {current_route}')
        print(f'Current fitness: {current_fitness}')
        print('--------------------------------')
        print(f'Candidate edges for segment {segment_index}:')
        for edge, fitness in candidate_edges:
            print(f'Edge: {edge}, Fitness: {fitness}')
        print('--------------------------------')
        print()
        print()
        print()
    
    # Expand DFS with each candidate edge.
    for edge, fitness in candidate_edges:
        new_route = current_route + [edge]
        new_fitness = current_fitness + fitness
        dfs_search_routes(segment_index + 1, new_route, new_fitness, segments, graph, candidate_routes, max_routes)
        if len(candidate_routes) >= max_routes:
            return  # Stop if maximum routes are reached

# -----------------------------
# Main Function to Infer Routes
# -----------------------------
def infer_flight_routes(segments, graph, max_routes=5):
    """
    Given a list of flight segments and a full air route network (as a networkx graph),
    infer candidate routes.
    
    Returns a list of tuples (route, total_fitness), where route is a list of edge tuples.
    """
    subgraph = get_relevant_subgraph(graph, segments, margin=0.5)
    
    candidate_routes = []
    dfs_search_routes(0, [], 0.0, segments, subgraph, candidate_routes, max_routes)
    candidate_routes.sort(key=lambda x: x[1], reverse=True)
    
    return candidate_routes[:max_routes]

# -----------------------------
# DataFrame Conversion Function
# -----------------------------
def convert_df_to_segment_format(flight_df):
    """
    Convert a DataFrame with flight data to the segment format required by the route inference algorithm.
    
    The DataFrame should contain the following columns:
        id, from_time, to_time, from_lat, from_lon, to_lat, to_lon, 
        from_alt, to_alt, from_speed, to_speed
    """
    segments = []
    for _, row in flight_df.iterrows():
        segment = {
            "id": row['id'],
            "from_time": row['from_time'],
            "to_time": row['to_time'],
            "from_lat": row['from_lat'],
            "from_lon": row['from_lon'],
            "to_lat": row['to_lat'],
            "to_lon": row['to_lon'],
            "from_alt": row['from_alt'],
            "to_alt": row['to_alt'],
            "from_speed": row['from_speed'],
            "to_speed": row['to_speed']
        }
        segments.append(segment)
    
    return segments

def convert_inferred_route_to_readable_format(inferred_route):
    """
    Convert an inferred route to a readable list format.
    
    Args:
        inferred_route: A tuple containing (route_edges, fitness_score) where
                        route_edges is a list of edge tuples (from_node, to_node, attributes)
    
    Returns:
        A string representation of the route as a space-separated sequence of waypoints
    """
    # Extract the route edges from the tuple
    route_edges, fitness_score = inferred_route
    
    # Initialize the route with the first 'from' waypoint
    if not route_edges:
        return ""
    
    waypoints = [route_edges[0][0]]  # Start with the first 'from' node
    
    # Add all 'to' waypoints
    for edge in route_edges:
        waypoints.append(edge[1])
    
    # Remove duplicates while preserving order
    unique_waypoints = []
    for wp in waypoints:
        if wp not in unique_waypoints:
            unique_waypoints.append(wp)
    
    # Join the waypoints with spaces
    return unique_waypoints

