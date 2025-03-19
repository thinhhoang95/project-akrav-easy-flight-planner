import math
import numpy as np
import networkx as nx

# ------------------------
# Helper functions
# ------------------------

def haversine(lat1, lon1, lat2, lon2):
    """
    Compute the great-circle distance between two points (in meters)
    using the haversine formula.
    """
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    a = math.sin(dphi/2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c

def latlon_to_ecef(lat, lon, R=6371000):
    """
    Convert latitude and longitude in degrees to ECEF coordinates.
    """
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    x = R * math.cos(lat_rad) * math.cos(lon_rad)
    y = R * math.cos(lat_rad) * math.sin(lon_rad)
    z = R * math.sin(lat_rad)
    return np.array([x, y, z])

def point_to_segment_distance(p, a, b):
    """
    Compute the Euclidean distance from point p to the segment ab.
    p, a, b are numpy arrays (ECEF coordinates).
    """
    ab = b - a
    ab_norm_sq = np.dot(ab, ab)
    if ab_norm_sq == 0:
        return np.linalg.norm(p - a)
    t = np.dot(p - a, ab) / ab_norm_sq
    t = max(0, min(1, t))
    projection = a + t * ab
    return np.linalg.norm(p - projection)

def compute_bearing(lat1, lon1, lat2, lon2):
    """
    Compute the bearing (in degrees) from point 1 to point 2.
    """
    lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2_rad)
    y = math.cos(lat1_rad)*math.sin(lat2_rad) - math.sin(lat1_rad)*math.cos(lat2_rad)*math.cos(dlon)
    bearing_rad = math.atan2(x, y)
    bearing_deg = (math.degrees(bearing_rad) + 360) % 360
    return bearing_deg

# ------------------------
# Fitness (distance) function
# ------------------------

def segment_edge_distance(flight_seg, edge_seg, DISTANCE_THRESHOLD=1000, MAX_ANGLE_DIFF=70):
    """
    Compute a normalized cost (0 to 1) between a flight segment and a graph edge.
    flight_seg and edge_seg are each a tuple (lat1, lon1, lat2, lon2).
    
    Cost components:
      - Angle cost: normalized difference between the segment bearings.
      - Distance cost: normalized maximum perpendicular distance (using ECEF).
      - Overlap cost: 1 minus the ratio of overlapping length along the edge.
    
    A perfect match yields cost 0; a very poor match yields cost 1.
    """
    # Unpack coordinates
    f_lat1, f_lon1, f_lat2, f_lon2 = flight_seg
    e_lat1, e_lon1, e_lat2, e_lon2 = edge_seg
    
    # --- Angle cost ---
    bearing_flight = compute_bearing(f_lat1, f_lon1, f_lat2, f_lon2)
    bearing_edge = compute_bearing(e_lat1, e_lon1, e_lat2, e_lon2)
    angle_diff = min(abs(bearing_flight - bearing_edge), 360 - abs(bearing_flight - bearing_edge))
    angle_cost = min(angle_diff / MAX_ANGLE_DIFF, 1)  # normalized: 0 for parallel, 1 if diff >= MAX_ANGLE_DIFF
    
    # --- Distance cost ---
    p_f1 = latlon_to_ecef(f_lat1, f_lon1)
    p_f2 = latlon_to_ecef(f_lat2, f_lon2)
    p_e1 = latlon_to_ecef(e_lat1, e_lon1)
    p_e2 = latlon_to_ecef(e_lat2, e_lon2)
    
    d1 = point_to_segment_distance(p_f1, p_e1, p_e2)
    d2 = point_to_segment_distance(p_f2, p_e1, p_e2)
    max_distance = max(d1, d2)
    distance_cost = min(max_distance / DISTANCE_THRESHOLD, 1)
    
    # --- Overlap cost ---
    # Compute flight and edge lengths (using ECEF distances)
    flight_length = np.linalg.norm(p_f2 - p_f1)
    edge_length = np.linalg.norm(p_e2 - p_e1)
    
    # Project flight endpoints onto the edge (using vector projection along the edge)
    AB = p_e2 - p_e1
    AB_norm_sq = np.dot(AB, AB)
    if AB_norm_sq == 0:
        t1 = t2 = 0
    else:
        def proj_t(P):
            t = np.dot(P - p_e1, AB) / AB_norm_sq
            return max(0, min(1, t))
        t1 = proj_t(p_f1)
        t2 = proj_t(p_f2)
    # The positions along the edge (in meters)
    proj1 = t1 * edge_length
    proj2 = t2 * edge_length
    overlap_length = abs(proj2 - proj1)
    
    # Overlap ratio relative to flight segment length (if flight_length is very small, set ratio = 0)
    overlap_ratio = overlap_length / flight_length if flight_length > 0 else 0
    overlap_cost = max(0, 1 - overlap_ratio)
    
    # --- Combine costs ---
    # We use a weighted sum of the three components. (Weights can be tuned.)
    w_angle = 0.3
    w_distance = 0.4
    w_overlap = 0.3
    total_cost = w_angle * angle_cost + w_distance * distance_cost + w_overlap * overlap_cost
    total_cost = min(total_cost, 1)
    
    return total_cost

# ------------------------
# Candidate generation and direction selection
# ------------------------

def get_candidate_direction(edge_seg, flight_seg):
    """
    Given an edge (with endpoints as (lat, lon)) and a flight segment,
    choose the ordering that makes the edge’s direction best match the flight segment.
    Returns the bearing (in degrees) of the candidate edge.
    """
    f_lat1, f_lon1, _, _ = flight_seg
    e_lat1, e_lon1, e_lat2, e_lon2 = edge_seg
    # Choose the endpoint (u or v) that is closer to the flight segment start:
    if haversine(f_lat1, f_lon1, e_lat1, e_lon1) < haversine(f_lat1, f_lon1, e_lat2, e_lon2):
        return compute_bearing(e_lat1, e_lon1, e_lat2, e_lon2)
    else:
        return compute_bearing(e_lat2, e_lon2, e_lat1, e_lon1)

def generate_candidate_edges(flight_seg, subgraph, DISTANCE_THRESHOLD=1000, MAX_ANGLE_DIFF=70, cost_threshold=1.0):
    """
    For a given flight segment (tuple: from_lat, from_lon, to_lat, to_lon), search through the 
    relevant subgraph (a networkx graph with 'lat' and 'lon' on each node) and generate candidate edges.
    For each edge we compute the segment_edge_distance and keep only candidates with cost < cost_threshold.
    
    Returns a list of candidate dictionaries with keys:
      - 'edge': a tuple (u, v) for the edge (as stored in the graph)
      - 'edge_seg': a tuple (lat_u, lon_u, lat_v, lon_v)
      - 'cost': the cost from segment_edge_distance
      - 'direction': the bearing for the edge in the chosen order
    """
    candidates = []
    f_lat1, f_lon1, f_lat2, f_lon2 = flight_seg
    flight_seg_tuple = (f_lat1, f_lon1, f_lat2, f_lon2)
    
    for u, v, data in subgraph.edges(data=True):
        # Get node coordinates from the subgraph (assumes they are stored under 'lat' and 'lon')
        try:
            u_lat, u_lon = subgraph.nodes[u]['lat'], subgraph.nodes[u]['lon']
            v_lat, v_lon = subgraph.nodes[v]['lat'], subgraph.nodes[v]['lon']
        except KeyError:
            continue  # skip if node attributes are missing
        
        edge_seg = (u_lat, u_lon, v_lat, v_lon)
        cost = segment_edge_distance(flight_seg_tuple, edge_seg, DISTANCE_THRESHOLD=DISTANCE_THRESHOLD, MAX_ANGLE_DIFF=MAX_ANGLE_DIFF)
        if cost < cost_threshold:
            candidate = {
                'edge': (u, v),
                'edge_seg': edge_seg,
                'cost': cost,
                'direction': get_candidate_direction(edge_seg, flight_seg_tuple)
            }
            candidates.append(candidate)
    return candidates

def get_relevant_subgraph(graph, flight_segments, buffer_degrees=0.1):
    """
    Extract the relevant part of the graph.
    flight_segments is a list of flight segment tuples (from_lat, from_lon, to_lat, to_lon).
    We compute the bounding box (with a buffer in degrees) covering all flight segments,
    and then return the subgraph induced by nodes within that box.
    """
    lats = []
    lons = []
    for seg in flight_segments:
        f_lat1, f_lon1, f_lat2, f_lon2 = seg
        lats.extend([f_lat1, f_lat2])
        lons.extend([f_lon1, f_lon2])
    
    min_lat, max_lat = min(lats) - buffer_degrees, max(lats) + buffer_degrees
    min_lon, max_lon = min(lons) - buffer_degrees, max(lons) + buffer_degrees
    
    # Filter nodes within the bounding box:
    nodes_in_box = [n for n, d in graph.nodes(data=True)
                    if (min_lat <= d.get('lat', 0) <= max_lat) and (min_lon <= d.get('lon', 0) <= max_lon)]
    subgraph = graph.subgraph(nodes_in_box).copy()
    return subgraph

# ------------------------
# Route search using dynamic programming (Viterbi-like)
# ------------------------

def find_best_route(flight_segments, graph,
                    DISTANCE_THRESHOLD=1000, MAX_ANGLE_DIFF=70,
                    turning_threshold=70, connection_bonus=-0.2,
                    segmentation_penalty=0.1, turning_cost_weight=0.05,
                    candidate_cost_threshold=1.0):
    """
    Given a list of flight_segments (each a tuple: from_lat, from_lon, to_lat, to_lon)
    and a networkx graph (with nodes having 'lat' and 'lon'), find the sequence of graph edges
    (one candidate per flight segment) that best “explains” the route.
    
    The cost for mapping a flight segment to an edge is computed via segment_edge_distance.
    In addition, we add:
      - A bonus if the candidate edges for consecutive segments are connected (share a node).
      - A segmentation penalty if consecutive flight segments are mapped to different edges.
      - A turning cost based on the angle between the chosen candidate directions.
      
    Transitions requiring a turn of more than turning_threshold degrees are disallowed.
    
    Returns the best route as a list of candidate dictionaries (one per flight segment).
    """
    n = len(flight_segments)
    if n == 0:
        return []
    
    # Extract a relevant subgraph from the (potentially huge) graph.
    subgraph = get_relevant_subgraph(graph, flight_segments)
    
    # For each flight segment, generate candidate edges.
    candidates_all = []
    for seg in flight_segments:
        cands = generate_candidate_edges(seg, subgraph,
                                         DISTANCE_THRESHOLD=DISTANCE_THRESHOLD,
                                         MAX_ANGLE_DIFF=MAX_ANGLE_DIFF,
                                         cost_threshold=candidate_cost_threshold)
        if not cands:
            raise ValueError(f"No candidate edges found for flight segment: {seg}")
        candidates_all.append(cands)
    
    # Dynamic programming tables:
    # dp[i][j] = best cumulative cost up to flight segment i, if candidate j is chosen for segment i.
    dp = [dict() for _ in range(n)]
    backpointer = [dict() for _ in range(n)]
    
    # Initialize with the cost for the first flight segment candidates.
    for j, cand in enumerate(candidates_all[0]):
        dp[0][j] = cand['cost']
        backpointer[0][j] = None  # no predecessor
    
    # For each subsequent flight segment:
    for i in range(1, n):
        for j, cand_j in enumerate(candidates_all[i]):
            best_cost = float('inf')
            best_prev = None
            for k, cand_k in enumerate(candidates_all[i-1]):
                # Get the candidate directions (precomputed for each candidate)
                dir_k = cand_k['direction']
                dir_j = cand_j['direction']
                # Compute turning angle difference
                turning_angle = min(abs(dir_j - dir_k), 360 - abs(dir_j - dir_k))
                if turning_angle > turning_threshold:
                    continue  # disallow this transition
                
                # Turning cost: a small cost proportional to turning angle.
                turning_cost = (turning_angle / turning_threshold) * turning_cost_weight
                
                # Connection bonus: if the two candidate edges share a common node.
                # (Edges are stored as (u, v). We check if any of the endpoints are the same.)
                if set(cand_k['edge']).intersection(set(cand_j['edge'])):
                    connection_cost = connection_bonus
                else:
                    connection_cost = 0
                
                # Segmentation penalty: if the edge changed (i.e. not the same edge)
                if cand_k['edge'] != cand_j['edge'] and cand_k['edge'] != cand_j['edge'][::-1]:
                    seg_penalty = segmentation_penalty
                else:
                    seg_penalty = 0
                
                total_cost = dp[i-1][k] + cand_j['cost'] + turning_cost + seg_penalty + connection_cost
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_prev = k
            if best_prev is None:
                # If no transition is allowed, assign a high cost.
                dp[i][j] = float('inf')
            else:
                dp[i][j] = best_cost
                backpointer[i][j] = best_prev
    
    # Backtrack to recover best route:
    # Find the candidate at the last segment with minimum cost.
    last_candidates = dp[-1]
    best_last_idx = min(last_candidates, key=lambda j: last_candidates[j])
    best_route = [None] * n
    best_route[-1] = candidates_all[-1][best_last_idx]
    
    # Backtrack
    prev_idx = best_last_idx
    for i in range(n-1, 0, -1):
        prev_idx = backpointer[i][prev_idx]
        best_route[i-1] = candidates_all[i-1][prev_idx]
    
    return best_route

# ------------------------
# Example usage (with made-up data)
# ------------------------

if __name__ == "__main__":
    # Example flight segments (from_lat, from_lon, to_lat, to_lon)
    flight_segments = [
        (42.94, 14.27, 46.18, 14.54),
        (46.18, 14.54, 35.85, 14.49)
    ]
    
    # Create a dummy graph with nodes having lat/lon.
    # (In practice, the graph would be loaded from real data.)
    G = nx.Graph()
    # Add nodes with latitude and longitude attributes:
    G.add_node("MEGAN", lat=42.9, lon=14.3)
    G.add_node("TIRSA", lat=46.2, lon=14.5)
    G.add_node("MIDPOINT", lat=44.0, lon=14.4)
    G.add_node("ALT", lat=36.0, lon=14.5)
    # Add edges (the route in this toy example)
    G.add_edge("MEGAN", "MIDPOINT")
    G.add_edge("MIDPOINT", "TIRSA")
    G.add_edge("TIRSA", "ALT")
    
    try:
        best_route = find_best_route(flight_segments, G)
        print("Best route candidate edges for each flight segment:")
        for i, cand in enumerate(best_route):
            print(f"Segment {i+1}: Edge {cand['edge']}, cost: {cand['cost']:.3f}, direction: {cand['direction']:.1f}°")
    except ValueError as e:
        print(e)
