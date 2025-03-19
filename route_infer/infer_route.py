import math
import networkx as nx

def latlon_to_xy(lat, lon, ref_lat):
    """
    Convert latitude and longitude (in degrees) to local Cartesian x,y coordinates (in km)
    using an equirectangular approximation.
    ref_lat is in degrees.
    """
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * math.cos(math.radians(ref_lat))
    x = lon * km_per_deg_lon
    y = lat * km_per_deg_lat
    return (x, y)

def point_line_distance(point, seg_start, seg_end):
    """
    Compute the Euclidean distance (in km) from a point to a line segment.
    """
    (x, y) = point
    (x1, y1) = seg_start
    (x2, y2) = seg_end
    
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(x - x1, y - y1)
    t = ((x - x1) * dx + (y - y1) * dy) / (dx*dx + dy*dy)
    if t < 0:
        return math.hypot(x - x1, y - y1)
    elif t > 1:
        return math.hypot(x - x2, y - y2)
    else:
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return math.hypot(x - proj_x, y - proj_y)

def compute_segment_edge_fitness(segment, edge, G):
    """
    Compute a fitness score between a flight segment and a directed graph edge.
    
    Parameters:
      segment: dict with keys 'from_lat','from_lon','to_lat','to_lon'
      edge: tuple (u,v) representing a directed edge in graph G.
      G: networkx directed graph with node attributes 'lat' and 'lon'
      
    Returns a score (float) where a higher value indicates a better match.
    The score is the product of:
        - overlap_score: fraction of the flight segment that overlaps the edge.
        - angle_score: cosine of the (adjusted) angle difference.
        - proximity_score: 1/(1+max_distance).
    """
    P1_lat = segment['from_lat']
    P1_lon = segment['from_lon']
    P2_lat = segment['to_lat']
    P2_lon = segment['to_lon']
    
    u, v = edge
    Q1_lat = G.nodes[u]['lat']
    Q1_lon = G.nodes[u]['lon']
    Q2_lat = G.nodes[v]['lat']
    Q2_lon = G.nodes[v]['lon']
    
    ref_lat = (P1_lat + P2_lat + Q1_lat + Q2_lat) / 4.0
    
    P1 = latlon_to_xy(P1_lat, P1_lon, ref_lat)
    P2 = latlon_to_xy(P2_lat, P2_lon, ref_lat)
    Q1 = latlon_to_xy(Q1_lat, Q1_lon, ref_lat)
    Q2 = latlon_to_xy(Q2_lat, Q2_lon, ref_lat)
    
    seg_vector = (P2[0] - P1[0], P2[1] - P1[1])
    edge_vector = (Q2[0] - Q1[0], Q2[1] - Q1[1])
    seg_length = math.hypot(seg_vector[0], seg_vector[1])
    edge_length = math.hypot(edge_vector[0], edge_vector[1])
    if seg_length == 0 or edge_length == 0:
        return 0.0

    dot = seg_vector[0]*edge_vector[0] + seg_vector[1]*edge_vector[1]
    cos_angle = max(min(dot/(seg_length*edge_length), 1.0), -1.0)
    angle = math.degrees(math.acos(cos_angle))
    if angle > 90:
        angle = 180 - angle
    if abs(angle) > 10: # limit angle to max 10 degrees
        angle_score = math.cos(math.radians(angle)) * 0.001 # for large angles, reduce the weight even more
    else:
        angle_score = math.cos(math.radians(angle)) # ideal: 1.0, range: 0.0-1.0
    
    d1 = point_line_distance(P1, Q1, Q2)
    d2 = point_line_distance(P2, Q1, Q2)
    d3 = point_line_distance(Q1, P1, P2)
    d4 = point_line_distance(Q2, P1, P2)
    max_distance = max(d1, d2, d3, d4)
    proximity_score = 12.0 / (12.0 + max_distance) # adjust the sensitivity of proximity, e.g., with 10.0, score = 0.5 when max_distance is 10km; ideal: 1.0, range: 0.0-1.0
    
    edge_norm_sq = edge_length**2
    t1 = ((P1[0]-Q1[0])*edge_vector[0] + (P1[1]-Q1[1])*edge_vector[1]) / edge_norm_sq
    t2 = ((P2[0]-Q1[0])*edge_vector[0] + (P2[1]-Q1[1])*edge_vector[1]) / edge_norm_sq
    t_min, t_max = min(t1, t2), max(t1, t2)
    overlap_t = max(0.0, min(t_max, 1.0) - max(t_min, 0.0))
    overlap_length = overlap_t * edge_length
    overlap_score = max(0.0, min(1.0, overlap_length / seg_length)) # ideal: 1.0, range: 0.0-1.0
    
    fitness = overlap_score * angle_score * proximity_score
    return fitness # best: 1.0, worst: 0.0

def extract_relevant_subgraph(G, segments, buffer=0.5):
    """
    Extract a subgraph of G that contains nodes within the bounding box of the flight segments,
    extended by a buffer (in degrees).
    """
    lats = []
    lons = []
    for seg in segments:
        lats.extend([seg['from_lat'], seg['to_lat']])
        lons.extend([seg['from_lon'], seg['to_lon']])
    min_lat, max_lat = min(lats) - buffer, max(lats) + buffer
    min_lon, max_lon = min(lons) - buffer, max(lons) + buffer
    
    selected_nodes = [n for n, data in G.nodes(data=True)
                      if (min_lat <= data.get('lat', 0) <= max_lat) and (min_lon <= data.get('lon', 0) <= max_lon)]
    return G.subgraph(selected_nodes).copy()

def get_candidate_edges(segment, subgraph, candidate_limit=10, bbox_margin=0.2):
    """
    For a given flight segment, compute a candidate list of directed graph edges (from subgraph)
    along with their fitness scores, filtered by a rough bounding box test.
    """
    seg_min_lat = min(segment['from_lat'], segment['to_lat']) - bbox_margin
    seg_max_lat = max(segment['from_lat'], segment['to_lat']) + bbox_margin
    seg_min_lon = min(segment['from_lon'], segment['to_lon']) - bbox_margin
    seg_max_lon = max(segment['from_lon'], segment['to_lon']) + bbox_margin
    
    candidates = []
    # Loop over all directed edges in the subgraph.
    for u, v in subgraph.edges():
        u_lat = subgraph.nodes[u].get('lat', None)
        u_lon = subgraph.nodes[u].get('lon', None)
        v_lat = subgraph.nodes[v].get('lat', None)
        v_lon = subgraph.nodes[v].get('lon', None)
        if u_lat is None or v_lat is None:
            continue
        edge_min_lat = min(u_lat, v_lat)
        edge_max_lat = max(u_lat, v_lat)
        edge_min_lon = min(u_lon, v_lon)
        edge_max_lon = max(u_lon, v_lon)
        if (edge_max_lat < seg_min_lat or edge_min_lat > seg_max_lat or
            edge_max_lon < seg_min_lon or edge_min_lon > seg_max_lon):
            continue
        
        fitness = compute_segment_edge_fitness(segment, (u,v), subgraph)
        if fitness > 0:
            candidates.append(((u, v), fitness))
    
    candidates.sort(key=lambda x: x[1], reverse=True) # highest fitness first
    return candidates[:candidate_limit]

import math

def infer_n_routes(flight_segments, G, connection_weight=0.0, candidate_limit=8, n_routes=3, segment_penalty=1.0, turn_angle_limit=70.0):
    """
    Infer the top n most likely sequences of directed graph edges corresponding to the flight segments.
    
    For each flight segment, candidate graph edges and their fitness scores are computed.
    Then a beam search is performed to choose sequences that minimize the sum of costs.
    In the transition between segments, a penalty is added if the destination of the previous edge
    doesn't match the source of the current edge. Additionally, a penalty is added for the number 
    of segments in a candidate edge, and connected edges that require a turn greater than turn_angle_limit 
    degrees are discarded.
    
    Parameters:
      flight_segments: list of flight segment dictionaries.
      G: networkx directed graph (nx.DiGraph) with node attributes 'lat' and 'lon'
      connection_weight: penalty weight for consecutive edges that aren't connected (prev_edge[1] != curr_edge[0]).
      candidate_limit: maximum number of candidate edges per flight segment.
      n_routes: number of best routes to return.
      segment_penalty: cost penalty per segment in the candidate edge (to discourage extra edges).
      turn_angle_limit: maximum allowed turn angle in degrees between consecutive connected edges.
      
    Returns:
      A list of tuples (cost, route) where each route is a list of directed graph edges (each as a tuple (u,v)).
    """
    import time
    time_start = time.time()
    subG = extract_relevant_subgraph(G, flight_segments, buffer=0.5)
    print(f'Extracted relevant subgraph in {time.time() - time_start:.2f} seconds')

    time_start = time.time()
    # Compute candidate edges for each flight segment.
    candidates_per_segment = []
    for seg in flight_segments:
        candidates = get_candidate_edges(seg, subG, candidate_limit=candidate_limit)
        if not candidates:
            # If no candidate found, use a dummy candidate (with default segment count 1).
            candidates = [((None, None), 0, 1)]
        new_candidates = []
        for cand in candidates:
            # If candidate tuple has two elements, assume it covers one segment.
            if len(cand) == 2:
                edge, score = cand
                seg_count = 1
            else:
                edge, score, seg_count = cand
            # Convert fitness score into a cost (lower is better) and add segment penalty.
            cost = (1.0 - score) + segment_penalty * seg_count
            new_candidates.append((edge, cost))
        candidates_per_segment.append(new_candidates)
    print(f'Computed candidate edges in {time.time() - time_start:.2f} seconds')

    time_start = time.time()
    # Initialize the beam: each candidate for the first segment starts a route.
    beam = []
    for cand_edge, cost in candidates_per_segment[0]:
        print(f'Adding edge {cand_edge} to beam with cost {cost}')
        beam.append((cost, [cand_edge]))

    # Beam search over the remaining flight segments.
    for i in range(1, len(flight_segments)):
        new_beam = []
        for (cum_cost, route) in beam:
            for cand_edge, cand_cost in candidates_per_segment[i]:
                # Initialize penalty for this transition.
                penalty = 0.0
                # If either the previous edge or the candidate is a dummy, or if they're disconnected,
                # add the connection penalty.
                if route[-1] == (None, None) or cand_edge == (None, None) or route[-1][1] != cand_edge[0]:
                    penalty += connection_weight
                else:
                    # Both edges are connected, so check the turn angle.
                    prev_u, prev_v = route[-1]
                    curr_u, curr_v = cand_edge
                    # Ensure the nodes exist in the graph.
                    if prev_u in G.nodes and prev_v in G.nodes and curr_v in G.nodes:
                        # Get coordinates (assume lon is x, lat is y).
                        x1, y1 = G.nodes[prev_u]['lon'], G.nodes[prev_u]['lat']
                        x2, y2 = G.nodes[prev_v]['lon'], G.nodes[prev_v]['lat']
                        x3, y3 = G.nodes[curr_v]['lon'], G.nodes[curr_v]['lat']
                        # Compute vectors for the previous edge and the candidate edge.
                        vecA = (x2 - x1, y2 - y1)
                        vecB = (x3 - x2, y3 - y2)
                        normA = math.sqrt(vecA[0]**2 + vecA[1]**2)
                        normB = math.sqrt(vecB[0]**2 + vecB[1]**2)
                        if normA > 0 and normB > 0:
                            dot = vecA[0]*vecB[0] + vecA[1]*vecB[1]
                            # Clamp to avoid domain errors.
                            cos_angle = max(min(dot/(normA*normB), 1.0), -1.0)
                            angle = math.degrees(math.acos(cos_angle))
                            # If the turn angle exceeds the limit, skip this candidate.
                            if angle > turn_angle_limit:
                                continue
                new_cost = cum_cost + cand_cost + penalty
                new_route = route + [cand_edge]
                print(f'Adding edge {cand_edge} to route {new_route} with cost {new_cost}')
                new_beam.append((new_cost, new_route))
        # Keep only the best n_routes (lowest cost routes).
        beam = sorted(new_beam, key=lambda x: x[0])[:n_routes]
    print(f'Beam search in {time.time() - time_start:.2f} seconds')

    # Optionally, merge consecutive duplicate edges in each route.
    final_routes = []
    for cost, route in beam:
        merged = []
        for edge in route:
            if not merged or merged[-1] != edge:
                merged.append(edge)
        final_routes.append((cost, merged))
    return final_routes


def convert_final_routes_to_waypoints(final_routes):
    waypoints = []
    for edge in final_routes:
        if not waypoints or edge[0] != waypoints[-1]:
            waypoints.append(edge[0])
        waypoints.append(edge[1])
    return waypoints

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

from utils.haversine import haversine_distance

def trim_mini_segments(flight_segments, min_length_nm=10.0):
    """
    Trims short segments from the beginning and end of the flight path.
    
    Parameters:
      flight_segments: list of flight segment dictionaries
      min_length_nm: minimum segment length in nautical miles to keep (default: 5.0 nm)
      
    Returns:
      A new list with short segments at the beginning and end removed
    """
    if not flight_segments:
        return []
    
    # Convert nautical miles to kilometers (1 nm = 1.852 km)
    min_length_km = min_length_nm * 1.852
    
    # Calculate length of each segment
    segment_lengths = []
    for seg in flight_segments:
        length = haversine_distance(
            seg['from_lat'], seg['from_lon'], 
            seg['to_lat'], seg['to_lon']
        )
        segment_lengths.append(length)
    
    # Find first segment from start that exceeds minimum length
    start_idx = 0
    while start_idx < len(flight_segments) and segment_lengths[start_idx] < min_length_km:
        start_idx += 1
    
    # Find first segment from end that exceeds minimum length
    end_idx = len(flight_segments) - 1
    while end_idx >= 0 and segment_lengths[end_idx] < min_length_km:
        end_idx -= 1
    
    # If all segments are too short, return at least one segment (the longest one)
    if start_idx > end_idx:
        if not flight_segments:
            return []
        longest_idx = segment_lengths.index(max(segment_lengths))
        return [flight_segments[longest_idx]]
    
    # Return the trimmed list of segments
    return flight_segments[start_idx:end_idx+1]

# Example usage:
if __name__ == '__main__':
    # Create a sample networkx graph with node coordinates.
    G = nx.Graph()
    # (Here we create a toy graph with nodes labeled by waypoint names; in practice, you'd have your full graph.)
    G.add_node("MEGAN", lat=42.9, lon=14.27)
    G.add_node("TIRSA", lat=46.18, lon=14.54)
    G.add_node("MID1", lat=44.0, lon=14.4)
    # Connect nodes (assume bidirectional edges)
    G.add_edge("MEGAN", "MID1")
    G.add_edge("MID1", "TIRSA")
    
    # Define two flight segments (example from your CSV)
    flight_segments = [
        {
            'id': '000042HMJ225',
            'from_time': 1680349319.0,
            'to_time': 1680364679.0,
            'from_lat': 42.94166564941406,
            'from_lon': 14.271751226380816,
            'to_lat': 46.17704772949219,
            'to_lon': 14.543553794302593,
            'from_alt': 11521.44,
            'to_alt': 1569.72,
            'from_speed': 0.232055441288291,
            'to_speed': 0.1399177216897295
        },
        {
            'id': '000042HMJ225',
            'from_time': 1680364679.0,
            'to_time': 1680382679.0,
            'from_lat': 46.17704772949219,
            'from_lon': 14.543553794302593,
            'to_lat': 35.847457627118644,
            'to_lon': 14.489973352310503,
            'from_alt': 1569.72,
            'to_alt': 114.3,
            'from_speed': 0.1399177216897295,
            'to_speed': 0.0
        }
    ]
    
    inferred_edges = infer_route(flight_segments, G, connection_weight=0.1, candidate_limit=10)
    print("Inferred route edges:")
    for edge in inferred_edges:
        print(edge)
