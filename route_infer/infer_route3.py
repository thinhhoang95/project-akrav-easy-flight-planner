import networkx as nx
import math
import numpy as np
from scipy.spatial import cKDTree

def point_to_segment_distance(px, py, ax, ay, bx, by):
    """
    Compute the Euclidean distance (in degrees) from a point (px, py) to 
    the line segment defined by (ax, ay) and (bx, by).
    
    Note: This is a flat approximation. For small tolerances or limited
    extents, it should be sufficient.
    """
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    
    # Compute projection factor
    t = ((px - ax) * dx + (py - ay) * dy) / (dx*dx + dy*dy)
    
    # Clamp t to the range [0, 1] to find the nearest point on the segment
    if t < 0:
        nearest_x, nearest_y = ax, ay
    elif t > 1:
        nearest_x, nearest_y = bx, by
    else:
        nearest_x, nearest_y = ax + t * dx, ay + t * dy
    
    return math.hypot(px - nearest_x, py - nearest_y)

def build_kd_tree(graph):
    node_keys = []
    node_coords = []
    for n, attr in graph.nodes(data=True):
        lat = attr.get('lat')
        lon = attr.get('lon')
        if lat is None or lon is None:
            continue
        node_keys.append(n)
        node_coords.append((lat, lon))
    node_coords = np.array(node_coords)
    
    # Build the KDTree for quick spatial look-up
    tree = cKDTree(node_coords)
    
    return tree, node_keys, node_coords
        

def derive_subgraph_tube_spatial(graph, segments, tolerance, kd_tree=None):
    """
    Derive a subgraph containing only nodes within a "tube" around the given flight segments,
    using a KDTree for efficient spatial indexing.
    
    Parameters:
      graph: networkx.Graph with node attributes 'lat' and 'lon'.
      segments: list of dicts, each with keys 'from_lat', 'from_lon', 'to_lat', 'to_lon'.
      tolerance: float, maximum distance (in degrees) from a segment for a node to be included.
    
    Returns:
      subgraph: networkx.Graph of nodes (and edges among them) that lie within the tube.
    """
    # Build a list of node coordinates and keep track of their corresponding keys
    if kd_tree is None:
        print('WARNING: KDTree was not provided, building it from scratch...')
        node_keys = []
        node_coords = []
        for n, attr in graph.nodes(data=True):
            lat = attr.get('lat')
            lon = attr.get('lon')
            if lat is None or lon is None:
                continue
            node_keys.append(n)
            node_coords.append((lat, lon))
        node_coords = np.array(node_coords)
        
        # Build the KDTree for quick spatial look-up
        tree = cKDTree(node_coords)
    else:
        tree = kd_tree
        node_keys = list(graph.nodes())
        node_coords = np.array([(graph.nodes[n]['lat'], graph.nodes[n]['lon']) for n in node_keys])
    
    selected_nodes = set()
    
    # Process each flight segment
    for seg in segments:
        lat1 = seg['from_lat']
        lon1 = seg['from_lon']
        lat2 = seg['to_lat']
        lon2 = seg['to_lon']
        
        # Create a rough bounding box around the segment (expanded by tolerance)
        min_lat = min(lat1, lat2) - tolerance
        max_lat = max(lat1, lat2) + tolerance
        min_lon = min(lon1, lon2) - tolerance
        max_lon = max(lon1, lon2) + tolerance
        
        # Use vectorized filtering on the node coordinates to get candidate indices
        candidate_idx = np.where(
            (node_coords[:, 0] >= min_lat) & (node_coords[:, 0] <= max_lat) &
            (node_coords[:, 1] >= min_lon) & (node_coords[:, 1] <= max_lon)
        )[0]
        
        # For each candidate, check if the distance to the segment is within tolerance
        for idx in candidate_idx:
            px, py = node_coords[idx]
            d = point_to_segment_distance(px, py, lat1, lon1, lat2, lon2)
            if d <= tolerance:
                selected_nodes.add(node_keys[idx])
    
    # Build and return the subgraph from the selected nodes
    subgraph = graph.subgraph(selected_nodes).copy()
    return subgraph

def sample_points_from_segment(segment, min_points=4, desired_spacing=0.001):
    """
    Sample points uniformly along a segment with approximately desired_spacing between them.
    
    Parameters:
      segment: dict with keys 'from_lat', 'from_lon', 'to_lat', 'to_lon'
      min_points: Minimum number of points to sample from any segment
      desired_spacing: Approximate distance between sampled points (in degrees)
    
    Returns:
      List of (lat, lon) tuples representing sampled points
    """
    lat1, lon1 = segment['from_lat'], segment['from_lon']
    lat2, lon2 = segment['to_lat'], segment['to_lon']
    
    # Calculate segment length
    segment_length = math.hypot(lat2 - lat1, lon2 - lon1)
    
    # Calculate how many points to sample
    num_points = max(min_points, int(segment_length / desired_spacing) + 1)
    
    points = []
    for i in range(num_points):
        t = i / (num_points - 1)  # parameter between 0 and 1
        lat = lat1 + t * (lat2 - lat1)
        lon = lon1 + t * (lon2 - lon1)
        points.append((lat, lon))
    return points

def compute_edge_length(graph, edge):
    """
    Compute the length of an edge in the graph.
    The edge is given as a tuple (u, v) and node attributes 'lat' and 'lon' are used.
    """
    u, v = edge
    lat1, lon1 = graph.nodes[u]['lat'], graph.nodes[u]['lon']
    lat2, lon2 = graph.nodes[v]['lat'], graph.nodes[v]['lon']
    return math.hypot(lat2 - lat1, lon2 - lon1)

def get_emission_cost(point, edge, graph, sigma):
    """
    Compute the emission cost for a point given a candidate edge.
    Uses the squared distance from the point to the edge normalized by sigma.
    """
    u, v = edge
    lat1, lon1 = graph.nodes[u]['lat'], graph.nodes[u]['lon']
    lat2, lon2 = graph.nodes[v]['lat'], graph.nodes[v]['lon']
    d = point_to_segment_distance(point[0], point[1], lat1, lon1, lat2, lon2)
    return (d ** 2) / (sigma ** 2)

def get_transition_cost(edge_from, edge_to, graph, w_complex, w_dist):
    """
    Compute the transition cost from one candidate edge to the next.
    - If staying on the same edge: cost = 0.
    - If moving from edge_from to edge_to and they are connected (head of edge_from equals tail of edge_to),
      cost = w_complex + w_dist * (length of edge_to).
    - Otherwise, the transition is invalid (returns infinity).
    """
    if edge_from == edge_to:
        return 0.0
    
    # Check connectivity: tail of edge_to should equal head of edge_from.
    u1, v1 = edge_from
    u2, v2 = edge_to
    if v1 == u2:
        return w_complex + w_dist * compute_edge_length(graph, edge_to)
    
    return float('inf')

def hmm_map_match(graph, route_segments, sigma=0.001, w_dist=1.0, w_complex=1.0, desired_spacing=0.01, min_points=4):
    """
    Perform HMM-based map matching.
    
    Parameters:
      graph: A NetworkX directed graph with node attributes 'lat' and 'lon'.
      route_segments: A list of dicts, each defining a segment with keys:
                      'from_lat', 'from_lon', 'to_lat', 'to_lon'.
      sigma: Parameter for emission cost (controls sensitivity to distance).
      w_dist: Weight for distance in the transition cost.
      w_complex: Weight for complexity (penalizing new edge transitions).
      desired_spacing: Desired spacing between points when sampling (in degrees)
      min_points: Minimum number of points to sample from any segment
    
    Returns:
      final_path: A list of edges representing the matched route.
    """
    # Step 1: Sample points along the dataset route.
    observations = []
    for seg in route_segments:
        pts = sample_points_from_segment(seg, min_points=min_points, desired_spacing=desired_spacing)
        observations.extend(pts)
    
    # Step 2: Prepare candidate states.
    # For simplicity, consider every edge in the graph as a candidate.
    edges = list(graph.edges())
    num_obs = len(observations)
    num_edges = len(edges)
    
    print(f'Number of observations: {num_obs}')
    print(f'Number of edges: {num_edges}')
    
    import time 
    time_start = time.time()
    
    # Build the emission cost table
    # emission_cost[i, j] is the emission cost for the i-th observation and j-th edge
    emission_cost = np.zeros((num_obs, num_edges))
    for i, point in enumerate(observations):
        for j, edge in enumerate(edges):
            cost = get_emission_cost(point, edge, graph, sigma)
            emission_cost[i, j] = cost
    
    print(f'Time taken to build emission cost: {time.time() - time_start} seconds')

    time_start = time.time()
    # Precompute edge successors and predecessors
    edge_successors = {}
    edge_predecessors = {j: [] for j in range(num_edges)}
    
    # First create an adjacency mapping from nodes to edge indices
    node_to_outgoing_edges = {}
    for k, (u, v) in enumerate(edges):
        if v not in node_to_outgoing_edges:
            node_to_outgoing_edges[v] = []
        if u not in node_to_outgoing_edges:
            node_to_outgoing_edges[u] = []
        node_to_outgoing_edges[v].append(k)
    
    # Now build the edge successors and predecessors in O(E) time
    for k, (u, v) in enumerate(edges):
        # Include self (staying on same edge)
        successors = [k]
        # Add all edges starting with v
        if v in node_to_outgoing_edges:
            for j in node_to_outgoing_edges[v]:
                if j != k:  # Avoid adding self twice
                    successors.append(j)
                    edge_predecessors[j].append(k)
        
        # Add self as predecessor (staying on same edge)
        edge_predecessors[k].append(k)
        edge_successors[k] = successors
    
    print(f'Time taken to build edge successors and predecessors: {time.time() - time_start} seconds')
    
    # On-the-fly transition cost cache
    transition_costs = {}
    
    # Function to get transition cost with caching
    def get_cached_transition_cost(k, j):
        if (k, j) not in transition_costs:
            edge_from = edges[k]
            edge_to = edges[j]
            trans_cost = get_transition_cost(edge_from, edge_to, graph, w_complex, w_dist)
            transition_costs[(k, j)] = trans_cost
        return transition_costs[(k, j)]

    # Initialize DP table for the Viterbi algorithm.
    dp = np.full((num_obs, num_edges), float('inf'))
    backpointer = np.full((num_obs, num_edges), -1, dtype=int)
    
    # Initialization for the first observation.
    for j in range(num_edges):
        dp[0, j] = emission_cost[0, j]
        
    from tqdm import tqdm
    
    # Recurrence: compute best cost for each observation and candidate edge.
    for i in tqdm(range(1, num_obs), desc='Running Viterbi algorithm'):
        for j in range(num_edges):
            # Only consider previous edges that could lead to the current edge
            for k in edge_predecessors[j]:
                # Get transition cost from cache or compute it on-demand
                trans_cost = get_cached_transition_cost(k, j)
                cost = dp[i - 1, k] + trans_cost + emission_cost[i, j]
                if cost < dp[i, j]:
                    dp[i, j] = cost
                    backpointer[i, j] = k
    
    # Backtrack to recover the optimal sequence of edges.
    best_last = np.argmin(dp[num_obs - 1])
    best_path_edges = []
    j = best_last
    for i in range(num_obs - 1, -1, -1):
        best_path_edges.append(edges[j])
        j = backpointer[i, j]
    
    # Post-process: remove consecutive duplicate edges.
    final_path = []
    prev = None
    for e in best_path_edges:
        if e != prev:
            final_path.append(e)
        prev = e
    
    return final_path

# Example usage:
if __name__ == "__main__":
    # Build a simple directed graph for demonstration.
    G = nx.DiGraph()
    # Add nodes with 'lat' and 'lon'
    G.add_node(1, lat=0.0, lon=0.0)
    G.add_node(2, lat=0.0, lon=1.0)
    G.add_node(3, lat=1.0, lon=1.0)
    G.add_node(4, lat=1.0, lon=0.0)
    
    # Add directed edges between nodes.
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    G.add_edge(3, 4)
    G.add_edge(4, 1)
    
    # Create a sample route (simulated ADS-B segments).
    # Here the route roughly goes from near node 1 to near node 3.
    route_segments = [
        {'from_lat': 0.1, 'from_lon': 0.1, 'to_lat': 0.0, 'to_lon': 0.9},
        {'from_lat': 0.0, 'from_lon': 0.9, 'to_lat': 0.9, 'to_lon': 1.0},
        {'from_lat': 0.9, 'from_lon': 1.0, 'to_lat': 0.9, 'to_lon': 0.2},
    ]
    
    # Run the map-matching algorithm.
    matched_path = hmm_map_match(G, route_segments, sigma=0.1, w_dist=1.0, w_complex=2.0, desired_spacing=0.001, min_points=4)
    print("Matched path:", matched_path)