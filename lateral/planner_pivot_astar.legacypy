# Global cache for admissible pivot nodes computations.
# We include the key node (start), endpoint, direction, and a hashable version of nodes_to_exclude.
pivot_cache = {}

def is_same_direction(origin_latlon, dest_latlon, node_latlon, neighbor_latlon):
    """
    Determine if two segments are in the same general direction.
    
    Parameters:
        origin_latlon (tuple): (latitude, longitude) of the origin point.
        dest_latlon (tuple): (latitude, longitude) of the destination of the first segment.
        node_latlon (tuple): (latitude, longitude) of the starting point of the second segment.
        neighbor_latlon (tuple): (latitude, longitude) of the ending point of the second segment.
        
    Returns:
        bool: True if the two segments are in the same general direction (dot product > 0), 
              indicating they are not "turning around", False otherwise.
    """
    # Create vectors for the segments
    vector1 = (dest_latlon[0] - origin_latlon[0], dest_latlon[1] - origin_latlon[1])
    vector2 = (neighbor_latlon[0] - node_latlon[0], neighbor_latlon[1] - node_latlon[1])
    
    # Compute the dot product
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    
    # If dot_product > 0, then the segments point roughly in the same direction
    return dot_product > 0



def is_same_direction_by_wp_name(G, origin_wp_name, dest_wp_name, node_wp_name, neighbor_wp_name):
    # Extract coordinates for each waypoint
    origin_latlon = (G.nodes[origin_wp_name]['lat'], G.nodes[origin_wp_name]['lon'])
    dest_latlon = (G.nodes[dest_wp_name]['lat'], G.nodes[dest_wp_name]['lon'])
    node_latlon = (G.nodes[node_wp_name]['lat'], G.nodes[node_wp_name]['lon'])
    neighbor_latlon = (G.nodes[neighbor_wp_name]['lat'], G.nodes[neighbor_wp_name]['lon'])
    
    # Use the existing is_same_direction function with the extracted coordinates
    return is_same_direction(origin_latlon, dest_latlon, node_latlon, neighbor_latlon)

def is_same_direction_by_wp_name_sham(G, origin_wp_name, dest_wp_name, node_wp_name, neighbor_wp_name):
    return True


from heapq import heappush, heappop
from utils.haversine import haversine_distance


def find_admissible_pivot_nodes_astar(G, start, target_coords, nodes_to_exclude, max_depth=8,
                                     prevent_backtracking=False, origin=None, dest=None,
                                     direction='forward', node_coords=None, max_pivots=100):
    from heapq import heappop, heappush
    open_set = [(0, 0, start)]  # (f_score, depth, node)
    g_scores = {start: 0}       # Cost from start to node
    depths = {start: 0}         # Depth of each node
    admissible_nodes = set()    # Set of pivot points
    pivot_count = 0             # Counter for pivot points

    is_first_node = True 

    while open_set and pivot_count < max_pivots:
        f_score, depth, node = heappop(open_set)
        print(f'f_score: {f_score}, depth: {depth}, node: {node}')
        if depth > max_depth:
            continue
        if not is_first_node and ('_' in node or node in nodes_to_exclude):
            continue

        admissible_nodes.add(node)
        pivot_count += 1

        # Explore neighbors
        neighbors = G.successors(node) if direction == 'forward' else G._predecessors_map.get(node, [])
        for neighbor in neighbors:
            if neighbor in nodes_to_exclude or '_' in neighbor:
                continue

            # Optional backtracking prevention
            if prevent_backtracking:
                ref_from, ref_to = (origin, dest) if direction == 'forward' else (dest, origin)
                if not is_same_direction_by_wp_name(G, ref_from, ref_to, node, neighbor):
                    continue

            # Calculate edge distance and new g_score
            edge_dist = G.edges[node, neighbor].get('distance', 1) if direction == 'forward' else \
                        G.edges[neighbor, node].get('distance', 1)
            new_g = g_scores[node] + edge_dist
            new_depth = depths[node] + 1

            if new_depth > max_depth:
                continue

            # Heuristic calculation
            h = haversine_distance(*node_coords[neighbor], *target_coords) if node_coords else 0
            f_score = new_g + h

            # Update neighbor if a better path is found
            if (neighbor not in g_scores or new_g < g_scores[neighbor] or
                (new_g == g_scores[neighbor] and new_depth < depths[neighbor])):
                g_scores[neighbor] = new_g
                depths[neighbor] = new_depth
                heappush(open_set, (f_score, new_depth, neighbor))

    return admissible_nodes, g_scores

# Main procedure
def find_pivot_points_with_astar(G, node_from, node_to, nodes_to_exclude,
                                 max_depth=8, node_from_coords=None, node_to_coords=None, max_pivots=100,
                                 prevent_backtracking=True, origin=None, dest=None):
    # Forward A* search
    forward_nodes, forward_dist = find_admissible_pivot_nodes_astar(
        G, node_from, node_to_coords, nodes_to_exclude, max_depth,
        prevent_backtracking=prevent_backtracking, origin=origin, dest=dest, direction='forward',
        max_pivots=max_pivots
    )
    # Backward A* search
    backward_nodes, backward_dist = find_admissible_pivot_nodes_astar(
        G, node_to, node_from_coords, nodes_to_exclude, max_depth,
        prevent_backtracking=prevent_backtracking, origin=origin, dest=dest, direction='backward',
        max_pivots=max_pivots
    )
    # Intersection of admissible nodes
    V = forward_nodes & backward_nodes
    # Distance dictionary
    V_distances = {v: {'forward_dist': forward_dist[v], 'backward_dist': backward_dist[v]} for v in V}
    return V, V_distances



def find_admissible_pivot_nodes(G, node_from, nodes_to_exclude, max_depth = 8, 
                                prevent_backtracking = False, origin = None, dest = None,
                                direction = 'forward'):
    
    # Initialize with starting node at depth 0 and distance 0.
    dict_distances = {node_from: (0, 0)}  # node: (depth, total_distance)
    buff = [node_from] # list of nodes to explore/expand
    admissible_nodes = set()  # use a set to avoid duplicates

    while buff:
        next_level = []
        # Peek at current depth from any node in the current level
        current_depth = dict_distances[buff[0]][0] if buff else 0
        if current_depth >= max_depth:
            break
        
        
        for node in buff: # node to explore/expand
            if '_' in node:
                continue # ignore SID/STAR nodes
            
            # time_start = time.time()
            # Get the set of neighbors to explore (forward will consider successors, backward will consider predecessors)
            if direction == 'forward':
                neighbors = G.successors(node)
            else: 
                if node in G._predecessors_map:
                    neighbors = G._predecessors_map[node]
                else:
                    neighbors = []

            # Explore the neighbors
            for neighbor in neighbors:
                if neighbor in nodes_to_exclude:
                    continue  # Skip excluded nodes
                
                # time_start = time.time()
                if prevent_backtracking:
                    if direction == 'forward':
                        backtrack = not is_same_direction_by_wp_name_sham(G, origin, dest, node, neighbor)
                    else: # backward
                        backtrack = not is_same_direction_by_wp_name_sham(G, dest, origin, node, neighbor)
                    if backtrack:
                        continue # skip this neighbor because it is in the wrong direction
                # print(f'Time taken to check backtrack: {time.time() - time_start}')

                # time_start = time.time()
                # Compute the new path's distance
                if direction == 'forward':
                    edge_distance = G.edges[node, neighbor].get('distance', 1)
                else:
                    edge_distance = G.edges[neighbor, node].get('distance', 1)
                new_distance = dict_distances[node][1] + edge_distance
                new_depth = current_depth + 1

                # Update if the neighbor is either new, or if a better (shorter distance) route is found at the same depth,
                # or if we reached the neighbor with a lower depth.
                if (neighbor not in dict_distances or 
                    new_depth < dict_distances[neighbor][0] or 
                    (new_depth == dict_distances[neighbor][0] and new_distance < dict_distances[neighbor][1])):
                    dict_distances[neighbor] = (new_depth, new_distance)
                    next_level.append(neighbor)
                    admissible_nodes.add(neighbor)
                # print(f'Time taken to update distance: {time.time() - time_start}')
        buff = next_level
    return list(admissible_nodes), dict_distances


import networkx as nx



import networkx as nx

def minimum_description_route(G, route):
    if not route:
        return []
    
    min_route = [route[0]]
    i = 0
    while i < len(route) - 1:
        # Compute all shortest paths from route[i] once.
        sp_dict = nx.single_source_shortest_path(G, route[i])
        # Try candidates from the end of the route backwards.
        for j in range(len(route) - 1, i, -1):
            # Only proceed if route[j] is reachable from route[i]
            if route[j] in sp_dict and sp_dict[route[j]] == route[i:j+1]:
                min_route.append(route[j])
                i = j  # jump to the farthest node found
                break
        else:
            # In a well-formed route, this should not occur.
            i += 1

    return min_route




def check_ac_valid(a, c, route, mdr = None, route_graph = None):
    if mdr is None:
        if route_graph is None:
            raise Exception("route_graph is required when mdr is not provided")
        mdr = minimum_description_route(route_graph, route)
    
    if a > c:
        return False, -1, [] # a should be before c in the route
    
    # Check if there is at most one key node in the range a (inclusive) to c (exclusive)
    key_node_count = 0
    key_nodes = []
    for i in range(a, c):
        if route[i] in mdr:
            key_nodes.append(route[i])
            key_node_count += 1

    if key_node_count > 1:
        return False, key_node_count, key_nodes
    
    return True, key_node_count, key_nodes

import numpy as np



def collapse_pivot_options(V_distances, threshold=1e-6):
    # Create a list of (node, total_distance) tuples
    nodes_with_distances = []
    for node, distances in V_distances.items():
        total_dist = distances['forward_dist'][1] + distances['backward_dist'][1]
        nodes_with_distances.append((node, total_dist))
    
    # Sort by total distance
    nodes_with_distances.sort(key=lambda x: x[1])
    
    # Group nodes with similar distances
    V_distances_collapsed = {}
    node_to_collapsed_node_mapping = {}  # Maps each node to its representative
    current_group = []
    current_dist = None
    
    for node, dist in nodes_with_distances:
        if current_dist is None or abs(dist - current_dist) > threshold:
            # Start a new group
            if current_group:
                # Add the first node from previous group to collapsed dict
                representative = current_group[0]
                V_distances_collapsed[representative] = V_distances[representative]
                
                # Map all nodes in the group to their representative
                for grouped_node in current_group:
                    node_to_collapsed_node_mapping[grouped_node] = representative
            
            current_group = [node]
            current_dist = dist
        else:
            # Add to current group
            current_group.append(node)
    
    # Don't forget to add the last group
    if current_group:
        representative = current_group[0]
        V_distances_collapsed[representative] = V_distances[representative]
        
        # Map all nodes in the last group to their representative
        for grouped_node in current_group:
            node_to_collapsed_node_mapping[grouped_node] = representative
    
    return V_distances_collapsed, node_to_collapsed_node_mapping

def compute_pivot_probabilities(pivots, mu = 1):
    # List pivot keys and compute total distances.
    keys = list(pivots.keys())
    totals = np.array([
        pivots[key]['forward_dist'][1] + pivots[key]['backward_dist'][1]
        for key in keys
    ])
    
    # Compute softmax probabilities (subtract max for numerical stability)
    exp_totals = np.exp(-(totals - np.min(totals)) / mu)
    probabilities = exp_totals / exp_totals.sum()
    return probabilities


def sample_pivot(pivots, probabilities = None, mu = 1, num_samples = 1,
                 old_pivot = None, node_to_collapsed_node_mapping = None): # mu: temperature, larger mu -> more uniform sampling
    keys = list(pivots.keys())
    if probabilities is None:
        probabilities = compute_pivot_probabilities(pivots, mu)

    # If old_pivot is provided, find its probability
    old_pivot_prob = None
    if old_pivot is not None:
        # If old_pivot is directly in keys, use its probability
        if old_pivot in keys:
            old_pivot_index = keys.index(old_pivot)
            old_pivot_prob = probabilities[old_pivot_index]
        # If old_pivot is not in keys but we have a mapping, use the representative's probability
        elif node_to_collapsed_node_mapping is not None and old_pivot in node_to_collapsed_node_mapping:
            representative = node_to_collapsed_node_mapping[old_pivot]
            if representative in keys:
                representative_index = keys.index(representative)
                old_pivot_prob = probabilities[representative_index]
    
    # Sample pivots based on softmax probability distribution
    sampled_indices = np.random.choice(len(keys), p=probabilities, size=num_samples)
    sampled_pivots = [keys[i] for i in sampled_indices]
    sampled_probs = [probabilities[i] for i in sampled_indices]
    
    # Return both the sampled pivots and their corresponding probabilities
    if num_samples == 1:
        return sampled_pivots[0], sampled_probs[0], old_pivot_prob
    else:
        return sampled_pivots, sampled_probs, old_pivot_prob



def get_shortest_path(G, node1, node2):
    return nx.shortest_path(G, node1, node2)

def replace_route_segment(G, route, a, c, v):
    """
    Replaces the segment route[a:c] (with a included and c excluded) with 
    the concatenation of the shortest path from route[a-1] to v and from v to route[c].
    
    The new route is built as follows:
      - prefix: all nodes before index a (i.e. route[:a])
      - sp1: shortest path from route[a-1] to v, with its first node removed
             (because route[a-1] is already in the prefix)
      - sp2: shortest path from v to route[c], with its first node removed 
             (to avoid duplicating v)
      - suffix: all nodes after index c (i.e. route[c+1:])
    
    Thus:
      new_route = route[:a] + sp1[1:] + sp2[1:] + route[c+1:]
    
    The marker indices are updated so that:
      - a' remains a (since the prefix is unchanged)
      - c' becomes the index of the node that was originally at route[c] in new_route.
        Since sp1[1:] has length L1 = len(get_shortest_path(route[a-1], v)) - 1 and 
        sp2[1:] has length L2 = len(get_shortest_path(v, route[c])) - 1,
        the new index of route[c] is:
          new_c = a + L1 + L2 - 1  (subtract 1 because the last element of sp2[1:] is route[c])
    
    Additionally, we count consecutive duplicate entries between the new inserted segment
    and the original replaced segment (route[a:c]) in two parts:
      - dup_count1: for the part from index a up to (and including) the occurrence of v,
      - dup_count2: for the part from the occurrence of v to index c-1.
    
    (This counting assumes that v appears in the original route segment; if not, you might
     decide to set the counts to zero.)
    
    Returns:
      new_route, new_a, new_c, dup_count1, dup_count2
    """
    # Get the prefix and suffix
    prefix = route[:a]               # indices 0 ... a-1 (includes route[a-1])
    suffix = route[c+1:]             # nodes after route[c]
    
    # Get the two shortest paths:
    sp1 = get_shortest_path(G, route[a-1], v)  # from route[a-1] to v, includes both endpoints
    sp2 = get_shortest_path(G, v, route[c])      # from v to route[c], includes both endpoints
    
    # Remove duplicate endpoints:
    # - sp1[0] is route[a-1], already the last node in prefix
    # - sp2[0] is v, which is the same as sp1[-1]
    sp1_part = sp1[1:]  # from after route[a-1] to v
    sp2_part = sp2[1:]  # from after v to route[c]
    
    # Build the new route.
    new_route = prefix + sp1_part + sp2_part + suffix # sp2_part includes node[c], suffix includes node[c+1] onwards
    
    # new_a stays the same (since prefix did not change)
    new_a = a
    
    # new_c: route[c] is now the last node in sp2 (which is the last element of sp2_part)
    # Its index is: len(prefix) + len(sp1_part) + len(sp2_part) - 1
    new_c = len(prefix) + len(sp1_part) + len(sp2_part) - 1

    # Now, count consecutive duplicated entries between the new segment and the old segment.
    # We consider the replaced segment in the original route:
    old_segment = route[a:c]
    # In the new route, the inserted segment is:
    new_segment = sp1_part + sp2_part[:-1] # sp2_part includes node[c], so we exclude it

    # For the duplicate count we assume that v is present in the original segment.
    # Find its index in the original replaced segment (if present)
    try:
        old_v_index = old_segment.index(v)
    except ValueError:
        # v is not in the old segment; in that case, we set both counts to zero.
        dup_count1 = 0
        dup_count2 = 0
    else:
        # In the new segment, v is at the junction: its index is at the end of sp1_part.
        new_v_index = len(sp1_part) - 1

        def count_consecutive_matches(lst1, lst2):
            count = 0
            for x, y in zip(lst1, lst2):
                if x == y:
                    count += 1
                else:
                    break
            return count

        dup_count1 = count_consecutive_matches(old_segment[:old_v_index+1],
                                               new_segment[:new_v_index+1])
        dup_count2 = count_consecutive_matches(old_segment[old_v_index:],
                                               new_segment[new_v_index:])

    return new_route, new_a, new_c, dup_count1, dup_count2



def evaluate_route(G, route):
    """
    Calculate the total distance of a route.
    
    Parameters:
        G (networkx.DiGraph): The graph containing the nodes and edges.
        route (list): A list of node IDs representing the route.
        
    Returns:
        float: The total distance of the route, which is the sum of all edges' distance property.
    """
    if not route or len(route) < 2:
        return 0
    
    total_distance = 0
    for i in range(len(route) - 1):
        current_node = route[i]
        next_node = route[i + 1]
        
        # Check if the edge exists in the graph
        if G.has_edge(current_node, next_node):
            # Add the distance of this edge to the total
            edge_distance = G.edges[current_node, next_node].get('distance', 0)
            total_distance += edge_distance
        else:
            # If the edge doesn't exist, we might want to handle this case
            # For now, we'll just skip it and not add any distance
            pass
    
    return total_distance



def mh_acceptance(cost_new, cost_old, p_forward, p_backward):
    import math
    
    # Calculate the cost difference term
    cost_diff = -(cost_new - cost_old)
    
    # Calculate the proposal probability ratio
    prob_ratio = p_backward / p_forward
    
    # Calculate the acceptance rate using the Metropolis-Hastings formula
    alpha = min(1, math.exp(cost_diff) * prob_ratio)
    
    return alpha



def find_eligible_range_of_nodes_to_sample(route):
    """
    Given a list of waypoints (route), return a tuple (start_idx, end_idx) representing 
    the contiguous index range (inclusive) in the middle of the route (i.e. excluding 
    the first and last waypoints) that contains no underscores.
    
    If there are multiple valid contiguous blocks, the longest one is returned.
    If no eligible middle waypoints are found, returns None.
    """
    # There must be at least three waypoints (start, at least one middle, end)
    if len(route) < 3:
        return None

    # Consider only the middle nodes: indices 1 to len(route)-2 inclusive.
    # Create a list of indices where the waypoint does not contain an underscore.
    eligible_indices = [i for i in range(1, len(route) - 1) if '_' not in route[i]]
    
    # If there are no eligible middle waypoints, return None.
    if not eligible_indices:
        return None

    # Find the longest contiguous block of eligible indices.
    longest_start = current_start = eligible_indices[0]
    longest_length = current_length = 1

    for idx in eligible_indices[1:]:
        # If this index continues the current contiguous block...
        if idx == current_start + current_length:
            current_length += 1
        else:
            # Check if the current block is the longest so far.
            if current_length > longest_length:
                longest_length = current_length
                longest_start = current_start
            # Reset the current block
            current_start = idx
            current_length = 1

    # Final check in case the last block is the longest.
    if current_length > longest_length:
        longest_length = current_length
        longest_start = current_start

    longest_end = longest_start + longest_length - 1

    return [longest_start, longest_end]


import time


import heapq

def find_bidirectional_admissible_pivot_nodes(G, start_node, end_node, nodes_to_exclude, max_depth=8,
                                             prevent_backtracking=False, origin=None, dest=None):
    """
    Find admissible pivot nodes by performing a single bidirectional search from start_node and end_node.
    
    Parameters:
        G: The graph to traverse
        start_node: The starting node (route[a-1])
        end_node: The ending node (route[c])
        nodes_to_exclude: Set of nodes to exclude from consideration
        max_depth: Maximum exploration depth
        prevent_backtracking: Whether to prevent backtracking
        origin, dest: Used for backtracking prevention
        
    Returns:
        V_distances: Dictionary of admissible pivot nodes with their distances from both directions
    """
    # Initialize distance dictionaries
    forward_distances = {start_node: (0, 0)}  # node: (depth, total_distance)
    backward_distances = {end_node: (0, 0)}  # node: (depth, total_distance)
    
    # Initialize priority queues for distance-ordered traversal
    # Format: (total_distance, depth, node)
    forward_queue = [(0, 0, start_node)]
    backward_queue = [(0, 0, end_node)]
    
    # Keep track of visited nodes in each direction
    forward_visited = set([start_node])
    backward_visited = set([end_node])
    
    # Dictionary for admissible nodes (nodes reached from both directions)
    V_distances = {}
    
    # Early termination threshold - tunable parameter
    early_termination_threshold = 200
    
    # Process both queues alternately until both are empty or max depth is reached
    while forward_queue and backward_queue:
        # Process a node from the forward direction
        if forward_queue:
            _, current_forward_depth, node = heapq.heappop(forward_queue)
            
            # Skip if we've reached max depth
            if current_forward_depth >= max_depth:
                continue
                
            # Skip SID/STAR nodes
            if '_' in node:
                continue
                
            # Check if this node has been visited from the backward direction
            if node in backward_visited and '_' not in node:
                # This is an admissible node, add it to V_distances
                V_distances[node] = {
                    'forward_dist': forward_distances[node],
                    'backward_dist': backward_distances[node]
                }
                
                # Check for early termination
                if len(V_distances) >= early_termination_threshold:
                    return V_distances
            
            # Get successors for forward traversal
            for neighbor in G.successors(node):
                if neighbor in nodes_to_exclude or neighbor in forward_visited:
                    continue
                
                # Check for backtracking if needed
                if prevent_backtracking:
                    backtrack = not is_same_direction_by_wp_name(G, origin, dest, node, neighbor)
                    if backtrack:
                        continue
                
                # Compute new distance
                edge_distance = G.edges[node, neighbor].get('distance', 1)
                new_distance = forward_distances[node][1] + edge_distance
                new_depth = current_forward_depth + 1
                
                # Update if better path found or if node is new
                if (neighbor not in forward_distances or 
                    new_depth < forward_distances[neighbor][0] or 
                    (new_depth == forward_distances[neighbor][0] and new_distance < forward_distances[neighbor][1])):
                    forward_distances[neighbor] = (new_depth, new_distance)
                    heapq.heappush(forward_queue, (new_distance, new_depth, neighbor))
                    forward_visited.add(neighbor)
                    
                    # Check if neighbor has been visited from backward direction
                    if neighbor in backward_visited and '_' not in neighbor:
                        V_distances[neighbor] = {
                            'forward_dist': forward_distances[neighbor],
                            'backward_dist': backward_distances[neighbor]
                        }
                        
                        # Check for early termination
                        if len(V_distances) >= early_termination_threshold:
                            return V_distances
        
        # Process a node from the backward direction
        if backward_queue:
            _, current_backward_depth, node = heapq.heappop(backward_queue)
            
            # Skip if we've reached max depth
            if current_backward_depth >= max_depth:
                continue
                
            # Skip SID/STAR nodes
            if '_' in node:
                continue
                
            # Get predecessors for backward traversal
            neighbors = G._predecessors_map.get(node, [])
            
            for neighbor in neighbors:
                if neighbor in nodes_to_exclude or neighbor in backward_visited:
                    continue
                
                # Check for backtracking if needed
                if prevent_backtracking:
                    backtrack = not is_same_direction_by_wp_name_sham(G, dest, origin, node, neighbor)
                    if backtrack:
                        continue
                
                # Compute new distance
                edge_distance = G.edges[neighbor, node].get('distance', 1)
                new_distance = backward_distances[node][1] + edge_distance
                new_depth = current_backward_depth + 1
                
                # Update if better path found or if node is new
                if (neighbor not in backward_distances or 
                    new_depth < backward_distances[neighbor][0] or 
                    (new_depth == backward_distances[neighbor][0] and new_distance < backward_distances[neighbor][1])):
                    backward_distances[neighbor] = (new_depth, new_distance)
                    heapq.heappush(backward_queue, (new_distance, new_depth, neighbor))
                    backward_visited.add(neighbor)
                    
                    # Check if neighbor has been visited from forward direction
                    if neighbor in forward_visited and '_' not in neighbor:
                        V_distances[neighbor] = {
                            'forward_dist': forward_distances[neighbor],
                            'backward_dist': backward_distances[neighbor]
                        }
                        
                        # Check for early termination
                        if len(V_distances) >= early_termination_threshold:
                            return V_distances
    
    return V_distances


def mcmc_step(route_graph, route, temperature = 1000, max_depth = 4,
              verbose = False):
    start_time = time.time()
    # ================================
    # SAMPLE A AND C
    # ================================

    # Find the minimum description route in order to ensure the reversibility of SPLICE operation
    time_start = time.time()
    # route_minimum = minimum_description_route(route_graph, route)
    if verbose:
        print(f'Time taken to find the minimum description route: {time.time() - time_start}')
    route_len = len(route)

    # Find the eligible range of nodes to be sampled
    eligible_range = find_eligible_range_of_nodes_to_sample(route)
    if eligible_range[0] > 0:
        eligible_range[0] += 1 # SID and STAR nodes were involved, we don't get the node next to SID or STAR entries/exits because the distance is zero 
    if eligible_range[1] < route_len - 1:
        eligible_range[1] -= 1 # like above

    # Sample index a, the route at index a (inclusive) onward until c-1 (inclusive) will probably be replaced
    a = np.random.randint(eligible_range[0], eligible_range[1] + 1)

    # # Sample index c, ensuring that between a (inclusive) and c-1 (inclusive),
    # # there is at most one node that belongs to route_minimum (i.e., 0 or 1 key node)
    
    # # Initialize c to be one position after a
    # c = a + 1
    # if c > eligible_range[1]:
    #     return route, False # reject the move because c is out of eligible range
    
    # # Count how many nodes in route_minimum are in the current range
    # key_node_count = 0
    
    # # Keep track of which nodes in the range are in route_minimum
    # key_nodes = []
    
    # # Extend c as far as possible while maintaining at most one key node
    # while c < eligible_range[1]:
    #     # Check if the current node is in route_minimum
    #     if route[c-1] in route_minimum:
    #         key_node_count += 1
    #         key_nodes.append(route[c-1])
            
    #     # If we've exceeded our limit of key nodes, back up one position
    #     if key_node_count > 1:
    #         c -= 1
    #         key_nodes.pop()  # Remove the last added key node
    #         break
            
    #     # Otherwise, try to extend further
    #     c += 1
    
    # # If we've reached the end of the route, make sure c doesn't exceed eligible_range[1]
    # c = min(c, eligible_range[1])

    # # Resample c between a+1 (inclusive) and c (inclusive) to account for both zero and one key node cases
    # # First, check if we have valid range to sample from
    # if c > a + 1:
    #     # We have a range to sample from
    #     new_c = np.random.randint(a + 1, c + 1)  # +1 because randint upper bound is exclusive
    #     c = new_c
    
    # # Recount key nodes in the potentially new range
    # key_node_count = 0
    # key_nodes = []
    # for i in range(a, c):
    #     if route[i] in route_minimum:
    #         key_nodes.append(route[i])
    #         key_node_count += 1
            
    # if len(key_nodes) > 1:
    #     raise ValueError(f'len(key_nodes) = {len(key_nodes)}, maximum is 1!')

    # if verbose: 
    #     print(f'Time taken to sample a and c: {time.time() - start_time}')
    
    # Sample c from a+1 to eligible_range[1]
    c = np.random.randint(a+1, eligible_range[1] + 1)
    
    if verbose:
        print(f'Time taken to sample a and c: {time.time() - start_time}')
    start_time = time.time()
    # ================================
    # FORWARD AND BACKWARD ADMISSIBLE PIVOT NODES
    # ================================
    nodes_to_exclude = set(route[:a]) | set(route[c:])
    admissible_nodes_forward, forward_distances = find_admissible_pivot_nodes(route_graph, route[a-1], nodes_to_exclude,
                                                                          max_depth=max_depth, prevent_backtracking = True, origin = route[a-1], dest = route[c])
    print(f'Time taken to find forward admissible nodes: {time.time() - start_time}')
    start_time = time.time()
    admissible_nodes_backward, backward_distances = find_admissible_pivot_nodes(route_graph, route[c], nodes_to_exclude,
                                                                          max_depth=max_depth, prevent_backtracking = True, origin = route[a-1], dest = route[c], direction = 'backward')
    print(f'Time taken to find backward admissible nodes: {time.time() - start_time}')
    start_time = time.time()
    # Find the intersection of forward and backward admissible nodes
    V = set(admissible_nodes_forward) & set(admissible_nodes_backward)
    # We also remove all nodes that are part of a SID or STAR node (containing an underscore)
    V = {v for v in V if '_' not in v}
    # Create a dictionary of nodes with their forward and backward distances
    V_distances = {v: {
        'forward_dist': forward_distances[v], 
        'backward_dist': backward_distances[v]
    } for v in V}

    # node_from_coords = route_graph.nodes[route[a-1]]['lat'], route_graph.nodes[route[a-1]]['lon']
    # node_to_coords = route_graph.nodes[route[c]]['lat'], route_graph.nodes[route[c]]['lon']
    # V, V_distances = find_pivot_points_with_astar(route_graph,
    #                                               route[a-1], route[c], nodes_to_exclude,
    #                                               max_depth=max_depth, prevent_backtracking = True,
    #                                               origin = route[a-1], dest = route[c],
    #                                               node_from_coords = node_from_coords, node_to_coords = node_to_coords)
    
    
    
    # V_distances = find_bidirectional_admissible_pivot_nodes(
    # route_graph, route[a-1], route[c], nodes_to_exclude,
    # max_depth=max_depth, prevent_backtracking=True, origin=route[a-1], dest=route[c])
    
    if verbose:
        print(f'Time taken to find bidirectional admissible pivot nodes: {time.time() - start_time}')
    
    # Collapse pivot nodes with similar distances
    V_distances_collapsed, node_to_collapsed_node_mapping = collapse_pivot_options(V_distances)
    
    old_pivot_node = min(V_distances_collapsed, key=lambda x: V_distances_collapsed[x]['forward_dist'][1] + V_distances_collapsed[x]['backward_dist'][1]) 

    # if len(key_nodes) == 0:
    #     # If key_nodes is empty, the current pivot is the one with lowest total distance in V_distances_collapsed
        
    # elif len(key_nodes) == 1: # len(key_nodes) == 1
    #     # If key_nodes is not empty, the current pivot is the one in key_nodes
    #     old_pivot_node = key_nodes[0]
    # else: # len(key_nodes) > 1
    #     raise ValueError(f'len(key_nodes) = {len(key_nodes)}, maximum is 1!')
    
    if verbose:
        print(f'Time taken to find the pivot node: {time.time() - start_time}')
    start_time = time.time()
    # ================================
    # SAMPLE PIVOT NODE V
    # ================================
    pivot_probabilities = compute_pivot_probabilities(V_distances_collapsed, mu = temperature)
    sampled_V, prob_sampled_V, old_pivot_prob = sample_pivot(V_distances_collapsed, pivot_probabilities, num_samples=1,
                                                             old_pivot = old_pivot_node, node_to_collapsed_node_mapping = node_to_collapsed_node_mapping)

    if old_pivot_node is None:
        raise ValueError('old_pivot_node is None!')

    if verbose:
        print(f'Time taken to sample the pivot node: {time.time() - start_time}')
    start_time = time.time()
    # ================================
    # PROPOSE NEW ROUTE BY SPLICING THE NODE V INTO THE ROUTE
    # ================================
    new_route, new_a, new_c, dup_count1, dup_count2 = replace_route_segment(route_graph, route, a, c, sampled_V)

    if verbose:
        print(f'Time taken to propose the new route: {time.time() - start_time}')
    start_time = time.time()
    # ================================
    # EVALUATE THE COST OF THE NEW ROUTE
    # ================================
    cost_new = evaluate_route(route_graph, new_route)
    cost_old = evaluate_route(route_graph, route)

    if verbose:
        print(f'Time taken to evaluate the cost of the new route: {time.time() - start_time}')
    start_time = time.time()
    # ================================
    # ACCEPTANCE/REJECTION
    # ================================
    acceptance_prob = mh_acceptance(cost_new, cost_old, prob_sampled_V, old_pivot_prob)

    if verbose:
        print(f'Time taken to compute the acceptance probability: {time.time() - start_time}')
    # ================================
    # ACCEPTANCE/REJECTION
    # ================================
    # Accept the new route with probability acceptance_prob
    if np.random.random() < acceptance_prob:
        # Accept the new route
        return new_route, True
    else:
        # Reject the new route, keep the old one
        return route, False
    