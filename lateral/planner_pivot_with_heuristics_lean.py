import time 


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



import math

def find_admissible_pivot_nodes_with_heuristics(G, node_from, nodes_to_exclude, max_depth=8, 
                                prevent_backtracking=False, origin=None, dest=None,
                                direction='forward', branching_factor=None, heuristic=None):
    """
    Finds admissible pivot nodes using a layered expansion (BFS-like) combined with 
    a heuristic to prioritize node expansion within each depth level.
    
    Parameters:
        G: NetworkX graph with nodes having 'lat' and 'lon' attributes.
        node_from (str): Starting node.
        nodes_to_exclude (set): Nodes to avoid during expansion.
        max_depth (int): Maximum depth (number of layers) to search.
        prevent_backtracking (bool): Whether to prevent backtracking via directional checks.
        origin (str): Name of the origin waypoint (used in directional check).
        dest (str): Name of the destination waypoint (used in directional check).
        direction (str): 'forward' (use successors) or 'backward' (use predecessors).
        branching_factor (int): Maximum number of nodes to explore per depth layer.
        heuristic (callable): Function that accepts (node, current_distance) and returns a numeric score.
                              Lower scores mean higher priority.
                              If None and dest is provided, a default heuristic is used.
    
    Returns:
        admissible_nodes (list): List of nodes that were admissibly reached.
        dict_distances (dict): Mapping of node -> (depth, total_distance)
    """
    # Initialize with starting node at depth 0 and distance 0.
    dict_distances = {node_from: (0, 0)}  # node: (depth, total_distance)
    buff = [node_from]  # current layer of nodes to expand
    admissible_nodes = set()  # use a set to avoid duplicates

    def euclidean_distance(latlon1, latlon2):
        return math.sqrt((latlon1[0]-latlon2[0])**2 + (latlon1[1]-latlon2[1])**2)
    
    # If no heuristic is provided but we have a destination, define a default one.
    if heuristic is None:
        if dest is not None:
            # For forward search, target is dest; for backward, use origin.
            target = dest if direction == 'forward' else origin
            def default_heuristic(node, current_distance):
                node_coord = (G.nodes[node]['lat'], G.nodes[node]['lon'])
                target_coord = (G.nodes[target]['lat'], G.nodes[target]['lon'])
                # Combine current distance with Euclidean distance as the heuristic score.
                return current_distance + euclidean_distance(node_coord, target_coord)
            heuristic = default_heuristic
        else:
            # If no target is provided, use the accumulated distance as the heuristic.
            heuristic = lambda node, current_distance: current_distance

    while buff:
        next_level = []
        # Peek at current depth from any node in the current level.
        current_depth = dict_distances[buff[0]][0]
        if current_depth >= max_depth:
            break

        for node in buff:
            # Skip nodes with an underscore (e.g., SID/STAR nodes)
            if '_' in node:
                continue

            # Determine neighbors based on the search direction.
            if direction == 'forward':
                neighbors = G.successors(node)
            else:
                # For backward, use predecessors if available.
                neighbors = G._predecessors_map[node] if node in G._predecessors_map else []

            # Process neighbors.
            for neighbor in neighbors:
                if neighbor in nodes_to_exclude:
                    continue  # Skip excluded nodes

                if prevent_backtracking:
                    # Check if the neighbor is going in the proper direction.
                    if direction == 'forward':
                        backtrack = not is_same_direction_by_wp_name(G, origin, dest, node, neighbor)
                    else:
                        backtrack = not is_same_direction_by_wp_name(G, dest, origin, node, neighbor)
                    if backtrack:
                        continue  # Skip neighbor that is "turning around"

                # Compute the new path's distance.
                if direction == 'forward':
                    edge_distance = G.edges[node, neighbor].get('distance', 1)
                else:
                    edge_distance = G.edges[neighbor, node].get('distance', 1)
                new_distance = dict_distances[node][1] + edge_distance
                new_depth = current_depth + 1

                # Update if the neighbor is either new or if a better route is found.
                if (neighbor not in dict_distances or 
                    new_depth < dict_distances[neighbor][0] or 
                    (new_depth == dict_distances[neighbor][0] and new_distance < dict_distances[neighbor][1])):
                    dict_distances[neighbor] = (new_depth, new_distance)
                    next_level.append(neighbor)
                    admissible_nodes.add(neighbor)

        # Within the current layer, prioritize nodes using the heuristic.
        # Remove duplicates while preserving order.
        unique_next_level = list(set(next_level))
        unique_next_level.sort(key=lambda n: heuristic(n, dict_distances[n][1]))
        # If a branching factor is provided, only keep the top candidates.
        if branching_factor is not None:
            unique_next_level = unique_next_level[:branching_factor]

        buff = unique_next_level

    return list(admissible_nodes), dict_distances


import networkx as nx
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

    # # For the duplicate count we assume that v is present in the original segment.
    # # Find its index in the original replaced segment (if present)
    # try:
    #     old_v_index = old_segment.index(v)
    # except ValueError:
    #     # v is not in the old segment; in that case, we set both counts to zero.
    #     dup_count1 = 0
    #     dup_count2 = 0
    # else:
    #     # In the new segment, v is at the junction: its index is at the end of sp1_part.
    #     new_v_index = len(sp1_part) - 1

    #     def count_consecutive_matches(lst1, lst2):
    #         count = 0
    #         for x, y in zip(lst1, lst2):
    #             if x == y:
    #                 count += 1
    #             else:
    #                 break
    #         return count

    #     dup_count1 = count_consecutive_matches(old_segment[:old_v_index+1],
    #                                            new_segment[:new_v_index+1])
    #     dup_count2 = count_consecutive_matches(old_segment[old_v_index:],
    #                                            new_segment[new_v_index:])

    return new_route, new_a, new_c



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
    # Sample index c, the route at index c (inclusive) onward until the end of the route will probably be replaced
    c_max = eligible_range[1] + 1
    if c_max <= a+1:
        if verbose:
            print(f'c_max = {c_max}, a = {a}, eligible_range = {eligible_range} - Early rejection')
        return route, False # reject the move because c is out of eligible range
    else:
        c = np.random.randint(a+1, c_max)
        if verbose:
            print(f'a = {a}, c = {c}. Modify: {route[a-2]} {route[a-1]} [{" ".join(route[a:c])}] {route[c]} {route[c+1]}')
    
    if verbose:
        print(f'Time taken to sample a and c: {time.time() - start_time}')
    start_time = time.time()
    # ================================
    # FORWARD AND BACKWARD ADMISSIBLE PIVOT NODES
    # ================================
    nodes_to_exclude = set(route[:a]) | set(route[c:])
    admissible_nodes_forward, forward_distances = find_admissible_pivot_nodes_with_heuristics(route_graph, route[a-1], nodes_to_exclude,
                                                                          max_depth=max_depth, prevent_backtracking = False,
                                                                          origin = route[a-1], dest = route[c],
                                                                          branching_factor = 16)
    if verbose:
        print(f'Time taken to find forward admissible nodes: {time.time() - start_time}')
        print(f'Total forward admissible nodes: {len(admissible_nodes_forward)}')
        for node in admissible_nodes_forward[:10]:
            print(f'  {node}: {forward_distances[node]}')
    start_time = time.time()
    admissible_nodes_backward, backward_distances = find_admissible_pivot_nodes_with_heuristics(route_graph, route[c], nodes_to_exclude,
                                                                          max_depth=max_depth, prevent_backtracking = False,
                                                                          origin = route[a-1], dest = route[c], direction = 'backward',
                                                                          branching_factor = 16)
    if verbose:
        print(f'Time taken to find backward admissible nodes: {time.time() - start_time}')
        print(f'Total backward admissible nodes: {len(admissible_nodes_backward)}')
        for node in admissible_nodes_backward[:10]:
            print(f'  {node}: {backward_distances[node]}')
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


    if verbose:
        print(f'Found {len(V_distances)} admissible pivot nodes')
    
    # Collapse pivot nodes with similar distances
    V_distances_collapsed, node_to_collapsed_node_mapping = collapse_pivot_options(V_distances)

    if len(V_distances_collapsed) == 0:
        if verbose:
            print('No admissible pivot nodes found! Skipping the move.')
        return route, False

    if verbose:
        print(f'Found {len(V_distances_collapsed)} collapsed pivot nodes')
    
    old_pivot_node = min(V_distances_collapsed, key=lambda x: V_distances_collapsed[x]['forward_dist'][1] + V_distances_collapsed[x]['backward_dist'][1]) 
    
    start_time = time.time()
    # ================================
    # SAMPLE PIVOT NODE V
    # ================================
    pivot_probabilities = compute_pivot_probabilities(V_distances_collapsed, mu = temperature)
    sampled_V, prob_sampled_V, old_pivot_prob = sample_pivot(V_distances_collapsed, pivot_probabilities, num_samples=1,
                                                             old_pivot = old_pivot_node, node_to_collapsed_node_mapping = node_to_collapsed_node_mapping)
    
    if verbose:
        # Print some pivot probabilities, sorted by probability:
        print('Pivot probabilities:')
        # Print the first 10 highest pivot probabilities
        keys = list(V_distances_collapsed.keys())
        # Get indices that would sort the probabilities in descending order
        sorted_indices = np.argsort(-pivot_probabilities)
        # Print the top 10 (or fewer if there are less than 10 pivots)
        for i in range(min(10, len(sorted_indices))):
            idx = sorted_indices[i]
            pivot_name = keys[idx]
            probability = pivot_probabilities[idx]
            total_dist = V_distances_collapsed[pivot_name]['forward_dist'][1] + V_distances_collapsed[pivot_name]['backward_dist'][1]
            print(f'  {pivot_name}: prob={probability:.4f}, dist={total_dist:.2f}')
        print(f'Sampled: {sampled_V} with prob {prob_sampled_V:.4f} and total dist {V_distances_collapsed[sampled_V]["forward_dist"][1] + V_distances_collapsed[sampled_V]["backward_dist"][1]:.2f}')

    if old_pivot_node is None:
        raise ValueError('old_pivot_node is None!')

    if verbose:
        print(f'Time taken to sample the pivot node: {time.time() - start_time}')
    start_time = time.time()
    # ================================
    # PROPOSE NEW ROUTE BY SPLICING THE NODE V INTO THE ROUTE
    # ================================
    new_route, new_a, new_c = replace_route_segment(route_graph, route, a, c, sampled_V)

    if verbose:
        print(f'Time taken to propose the new route: {time.time() - start_time}')
        print(f'New route: {new_route[new_a-2]} {new_route[new_a-1]} [{" ".join(new_route[new_a:new_c])}] {new_route[new_c]} {new_route[new_c+1]}')
        print(f'Full new route: {" ".join(new_route)}')
        print(f'Total distance: {V_distances_collapsed[sampled_V]["forward_dist"][1] + V_distances_collapsed[sampled_V]["backward_dist"][1]:.2f}')
    start_time = time.time()
    # ================================
    # EVALUATE THE COST OF THE NEW ROUTE
    # ================================
    cost_new = evaluate_route(route_graph, new_route)
    cost_old = evaluate_route(route_graph, route)
    if verbose:
        print(f'Cost of the new route: {cost_new}')
        print(f'Cost of the old route: {cost_old}')

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
    