from utils.haversine import haversine_distance
import math

def haversine_heuristic(graph, node_from, node_to):
    node_from_coords = graph.nodes[node_from]['lat'], graph.nodes[node_from]['lon']
    node_to_coords = graph.nodes[node_to]['lat'], graph.nodes[node_to]['lon']
    return haversine_distance(*node_from_coords, *node_to_coords)



def compute_bearing(lat1, lon1, lat2, lon2):
    """
    Compute the initial bearing (in degrees) from point A to point B.
    """
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dLon = lon2 - lon1
    x = math.sin(dLon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)
    initial_bearing = math.degrees(math.atan2(x, y))
    # Normalize to [0, 360)
    return (initial_bearing + 360) % 360



def sample_paths(graph, node_from, node_to, max_paths=None, heuristic_fn=None,
                 nodes_to_exclude=None, max_global_deviation=85, max_local_deviation=80):
    """
    Sample paths from node_from to node_to using A* heuristics, ensuring that
    the freshly developed segment of the path does not deviate more than
    max_deviation degrees from the overall direction from node_from to node_to.
    
    Args:
        graph: The graph G
        node_from: Starting node
        node_to: Target node
        max_paths: Maximum number of paths to sample (None for all possible simple paths)
        heuristic_fn: Function that estimates distance from a node to node_to.
                     If None, uses a default heuristic of 0.
        nodes_to_exclude: Optional set of nodes to exclude from paths
        max_deviation: Maximum allowed deviation (in degrees) from the global bearing.
    Returns:
        List of paths, where each path is a list of nodes.
    """
    import heapq
    
    if heuristic_fn is None:
        # Default heuristic: just return 0
        heuristic_fn = lambda graph, node, target: 0

    # Compute the overall bearing from node_from to node_to.
    start_coords = graph.nodes[node_from]['lat'], graph.nodes[node_from]['lon']
    target_coords = graph.nodes[node_to]['lat'], graph.nodes[node_to]['lon']
    global_bearing = compute_bearing(start_coords[0], start_coords[1],
                                     target_coords[0], target_coords[1])
    
    paths = []
    best_path_length = float('inf')  # Track the best path length found so far
    
    # Priority queue of (f_value, path_id, path)
    # f_value = g_value + h_value (path length + heuristic)
    # path_id is used to break ties consistently
    open_set = [(heuristic_fn(graph, node_from, node_to), 0, [node_from])]
    path_id = 1
    visited = set()  # Track visited nodes to avoid redundant exploration

    if nodes_to_exclude is None:
        nodes_to_exclude = set()
    
    # Continue until we've either found max_paths or exhausted all possibilities
    while open_set and (max_paths is None or len(paths) < max_paths):
        # Get the path with the lowest f_value
        f_value, _, path = heapq.heappop(open_set)
        # print(f'{f_value} {" ".join(path)}')
        current = path[-1]
        
        # Skip if we've already visited this node with a better path
        current_path_key = (current, tuple(sorted(set(path))))
        if current_path_key in visited:
            continue
        visited.add(current_path_key)
        
        # If we've reached the target, add this path to our results
        if current == node_to:
            paths.append(path)
            print(f'Found a path: {" ".join(path)}')
            path_length = len(path) - 1
            best_path_length = min(best_path_length, path_length)
            continue
        
        # Explore neighbors
        g_value = len(path) - 1  # Path length so far
        
        for neighbor in graph.neighbors(current):
            # Avoid cycles in the current path and excluded nodes
            if neighbor not in path and neighbor not in nodes_to_exclude:
                # Prevent backtracking heuristics
                # Compute the bearing from the current node to the neighbor
                current_coords = graph.nodes[current]['lat'], graph.nodes[current]['lon']
                neighbor_coords = graph.nodes[neighbor]['lat'], graph.nodes[neighbor]['lon']
                local_bearing = compute_bearing(current_coords[0], current_coords[1],
                                                neighbor_coords[0], neighbor_coords[1])
                
                # Compute the smallest angular difference (accounting for circularity)
                global_bearing_angle_diff = abs(local_bearing - global_bearing)
                
                if global_bearing_angle_diff > 180:
                    global_bearing_angle_diff = 360 - global_bearing_angle_diff

                # Check for sharp turnbacks by comparing with the previous segment's bearing
                if len(path) > 1: # there is a previous segment
                    prev_node = path[-2]
                    prev_coords = graph.nodes[prev_node]['lat'], graph.nodes[prev_node]['lon']
                    prev_bearing = compute_bearing(prev_coords[0], prev_coords[1],
                                                  current_coords[0], current_coords[1])
                    
                    # Calculate angle difference between previous segment and potential new segment
                    segment_bearing_diff = abs(local_bearing - prev_bearing)
                    if segment_bearing_diff > 180:
                        segment_bearing_diff = 360 - segment_bearing_diff
                    
                    # Skip if the turn is too sharp (e.g., more than max_local_deviation degrees)
                    if segment_bearing_diff > max_local_deviation:
                        continue
                
                # Skip if the neighbor deviates too much from the general direction
                if global_bearing_angle_diff > max_global_deviation:
                    continue

                

                new_path = path + [neighbor]
                new_g_value = g_value + 1
                
                # Calculate heuristic from neighbor to target (not from start to neighbor)
                h_value = heuristic_fn(graph, neighbor, node_to)
                f_value = new_g_value + h_value
                
                # Don't push unpromising candidates - paths that are already longer than 
                # the best path we've found plus a tolerance factor
                if paths and new_g_value > best_path_length * 1.5:
                    continue
                
                heapq.heappush(open_set, (f_value, path_id, new_path))
                path_id += 1
    
    return paths



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
import numpy as np

def mcmc_step(route_graph, route, temperature = 1000, max_depth = 4,
              verbose = False):
    start_time = time.time()
    # ================================
    # SAMPLE A AND C
    # ================================
    time_start = time.time()
    # Find the minimum description route in order to ensure the reversibility of SPLICE operation
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
    
    # Sample c from a+1 to eligible_range[1]
    c = np.random.randint(a+1, eligible_range[1] + 1)
    
    if verbose:
        print(f'Time taken to sample a and c: {time.time() - start_time}')

    
    # ================================
    # FORWARD AND BACKWARD ADMISSIBLE PIVOT NODES
    # ================================
    start_time = time.time()
    print(f'Begin finding candidate paths from {route[a-1]} to {route[c]}... | {route[a-3:c+2]}')
    nodes_to_exclude = set(route[:a]) | set(route[c:])
    
    candidate_paths = sample_paths(route_graph, route[a-1], route[c],
                                     max_paths=100, nodes_to_exclude=nodes_to_exclude,
                                     heuristic_fn=haversine_heuristic)
    print(f'Time taken to find candidate paths: {time.time() - start_time}')

    if verbose:
        print(f'Time taken to find bidirectional admissible pivot nodes: {time.time() - start_time}')
    
    return None