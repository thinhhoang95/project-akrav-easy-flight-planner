"""
Lateral Planner Module
======================

This module provides path planning algorithms for lateral movement in networks.
It implements Metropolis-Hastings path sampling for generating diverse paths
between nodes in a directed graph, with various operations for path manipulation.

Author: Thinh Hoang
Date: 2025-03-03
Version: 0.1
"""


import networkx as nx
import numpy as np
import random
from typing import List, Tuple, Dict, Set, Optional

def mh_path_sampling(
    G: nx.DiGraph,
    origin: int,
    destination: int,
    mu: float = 1.0,
    max_iterations: int = 10000,
    thinning: int = 2500,
    burn_in: int = 2500,
    w: float = 0.5  # Probability of SPLICE operation
) -> List[List[int]]:
    """
    Sample paths from a directed graph using Metropolis-Hastings algorithm.
    
    Args:
        G: NetworkX directed graph with 'cost' attribute on edges
        origin: Source node
        destination: Target node
        mu: Scale parameter for exp(-mu * path_cost) weighting
        max_iterations: Maximum number of iterations
        thinning: Number of iterations between each sample
        burn_in: Number of iterations to discard at the beginning
        w: Probability of SPLICE operation vs SHUFFLE
    
    Returns:
        List of sampled paths, where each path is a list of node IDs
    """
    # Helper functions for shortest paths
    def shortest_path(from_node, to_node, excluded_nodes=None):
        """Find shortest path avoiding excluded nodes."""
        if excluded_nodes is None:
            excluded_nodes = set()
        
        # Create a copy of the graph without excluded nodes
        H = G.copy()
        for node in excluded_nodes:
            if node in H:
                H.remove_node(node)
        
        try:
            path = nx.shortest_path(H, from_node, to_node, weight='cost')
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
    
    def path_cost(path):
        """Calculate the total cost of a path."""
        cost = 0
        for i in range(len(path) - 1):
            cost += G[path[i]][path[i+1]].get('cost', 1)
        return cost

    def is_cycle_free(path):
        """Check if a path is cycle-free."""
        return len(path) == len(set(path))
    
    def target_weight(path, a, b, c):
        """Calculate the target weight for a state."""
        denominator = len(path) * (len(path) - 1) * (len(path) - 2) / 6
        return np.exp(-mu * path_cost(path)) / denominator
    
    def is_spliceable(path, a, b, c):
        """Check if a state is spliceable."""
        # Get the segment of the path between a and c
        segment = path[a-1:c]
        
        # Get the shortest path from path[a-1] to path[b-1]
        sp1 = shortest_path(path[a-1], path[b-1])
        if sp1 is None:
            return False
        
        # Get the shortest path from path[b-1] to path[c-1]
        sp2 = shortest_path(path[b-1], path[c-1])
        if sp2 is None:
            return False
        
        # Check if the concatenation equals the segment
        # We need to remove duplicates at the connection point
        combined = sp1 + sp2[1:]
        return combined == segment
    
    # Initialize with the shortest path from origin to destination
    current_path = nx.shortest_path(G, origin, destination, weight='cost')
    
    # Initialize a, b, c indices
    if len(current_path) < 3:
        raise ValueError("Shortest path has fewer than 3 nodes, cannot initialize indices.")
    
    current_a = 1
    current_b = 2
    current_c = 3
    
    # Lists to store sampling results
    sampled_paths = []
    
    # Main MCMC loop
    for iteration in range(max_iterations):
        # Determine whether to use SPLICE or SHUFFLE
        use_splice = (random.random() < w) and is_spliceable(current_path, current_a, current_b, current_c)
        
        if use_splice:
            # SPLICE operation
            new_path, new_a, new_b, new_c = splice_operation(
                G, current_path, current_a, current_b, current_c, mu
            )
        else:
            # SHUFFLE operation
            new_path, new_a, new_b, new_c = shuffle_operation(
                current_path, current_a, current_b, current_c
            )
        
        # Calculate acceptance probability
        current_weight = target_weight(current_path, current_a, current_b, current_c)
        new_weight = target_weight(new_path, new_a, new_b, new_c)
        
        # Calculate proposal probabilities (simplified for demonstration)
        # In a full implementation, these would be calculated based on the SPLICE and SHUFFLE
        # operations as defined in the paper
        
        # For now, we'll use a simple Metropolis algorithm (symmetric proposals)
        acceptance_prob = min(1, new_weight / current_weight)
        
        # Accept or reject
        if random.random() < acceptance_prob:
            current_path = new_path
            current_a = new_a
            current_b = new_b
            current_c = new_c
        
        # Collect samples after burn-in and according to thinning
        if iteration >= burn_in and (iteration - burn_in) % thinning == 0:
            sampled_paths.append(current_path.copy())
    
    return sampled_paths


def splice_operation(G, current_path, a, b, c, mu):
    """
    Implement the SPLICE operation as described in the paper.
    
    Args:
        G: The graph
        current_path: Current path
        a, b, c: Current indices
        mu: Scale parameter for node selection probability
    
    Returns:
        (new_path, new_a, new_b, new_c)
    """
    # Define N1 and N2 sets
    N1 = set(G.nodes()) - set(current_path[:a-1]) - set(current_path[c-1:])
    N2 = set(G.nodes()) - set(current_path[:a]) - set(current_path[c:])
    
    # Find potential insertion nodes V(i)
    V = set()
    for v in G.nodes():
        if v in current_path:
            continue  # Skip nodes already in the path
            
        # Check if there's a path from current_path[a-1] to v avoiding N\N1
        path1 = find_path_avoiding(G, current_path[a-1], v, set(G.nodes()) - N1)
        if not path1:
            continue
            
        # Check if there's a path from v to current_path[c-1] avoiding N\N2
        path2 = find_path_avoiding(G, v, current_path[c-1], set(G.nodes()) - N2)
        if not path2:
            continue
            
        V.add(v)
    
    if not V:
        # No valid insertion nodes, return the original state
        return current_path, a, b, c
    
    # Select an insertion node based on probabilities
    weights = {}
    for v in V:
        path1 = find_path_avoiding(G, current_path[a-1], v, set(G.nodes()) - N1)
        path2 = find_path_avoiding(G, v, current_path[c-1], set(G.nodes()) - N2)
        
        weight = np.exp(-mu * (path_cost(G, path1) + path_cost(G, path2)))
        weights[v] = weight
    
    total_weight = sum(weights.values())
    probs = {v: w/total_weight for v, w in weights.items()}
    
    v = random.choices(list(probs.keys()), weights=list(probs.values()), k=1)[0]
    
    # Construct the new path
    path1 = find_path_avoiding(G, current_path[a-1], v, set(G.nodes()) - N1)
    path2 = find_path_avoiding(G, v, current_path[c-1], set(G.nodes()) - N2)
    
    new_path = current_path[:a-1] + path1[:-1] + path2 + current_path[c:]
    
    # Check if the new path has cycles
    if len(new_path) != len(set(new_path)):
        # Cycle detected, return original state
        return current_path, a, b, c
    
    # Adjust indices for the new path
    new_a = a
    new_b = len(current_path[:a-1]) + len(path1)
    new_c = new_b + len(path2) - 1
    
    return new_path, new_a, new_b, new_c


def shuffle_operation(current_path, a, b, c):
    """
    Implement the SHUFFLE operation.
    
    Args:
        current_path: Current path
        a, b, c: Current indices
    
    Returns:
        (new_path, new_a, new_b, new_c)
    """
    path_len = len(current_path)
    
    # Randomly select new a, b, c indices
    valid_indices = False
    while not valid_indices:
        indices = sorted(random.sample(range(1, path_len + 1), 3))
        new_a, new_b, new_c = indices
        if 1 <= new_a < new_b < new_c <= path_len:
            valid_indices = True
    
    return current_path, new_a, new_b, new_c


def find_path_avoiding(G, start, end, avoid_nodes):
    """Find a path from start to end avoiding the nodes in avoid_nodes."""
    # Create a subgraph without the nodes to avoid
    H = G.copy()
    for node in avoid_nodes:
        if node in H:
            H.remove_node(node)
    
    try:
        return nx.shortest_path(H, start, end, weight='cost')
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def path_cost(G, path):
    """Calculate the cost of a path in a graph."""
    cost = 0
    for i in range(len(path) - 1):
        cost += G[path[i]][path[i+1]].get('cost', 1)
    return cost