import networkx as nx
import numpy as np
import random
import math

def compute_boltzmann_partition_functions(graph, goal_node, max_iterations=1000, tol=1e-6):
    """
    Computes Z(u) = sum_{paths p from u to G} exp(-Cost(p)) for all nodes u.
    Assumes positive edge costs ('cost' attribute).
    Uses value iteration.
    """
    nodes = list(graph.nodes())
    z_values = {node: 0 for node in nodes}
    if goal_node in z_values:
        z_values[goal_node] = 1.0  # exp(-0) for the path of cost 0 at the goal

    for iteration in range(max_iterations):
        z_values_old = z_values.copy()
        delta = 0.0

        for u in nodes:
            if u == goal_node:
                continue

            sum_exp_z = 0.0
            for v in graph.neighbors(u):
                cost = graph.edges[u, v].get('weight', 0.0) # Default cost is 1 if not specified
                if cost < 0:
                    print(f"Warning: Negative edge cost detected ({u}, {v}: {cost}). Convergence not guaranteed.")
                
                # Use log-sum-exp for potentially better numerical stability if needed
                # For simplicity here, using direct sum
                if v in z_values_old: # Ensure neighbor exists in dict keys
                    sum_exp_z += math.exp(-cost) * z_values_old.get(v, 0.0) # Use get for robustness

            z_values[u] = sum_exp_z
            delta = max(delta, abs(z_values[u] - z_values_old.get(u, 0.0))) # Use get for robustness

        # print(f"Iteration {iteration}, Max Delta: {delta}") # Debugging
        if delta < tol and iteration > 50:
            print(f"Converged after {iteration + 1} iterations.")
            break
    else:
        print(f"Warning: Partition function computation did not converge after {max_iterations} iterations.")
        
    return z_values

def sample_boltzmann_path(graph, start_node, goal_node, z_values):
    """
    Samples a path from start_node to goal_node using precomputed Z values.
    Probability P(path) is proportional to exp(-sum_cost(path)).
    """
    if start_node == goal_node:
        return [start_node]
    if z_values.get(start_node, 0.0) == 0.0:
         # If Z(start_node) is 0, it means G is unreachable from S
         # according to the paths considered by Z computation.
         print(f"Warning: Goal node {goal_node} might be unreachable from {start_node} (Z={z_values.get(start_node)}).")
         return None # Or raise an error

    path = [start_node]
    current_node = start_node

    while current_node != goal_node:
        neighbors = list(graph.neighbors(current_node))
        if not neighbors:
            # Should not happen if G is reachable and Z(current_node) > 0
            print(f"Error: Dead end reached at node {current_node} before reaching goal.")
            return None 

        weights = []
        valid_neighbors = []
        total_weight = 0.0

        for neighbor in neighbors:
            cost = graph.edges[current_node, neighbor].get('weight', 0.0)
            z_neighbor = z_values.get(neighbor, 0.0) # Z can be 0 if G is unreachable from neighbor
            
            # Only consider neighbors from which G is potentially reachable (Z > 0)
            # and avoid numerical issues with exp(-large_cost) * small_z
            # Note: If Z is computed correctly, Z(neighbor) should only be 0 if
            # G is truly unreachable from there.
            if z_neighbor > 1e-12: # Add a small tolerance
                weight = math.exp(-cost) * z_neighbor
                if weight > 0:
                    weights.append(weight)
                    valid_neighbors.append(neighbor)
                    total_weight += weight
        
        if total_weight == 0 or not valid_neighbors:
             # This implies that from current_node, all next steps lead to nodes
             # from which G is considered unreachable (Z=0), or numerical underflow.
             # This shouldn't happen if Z(current_node) > 0 and calculated correctly.
             print(f"Error: Cannot proceed from {current_node}. No valid transitions found. Z={z_values.get(current_node)}")
             # print(f"Neighbors: {neighbors}")
             # for n in neighbors: print(f"  Neighbor {n}: cost={graph.edges[current_node, n].get('cost', 1.0)}, Z={z_values.get(n)}")
             return None # Stuck

        # Normalize weights to get probabilities
        probabilities = [w / total_weight for w in weights]

        # Choose the next node based on probabilities
        # Use np.random.choice for potentially better handling of floating point inaccuracies
        next_node = random.choices(valid_neighbors, weights=probabilities, k=1)[0]
        # Or: next_node = np.random.choice(valid_neighbors, p=probabilities) 

        path.append(next_node)
        current_node = next_node

        if len(path) > graph.number_of_nodes() * 2: # Heuristic loop break
             print(f"Warning: Path seems excessively long ({len(path)} steps). Breaking loop. Possibly stuck in a cycle?")
             return None

    return path

# --- Example Usage ---
# Create a sample graph
# G = nx.DiGraph()
# G.add_edge('S', 'A', cost=1.0)
# G.add_edge('S', 'B', cost=4.0)
# G.add_edge('A', 'C', cost=1.0)
# G.add_edge('A', 'D', cost=1.0)
# G.add_edge('B', 'D', cost=1.0)
# G.add_edge('C', 'G', cost=1.0)
# G.add_edge('D', 'G', cost=5.0)
# # Add a cycle and alternative path
# G.add_edge('D', 'A', cost=0.5) # Cycle A->D->A
# G.add_edge('B', 'G', cost=1.0) # Faster path from B

# start = 'S'
# goal = 'G'

# # 1. Compute Z values
# z_vals = compute_boltzmann_partition_functions(G, goal, max_iterations=200)
# print("Computed Z values:", z_vals)

# # Check if start node has non-zero Z value
# if z_vals.get(start, 0) == 0:
#     print(f"Goal {goal} is not reachable from Start {start}")
# else:
#     # 2. Sample paths
#     print("\nSampling Paths:")
#     for i in range(10):
#         path = sample_boltzmann_path(G, start, goal, z_vals)
#         if path:
#             cost = sum(G.edges[u, v]['cost'] for u, v in zip(path[:-1], path[1:]))
#             print(f"Sample {i+1}: Path={path}, Cost={cost:.2f}, Exp(-Cost)={math.exp(-cost):.4f}")
#         else:
#             print(f"Sample {i+1}: Failed to generate path.")
            
# # Expected Behavior:
# # Path S->A->C->G (Cost 3) -> exp(-3) ~ 0.0498
# # Path S->A->D->G (Cost 7) -> exp(-7) ~ 0.0009
# # Path S->B->D->G (Cost 10) -> exp(-10) ~ 0.000045
# # Path S->B->G (Cost 5) -> exp(-5) ~ 0.0067
# # The sampling should favor lower-cost paths like S->A->C->G and S->B->G exponentially more.
# # The cycle A->D->A might cause issues if costs allow negative cycles or Z computation doesn't converge well.