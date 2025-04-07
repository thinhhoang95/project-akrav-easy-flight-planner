import networkx as nx
import numpy as np
from scipy.special import logsumexp
import copy
from collections import defaultdict

# Basic idea:
# Initialization: The code starts by ensuring all edges have distance and preference_score attributes.
# calculate_edge_costs: Simple function to compute cost = distance + alpha * preference_score for all edges.
# soft_bellman_ford: Implements the iterative soft Bellman update using logsumexp for numerical stability. It initializes values to infinity except for the target node (0) and iterates until convergence or max_iterations.
# calculate_transition_probs: Computes p(j -> i) using the calculated costs and V_soft. It normalizes probabilities for transitions leaving each node j. Uses logsumexp again for stability.
# calculate_expected_visitations: This is a crucial part. It iteratively solves the linear system D(i) = I(i=start) + sum_{j} D(j) * p(j -> i) to find the expected number of times each node i is visited (D). It then uses D and p(j -> i) to calculate the expected number of times each edge (j, i) is traversed (E_pi).
# calculate_empirical_visitations: Counts edge traversals in the demonstrations and normalizes by the number of demonstrations to get E_D.
# calculate_gradient: Implements the MaxEnt IRL gradient: alpha * (E_pi - E_D) plus the L2 regularization term alpha_r * rho.
# update_preferences: Performs a standard gradient descent step: rho_new = rho_old - lr * grad.
# optimize_preferences: The main loop orchestrates the process: calculates costs, runs SBF, computes probabilities, computes expected visitations (E_pi), computes the gradient using E_pi and the pre-computed E_D, checks for convergence, and updates the preference scores in the graph.

# Tollerances and limits for numerical stability and convergence
DEFAULT_TOLERANCE = 1e-5
DEFAULT_MAX_ITER = 1000
DEFAULT_GAMMA = 1.0 # Discount factor (implicitly 1 in undiscounted soft value iteration)
LOG_PROB_THRESHOLD = -20 # Threshold to consider a transition probability effectively zero



def initialize_preference_scores(graph: nx.DiGraph, initial_value: float = 0.0):
    """Initializes 'preference_score' attribute for all edges if not present."""
    for u, v, data in graph.edges(data=True):
        if 'preference_score' not in data:
            data['preference_score'] = initial_value
        if 'distance' not in data:
             # Ensure distance exists, default to 1.0 if missing
             print(f"Warning: Edge ({u}, {v}) missing 'distance'. Defaulting to 1.0.")
             data['distance'] = 1.0

def calculate_edge_costs(graph: nx.DiGraph, alpha: float) -> dict:
    """
    Calculates the cost for each edge.
    cost = distance + alpha * preference_score
    """
    costs = {}
    for u, v, data in graph.edges(data=True):
        distance = data.get('distance', 1.0) # Default distance if missing
        preference_score = data.get('preference_score', 0.0) # Default preference if missing
        costs[(u, v)] = distance - alpha * preference_score
    return costs

def soft_bellman_ford(graph: nx.DiGraph, target_node, costs: dict,
                      max_iterations: int = DEFAULT_MAX_ITER,
                      tolerance: float = DEFAULT_TOLERANCE) -> dict:
    """
    Computes the soft value function V_soft = -log Z using soft Bellman-Ford.
    V_soft(i) = -log [ sum_{k in successors(i)} exp(-cost(i, k) - V_soft(k)) ]
    Handles graphs with cycles. Converges if costs guarantee paths to target are finite.
    """
    nodes = list(graph.nodes())
    V_soft = {node: np.inf for node in nodes}
    
    if target_node not in V_soft:
        raise ValueError(f"Target node {target_node} not in graph.")
        
    V_soft[target_node] = 0.0

    for iteration in range(max_iterations):
        V_soft_old = V_soft.copy()
        max_delta = 0.0

        # Iterate in reverse topological order if DAG, otherwise iterate until convergence
        # For general graphs, simple iteration works but might be slower
        for node in nodes:
            if node == target_node:
                continue

            successors = list(graph.successors(node))
            if not successors:
                # Node has no successors, its value remains inf unless it's the target
                continue

            # Calculate logsumexp term: logsumexp([-cost(node, k) - V_soft_old[k]])
            terms = []
            for k in successors:
                edge = (node, k)
                if edge in costs and np.isfinite(V_soft_old[k]):
                   terms.append(-costs[edge] - V_soft_old[k])

            if not terms:
                 # No path to target through successors with finite cost/value
                 new_value = np.inf
            else:
                 new_value = -logsumexp(terms)


            # Update value and track change
            delta = abs(new_value - V_soft[node])
            max_delta = max(max_delta, delta)
            V_soft[node] = new_value

        # Check for convergence
        if max_delta < tolerance:
            # print(f"Soft Bellman-Ford converged in {iteration + 1} iterations.")
            return V_soft

    print(f"Warning: Soft Bellman-Ford did not converge within {max_iterations} iterations. Max delta: {max_delta}")
    return V_soft


def calculate_transition_probs(graph: nx.DiGraph, costs: dict, V_soft: dict) -> dict:
    """
    Calculates the Boltzmann transition probabilities p(j -> i).
    p(j -> i) = exp(-cost(j, i) - V_soft(i) + V_soft(j))
    """
    probs = {}
    nodes = list(graph.nodes())

    for u in nodes:
        if not np.isfinite(V_soft[u]): # Cannot transition from a state with infinite value
             continue
             
        successors = list(graph.successors(u))
        if not successors:
            continue # No transitions out

        log_probs_terms = []
        valid_successors = []

        for v in successors:
             edge = (u, v)
             if edge in costs and np.isfinite(V_soft[v]):
                 log_prob_unnormalized = -costs[edge] - V_soft[v] + V_soft[u]
                 log_probs_terms.append(log_prob_unnormalized)
                 valid_successors.append(v)
             # else: edge doesn't exist or leads to inf value state, prob is 0


        if not log_probs_terms:
             continue # No valid transitions with finite cost/value

        # Normalize probabilities using logsumexp trick for stability
        log_Z_u = logsumexp(log_probs_terms) # This should be close to 0 if V_soft is correct, but recalculate for safety

        for i, v in enumerate(valid_successors):
             log_prob = log_probs_terms[i] - log_Z_u
             # Store probability if it's numerically significant
             if log_prob > LOG_PROB_THRESHOLD:
                 probs[(u, v)] = np.exp(log_prob)
             # else: probability is effectively zero, omit from sparse dict


    # Ensure probabilities sum to 1 (approximately) for each node with finite V_soft
    # for u in nodes:
    #    if np.isfinite(V_soft[u]):
    #        total_prob = sum(probs.get((u, v), 0) for v in graph.successors(u))
    #        if abs(total_prob - 1.0) > 1e-3 and total_prob > 1e-9 : # Allow some tolerance, ignore nodes with near zero total prob çıkışı
    #            print(f"Warning: Probabilities from node {u} sum to {total_prob:.4f}")

    return probs


def calculate_expected_visitations(graph: nx.DiGraph, start_node, target_node,
                                   transition_probs: dict,
                                   max_iterations: int = DEFAULT_MAX_ITER,
                                   tolerance: float = DEFAULT_TOLERANCE) -> (dict, dict):
    """
    Calculates expected state visitation frequencies (D) and expected edge frequencies (E_pi).
    D(i) = I(i=start) + sum_{j in predecessors(i)} D(j) * p(j -> i)
    Solved iteratively.
    E_pi(j, i) = D(j) * p(j -> i)
    """
    nodes = list(graph.nodes())
    D = {node: 0.0 for node in nodes}
    if start_node not in D:
        raise ValueError(f"Start node {start_node} not in graph.")
    D[start_node] = 1.0 # Initial state probability

    # Precompute predecessors for efficiency
    predecessors = {node: list(graph.predecessors(node)) for node in nodes}

    for iteration in range(max_iterations):
        D_old = D.copy()
        max_delta = 0.0

        for i in nodes:
            # Start node contribution
            new_D_i = 1.0 if i == start_node else 0.0

            # Sum over predecessors
            in_flow = 0.0
            for j in predecessors[i]:
                 prob_ji = transition_probs.get((j, i), 0.0) # Get p(j -> i), default 0
                 if prob_ji > 0: # Avoid unnecessary multiplication
                    in_flow += D_old[j] * prob_ji

            new_D_i += in_flow

            # Update D and track change
            delta = abs(new_D_i - D[i])
            max_delta = max(max_delta, delta)
            D[i] = new_D_i

        # Check for convergence
        if max_delta < tolerance:
            # print(f"Expected state visitations (D) converged in {iteration + 1} iterations.")
            break
    else: # No break
        print(f"Warning: Expected state visitations (D) did not converge within {max_iterations} iterations. Max delta: {max_delta}")

    # Calculate expected edge frequencies E_pi
    E_pi = defaultdict(float)
    for (u, v), prob in transition_probs.items():
        if D[u] > 0 and prob > 0: # Only consider reachable states and valid transitions
            E_pi[(u, v)] = D[u] * prob

    return D, dict(E_pi)


def calculate_empirical_visitations(graph: nx.DiGraph, demonstrations: list) -> dict:
    """
    Calculates empirical edge traversal frequencies (E_D) from demonstrations.
    """
    if not demonstrations:
        return defaultdict(float)

    edge_counts = defaultdict(int)
    num_demos = len(demonstrations)

    for path in demonstrations:
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if graph.has_edge(u, v):
                edge_counts[(u, v)] += 1
            else:
                # This might indicate an issue with the demonstrations or the graph
                print(f"Warning: Edge ({u}, {v}) from demonstration path not found in graph.")

    E_D = {edge: count / num_demos for edge, count in edge_counts.items()}
    return E_D


def calculate_gradient(graph: nx.DiGraph, alpha: float, alpha_r: float,
                       E_pi: dict, E_D: dict) -> dict:
    """
    Calculates the gradient of the NLL w.r.t. preference scores (rho).
    grad(rho_uv) = alpha * (E_pi[u,v] - E_D[u,v]) + alpha_r * rho_uv
    """
    grad_rho = {}
    all_edges = set(E_pi.keys()) | set(E_D.keys()) | set(graph.edges())

    for u, v in all_edges:
         # Check if edge exists in graph before accessing preference score
         if graph.has_edge(u,v):
             e_pi_uv = E_pi.get((u, v), 0.0)
             e_d_uv = E_D.get((u, v), 0.0)
             
             # Retrieve current preference score for regularization term
             current_rho_uv = graph.edges[u, v].get('preference_score', 0.0)
             
             # Gradient calculation
             gradient = alpha * (e_pi_uv - e_d_uv) + alpha_r * current_rho_uv
             grad_rho[(u, v)] = gradient
         # else: Edge might be in demonstrations but not graph, or only in E_pi theoretically.
         # If edge isn't in the graph, we can't calculate/update its preference score.
         # Consider logging a warning if an edge appears in E_D but not the graph.

    return grad_rho


def update_preferences(graph: nx.DiGraph, grad_rho: dict, lrate: float, infer_mode: bool = False):
    """
    Updates the 'preference_score' attribute of edges using gradient descent.
    rho_new = rho_old - lrate * grad(rho)
    """
    for (u, v), grad in grad_rho.items():
        if graph.has_edge(u, v): # Ensure edge exists
            current_rho = graph.edges[u, v].get('preference_score', 0.0)
            if not infer_mode: # Only update preferences if not in inference mode (i.e. training)
                graph.edges[u, v]['preference_score'] = current_rho - lrate * grad
            else:
                print(f"WARNING: Inference mode. Not updating preferences.")
        else:
             print(f"Warning: Trying to update preference for non-existent edge ({u}, {v}). Gradient ignored.")


def calculate_loss(graph: nx.DiGraph, V_soft: dict, start_node, alpha_r: float, 
                  demonstrations: list, costs: dict):
    """
    Calculates the MaxEnt IRL loss function:
    L = -log(P(demonstrations)) + alpha_r * ||rho||^2 / 2
    """
    # 1. Calculate negative log-likelihood of demonstrations
    nll = 0.0
    for path in demonstrations:
        path_cost = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if graph.has_edge(u, v) and (u, v) in costs:
                path_cost += costs[(u, v)]
            else:
                print(f"Warning: Edge ({u}, {v}) missing in graph or costs.")
                path_cost += float('inf')
        
        # Add path probability: -log P(path) = path_cost + V_soft[start] - V_soft[goal]
        if np.isfinite(path_cost) and np.isfinite(V_soft[start_node]):
            # nll += path_cost + V_soft[start_node] - V_soft[path[-1]]
            nll += path_cost + V_soft[path[-1]] - V_soft[start_node]
        else:
            nll += float('inf')
    
    # 2. Calculate L2 regularization term
    reg_term = 0.0
    for _, _, data in graph.edges(data=True):
        pref = data.get('preference_score', 0.0)
        reg_term += pref * pref
    reg_term = alpha_r * reg_term / 2.0
    
    # 3. Return total loss
    return nll / len(demonstrations) + reg_term


# --- Main Optimization Function ---

def optimize_preferences(graph: nx.DiGraph,
                         demonstrations: list,
                         start_node,
                         target_node,
                         alpha: float,
                         alpha_r: float,
                         learning_rate: float,
                         num_epochs: int,
                         initial_preference: float = 0.0,
                         convergence_threshold: float = 1e-4,
                         early_stopping_patience: int = 5,
                         infer_mode: bool = False):
    """
    Performs gradient descent to optimize edge preference scores.
    
    Parameters:
        early_stopping_patience: Number of epochs to wait after loss increases
                                before stopping training
    """
    if not graph.is_directed():
        print("Warning: Graph is undirected. Converting to a directed graph.")
        graph = graph.to_directed()

    # Initialize preference scores if they don't exist
    initialize_preference_scores(graph, initial_value=initial_preference)

    # Calculate empirical frequencies once (they don't change)
    E_D = calculate_empirical_visitations(graph, demonstrations)
    print(f"Calculated Empirical Visitations (E_D). Example: {dict(list(E_D.items())[:5])}")

    # Early stopping variables
    best_loss = float('inf')
    best_epoch = 0
    best_graph = None
    patience_counter = 0

    history = {'epoch': [], 'total_grad_norm': [], 'max_grad': [], 'loss': []}
    
    from tqdm import tqdm
    progress_bar = tqdm(range(num_epochs), total=num_epochs)
    for epoch in progress_bar:
        # print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

        # 1. Calculate edge costs based on current preferences
        costs = calculate_edge_costs(graph, alpha)
        # print(f"Calculated costs. Example: {dict(list(costs.items())[:5])}")

        # 2. Compute soft value function
        V_soft = soft_bellman_ford(graph, target_node, costs)
        # Check for issues
        if not np.isfinite(V_soft[start_node]):
             print(f"Error: Start node {start_node} has infinite soft value. Check graph connectivity or costs.")
             print(f"V_soft example: {dict(list(V_soft.items())[:5])}")
             # Optional: Add check for V_soft[target_node] != 0? Should be 0 by definition.
             return graph, history # Stop optimization

        # 3. Calculate transition probabilities
        transition_probs = calculate_transition_probs(graph, costs, V_soft)
        # print(f"Calculated transition probs. Example: {dict(list(transition_probs.items())[:5])}")

        # 4. Calculate expected state and edge visitations
        D, E_pi = calculate_expected_visitations(graph, start_node, target_node, transition_probs)
        # print(f"Calculated Expected Visitations (E_pi). Example: {dict(list(E_pi.items())[:5])}")
        # print(f"Calculated State Visitations (D). Example: {dict(list(D.items())[:5])}")

        # 5. Calculate gradient
        grad_rho = calculate_gradient(graph, alpha, alpha_r, E_pi, E_D)
        # print(f"Calculated gradient. Example: {dict(list(grad_rho.items())[:5])}")

        # --- Gradient Check & Convergence ---
        if not grad_rho:
             print("Gradient is empty. Check calculations or graph structure.")
             break
             
        total_grad_norm = np.linalg.norm(list(grad_rho.values()))
        max_grad = max(abs(g) for g in grad_rho.values()) if grad_rho else 0.0
        
        # 5.5 Calculate and record loss
        loss = calculate_loss(graph, V_soft, start_node, alpha_r, demonstrations, costs)
        
        history['epoch'].append(epoch + 1)
        history['total_grad_norm'].append(total_grad_norm)
        history['max_grad'].append(max_grad)
        history['loss'].append(loss)
        
        progress_bar.set_postfix({
            'Epoch': epoch + 1, 
            'Loss': f'{loss:.4e}', 
            'Grad Norm': f'{total_grad_norm:.4e}', 
            'Max Grad': f'{max_grad:.4e}'
        })

        # if epoch % 100 == 0:
            # print(f"Epoch {epoch + 1}: Grad Norm = {total_grad_norm:.4e}, Max Grad = {max_grad:.4e}, Loss = {loss:.4e}")

        # Check for early stopping
        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch
            best_graph = copy.deepcopy(graph)  # Save the best model
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch + 1}. Loss hasn't improved for {early_stopping_patience} epochs.")
                print(f"Best loss of {best_loss:.4e} was at epoch {best_epoch + 1}")
                graph = best_graph  # Restore the best model
                break

        # Check for convergence based on gradient magnitude
        if max_grad < convergence_threshold:
            print(f"\nConvergence reached at epoch {epoch + 1}. Max gradient below threshold.")
            break

        # 6. Update preference scores
        if not infer_mode: # Only update preferences if not in inference mode (i.e. training)
            update_preferences(graph, grad_rho, learning_rate)
        else:
            update_preferences(graph, grad_rho, learning_rate, infer_mode=True)
            print(f"WARNING: Inference mode. Only one epoch is run.")
            break 

        # Example of updated preference:
        # edge_example = list(grad_rho.keys())[0] if grad_rho else None
        # if edge_example:
        #     print(f"Updated preference for {edge_example}: {graph.edges[edge_example]['preference_score']:.4f}")

    else:  # No break from loop
        print(f"\nOptimization finished after {num_epochs} epochs.")
        if best_graph is not None and best_loss < loss:
            print(f"Restoring best model from epoch {best_epoch + 1} with loss {best_loss:.4e}")
            graph = best_graph  # Restore the best model if it's better than the final one

#    return graph, history
    return graph, history

def optimize_preferences_with_restarts(graph: nx.DiGraph,
                                       demonstrations: list,
                                       start_node,
                                       target_node,
                                       alpha: float,
                                       alpha_r: float,
                                       learning_rate: float,
                                       num_epochs: int,
                                       num_restarts: int = 5,
                                       initial_preference: float = 0.0,
                                       init_std: float = 0.1,
                                       convergence_threshold: float = 1e-4,
                                       early_stopping_patience: int = 5,
                                       infer_mode: bool = False):
    """
    Performs gradient descent optimization with multiple random restarts.
    
    For each restart, the graph is reinitialized with random preference scores
    drawn from a normal distribution centered at 'initial_preference' with standard deviation 'init_std'.
    The best performing graph (lowest final loss) across all restarts is returned.
    
    Returns:
        best_graph: The graph with optimized preference scores corresponding to the lowest loss.
        best_history: The optimization history for the best run.
    """
    best_loss = float('inf')
    best_graph = None
    best_history = None

    for restart in range(num_restarts):
        print(f"\nRestart {restart + 1}/{num_restarts}")
        # Deep copy the original graph to avoid side effects
        graph_copy = copy.deepcopy(graph)
        
        # Randomize initial preferences for each edge
        for u, v, data in graph_copy.edges(data=True):
            data['preference_score'] = initial_preference + init_std * np.random.randn()
        
        # Run the standard optimization on the graph copy
        optimized_graph, history = optimize_preferences(
            graph=graph_copy,
            demonstrations=demonstrations,
            start_node=start_node,
            target_node=target_node,
            alpha=alpha,
            alpha_r=alpha_r,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            initial_preference=initial_preference,
            convergence_threshold=convergence_threshold,
            early_stopping_patience=early_stopping_patience,
            infer_mode=infer_mode
        )

        # Get the final loss from the history
        final_loss = history['loss'][-1] if history['loss'] else float('inf')
        print(f"Restart {restart + 1} final loss: {final_loss:.4e}")

        if final_loss < best_loss:
            best_loss = final_loss
            best_graph = optimized_graph
            best_history = history

    print(f"\nBest loss after {num_restarts} restarts: {best_loss:.4e}")
    return best_graph, best_history

# --- Example Usage ---
if __name__ == "__main__":
    # 1. Create a sample graph
    G = nx.DiGraph()
    # Add nodes (waypoints)
    G.add_nodes_from(['A', 'B', 'C', 'D', 'E', 'T']) # T is target
    # Add edges with distance and initial preference_score
    G.add_edge('A', 'B', distance=1.0)
    G.add_edge('A', 'C', distance=2.0)
    G.add_edge('B', 'D', distance=1.5)
    G.add_edge('B', 'T', distance=5.0)
    G.add_edge('C', 'D', distance=1.0)
    G.add_edge('C', 'E', distance=2.5)
    G.add_edge('D', 'T', distance=2.0)
    G.add_edge('E', 'T', distance=1.0)
    
    # Add a less desirable edge for testing
    G.add_edge('A', 'D', distance=4.0) 
    G.add_edge('D', 'B', distance=1.5) # Cycle


    # 2. Define demonstrated routes
    # Assume the "expert" prefers paths going through C and E
    demonstrations = [
        ['A', 'C', 'E', 'T'],
        ['A', 'C', 'E', 'T'],
        ['A', 'C', 'D', 'T'], # Include some variation
        ['A', 'C', 'E', 'T'],
    ]

    # 3. Define parameters
    start_node = 'A'
    target_node = 'T'
    alpha = 1.0       # Weight for preference score in cost
    alpha_r = 0.01    # Regularization strength (L2)
    learning_rate = 0.1 # Step size for gradient descent
    num_epochs = 100    # Number of optimization iterations
    initial_preference = 0.0 # Initial guess for preference scores
    convergence_threshold = 1e-4 # Stop if max gradient component is small


    # 4. Run the optimization with multiple restarts
    num_restarts = 5  # Adjust the number of restarts as desired
    optimized_graph, history = optimize_preferences_with_restarts(
        graph=G,
        demonstrations=demonstrations,
        start_node=start_node,
        target_node=target_node,
        alpha=alpha,
        alpha_r=alpha_r,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        num_restarts=num_restarts,
        initial_preference=initial_preference,
        init_std=0.1,  # Standard deviation for random initialization
        convergence_threshold=convergence_threshold,
        early_stopping_patience=1
    )

    # 5. Inspect the results
    print("\n--- Optimization Results ---")
    print("Final Preference Scores:")
    for u, v, data in optimized_graph.edges(data=True):
        print(f"Edge ({u}, {v}): Distance={data['distance']:.1f}, Preference={data['preference_score']:.4f}")
        
    # Plotting convergence (optional)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(history['epoch'], history['total_grad_norm'])
        plt.xlabel("Epoch")
        plt.ylabel("Total Gradient Norm (L2)")
        plt.title("Gradient Norm Convergence")
        plt.yscale('log')
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(history['epoch'], history['max_grad'])
        plt.xlabel("Epoch")
        plt.ylabel("Max Absolute Gradient")
        plt.title("Max Gradient Convergence")
        plt.yscale('log')
        plt.axhline(convergence_threshold, color='r', linestyle='--', label='Threshold')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        plt.plot(history['epoch'], history['loss'])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Function")
        plt.yscale('log')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("\nMatplotlib not found. Cannot plot convergence history.")

    # You can now use the optimized_graph with its adjusted preference scores
    # for sampling routes using Boltzmann probabilities based on the learned costs.