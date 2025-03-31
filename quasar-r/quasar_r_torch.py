import networkx as nx
import numpy as np
from scipy.special import logsumexp as scipy_logsumexp # For comparison/verification if needed
import copy
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
import torch_scatter
from tqdm import tqdm
import time
import warnings

# --- PyTorch Configuration ---
# Automatically select GPU if available, otherwise CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print("Using CPU")

# --- Constants ---
DEFAULT_TOLERANCE = 1e-5
DEFAULT_MAX_ITER = 1000 # Max iterations for SBF and Visitations
LOG_PROB_THRESHOLD_Tensor = torch.tensor(-20.0, device=device) # Threshold for probabilities
INF_REPLACEMENT = 1e18 # Large number to represent infinity safely in tensors

# --- Helper Functions ---

def networkx_to_pyg_data(graph: nx.DiGraph, initial_preference: float = 0.0):
    """Converts a NetworkX graph to a PyTorch Geometric Data object."""
    nodes = list(graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    idx_to_node = {i: node for node, i in node_to_idx.items()}
    num_nodes = len(nodes)

    edge_list = list(graph.edges(data=True))
    num_edges = len(edge_list)

    # Ensure all edges have distance and preference_score
    distances = []
    initial_prefs = []
    edge_indices = [[], []] # Source nodes, Target nodes

    for i, (u, v, data) in enumerate(edge_list):
        if 'distance' not in data:
            warnings.warn(f"Edge ({u}, {v}) missing 'distance'. Defaulting to 1.0.")
            data['distance'] = 1.0
        if 'preference_score' not in data:
             data['preference_score'] = initial_preference # Use provided initial value

        distances.append(data['distance'])
        initial_prefs.append(data['preference_score'])
        edge_indices[0].append(node_to_idx[u])
        edge_indices[1].append(node_to_idx[v])

    # Create tensors
    edge_index = torch.tensor(edge_indices, dtype=torch.long, device=device)
    distance = torch.tensor(distances, dtype=torch.float32, device=device)
    # Learnable parameter: preference_score
    preference_score = nn.Parameter(torch.tensor(initial_prefs, dtype=torch.float32, device=device))

    # Create PyG Data object
    data = Data(
        edge_index=edge_index,
        distance=distance,
        preference_score=preference_score, # This is the learnable tensor
        num_nodes=num_nodes
    )

    # Add mappings for convenience
    data.node_to_idx = node_to_idx
    data.idx_to_node = idx_to_node
    data.edge_list_map = edge_list # Store original edge tuples for mapping E_D/results back

    print(f"Converted graph: {num_nodes} nodes, {num_edges} edges.")
    return data

def preprocess_demonstrations(demonstrations: list, data: Data):
    """
    Calculates empirical edge counts (E_D) and converts demo paths to edge indices.
    """
    node_to_idx = data.node_to_idx
    edge_index_np = data.edge_index.cpu().numpy() # For faster lookup
    num_edges = data.edge_index.shape[1]

    # Create a fast lookup from (u_idx, v_idx) tuple to edge index in the PyG data
    edge_tuple_to_edge_idx = {
        (edge_index_np[0, i], edge_index_np[1, i]): i
        for i in range(num_edges)
    }

    E_D_counts = torch.zeros(num_edges, dtype=torch.float32, device=device)
    demo_edge_indices = [] # List of lists containing edge indices for each demo path

    num_valid_demos = 0
    for path in demonstrations:
        path_edge_idxs = []
        valid_path = True
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            u_idx, v_idx = node_to_idx.get(u), node_to_idx.get(v)

            if u_idx is None or v_idx is None:
                warnings.warn(f"Node {u} or {v} in demo path not in graph. Skipping edge.")
                valid_path = False
                continue # Skip this edge

            edge_idx = edge_tuple_to_edge_idx.get((u_idx, v_idx))

            if edge_idx is not None:
                E_D_counts[edge_idx] += 1
                path_edge_idxs.append(edge_idx)
            else:
                # This might indicate an issue with the demonstrations or the graph
                warnings.warn(f"Edge ({u} [{u_idx}], {v} [{v_idx}]) from demonstration path not found in graph structure. Skipping edge in demo.")
                valid_path = False # Mark path as potentially problematic for loss calc

        if path_edge_idxs and valid_path: # Only add non-empty, fully valid paths for loss calc
             demo_edge_indices.append(torch.tensor(path_edge_idxs, dtype=torch.long, device=device))
             num_valid_demos += 1
        elif not path_edge_idxs and len(path) > 1:
             warnings.warn(f"Demonstration path {path} resulted in zero valid edges.")


    if num_valid_demos == 0 and len(demonstrations) > 0:
         raise ValueError("No valid demonstration paths found or processed. Check warnings.")
    elif num_valid_demos < len(demonstrations):
         warnings.warn(f"Processed {num_valid_demos} valid demonstrations out of {len(demonstrations)} provided.")


    # Normalize E_D counts to get frequencies
    E_D_tensor = E_D_counts / num_valid_demos if num_valid_demos > 0 else torch.zeros_like(E_D_counts)

    print(f"Preprocessed demonstrations. Calculated E_D for {num_edges} edges. Found {num_valid_demos} valid demo paths.")
    return E_D_tensor, demo_edge_indices

# --- Core Algorithm Components (PyTorch) ---

def calculate_edge_costs_pt(data: Data, alpha: float) -> torch.Tensor:
    """ Calculates edge costs using PyTorch tensors. """
    # Ensure preference_score is treated correctly
    prefs = data.preference_score
    return data.distance - alpha * prefs

def soft_bellman_ford_pt(data: Data, target_idx: int, costs: torch.Tensor,
                         max_iterations: int = DEFAULT_MAX_ITER,
                         tolerance: float = DEFAULT_TOLERANCE) -> torch.Tensor:
    """ Computes soft value function V_soft using PyTorch and torch_scatter. """
    num_nodes = data.num_nodes
    edge_index = data.edge_index
    source_nodes, target_nodes = edge_index[0], edge_index[1]

    # Initialize V_soft with large values (acting as infinity)
    V_soft = torch.full((num_nodes,), INF_REPLACEMENT, dtype=torch.float32, device=device)
    V_soft[target_idx] = 0.0

    converged = False
    for iteration in range(max_iterations):
        V_soft_old = V_soft.clone()

        # Gather V_soft values for target nodes of edges (k)
        v_k = V_soft_old[target_nodes]

        # Calculate terms: -cost(i, k) - V_soft_old(k)
        # Mask out edges leading to 'infinite' value nodes
        finite_v_k_mask = v_k < INF_REPLACEMENT / 2 # Check if target V is finite
        
        # Only compute terms for edges where target node has finite value
        # Important: Ensure costs is indexed correctly if mask is applied *before* scatter
        # Approach: Compute all terms, use mask in scatter or after
        terms = -costs - v_k 
        
        # Perform scatter_logsumexp for valid terms only
        # Aggregate terms based on the source node (i)
        # Use fill_value=INF_REPLACEMENT for nodes with no valid outgoing paths? scatter doesn't have it directly
        # -> Compute for all, then mask nodes that didn't receive updates?
        
        logsumexp_vals = torch_scatter.scatter_logsumexp(terms[finite_v_k_mask], 
                                                         source_nodes[finite_v_k_mask], 
                                                         dim=0, 
                                                         dim_size=num_nodes)
                                                        # fill_value=float('-inf')) # Not supported by scatter_lsp

        # Nodes that had no finite path successors will have -inf from logsumexp
        # Replace -inf with our INF_REPLACEMENT value
        # Also, nodes with no outgoing edges *at all* won't be updated by scatter -> keep old value (inf)
        new_V_soft = -logsumexp_vals
        new_V_soft = torch.where(torch.isinf(new_V_soft), INF_REPLACEMENT, new_V_soft) # Handle -inf
        
        # Keep old inf value for nodes that were not updated by scatter (no outgoing edges or no finite successors)
        # Scatter operates only on indices present in source_nodes[finite_v_k_mask]
        updated_nodes_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        updated_nodes_mask[source_nodes[finite_v_k_mask].unique()] = True

        # Combine: Use new value if updated, otherwise keep old value
        V_soft = torch.where(updated_nodes_mask, new_V_soft, V_soft_old)
        
        # MUST Reset target node value to 0.0 after update
        V_soft[target_idx] = 0.0
        
        # Check for NaNs (debugging)
        if torch.isnan(V_soft).any():
            warnings.warn(f"NaN detected in V_soft during SBF iteration {iteration+1}. Check costs/graph structure.")
            V_soft = torch.nan_to_num(V_soft, nan=INF_REPLACEMENT, posinf=INF_REPLACEMENT, neginf=-INF_REPLACEMENT) # Try to recover

        # Check convergence
        delta = torch.abs(V_soft - V_soft_old)
        # Ignore change if both old and new are 'inf'
        max_delta = torch.max(delta[V_soft_old < INF_REPLACEMENT / 2]) if (V_soft_old < INF_REPLACEMENT / 2).any() else 0.0

        if max_delta < tolerance:
            # print(f"Soft Bellman-Ford converged in {iteration + 1} iterations.")
            converged = True
            break

    if not converged:
        warnings.warn(f"Soft Bellman-Ford did not converge within {max_iterations} iterations. Max delta: {max_delta:.4e}")

    # Final check for safety
    V_soft = torch.nan_to_num(V_soft, nan=INF_REPLACEMENT, posinf=INF_REPLACEMENT, neginf=-INF_REPLACEMENT)
    return V_soft


def calculate_transition_probs_pt(data: Data, costs: torch.Tensor, V_soft: torch.Tensor) -> torch.Tensor:
    """ Calculates Boltzmann transition probabilities p(j -> i) using PyTorch. """
    num_nodes = data.num_nodes
    edge_index = data.edge_index
    source_nodes, target_nodes = edge_index[0], edge_index[1]

    # Gather V_soft for source (j) and target (i) nodes
    v_j = V_soft[source_nodes]
    v_i = V_soft[target_nodes]

    # Mask edges where source or target V_soft is infinite
    finite_mask = (v_j < INF_REPLACEMENT / 2) & (v_i < INF_REPLACEMENT / 2)

    # Calculate unnormalized log probabilities only for valid edges
    log_probs_unnorm = torch.full_like(costs, float('-inf')) # Initialize with -inf
    log_probs_unnorm[finite_mask] = -costs[finite_mask] - v_i[finite_mask] + v_j[finite_mask]

    # Normalize per source node using scatter_logsumexp
    # Need fill_value=-inf for nodes with no outgoing valid edges -> handle manually
    log_Z_j = torch_scatter.scatter_logsumexp(log_probs_unnorm[finite_mask],
                                              source_nodes[finite_mask],
                                              dim=0,
                                              dim_size=num_nodes)
    # Nodes without any valid outgoing edge will have log_Z = -inf here

    # Gather log_Z back to edge shape
    log_Z_edge = log_Z_j[source_nodes]

    # Calculate normalized log probabilities
    # Subtracting -inf Z results in +inf, exp(inf)=inf. Subtracting finite Z from -inf prob = -inf.
    log_probs = log_probs_unnorm - log_Z_edge

    # Apply threshold and convert to probabilities
    probs = torch.exp(log_probs)
    probs[log_probs < LOG_PROB_THRESHOLD_Tensor] = 0.0 # Thresholding
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0) # Handle NaNs/Infs from exp(-inf) or exp(+inf)

    # Optional: Check if probs sum to 1 per node (for debugging)
    # sum_probs = torch_scatter.scatter_add(probs, source_nodes, dim=0, dim_size=num_nodes)
    # print("Probs sum check (should be close to 1 for nodes with finite V_soft and outgoing edges):", sum_probs[V_soft < INF_REPLACEMENT/2])

    return probs


def calculate_expected_visitations_pt(data: Data, start_idx: int, probs: torch.Tensor,
                                      max_iterations: int = DEFAULT_MAX_ITER,
                                      tolerance: float = DEFAULT_TOLERANCE) -> (torch.Tensor, torch.Tensor):
    """ Calculates expected state (D) and edge (E_pi) visitations using PyTorch. """
    num_nodes = data.num_nodes
    edge_index = data.edge_index
    source_nodes, target_nodes = edge_index[0], edge_index[1]

    # Initialize D (expected state visitations)
    D = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    D[start_idx] = 1.0

    converged = False
    for iteration in range(max_iterations):
        D_old = D.clone()

        # Gather D for source nodes (j)
        d_j = D_old[source_nodes]

        # Calculate flow along each edge: D(j) * p(j -> i)
        flow = d_j * probs # probs should already be 0 for invalid transitions

        # Aggregate incoming flow per target node (i) using scatter_add
        in_flow = torch_scatter.scatter_add(flow, target_nodes, dim=0, dim_size=num_nodes)

        # Update D: D_new(i) = I(i=start) + sum_{j} D_old(j) * p(j -> i)
        # Note: D(start) gets its own contribution + incoming flow
        D = in_flow
        D[start_idx] = D[start_idx] + 1.0 # Add the starting probability contribution

        # Check for convergence
        max_delta = torch.max(torch.abs(D - D_old))
        if max_delta < tolerance:
            # print(f"Expected state visitations (D) converged in {iteration + 1} iterations.")
            converged = True
            break

    if not converged:
         warnings.warn(f"Expected state visitations (D) did not converge within {max_iterations} iterations. Max delta: {max_delta:.4e}")

    # Calculate expected edge frequencies E_pi = D(j) * p(j -> i)
    E_pi = D[source_nodes] * probs
    
    # Check for NaNs (debugging)
    if torch.isnan(D).any() or torch.isnan(E_pi).any():
        warnings.warn(f"NaN detected in Visitations D or E_pi. Check probabilities/graph.")
        D = torch.nan_to_num(D, nan=0.0)
        E_pi = torch.nan_to_num(E_pi, nan=0.0)


    return D, E_pi


def calculate_maxent_loss(data: Data, V_soft: torch.Tensor, costs: torch.Tensor,
                          start_idx: int, alpha_r: float,
                          demo_edge_indices: list) -> torch.Tensor:
    """ Calculates the MaxEnt IRL NLL loss + L2 regularization. """

    # 1. NLL of demonstrations
    total_demo_cost = 0.0
    num_demos = len(demo_edge_indices)
    if num_demos == 0:
        # Should have been caught earlier, but handle defensively
        warnings.warn("Calculating loss with zero valid demonstrations.")
        # Loss becomes just partition function + regularization
        nll_demos = torch.tensor(0.0, device=device)
    else:
        for path_indices in demo_edge_indices:
            if len(path_indices) > 0:
                total_demo_cost += torch.sum(costs[path_indices])
            # else: path had no valid edges, ignore
        nll_demos = total_demo_cost / num_demos

    # 2. Log Partition Function Term (approximated by -V_soft[start])
    # V_soft(start) = -log Z_start => -log Z_start = V_soft(start)
    log_partition_term = V_soft[start_idx]
    
    # Handle case where start node has infinite value (no path to target)
    if log_partition_term >= INF_REPLACEMENT / 2 :
        warnings.warn(f"Start node {start_idx} has infinite soft value ({log_partition_term:.2e}). Loss will be infinite. Check graph connectivity/costs.")
        # Return INF loss to signal problem - optimizer likely won't proceed.
        # Need to make sure it's differentiable inf? Or just large number?
        # Use a large number compatible with the gradient calculation.
        # Autograd might struggle with Inf. Let's return a very large loss.
        log_partition_term = torch.tensor(INF_REPLACEMENT * 10, device=device, dtype=torch.float32) # Make it huge


    # 3. L2 Regularization Term
    reg_term = alpha_r * torch.sum(data.preference_score**2) / 2.0

    # 4. Total Loss
    total_loss = nll_demos + log_partition_term + reg_term

    return total_loss

# --- Main Optimization Function (PyTorch) ---

def optimize_preferences_pt(graph: nx.DiGraph,
                            demonstrations: list,
                            start_node,
                            target_node,
                            alpha: float,
                            alpha_r: float,
                            learning_rate: float,
                            num_epochs: int,
                            initial_preference: float = 0.0,
                            convergence_threshold_grad: float = 1e-4, # Gradient norm based
                            early_stopping_patience: int = 10,
                            max_iter_sbf: int = DEFAULT_MAX_ITER,
                            max_iter_vis: int = DEFAULT_MAX_ITER,
                            tolerance_sbf: float = DEFAULT_TOLERANCE,
                            tolerance_vis: float = DEFAULT_TOLERANCE,
                            print_interval: int = 10,
                            infer_mode: bool = False):
    """
    Performs gradient descent using PyTorch and Autograd.
    """
    start_time = time.time()

    # 1. Setup Graph Data and Parameters
    data = networkx_to_pyg_data(graph, initial_preference=initial_preference)
    # Ensure parameters are tracked for optimization
    preference_param = data.preference_score # This IS the nn.Parameter

    # Get node indices
    try:
        start_idx = data.node_to_idx[start_node]
        target_idx = data.node_to_idx[target_node]
    except KeyError as e:
        raise ValueError(f"Start or Target node '{e}' not found in the graph nodes.") from e

    # 2. Preprocess Demonstrations
    E_D_tensor, demo_edge_indices = preprocess_demonstrations(demonstrations, data)

    # Check if preprocessing failed
    if len(demo_edge_indices) == 0 and not infer_mode:
         print("Error: No valid demonstration paths to learn from. Aborting.")
         return data, None # Return initial data

    # 3. Setup Optimizer
    optimizer = optim.Adam([preference_param], lr=learning_rate)

    # 4. Training Loop
    history = {'epoch': [], 'loss': [], 'grad_norm': []}
    best_loss = float('inf')
    best_epoch = -1
    best_prefs = None
    patience_counter = 0

    print(f"\nStarting optimization on {device} for {num_epochs} epochs...")
    progress_bar = tqdm(range(num_epochs), desc="Epochs", total=num_epochs)

    for epoch in progress_bar:
        optimizer.zero_grad()

        # --- Forward Pass ---
        # 1. Calculate Costs
        costs = calculate_edge_costs_pt(data, alpha)

        # 2. Compute Soft Value Function (SBF)
        V_soft = soft_bellman_ford_pt(data, target_idx, costs, max_iter_sbf, tolerance_sbf)
        
        # Check for fatal error from SBF (start node unreachable)
        if V_soft[start_idx] >= INF_REPLACEMENT / 2 and not infer_mode:
             print(f"\nEpoch {epoch+1}: Error - Start node {start_node} has infinite value. Cannot proceed. Check graph connectivity/costs.")
             # Potentially restore best state if one existed? Or just stop.
             if best_prefs is not None:
                 with torch.no_grad():
                     preference_param.copy_(best_prefs)
                 print("Restored preferences from best epoch.")
             return data, history # Stop optimization


        # (Steps 3 & 4 - Probs & Visitations are implicitly handled by loss calculation via V_soft & costs)
        # For pure inference or debugging, you might calculate them explicitly:
        # probs = calculate_transition_probs_pt(data, costs, V_soft)
        # D, E_pi = calculate_expected_visitations_pt(data, start_idx, probs, max_iter_vis, tolerance_vis)

        # 5. Calculate Loss (using the NLL formulation for Autograd)
        loss = calculate_maxent_loss(data, V_soft, costs, start_idx, alpha_r, demo_edge_indices)

        # --- Backward Pass & Update ---
        if not infer_mode and torch.isfinite(loss):
            # 6. Backpropagate
            loss.backward()

            # Check for gradient issues
            grad_norm = torch.tensor(0.0)
            if preference_param.grad is not None:
                 grad_norm = torch.linalg.norm(preference_param.grad).item()
                 if torch.isnan(preference_param.grad).any():
                     warnings.warn(f"Epoch {epoch+1}: NaN gradient detected! Skipping optimizer step.")
                     optimizer.zero_grad() # Clear the bad gradient
                 else:
                     # Optional: Gradient clipping
                     # torch.nn.utils.clip_grad_norm_([preference_param], max_norm=1.0)
                     # 7. Optimizer Step
                     optimizer.step()
            else:
                 warnings.warn(f"Epoch {epoch+1}: Gradient is None. Check loss computation and graph connectivity.")


        elif not torch.isfinite(loss):
            warnings.warn(f"Epoch {epoch+1}: Loss is {loss}. Skipping backward pass and optimizer step.")
            optimizer.zero_grad() # Clear any potential stale grads


        # --- Logging & Convergence ---
        loss_item = loss.item() if torch.isfinite(loss) else float('inf')
        history['epoch'].append(epoch + 1)
        history['loss'].append(loss_item)
        history['grad_norm'].append(grad_norm if not infer_mode and torch.isfinite(loss) else 0.0)

        postfix_dict = {'Loss': f'{loss_item:.4e}'}
        if not infer_mode:
             postfix_dict['GradNorm'] = f'{grad_norm:.4e}'
        progress_bar.set_postfix(postfix_dict)
        
        if infer_mode:
             print(f"\nInference mode complete after 1 epoch. Loss: {loss_item:.4e}")
             break # Run only one epoch in inference mode

        # Early Stopping & Best Model Saving
        if loss_item < best_loss:
            best_loss = loss_item
            best_epoch = epoch
            # Save the best preference parameters
            best_prefs = preference_param.detach().clone()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}. Loss hasn't improved for {early_stopping_patience} epochs.")
                break

        # Convergence Check (Gradient Norm)
        if grad_norm < convergence_threshold_grad and epoch > 0: # Avoid stopping at epoch 0
             print(f"\nConvergence reached at epoch {epoch + 1}. Gradient norm below threshold.")
             break

    # --- End of Training ---
    end_time = time.time()
    print(f"\nOptimization finished in {end_time - start_time:.2f} seconds.")

    # Restore best model if early stopping occurred or if final loss is worse
    if best_prefs is not None and (patience_counter >= early_stopping_patience or history['loss'][-1] > best_loss):
         print(f"Restoring best preferences from epoch {best_epoch + 1} with loss {best_loss:.4e}")
         with torch.no_grad():
             preference_param.copy_(best_prefs)
    elif best_prefs is not None:
         print(f"Best loss was {best_loss:.4e} at epoch {best_epoch + 1}")


    # Return the data object with optimized parameters and history
    # Ensure final preference scores are updated in the data object if Adam's internal state was used
    data.preference_score = preference_param # Re-assign the parameter tensor

    return data, history


def optimize_preferences_pt_with_restarts(*args, num_restarts: int = 5, init_std: float = 0.1, **kwargs):
    """ Runs the PyTorch optimization multiple times with random initializations. """
    best_final_loss = float('inf')
    best_data = None
    best_history = None
    
    # Store initial graph to reset from
    graph_orig = args[0]
    initial_preference_mean = kwargs.get('initial_preference', 0.0)

    for restart in range(num_restarts):
        print(f"\n--- Restart {restart + 1}/{num_restarts} ---")
        # Need to re-initialize the preference parameter
        # Easiest way: Create a new Data object or reset the parameter
        
        # Modify kwargs for initialization in this run
        current_kwargs = kwargs.copy()
        # We handle init inside optimize_preferences_pt by passing a new initial_preference value if needed
        # OR, more robustly, re-create the data object with noise
        
        graph_copy = copy.deepcopy(graph_orig) # Start from the original graph structure
        
        # Create PyG data with randomized initial prefs for this restart
        data_restart = networkx_to_pyg_data(graph_copy, initial_preference=0.0) # Temp init
        with torch.no_grad():
             noise = torch.randn_like(data_restart.preference_score) * init_std
             data_restart.preference_score.copy_(initial_preference_mean + noise)
        
        # Prepare args for optimize_preferences_pt, replacing graph with data_restart
        current_args = list(args)
        current_args[0] = data_restart # Pass the initialized Data object

        # Run optimization
        optimized_data, history = optimize_preferences_pt(*current_args, **current_kwargs)

        final_loss = history['loss'][-1] if history and history['loss'] else float('inf')
        print(f"Restart {restart + 1} final loss: {final_loss:.4e}")

        if final_loss < best_final_loss:
            best_final_loss = final_loss
            best_data = optimized_data # Keep the Data object with optimized params
            best_history = history

    print(f"\nBest loss after {num_restarts} restarts: {best_final_loss:.4e}")
    
    # If best_data is None (all restarts failed), maybe return the initial data?
    if best_data is None:
         warnings.warn("All restarts failed or produced non-finite loss. Returning initial graph structure.")
         best_data = networkx_to_pyg_data(graph_orig, kwargs.get('initial_preference', 0.0))


    return best_data, best_history

def extract_results_to_networkx(final_data: Data, original_graph: nx.DiGraph):
    """Copies the optimized preferences back to a NetworkX graph."""
    output_graph = original_graph.copy()
    
    optimized_prefs = final_data.preference_score.detach().cpu().numpy()
    edge_list_map = final_data.edge_list_map # Original (u, v, data) tuples used for indexing
    
    if len(edge_list_map) != len(optimized_prefs):
        warnings.warn("Mismatch between number of edges in map and optimized preferences. Results might be incorrect.")
        return output_graph # Return original on error

    for i, (u, v, _) in enumerate(edge_list_map):
        if output_graph.has_edge(u, v):
             # Ensure the edge has the attribute field before assigning
             if 'preference_score' not in output_graph.edges[u,v]:
                  output_graph.edges[u, v]['preference_score'] = 0.0 # Initialize if missing
             output_graph.edges[u, v]['preference_score'] = optimized_prefs[i]
        else:
             warnings.warn(f"Edge ({u}, {v}) from mapping not found in target NetworkX graph.")

    return output_graph


# --- Example Usage ---
if __name__ == "__main__":
    # 1. Create a sample graph (same as before)
    G = nx.DiGraph()
    G.add_nodes_from(['A', 'B', 'C', 'D', 'E', 'T']) # T is target
    G.add_edge('A', 'B', distance=1.0)
    G.add_edge('A', 'C', distance=2.0)
    G.add_edge('B', 'D', distance=1.5)
    G.add_edge('B', 'T', distance=5.0)
    G.add_edge('C', 'D', distance=1.0)
    G.add_edge('C', 'E', distance=2.5)
    G.add_edge('D', 'T', distance=2.0)
    G.add_edge('E', 'T', distance=1.0)
    G.add_edge('A', 'D', distance=4.0)
    G.add_edge('D', 'B', distance=1.5) # Cycle

    # 2. Define demonstrated routes
    demonstrations = [
        ['A', 'C', 'E', 'T'],
        ['A', 'C', 'E', 'T'],
        ['A', 'C', 'D', 'T'],
        ['A', 'C', 'E', 'T'],
        ['A', 'B', 'D', 'T'], # Add another path
    ]

    # 3. Define parameters
    start_node = 'A'
    target_node = 'T'
    alpha = 1.0       # Weight for preference score in cost
    alpha_r = 0.01    # Regularization strength (L2)
    learning_rate = 0.05 # Often needs to be smaller with Adam
    num_epochs = 200    # Number of optimization iterations
    initial_preference = 0.0 # Initial guess for preference scores
    convergence_threshold_grad = 1e-5 # Stop if max gradient component is small
    early_stopping_patience = 15

    # 4. Run the optimization (potentially with restarts)
    # Option 1: Single run
    # final_data, history = optimize_preferences_pt(
    #     graph=G,
    #     demonstrations=demonstrations,
    #     start_node=start_node,
    #     target_node=target_node,
    #     alpha=alpha,
    #     alpha_r=alpha_r,
    #     learning_rate=learning_rate,
    #     num_epochs=num_epochs,
    #     initial_preference=initial_preference,
    #     convergence_threshold_grad=convergence_threshold_grad,
    #     early_stopping_patience=early_stopping_patience,
    # )

    # Option 2: With restarts
    final_data, history = optimize_preferences_pt_with_restarts(
        # Args for optimize_preferences_pt
        G, # Pass the graph for the first arg
        demonstrations,
        start_node,
        target_node,
        alpha,
        alpha_r,
        learning_rate,
        num_epochs,
        # Kwargs for optimize_preferences_pt
        initial_preference=initial_preference,
        convergence_threshold_grad=convergence_threshold_grad,
        early_stopping_patience=early_stopping_patience,
        # Kwargs specific to restarts
        num_restarts=3,
        init_std=0.1
    )


    # 5. Inspect the results
    if final_data and history:
        print("\n--- Optimization Results ---")
        # Extract final preferences back to NetworkX graph for easy viewing
        optimized_graph = extract_results_to_networkx(final_data, G)

        print("Final Preference Scores (in NetworkX graph):")
        for u, v, data in optimized_graph.edges(data=True):
            pref = data.get('preference_score', 'N/A')
            dist = data.get('distance', 'N/A')
            print(f"Edge ({u}, {v}): Distance={dist}, Preference={pref:.4f}")

        # Example: Print final preferences directly from tensor
        print("\nFinal Preference Scores (Tensor):")
        final_prefs_tensor = final_data.preference_score.detach().cpu().numpy()
        edge_list_map = final_data.edge_list_map
        for i, (u, v, _) in enumerate(edge_list_map):
            print(f"Edge ({u}, {v}) - Index {i}: {final_prefs_tensor[i]:.4f}")

        # Plotting convergence
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.plot(history['epoch'], history['loss'])
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Loss Convergence")
            plt.yscale('log')
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.plot(history['epoch'], history['grad_norm'])
            plt.xlabel("Epoch")
            plt.ylabel("Gradient Norm (L2)")
            plt.title("Gradient Norm Convergence")
            plt.yscale('log')
            plt.axhline(convergence_threshold_grad, color='r', linestyle='--', label='Threshold')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.show()
        except ImportError:
            print("\nMatplotlib not found. Cannot plot convergence history.")
        except Exception as e:
            print(f"\nError plotting history: {e}") # Catch other potential plotting errors
            print(f"History content: Loss length {len(history.get('loss',[]))}, GradNorm length {len(history.get('grad_norm',[]))}")


    else:
        print("\nOptimization did not complete successfully or was aborted.")