import numpy as np
from scipy import sparse
import networkx as nx
from typing import Union
from scipy.sparse.linalg import spsolve

# ===============================
# SINGULAR COST FUNCTION
# ===============================

def compute_edge_costs_dict(G: nx.DiGraph, alpha_pref: float = 1.0):
    """
    Compute edge costs for a directed graph by combining distance and preference scores.

    This function calculates the cost of each edge in a directed graph by subtracting 
    a weighted preference score from the edge's distance. This allows for incorporating 
    both distance and preference information into edge costs.

    Parameters:
    -----------
    G : nx.DiGraph
        The input directed graph with edges that may have 'distance' and 'preference' attributes.
    alpha_pref : float, optional
        A weight factor for the preference score. Defaults to 1.0.
        - Larger values give more importance to preference scores
        - Smaller values reduce the impact of preference scores

    Returns:
    --------
    dict
        A dictionary of edge costs, where keys are edge tuples (source, target) 
        and values are the computed costs.
        Cost = distance - alpha_pref * preference_score

    Notes:
    ------
    - If 'distance' is not specified for an edge, it defaults to 1.0
    - If 'preference' is not specified for an edge, it defaults to 0.0
    - The cost can be negative if the preference score is high relative to the distance

    Example:
    --------
    >>> import networkx as nx
    >>> G = nx.DiGraph()
    >>> G.add_edge(1, 2, distance=2.0, preference=0.5)
    >>> costs = compute_edge_costs_dict(G, alpha_pref=1.0)
    >>> print(costs[(1, 2)])  # Output: 1.5
    """
    
    costs = {}
    for u, v, data in G.edges(data=True):
        distance = data.get('distance', 1.0) # Default distance if missing
        preference_score = data.get('preference', 0.0) # Default preference if missing
        costs[(u, v)] = distance - alpha_pref * preference_score
    return costs

def graph_to_sparse_matrices(G: nx.DiGraph, costs_dict: dict = None):
    """
    Convert a NetworkX directed graph to sparse adjacency and cost matrices.
    
    This function transforms a NetworkX DiGraph into sparse matrix representations
    suitable for efficient graph algorithms. It creates both an adjacency matrix
    and a corresponding cost matrix in CSR (Compressed Sparse Row) format.
    
    Parameters:
    -----------
    G : nx.DiGraph
        The input directed graph to convert.
    
    costs_dict : dict, optional
        A dictionary mapping edge tuples (u, v) to their costs.
        If None, all edges will have a default cost of 1.0.
    
    Returns:
    --------
    tuple (sparse.csr_matrix, sparse.csr_matrix, dict)
        A tuple containing:
        - A sparse adjacency matrix where A[j, i] = 1 indicates an edge from j to i
        - A sparse cost matrix where costs[j, i] contains the cost of the edge from j to i
        - A dictionary mapping node indices to original node names/IDs
    
    Notes:
    ------
    - The matrices use the same node ordering as G.nodes()
    - For nodes represented by non-integer values, a mapping is created internally
    - The adjacency matrix has entries of 1 where edges exist
    - The cost matrix contains the actual edge costs from the costs_dict
    
    Example:
    --------
    >>> import networkx as nx
    >>> G = nx.DiGraph()
    >>> G.add_edge('A', 'B')
    >>> G.add_edge('B', 'C')
    >>> costs = {('A', 'B'): 2.0, ('B', 'C'): 1.5}
    >>> A, C, node_mapping = graph_to_sparse_matrices(G, costs)
    """
    # Get list of nodes and create a mapping from node to index
    nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    idx_to_node = {i: node for i, node in enumerate(nodes)}
    n = len(nodes)
    
    # Create lists for constructing sparse matrices
    adj_row_indices = []
    adj_col_indices = []
    adj_data = []
    cost_row_indices = []
    cost_col_indices = []
    cost_data = []
    
    # Populate the lists
    for u, v in G.edges():
        j = node_to_idx[u]  # Source node index
        i = node_to_idx[v]  # Target node index
        
        # Add to adjacency matrix data
        adj_row_indices.append(j)
        adj_col_indices.append(i)
        adj_data.append(1)
        
        # Add to cost matrix data
        cost = 1.0  # Default cost
        if costs_dict and (u, v) in costs_dict:
            cost = costs_dict[(u, v)]
        
        cost_row_indices.append(j)
        cost_col_indices.append(i)
        cost_data.append(cost)
    
    # Create sparse matrices in CSR format directly for potential efficiency
    adjacency_matrix = sparse.csr_matrix(
        (adj_data, (adj_row_indices, adj_col_indices)),
        shape=(n, n)
    )

    cost_matrix = sparse.csr_matrix(
        (cost_data, (cost_row_indices, cost_col_indices)),
        shape=(n, n)
    )
    
    return adjacency_matrix, cost_matrix, idx_to_node

def check_sparsity_pattern_match(A, costs):
    """
    Check if two sparse matrices have the same sparsity pattern.
    
    Parameters:
    -----------
    A : sparse.csr_matrix
        First sparse matrix
    costs : sparse.csr_matrix
        Second sparse matrix
    
    Returns:
    --------
    bool
        True if the sparsity patterns match, False otherwise
    """
    # Ensure both matrices are in CSR format for consistent comparison
    if not sparse.isspmatrix_csr(A):
        A = A.tocsr()
    if not sparse.isspmatrix_csr(costs):
        costs = costs.tocsr()
    
    # Efficient check using internal attributes
    pattern_match = (
        A.shape == costs.shape and
        np.array_equal(A.indptr, costs.indptr) and
        np.array_equal(A.indices, costs.indices)
    )

    if not pattern_match:
        # Report the differences if needed (can be computationally expensive)
        A_rows, A_cols = A.nonzero()
        costs_rows, costs_cols = costs.nonzero()
        A_indices = set(zip(A_rows, A_cols))
        costs_indices = set(zip(costs_rows, costs_cols))
        only_in_A = A_indices - costs_indices
        only_in_costs = costs_indices - A_indices
        print(f"Sparsity patterns differ.")
        print(f"Entries only in A: {len(only_in_A)}")
        print(f"Entries only in costs: {len(only_in_costs)}")
        if len(only_in_A) > 0:
            print(f"Sample entries only in A: {list(only_in_A)[:5]}")
        if len(only_in_costs) > 0:
            print(f"Sample entries only in costs: {list(only_in_costs)[:5]}")

    return pattern_match

# ===============================
# STRUCTURED COST FUNCTION
# ===============================
# To store the components of a structured cost function, we need to divide the components into three categories:
# - Intrinsic properties such as distance
# - Edge specific properties such as preference
# - Shared properties such as "weather, sector load attractiveness..."
#
# How to store the components:
# - The intrinsic properties are hard-coded into the cost function.
# - The edge specific properties are stored in a sparse matrix with the same sparsity pattern as the adjacency matrix.
# - The shared properties, if follow piece-wise linear functions, can be store in a set of three vectors: one for breakpoints, one for the slopes, and one for the intercepts.

def get_cost_matrix(
    A: sparse.csr_matrix,                  # Adjacency matrix defining graph structure
    intrinsic_costs: sparse.csr_matrix,    # Base costs (e.g., distances)
    edge_properties: dict = None,          # Dict of {property_name: (sparse_matrix, weight)}
    shared_properties: dict = None,        # Dict of {property_name: (values, breakpoints, slopes, intercepts, weight)}
    default_weight: float = 1.0            # Default weight for intrinsic costs
) -> sparse.csr_matrix:
    """
    Constructs a cost matrix by combining intrinsic costs with edge-specific and shared properties.
    
    Parameters:
    -----------
    A : sparse.csr_matrix
        Adjacency matrix that defines the graph structure and sparsity pattern.
    
    intrinsic_costs : sparse.csr_matrix
        Base intrinsic costs (e.g., distances) with same sparsity pattern as A.
    
    edge_properties : dict, optional
        Dictionary mapping property names to tuples of (sparse_matrix, weight).
        Each sparse_matrix must have the same sparsity pattern as A.
        The weight controls the influence of the property on the final cost.
    
    shared_properties : dict, optional
        Dictionary mapping property names to tuples of (values, breakpoints, slopes, intercepts, weight).
        - values: np.ndarray of property values for each node or edge
        - breakpoints: np.ndarray defining piece-wise linear function breakpoints
        - slopes: np.ndarray defining slopes between breakpoints
        - intercepts: np.ndarray defining intercepts for each segment
        - weight: float controlling the influence of this property
    
    default_weight : float, optional
        Weight factor for the intrinsic costs. Default is 1.0.
        
    # Example usage of get_cost_matrix with various properties
    
    # Create a sample adjacency matrix
    N = 5  # Number of nodes
    row_indices = [0, 0, 1, 1, 2, 3]
    col_indices = [1, 2, 2, 3, 4, 4]
    A = sparse.csr_matrix(([1]*len(row_indices), (row_indices, col_indices)), shape=(N, N))
    
    # Create intrinsic costs matrix (distances)
    intrinsic_costs_data = [2.0, 1.5, 3.0, 2.5, 1.0, 1.8]
    intrinsic_costs = sparse.csr_matrix((intrinsic_costs_data, (row_indices, col_indices)), shape=(N, N))
    
    # Create edge-specific properties: preference scores
    edge_pref_data = [0.5, 0.2, 0.7, 0.3, 0.1, 0.8]
    edge_pref_matrix = sparse.csr_matrix((edge_pref_data, (row_indices, col_indices)), shape=(N, N))
    edge_properties = {'preference': (edge_pref_matrix, 2.0)}  # weight of 2.0 for preferences
    
    # Create shared properties: node congestion values with piecewise linear transform
    node_congestion = np.array([0.1, 0.5, 0.8, 0.3, 0.9])  # Congestion value for each node
    # Piecewise linear function: breakpoints, slopes, and intercepts
    breakpoints = np.array([0.0, 0.3, 0.7, 1.0])
    slopes = np.array([0.5, 1.0, 2.0])
    intercepts = np.array([0.0, 0.15, 0.5])
    shared_properties = {'congestion': (node_congestion, breakpoints, slopes, intercepts, 1.5)}
    
    # Calculate the combined cost matrix
    final_costs = get_cost_matrix(
        A=A,
        intrinsic_costs=intrinsic_costs,
        edge_properties=edge_properties,
        shared_properties=shared_properties,
        default_weight=1.0
    )
    
    Returns:
    --------
    sparse.csr_matrix
        Combined cost matrix with the same sparsity pattern as A.
    """
    # Ensure matrices are in CSR format
    if not sparse.isspmatrix_csr(A):
        A = A.tocsr()
    if not sparse.isspmatrix_csr(intrinsic_costs):
        intrinsic_costs = intrinsic_costs.tocsr()
    
    # Check sparsity pattern match
    if not check_sparsity_pattern_match(A, intrinsic_costs):
        raise ValueError("Intrinsic costs matrix must have the same sparsity pattern as adjacency matrix.")
    
    # Start with weighted intrinsic costs
    final_costs = intrinsic_costs.copy()
    final_costs.data *= default_weight
    
    # Add edge-specific properties
    if edge_properties: # for example: {'preference': (edge_pref_matrix, 2.0)}
        for prop_name, (prop_matrix, prop_weight) in edge_properties.items():
            if not sparse.isspmatrix_csr(prop_matrix):
                prop_matrix = prop_matrix.tocsr()
            
            if not check_sparsity_pattern_match(A, prop_matrix):
                raise ValueError(f"Edge property matrix '{prop_name}' must have the same sparsity pattern as adjacency matrix.")
            
            # Add weighted property to the final costs
            weighted_prop = prop_matrix.copy()
            weighted_prop.data *= prop_weight
            final_costs = final_costs + weighted_prop
    
    # Process shared properties with piece-wise linear functions
    if shared_properties: # for example: {'congestion': (node_congestion, breakpoints, slopes, intercepts, 1.5)}
        sources, targets = A.nonzero()  # Get source and target nodes for all edges
        
        for prop_name, (values, breakpoints, slopes, intercepts, prop_weight) in shared_properties.items():
            # Apply piece-wise linear function to the property values
            transformed_values = np.zeros_like(values, dtype=float)
            
            for i, val in enumerate(values):
                # Find the segment this value belongs to
                idx = np.searchsorted(breakpoints, val, side='right') - 1
                idx = max(0, min(idx, len(slopes) - 1))  # Ensure valid index
                
                # Apply the corresponding linear function
                transformed_values[i] = slopes[idx] * val + intercepts[idx]
            
            # Create a sparse matrix for this shared property
            prop_data = transformed_values * prop_weight
            shared_prop_matrix = sparse.csr_matrix((prop_data, (sources, targets)), shape=A.shape)
            
            # Add to final costs
            final_costs = final_costs + shared_prop_matrix
    
    return final_costs

import numpy as np
from scipy import sparse
from typing import Union

def bellman_fw(A: sparse.csr_matrix, costs: sparse.csr_matrix, source_node: Union[int, str],
            max_iter: int = 1000, tol: float = 1e-3, epsilon = 1e-10):
    """
    Compute the value function using a sparse Bellman-Ford-like algorithm with exponential updates.
    
    This function performs an iterative value function computation on a sparse graph,
    using an exponential update rule. It is particularly useful for computing shortest
    paths or value propagation in sparse graph structures.
    
    Parameters:
    -----------
    A : sparse.csr_matrix
        The adjacency matrix of the graph in Compressed Sparse Row (CSR) format.
        A[j, i] = 1 indicates an edge from node j to node i.
    
    costs : sparse.csr_matrix
        A sparse matrix representing the edge costs between nodes.
        costs[j, i] represents the cost of the edge from node j to node i.
    
    source_node : int or str
        The index of the source node from which to start the value computation.
    
    max_iter : int, optional (default=1000)
        Maximum number of iterations to perform before stopping.
    
    tol : float, optional (default=1e-3)
        Convergence tolerance. The algorithm stops when the maximum change 
        in the value function between iterations is less than this threshold.
    
    Returns:
    --------
    numpy.ndarray
        The computed value function, where each entry represents the 
        value/cost associated with each node from the source.
    """
    # Number of nodes
    N = A.shape[0]
    
    # Initialize the value function: all nodes start with infinity except the source
    V_k = np.full(N, np.inf)
    V_k[source_node] = 0.0
    
    # Ensure matrices are in CSR format
    if not sparse.isspmatrix_csr(A):
        A = A.tocsr()
    if not sparse.isspmatrix_csr(costs):
        costs = costs.tocsr()
    
    # Pre-extract non-zero elements and corresponding costs
    # Assumes A and costs have the same sparsity pattern and internal data order
    sources, targets = A.nonzero() # j -> i
    if len(costs.data) != len(sources):
         raise ValueError(f"Mismatch in number of non-zero elements between A ({len(sources)}) and costs ({len(costs.data)}). Ensure they have the same sparsity pattern and are both CSR.")
    edge_costs = costs.data

    for _ in range(max_iter):
        # print(f"Iteration {_}, V_k[231] = {V_k[231]}")
        # Save the previous value function for convergence checking
        V_prev = V_k.copy()
        
        # Compute the exponential update term for each edge:
        # exp(-(V[source] + cost)) for each edge from source j -> target i
        # V_k[sources] retrieves V_k[j] for all source nodes j of existing edges
        edge_exp_terms = np.exp(-(V_k[sources] + edge_costs))
        
        # Sum the contributions for each target node i using np.bincount
        # weights=edge_exp_terms sums the exp terms for edges pointing *to* the same target i
        Sum_Exp_vector = np.bincount(targets, weights=edge_exp_terms, minlength=N)
        
        # Update the value function using the -log transformation
        # Add epsilon to prevent log(0)
        V_k = -np.log(Sum_Exp_vector + epsilon)
        
        # Re-apply the source condition (value at source is always 0)
        V_k[source_node] = 0.0
        
        # Check for convergence: if maximum change is below the tolerance, stop iterating
        if np.max(np.abs(V_k - V_prev)) < tol:
            break
    
    return V_k

def bellman_bw(A: sparse.csr_matrix, costs: sparse.csr_matrix, goal_node: Union[int, str],
               max_iter: int = 1000, tol: float = 1e-3, epsilon = 1e-10):
    """
    Compute the cost-to-go function using a sparse Bellman-Ford-like algorithm with exponential updates.
    
    This function performs an iterative computation on a sparse graph, but instead of propagating
    values from a source, it propagates them in reverse from a goal node, computing the cost-to-go.
    
    Parameters:
    -----------
    A : sparse.csr_matrix
        The adjacency matrix of the graph in Compressed Sparse Row (CSR) format.
        A[j, i] = 1 indicates an edge from node j to node i.
    
    costs : sparse.csr_matrix
        A sparse matrix representing the edge costs between nodes.
        costs[j, i] represents the cost of the edge from node j to node i.
    
    goal_node : int or str
        The index of the goal node at which the cost is zero.
    
    max_iter : int, optional (default=1000)
        Maximum number of iterations to perform before stopping.
    
    tol : float, optional (default=1e-3)
        Convergence tolerance. The algorithm stops when the maximum change in the value function 
        between iterations is less than this threshold.
    
    Returns:
    --------
    numpy.ndarray
        The computed cost-to-go value function, where each entry represents the cost associated 
        with reaching the goal from that node.
    """
    # Number of nodes
    N = A.shape[0]
    
    # Initialize the value function: all nodes start with infinity except the goal node
    V_k = np.full(N, np.inf)
    V_k[goal_node] = 0.0
    
    # Ensure matrices are in CSR format
    if not sparse.isspmatrix_csr(A):
        A = A.tocsr()
    if not sparse.isspmatrix_csr(costs):
        costs = costs.tocsr()
    
    # Pre-extract non-zero elements and corresponding costs
    # Assumes A and costs have the same sparsity pattern and internal data order
    sources, targets = A.nonzero() # j -> i
    if len(costs.data) != len(sources):
         raise ValueError(f"Mismatch in number of non-zero elements between A ({len(sources)}) and costs ({len(costs.data)}). Ensure they have the same sparsity pattern and are both CSR.")
    edge_costs = costs.data

    for _ in range(max_iter):
        # Save the previous value function for convergence checking
        V_prev = V_k.copy()
        
        # Compute the exponential update term for each edge:
        # For each edge from node j (source) to node i (target), we use V_k[i] (cost-to-go from target)
        # V_k[targets] retrieves V_k[i] for all target nodes i of existing edges
        edge_exp_terms = np.exp(-(V_k[targets] + edge_costs))
        
        # Sum the contributions for each source node j using np.bincount
        # weights=edge_exp_terms sums the exp terms for edges originating *from* the same source j
        Sum_Exp_vector = np.bincount(sources, weights=edge_exp_terms, minlength=N)
        
        # Update the value function using the -log transformation
        V_k = -np.log(Sum_Exp_vector + epsilon)
        
        # print(f'V[21] = {V_prev[21]} -> {V_k[21]}')
        
        # Re-apply the goal condition: cost at goal remains zero
        V_k[goal_node] = 0.0
        
        # Check for convergence: if maximum change is below the tolerance, stop iterating
        if np.max(np.abs(V_k - V_prev)) < tol:
            break
    
    return V_k

def compute_edge_traversal_likelihoods(A: sparse.csr_matrix, costs: sparse.csr_matrix,
                                       V_fw: np.ndarray, V_bw: np.ndarray,
                                       overall_source_node: int) -> sparse.csr_matrix:
    """
    Computes the likelihood of traversing each edge (j, i) as part of paths
    from a given overall_source_node to the goal node used for V_bw.

    The likelihood L(j, i) is calculated relative to the total likelihood of all
    paths from the overall_source_node to the goal node. It is proportional to:
    exp(-(V_fw[j] + cost(j, i) + V_bw[i]))
    normalized by exp(-V_bw[overall_source_node]).

    Parameters:
    -----------
    A : sparse.csr_matrix
        Adjacency matrix (A[j, i] = 1 for edge j -> i). Assumed CSR format.
    costs : sparse.csr_matrix
        Cost matrix (costs[j, i] = cost of edge j -> i). Assumed CSR format.
        Must have the same sparsity pattern as A.
    V_fw : np.ndarray
        Forward value function (cost from overall_source_node to node j). Shape (N,).
        Assumes V_fw[overall_source_node] == 0.
    V_bw : np.ndarray
        Backward value function (cost from node i to the goal node). Shape (N,).
    overall_source_node : int
        The index of the source node 's' used to compute V_fw.

    Returns:
    --------
    sparse.csr_matrix
        A sparse matrix L where L[j, i] contains the likelihood
        of traversing edge j -> i in paths from overall_source_node to the goal.
        The matrix has the same sparsity pattern as A and costs.
        The sum of all elements L[j, i] should approximate 1 if the goal is reachable.

    Raises:
    -------
    ValueError
        If A and costs do not have the same shape or sparsity pattern.
        If V_fw or V_bw dimensions do not match the matrix dimensions.
        If overall_source_node index is invalid.
    """
    N = A.shape[0]
    if A.shape != costs.shape:
        raise ValueError(f"Adjacency matrix shape {A.shape} and cost matrix shape {costs.shape} must match.")
    if V_fw.shape != (N,) or V_bw.shape != (N,):
        raise ValueError(f"Value function shapes {V_fw.shape}, {V_bw.shape} must match matrix dimension {N}.")
    if not (0 <= overall_source_node < N):
        raise ValueError(f"overall_source_node index {overall_source_node} out of bounds for N={N}.")

    # Ensure matrices are in CSR format
    if not sparse.isspmatrix_csr(A):
        A = A.tocsr()
    if not sparse.isspmatrix_csr(costs):
        costs = costs.tocsr()

    # Extract edges (sources j, targets i) for edges j -> i
    sources, targets = A.nonzero()

    # Get corresponding costs
    # Assume A and costs have identical .data ordering when in CSR format
    if len(costs.data) != len(sources):
         check_sparsity_pattern_match(A, costs) # Assuming check_sparsity_pattern_match is defined elsewhere
         raise ValueError(f"Mismatch in number of non-zero elements between A ({len(sources)}) and costs ({len(costs.data)}). Ensure they have the same sparsity pattern and were created/converted consistently to CSR.")
    edge_costs = costs.data

    # Gather forward values at source nodes (V_fw[j])
    V_fw_sources = V_fw[sources]

    # Gather backward values at target nodes (V_bw[i])
    V_bw_targets = V_bw[targets]

    # Compute the unnormalized weight for each edge: exp(-(V_fw[j] + cost(j,i) + V_bw[i]))
    path_costs = V_fw_sources + edge_costs + V_bw_targets
    edge_weights = np.exp(-path_costs)

    # Check for invalid values (NaN or Inf) in weights that might arise from V_fw/V_bw
    # Treat NaN/Inf weights as zero probability contribution.
    edge_weights = np.nan_to_num(edge_weights, nan=0.0, posinf=0.0, neginf=0.0)

    # Calculate the normalization factor Z = exp(-V_bw[overall_source_node])
    # This is the total likelihood sum over all paths s -> g
    total_likelihood_Z = np.exp(-V_bw[overall_source_node])
    total_likelihood_Z = np.nan_to_num(total_likelihood_Z, nan=0.0, posinf=0.0, neginf=0.0) # Handle V_bw[s] = inf

    # Initialize likelihoods
    edge_likelihoods = np.zeros_like(edge_weights)

    # Normalize edge weights by the total likelihood Z
    if total_likelihood_Z > 0:
        edge_likelihoods = edge_weights / total_likelihood_Z
    # If total_likelihood_Z is 0 (goal unreachable), likelihoods remain 0.

    # Create the sparse likelihood matrix L[j, i] = likelihood of traversal j -> i
    L = sparse.csr_matrix((edge_likelihoods, (sources, targets)), shape=A.shape)

    return L

import numpy as np
from scipy import sparse
import warnings

def sample_boltzmann_path(A: sparse.csr_matrix, costs: sparse.csr_matrix,
                          V_bw: np.ndarray, source_node: int, goal_node: int,
                          max_path_len: int = None) -> list[int]:
    """
    Samples a single path from source_node to goal_node based on Boltzmann
    probabilities derived from edge costs and the backward value function V_bw.

    Transition probability P(next=i | current=j) is proportional to
    exp(-(cost(j, i) + V_bw[i])).

    Parameters:
    -----------
    A : sparse.csr_matrix
        Adjacency matrix (A[j, i] = 1 for edge j -> i). Assumed CSR format.
    costs : sparse.csr_matrix
        Cost matrix (costs[j, i] = cost of edge j -> i). Assumed CSR format.
        Must have the same sparsity pattern and CSR data ordering as A.
    V_bw : np.ndarray
        Backward value function (-log(gamma), where gamma is the sum of exp(-cost)
        for paths from node i to the goal). Shape (N,).
    source_node : int
        The index of the starting node for the path.
    goal_node : int
        The index of the target node for the path.
    max_path_len : int, optional
        Maximum allowed length of the sampled path. If None, defaults to A.shape[0].
        Helps prevent infinite loops in case of cycles or unreachable goals.

    Returns:
    --------
    list[int]
        A list of node indices representing the sampled path from source to goal.
        Returns a partial path if the goal is not reached within max_path_len
        or if a dead end (node with no viable outgoing transitions) is encountered.

    Raises:
    -------
    ValueError
        If A and costs shapes don't match or node indices are invalid.
        If the source node is the goal node.
    """
    N = A.shape[0]
    if A.shape != costs.shape:
        raise ValueError("Adjacency and cost matrix shapes must match.")
    if not (0 <= source_node < N and 0 <= goal_node < N):
        raise ValueError("Source or goal node index out of bounds.")
    if source_node == goal_node:
        # Handle this case explicitly, maybe return [source_node] or raise error
        warnings.warn("Source node is the same as goal node.")
        return [source_node]
    if V_bw.shape != (N,):
        raise ValueError("V_bw shape must match matrix dimension.")

    if max_path_len is None:
        max_path_len = N # A simple path won't have more than N nodes

    # Ensure CSR format for efficient row slicing
    if not sparse.isspmatrix_csr(A):
        A = A.tocsr()
    if not sparse.isspmatrix_csr(costs):
        costs = costs.tocsr()

    path = [source_node]
    current_node = source_node

    for _ in range(max_path_len):
        if current_node == goal_node:
            break # Successfully reached the goal

        # Find neighbors and corresponding edge costs efficiently using CSR properties
        # This relies on A and costs having the *exact* same sparsity pattern
        # and internal data ordering for corresponding edges.
        start_ptr = A.indptr[current_node]
        end_ptr = A.indptr[current_node + 1]

        neighbors = A.indices[start_ptr:end_ptr]
        edge_costs_out = costs.data[start_ptr:end_ptr] # Corresponding costs

        if len(neighbors) == 0:
            warnings.warn(f"Node {current_node} has no outgoing edges. Path terminated prematurely.")
            break # Dead end

        # Get V_bw values for neighbors
        V_bw_neighbors = V_bw[neighbors]

        # Calculate unnormalized weights: exp(-(cost + V_bw))
        # Use np.exp directly, handle potential infinities in V_bw
        log_weights = -(edge_costs_out + V_bw_neighbors)

        # --- Numerical Stability Trick ---
        # Subtract max(log_weights) before exponentiating to prevent overflow
        # and handle underflow gracefully (weights become close to 0).
        # This doesn't change the resulting probabilities.
        max_log_weight = np.max(log_weights[np.isfinite(log_weights)]) if np.any(np.isfinite(log_weights)) else 0
        weights = np.exp(log_weights - max_log_weight)
        # Set weights corresponding to infinite V_bw (unreachable goal) to 0
        weights[~np.isfinite(log_weights)] = 0.0
        #--------------------------------

        total_weight = np.sum(weights)

        if total_weight <= 0 or not np.isfinite(total_weight):
             warnings.warn(f"No viable path from node {current_node} towards goal {goal_node} (all paths have zero or infinite weight). Path terminated prematurely.")
             break # Dead end towards the goal

        # Calculate probabilities
        probabilities = weights / total_weight
        # Clean up potential small numerical errors leading to non-sum-to-1
        probabilities = probabilities / np.sum(probabilities)

        # Sample the *index* of the next node within the neighbors list
        chosen_neighbor_index = np.random.choice(len(neighbors), p=probabilities)

        # Get the actual node ID
        next_node = neighbors[chosen_neighbor_index]

        path.append(next_node)
        current_node = next_node

    else: # Loop finished without break (max_path_len reached)
         warnings.warn(f"Path sampling terminated after reaching max_path_len ({max_path_len}) before reaching goal.")

    return path

from collections import Counter

def count_link_probabilities(paths):
    # Extract edges from all paths and count occurrences
    edge_counter = Counter()
    
    for path in paths:
        # Create pairs of consecutive nodes (edges)
        edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        # Use a set to ensure each edge is counted only once per path
        unique_edges = set(edges)
        for edge in unique_edges:
            edge_counter[edge] += 1
    
    return dict(edge_counter)

def compute_empirical_edge_traversals(demonstrations: list[list[int]], A: sparse.csr_matrix):
    """
    Computes the empirical traversal frequency for each edge based on demonstrations.

    Parameters:
    -----------
    demonstrations : list[list[int]]
        A list of paths, where each path is a list of node indices.
    A : sparse.csr_matrix
        The adjacency matrix of the graph, used to determine the shape of the output matrix.

    Returns:
    --------
    sparse.csr_matrix
        A sparse matrix L where L[j, i] contains the empirical traversal frequency
        (count / number of demonstrations) for the edge j -> i.
    """
    if not demonstrations:
        # Return an empty sparse matrix of the correct shape if no demonstrations provided
        return sparse.csr_matrix(A.shape)

    N = A.shape[0]
    empirical_link_counts = Counter()
    empirical_number_of_paths = len(demonstrations)

    for path in demonstrations:
        # Create pairs of consecutive nodes (edges)
        edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        # Use a set to ensure each edge is counted only once per path
        unique_edges = set(edges)
        for edge in unique_edges:
            # Check if node indices are within bounds
            u, v = edge
            if not (0 <= u < N and 0 <= v < N):
                 warnings.warn(f"Edge {edge} in demonstration contains node index out of bounds (N={N}). Skipping.")
                 continue
            empirical_link_counts[edge] += 1

    # Prepare data for sparse matrix construction
    row_indices = []
    col_indices = []
    data = []

    for edge, count in empirical_link_counts.items():
        u, v = edge
        frequency = count / empirical_number_of_paths
        row_indices.append(u)
        col_indices.append(v)
        data.append(frequency)

    # Create the sparse matrix
    L_empirical = sparse.csr_matrix((data, (row_indices, col_indices)), shape=(N, N))

    return L_empirical

def fit(A: sparse.csr_matrix,
                          max_iter_for_vi: int = 100, tol_for_vi: float = 1e-3,
                          max_iter_gd: int = 10_000, tol_gd: float = 5e-3,
                          learning_rate: float = 0.01,
                          demonstrations: list[list[int]] = None,
                          intrinsic_costs: sparse.csr_matrix = None,
                          edge_specific_costs: dict = None, # {'preference': (edge_pref_matrix, 1.0)}
                          shared_costs: dict = None, # {'congestion': (node_congestion, breakpoints, slopes, intercepts, 1.0)}
                          source_node: int = None,
                          goal_node: int = None):
    
    # Ensure source_node and goal_node are defined
    if source_node is None or goal_node is None:
        raise ValueError("source_node and goal_node must be defined")
    
    if demonstrations is None:
        raise ValueError("demonstrations must be defined")
    
    # Compute the empirical traversals of the edges (only needed to be computed once)
    # Now returns a sparse matrix L_empirical[j, i] = empirical frequency of edge j->i
    empirical_traversals_matrix = compute_empirical_edge_traversals(demonstrations, A)
    
    for iter_gd in range(max_iter_gd):
        # Compute the cost matrix
        costs = get_cost_matrix(A, intrinsic_costs, edge_specific_costs, shared_costs)

        # Compute V_fw and V_bw
        V_fw = bellman_fw(A, costs, source_node, max_iter=max_iter_for_vi, tol=tol_for_vi)
        V_bw = bellman_bw(A, costs, goal_node, max_iter=max_iter_for_vi, tol=tol_for_vi)
    
        # Compute expected edge traversals
        expected_traversals_matrix = compute_edge_traversal_likelihoods(A, costs, V_fw, V_bw, source_node)
        
        # Compute the gradient of the cost function
        gradient_matrix = - expected_traversals_matrix + empirical_traversals_matrix
        
        # Assuming linear cost function in terms of edge cost components
        for prop_name, (prop_matrix, prop_weight) in edge_specific_costs.items():
            prop_matrix.data -= learning_rate * gradient_matrix.data * prop_weight
            
        # TODO: Implement gradient descent for shared costs (will be done later)
        
        # Check convergence by computing the maximum absolute value in the gradient matrix
        if sparse.issparse(gradient_matrix):
            gradient_norm = np.max(np.abs(gradient_matrix.data))
        else:
            gradient_norm = np.max(np.abs(gradient_matrix))
            
        print(f"Iteration {iter_gd}: Gradient norm = {gradient_norm:.6f}")
        
        if gradient_norm < tol_gd:
            print(f"Converged after {iter_gd+1} iterations. Gradient norm: {gradient_norm:.6f}")
            break
            
        if iter_gd == max_iter_gd - 1:
            print(f"Warning: Maximum iterations ({max_iter_gd}) reached without convergence. Final gradient norm: {gradient_norm:.6f}")

    return edge_specific_costs, shared_costs
        
