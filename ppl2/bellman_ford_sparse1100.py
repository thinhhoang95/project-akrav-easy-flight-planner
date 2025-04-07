import numpy as np
from scipy import sparse
import networkx as nx
from typing import Union
from scipy.sparse.linalg import spsolve

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

def compute_edge_transition_probabilities(A: sparse.csr_matrix, costs: sparse.csr_matrix,
                               V_fw: np.ndarray, V_bw: np.ndarray) -> sparse.csr_matrix:
    """
    Computes edge transition probabilities based on forward and backward value functions.

    The probability of transitioning from node j to node i via edge (j, i) is proportional to:
    exp(-(V_fw[j] + cost(j, i) + V_bw[i]))
    where V_fw is the cost from the overall origin and V_bw is the cost to the overall destination.

    The probabilities are normalized such that the sum of probabilities for outgoing edges
    from each node j equals 1 (unless the node has no outgoing paths with finite weight).

    Parameters:
    -----------
    A : sparse.csr_matrix
        Adjacency matrix (A[j, i] = 1 for edge j -> i). Assumed CSR format.
    costs : sparse.csr_matrix
        Cost matrix (costs[j, i] = cost of edge j -> i). Assumed CSR format.
        Must have the same sparsity pattern as A.
    V_fw : np.ndarray
        Forward value function (cost from source to node j). Shape (N,).
    V_bw : np.ndarray
        Backward value function (cost from node i to goal). Shape (N,).

    Returns:
    --------
    sparse.csr_matrix
        A sparse matrix P where P[j, i] contains the normalized probability
        of transitioning from node j to node i. The matrix has the
        same sparsity pattern as A and costs. The rows of P sum to 1
        (except for nodes with no outgoing edges or only infinite cost paths).

    Raises:
    -------
    ValueError
        If A and costs do not have the same shape or sparsity pattern.
        If V_fw or V_bw dimensions do not match the matrix dimensions.
    """
    N = A.shape[0]
    if A.shape != costs.shape:
        raise ValueError(f"Adjacency matrix shape {A.shape} and cost matrix shape {costs.shape} must match.")
    if V_fw.shape != (N,) or V_bw.shape != (N,):
        raise ValueError(f"Value function shapes {V_fw.shape}, {V_bw.shape} must match matrix dimension {N}.")

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
         check_sparsity_pattern_match(A, costs)
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

    # Calculate the sum of weights for edges originating from each source node j
    sum_weights_per_source = np.bincount(sources, weights=edge_weights, minlength=N)

    # Avoid division by zero for nodes with no outgoing edges or only zero-weight edges.
    # Get the sum corresponding to the source node of each edge
    relevant_sums = sum_weights_per_source[sources]

    # Initialize probabilities
    edge_probabilities = np.zeros_like(edge_weights)

    # Calculate probabilities only where the sum of outgoing weights is positive
    valid_indices = relevant_sums > 0
    edge_probabilities[valid_indices] = edge_weights[valid_indices] / relevant_sums[valid_indices]

    # Create the sparse probability matrix P[j, i] = probability of transition j -> i
    P = sparse.csr_matrix((edge_probabilities, (sources, targets)), shape=A.shape)

    return P

def compute_expected_traversals(P: sparse.csr_matrix, source_node: int, goal_node: int):
    """
    Computes the expected number of times each node and edge is traversed
    in a path starting from source_node and ending upon first reaching goal_node.

    Assumes the process stops once the goal_node is reached. Uses the
    fundamental matrix method for absorbing Markov chains.

    Parameters:
    -----------
    P : sparse.csr_matrix
        N x N transition probability matrix where P[j, i] is the probability
        of transitioning from node j to node i. Rows should sum to 1 (or less
        for nodes with no outgoing edges or absorbing states).
    source_node : int
        Index of the starting node (0 to N-1).
    goal_node : int
        Index of the absorbing goal node (0 to N-1).

    Returns:
    --------
    tuple (np.ndarray, sparse.csr_matrix)
        - expected_node_traversals: N-element array where element k is the
          expected number of times node k is visited. E[goal_node] = 1.
        - expected_edge_traversals: N x N sparse matrix where element (j, i)
          is the expected number of times edge j -> i is traversed before
          absorption at the goal node.

    Raises:
    -------
    ValueError
        If source_node or goal_node indices are invalid.
    np.linalg.LinAlgError or similar from spsolve
        If the goal node is unreachable from the source node, potentially
        indicating a singular matrix (I - P_T).
    """
    N = P.shape[0]
    if not (0 <= source_node < N and 0 <= goal_node < N):
        raise ValueError(f"source_node ({source_node}) or goal_node ({goal_node}) index out of bounds for N={N}.")

    # Handle the trivial case where the start is the goal
    if source_node == goal_node:
        expected_node_traversals = np.zeros(N)
        expected_node_traversals[goal_node] = 1.0
        # No edges are traversed if starting at the goal
        expected_edge_traversals = sparse.csr_matrix((N, N), dtype=float)
        return expected_node_traversals, expected_edge_traversals

    # Identify transient nodes (all nodes except the goal_node)
    transient_nodes_mask = np.ones(N, dtype=bool)
    transient_nodes_mask[goal_node] = False
    transient_nodes = np.arange(N)[transient_nodes_mask]

    num_transient = len(transient_nodes)
    if num_transient == 0: # Only the goal node exists? Or N=1 graph?
         expected_node_traversals = np.zeros(N)
         expected_node_traversals[goal_node] = 1.0
         expected_edge_traversals = sparse.csr_matrix((N, N), dtype=float)
         return expected_node_traversals, expected_edge_traversals


    # Create mapping from original node index to transient state index
    node_to_transient_idx = {node: idx for idx, node in enumerate(transient_nodes)}

    # Source node must be a transient state if source != goal
    if source_node not in node_to_transient_idx:
         # This case should not happen based on earlier check
         raise RuntimeError("Internal logic error: source node not found in transient set.")
    s_prime = node_to_transient_idx[source_node] # Index of source within transient states

    # Extract the submatrix P_T for transitions *between* transient states
    # P needs to be CSR/CSC for this type of indexing to work well
    if not sparse.isspmatrix_csr(P) and not sparse.isspmatrix_csc(P):
        P = P.tocsr() # Convert to CSR if not already suitable format

    # Efficiently extract P_T using boolean indexing if possible,
    # otherwise use list indexing (less efficient for large sparse)
    # P[transient_nodes, :][:, transient_nodes] extracts rows then columns
    P_T = P[transient_nodes_mask, :][:, transient_nodes_mask]
    P_T = P_T.tocsr() # Ensure CSR format for calculations

    # Build the matrix M = I - P_T for the fundamental matrix calculation
    I_T = sparse.identity(num_transient, format='csr', dtype=P.dtype)
    M = I_T - P_T

    # Prepare the right-hand side vector b for the linear system.
    # b is a column vector with 1 at the source node's transient index (s_prime).
    b = np.zeros(num_transient, dtype=P.dtype)
    b[s_prime] = 1.0

    # Solve M^T * x = b to find the expected visits vector E_T (as x^T)
    # M^T * E_T^T = e_s'
    # We solve for x = E_T^T
    try:
        # Use CSC format for the solver as it's often preferred for column operations
        E_T_transpose = spsolve(M.tocsc().T, b)
    except Exception as e:
        # Catch potential errors like singularity
        print(f"Error solving linear system with spsolve: {e}")
        print("This might indicate that the goal node is unreachable from the source node "
              "or other numerical issues.")
        raise

    # E_T now holds the expected number of visits for each transient node
    E_T = E_T_transpose # spsolve returns a dense 1D array

    # Assemble the full expected node traversals vector E (size N)
    expected_node_traversals = np.zeros(N, dtype=float)
    expected_node_traversals[goal_node] = 1.0  # Goal node is visited exactly once to end
    expected_node_traversals[transient_nodes] = E_T # Assign computed values

    # --- Compute expected edge traversals ---
    # E_ji = E_j * P_ji, where E_j is expected visits to node j
    # This is calculated as D_E' @ P, where D_E' is diagonal matrix of E
    # with the entry for the goal node set to 0 (no transitions out of goal).

    E_prime = expected_node_traversals.copy()
    E_prime[goal_node] = 0.0  # Zero out visits from the goal state

    # Create the sparse diagonal matrix D_E'
    D_E_prime = sparse.diags(E_prime, offsets=0, shape=(N, N), format='csr', dtype=float)

    # Compute expected edge traversals: D_E' @ P
    expected_edge_traversals = D_E_prime @ P
    # Ensure the result has the correct dtype if P was integer etc.
    expected_edge_traversals = expected_edge_traversals.astype(float)


    return expected_node_traversals, expected_edge_traversals