import numpy as np
from scipy import sparse
import networkx as nx
from typing import Union

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
    >>> costs = compute_edge_costs(G, alpha_pref=1.0)
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
    row_indices = []
    col_indices = []
    adjacency_data = []
    cost_data = []
    
    # Populate the lists
    for u, v in G.edges():
        j = node_to_idx[u]  # Source node index
        i = node_to_idx[v]  # Target node index
        
        # Add to adjacency matrix data
        row_indices.append(j)
        col_indices.append(i)
        adjacency_data.append(1)
        
        # Add to cost matrix data
        cost = 1.0  # Default cost
        if costs_dict and (u, v) in costs_dict:
            cost = costs_dict[(u, v)]
        
        row_indices.append(j)
        col_indices.append(i)
        cost_data.append(cost)
    
    adjacency_matrix = sparse.csr_matrix(
        (adjacency_data, (row_indices[::2], col_indices[::2])),
        shape=(n, n)
    )

    cost_matrix = sparse.csr_matrix(
        (cost_data, (row_indices[1::2], col_indices[1::2])),
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
    
    # Get non-zero indices
    A_rows, A_cols = A.nonzero()
    costs_rows, costs_cols = costs.nonzero()
    
    # Compare the non-zero entry coordinates
    A_indices = set(zip(A_rows, A_cols))
    costs_indices = set(zip(costs_rows, costs_cols))
    
    # Check if the sets are equal
    match = A_indices == costs_indices
    
    if not match:
        # Report the differences
        only_in_A = A_indices - costs_indices
        only_in_costs = costs_indices - A_indices
        print(f"Entries only in A: {len(only_in_A)}")
        print(f"Entries only in costs: {len(only_in_costs)}")
        if len(only_in_A) > 0:
            print(f"Sample entries only in A: {list(only_in_A)[:5]}")
        if len(only_in_costs) > 0:
            print(f"Sample entries only in costs: {list(only_in_costs)[:5]}")
    
    return match

import numpy as np
from scipy import sparse
from typing import Union

def bellman(A: sparse.csr_matrix, costs: sparse.csr_matrix, source_node: Union[int, str],
            max_iter: int = 1000, tol: float = 1e-3):
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
        value/cost associated with each node.
    """
    # Number of nodes
    N = A.shape[0]
    
    # Initialize the value function: all nodes start with infinity except the source
    V_k = np.full(N, np.inf)
    V_k[source_node] = 0.0
    
    # Epsilon for numerical stability in the logarithm computation
    epsilon = 1e-10
    
    # Ensure costs is in CSR format for consistent ordering
    if not sparse.isspmatrix_csr(costs):
        costs = costs.tocsr()
    
    for _ in range(max_iter):
        # print(f"Iteration {_}, V_k[231] = {V_k[231]}")
        # Save the previous value function for convergence checking
        V_prev = V_k.copy()
        
        # Extract edges from the adjacency matrix
        sources, targets = A.nonzero()
        
        # Directly extract the edge costs assuming the structure of A and costs are aligned
        edge_costs = costs.data  
        
        # Compute the exponential update term for each edge:
        # exp(-(V[source] + cost)) for each edge from source -> target
        edge_exp_terms = np.exp(-(V_k[sources] + edge_costs))
        
        # Sum the contributions for each target node using np.bincount
        Sum_Exp_vector = np.bincount(targets, weights=edge_exp_terms, minlength=N)
        
        # Update the value function using the -log transformation
        V_k = -np.log(Sum_Exp_vector + epsilon)
        
        # Re-apply the source condition
        V_k[source_node] = 0.0
        
        # Check for convergence: if maximum change is below the tolerance, stop iterating
        if np.max(np.abs(V_k - V_prev)) < tol:
            break
    
    return V_k

# Example Inputs (Same graph as before, but in sparse format)
# N=4 nodes
V_k = np.array([0.5, 1.0, 1.5, 2.0])
N = len(V_k)

# Edge list representation: (source, target, cost)
# Edges: 0->1 (0.1), 0->2 (0.2)
#        1->2 (0.3), 1->3 (0.4)
#        2->3 (0.5)
#        3->0 (0.6)
sources = np.array([0, 0, 1, 1, 2, 3])
targets = np.array([1, 2, 2, 3, 3, 0])
costs   = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
E = len(sources) # Number of edges

epsilon = 1e-10 # Small value for numerical stability

# --- Vectorized Calculation using Sparse Matrix ---

# 1. Calculate exp(-(V_k[j] + c(j,i))) for each edge
#    First, gather V_k values for the source node of each edge
Vk_j = V_k[sources]  # Shape: (E,)
#    Then, calculate the exponent term for each edge
edge_exp_terms = np.exp(-(Vk_j + costs)) # Shape: (E,)
# print("Edge Exp Terms:", edge_exp_terms)

# 2. Construct the Aggregation Matrix (N x E)
#    Row indices are the target nodes
#    Column indices are the edge indices (0 to E-1)
#    Data is 1 for each entry (we just want to sum)
row_indices = targets
col_indices = np.arange(E)
data = np.ones(E)

# Create a Coordinate format (COO) matrix and convert to CSR for efficiency
Agg_Matrix = sparse.coo_matrix((data, (row_indices, col_indices)), shape=(N, E))
Agg_Matrix_csr = Agg_Matrix.tocsr()
# print("Aggregation Matrix (CSR):\n", Agg_Matrix_csr.toarray()) # Dense view for inspection

# 3. Perform the sum via Matrix-Vector multiplication
#    Agg_Matrix_csr @ edge_exp_terms computes:
#    Sum_Exp[i] = sum_{k where targets[k]==i} (1 * edge_exp_terms[k])
Sum_Exp_vector = Agg_Matrix_csr @ edge_exp_terms # Shape: (N,)
# print("Sum Exp Vector:", Sum_Exp_vector)

# 4. Apply final -log
V_new_sparse = -np.log(Sum_Exp_vector + epsilon)

# --- Verification (Optional: Compare with dense calculation from previous example) ---
# print("\n--- Dense Calculation Result (for comparison) ---")
# A = np.array([
#     [0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1], [1, 0, 0, 0]
# ])
# C = np.zeros((N,N)) # Rebuild cost matrix carefully
# C[sources, targets] = costs # Fill costs based on sparse info
# S = V_k[:, None] + C
# Exp_Terms_All = np.exp(-S)
# Masked_Exp_Terms = A * Exp_Terms_All # Assuming A is correct! Need A[j,i] here!
# A_transpose_for_dense = np.array([ # A[j,i]=1 if j->i
#     [0, 0, 0, 1], # preds of 0 are {3}
#     [1, 0, 0, 0], # preds of 1 are {0}
#     [1, 1, 0, 0], # preds of 2 are {0, 1}
#     [0, 1, 1, 0]  # preds of 3 are {1, 2}
# ]).T # Transpose because my original A was A[i,j] convention? Let's use the one from prev ex.
A_dense = np.array([
    [0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1], [1, 0, 0, 0]
]) # This A is A[j,i]=1 if j->i, matching the sparse definition
C_dense = np.zeros((N,N))
C_dense[sources, targets] = costs
S_dense = V_k[:, None] + C_dense
Exp_Terms_All_dense = np.exp(-S_dense)
Masked_Exp_Terms_dense = A_dense * Exp_Terms_All_dense
Sum_Exp_dense = Masked_Exp_Terms_dense.sum(axis=0)
V_new_dense = -np.log(Sum_Exp_dense + epsilon)
# print("V_new (Dense): ", V_new_dense)
# print("V_new (Sparse):", V_new_sparse)
# print("Match:", np.allclose(V_new_dense, V_new_sparse))

print("V_k:           ", V_k)
print("V_new (Sparse):", V_new_sparse)