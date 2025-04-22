import networkx as nx
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from context import GraphRouteContext


def get_adjacency_matrix(G: nx.DiGraph) -> np.ndarray:
    """
    Get the adjacency matrix of a directed graph.
    """
    return nx.to_numpy_array(G, dtype=int)
