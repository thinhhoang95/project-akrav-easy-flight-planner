import networkx as nx
from lateral.splicer import splicer

def networkx_to_splicer(nx_graph, latlon_coeff = 1.0):
    """
    Convert a networkx graph with string waypoint names into a splicer Graph 
    (using integer IDs). Assumes each node has 'lat' and 'lon' attributes.
    
    Parameters:
      nx_graph (networkx.Graph): A NetworkX graph with nodes as strings and 
                                 attributes 'lat' and 'lon'. Edge distances are
                                 taken from the 'weight' attribute, or default to 1.0.
    
    Returns:
      splicer_graph (planner.Graph): A splicer graph with integer-based node IDs.
      name_to_id (dict): Mapping from the original string waypoint names to integer IDs.
    """
    splicer_graph = splicer.Graph()
    name_to_id = {}
    
    # Assign an integer ID for each node in the networkx graph.
    for idx, node in enumerate(nx_graph.nodes()):
        name_to_id[node] = idx
        # Retrieve coordinates from node attributes; default to (0.0, 0.0) if not provided.
        node_attrs = nx_graph.nodes[node]
        lat = node_attrs.get("row", 0.0) * latlon_coeff
        lon = node_attrs.get("col", 0.0) * latlon_coeff
        splicer_graph.add_node(idx, lat, lon)
        print(f"Added node {idx} at {lat}, {lon}")
    
    # Add edges to the splicer graph using the new integer IDs.
    for u, v, data in nx_graph.edges(data=True):
        # Use the 'weight' attribute for the distance if available; default to 1.0.
        distance = data.get("weight", 1.0)
        splicer_graph.add_edge(name_to_id[u], name_to_id[v], distance)
        print(f"Added edge {name_to_id[u]} -> {name_to_id[v]} with distance {distance}")
    
    return splicer_graph, name_to_id
