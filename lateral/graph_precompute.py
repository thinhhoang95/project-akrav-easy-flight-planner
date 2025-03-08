import networkx as nx
import multiprocessing
from tqdm import tqdm

def process_source(args):
    source, G, max_depth = args
    local_distance_dict = {}
    
    # Compute all shortest paths from this source node using Dijkstra’s algorithm
    try:
        distances, paths = nx.single_source_dijkstra(G, source, weight='distance')
    except Exception:
        return local_distance_dict

    for target, path in paths.items():
        if source == target:
            continue
        # Only consider paths with number of hops <= max_depth
        if len(path) - 1 > max_depth:
            continue
        
        # Precompute cumulative distances along the path:
        # cum_dist[i] is the total distance from path[0] to path[i].
        cum_dist = [0]
        for i in range(1, len(path)):
            edge_weight = G.edges[path[i-1], path[i]].get('distance', 1)
            cum_dist.append(cum_dist[-1] + edge_weight)
        
        # Update the local dictionary for every subpath of this shortest path.
        for i in range(len(path)):
            for j in range(i+1, len(path)):
                key = f"{path[i]}_{path[j]}"
                subpath_distance = cum_dist[j] - cum_dist[i]
                if key not in local_distance_dict or subpath_distance < local_distance_dict[key]:
                    local_distance_dict[key] = subpath_distance
                    # For undirected graphs, store the reverse key as well.
                    if not G.is_directed():
                        local_distance_dict[f"{path[j]}_{path[i]}"] = subpath_distance
    return local_distance_dict

def precompute_distance_graph_parallel(G, max_depth=4):
    # Prepare a list of tasks: each task is a tuple (source, G, max_depth)
    tasks = [(source, G, max_depth) for source in G.nodes()]
    
    combined_distance_dict = {}
    # Use a multiprocessing Pool to distribute the tasks and wrap the imap in a TQDM progress bar.
    # Get number of cores
    num_cores = multiprocessing.cpu_count()
    print(f"Using {num_cores} cores")
    with multiprocessing.Pool(processes=num_cores - 1) as pool:
        for local_dict in tqdm(pool.imap(process_source, tasks), total=len(tasks), desc=f"Processing nodes"):
            for key, distance in local_dict.items():
                if key not in combined_distance_dict or distance < combined_distance_dict[key]:
                    combined_distance_dict[key] = distance
    return combined_distance_dict