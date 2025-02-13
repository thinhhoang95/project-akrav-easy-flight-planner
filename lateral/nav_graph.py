import networkx as nx
from math import radians, sin, cos, sqrt, atan2
from tqdm import tqdm
from utils.sid_star_reader import load_procedures
import os, sys
from utils.haversine import haversine_distance
from utils.sid_star_reader import load_procedures

PROJECT_ROOT = os.getenv('PROJECT_ROOT')
sys.path.append(PROJECT_ROOT)

import networkx as nx
from tqdm import tqdm
import os

def is_in_europe(lat, lon):
    """
    Checks if a given latitude and longitude fall within an approximate bounding
    box for the European continent.
    
    Europe is approximated as:
      Latitude: 30°N to 72°N
      Longitude: -25°E to 45°E
    """
    return 30 <= lat <= 72 and -25 <= lon <= 45

def convert_coord_for_fra_df(coord_str):
        """Convert coordinates from format like 'N404519' or 'E0183830' to decimal degrees"""
        try:
            direction = coord_str[0]
            degrees = float(coord_str[1:-4])
            decimals = float(coord_str[-4:]) / 10000
            decimal = round(degrees + decimals, 4)
            return decimal if direction in ['N', 'E'] else -decimal
        except (ValueError, IndexError):
            return None

def point_line_distance(point_lat, point_lon, line_start_lat, line_start_lon, line_end_lat, line_end_lon):
    # Calculate distance from point to great circle path
    # Using cross track distance formula
    d13 = haversine_distance(line_start_lat, line_start_lon, point_lat, point_lon)
    bearing13 = atan2(sin(radians(point_lon - line_start_lon)) * cos(radians(point_lat)),
                    cos(radians(line_start_lat)) * sin(radians(point_lat)) - 
                    sin(radians(line_start_lat)) * cos(radians(point_lat)) * 
                    cos(radians(point_lon - line_start_lon)))
    bearing12 = atan2(sin(radians(line_end_lon - line_start_lon)) * cos(radians(line_end_lat)),
                    cos(radians(line_start_lat)) * sin(radians(line_end_lat)) - 
                    sin(radians(line_start_lat)) * cos(radians(line_end_lat)) * 
                    cos(radians(line_end_lon - line_start_lon)))
    return abs(d13 * sin(bearing13 - bearing12))

def load_ats_graph(origin_lat, origin_lon, dest_lat, dest_lon, max_dist_from_gcp = 100,
                   ats_graph = None, ats_graph_path = None):
    """
    Load the ATS route graph from a GraphML file
    Args:
        graph_path: Path to the GraphML file
    Returns:
        NetworkX DiGraph object containing the ATS route network
    """
    if ats_graph is None:
        if ats_graph_path is None:
            raise ValueError("Either ats_graph or ats_graph_path must be provided")
        ats_graph = nx.read_graphml(ats_graph_path)

    print(f'Subsetting the ATS graph to the great circle path between origin and destination...')
    # Create a new graph for the subset
    subset_graph = nx.DiGraph()

    # For each node in the original graph
    for node in tqdm(ats_graph.nodes(), desc="Adding nodes to subset"):
        node_lat = float(ats_graph.nodes[node]['lat'])
        node_lon = float(ats_graph.nodes[node]['lon'])
        
        # Calculate distance from node to great circle path
        dist_to_path = point_line_distance(node_lat, node_lon,
                                         origin_lat, origin_lon, 
                                         dest_lat, dest_lon)
        
        # If node is within max_dist_from_gcp of path, add it to subset
        if dist_to_path <= max_dist_from_gcp:
            subset_graph.add_node(node, **ats_graph.nodes[node])

    # Add relevant edges
    for edge in tqdm(ats_graph.edges(data=True), desc="Adding edges to subset"):
        source, target, data = edge
        # If both nodes are in subset, add the edge
        if source in subset_graph and target in subset_graph:
            subset_graph.add_edge(source, target, **data)

    ats_graph = subset_graph

    print(f'ATS graph loaded. Nodes: {ats_graph.number_of_nodes()}, edges: {ats_graph.number_of_edges()}')
    
    return ats_graph

import pandas as pd
import random

def generate_random_two_digits():
    return str(random.randint(0, 99)).zfill(2)

def add_node_with_refs(G, node_id, lat, lon, type='FRA'):
    if node_id in G.nodes:
        # Fix already exists, check if coordinates match
        if abs(G.nodes[node_id]['lat'] - lat) > 1e-4 or abs(G.nodes[node_id]['lon'] - lon) > 1e-4:
            # print(f"Fix {node_id} has different coordinates: {G.nodes[node_id]['lat']}, {G.nodes[node_id]['lon']} != {lat}, {lon}")
            original_node_id = node_id
            node_id = node_id + '_' + generate_random_two_digits()
            G.add_node(node_id, lat=lat, lon=lon, type=type, refs='')
            # Modify the refs of the original node
            G.nodes[original_node_id]['refs'] = G.nodes[original_node_id]['refs'] + f'{node_id}, '
        return node_id
    else:
        G.add_node(node_id, lat=lat, lon=lon, type=type, refs='')
        return node_id
    
def collapse_duplicate_nodes(G):
    """
    Collapses nodes that reference each other and have virtually identical coordinates.
    
    Args:
        G (networkx.Graph): Input graph
        
    Returns:
        networkx.Graph: Graph with duplicate nodes collapsed
    """
    # Copy the graph first
    G = G.copy()
    nodes_removed = 0
    
    # Make a copy to avoid modifying graph during iteration
    nodes = list(G.nodes(data=True))
    
    # Track nodes that have been merged to avoid repeat processing
    merged = set()
    
    for node, data in nodes:
        if node in merged:
            continue
            
        # Skip if no refs field or empty
        if 'refs' not in data or not data['refs']:
            continue
            
        # Get referenced nodes
        refs = data['refs'].split(',')
        refs = [r.strip() for r in refs if r.strip()]
        # Remove empty strings from refs
        refs = [r for r in refs if r]
        # Add the current node to refs for consideration
        refs.append(node)
            
        # Get coordinates for all related refs
        ref_coords = {}
        for r in refs:
            if r in G and 'lat' in G.nodes[r] and 'lon' in G.nodes[r]:
                ref_coords[r] = (float(G.nodes[r]['lat']), float(G.nodes[r]['lon']))
        
        # Group refs by identical coordinates
        coord_groups = {}
        for r, coords in ref_coords.items():
            found = False
            for group_coords, group_refs in coord_groups.items():
                if (abs(coords[0] - group_coords[0]) < 1e-4 and 
                    abs(coords[1] - group_coords[1]) < 1e-4):
                    group_refs.append(r)
                    found = True
                    break
            if not found:
                coord_groups[coords] = [r]
        
        # Keep only first ref from each coordinate group
        refs_to_keep = list(set([group[0] for group in coord_groups.values()]))
        print(f'Refs to keep: {refs_to_keep}')
        refs_to_remove = list(set([r for r in refs if r not in refs_to_keep]))
        
        for refrm in refs_to_remove:
            print(f'Attempting to remove {refrm}')
            print(f'Refs to remove: {refs_to_remove}')
            # Find the ref in refs_to_keep that have the same coordinates
            # Find ref with matching coordinates in refs_to_keep
            ref_coords = (float(G.nodes[refrm]['lat']), float(G.nodes[refrm]['lon']))
            for refkeep in refs_to_keep:
                keep_coords = (float(G.nodes[refkeep]['lat']), float(G.nodes[refkeep]['lon']))
                if (abs(ref_coords[0] - keep_coords[0]) < 1e-4 and 
                    abs(ref_coords[1] - keep_coords[1]) < 1e-4):
                    rk = refkeep
                    break
            # Redirect all edges from ref node to original node
            for pred in G.predecessors(refrm):
                edge_data = G.get_edge_data(pred, refrm)
                G.add_edge(pred, rk, **edge_data)
                
            for succ in G.successors(refrm):
                edge_data = G.get_edge_data(refrm, succ)
                G.add_edge(rk, succ, **edge_data)
            
            # Remove the duplicate node
            print(f'Removing node {refrm}')
            G.remove_node(refrm)

            merged.add(refrm)
            nodes_removed += 1
                
    print(f"Removed {nodes_removed} duplicate nodes")

    print(f'Revising refs properties...')
    # Revise the refs property
    # Update refs property to only include existing nodes
    for node in G.nodes():
        refs_str = G.nodes[node].get('refs', '')
        if refs_str:
            # Split refs string into list and remove empty strings
            refs = [r.strip() for r in refs_str.split(',') if r.strip()]
            
            # Filter to only keep refs that exist in graph
            existing_refs = [r for r in refs if r in G.nodes]
            
            # Update the refs property with filtered list
            G.nodes[node]['refs'] = ', '.join(existing_refs) if existing_refs else ''
    
    return G

def create_edge_with_closest_refs(graph_ats, from_fix, to_fix, threshold = 100, type='DCT'):
    if graph_ats.has_edge(from_fix, to_fix):
        return
    from_fix_lat = graph_ats.nodes[from_fix]['lat']
    from_fix_lon = graph_ats.nodes[from_fix]['lon']
    to_fix_lat = graph_ats.nodes[to_fix]['lat']
    to_fix_lon = graph_ats.nodes[to_fix]['lon']
    distance = haversine_distance(from_fix_lat, from_fix_lon, to_fix_lat, to_fix_lon)
    
    # print(f'DCT link between {from_fix} and {to_fix} is {distance:.2f}nm. Considering alternatives...')
    if distance < threshold:
        if not graph_ats.has_edge(from_fix, to_fix):
            graph_ats.add_edge(from_fix, to_fix,
                            distance=distance,
                            airway='',
                            edge_type=type)
        return # do nothing if the distance is less than the threshold
    
    from_fix_refs = graph_ats.nodes[from_fix]['refs']
    to_fix_refs = graph_ats.nodes[to_fix]['refs']
    # Split refs strings into lists
    from_fix_refs = from_fix_refs.split(',') if from_fix_refs else []
    to_fix_refs = to_fix_refs.split(',') if to_fix_refs else []
    if len(from_fix_refs) == 0 and len(to_fix_refs) == 0:
        # No refs, add the edge directly
        if not graph_ats.has_edge(from_fix, to_fix):
            graph_ats.add_edge(from_fix, to_fix,
                            distance=distance,
                            airway='',
                            edge_type=type)
        return
    # Add the current fixes to the refs
    from_fix_refs.append(from_fix)
    to_fix_refs.append(to_fix)
    
    # Find the closest pair of refs
    min_distance = float('inf')
    best_from_ref = None 
    best_to_ref = None
    
    for from_ref in from_fix_refs:
        if from_ref not in graph_ats.nodes:
            continue
        for to_ref in to_fix_refs:
            if to_ref not in graph_ats.nodes:
                continue
            # Get coordinates
            from_ref_lat = graph_ats.nodes[from_ref]['lat']
            from_ref_lon = graph_ats.nodes[from_ref]['lon']
            to_ref_lat = graph_ats.nodes[to_ref]['lat']
            to_ref_lon = graph_ats.nodes[to_ref]['lon']
            
            # Calculate simple distance (absolute difference)
            dist = abs(from_ref_lat - to_ref_lat) + abs(from_ref_lon - to_ref_lon)
            
            if dist < min_distance:
                min_distance = dist
                best_from_ref = from_ref
                best_to_ref = to_ref

    # Recalculate the distance in nm
    best_min_distance = haversine_distance(
        graph_ats.nodes[best_from_ref]['lat'], 
        graph_ats.nodes[best_from_ref]['lon'], 
        graph_ats.nodes[best_to_ref]['lat'], 
        graph_ats.nodes[best_to_ref]['lon']
    )
    
    # Create edge between closest refs if found
    if best_from_ref and best_to_ref:
        if not graph_ats.has_edge(best_from_ref, best_to_ref):
            graph_ats.add_edge(best_from_ref, best_to_ref,
                            distance=best_min_distance,
                            airway='',
                            edge_type=type)
            # print(f'Established link between {best_from_ref} and {best_to_ref} instead. New distance is {best_min_distance}')

    return

def load_fra_graph_into_ats_graph(ats_graph, fra_df = None, fra_df_path = None,
                                  max_dist_from_gcp = 100, origin_lat = None, origin_lon = None,
                                  dest_lat = None, dest_lon = None):
    
    ats_graph = ats_graph.copy()

    print(f'Building FRA routing options...')
    # Load the FRA graph from a GraphML file
    if fra_df is None:
        if fra_df_path is None:
            raise ValueError("Either fra_graph or fra_graph_path must be provided")
        fra_df = pd.read_csv(fra_df_path)
        # columns of fra_df: chg_rec,pt_type,fra_pt,fra_lat,fra_lon,fra_name,fra_rel_enroute,fra_rel_arr_dep,arr_apt,dep_apt,flos,lvl_avail,time_avail,loc_ind,rmk

    # Uppercase and strip the fra_name column to standardize the names
    fra_df['fra_name'] = fra_df['fra_name'].str.upper().str.strip()

    # Convert fra_lat and fra_lon to decimal degrees
    fra_df['fra_lat'] = fra_df['fra_lat'].apply(convert_coord_for_fra_df)
    fra_df['fra_lon'] = fra_df['fra_lon'].apply(convert_coord_for_fra_df)
    
    # Filter out only FRA points within 100nm of the great circle path between origin and destination
    fra_df['dist_to_path'] = fra_df.apply(
        lambda row: point_line_distance(
            row['fra_lat'], row['fra_lon'],
            origin_lat, origin_lon,
            dest_lat, dest_lon
        ), axis=1
    )
    
    # Filter to keep only points within max_dist_from_gcp
    fra_df = fra_df[fra_df['dist_to_path'] <= max_dist_from_gcp]

    print(f'Found {len(fra_df)} FRA points within {max_dist_from_gcp}nm of the great circle path between origin and destination.\
          \nMerging these into the ATS graph...')
    
    fra_names = fra_df['fra_name'].unique()

    for fra in fra_names:
        fra_df_subset = fra_df[fra_df['fra_name'] == fra]
        # Add all FRA points from this subset to the graph if they don't exist
        fra_map = {} # stores mapping ABADI -> ABADI_01 if they are not unique
        for _, row in fra_df_subset.iterrows():
            node_id = row['fra_pt']
            renamed_id = add_node_with_refs(ats_graph, node_id, row['fra_lat'], row['fra_lon'], type='FRA')
            if renamed_id != node_id:
                fra_map[node_id] = renamed_id

        print(f'{len(fra_map)} FRA points renamed for {fra}')

        # Get all FRA points for this name and their types
        fra_points = [(row['fra_pt'], row['fra_rel_enroute']) 
                     for _, row in fra_df_subset.iterrows()]

        # Rename fra_points using fra_map if any points were renamed
        # fra_points = [(fra_map.get(point, point), type_) 
        #              for point, type_ in fra_points]

        # Connect points according to rules
        for point1, type1 in tqdm(fra_points, desc=f"{fra}"):
            for point2, type2 in fra_points:
                # Skip self-connections
                if point1 == point2:
                    continue
                    
                # Rule 2: EX can connect to X, EX or I
                if type1 == 'EX' and type2 in ['X', 'EX', 'I']:
                    create_edge_with_closest_refs(ats_graph, point1, point2, threshold=150, type='FRA')
                
                # Rule 3: E can connect to X, EX or I  
                elif type1 == 'E' and type2 in ['X', 'EX', 'I']:
                    create_edge_with_closest_refs(ats_graph, point1, point2, threshold=150, type='FRA')
                
                # Rule 4: I can connect bilaterally to I
                elif type1 == 'I' and type2 == 'I':
                    create_edge_with_closest_refs(ats_graph, point1, point2, threshold=150, type='FRA')
    
    print(f'FRA graph merged into ATS graph. Nodes: {ats_graph.number_of_nodes()}, edges: {ats_graph.number_of_edges()}')
    return ats_graph

def route_graph_subset(route_graph, icao_origin, icao_dest,
                       origin_lat, origin_lon, dest_lat, dest_lon,
                       w_dct = 1.0, origin_runway = '0', destination_runway = '0',
                       use_sid_star = True, w_proc = 0.2, w_fra=1.0):

    # We get the relevant part of the graph (nodes and edges)
    # that locate within 500nm from the great circle path between origin and destination

    # Create a new graph for the subset
    subset_graph = nx.DiGraph()


    print(f'Computing cost for the ATS-FRA route graph...')
    # For each node in the original graph
    for node in tqdm(route_graph.nodes(), desc="Adding nodes to subset"):
        node_lat = float(route_graph.nodes[node]['lat'])
        node_lon = float(route_graph.nodes[node]['lon'])
        subset_graph.add_node(node, **route_graph.nodes[node])

    # Add relevant edges
    for edge in tqdm(route_graph.edges(data=True), desc="Adding edges to subset"):
        source, target, data = edge
        # If both nodes are in subset, add the edge
        if source in subset_graph and target in subset_graph:
            # Compute cost of edge
            if data['edge_type'] == 'DCT':
                data['cost'] = data['distance'] * w_dct
            elif data['edge_type'] == 'FRA':
                data['cost'] = data['distance'] * w_fra
            else:
                data['cost'] = data['distance']
            subset_graph.add_edge(source, target, **data)

    # We add the origin and destination nodes to the subset
    subset_graph.add_node(icao_origin, lat=origin_lat, lon=origin_lon)
    subset_graph.add_node(icao_dest, lat=dest_lat, lon=dest_lon)

    sid_avail = True
    star_avail = True

    if use_sid_star:
        print(f'Adding SID and STAR graphs to the subset...')
        # We add the SID and STAR graphs to the subset
        # Read the procedures from the file
        sid_file = os.path.join(PROJECT_ROOT, 'data', 'airac', 'proc', f'{icao_origin}.txt')
        star_file = os.path.join(PROJECT_ROOT, 'data', 'airac', 'proc', f'{icao_dest}.txt')
        # If file exists, read the procedures
        if os.path.exists(sid_file):
            data = open(sid_file, 'r').read()
            subset_graph, sid_avail = load_procedures(data, icao_origin, origin_runway, subset_graph, procedure='SID', w_proc=w_proc)
        if os.path.exists(star_file):
            data = open(star_file, 'r').read()
            subset_graph, star_avail = load_procedures(data, icao_dest, destination_runway, subset_graph, procedure='STAR', w_proc=w_proc)


    # Haversine distance between origin and destination
    dist_origin_dest = haversine_distance(origin_lat, origin_lon, dest_lat, dest_lon)

    # We add the edges to the origin and destination nodes
    # Add edges to/from origin and destination for nodes within 40nm
    if not use_sid_star and sid_avail and star_avail:
        for node in tqdm(subset_graph.nodes(), desc="Adding edges to/from origin and destination"):
            if node not in [icao_origin, icao_dest]:
                # Check distance to origin
                node_lat = float(subset_graph.nodes[node]['lat'])
                node_lon = float(subset_graph.nodes[node]['lon'])
                
                dist_to_origin = haversine_distance(origin_lat, origin_lon, node_lat, node_lon)
                if dist_to_origin <= 40:
                    subset_graph.add_edge(icao_origin, node, 
                                        edge_type='DCT', airway='', cost=dist_to_origin)
                    # subset_graph.add_edge(node, self.origin_icao, edge_type='DCT', airway='')
                

                # Check distance to destination    
                dist_to_dest = haversine_distance(dest_lat, dest_lon, node_lat, node_lon)
                if dist_to_dest <= 40:
                    subset_graph.add_edge(node, icao_dest, 
                                          edge_type='DCT', airway='', cost=dist_to_dest)
                    # subset_graph.add_edge(icao_dest, node)

    print(f'Route graph subset created. Nodes: {subset_graph.number_of_nodes()}, edges: {subset_graph.number_of_edges()}')
    print(f'Great circle distance between origin and destination: {dist_origin_dest:.2f} nm')

    return subset_graph, dist_origin_dest

def generate_navigraph(icao_origin, icao_dest, origin_lat, origin_lon, dest_lat, dest_lon,
                       origin_runway, destination_runway, w_dct = 1.0, w_fra = 1.0, w_proc = 0.2):
    ats_graph = load_ats_graph(origin_lat, origin_lon, dest_lat, dest_lon,
                           ats_graph_path=os.path.join(PROJECT_ROOT, "data", "graphs", "route_graph_dct3.graphml"))
    
    ats_fra_graph = load_fra_graph_into_ats_graph(ats_graph,
                                              fra_df_path=os.path.join(PROJECT_ROOT, "data", "rad", "FRA_PTS.csv"),
                                              origin_lat=origin_lat, origin_lon=origin_lon,
                                              dest_lat=dest_lat, dest_lon=dest_lon)
    
    route_graph, gcd = route_graph_subset(ats_fra_graph, icao_origin, icao_dest,
                                  origin_lat=origin_lat, origin_lon=origin_lon,
                                  dest_lat=dest_lat, dest_lon=dest_lon,
                                  origin_runway=origin_runway, destination_runway=destination_runway,
                                  use_sid_star=True, w_dct=w_dct, w_fra=w_fra, w_proc=w_proc)

    return route_graph



def plan(search_graph, icao_origin, icao_dest):
    if search_graph is None:
        raise ValueError("Route graph not found. Please call route_graph_subset first.")

    # Get the shortest path in the search graph
    shortest_path = nx.shortest_path(search_graph, source=icao_origin,
                                        target=icao_dest, weight='cost')
    # Convert shortest path into list of dictionaries with edge information
    result = []
    cumulative_cost = 0
    cumulative_distance = 0
    for i in range(len(shortest_path)-1):
        from_node = shortest_path[i]
        to_node = shortest_path[i+1]
        
        # Get node coordinates
        from_lat = float(search_graph.nodes[from_node]['lat'])
        from_lon = float(search_graph.nodes[from_node]['lon'])
        to_lat = float(search_graph.nodes[to_node]['lat'])
        to_lon = float(search_graph.nodes[to_node]['lon'])
        
        # Get edge data
        edge_data = search_graph.edges[from_node, to_node]
        cumulative_cost += edge_data['cost']
        cumulative_distance += edge_data.get('distance', haversine_distance(from_lat, from_lon, to_lat, to_lon))

        result.append({
            'from_node': from_node,
            'to_node': to_node,
            'from_lat': from_lat,
            'from_lon': from_lon,
            'to_lat': to_lat,
            'to_lon': to_lon,
            'distance': edge_data.get('distance', haversine_distance(from_lat, from_lon, to_lat, to_lon)),
            'cost': edge_data['cost'],
            'edge_type': edge_data['edge_type'],
            'airway': edge_data['airway'],
        })
        

    return result, cumulative_cost, cumulative_distance

def plan_e2e(ORIGIN_ICAO, DEST_ICAO, ORIGIN_RWY, DEST_RWY,
             airports_df = None, airports_df_path = None):
    if airports_df is None:
        if airports_df_path is None:
            raise ValueError("Either airports_df or airports_df_path must be provided")
        airports_df = pd.read_csv(airports_df_path)

    # Get the latitude and longitude of the origin and destination
    origin_lat = airports_df[airports_df['icao'] == ORIGIN_ICAO]['latitude'].values[0]
    origin_lon = airports_df[airports_df['icao'] == ORIGIN_ICAO]['longitude'].values[0]
    dest_lat = airports_df[airports_df['icao'] == DEST_ICAO]['latitude'].values[0]
    dest_lon = airports_df[airports_df['icao'] == DEST_ICAO]['longitude'].values[0]
    # Origin and destination airport names
    origin_name = airports_df[airports_df['icao'] == ORIGIN_ICAO]['name'].values[0]
    dest_name = airports_df[airports_df['icao'] == DEST_ICAO]['name'].values[0]

    print(f'Origin: {ORIGIN_ICAO} - {origin_name} ({origin_lat}, {origin_lon}, {ORIGIN_RWY})')
    print(f'Destination: {DEST_ICAO} - {dest_name} ({dest_lat}, {dest_lon}, {DEST_RWY})')

    route_graph = generate_navigraph(ORIGIN_ICAO, DEST_ICAO, origin_lat, origin_lon, dest_lat, dest_lon,
                                    ORIGIN_RWY, DEST_RWY,
                                    w_dct=1.0, w_fra=1.0, w_proc=0.2)

    result, cumulative_cost, cumulative_distance = plan(route_graph, ORIGIN_ICAO, DEST_ICAO)

    return result, cumulative_cost, cumulative_distance


