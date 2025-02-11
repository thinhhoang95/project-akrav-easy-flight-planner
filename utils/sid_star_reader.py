import networkx as nx
from utils.haversine import haversine_distance

def load_procedures(data, airport_node, runway, G, w_proc = 0.2, procedure='SID'):
    """
    Loads SID and STAR procedure data into a directed graph.
    
    For each procedure header line (starting with "SID" or "STAR"),
    subsequent lines are parsed as fix (waypoint) definitions.
    An edge is created from the provided airport node to the first fix
    (using the distance from the first fix’s row) and then from each fix
    to the next in the order of appearance.
    
    Edge attributes:
      - edge_type: "PROC"
      - edge_subtype: "SID" or "STAR" (from the header)
      - max_alt: 0
      - min_alt: 0
      - airway: procedure name (from header)
      - distance: distance value (nautical miles)
      - cost: same as distance
      
    Parameters:
      data (str): A string containing the CSV–formatted procedure data.
      airport_node (str): The ID of the airport node.
      runway (str): The runway number.
    
    Returns:
      G (nx.DiGraph): A NetworkX directed graph with the procedure nodes and edges.
    """    
    # Make sure the airport node exists.
    if airport_node not in G:
        G.add_node(airport_node)
    
    # Variables to track the current procedure.
    current_proc_type = None  # "SID" or "STAR"
    current_airway = None     # procedure name from header (e.g., "VETO1G")
    first_fix_in_proc = None  # will hold the first fix id for the current procedure
    previous_fix = None       # to chain the fixes

    scheduled_link_to_airport = None # this is a variable to temporarily store
    # the edge from the last fix to the arrival airport. The problem is that we don't
    # whether the current fix is the last fix in the procedure for STARs.
    # So every time we see a fix, we reassign it to this variable,
    # and when the read_phase switches to HEADER, we add the edge.
    
    read_phase = 'HEADER' # we wait for a header
    sid_star_avail = False

    # Process the data line by line.
    for line in data.strip().splitlines():
        line = line.strip()

        if not line:
            # There is an empty line
            # If the link between the last STAR fix and the arrival airport is scheduled, add it.
            if scheduled_link_to_airport is not None:
                G.add_edge(scheduled_link_to_airport['fix_id'], scheduled_link_to_airport['airport_node'],
                           edge_type=scheduled_link_to_airport['edge_type'],
                           edge_subtype=scheduled_link_to_airport['edge_subtype'],
                           max_alt=scheduled_link_to_airport['max_alt'],
                           min_alt=scheduled_link_to_airport['min_alt'],
                           airway=scheduled_link_to_airport['airway'],
                           distance=scheduled_link_to_airport['distance'],
                           cost=scheduled_link_to_airport['cost'])
                scheduled_link_to_airport = None
            read_phase = 'HEADER' # when there is a blank line, we wait for a new header
            continue
        

        # Split the line by comma.
        # (If needed, you can use the csv module to be more robust.)
        fields = [field.strip() for field in line.split(',')]
        
        # If the line begins with "SID" or "STAR", it is a header line.
        if read_phase == 'HEADER' and fields[0] in ("SID", "STAR") and (fields[2] == runway or fields[2]=='ALL'):
            current_proc_type = fields[0]     # e.g., "SID" or "STAR"

            current_airway = fields[1]          # e.g., "VETO1G" or "ANTR1G"
            # (fields[2] and fields[3] might be runway or other info.)

            first_fix_in_proc = None
            previous_fix = None
            read_phase = 'FIXES' # we are allowed to read all the fixes for this procedure
            continue  # move on to the next line
        else:
            if read_phase == 'HEADER':
                read_phase = 'SKIP' # skip until next header


        if read_phase == 'FIXES': # we are allowed to read all the fixes for this procedure
            # We assume the fix ID is in the second column.
            if fields[0] in ['CF','TF','DF','IF']: # this is a valid fix
                fix_id = current_airway + '_' + fields[1] # e.g. VETO1G_PG271 -- to avoid procedure crossings
                real_fix_id = fields[1] # e.g. PG271
                sid_star_avail = True
            else:
                read_phase = 'SKIP' # skip until next header
                continue # skip the rest of the procedure

            # Try to get latitude and longitude from columns 3 and 4.
            try:
                lat = float(fields[2])
                lon = float(fields[3])
            except (IndexError, ValueError):
                lat = None
                lon = None
                print(f"Error: could not parse latitude or longitude for fix {fix_id}")
                continue # skip this fix
            
            # Add the fix node if it is not already in the graph.
            if fix_id not in G:
                G.add_node(fix_id, lat=lat, lon=lon, type='PROC')

            # Add connection between a SID/STAR fix and all nearby fixes, within 20nm
            # Add edges between this fix and all nearby fixes within 20nm
            for node in G.nodes():
                # Skip if node is the same as current fix or is the airport
                if node == fix_id or node == airport_node:
                    continue

                # Heuristics
                # Skip if euclidean distance between nodes is more than 3 degrees
                if abs(lat - float(G.nodes[node]['lat'])) > 3 or abs(lon - float(G.nodes[node]['lon'])) > 3:
                    continue
                
                # Get coordinates of other node
                try:
                    node_lat = float(G.nodes[node]['lat'])
                    node_lon = float(G.nodes[node]['lon'])
                except (KeyError, ValueError):
                    continue

                # Calculate distance between fixes
                dist = haversine_distance(lat, lon, node_lat, node_lon)
                
                # Add bidirectional edges if within 20nm
                if dist <= 30 and ('type' not in G.nodes[node] or G.nodes[node]['type'] != 'PROC') and node != airport_node:
                    # recall: fix_id is the SID/STAR node, and node is the nearby fix (not a SID/STAR since type is not PROC)
                    if procedure == 'SID' and current_proc_type == 'SID':
                        G.add_edge(fix_id, node,
                                edge_type='DCT',
                                edge_subtype='',
                                max_alt=0,
                                min_alt=0, 
                                airway='',
                                distance=dist,
                                cost=dist)
                    elif procedure == 'STAR' and current_proc_type == 'STAR':
                        G.add_edge(node, fix_id,
                                 edge_type='DCT',
                                 edge_subtype='',
                                 max_alt=0,
                                 min_alt=0, 
                                 airway='',
                                distance=dist,
                                cost=dist)
            # Determine the distance for the edge.
            # For the first fix in a procedure, we assume the row holds
            # the distance from the airport to this fix in field index 6 (if available).
            # For subsequent fixes, we use field index 10 (if available).

            if first_fix_in_proc is None:
                first_fix_in_proc = fix_id
                try:
                    dist = haversine_distance(G.nodes[airport_node]['lat'], G.nodes[airport_node]['lon'], lat, lon)
                except (IndexError, ValueError):
                    dist = 0.0
                    print(f"Error: could not parse distance for fix {fix_id}")
                    continue # skip this fix
                # Add an edge from the airport node to the first fix.
                if procedure == 'SID':
                    G.add_edge(
                        airport_node, fix_id,
                        edge_type="PROC",
                        edge_subtype=current_proc_type,
                        max_alt=0,
                        min_alt=0,
                        airway=current_airway,
                        distance=dist,
                        cost=dist * w_proc
                    )
            else:
                # For subsequent fixes (not the first fix), use field index 10 for the distance.
                try:
                    dist = haversine_distance(G.nodes[previous_fix]['lat'], G.nodes[previous_fix]['lon'], lat, lon)
                except (IndexError, ValueError):
                    dist = 0.0
                    print(f"Error: could not parse distance for fix {fix_id}")
                    continue # skip this fix
                # Create an edge from the previous fix to the current fix.

                G.add_edge(
                    previous_fix, fix_id,
                    edge_type="PROC",
                    edge_subtype=current_proc_type,
                    max_alt=0,
                    min_alt=0,
                    airway=current_airway,
                    distance=dist,
                    cost=dist * w_proc
                )

            if procedure == 'STAR':
                scheduled_link_to_airport = {
                    'fix_id': fix_id,
                    'airport_node': airport_node,
                    'edge_type': "PROC-APT",
                    'edge_subtype': current_proc_type,
                    'max_alt': 0,
                    'min_alt': 0,
                    'airway': current_airway,
                    'distance': dist,
                    'cost': dist * w_proc
                } # we schedule to add the edge when the read_phase switches to HEADER (when it reads an empty line again)

                
            # Update previous_fix to the current fix.
            previous_fix = fix_id


    return G, sid_star_avail

# -------------------------
# Example usage:
if __name__ == '__main__':
    with open('data/airac/proc/LFPG.txt', 'r') as file:
        data = file.read()
    G = load_procedures(data, 'LFPG', '27L')
    print(G.nodes(data=True))
    print(G.edges(data=True))
