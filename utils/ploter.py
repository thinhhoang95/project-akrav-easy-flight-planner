# Example fp:
# {'from_node': 'LFPG',
#   'to_node': 'RESMI',
#   'from_lat': 49.009722,
#   'from_lon': 2.547778,
#   'to_lat': 48.568611,
#   'to_lon': 2.191944,
#   'distance': 29.992426028369458,
#   'cost': 29.992426028369458,
#   'edge_type': 'DCT',
#   'airway': '',
#   'min_alt': 0,
#   'max_alt': 0},

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_flightplan(fp1, flight_number):
    # Create figure and axis with projection
    plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # Plot flight path
    for leg in fp1:
        ax.plot([leg['from_lon'], leg['to_lon']], 
                [leg['from_lat'], leg['to_lat']], 
                color='red', linewidth=2, 
                transform=ccrs.Geodetic())
        
        # Add waypoint markers
        ax.plot(leg['from_lon'], leg['from_lat'], 
                marker='o', color='blue', markersize=5,
                transform=ccrs.Geodetic())
        ax.plot(leg['to_lon'], leg['to_lat'], 
                marker='o', color='blue', markersize=5,
                transform=ccrs.Geodetic())
        
        # Add waypoint labels
        ax.text(leg['from_lon'], leg['from_lat'], leg['from_node'],
                fontsize=12, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                transform=ccrs.Geodetic())
        ax.text(leg['to_lon'], leg['to_lat'], leg['to_node'],
                fontsize=12, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                transform=ccrs.Geodetic())
        


    # Set map bounds with some padding around the route
    lats = [leg['from_lat'] for leg in fp1] + [leg['to_lat'] for leg in fp1]
    lons = [leg['from_lon'] for leg in fp1] + [leg['to_lon'] for leg in fp1]
    ax.set_extent([min(lons)-1, max(lons)+1, min(lats)-1, max(lats)+1])

    plt.title(f'Flight Plan: {fp1[0]["from_node"]} to {fp1[-1]["to_node"]}')
    plt.show()

def plot_flightplan2(graph, list_of_nodes):
    """
    Plot a flight plan using a graph and a list of nodes.
    
    Parameters:
        graph (networkx.Graph): Graph containing node coordinates
        list_of_nodes (list): Ordered list of node IDs representing the flight path
    """
    if not list_of_nodes or len(list_of_nodes) < 2:
        print("Not enough nodes to plot a flight path")
        return
    
    # Create figure and axis with projection
    plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    
    # Extract coordinates from the graph
    lats = []
    lons = []
    
    # Plot the route segments
    for i in range(len(list_of_nodes) - 1):
        from_node = list_of_nodes[i]
        to_node = list_of_nodes[i + 1]
        
        # Get coordinates from graph
        from_lat = graph.nodes[from_node]['lat']
        from_lon = graph.nodes[from_node]['lon']
        to_lat = graph.nodes[to_node]['lat']
        to_lon = graph.nodes[to_node]['lon']
        
        # Store for bounds calculation
        lats.extend([from_lat, to_lat])
        lons.extend([from_lon, to_lon])
        
        # Plot route segment
        ax.plot([from_lon, to_lon], [from_lat, to_lat], 
                color='red', linewidth=2, 
                transform=ccrs.Geodetic())
        
        # Add waypoint markers
        ax.plot(from_lon, from_lat, 
                marker='o', color='blue', markersize=5,
                transform=ccrs.Geodetic())
    
    # Add the last waypoint marker
    last_node = list_of_nodes[-1]
    last_lat = graph.nodes[last_node]['lat']
    last_lon = graph.nodes[last_node]['lon']
    ax.plot(last_lon, last_lat, 
            marker='o', color='blue', markersize=5,
            transform=ccrs.Geodetic())
    
    # Add waypoint labels for all nodes
    for node in list_of_nodes:
        lat = graph.nodes[node]['lat']
        lon = graph.nodes[node]['lon']
        ax.text(lon + 0.05, lat, node,
                fontsize=8, ha='left', va='center',
                bbox=dict(facecolor='white', alpha=0.2, edgecolor='none'),
                transform=ccrs.Geodetic())
    
    # Set map bounds with some padding around the route
    ax.set_extent([min(lons)-1, max(lons)+1, min(lats)-1, max(lats)+1])

    # Set title
    plt.title(f'Flight Plan: {list_of_nodes[0]} to {list_of_nodes[-1]}')
    plt.show()

def plot_route_of_flight(flight_route_df, figsize=(12, 8), arrow_scale=30):
    """
    Plot flight route segments using Cartopy with directional arrows.
    
    Args:
        flight_route_df: DataFrame containing flight route data with columns:
            from_lat, from_lon, to_lat, to_lon
        figsize: Tuple specifying figure dimensions (width, height) in inches
        arrow_scale: Controls the size of directional arrows (higher = smaller arrows)
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import numpy as np
    
    # Create figure and axis with a map projection
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Add coastlines and borders for reference
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    
    # Lists to store all coordinates for setting bounds
    lats = []
    lons = []
    
    # Plot each flight segment
    for _, row in flight_route_df.iterrows():
        from_lat = row['from_lat']
        from_lon = row['from_lon']
        to_lat = row['to_lat']
        to_lon = row['to_lon']
        
        # Store coordinates for bounds calculation
        lats.extend([from_lat, to_lat])
        lons.extend([from_lon, to_lon])
        
        # Plot the flight segment
        ax.plot([from_lon, to_lon], [from_lat, to_lat],
                color='red', linewidth=2,
                transform=ccrs.Geodetic())
        
        # Add directional arrows
        # Calculate midpoint for arrow placement
        mid_lon = (from_lon + to_lon) / 2
        mid_lat = (from_lat + to_lat) / 2
        
        # Calculate direction vector
        dx = to_lon - from_lon
        dy = to_lat - from_lat
        
        # Normalize and scale the direction vector
        magnitude = np.sqrt(dx**2 + dy**2)
        if magnitude > 0:  # Avoid division by zero
            # Scale arrow size based on the map extent
            # This ensures arrows remain visible at different zoom levels
            dx = dx / arrow_scale
            dy = dy / arrow_scale
            
            # Add the arrow
            ax.arrow(mid_lon - dx/2, mid_lat - dy/2, dx, dy, 
                    transform=ccrs.PlateCarree(),
                    head_width=max(magnitude/arrow_scale, 0.05),  # Minimum size to ensure visibility
                    head_length=max(magnitude/arrow_scale, 0.08),  # Minimum size to ensure visibility
                    fc='blue', ec='blue',
                    length_includes_head=True,
                    zorder=3)  # Place arrows above the route line
    
    # Set map bounds with some padding around the route
    if lats and lons:  # Check if lists are not empty
        ax.set_extent([min(lons)-1, max(lons)+1, min(lats)-1, max(lats)+1])
    
    plt.title(f'Flight Route of {flight_route_df.iloc[0]["id"]}')
    plt.show()

def plot_route_with_graph_and_nodes(graph, flight_segments, new_nodes):
    """
    Plot a route on a map with graph nodes and flight segments.
    
    Args:
        graph: NetworkX graph containing nodes with lat/lon attributes
        flight_segments: List of tuples representing flight segments [(node1, node2), ...]
        new_nodes: Dictionary of new nodes with their coordinates {'node_name': {'lat': lat, 'lon': lon}, ...}
    """
    # Create figure and axis with projection
    plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    
    # Lists to store all coordinates for setting map bounds
    all_lats = []
    all_lons = []
    
    # Plot flight segments
    i = -1
    for from_node, to_node, distance_AB in flight_segments:
        i += 1
        # Get coordinates for from_node
        if from_node == '__XXX__':
            from_node = flight_segments[i-1][1]
        if from_node.startswith('_'):
            # Node is in new_nodes dictionary
            from_lat = new_nodes[from_node]['lat']
            from_lon = new_nodes[from_node]['lon']
        else:
            # Node is in the graph
            from_lat = graph.nodes[from_node]['lat']
            from_lon = graph.nodes[from_node]['lon']
            
        # Get coordinates for to_node
        if to_node == '__XXX__':
            to_node = flight_segments[i+1][0]
        if to_node.startswith('_'):
            # Node is in new_nodes dictionary
            to_lat = new_nodes[to_node]['lat']
            to_lon = new_nodes[to_node]['lon']
        else:
            # Node is in the graph
            to_lat = graph.nodes[to_node]['lat']
            to_lon = graph.nodes[to_node]['lon']
        
        # Store coordinates for bounds calculation
        all_lats.extend([from_lat, to_lat])
        all_lons.extend([from_lon, to_lon])
        
        # Plot the flight segment
        ax.plot([from_lon, to_lon], [from_lat, to_lat], 
                color='red', linewidth=2, 
                transform=ccrs.Geodetic())
        
        # Add waypoint markers
        ax.plot(from_lon, from_lat, 
                marker='o', color='blue', markersize=5,
                transform=ccrs.Geodetic())
        ax.plot(to_lon, to_lat, 
                marker='o', color='blue', markersize=5,
                transform=ccrs.Geodetic())
        
        # Add waypoint labels
        ax.text(from_lon, from_lat, from_node,
                fontsize=6, ha='center', va='bottom',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'),
                transform=ccrs.Geodetic())
        ax.text(to_lon, to_lat, to_node,
                fontsize=6, ha='center', va='bottom',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'),
                transform=ccrs.Geodetic())
    
    # Set map bounds with some padding
    if all_lats and all_lons:
        ax.set_extent([min(all_lons)-1, max(all_lons)+1, min(all_lats)-1, max(all_lats)+1])
    
    plt.title('Flight Route with Graph Nodes')
    plt.show()

def plot_route_with_graph_and_nodes_list_format(graph, flight_segments, new_nodes):
    """
    Plot a flight route on a map with graph nodes and waypoints.
    
    Parameters:
        graph: A networkx graph with nodes having 'lat' and 'lon' attributes
        flight_segments: List of waypoint names in order ['GUDIS', '_KtZiRWMK', 'BABIT', ...]
        new_nodes: Dictionary of new nodes with their coordinates {node_name: {'lat': lat, 'lon': lon}}
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    
    # Create a figure and axis with a specific projection
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Add map features
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    
    # Lists to store all coordinates for bounds calculation
    all_lats = []
    all_lons = []
    
    # Plot each segment of the flight route
    for i in range(len(flight_segments) - 1):
        from_node = flight_segments[i]
        to_node = flight_segments[i + 1]
        
        # Get coordinates for from_node
        if from_node.startswith('_'):
            # Node is in new_nodes dictionary
            from_lat = new_nodes[from_node]['lat']
            from_lon = new_nodes[from_node]['lon']
        else:
            # Node is in the graph
            from_lat = graph.nodes[from_node]['lat']
            from_lon = graph.nodes[from_node]['lon']
        
        # Get coordinates for to_node
        if to_node.startswith('_'):
            # Node is in new_nodes dictionary
            to_lat = new_nodes[to_node]['lat']
            to_lon = new_nodes[to_node]['lon']
        else:
            # Node is in the graph
            to_lat = graph.nodes[to_node]['lat']
            to_lon = graph.nodes[to_node]['lon']
        
        # Store coordinates for bounds calculation
        all_lats.extend([from_lat, to_lat])
        all_lons.extend([from_lon, to_lon])
        
        # Plot the flight segment
        ax.plot([from_lon, to_lon], [from_lat, to_lat], 
                color='red', linewidth=2, 
                transform=ccrs.Geodetic())
        
        # Add waypoint markers
        ax.plot(from_lon, from_lat, 
                marker='o', color='blue', markersize=5,
                transform=ccrs.Geodetic())
        ax.plot(to_lon, to_lat, 
                marker='o', color='blue', markersize=5,
                transform=ccrs.Geodetic())
        
        # Add waypoint labels
        ax.text(from_lon, from_lat, from_node,
                fontsize=6, ha='center', va='bottom',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'),
                transform=ccrs.Geodetic())
        ax.text(to_lon, to_lat, to_node,
                fontsize=6, ha='center', va='bottom',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'),
                transform=ccrs.Geodetic())
    
    # Set map bounds with some padding
    if all_lats and all_lons:
        ax.set_extent([min(all_lons)-1, max(all_lons)+1, min(all_lats)-1, max(all_lats)+1])
    
    plt.title('Flight Route with Graph Nodes')
    plt.show()

def plot_route_with_graph_and_nodes_list_format_with_synth_wps(graph, flight_segments, df_synth_wps):
    """
    Plot a flight route with graph nodes and synthetic waypoints.
    
    Parameters:
        graph (networkx.Graph): Graph containing node coordinates
        flight_segments (list): List of strings in format 'FROM_NODE TO_NODE'
        new_nodes (dict): Dictionary of new nodes (not used in this function)
        df_synth_wps (DataFrame): DataFrame containing synthetic waypoints with columns 'id', 'lat', 'lon'
    """
    # Create figure and axis with projection
    plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    
    # Lists to store all coordinates for setting map bounds
    all_lats = []
    all_lons = []
    
    # Convert df_synth_wps to a dictionary for easier lookup
    synth_wps = {}
    for _, row in df_synth_wps.iterrows():
        synth_wps[row['id']] = {'lat': row['lat'], 'lon': row['lon']}
    
    # Plot the flight segment
    # Split the flight_segments string into a list of waypoints
    waypoints = flight_segments.split()
    
    # Process each waypoint pair (from_node to to_node)
    for i in range(len(waypoints) - 1):
        from_node = waypoints[i]
        to_node = waypoints[i + 1]
        
        # Get coordinates for from_node
        if from_node.startswith('_'):
            # Node is in synth_wps dictionary
            from_lat = synth_wps[from_node]['lat']
            from_lon = synth_wps[from_node]['lon']
        else:
            # Node is in the graph
            from_lat = graph.nodes[from_node]['lat']
            from_lon = graph.nodes[from_node]['lon']
        
        # Get coordinates for to_node
        if to_node.startswith('_'):
            # Node is in synth_wps dictionary
            to_lat = synth_wps[to_node]['lat']
            to_lon = synth_wps[to_node]['lon']
        else:
            # Node is in the graph
            to_lat = graph.nodes[to_node]['lat']
            to_lon = graph.nodes[to_node]['lon']
        
        # Store coordinates for bounds calculation
        all_lats.extend([from_lat, to_lat])
        all_lons.extend([from_lon, to_lon])
        
        # Plot the flight segment
        ax.plot([from_lon, to_lon], [from_lat, to_lat], 
                color='red', linewidth=2, 
                transform=ccrs.Geodetic())
        
        # Add waypoint markers
        ax.plot(from_lon, from_lat, 
                marker='o', color='blue', markersize=5,
                transform=ccrs.Geodetic())
        ax.plot(to_lon, to_lat, 
                marker='o', color='blue', markersize=5,
                transform=ccrs.Geodetic())
        
        # Add waypoint labels
        ax.text(from_lon, from_lat, from_node,
                fontsize=6, ha='center', va='bottom',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'),
                transform=ccrs.Geodetic())
        ax.text(to_lon, to_lat, to_node,
                fontsize=6, ha='center', va='bottom',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'),
                transform=ccrs.Geodetic())
    
    # Set map bounds with some padding
    if all_lats and all_lons:
        ax.set_extent([min(all_lons)-1, max(all_lons)+1, min(all_lats)-1, max(all_lats)+1])
    
    plt.title('Flight Route with Graph Nodes and Synthetic Waypoints')
    plt.show()
