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
        ax.text(lon, lat, node,
                fontsize=12, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                transform=ccrs.Geodetic())
    
    # Set map bounds with some padding around the route
    ax.set_extent([min(lons)-1, max(lons)+1, min(lats)-1, max(lats)+1])

    # Set title
    plt.title(f'Flight Plan: {list_of_nodes[0]} to {list_of_nodes[-1]}')
    plt.show()
