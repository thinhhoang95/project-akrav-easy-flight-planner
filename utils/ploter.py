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
