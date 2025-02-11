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

def get_first_part(text):
    """
    Returns the part of text after the last underscore.
    If no underscore exists, returns the original text.
    
    Args:
        text: String that may contain underscores
        
    Returns:
        String containing part after last underscore, or original if no underscore
    """
    if '_' not in text:
        return text
    return text.split('_')[0]

def remove_same_waypoints(fpl_str):
    """
    Removes consecutive duplicate waypoints from a flight plan string.
    For example "LASTI NIK NIK MOKUN" becomes "LASTI NIK MOKUN"
    
    Args:
        fpl_str: String containing space-separated waypoints
        
    Returns:
        String with consecutive duplicates removed
    """
    if not fpl_str:
        return fpl_str
        
    points = fpl_str.split()
    result = [points[0]]  # Keep first point
    
    for i in range(1, len(points)):
        if points[i] != result[-1]:  # Only add if different from previous
            result.append(points[i])
            
    return ' '.join(result)


def format_flightplan(route_points):
    """
    Format route points into a standard flight plan string.
    For airways, includes airway name followed by last point on that airway.
    For direct routings, just includes the waypoint ID.
    
    Args:
        route_points: List of dictionaries containing route point information
        
    Returns:
        String containing formatted flight plan
    """
    formatted_route = []
    current_airway = None
    
    # Add origin
    if route_points:
        formatted_route.append(get_first_part(route_points[0]['from_node']))
    
    # Process each point
    for point in route_points:
        if point['edge_type'] == 'DCT':
            # Direct routing - just add the waypoint
            formatted_route.append(get_first_part(point['to_node']))
            current_airway = None
            

        else:
            # Airway routing
            if point['airway'] != current_airway:
                # New airway - add airway name and waypoint
                formatted_route.append(get_first_part(point['airway']))
                formatted_route.append(get_first_part(point['to_node']))
                current_airway = get_first_part(point['airway'])
            else:
                # Same airway - only add final waypoint
                formatted_route[-1] = get_first_part(point['to_node'])
                
    return remove_same_waypoints(' '.join(formatted_route))
