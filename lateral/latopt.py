import networkx as nx
from math import radians, sin, cos, sqrt, atan2
from tqdm import tqdm
from utils.sid_star_reader import load_procedures
import os
from utils.haversine import haversine_distance

PROJECT_ROOT = os.getenv('PROJECT_ROOT')

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

import os

class LateralFlight:
    def __init__(self, aircraft_type, origin_icao, destination_icao):
        self.aircraft_type = aircraft_type
        self.origin_icao = origin_icao
        self.destination_icao = destination_icao

    def route_graph_subset(self, route_graph, airports_df, max_distance=500,
                           w_dct = 10.0, origin_runway = '0', destination_runway = '0',
                           use_sid_star = True, w_proc = 0.2):
        
        # Example of route_graph: 
        # <node id="ENILO">
            # <data key="d0">46.174722</data> # latitude
            # <data key="d1">38.031667</data> # longitude
        # </node>
        # <edge source="ENILO" target="LS103">
        #     <data key="d4">17.679304921328807</data> # distance in nm
        #     <data key="d7">0</data> # min_alt
        #     <data key="d8">0</data> # max_alt
        #     <data key="d5" /> # airway
        #     <data key="d6">DCT</data> # edge_type
        # </edge>

        # Lookup origin and destination airport coordinates
        origin_airport = airports_df[airports_df['icao'] == self.origin_icao]
        destination_airport = airports_df[airports_df['icao'] == self.destination_icao]
        
        # Validate airport data exists
        if origin_airport.empty or destination_airport.empty:
            raise ValueError(f"Could not find coordinates for airports {self.origin_icao} or {self.destination_icao}")
        
        # Extract latitude and longitude
        origin_lat = origin_airport['latitude'].values[0]
        origin_lon = origin_airport['longitude'].values[0]
        dest_lat = destination_airport['latitude'].values[0]
        dest_lon = destination_airport['longitude'].values[0]

        # We get the relevant part of the graph (nodes and edges)
        # that locate within 500nm from the great circle path between origin and destination

        # Create a new graph for the subset
        subset_graph = nx.DiGraph()

        # For each node in the original graph
        for node in tqdm(route_graph.nodes(), desc="Adding nodes to subset"):
            node_lat = float(route_graph.nodes[node]['lat'])
            node_lon = float(route_graph.nodes[node]['lon'])
            
            # Calculate distance from node to great circle path
            dist_to_path = point_line_distance(node_lat, node_lon, 
                                             origin_lat, origin_lon,
                                             dest_lat, dest_lon)
            
            # If node is within max_distance (nm) of path, add it to subset
            if dist_to_path <= max_distance:
                subset_graph.add_node(node, **route_graph.nodes[node])



        # Add relevant edges
        for edge in tqdm(route_graph.edges(data=True), desc="Adding edges to subset"):
            source, target, data = edge
            # If both nodes are in subset, add the edge
            if source in subset_graph and target in subset_graph:
                # Compute cost of edge
                if data['edge_type'] == 'DCT':
                    data['cost'] = data['distance'] * w_dct
                else:
                    data['cost'] = data['distance']
                subset_graph.add_edge(source, target, **data)

        # We add the origin and destination nodes to the subset
        subset_graph.add_node(self.origin_icao, lat=origin_lat, lon=origin_lon)
        subset_graph.add_node(self.destination_icao, lat=dest_lat, lon=dest_lon)

        sid_avail = True
        star_avail = True

        if use_sid_star:
            # We add the SID and STAR graphs to the subset
            # Read the procedures from the file
            sid_file = os.path.join(PROJECT_ROOT, 'data', 'airac', 'proc', f'{self.origin_icao}.txt')
            star_file = os.path.join(PROJECT_ROOT, 'data', 'airac', 'proc', f'{self.destination_icao}.txt')
            # If file exists, read the procedures
            if os.path.exists(sid_file):
                data = open(sid_file, 'r').read()
                subset_graph, sid_avail = load_procedures(data, self.origin_icao, origin_runway, subset_graph, procedure='SID', w_proc=w_proc)
            if os.path.exists(star_file):
                data = open(star_file, 'r').read()
                subset_graph, star_avail = load_procedures(data, self.destination_icao, destination_runway, subset_graph, procedure='STAR', w_proc=w_proc)


        # Haversine distance between origin and destination
        dist_origin_dest = haversine_distance(origin_lat, origin_lon, dest_lat, dest_lon)

        # We add the edges to the origin and destination nodes
        # Add edges to/from origin and destination for nodes within 40nm
        if not use_sid_star and sid_avail and star_avail:
            for node in tqdm(subset_graph.nodes(), desc="Adding edges to/from origin and destination"):
                if node not in [self.origin_icao, self.destination_icao]:
                    # Check distance to origin
                    node_lat = float(subset_graph.nodes[node]['lat'])
                    node_lon = float(subset_graph.nodes[node]['lon'])
                    
                    dist_to_origin = haversine_distance(origin_lat, origin_lon, node_lat, node_lon)
                    if dist_to_origin <= 40:
                        subset_graph.add_edge(self.origin_icao, node, max_alt=0, min_alt=0,
                                            edge_type='DCT', airway='', cost=dist_to_origin)
                        # subset_graph.add_edge(node, self.origin_icao, max_alt=0, min_alt=0, edge_type='DCT', airway='')
                    

                    # Check distance to destination    
                    dist_to_dest = haversine_distance(dest_lat, dest_lon, node_lat, node_lon)
                    if dist_to_dest <= 40:
                        subset_graph.add_edge(node, self.destination_icao, max_alt=0,
                                            min_alt=0, edge_type='DCT', airway='', cost=dist_to_dest)
                        # subset_graph.add_edge(self.destination_icao, node)

        self.search_graph = subset_graph
        return subset_graph, dist_origin_dest



    def plan(self):
        if self.search_graph is None:
            raise ValueError("Route graph not found. Please call route_graph_subset first.")

        # Get the shortest path in the search graph
        shortest_path = nx.shortest_path(self.search_graph, source=self.origin_icao,
                                         target=self.destination_icao, weight='cost')
        # Convert shortest path into list of dictionaries with edge information
        result = []
        cumulative_cost = 0
        cumulative_distance = 0
        for i in range(len(shortest_path)-1):
            from_node = shortest_path[i]
            to_node = shortest_path[i+1]
            
            # Get node coordinates
            from_lat = float(self.search_graph.nodes[from_node]['lat'])
            from_lon = float(self.search_graph.nodes[from_node]['lon'])
            to_lat = float(self.search_graph.nodes[to_node]['lat'])
            to_lon = float(self.search_graph.nodes[to_node]['lon'])
            
            # Get edge data
            edge_data = self.search_graph.edges[from_node, to_node]
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
                'min_alt': edge_data['min_alt'],
                'max_alt': edge_data['max_alt']
            })
            

        return result, cumulative_cost, cumulative_distance
        



