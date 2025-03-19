from dotenv import load_dotenv
import os
load_dotenv()
PROJECT_ROOT = os.getenv('PROJECT_ROOT')

# Add PROJECT_ROOT to the Python path
import sys
sys.path.append(PROJECT_ROOT)

import networkx as nx

# Load the comprehensive route graph
import time 
time_start = time.time()
print('Reading navigation graph...')
import pickle
G = pickle.load(open(os.path.join(PROJECT_ROOT, 'data', 'graphs', 'ats_fra_graph.pkl'), 'rb'))
print(f'Navigation graph loaded in {time.time() - time_start:.2f} seconds')

from infer_route import infer_route, infer_n_routes, convert_final_routes_to_waypoints, convert_df_to_segment_format
import pandas as pd
from tqdm import tqdm

def process_one_day(day_df, G, output_path):
    flight_ids = day_df['id'].unique()
    print(f'There are {len(flight_ids)} unique flight ids in the routes data')
    result_df = pd.DataFrame(columns=['id', 'route_string'])
    for selected_flight_id in tqdm(flight_ids):
        try:
            selected_flight_df = day_df[day_df['id'] == selected_flight_id]
            selected_flight_route = convert_df_to_segment_format(selected_flight_df)
            
            # Infer route
            candidate_routes = infer_n_routes(selected_flight_route, G, connection_weight=1e-3, candidate_limit=32, n_routes=1)

            # Convert the final route to waypoints
            simplified_routes = [x for _, x in candidate_routes]
            waypoints = [convert_final_routes_to_waypoints(x) for x in simplified_routes]
            
            best_route = waypoints[0]
            result_df = pd.concat([result_df, pd.DataFrame({'id': [selected_flight_id], 'route_string': [' '.join(best_route)]})], ignore_index=True)
        except Exception as e:
            print(f'Error processing flight id {selected_flight_id}: {e}')
            continue
        
    # Save the result to a CSV file
    result_df.to_csv(output_path, index=False)
    
if __name__ == '__main__':
    day_df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'routes_data', 'cs_2023-04-01.csv'))
    # Create the output path if it does not exist
    import os
    os.makedirs(os.path.join(PROJECT_ROOT, 'data', 'infered_routes'), exist_ok=True)
    output_path = os.path.join(PROJECT_ROOT, 'data', 'infered_routes', 'cs_2023-04-01.csv')
    process_one_day(day_df, G, output_path)