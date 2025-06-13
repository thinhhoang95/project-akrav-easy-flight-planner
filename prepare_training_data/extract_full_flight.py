import pandas as pd
import os
from pathlib import Path
import multiprocessing as mp
from functools import partial

import networkx as nx

# Load the ATS FRA graph
G = nx.read_gml('data/graphs/ats_fra_nodes_only.gml')


def process_flight_csv(input_filename, output_directory):
    """
    Process a CSV file containing flight data and filter flights that start and end
    at altitudes less than 2000ft based on the full_alts column.
    
    Args:
        input_filename (str): Path to the input CSV file
        output_directory (str): Directory where the filtered CSV will be saved
    
    Returns:
        int: Number of flights kept after filtering
    """
    # Read the CSV file
    try:
        df = pd.read_csv(input_filename)
        print(f"Loaded {len(df)} flights from {input_filename}")
    except Exception as e:
        print(f"Error reading {input_filename}: {e}")
        return 0
    
    # Check if the required column exists
    if 'full_alts' not in df.columns:
        print(f"Error: 'full_alts' column not found in {input_filename}")
        return 0
    
    # Filter flights based on altitude criteria
    filtered_flights = []
    
    for idx, row in df.iterrows():
        full_alts_str = str(row['full_alts'])
        
        # Skip if full_alts is empty or NaN
        if pd.isna(row['full_alts']) or full_alts_str.strip() == '':
            continue
            
        try:
            # Split the altitude string and convert to floats
            altitudes = [float(alt) for alt in full_alts_str.split()]
            
            # Check if flight starts and ends at less than 2000ft
            if len(altitudes) >= 2:  # Need at least start and end altitudes
                start_alt = altitudes[0]
                end_alt = altitudes[-1]
                
                if start_alt < 2000 and end_alt < 2000:
                    filtered_flights.append(row)
                    
        except (ValueError, AttributeError) as e:
            # Skip rows with invalid altitude data
            print(f"Warning: Invalid altitude data in row {idx}: {full_alts_str}")
            continue
    
    # Create DataFrame from filtered flights
    if filtered_flights:
        filtered_df = pd.DataFrame(filtered_flights)
    else:
        # Create empty DataFrame with same columns if no flights match criteria
        filtered_df = pd.DataFrame(columns=df.columns)
    
    # Create output directory if it doesn't exist
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    input_path = Path(input_filename)
    output_filename = f"filtered_{input_path.name}"
    output_path = Path(output_directory) / output_filename
    
    # Save filtered data
    try:
        filtered_df.to_csv(output_path, index=False)
        print(f"Saved {len(filtered_df)} filtered flights to {output_path}")
    except Exception as e:
        print(f"Error saving filtered data: {e}")
        return 0
    
    return len(filtered_df)


def process_all_csv_files(input_directory, output_directory, file_pattern="*.routes.csv"):
    """
    Process all CSV files in a directory that match the given pattern using multiprocessing.
    
    Args:
        input_directory (str): Directory containing input CSV files
        output_directory (str): Directory where filtered CSVs will be saved
        file_pattern (str): Pattern to match CSV files (default: "*.routes.csv")
    
    Returns:
        dict: Dictionary with filename as key and number of kept flights as value
    """
    input_path = Path(input_directory)
    results = {}
    total_kept = 0
    
    # Find all matching CSV files
    csv_files = list(input_path.glob(file_pattern))
    
    if not csv_files:
        print(f"No CSV files found matching pattern '{file_pattern}' in {input_directory}")
        return results
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Determine number of processes to use (all but one CPU core)
    num_processes = max(1, mp.cpu_count() - 1)
    print(f"Using {num_processes} processes for parallel processing")
    
    # Create a partial function with the output_directory parameter fixed
    process_func = partial(process_flight_csv_wrapper, output_directory=output_directory)
    
    # Process files in parallel
    with mp.Pool(processes=num_processes) as pool:
        # Map the processing function to all CSV files
        pool_results = pool.map(process_func, [str(csv_file) for csv_file in csv_files])
    
    # Combine results
    for csv_file, kept_flights in zip(csv_files, pool_results):
        results[csv_file.name] = kept_flights
        total_kept += kept_flights
    
    # Print summary
    print(f"\n{'='*50}")
    print("PROCESSING SUMMARY")
    print(f"{'='*50}")
    for filename, count in results.items():
        print(f"{filename}: {count} flights kept")
    print(f"{'='*50}")
    print(f"Total flights kept across all files: {total_kept}")
    
    return results


def process_flight_csv_wrapper(input_filename, output_directory):
    """
    Wrapper function for process_flight_csv to work with multiprocessing.
    This function also prints the processing status for each file.
    
    Args:
        input_filename (str): Path to the input CSV file
        output_directory (str): Directory where the filtered CSV will be saved
    
    Returns:
        int: Number of flights kept after filtering
    """
    filename = Path(input_filename).name
    print(f"Processing: {filename}")
    result = process_flight_csv(input_filename, output_directory)
    print(f"Completed: {filename} - {result} flights kept")
    return result


# Example usage
if __name__ == "__main__":
    # Process a single file
    # kept = process_flight_csv("data/routes_wps/cs_2023-04-01.routes.csv", "filtered_data")
    # print(f"Kept {kept} flights")
    
    # Process all .routes.csv files in the directory
    results = process_all_csv_files("data/routes_wps", "filtered_data", "*.routes.csv")