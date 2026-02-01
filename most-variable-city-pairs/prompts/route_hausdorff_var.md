# Goal

We want to compute the variance of the Frechet distances between the flight routes corresponding to a particular city pair. The goal is to reveal city pairs have the most variable routes (along with other statistics).

# Input Files

The routes could be located at `matched_filtered_data_by_origin_region`. In there there are many CSV files. For example: `LF.csv` will contain all routes from any French airport such as `LFPG`, `LFBO`, `LFPO`...

For example:

```csv
flight_id,real_waypoints,pass_times,speeds,alts,real_full_waypoints,full_pass_times,full_speeds,full_alts
4D2395EWG3BH,BKPR ALELU GOMIG ETAGO EDDS,1684775555 1684776059 1684779839 1684781099 1684781279,0.1392 0.2126 0.2144 0.0948 0.0000,1943 7178 9365 1311 495,BKPR _dZEcROHC _yelziyXW ALELU GOMIG ETAGO EDDS,1684775555 1684775519 1684775579 1684776059 1684779839 1684781099 1684781279,0.1392 0.1392 0.1427 0.2126 0.2144 0.0948 0.0000,1943 1943 2347 7178 9365 1311 495
```

This shows a route from `BKPR` to `EDDS`. 

It is also necessary to load the `data/graphs/ats_fra_nodes_only.gml` graph (which contains all the waypoint definitions so you could get the waypoint/airport to geographical coordinates). Each waypoint is a node, with `lat` and `lon` attributes.

# General idea

The general steps are:

1. Compute pairwise Frechet distances between the routes in a city pair. Use shapely library for this purpose.
2. Compute the following values:
- Average distance
- Standard deviation of distance
- Minimum distance
- Maximum distance
- Average Frechet distance
- Standard deviation of Frechet distance (stand in for variance)
- Minimum Frechet distance
- Maximum Frechet distance
- Number of routes in the cohort.

# Requirements

1. Use multiprocessing to process multiple files at the same time, use maximum `n_cpu - 1` processes.
2. Carefully design so that the process's output could be assimilated across processes. To do this, we could produce a dictionary for each process, then merge the dictionaries later into a master dictionary.
3. Save the dictionary output to an external file: `output/most-variable-city-pairs` directory with the outputs listed above.

# Pitfalls
- Ensure minimum 10 routes in the city pair. If not, skip the city pair, but in the end, note down how many city pairs were skipped due to insufficient routes.

