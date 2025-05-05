# Notebook Context
- route_df column names are: flight_id	real_waypoints	pass_times	speeds	alts
- Example: 0	'000042HMJ225'	'INKIM LJLJ LMML MALTI'	'1680349300 1680364679 1680382679 1680382679'	'0.2321 0.1399 0.0000 0.0000'	'11521 1570 114 114'
- departures and arrivals are the set of airports: departures: {'BKPR', 'DAAE', 'DAAG',...}, arrivals: {'BKPR', 'DAAE', 'DAAG',...}
- G: no edges, nodes are waypoints. Each has the lat and lon attributes.
- Gm: no edges, nodes are waypoints, like G. Each has the lat and lon attributes also. Gm is segregated from G, to contain only relevant waypoints to a given origin-destination.
- Gmx: the graph with edges added, as well as a little more connections to ensure it is better connected (the idea is we extend the "key edges" - edges that are long, aligned with the great circle between OD pair) as to form continuous path.
- Gmxx: the processed route graph, for each OD pair, limited to the maximum turn angles between consecutive legs. Each node is a waypoint with ID (e.g., MIRAX) with lat and lon attributes. Each edge is a possible routing option (based on historical data).
