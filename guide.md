# Project Akrav

## Preprocessing data

### Waypoint Route Inferral
In `route_infer`, we have the script `infer_route_auto_server.py` which can be used to convert the `cs` routes to proper routes with waypoints. For example: 

`id,from_time,to_time,from_lat,from_lon,to_lat,to_lon,from_alt,to_alt,from_speed,to_speed
000042HMJ225,1680349319.0,1680364679.0,42.94166564941406,14.271751226380816,46.17704772949219,14.543553794302593,11521.440000000002,1569.7200000000005,0.232055441288291,0.1399177216897295
000042HMJ225,1680364679.0,1680382679.0,46.17704772949219,14.543553794302593,35.847457627118644,14.489973352310503,1569.7200000000005,114.3,0.1399177216897295,0.0`

Can be converted into:
`000042HMJ225,INKIM LJLJ LMML MALTI,1680349300.423041 1680364679.0 1680382679.0 1680382679.0,0.232055441288291 0.1399177216897295 0.0 0.0,INKIM LJLJ LMML MALTI,1680349300.423041 1680364679.0 1680382679.0 1680382679.0,0.232055441288291 0.1399177216897295 0.0 0.0`

The `cs` routes can be built from ADS-B data using the scripts in `hlybokiy_potik`, check the guide in that project to know how.