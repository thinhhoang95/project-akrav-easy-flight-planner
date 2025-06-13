# Context for Wind Extraction
- The data are CDF files located in project_root/data/wx/cdfs
- The file names are UNIX timestamp of the analysis forecast. For example: 1700827200.nc, the UNIX timestamp is 1700827200.
- Description of an xarray ds object:
```
xarray.Dataset {
dimensions:
	longitude = 111 ;
	latitude = 85 ;

variables:
	float64 longitude(longitude) ;
	float64 cape(latitude, longitude) ;
		cape:long_name = Convective Available Potential Energy ;
		cape:units = J/kg ;
		cape:standard_name = atmosphere_convective_available_potential_energy ;
	float64 cin(latitude, longitude) ;
		cin:long_name = Convective Inhibition ;
		cin:units = J/kg ;
		cin:standard_name = atmosphere_convective_inhibition ;
	float64 u_wind(latitude, longitude) ;
		u_wind:long_name = U-component of wind ;
		u_wind:units = m/s ;
		u_wind:standard_name = eastward_wind ;
	float64 v_wind(latitude, longitude) ;
		v_wind:long_name = V-component of wind ;
		v_wind:units = m/s ;
		v_wind:standard_name = northward_wind ;
	float64 wind_speed(latitude, longitude) ;
		wind_speed:long_name = Wind Speed ;
		wind_speed:units = m/s ;
		wind_speed:standard_name = wind_speed ;
	float64 latitude(latitude) ;

// global attributes:
	:description = European meteorological data extracted from GRIB file ;
	:region = Europe ;
	:source = GRIB file ;
}None

Detailed Information:
<xarray.Dataset> Size: 379kB
Dimensions:     (longitude: 111, latitude: 85)
Coordinates:
  * longitude   (longitude) float64 888B -15.0 -14.5 -14.0 ... 39.0 39.5 40.0
  * latitude    (latitude) float64 680B 30.0 30.5 31.0 31.5 ... 71.0 71.5 72.0
Data variables:
    cape        (latitude, longitude) float64 75kB ...
    cin         (latitude, longitude) float64 75kB ...
    u_wind      (latitude, longitude) float64 75kB ...
    v_wind      (latitude, longitude) float64 75kB ...
    wind_speed  (latitude, longitude) float64 75kB ...
Attributes:
    description:  European meteorological data extracted from GRIB file
    region:       Europe
    source:       GRIB file
```
