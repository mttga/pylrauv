import pyproj

# origin is at Monterey bay area: lat: 36,799999, ong: -122,7
LAT_ORIGIN = 36.8
LON_ORIGIN = -122.7

def lat_lon_to_utm(latitude, longitude):
    zone_number = int((longitude + 180) / 6) + 1
    hemisphere = 'south' if latitude < 0 else 'north'

    utm_crs = pyproj.crs.CRS.from_proj4(f"+proj=utm +zone={zone_number} +ellps=WGS84 {'+south' if hemisphere == 'south' else ''}")
    lat_lon_crs = pyproj.crs.CRS.from_epsg(4326)
    transformer = pyproj.Transformer.from_crs(lat_lon_crs, utm_crs)
    easting, northing = transformer.transform(latitude, longitude)

    return easting, northing, zone_number, hemisphere

EASTING_ORIGIN, NORTHING_ORIGIN, ZONE_ORIGIN, HEMISPHERE_ORIGIN = lat_lon_to_utm(LAT_ORIGIN, LON_ORIGIN)

def custom_utm_to_lat_lon(easting, northing, zone_number=ZONE_ORIGIN, hemisphere=HEMISPHERE_ORIGIN):
    utm_crs = pyproj.crs.CRS.from_proj4(f"+proj=utm +zone={zone_number} +ellps=WGS84 {'+south' if hemisphere == 'south' else ''}")
    lat_lon_crs = pyproj.crs.CRS.from_epsg(4326)
    transformer = pyproj.Transformer.from_crs(utm_crs, lat_lon_crs)
    lat, lon = transformer.transform(easting, northing)
    return lat, lon

utm_crs = pyproj.crs.CRS.from_proj4(f"+proj=utm +zone={ZONE_ORIGIN} +ellps=WGS84 {'+south' if HEMISPHERE_ORIGIN == 'south' else ''}")
lat_lon_crs = pyproj.crs.CRS.from_epsg(4326)
transformer = pyproj.Transformer.from_crs(utm_crs, lat_lon_crs)

def utm_to_lat_lon(easting, northing, transformer=transformer):
    return transformer.transform(easting, northing)

if __name__=='__main__':
    print(lat_lon_to_utm(LAT_ORIGIN, LON_ORIGIN))
    print(custom_utm_to_lat_lon(EASTING_ORIGIN, NORTHING_ORIGIN))
    print(utm_to_lat_lon(EASTING_ORIGIN, NORTHING_ORIGIN))