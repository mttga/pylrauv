import math

def bearing_to_r_el_az(bearing):
    # get r, elevation and azymuth from the bearing
    x = bearing.x
    y = bearing.y
    z = bearing.z

    r = math.sqrt(x**2 + y**2 + z**2)
    elevation = math.asin(z / r)
    azimuth = math.atan2(y, x)
    return r, elevation, azimuth

def process_bearing(bearing):  
    r, elevation, azimuth = bearing_to_r_el_az(bearing)
    
    # Calculate Cartesian coordinates
    dx = r * math.cos(elevation) * math.cos(azimuth)
    dy = r * math.cos(elevation) * math.sin(azimuth)
    dz = r * math.sin(elevation)

    return dx, dy, dz