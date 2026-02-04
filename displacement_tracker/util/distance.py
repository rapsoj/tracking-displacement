EARTH_RADIUS_M = 6371000.0

def haversine_m(lat1, lon1, lat2, lon2):
    """
    Haversine distance in meters between two (lat, lon) pairs.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))

def interpolate_centroid(centroid, bounds, shape):
    """
    Convert centroid pixel coordinates to geographic coordinates using bounds.
    centroid: (y, x) pixel coordinates
    bounds: dict with lat_min, lat_max, lon_min, lon_max
    shape: (height, width) of the image
    Returns (latitude, longitude)
    """
    y, x = centroid
    height, width = shape
    lat_min = bounds.get('lat_min')
    lat_max = bounds.get('lat_max')
    lon_min = bounds.get('lon_min')
    lon_max = bounds.get('lon_max')
    if None in (lat_min, lat_max, lon_min, lon_max):
        raise ValueError("Missing bounds in metadata")
    lat = lat_max - (lat_max - lat_min) * (y / height)
    lon = lon_min + (lon_max - lon_min) * (x / width)
    return (lat, lon)