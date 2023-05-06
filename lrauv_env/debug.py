import math
from dataclasses import dataclass
from gz.math7 import SphericalCoordinates, Vector3d, Angle

fuel_model_url = "https://fuel.gazebosim.org/1.0/OpenRobotics/models/Portuguese Ledge"

@dataclass
class Tile:
    index: int
    lat_deg: float
    lon_deg: float
    height: float
    pos_enu: Vector3d = Vector3d()

# Center of all 18 tiles in degrees
tiles = [
    Tile(1, 36.693509, -121.936568, 25.34),
    Tile(2, 36.693583, -121.944962, 27.094),
    Tile(3, 36.693658, -121.953356, 11.602),
    Tile(4, 36.693731, -121.961751, 6.781),
    Tile(5, 36.693804, -121.970145, 6.689),
    Tile(6, 36.693876, -121.978539, 30.707),
    Tile(7, 36.700269, -121.936475, 20.746),
    Tile(8, 36.700343, -121.944870, 29.343),
    Tile(9, 36.700418, -121.953265, 6.851),
    Tile(10, 36.700491, -121.961660, 6.462),
    Tile(11, 36.700564, -121.970055, 29.339),
    Tile(12, 36.700636, -121.978450, 148.439),
    Tile(13, 36.707029, -121.936382, 6.799),
    Tile(14, 36.707103, -121.944777, 6.814),
    Tile(15, 36.707178, -121.953173, 8.834),
    Tile(16, 36.707251, -121.961569, 11.934),
    Tile(17, 36.707324, -121.969965, 75.378),
    Tile(18, 36.707396, -121.978360, 229.765)]

# Convert to world ENU coordinates
sc = SphericalCoordinates(
    SphericalCoordinates.EARTH_WGS84,
    Angle(math.radians(tiles[0].lat_deg)),
    Angle(math.radians(tiles[0].lon_deg)),
    0, Angle(0))

for tile in tiles:
    vec = Vector3d(math.radians(tile.lat_deg), math.radians(tile.lon_deg), 0)
    pos_enu = sc.position_transform(vec,
        SphericalCoordinates.SPHERICAL,
        SphericalCoordinates.LOCAL2)
    tile.pos_enu = pos_enu
    print(tile)