import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.affinity

from pathlib import Path
from shapely import geometry, segmentize
from skimage.measure import CircleModel
from skimage.measure import EllipseModel
from tqdm import tqdm

# TODO > multiple buffer (multi-ring, but as disk)
def minimum_rotated_rectangle(in_shp, out_shp):
    """
    Compute the minimum rotated rectangle for each polygon in a shapefile.
    
    Parameters
    ----------
    in_shp : str or Path
        Path to input shapefile containing polygons
    out_shp : str or Path
        Path to output shapefile where rotated rectangles will be saved
        
    Returns
    -------
    geopandas.GeoDataFrame
        DataFrame containing the minimum rotated rectangles with same 
        attributes as input
        
    Notes
    -----
    Uses shapely's minimum_rotated_rectangle algorithm to find the smallest
    rectangle that contains each polygon while allowing rotation.
    """
    gdf = gpd.read_file(in_shp)
    gdf_mrr = gdf.copy()
    n = gdf_mrr.shape[0]
    rotated_rectangle = []

    for index, row in tqdm(gdf_mrr.iterrows(), total=n):
        rotated_rectangle.append(row.geometry.minimum_rotated_rectangle)

    gdf_mrr["geometry"] = rotated_rectangle
    gdf_mrr.to_file(out_shp)
    return (gdf_mrr)


def camembert(in_shp_point, radius, angle_per_camembert, labels, out_shp):
    """
    Create a circular partition around a point divided into angular sectors.
    
    Parameters
    ----------
    in_shp_point : str or Path
        Path to input shapefile containing a single point
    radius : float
        Radius of the circle in map units
    angle_per_camembert : float
        Angular width of each sector in degrees
    labels : list of str
        Labels for each sector (e.g. ["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    out_shp : str or Path
        Path to output shapefile where sectors will be saved
        
    Returns
    -------
    geopandas.GeoDataFrame
        DataFrame containing the sector polygons with their labels
        
    Notes
    -----
    Creates a circular partition centered on the input point, divided into equal 
    angular sectors. Useful for directional analysis.
    """

    gdf = gpd.read_file(in_shp_point)
    xc = gdf.geometry.x.values[0]
    yc = gdf.geometry.y.values[0]

    angles = np.arange(0, 360, angle_per_camembert)
    angles_centered = angles - (angle_per_camembert / 2.0)  # so that they are centered around the direction of interest
    angles_centered[angles_centered < 0] = angles_centered[
                                               angles_centered < 0] + 360.0  # we don't want to have - degrees

    xr = np.cos(np.deg2rad(angles_centered)) * radius
    yr = np.sin(np.deg2rad(angles_centered)) * radius

    polygons = []

    for i, angle in enumerate(angles):
        try:
            polygon = [[xc, yc], [xc + xr[i], yc + yr[i]], [xc + xr[i + 1], yc + yr[i + 1]], [xc, yc]]
        except:
            polygon = [[xc, yc], [xc + xr[i], yc + yr[i]], [xc + xr[0], yc + yr[0]], [xc, yc]]
        polygons.append(geometry.Polygon(polygon))

    df = pd.DataFrame({'directions': labels})
    gdf_pol = gpd.GeoDataFrame(df, geometry=polygons)
    gdf_pol = gdf_pol.set_crs(gdf.crs.to_wkt())
    gdf_pol.to_file(out_shp)

    return (gdf_pol)

def ellipse(in_shp_polygon, res, out_shp):
    """
    Fit ellipses to polygon outlines.
    
    Parameters
    ----------
    in_shp_polygon : str or Path
        Path to input shapefile containing polygons
    res : float
        Resolution for segmentizing the polygons before fitting
    out_shp : str or Path
        Path to output shapefile where fitted ellipses will be saved
        
    Returns
    -------
    geopandas.GeoDataFrame
        DataFrame containing fitted ellipses and their parameters:
        - a : semi-major axis length
        - b : semi-minor axis length  
        - theta : rotation angle in radians
        
    Notes
    -----
    Uses scikit-image's EllipseModel to fit ellipses to polygon outlines.
    The input polygons are first segmentized to the specified resolution.
    If fitting fails for any polygon, the original shape is retained.
    """
    gdf = gpd.read_file(in_shp_polygon)
    gdf_ellipse = gdf.copy()
    gdf_ellipse["geometry"] = segmentize(gdf_ellipse.geometry, res) # two times the resolution.
    ellipses = []
    a = []
    b = []
    theta = []

    print("Fitting ellipses through boulder outlines....")
    for index, row in tqdm(gdf_ellipse.iterrows(), total=gdf_ellipse.shape[0]):
        try:
            data = fitEllipse(row)
            ellipses.append(data[0])
            a.append(data[1])
            b.append(data[2])
            theta.append(data[3])
        # if it does not work just paste original shapefile?
        except:
            print("The fitting of one ellipse failed if you see this message. The original shape is just copied...")
            ellipses.append(row.geometry)
            a.append(1.0)
            b.append(1.0)
            theta.append(1.0)

    gdf_ellipse["geometry"] = ellipses
    gdf_ellipse["a"] = a
    gdf_ellipse["b"] = b
    gdf_ellipse["theta"] = theta
    gdf_ellipse.to_file(out_shp)

    return gdf_ellipse

def circle(in_shp_polygon, res, out_shp):
    """
    Fit circles to polygon outlines.
    
    Parameters
    ----------
    in_shp_polygon : str or Path
        Path to input shapefile containing polygons
    res : float
        Resolution for segmentizing the polygons before fitting
    out_shp : str or Path
        Path to output shapefile where fitted circles will be saved
        
    Returns
    -------
    geopandas.GeoDataFrame
        DataFrame containing fitted circles and their parameters:
        - r : radius
        
    Notes
    -----
    Uses scikit-image's CircleModel to fit circles to polygon outlines.
    The input polygons are first segmentized to the specified resolution.
    """
    gdf = gpd.read_file(in_shp_polygon)
    gdf_circle = gdf.copy()
    gdf_circle["geometry"] = segmentize(gdf_circle.geometry, res)
    circles = []
    r = []

    print("Fitting circles through boulder outlines....")
    for index, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):
        data = fitCircle(row)
        circles.append(data[0])
        r.append(data[1])

    gdf_circle["geometry"] = circles
    gdf_circle["r"] = r
    gdf_circle.to_file(out_shp)

    return gdf_circle

def createEllipse(row):
    """
    Create an ellipse polygon from parameters.
    
    Parameters
    ----------
    row : pandas.Series
        Row containing ellipse parameters:
        - geometry : center point
        - long_axis : major axis length
        - short_axis : minor axis length
        - angle : rotation angle in degrees
        
    Returns
    -------
    shapely.geometry.Polygon
        Ellipse polygon
    """
    circ = shapely.geometry.Point(row.geometry.centroid.x, row.geometry.centroid.y).buffer(1)
    ell = shapely.affinity.scale(circ, row.long_axis/2.0, row.short_axis/2.0)
    elrv = shapely.affinity.rotate(ell, 90 - row.angle)
    return (elrv)

def createCircle(row):
    """
    Create a circle polygon from parameters.
    
    Parameters
    ----------
    row : pandas.Series
        Row containing circle parameters:
        - geometry : center point
        - diametersq : squared diameter
        
    Returns
    -------
    shapely.geometry.Polygon
        Circle polygon
    """
    circ = shapely.geometry.Point(row.geometry.centroid.x,row.geometry.centroid.y).buffer(1)
    circ = shapely.affinity.scale(circ, row.diametersq/2.0, row.diametersq/2.0)
    return (circ)

def fitEllipse(row):
    """
    Fit an ellipse to a polygon outline.
    
    Parameters
    ----------
    row : pandas.Series
        Row containing polygon geometry
        
    Returns
    -------
    tuple
        (shapely.geometry.Polygon, a, b, theta) where:
        - Polygon is the fitted ellipse
        - a is the semi-major axis length
        - b is the semi-minor axis length
        - theta is the rotation angle in radians
        
    Notes
    -----
    Uses scikit-image's EllipseModel to fit an ellipse to the polygon vertices.
    Coordinates are centered before fitting.
    """
    ell = EllipseModel()
    x = np.array(row.geometry.exterior.coords.xy[0])
    xc = row.geometry.centroid.x
    x = x - xc  # xc being center coordinate of the boulder

    y = np.array(row.geometry.exterior.coords.xy[1])
    yc = row.geometry.centroid.y
    y = y - yc  # yc being center coordinate of the boulder

    est = ell.estimate(np.column_stack((x, y)))

    xell, yell, a, b, theta = ell.params


    xy_ell = EllipseModel().predict_xy(np.linspace(0.0, 2 * np.pi, 128),
                                         params=(xell + xc, yell + yc, a, b, theta))

    return (geometry.Polygon(xy_ell), a, b, theta)

def fitCircle(row):
    """
    Fit a circle to a polygon outline.
    
    Parameters
    ----------
    row : pandas.Series
        Row containing polygon geometry
        
    Returns
    -------
    tuple
        (shapely.geometry.Polygon, r) where:
        - Polygon is the fitted circle
        - r is the radius
        
    Notes
    -----
    Uses scikit-image's CircleModel to fit a circle to the polygon vertices.
    Coordinates are centered before fitting.
    """
    circ = CircleModel()
    x = np.array(row.geometry.exterior.coords.xy[0])
    xc = row.geometry.centroid.x
    x = x - xc  # xc being center coordinate of the boulder

    y = np.array(row.geometry.exterior.coords.xy[1])
    yc = row.geometry.centroid.y
    y = y - yc  # yc being center coordinate of the boulder

    circ.estimate(np.column_stack((x, y)))

    xcircle, ycircle, r = circ.params

    xy_circle = CircleModel().predict_xy(np.linspace(0.0, 2 * np.pi, 128),
                                         params=(xcircle + xc, ycircle + yc, r))

    return(geometry.Polygon(xy_circle), r)

def multi_ring_buffer(in_shp, buffer_dist, number_of_rings, out_shp):
    """
    Create multiple concentric buffer rings around geometries.
    
    Parameters
    ----------
    in_shp : str or Path
        Path to input shapefile
    buffer_dist : float
        Distance between consecutive rings
    number_of_rings : int
        Number of buffer rings to create
    out_shp : str or Path
        Path to output shapefile
        
    Returns
    -------
    geopandas.GeoDataFrame
        DataFrame containing buffer rings with attributes:
        - ringid : ring number (0 to number_of_rings)
        - distance : distance from original geometry
        
    Notes
    -----
    Creates concentric buffer rings at equal intervals. Each ring includes
    the area of inner rings (cumulative buffers).
    """
    gdf = gpd.read_file(in_shp)
    gdf_buffer = gdf.copy()
    geoms = []
    for n in range(number_of_rings + 1):
        buffer_dist_tmp = (n) * buffer_dist
        geoms.append(gdf_buffer.geometry.buffer(buffer_dist_tmp).values[0])

    d = {'ringid': list(np.arange(number_of_rings + 1)), 'distance': np.arange(0,buffer_dist * (number_of_rings+1), buffer_dist),
         'geometry': geoms}
    gdf_multi_ring = gpd.GeoDataFrame(d, crs=gdf_buffer.crs)
    gdf_multi_ring.to_file(out_shp)
    return(gdf_multi_ring)


def multi_ring_donut(in_shp, buffer_dist, number_of_rings, out_shp):
    """
    Create multiple concentric donut rings around geometries.
    
    Parameters
    ----------
    in_shp : str or Path
        Path to input shapefile
    buffer_dist : float
        Distance between consecutive rings
    number_of_rings : int
        Number of donut rings to create
    out_shp : str or Path
        Path to output shapefile
        
    Returns
    -------
    geopandas.GeoDataFrame
        DataFrame containing donut rings with attributes:
        - ringid : ring number (0 to number_of_rings-1)
        - ring_start : inner radius index
        - ring_end : outer radius index
        - buffer_dis : distance between rings
        
    Notes
    -----
    Creates concentric donut rings at equal intervals. Each ring excludes
    the area of inner rings (non-cumulative buffers).
    """
    gdf = gpd.read_file(in_shp)
    gdf_buffer = gdf.copy()
    geoms = []
    for n in range(number_of_rings + 1):
        buffer_dist_tmp = (n) * buffer_dist
        geoms.append(gdf_buffer.geometry.buffer(buffer_dist_tmp).values[0])

    d = {'ringid': list(np.arange(number_of_rings + 1)),
         'distance': np.arange(0, buffer_dist * (number_of_rings + 1), buffer_dist),
         'geometry': geoms}
    gdf_multi_ring = gpd.GeoDataFrame(d, crs=gdf_buffer.crs)

    geoms = []

    for n in range(number_of_rings):
        geoms.append(gdf_multi_ring.iloc[n + 1].geometry.difference(gdf_multi_ring.iloc[n].geometry))

    distance = np.arange(0, buffer_dist * (number_of_rings + 1), buffer_dist)
    d = {'ringid': list(np.arange(number_of_rings)),
         'ring_start': np.arange(number_of_rings),
         'ring_end': np.arange(number_of_rings) + 1,
         'buffer_dis': [buffer_dist] * np.arange(number_of_rings).shape[0],
         'geometry': geoms}

    gdf_multi_donut_ring = gpd.GeoDataFrame(d, crs=gdf_buffer.crs)
    gdf_multi_donut_ring.to_file(out_shp)
    return (gdf_multi_donut_ring)

def dart(in_shp_point, radius, number_of_rings, angle_per_camembert, labels, out_shp):
    """
    Create a dartboard-like grid combining angular sectors and distance rings.
    
    Parameters
    ----------
    in_shp_point : str or Path
        Path to input shapefile containing a single point
    radius : float
        Distance between consecutive rings
    number_of_rings : int
        Number of distance rings
    angle_per_camembert : float
        Angular width of each sector in degrees
    labels : list of str
        Labels for each sector (e.g. ["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    out_shp : str or Path
        Path to output shapefile
        
    Returns
    -------
    geopandas.GeoDataFrame
        DataFrame containing grid cells with attributes:
        - id : unique cell identifier
        - directions : sector label
        - ringid : ring number
        - ring_start : inner radius index
        - ring_end : outer radius index
        - buffer_dis : distance between rings
        
    Notes
    -----
    Combines angular sectors (like camembert) with distance rings (like multi_ring_donut)
    to create a grid useful for directional and distance-based analysis.
    
    Examples
    --------
    Common label schemes:
    - 8 directions: ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    - 16 directions: ["E", "ENE", "NE", "NNE", "N", "NNW", "NW", "WNW", 
                     "W", "WSW", "SW", "SSW", "S", "SSE", "SE", "ESE"]
    """

    gdf = gpd.read_file(in_shp_point)
    xc = gdf.geometry.x.values[0]
    yc = gdf.geometry.y.values[0]

    # create camembert
    angles = np.arange(0, 360, angle_per_camembert)
    angles_centered = angles - (angle_per_camembert / 2.0)  # so that they are centered around the direction of interest
    angles_centered[angles_centered < 0] = angles_centered[
                                               angles_centered < 0] + 360.0  # we don't want to have - degrees

    xr = np.cos(np.deg2rad(angles_centered)) * (radius * (number_of_rings + 1))
    yr = np.sin(np.deg2rad(angles_centered)) * (radius * (number_of_rings + 1))

    polygons = []

    for i, angle in enumerate(angles):
        try:
            polygon = [[xc, yc], [xc + xr[i], yc + yr[i]], [xc + xr[i + 1], yc + yr[i + 1]], [xc, yc]]
        except:
            polygon = [[xc, yc], [xc + xr[i], yc + yr[i]], [xc + xr[0], yc + yr[0]], [xc, yc]]
        polygons.append(geometry.Polygon(polygon))

    df = pd.DataFrame({'directions': labels})
    gdf_pol = gpd.GeoDataFrame(df, geometry=polygons)
    gdf_pol = gdf_pol.set_crs(gdf.crs.to_wkt())

    # create donut based on radius
    # save automatically in temporary folder
    tmp_file = Path.home() / "tmp" / "shp" / "multi_ring_donut.shp"
    if tmp_file.parent.is_dir():
        None
    else:
        tmp_file.parent.mkdir(parents=True, exist_ok=True)

    gdf_donut = multi_ring_donut(in_shp_point, radius, number_of_rings, tmp_file)

    # combine camembert and donut :) into a dart-like grid
    gdf_dart = gpd.overlay(gdf_pol, gdf_donut, how='intersection', keep_geom_type=True, make_valid=True)
    gdf_dart["id"] = np.arange(gdf_dart.shape[0])

    # reorder columns so that id comes first
    cols = gdf_dart.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    gdf_dart = gdf_dart[cols]
    gdf_dart.to_file(out_shp)

    return (gdf_dart)


