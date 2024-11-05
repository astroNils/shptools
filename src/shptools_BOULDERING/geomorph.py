import geopandas as gpd
import numpy as np
import pandas as pd
import sys

import rastertools_BOULDERING.raster as raster
import rastertools_BOULDERING.metadata as raster_metadata

import shptools_BOULDERING.geometry as shp_geom
import shptools_BOULDERING.shp as shp

from pathlib import Path
from affine import Affine
from tqdm import tqdm

def distance(x1, y1, x2, y2):
    """
    Calculate Euclidean distance between two points.
    
    Parameters
    ----------
    x1 : float
        x-coordinate of first point
    y1 : float
        y-coordinate of first point
    x2 : float
        x-coordinate of second point
    y2 : float
        y-coordinate of second point
        
    Returns
    -------
    float
        Euclidean distance between the points
    """
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def azimuth(x1, y1, x2, y2):
    """
    Calculate azimuth angles between two points.
    
    Parameters
    ----------
    x1 : float
        x-coordinate of first point
    y1 : float
        y-coordinate of first point
    x2 : float
        x-coordinate of second point
    y2 : float
        y-coordinate of second point
        
    Returns
    -------
    tuple
        Three different angle representations:
        - angle in interval [-180, 180] degrees
        - angle in interval [0, 360] degrees
        - angle in interval [0, 180] degrees
        
    Notes
    -----
    The angles follow these conventions:
    - North is 0 degrees
    - East is 90 degrees
    - South is 180 degrees
    - West is either -90, 270, or 90 degrees depending on representation
    """
    # let's take the absolute value of y2 - y1 to avoid values above 180 degrees
    angle = np.arctan2(x2 - x1, y2 - y1)
    return (np.degrees(angle),
            np.degrees(angle) if angle > 0 else np.degrees(angle) + 360,
            np.degrees(angle) if angle > 0 else np.degrees(angle) + 180)

def filter_by_area(in_shp, area_threshold, out_shp, reindex=True):
    """
    Filter polygons in a shapefile by minimum area threshold.
    
    Parameters
    ----------
    in_shp : str or Path
        Path to input shapefile
    area_threshold : float
        Minimum area threshold for filtering
    out_shp : str or Path
        Path to output filtered shapefile
    reindex : bool, optional
        Whether to reindex the filtered features, by default True
        
    Returns
    -------
    geopandas.GeoDataFrame
        DataFrame containing filtered polygons with attributes:
        - x : centroid x-coordinate
        - y : centroid y-coordinate
        - poly_area : polygon area
        - id : feature index (if reindex=True)
    """
    gdf = gpd.read_file(in_shp)
    gdf["x"] = gdf.geometry.centroid.x.values
    gdf["y"] = gdf.geometry.centroid.y.values
    gdf["poly_area"] = gdf.geometry.area
    gdf_area_filtered = gdf[gdf["poly_area"] >= area_threshold]
    n = gdf_area_filtered.shape[0]
    if reindex:
        gdf_area_filtered["id"] = np.arange(n).astype("int")
    gdf_area_filtered.to_file(out_shp)
    return(gdf_area_filtered)

def boulder_row(row):
    """
    Calculate geometric parameters for a minimum rotated rectangle polygon.
    
    Parameters
    ----------
    row : pandas.Series
        Row containing polygon geometry from a GeoDataFrame
        
    Returns
    -------
    tuple
        (length, width, long_axis, short_axis, diameter, 
         rotation_angle_default, rotation_angle360, rotation_angle180)
        where:
        - length : longer side of rectangle
        - width : shorter side of rectangle
        - long_axis : maximum of length and width
        - short_axis : minimum of length and width
        - diameter : average of long and short axes
        - rotation_angle_default : angle in [-180, 180] degrees
        - rotation_angle360 : angle in [0, 360] degrees
        - rotation_angle180 : angle in [0, 180] degrees
        
    Notes
    -----
    The rotation angles are calculated relative to North (0 degrees)
    based on the orientation of the longer axis.
    """
    bbox = list(row.geometry.exterior.coords)
    width = distance(bbox[0][0], bbox[0][1], bbox[3][0], bbox[3][1])
    length = distance(bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1])
    long_axis = np.max(np.array([length, width]))
    short_axis = np.min(np.array([length, width]))
    diameter = (short_axis + long_axis) / 2.0

    if width <= length:
        rotation_angle_default, rotation_angle360, rotation_angle180 = azimuth(
            bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1])
    else:
        rotation_angle_default, rotation_angle360, rotation_angle180 = azimuth(
            bbox[0][0], bbox[0][1], bbox[3][0], bbox[3][1])

    return (length, width, long_axis, short_axis, diameter, rotation_angle_default,
            rotation_angle360, rotation_angle180)

def boulder(in_shp, out_shp_mmr, out_shp_geomorph,
            crater_centre_point, crater_diameter, is_boulder=True):
    """
    Calculate geometric parameters for boulder or ellipse polygons.
    
    Parameters
    ----------
    in_shp : str or Path
        Path to input shapefile containing polygons
    out_shp_mmr : str or Path
        Path to output minimum rotated rectangles shapefile
    out_shp_geomorph : str or Path
        Path to output geomorphology shapefile
    crater_centre_point : str or Path or None
        Path to crater center point shapefile, or None
    crater_diameter : float or None
        Crater diameter for normalization, or None
    is_boulder : bool, optional
        Whether polygons represent boulders (True) or ellipses (False),
        by default True
        
    Returns
    -------
    geopandas.GeoDataFrame
        DataFrame containing polygons with calculated parameters:
        - length : longer side of minimum rotated rectangle
        - width : shorter side of minimum rotated rectangle
        - long_axis : maximum of length and width
        - short_axis : minimum of length and width
        - aspect_ra : aspect ratio (long_axis / short_axis)
        - angle : rotation angle in [-180, 180] degrees
        - angle360 : rotation angle in [0, 360] degrees
        - angle180 : rotation angle in [0, 180] degrees
        - diameter : average of long and short axes
        - diametereq : equivalent diameter from area
        - dist_cc : distance to crater center (if provided)
        - ndist_cc : normalized distance to crater center (if provided)
        
    Notes
    -----
    This function should be run after filtering polygons by area:
    areal_threshold = (res * res) * (4.74 * 4.74)
    filter_by_area(in_shp, areal_threshold, out_shp_filtered)
    """
    if is_boulder:
        print("Computing the minimum rotated rectangles....")
    else:
        print("Computing the minimum rotated rectangles for ellipses....")

    gdf = gpd.read_file(in_shp)
    # minimum rotated rectangle (mrr)
    gdf_mrr = shp_geom.minimum_rotated_rectangle(in_shp, out_shp_mmr)

    # location of a crater centre or None (if none do nothing)
    if crater_centre_point:
        gdf_cc = gpd.read_file(crater_centre_point)
        geometry_cc = gdf_cc.geometry.values

    (length, width,
     long_axis, short_axis,
     diameter, rotation_angle_default,
     rotation_angle360, rotation_angle180) = [[] for _ in range(8)]
    if is_boulder:
        print("Computing boulder diameters, aspect ratios, orientations....")
    else:
        print("Computing boulder diameters, aspect ratios, orientations for ellipses....")

    for index, row in tqdm(gdf_mrr.iterrows(), total=gdf_mrr.shape[0]):
        data = boulder_row(row)
        length.append(data[0])
        width.append(data[1])
        long_axis.append(data[2])
        short_axis.append(data[3])
        diameter.append(data[4])
        rotation_angle_default.append(data[5])
        rotation_angle360.append(data[6])
        rotation_angle180.append(data[7])

    gdf_mrr["length"] = length
    gdf_mrr["width"] = width
    gdf_mrr["long_axis"] = long_axis
    gdf_mrr["short_axis"] = short_axis
    gdf_mrr["aspect_ra"] = np.array(long_axis) / np.array(short_axis)
    gdf_mrr["angle"] = rotation_angle_default
    gdf_mrr["angle360"] = rotation_angle360
    gdf_mrr["angle180"] = rotation_angle180
    gdf_mrr["diameter"] = (np.array(long_axis) + np.array(short_axis)) / 2.0
    gdf_mrr["diametereq"] = 2.0 * np.sqrt(gdf_mrr["poly_area"] / np.pi) #equivalent diameter calculated from area
    gdf_mrr["geometry"] = gdf.geometry.values

    if crater_centre_point:
        gdf_mrr["dist_cc"] = distance(gdf_mrr["x"], gdf_mrr["y"], geometry_cc.x, geometry_cc.y)
        gdf_mrr["ndist_cc"] = gdf_mrr["dist_cc"] / (crater_diameter / 2.0)

    gdf_mrr.to_file(out_shp_geomorph)
    return (gdf_mrr)


def density(in_shp, in_graticule, in_graticule_selection, in_raster, min_eq_diameter, 
           block_width, block_height, out_shp, out_raster):
    """
    Calculate boulder density statistics within a spatial grid.
    
    Parameters
    ----------
    in_shp : str or Path
        Path to input shapefile containing boulders with equivalent diameter calculated
    in_graticule : str or Path
        Path to complete graticule shapefile
    in_graticule_selection : str or Path
        Path to graticule selection shapefile (can be same as in_graticule)
    in_raster : str or Path
        Path to input raster for metadata reference
    min_eq_diameter : float
        Minimum equivalent diameter threshold for boulder selection
    block_width : int
        Width of analysis blocks in pixels
    block_height : int
        Height of analysis blocks in pixels
    out_shp : str or Path
        Path to output shapefile
    out_raster : str or Path
        Path to output raster files (will create count and percentage variants)
        
    Returns
    -------
    None
        Outputs are written to files:
        - count raster: number of boulders per block
        - percentage raster: percentage area covered by boulders
        
    Notes
    -----
    Block width must equal block height in current implementation.
    Input shapefile must have equivalent diameter pre-calculated.
    """

    assert block_width == block_height, "block width is different than block_height. This is not supported by the current algorithm."

    out_raster = Path(out_raster)
    out_raster_count = out_raster.with_name(out_raster.stem + "-count.tif")
    out_raster_area = out_raster.with_name(out_raster.stem + "-percentage-covered.tif")

    out_shp1 = in_shp.parent / (
        "selection-density-min-eq-diameter.shp")  # would be interesting to have the min_eq_diameter included
    out_shp2 = in_shp.parent / (
        "spatial-selection-density-min-eq-diameter.shp")  # would be interesting to have the min_eq_diameter included
    out_shp3 = in_shp.parent / ("spatial-selection-density-min-eq-diameter-centroids.shp")

    # read graticule
    gdf_graticule = gpd.read_file(in_graticule)
    gdf_graticule_selection = gpd.read_file(in_graticule_selection)

    # get metadata and resolution of in_raster
    out_meta = raster_metadata.get_profile(in_raster)
    res = raster_metadata.get_resolution(in_raster)[0]

    gdf_boulders = gpd.read_file(in_shp)
    gdf_boulders_selection = gdf_boulders[gdf_boulders.diametereq >= min_eq_diameter]
    gdf_boulders_selection.to_file(out_shp1)
    gdf_intersect = shp.intersect(out_shp1, in_graticule_selection, min_area_threshold=0, out_shp=out_shp2)

    gdf_centroid = shp.centroid(out_shp2, out_shp3)
    gdf_within = gpd.sjoin(gdf_centroid, gdf_graticule_selection, how="inner", op="within")

    # number of boulders per tile_id
    nboulders_per_tile = gdf_within.groupby('tile_id')['id'].nunique().astype('uint16')  # until here, it is correct
    tile_area = gdf_graticule_selection.iloc[0].geometry.area
    density_boulders = np.round(nboulders_per_tile / (tile_area * 10e-6), decimals=0).astype('uint16')

    # create a panda dataframe out of it
    d = {"tile_id": nboulders_per_tile.index.values,
         "nboulders": nboulders_per_tile.values,
         "dboulders": density_boulders.values}

    df = pd.DataFrame(data=d)

    # set 0 values for everything
    gdf_graticule["nboulders"] = 0
    gdf_graticule["dboulders"] = 0

    # only update the number
    gdf_graticule.nboulders[gdf_graticule.tile_id.isin(df.tile_id)] = df.nboulders.values
    gdf_graticule.dboulders[gdf_graticule.tile_id.isin(df.tile_id)] = df.dboulders.values

    # attribute joins
    nw = np.arange(0, out_meta["width"], block_width).shape[0]
    nh = np.arange(0, out_meta["height"], block_height).shape[0]

    # generate array
    array_count = gdf_graticule.nboulders.values.reshape((nw, nh)).T
    array_area = gdf_graticule.dboulders.values.reshape((nw, nh)).T
    bbox_unary = gdf_graticule.geometry.unary_union.bounds
    out_transform = Affine(block_width * res, 0.0, bbox_unary[0], 0.0, -block_width * res, bbox_unary[3])
    out_meta.update({'dtype': 'uint16', 'width': nw, 'height': nh, 'transform': out_transform})
    raster.save(out_raster_count, np.expand_dims(array_count, axis=2), out_meta, is_image=True)
    raster.save(out_raster_area, np.expand_dims(array_area, axis=2), out_meta, is_image=True)

def median(in_shp, in_graticule, column, in_meta, out_raster, dtype='uint8'):
    """
    Calculate median values of a column within spatial grid cells.
    
    Parameters
    ----------
    in_shp : str or Path
        Path to input shapefile
    in_graticule : str or Path
        Path to graticule shapefile defining grid cells
    column : str
        Name of column to calculate median for
    in_meta : dict
        Metadata dictionary containing 'width' and 'length' keys
    out_raster : str or Path
        Path to output raster file
    dtype : str, optional
        Output data type, by default 'uint8'
        
    Returns
    -------
    None
        Results are written to output raster file
    """

    gdf = gpd.read_file(in_shp)
    gdf_graticule = gpd.read_file(in_graticule)

    gdf_centroid = gdf.copy()
    gdf_centroid["geometry"] = gdf.geometry.centroid

    values = []
    for index, row in gdf_graticule.iterrows():
        gdf_selection = gdf[gdf_centroid.geometry.within(row.geometry)]
        values.append(np.median(gdf_selection[column]))

    values = np.array(values).reshape(in_meta["width"], in_meta["length"]).T
    values = np.reshape(values, (in_meta["length"], in_meta["width"], 1)).astype(dtype)

    out_meta = in_meta.copy()
    out_meta.update({'dtype': dtype})
    raster.save(out_raster, values, out_meta)
    print("...calculate median of " + column + "...")

def csfd(in_shp, mask_shp, min_area_threshold, out_shp):
    """
    Calculate Cumulative Size-Frequency Distribution for boulders.
    
    Parameters
    ----------
    in_shp : str or Path
        Path to input shapefile containing boulders
    mask_shp : str or Path
        Path to mask shapefile defining analysis area
    min_area_threshold : float
        Minimum area threshold for filtering intersected polygons
    out_shp : str or Path
        Path to output shapefile
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing:
        - boulder_diameter_eq : equivalent diameter values
        - density : cumulative number of boulders per area
        - confidence_interval : confidence interval for density
        
    Notes
    -----
    Area can be defined by single polygon or multiple squares.
    Duplicates in geometry are automatically removed.
    """

    # reading of files
    gdf = shp.intersect(in_shp, mask_shp, min_area_threshold, out_shp)
    gdf_mask = gpd.read_file(mask_shp)

    # just to double check that no duplicates are found in geometry
    gdf = gdf.drop_duplicates('geometry')

    # get the total area
    area = gpd.GeoDataFrame(geometry=gpd.GeoSeries([gdf.geometry.unary_union,
                                                    gdf_mask.geometry.unary_union],
                                                   crs= gdf.crs)).geometry.unary_union.area

    diameter_sorted = np.sort(gdf.diametereq.values)[::-1]
    N = np.arange(len(diameter_sorted)) + 1
    N_per_area = N / area
    confidence_interval_pred = (N_per_area + np.sqrt(N_per_area)) / area
    pd_csfd = pd.DataFrame(data=np.stack([diameter_sorted, N_per_area, confidence_interval_pred]).T,
                           columns=["boulder_diameter_eq","density","confidence_interval"])
    return (pd_csfd)

def csfd_binned(in_shp, mask_shp, min_area_threshold, out_shp, bins_range=False):
    """
    Calculate binned Size-Frequency Distribution for boulders.
    
    Parameters
    ----------
    in_shp : str or Path
        Path to input shapefile containing boulders
    mask_shp : str or Path
        Path to mask shapefile
    min_area_threshold : float
        Minimum area threshold for filtering intersected polygons
    out_shp : str or Path
        Path to output shapefile
    bins_range : tuple or False, optional
        Optional (min, max) range
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing:
        - boulder_diameter : boulder diameter values
        - density : cumulative number of boulders per area
        - confidence_interval : confidence interval for density
        
    Notes
    -----
    Area can be defined by single polygon or multiple squares.
    Duplicates in geometry are automatically removed.
    """

    # reading of files
    gdf = shp.intersect(in_shp, mask_shp, min_area_threshold, out_shp)
    gdf_mask = gpd.read_file(mask_shp)

    # get the total area
    area = gpd.GeoDataFrame(geometry=gpd.GeoSeries([gdf.geometry.unary_union,
                                                    gdf_mask.geometry.unary_union],
                                                   crs= gdf.crs)).geometry.unary_union.area

    s = pd.Series(gdf.diameter.values)
    if bins_range:
        bins= np.linspace(bins_range[0], bins_range[1], 20)
    else:
        bins= np.linspace(s.min(), s.max(), 20)
    step = bins[1] - bins[0]
    bins= np.append(bins, [bins[-1] + step], axis=0)
    out = pd.cut(s, bins=bins, right=False, include_lowest=True)
    N = out.value_counts(sort=False)
    diam_min = bins[:-1]
    density = N.values / area
    errors = (density / np.sqrt(N))

    pd_csfd = pd.DataFrame(data=np.stack(
        [diam_min, density, errors]).T,
                           columns=["boulder_diameter", "density",
                                    "confidence_interval"])
    return (pd_csfd)


## This function is very specific, camembert-related function (see camembert function in geometry.py)
def angle_from_pie(gdf_circle, gdf_oriented_bboxes, threshold_distance):
    """
    Calculate median angles of oriented boulders within circular sectors.
    
    Parameters
    ----------
    gdf_circle : geopandas.GeoDataFrame
        DataFrame containing circular sector geometries
    gdf_oriented_bboxes : geopandas.GeoDataFrame
        DataFrame containing oriented boulder boxes with attributes:
        - dist_from_cc : distance from center
        - aspect_ratio : length/width ratio
        - angle : orientation angle
    threshold_distance : float
        Minimum distance from center for boulder selection
        
    Returns
    -------
    geopandas.GeoDataFrame
        Input circle DataFrame with added column:
        - median_angle : median angle of selected boulders
        
    Notes
    -----
    Only boulders with aspect_ratio > 2.0 and distance > threshold_distance
    are used for angle calculation.
    """
    gdf_centroid = gdf_oriented_bboxes.copy()
    gdf_centroid["geometry"] = gdf_oriented_bboxes.geometry.centroid

    values = []
    for index, row in gdf_circle.iterrows():
        gdf_selection = gdf_oriented_bboxes[
            gdf_centroid.geometry.within(row.geometry)]
        gdf_angle = gdf_selection[gdf_selection["dist_from_cc"] > threshold_distance]
        gdf_angle = gdf_angle[
            gdf_angle["aspect_ratio"] > 2.0]
        print(gdf_angle.shape[0])
        if gdf_angle.shape[0] == 0:
            values.append(0.0)
        else:
            values.append(np.median(gdf_angle["angle"]))

    gdf_circle["median_angle"] = values
    return gdf_circle