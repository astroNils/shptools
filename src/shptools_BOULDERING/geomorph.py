import geopandas as gpd
import numpy as np
import pandas as pd
import sys

import rastertools_BOULDERING.raster as raster
import shptools_BOULDERING.geometry as shp_geom
import shptools_BOULDERING.shp as shp

from pathlib import Path
from affine import Affine
from tqdm import tqdm

def distance(x1, y1, x2, y2):
    "Distance between 2 points."
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def azimuth(x1, y1, x2, y2):
    """azimuth between 2 points. The north is degree 0.

    (QGIS) North 0; East 90; South 180; West 270; North 360
    (1) North 0; East 90; South 180; West -90; North -180 (interval -180 - 180) - OK
    (2) North 0; East 90; South 180; West 270; North 360 (interval 0-360) - OK
    (3) North 0; East 90; South 180; West 90; North 0 (interval 0-180) - double check
    """
    # let's take the absolute value of y2 - y1 to avoid values above 180 degrees
    angle = np.arctan2(x2 - x1, y2 - y1)
    return (np.degrees(angle),
            np.degrees(angle) if angle > 0 else np.degrees(angle) + 360,
            np.degrees(angle) if angle > 0 else np.degrees(angle) + 180)

def filter_by_area(in_shp, area_threshold, out_shp, reindex=True):
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
    Here is a function which takes the geometry of a minimum_rotated_rectangle
    polygon and compute the main basis parameter such as width, length, short
    and long axes, diameter, squared diameter, and rotation angle of the boulder.
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
    filter_by_area should be run before this function (as below).
    areal_threshold = (res * res) * (4.74 * 4.74)
    __ = filter_by_area(in_shp, areal_threshold, out_shp_filtered, reindex=True)
    __ = boulder(out_shp_filtered, out_shp_mmr, out_shp_geomorph,
             None, None, is_boulder=True)

    __ = ellipse(out_shp_filtered, res, out_shp_ellipse)
    __ = boulder(out_shp_ellipse, out_shp_mmr_ellipse, out_shp_geomorph_ellipse,
             None, None, is_boulder=False)

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



def density(in_shp, in_graticule, in_raster, block_width, block_height, out_shp, out_raster):

    """
    This function depends on the filtering apply to in_shp.
    """

    assert block_width == block_height, "block width is different than block_height. This is not supported by the current algorithm."

    out_raster = Path(out_raster)
    out_raster_count = out_raster.with_name(out_raster.stem + "-count.tif")
    out_raster_area = out_raster.with_name(out_raster.stem + "-percentage-covered.tif")

    # read graticule
    gdf_graticule = gpd.read_file(in_graticule)

    # get metadata and resolution of in_raster
    out_meta = raster.get_raster_profile(in_raster)
    res = raster.get_raster_resolution(in_raster)[0]

    gdf_centroid = shp.centroid(in_shp, out_shp)
    gdf_within = gpd.sjoin(gdf_centroid, gdf_graticule, how="inner", op="within")

    # number of boulders per tile_id
    nboulders_per_tile = gdf_within.groupby('tile_id')['id'].nunique()

    area_covered_per_tile = np.round((gdf_within.groupby('tile_id').area.sum() / (
                block_width * res * block_height * res)) * 100, decimals=0)

    area_covered_per_tile = area_covered_per_tile.astype('int')

    # create a panda dataframe out of it
    d = {"tile_id":nboulders_per_tile.index.values,
         "nboulders":nboulders_per_tile.values,
         "area_cov": area_covered_per_tile.values}

    df = pd.DataFrame(data=d)

    # set 0 values for everything
    gdf_graticule["nboulders"] = 0
    gdf_graticule["area_cov"] = 0

    # only update the number
    gdf_graticule.nboulders[gdf_graticule.tile_id.isin(df.tile_id)] = df.nboulders.values
    gdf_graticule.area_cov[gdf_graticule.tile_id.isin(df.tile_id)] = df.area_cov.values

    # attribute joins
    nw = np.arange(0, out_meta["width"], block_width).shape[0]
    nh = np.arange(0, out_meta["height"], block_height).shape[0]

    # generate array
    array_count = gdf_graticule.nboulders.values.reshape((nw, nh)).T
    array_area = gdf_graticule.area_cov.values.reshape((nw, nh)).T
    bbox_unary = gdf_graticule.geometry.unary_union.bounds
    out_transform = Affine(block_width * res, 0.0, bbox_unary[0], 0.0, -block_width * res, bbox_unary[3])
    out_meta.update({'width': nw, 'height': nh, 'transform': out_transform})
    raster.save(out_raster_count, np.expand_dims(array_count, axis=2), out_meta, is_image=True)
    raster.save(out_raster_area, np.expand_dims(array_area, axis=2), out_meta, is_image=True)

def median(in_shp, in_graticule, column, in_meta, out_raster , dtype='uint8'):


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
    area_polygon can be either a single polygon or multiple squares.
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
    '''

    :return:

    :note:
    I feel that you have two different ways to approach this problem:
    - either taking the median within a defined square area (if you have a too
    small area
    '''
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