import geopandas as gpd
import numpy as np
import pandas as pd

from pathlib import Path
from rasterio import features
from shapely.geometry import box
from tqdm import tqdm
import rastertools_BOULDERING.raster as raster

def bbox_to_shp(bbox, crs_wkt, out_shp):
    """
    Convert a bounding box to a shapefile.
    
    Parameters
    ----------
    bbox : tuple
        Bounding box coordinates (minx, miny, maxx, maxy)
    crs_wkt : str
        Coordinate reference system in WKT format
    out_shp : str or Path
        Path to output shapefile
        
    Returns
    -------
    geopandas.GeoSeries
        GeoSeries containing the bounding box polygon
    """
    gs = gpd.GeoSeries(box(*bbox), crs=crs_wkt)
    gs.to_file(out_shp)
    return (gs)

def buffer(in_shp, buffer_dist, out_shp):
    """
    Create a buffer around geometries in a shapefile.
    
    Parameters
    ----------
    in_shp : str or Path
        Path to input shapefile
    buffer_dist : float
        Buffer distance in map units
    out_shp : str or Path
        Path to output shapefile
        
    Returns
    -------
    geopandas.GeoDataFrame
        DataFrame containing buffered geometries
    """
    gdf = gpd.read_file(in_shp)
    gdf_buffer = gdf.copy()
    gdf_buffer.geometry = gdf.geometry.buffer(buffer_dist)
    gdf_buffer.to_file(out_shp)
    return (gdf_buffer)
def centroid(in_shp, out_shp):
    """
    Calculate centroids of geometries in a shapefile.
    
    Parameters
    ----------
    in_shp : str or Path
        Path to input shapefile
    out_shp : str or Path
        Path to output shapefile
        
    Returns
    -------
    geopandas.GeoDataFrame
        DataFrame containing centroid points
    """
    gdf = gpd.read_file(in_shp)
    gdf_centroids = gdf.copy()
    gdf_centroids.geometry = gdf.geometry.centroid
    gdf_centroids.to_file(out_shp)
    return (gdf_centroids)

def shift(in_shp, shift_x, shift_y, out_shp):
    """
    Shift geometries in a shapefile by given x and y offsets.
    
    Parameters
    ----------
    in_shp : str or Path
        Path to input shapefile
    shift_x : float
        Shift distance in x direction
    shift_y : float
        Shift distance in y direction
    out_shp : str or Path
        Path to output shapefile
        
    Returns
    -------
    geopandas.GeoDataFrame
        DataFrame containing shifted geometries
    """
    in_polygon = Path(in_shp)
    gdf = gpd.read_file(in_polygon)
    geom_shifted = gdf.geometry.translate(xoff=shift_x, yoff=shift_y)
    gdf_shifted = gdf.copy()
    gdf_shifted["geometry"] = geom_shifted
    gdf_shifted.to_file(out_shp)
    return gdf_shifted

def rasterize(shapes, out_meta, out_raster, initial_values=0):
    """
    Convert vector shapes to a raster.
    
    Parameters
    ----------
    shapes : iterable
        Iterable of (geometry, value) pairs
    out_meta : dict
        Output raster metadata including 'height', 'width', and 'transform'
    out_raster : str or Path
        Path to output raster file
    initial_values : int, optional
        Initial value for raster cells, by default 0
        
    Notes
    -----
    Example usage:
    >>> gdf_poly = gpd.read_file(polygon_shp)
    >>> gdf_poly.constant = 1
    >>> shapes = ((geom,value) for geom, value in zip(gdf_poly.geometry, 
                  gdf_poly.constant.values.astype('uint8')))
    >>> rasterize(shapes, out_meta, out_raster, initial_values=0)
    """
    burned = features.rasterize(shapes=shapes, fill=initial_values, out_shape=(out_meta["height"], out_meta["width"]),
                                transform=out_meta["transform"])

    burned_dim = np.expand_dims(burned, axis=2)
    raster.save(out_raster, burned_dim, out_meta, is_image=True)

def clip(in_shp, mask_shp, min_area_threshold, out_shp):
    """
    Clip features in a shapefile using a mask shapefile.
    
    Parameters
    ----------
    in_shp : str or Path
        Path to input shapefile
    mask_shp : str or Path
        Path to mask shapefile
    min_area_threshold : float
        Minimum area threshold for keeping clipped features
    out_shp : str or Path
        Path to output shapefile
        
    Returns
    -------
    geopandas.GeoDataFrame
        DataFrame containing clipped geometries
        
    Notes
    -----
    Performs intersection with clipping of edges. Features smaller than 
    min_area_threshold are removed. MultiPolygons are exploded into 
    separate polygons.
    """

    in_shp = Path(in_shp)
    mask_shp = Path(mask_shp)
    out_shp = Path(out_shp)

    # read
    gdf = gpd.read_file(in_shp)
    gdf_mask = gpd.read_file(mask_shp)

    # intersection of overlay actually clip to mask
    gdf_clip = gpd.overlay(gdf, gdf_mask, how='intersection', keep_geom_type=True, make_valid=True)
    gdf_clip["area"] = gdf_clip.geometry.area
    gdf_clip = gdf_clip[gdf_clip["area"] >= min_area_threshold]

    # explode Multipolygons (if existing) and keep only polygons above min_area_threshold
    gdf_clip = remove_multipolygon(gdf_clip, min_area_threshold)
    gdf_clip["id"] = np.arange(gdf_clip.shape[0]).astype('int')

    # save
    gdf_clip.to_file(out_shp)
    print("shapefile " + out_shp.as_posix() + " has been generated")
    return (gdf_clip)


def intersect(in_shp, mask_shp, min_area_threshold, out_shp):
    """
    Find intersection between features in two shapefiles without clipping edges.
    
    Parameters
    ----------
    in_shp : str or Path
        Path to input shapefile
    mask_shp : str or Path
        Path to mask shapefile
    min_area_threshold : float
        Minimum area threshold for keeping intersected features
    out_shp : str or Path
        Path to output shapefile
        
    Returns
    -------
    geopandas.GeoDataFrame
        DataFrame containing intersected geometries
        
    Notes
    -----
    Similar to clip() but preserves original geometries that intersect with
    the mask, rather than clipping them to the mask boundary.
    """

    in_shp = Path(in_shp)
    mask_shp = Path(mask_shp)
    out_shp = Path(out_shp)

    # reading
    gdf = gpd.read_file(in_shp)
    gdf["id"] = np.arange(gdf.shape[0]).astype('int')
    gdf_mask = gpd.read_file(mask_shp)

    # in order to get the intersection without clipping, we use the index and the original dataframe
    gdf_clip = gpd.overlay(gdf, gdf_mask, how='intersection', keep_geom_type=True, make_valid=True)
    try:
        gdf_intersect = gdf[gdf["id"].isin(gdf_clip["id"])]
    except:
        gdf_intersect = gdf[gdf["id_1"].isin(gdf_clip["id"])] # in case there is an id column in both
    gdf_intersect["area"] = gdf_intersect.geometry.area
    gdf_intersect = gdf_intersect[gdf_intersect["area"] >= min_area_threshold]
    gdf_intersect["id"] = np.arange(gdf_intersect.shape[0]).astype('int')
    gdf_intersect.to_file(out_shp)
    print("shapefile " + out_shp.as_posix() + " has been generated")
    return (gdf_intersect)

def remove_multipolygon(gdf, min_area_threshold):
    """
    Explode MultiPolygons into separate Polygons and filter by area.
    
    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input GeoDataFrame containing MultiPolygons
    min_area_threshold : float
        Minimum area threshold for keeping polygons
        
    Returns
    -------
    geopandas.GeoDataFrame
        DataFrame with MultiPolygons exploded into separate Polygons,
        filtered by minimum area
        
    Notes
    -----
    Prints number of MultiPolygons removed and number of new Polygons added.
    """
    gdf_multipolygon = gdf[gdf.geometry.geom_type == "MultiPolygon"]

    if gdf_multipolygon.shape[0] > 0:
        id_multipolygon = gdf_multipolygon.index.values
        gdf_explode = gdf_multipolygon.explode(index_parts=True, ignore_index=True)
        gdf_explode["area"] = gdf_explode.geometry.area
        gdf_explode = gdf_explode[gdf_explode["area"] > min_area_threshold]

        # drop multipolygons in original dataset
        gdf = gdf.drop(id_multipolygon)
        print(str(len(id_multipolygon)) + " MultiPolygon(s) removed")

        # only concatenate if at least one boulder is larger than min_area_threshold
        if gdf_explode.shape[0] > 0:
            # add exploded multipolygons above min_area_threshold
            gdf = gpd.GeoDataFrame(pd.concat([gdf, gdf_explode], ignore_index=True))
            print(str(gdf_explode.shape[0]) + " Polygon(s) added")
    else:
        None
    return (gdf)

#TODO minimal shift between two sets of polygons (which should overlap)
#TODO use distance_to_point example but in between each
#TODO also distance_to_point does not need to have a point as second input
#TODO could be a more generic distance_between
# or calculate shape of geometry and find most similar automatically?

def distance_between(in_shp1, in_shp2, reindex=False):
    """
    Calculate distances between all features in two shapefiles.
    
    Parameters
    ----------
    in_shp1 : str or Path
        Path to first input shapefile
    in_shp2 : str or Path
        Path to second input shapefile
    reindex : bool, optional
        Whether to reindex features with new IDs, by default False
        
    Returns
    -------
    numpy.ndarray
        2D array of distances with shape (n_features2, n_features1)
    """

    in_shp1= Path(in_shp1)
    in_shp2 = Path(in_shp2)

    gdf1 = gpd.read_file(in_shp1)
    gdf2 = gpd.read_file(in_shp2)

    if reindex:
        gdf1["id"] = np.arange(gdf1.shape[0]).astype('int')
        gdf2["id"] = np.arange(gdf2.shape[0]).astype('int')

    distance_matrix = np.ones((gdf2.shape[0], gdf1.shape[0]))
    for i, row in tqdm(gdf2.iterrows(), total=gdf2.shape[0]):
        distance_row = gdf1.distance(row.geometry)
        distance_matrix[i, :] = distance_row
    return (distance_matrix)

#TODO polygon to polygon co-registration
# see https://www.tandfonline.com/doi/full/10.1080/2150704X.2017.1317928
# see https://scikit-image.org/docs/stable/auto_examples/registration/plot_register_translation.html
# or conversion to raster for simplification? for use of

def shift_between(in_shp1, in_shp2):
    """
    Calculate shifts between corresponding features in two shapefiles.
    
    Parameters
    ----------
    in_shp1 : str or Path
        Path to first input shapefile
    in_shp2 : str or Path
        Path to second input shapefile
        
    Returns
    -------
    tuple of geopandas.GeoDataFrame
        Two GeoDataFrames with added columns:
        - cdistance: distance between corresponding features
        - shift_x: x-direction shift
        - shift_y: y-direction shift
        
    Notes
    -----
    Features must have matching IDs in both shapefiles. Calculates shifts
    between centroids of corresponding features.
    """

    """
    probably used in the notebook to calculate shift between two datasets.
    :param polygon_shp_ref:
    :param polygon_shp_tgt:
    :param column_name:
    :return:
    """
    in_shp1 = Path(in_shp1)
    in_shp2 = Path(in_shp2)

    gdf1 = gpd.read_file(in_shp1)
    gdf2 = gpd.read_file(in_shp2)

    gdf1_centroid = centroid(in_shp1, out_shapefile=False)
    gdf2_centroid = centroid(in_shp2, out_shapefile=False)

    # take nearest centroids with same id
    gdf1 = gdf1.sort_values(by=["id"])
    gdf2 = gdf2.sort_values(by=["id"])
    gdf1_centroid = gdf1_centroid.sort_values(by=["id"])
    gdf2_centroid = gdf2_centroid.sort_values(by=["id"])

    gdf1 = gdf1.set_index("id")
    gdf2 = gdf2.set_index("id")
    gdf1_centroid = gdf1_centroid.set_index("id")
    gdf2_centroid = gdf2_centroid.set_index("id")

    # gdf_iterrows
    distances = []
    shifts_x = []
    shifts_y = []
    for index, row in tqdm(gdf1_centroid.iterrows(), total=gdf1_centroid.shape[0]):
        x1 = row.geometry.x
        y1 = row.geometry.y
        x2 = gdf2_centroid.loc[index].geometry.x
        y2 = gdf2_centroid.loc[index].geometry.y

        distance = np.sqrt(((x2 - x1)**2.0) + ((y2 - y1)**2.0))
        distances.append(distance)
        shift_x = x2 - x1
        shifts_x.append(shift_x)
        shift_y = y2 - y1
        shifts_y.append(shift_y)

    gdf1["cdistance"] = distances
    gdf1["shift_x"] = shifts_x
    gdf1["shift_y"] = shifts_y
    gdf2["cdistance"] = distances
    gdf2["shift_x"] = shifts_x
    gdf2["shift_y"] = shifts_y

    return (gdf1, gdf2)

