from pathlib import Path

def rm_tree(pth):
    pth = Path(pth)
    for child in pth.glob('*'):
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    pth.rmdir()


def folder_structure(df, dataset_directory):
    dataset_directory = Path(dataset_directory)
    folders = list(df["dataset"].unique())
    sub_folders = ["images", "labels"]

    for f in folders:
        for s in sub_folders:
            new_folder = dataset_directory / f / s
            Path(new_folder).mkdir(parents=True, exist_ok=True)

"""
Leftovers of messy coding:

def azimuth_numpy(x0, y0, x1, y1):

    angles = np.arctan2(x1 - x0, y1 - y0)
    rotation_angle360 = angles.copy()
    rotation_angle180 = angles.copy()
    rotation_angle_default = np.degrees(angles)

    idx1 = np.where(angles > 0)
    idx2 = np.where(angles <= 0)

    rotation_angle360[idx1] = np.degrees(angles[idx1])
    rotation_angle360[idx2] = np.degrees(angles[idx2]) + 360.0

    rotation_angle180[idx1] = np.degrees(angles[idx1])
    rotation_angle180[idx2] = np.degrees(angles[idx2]) + 180.0

    return (rotation_angle_default, rotation_angle360, rotation_angle180)

def basis_parameter(gdf_minimum_rotated_rectangle, crater_centre_point, crater_diameter, out_shapefile):

    gdf_copy = gdf_minimum_rotated_rectangle.copy()
    gdf_cc = gpd.read_file(crater_centre_point)
    geometry_cc = gdf_cc.geometry.values

    gdf_copy["bbox"] = gdf_copy.apply(lambda row: list(row.geometry.exterior.coords), axis=1)
    bbox_flatten = np.concatenate(gdf_copy["bbox"].values).ravel()

    x0 = bbox_flatten[0::10]
    y0 = bbox_flatten[1::10]
    x1 = bbox_flatten[2::10]
    y1 = bbox_flatten[3::10]
    x3 = bbox_flatten[6::10]
    y3 = bbox_flatten[7::10]

    width = distance(x0, y0, x3, y3)
    length = distance(x0, y0, x1, y1)
    wl = np.column_stack((width, length))
    long_axis = np.max(wl, axis=1)
    short_axis = np.min(wl, axis=1)
    diameter = (short_axis + long_axis) / 2.0
    aspect_ra = long_axis / short_axis

    rotation_angle_default_1, rotation_angle360_1, rotation_angle180_1 = azimuth_numpy(x0, y0, x1, y1)
    rotation_angle_default_2, rotation_angle360_2, rotation_angle180_2 = azimuth_numpy(x0, y0, x3, y3)

    idx_w1 = np.where(width <= length)
    idx_w2 = np.where(width > length)

    angle = rotation_angle_default_1.copy()
    angle360 = rotation_angle360_1.copy()
    angle180 = rotation_angle180_1.copy()

    angle360[idx_w1] = rotation_angle360_1[idx_w1]
    angle360[idx_w2] = rotation_angle360_2[idx_w2]

    angle180[idx_w1] = rotation_angle180_1[idx_w1]
    angle180[idx_w2] = rotation_angle180_2[idx_w2]

    angle[idx_w1] = rotation_angle_default_1[idx_w1]
    angle[idx_w2] = rotation_angle_default_2[idx_w2]

    values = [length, width, long_axis, short_axis, aspect_ra,
              angle, angle360, angle180, diameter]

    columns = ["length", "width", "long_axis", "short_axis", "aspect_ra",
               "angle", "angle360", "angle180", "diameter"]

    for i, c in enumerate(columns):
        gdf_copy[c] = values[i]

    gdf_copy["diametereq"] = 2.0 * np.sqrt(gdf_copy["area"] / np.pi)
    gdf_copy["dist_cc"] = distance(gdf_copy["x"], gdf_copy["y"], geometry_cc.x, geometry_cc.y)
    gdf_copy["ndist_cc"] = gdf_copy["dist_cc"] / (crater_diameter / 2.0)
    gdf_copy = gdf_copy.drop(columns=["bbox"])
    gdf_copy.to_file(out_shapefile)

    return(gdf_copy)



def morphometry(boulder_polygon, crater_centre_point, crater_diameter, area_threshold, out_shapefile_mmr, out_shapefile):

    gdf = gpd.read_file(boulder_polygon)

    ## add boulder id, area and ++
    gdf["area"] = gdf.geometry.area
    gdf = gdf[gdf["area"] >= area_threshold]
    n = gdf.shape[0]
    gdf['id'] = np.arange(n)

    gdf["x"] = gdf.geometry.centroid.x.values
    gdf["y"] = gdf.geometry.centroid.y.values

    # minimum_rotated_rectangle (mrr)
    gdf_mmr = gdf.copy()
    gdf_mmr["geometry"] = gdf.apply(lambda row: row.geometry.minimum_rotated_rectangle, axis=1)
    gdf_mmr = basis_parameter(gdf_mmr, crater_centre_point, crater_diameter, out_shapefile_mmr)

    # copy calculated morphometry of boulders back to original boulder polygons
    # equivalent to only changing the geometry
    gdf_mmr["geometry"] = gdf.geometry.values
    gdf_mmr.to_file(out_shapefile)

    return(gdf_mmr)
"""
