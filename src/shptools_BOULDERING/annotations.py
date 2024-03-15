import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import rastertools_BOULDERING.raster as raster
import rastertools_BOULDERING.metadata as raster_metadata

import shptools_BOULDERING.misc as shptools_misc

from pathlib import Path
from pycocotools import mask
from shapely.geometry import box
from tqdm import tqdm

def tile_from_dataframe(df, dataset_directory, resolution_limit):
    print("...Generating one boulder outline shapefile per image patch...")

    dataset_directory = Path(dataset_directory)
    datasets = df.dataset.unique()

    nshapefiles = 0
    for d in datasets:
        label_directory = (dataset_directory / d / "labels")
        n = len(list(label_directory.glob("*.shp")))
        nshapefiles = nshapefiles + n

    ntiles = df.shape[0]

    if nshapefiles == ntiles:
        print("Number of tiles == Number of shapefiles in specified folder(s). No tiling of boulders required.")
    # if for some reasons they don't match, it just need to be re-tiled
    # we delete the image directory(ies) just to start from a clean folder
    else:
        for d in datasets:
            label_directory = (dataset_directory / d / "labels")
            shptools_misc.rm_tree(label_directory)

        # re-creating folder structure
        shptools_misc.folder_structure(df, dataset_directory)

        for index, row in tqdm(df.iterrows(), total=ntiles):

            # this is only useful within the loop if generating tiling on multiple images
            in_raster = row.raster_ap
            in_boulders = row.boulder_ap
            bbox = box(*row.bbox_im)
            gdf_boulders = gpd.read_file(in_boulders, bbox=bbox) # equivalent to clip
            gdf_boulders["id"] = np.arange(gdf_boulders.shape[0]).astype('int') # equivalent to clip
            gdf_clip = gpd.clip(gdf_boulders, mask=bbox, keep_geom_type=False)  # to clip at edges
            gdf_clip = gdf_clip[gdf_clip.geometry.geom_type == "Polygon"]
            gdf_clip["area"] = gdf_clip.geometry.area
            gdf_clip = gdf_clip[gdf_clip.area > (resolution_limit * row.pix_res) ** 2.0]  # at least 2 px x 2px areal is adviced
            filename_shp = (dataset_directory / row.dataset / "labels" / row.file_name.replace("_image.png", "_mask.shp"))
            if gdf_clip.shape[0] > 0:
                gdf_clip.to_file(filename_shp)
            else:
                schema = {"geometry": "Polygon", "properties": {"id": "int", "area": "float"}}
                gdf_empty = gpd.GeoDataFrame(geometry=[])
                gdf_empty.to_file(filename_shp, driver='ESRI Shapefile', schema=schema, crs=row.coord_sys)
def split_global(df, gdf, split):
    """
    Shuffle tiles and randomly distributes the selection rectangle grids /
    graticules into a train / validation / test datasets (respecting the
    split values specified in <split>).

    A "dataset" column is added to both the DataFrame and GeoDataFrame.

    Note that the split here is a global split, meaning that regardless of the
    number of rectangle grids per images, it just randomly distributes across
    the train / validation / test datasets. Another function needs to be written
    if a split per image is wanted... TODO...

    :param df_selection_tiles:
    :param gdf_selection_tiles_updated:
    :param split:
    :return:
    """
    #out_shapefile = Path(out_shapefile)

    np.random.seed(seed=27)
    n = df.shape[0]
    idx_shuffle = np.random.permutation(n)

    training_idx, remaining_idx = np.split(idx_shuffle, [int(split[0] * len(idx_shuffle))])
    split_val = split[1] / (1 - split[0])  # split compare to remaining data
    val_idx, test_idx = np.split(remaining_idx, [int(split_val * len(remaining_idx))])

    df["dataset"] = "train"
    df["dataset"].iloc[val_idx] = "validation"
    df["dataset"].iloc[test_idx] = "test"

    gdf["dataset"] = "train"
    gdf["dataset"].iloc[val_idx] = "validation"
    gdf["dataset"].iloc[test_idx] = "test"

    return (df, gdf)


def split_per_image(df, split):
    np.random.seed(seed=27)
    print("...Assigning train/validation/test datasets to tiles...")
    train_tiles = []
    validation_tiles = []
    test_tiles = []

    unique_image_id = df.image_id.unique()

    numpy_split = [np.round(split[0], decimals=2),
                   np.round(split[0] + split[1], decimals=2)]

    for i in unique_image_id:
        df_selection = df[df.image_id == i]
        train_tiles_tmp, val_tiles_tmp, test_tiles_tmp = np.split(
            df_selection.sample(frac=1, random_state=27),
            [int(numpy_split[0] * len(df_selection)),
             int(numpy_split[1] * len(df_selection))])

        train_tiles.append(train_tiles_tmp)
        validation_tiles.append(val_tiles_tmp)
        test_tiles.append(test_tiles_tmp)

    df_train_tiles = pd.concat(train_tiles)
    df_train_tiles["dataset"] = "train"
    df_val_tiles = pd.concat(validation_tiles)
    df_val_tiles["dataset"] = "validation"
    df_test_tiles = pd.concat(test_tiles)
    df_test_tiles["dataset"] = "test"

    df_selection_tiles_split = pd.concat(
        [df_train_tiles, df_val_tiles, df_test_tiles])
    df_selection_tiles_split = df_selection_tiles_split.sample(frac=1, random_state=27)

    return (df_selection_tiles_split)

def semantic_segm_mask(image, labels):
    """No filtering including here"""

    image = Path(image)
    labels = Path(labels)
    seg_mask_filename = Path(
        labels.as_posix().replace("_mask.shp", "_segmask.tif"))

    gdf = gpd.read_file(labels)
    out_meta = raster_metadata.get_profile(image)

    with rio.open(image) as src:
        arr = src.read()  # always read as channel, height, width

        try:
            seg_mask, __ = rio.mask.mask(src, gdf.geometry, all_touched=False, invert=False)
            seg_mask_byte = (seg_mask > 0).astype('uint8')
        # if no values are in there...
        except:
            seg_mask_byte = np.zeros_like(arr).astype('uint8')

    raster.save(seg_mask_filename, seg_mask_byte, out_meta, False)

def gen_semantic_segm_mask(df, dataset_directory):
    print("...Generating semantic segmentation masks...")
    ntiles = df.shape[0]
    for index, row in tqdm(df.iterrows(), total=ntiles):
        image = dataset_directory / row.dataset / "images" / row.file_name.replace("_image.png", "_image.tif")
        labels = dataset_directory / row.dataset / "labels" / row.file_name.replace("_image.png", "_mask.shp")
        semantic_segm_mask(image, labels)

def annotations_to_df(df, dataset_directory, block_width, block_height, add_one, json_out):
    print("...Generating Detectron2 custom dataset from dataframe...")

    ntiles = df.shape[0]

    df_json = pd.DataFrame([])
    df_json["file_name"] = df.file_name
    df_json["height"] = block_height
    df_json["width"] = block_width
    df_json["image_id"] = np.arange(df_json.shape[0]).astype('int')
    df_json["dataset"] = df.dataset
    annotations = []

    for index, row in tqdm(df.iterrows(), total=ntiles):

        rle_mask = []
        bbox_xyxy = []

        df_annotations = pd.DataFrame([])

        image = dataset_directory / row.dataset / "images" / row.file_name.replace(
            "_image.png", "_image.tif")
        labels = dataset_directory / row.dataset / "labels" / row.file_name.replace(
            "_image.png", "_mask.shp")

        gdf = gpd.read_file(labels)

        with rio.open(image) as src:
            arr = src.read()  # always read as channel, height, width
            masks = np.zeros((gdf.shape[0], arr.shape[1], arr.shape[2])).astype(
                'uint8')

            # https://rasterio.readthedocs.io/en/stable/api/rasterio.mask.html
            for i, row_gdf in gdf.iterrows():
                out, tt = rio.mask.mask(src, [row_gdf.geometry],
                                        all_touched=False, invert=False)
                masks[i, :, :] = (out[0] > 0).astype('uint8')
                bbox_xyxy.append(bbox_numpy(out[0] > 0, add_one))

            # Then I have to convert the masks (as rle...) bit mask...
            # I could add skimage.morphology.remove_small_holes to be sure to get rid of artefacts?
            for m in masks:
                rle_mask.append(mask.encode(np.asarray(m, order="F")))

        df_annotations["bbox"] = bbox_xyxy
        df_annotations["bbox_mode"] = 0
        df_annotations["category_id"] = 0
        df_annotations["segmentation"] = rle_mask

        annotations.append(df_annotations.to_dict(orient="records"))

    df_json["annotations"] = annotations
    df_json.to_json(json_out, orient="records", indent=2)

    return (df_json)

def bbox_numpy(img, add_one=True):
    # similar behavior as Detectron2 (with +1, make sense in QGIS)
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    if add_one:
        return [float(xmin), float(ymin), float(xmax + 1), float(ymax + 1)]
    else:
        return [float(xmin), float(ymin), float(xmax), float(ymax)]


