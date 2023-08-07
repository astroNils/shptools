import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import torchvision
from rastertools_BOULDERING import raster, metadata as raster_metadata

#TODO: improve functions (e.g., when there are no boulders within size range).
def batch_calculate_iou(ground_truth, predictions, in_raster, area,
                        iou_threshold):
    """
    Currently used.

    Area needs to be tuple (min_area, max_area)
    In number of pixels/area

    small : (6*6, 12*12)
    medium : (12*12, 24*24)
    large : (24*24, 48*48)
    very large : (48*48, 512*512)
    all : (0, 1024*1024) # don't expect boulders larger than 1024x1024 pixels!
    """

    # raster resolution
    res = raster_metadata.get_resolution(in_raster)[0]

    # loading of the data
    gdf_gt = gpd.read_file(ground_truth)
    gdf_pred = gpd.read_file(predictions)

    # batch
    if isinstance(area, tuple):
        areas = [area]
    else:
        areas = area

    if isinstance(iou_threshold, int) or isinstance(iou_threshold, float):
        iou_thresholds = [iou_threshold]
    else:
        iou_thresholds = iou_threshold

    precisions = []
    recalls = []
    f1_scores = []
    n_boulders = []

    for area in areas:

        min_area = area[0] * (res * res)
        max_area = area[1] * (res * res)

        # filtering based on area
        gdf_gt_filtered = gdf_gt[
            np.logical_and(gdf_gt.geometry.area >= min_area,
                           gdf_gt.geometry.area < max_area)]
        gdf_pred_filtered = gdf_pred[
            np.logical_and(gdf_pred.geometry.area >= min_area,
                           gdf_pred.geometry.area < max_area)]

        if (gdf_gt_filtered.shape[0] == 0) and (
                gdf_pred_filtered.shape[0] == 0):
            precisions.append(np.nan)
            recalls.append(np.nan)
            f1_scores.append(np.nan)

        elif (gdf_gt_filtered.shape[0] != 0) and (
                gdf_pred_filtered.shape[0] == 0):
            precisions.append(0.00)
            recalls.append(0.00)
            f1_scores.append(0.00)
        else:
            # generation of bounding boxes
            bboxes_gt = gdf_gt_filtered.geometry.bounds.to_numpy()
            bboxes_pred = gdf_pred_filtered.geometry.bounds.to_numpy()
            bboxes_gt_torch = torch.from_numpy(bboxes_gt)
            bboxes_pred_torch = torch.from_numpy(bboxes_pred)

            # pairwise iou
            pairwise_iou = torchvision.ops.box_iou(bboxes_gt_torch,
                                                   bboxes_pred_torch)

            # selection of best match
            best_matches_ious, best_matches_idxs = pairwise_iou.max(axis=1)

            # filtering of ious lower than the threshold
            gdf_gt_filtered["iou"] = best_matches_ious.numpy()

            for iou_threshold in iou_thresholds:
                gdf_good_matches = gdf_gt_filtered[
                    gdf_gt_filtered.iou >= iou_threshold]

                # calculating precision and recall
                precision = gdf_good_matches.shape[0] / gdf_pred_filtered.shape[0]
                recall = gdf_good_matches.shape[0] / gdf_gt_filtered.shape[0]
                deno = precision + recall
                nume = (2.0 * precision * recall)
                f1_score = np.divide(nume, deno, out=np.zeros_like(nume), where=deno!=0)

                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1_score)
                n_boulders.append(gdf_good_matches.shape[0])

    return (areas, iou_thresholds, precisions, recalls, f1_scores, n_boulders,
            gdf_gt_filtered, gdf_pred_filtered, best_matches_ious, best_matches_idxs)


def format_iou_to_pd(areas, iou_thresholds, precisions, recalls, f1_scores,
                     n_boulders):
    recalls_reshaped = np.array(recalls).reshape(len(areas),
                                                 len(iou_thresholds))
    precisions_reshaped = np.array(precisions).reshape(len(areas),
                                                       len(iou_thresholds))
    f1_scores_reshaped = np.array(f1_scores).reshape(len(areas),
                                                     len(iou_thresholds))
    n_boulders_reshaped = np.array(n_boulders).reshape(len(areas),
                                                       len(iou_thresholds))

    dfs = []

    for i in range(len(areas)):
        df = pd.DataFrame([])
        df["recall"] = recalls_reshaped[i]
        df["precision"] = precisions_reshaped[i]
        df["f1_score"] = f1_scores_reshaped[i]
        df["n_boulders"] = n_boulders_reshaped[i]
        df = df.set_index((np.array(iou_thresholds)*100).astype('int'))
        dfs.append(df)

    return (dfs)

'''
## OUTDATED: To be deleted...

def process_iou(ground_truth, predictions):
    gdf_gt = gpd.read_file(ground_truth)
    gdf_pred = gpd.read_file(predictions)
    gdf_gt = gdf_gt.set_index(np.arange(gdf_gt.shape[0]).astype('int'))
    gdf_pred = gdf_pred.set_index(np.arange(gdf_pred.shape[0]).astype('int'))
    bboxes_gt = gdf_gt.geometry.bounds.to_numpy()
    bboxes_pred = gdf_pred.geometry.bounds.to_numpy()
    bboxes_gt_torch = torch.from_numpy(bboxes_gt)
    bboxes_pred_torch = torch.from_numpy(bboxes_pred)
    pairwise_iou = torchvision.ops.box_iou(bboxes_gt_torch, bboxes_pred_torch)
    best_matches_ious, best_matches_idxs = pairwise_iou.max(axis=1)
    gdf_gt["iou"] = best_matches_ious.numpy()
    return gdf_gt, gdf_pred, best_matches_ious, best_matches_idxs

def calculate_iou(boulder_polygon_mapped, boulder_polygon_inferred, res):

    """
    make it a bit more condensed.... I think this is the old function, and it does not work!
    """

    boulder_polygon_mapped = Path(boulder_polygon_mapped)
    boulder_polygon_inferred = Path(boulder_polygon_inferred)

    gdf_manual = gpd.read_file(boulder_polygon_mapped)
    gdf_inferred = gpd.read_file(boulder_polygon_inferred)

    gdf_manual["area"] = gdf_manual.geometry.area
    gdf_inferred["area"] = gdf_inferred.geometry.area

    area_ranges = [[(res * 6.0) ** 2.0, (res * 501.0) ** 2.0],
                   [(res * 6.0) ** 2.0, (res * 16.0) ** 2.0],
                   [(res * 16.0) ** 2.0, (res * 32.0) ** 2.0],
                   [(res * 32.0) ** 2.0, (res * 501.0) ** 2.0]]

    gdf_manual_all = gdf_manual[np.logical_and(gdf_manual["area"] >= area_ranges[0][0], gdf_manual["area"] < area_ranges[0][1])]
    gdf_manual_small = gdf_manual[np.logical_and(gdf_manual["area"] >= area_ranges[1][0], gdf_manual["area"] < area_ranges[1][1])]
    gdf_manual_mediu = gdf_manual[np.logical_and(gdf_manual["area"] >= area_ranges[2][0], gdf_manual["area"] < area_ranges[2][1])]
    gdf_manual_large = gdf_manual[np.logical_and(gdf_manual["area"] >= area_ranges[3][0], gdf_manual["area"] < area_ranges[3][1])]

    gdf_inferred_all = gdf_inferred[np.logical_and(gdf_inferred["area"] >= area_ranges[0][0], gdf_inferred["area"] < area_ranges[0][1])]
    gdf_inferred_small = gdf_inferred[np.logical_and(gdf_inferred["area"] >= area_ranges[1][0], gdf_inferred["area"] < area_ranges[1][1])]
    gdf_inferred_mediu = gdf_inferred[np.logical_and(gdf_inferred["area"] >= area_ranges[2][0], gdf_inferred["area"] < area_ranges[2][1])]
    gdf_inferred_large = gdf_inferred[np.logical_and(gdf_inferred["area"] >= area_ranges[3][0], gdf_inferred["area"] < area_ranges[3][1])]

    # this does not work if one of the two GeoDataframes are empty
    try:
        gdf_intersect_all = gpd.overlay(gdf_manual_all, gdf_inferred_all, how="intersection")
    except:
        gdf_intersect_all = gpd.GeoDataFrame(geometry=[], columns=gdf_intersect_all.columns.values)
    try:
        gdf_intersect_small = gpd.overlay(gdf_manual_small, gdf_inferred_small, how="intersection")
    except:
        gdf_intersect_small = gpd.GeoDataFrame(geometry=[], columns=gdf_intersect_all.columns.values)
    try:
        gdf_intersect_mediu = gpd.overlay(gdf_manual_mediu, gdf_inferred_mediu, how="intersection")
    except:
        gdf_intersect_mediu = gpd.GeoDataFrame(geometry=[], columns=gdf_intersect_all.columns.values)
    try:
        gdf_intersect_large = gpd.overlay(gdf_manual_large, gdf_inferred_large, how="intersection")
    except:
        gdf_intersect_large = gpd.GeoDataFrame(geometry=[], columns=gdf_intersect_all.columns.values)

    IoUs = []
    gdfs_intersect = [gdf_intersect_all, gdf_intersect_small, gdf_intersect_mediu, gdf_intersect_large]
    gdfs_manual = [gdf_manual_all, gdf_manual_small, gdf_manual_mediu, gdf_manual_large]
    gdfs_inferred = [gdf_inferred_all, gdf_inferred_small, gdf_inferred_mediu, gdf_inferred_large]
    for i, gdf in enumerate(gdfs_intersect):
        IoU = []
        gdf_manual_tmp = gdfs_manual[i]
        gdf_inferred_tmp = gdfs_inferred[i]
        for idx, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):
            geom_list = []
            geom_list.append(gdf_manual_tmp[gdf_manual_tmp.boulder_id == row.boulder_id_1].geometry.values[0])
            geom_list.append(gdf_inferred_tmp[gdf_inferred_tmp.boulder_id == row.boulder_id_2].geometry.values[0])
            area_union = gpd.GeoDataFrame(geometry=gpd.GeoSeries(geom_list), crs=gdf_inferred.crs).unary_union.area
            IoU.append(row.geometry.area / area_union)
        IoUs.append(IoU)

    gdf_intersect_all["iou"] = IoUs[0]
    gdf_sorted_all = gdf_intersect_all.sort_values(by=['boulder_id_1', 'iou'], ascending=True)
    gdf_drop_all = gdf_sorted_all.drop_duplicates(subset=['boulder_id_1'], keep='last')

    gdf_intersect_small["iou"] = IoUs[1]
    gdf_sorted_small = gdf_intersect_small.sort_values(by=['boulder_id_1', 'iou'], ascending=True)
    gdf_drop_small = gdf_sorted_small.drop_duplicates(subset=['boulder_id_1'], keep='last')

    gdf_intersect_mediu["iou"] = IoUs[2]
    gdf_sorted_mediu = gdf_intersect_mediu.sort_values(by=['boulder_id_1', 'iou'], ascending=True)
    gdf_drop_mediu = gdf_sorted_mediu.drop_duplicates(subset=['boulder_id_1'], keep='last')

    gdf_intersect_large["iou"] = IoUs[3]
    gdf_sorted_large = gdf_intersect_large.sort_values(by=['boulder_id_1', 'iou'], ascending=True)
    gdf_drop_large = gdf_sorted_large.drop_duplicates(subset=['boulder_id_1'], keep='last')

    # is true positive based on
    for i in np.arange(0.50, 1.00, 0.05):
        column_name = "tp_" + str(int(i * 100)).zfill(2)
        a = gdf_drop_all.iou > i
        gdf_drop_all[column_name] = a + 0

    for i in np.arange(0.50, 1.00, 0.05):
        column_name = "tp_" + str(int(i * 100)).zfill(2)
        a = gdf_drop_small.iou > i
        gdf_drop_small[column_name] = a + 0

    for i in np.arange(0.50, 1.00, 0.05):
        column_name = "tp_" + str(int(i * 100)).zfill(2)
        a = gdf_drop_mediu.iou > i
        gdf_drop_mediu[column_name] = a + 0

    for i in np.arange(0.50, 1.00, 0.05):
        column_name = "tp_" + str(int(i * 100)).zfill(2)
        a = gdf_drop_large.iou > i
        gdf_drop_large[column_name] = a + 0

    ious = pd.DataFrame(np.zeros((11, 8)),
                        index=["50%", "55%", "60%", "65%", "70%", "75%", "80%", "85%",
                               "90%", "95%", "mean"], columns=["precision_all", "recall_all",
                                                               "precision_small", "recall_small",
                                                               "precision_medium", "recall_medium",
                                                               "precision_large", "recall_large"])

    precision_all = []
    recall_all = []
    precision_small = []
    recall_small = []
    precision_mediu = []
    recall_mediu = []
    precision_large = []
    recall_large = []

    for i in np.arange(0.50, 1.00, 0.05):
        column_name = "tp_" + str(int(i * 100)).zfill(2)
        tp = gdf_drop_all[column_name].sum()
        fp = len(gdf_drop_all[gdf_drop_all[column_name] == 0])
        precision_all.append(tp / (tp + fp))
        recall_all.append(tp / gdf_manual_all.shape[0]) # there is an error, it should be the recall
    precision_all.append(np.mean(np.array(precision_all)))
    recall_all.append(np.mean(np.array(recall_all)))
    ious["precision_all"] = precision_all
    ious["recall_all"] = recall_all

    for i in np.arange(0.50, 1.00, 0.05):
        column_name = "tp_" + str(int(i * 100)).zfill(2)
        tp = gdf_drop_small[column_name].sum()
        fp = len(gdf_drop_small[gdf_drop_small[column_name] == 0])
        precision_small.append(tp / (tp + fp))
        recall_small.append(tp / gdf_manual_small.shape[0])
    precision_small.append(np.mean(np.array(precision_small)))
    recall_small.append(np.mean(np.array(recall_small)))
    ious["precision_small"] = precision_small
    ious["recall_small"] = recall_small

    for i in np.arange(0.50, 1.00, 0.05):
        column_name = "tp_" + str(int(i * 100)).zfill(2)
        tp = gdf_drop_mediu[column_name].sum()
        fp = len(gdf_drop_mediu[gdf_drop_mediu[column_name] == 0])
        precision_mediu.append(tp / (tp + fp))
        recall_mediu.append(tp / gdf_manual_mediu.shape[0])
    precision_mediu.append(np.mean(np.array(precision_mediu)))
    recall_mediu.append(np.mean(np.array(recall_mediu)))
    ious["precision_medium"] = precision_mediu
    ious["recall_medium"] = recall_mediu

    for i in np.arange(0.50, 1.00, 0.05):
        column_name = "tp_" + str(int(i * 100)).zfill(2)
        tp = gdf_drop_large[column_name].sum()
        fp = len(gdf_drop_large[gdf_drop_large[column_name] == 0])
        precision_large.append(tp / (tp + fp))
        recall_large.append(tp / gdf_manual_large.shape[0])
    precision_large.append(np.mean(np.array(precision_large)))
    recall_large.append(np.mean(np.array(recall_large)))
    ious["precision_large"] = precision_large
    ious["recall_large"] = recall_large

    # saving shapefiles for graphical inspection (with iou)
    over_50iou = gdf_drop_all[gdf_drop_all.tp_50.astype('bool')]

    gdf_inferred_tp = gdf_inferred[gdf_inferred.boulder_id.isin(over_50iou.boulder_id_2.values)]

    # add iou
    ious_tp = []
    for idx, row in gdf_inferred_tp.iterrows():
        over_50iou_tmp = over_50iou[over_50iou.boulder_id_2 == row.boulder_id]
        ious_tp.append(over_50iou_tmp.iou.values[0])
    gdf_inferred_tp["iou"] = ious_tp

    filename_tp = boulder_polygon_inferred.with_name(boulder_polygon_inferred.name.split(".")[0] + "_tp_iou50" + boulder_polygon_inferred.suffix)
    gdf_inferred_tp.to_file(filename_tp)
    gdf_inferred_fp = gdf_inferred[~(gdf_inferred.boulder_id.isin(over_50iou.boulder_id_2.values))]
    filename_fp = boulder_polygon_inferred.with_name(boulder_polygon_inferred.name.split(".")[0] + "_fp_iou50" + boulder_polygon_inferred.suffix)
    if gdf_inferred_fp.shape[0] > 0:
        gdf_inferred_fp.to_file(filename_fp)

    return (ious)

'''