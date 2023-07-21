import geopandas as gpd
import numpy as np
import shapely.affinity

from shapely import geometry, segmentize
from skimage.measure import CircleModel
from skimage.measure import EllipseModel
from tqdm import tqdm

# TODO > multiple buffer (multi-ring, but as disk)
def minimum_rotated_rectangle(in_shp, out_shp):
    gdf = gpd.read_file(in_shp)
    gdf_mrr = gdf.copy()
    n = gdf_mrr.shape[0]
    rotated_rectangle = []

    for index, row in tqdm(gdf_mrr.iterrows(), total=n):
        rotated_rectangle.append(row.geometry.minimum_rotated_rectangle)

    gdf_mrr["geometry"] = rotated_rectangle
    gdf_mrr.to_file(out_shp)
    return (gdf_mrr)

def camembert(in_shp_point, radius, angle_per_camembert, out_shp):
    """
    Create a pie chart/camembert of radius radius at coordinates in point shapefile.
    Only work if in_shp is a point shapefile, and has only one entry.
    360 need to be divisible by angle_per_camembert (e.g., 20, 40, ...)
    """
    gdf = gpd.read_file(in_shp_point)
    xc = gdf.geometry.x.values[0]
    yc = gdf.geometry.y.values[0]

    # fitting a circle
    model = CircleModel()
    model.params = (xc, yc, radius)

    n = int(360 / angle_per_camembert)
    t = np.linspace(0, 2 * np.pi, n)
    angle = np.arange(0, 360, angle_per_camembert)
    xy = model.predict_xy(t)
    x = xy[:,0]
    y = xy[:,1]

    coords = []

    for i in range(len(t)):
        if i == len(t) - 1:
            pol = [(xc, yc), (x[i], y[i]), (x[0], y[0]), (xc, yc)]
            coords.append(geometry.Polygon(pol))
        else:
            pol = [(xc, yc), (x[i], y[i]), (x[i + 1], y[i + 1]), (xc, yc)]
            coords.append(geometry.Polygon(pol))

    gdf_cam = gpd.GeoDataFrame(angle, columns=['angle'], geometry=coords)
    gdf_cam = gdf_cam.set_crs(gdf.to_wkt())
    gdf_cam.to_file(out_shp)

def ellipse(in_shp_polygon, res, out_shp):
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
    circ = shapely.geometry.Point(row.geometry.centroid.x, row.geometry.centroid.y).buffer(1)
    ell = shapely.affinity.scale(circ, row.long_axis/2.0, row.short_axis/2.0)
    elrv = shapely.affinity.rotate(ell, 90 - row.angle)
    return (elrv)

def createCircle(row):
    circ = shapely.geometry.Point(row.geometry.centroid.x,row.geometry.centroid.y).buffer(1)
    circ = shapely.affinity.scale(circ, row.diametersq/2.0, row.diametersq/2.0)
    return (circ)

def fitEllipse(row):
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

