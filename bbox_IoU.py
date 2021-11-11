import numpy as np
import math
import matplotlib.pyplot as plt
from pandas import DataFrame
from scipy.spatial import ConvexHull
from matplotlib.patches import Ellipse


def rectangle_bbox(points):
    """
    generate a bounding rectangle around a set of points/a object"
    :param  points: np_darray, a set of points belong to one object
    :return  corners: four corners of the bounding box
             conv_indices: bbox coordinates (five)
    """

    # calculate the minimal and maximal x-y coordinates
    unipoints = DataFrame(points).drop_duplicates().values
    if (len(points) == 1) or (len(unipoints) == 1):  # if only one point, make the side length 0.1
        points = unipoints
        min_x = points[0,0] - 0.1
        min_y = points[0,1] - 0.1
        max_x = points[0,0] + 0.1
        max_y = points[0,1] + 0.1
    else: # more than one point
        min_x = np.min(points[:,0])
        min_y = np.min(points[:,1])
        max_x = np.max(points[:,0])
        max_y = np.max(points[:,1])

    if min_x == max_x:
        min_x -= 0.05
        max_x += 0.05
    if min_y == max_y:
        min_y -= 0.05
        max_y += 0.05

    # calculate the coordinates of four corners
    top_left = np.array([min_x, max_y])
    top_right = np.array([max_x, max_y])
    bottom_right = np.array([max_x, min_y])
    bottom_left = np.array([min_x, min_y])
    corners = np.vstack((top_left, top_right, bottom_right, bottom_left))

    # use four corners to generate the rectangle bbox
    #convex_hulls = ConvexHull(corners)
    #conv_indices = corners[convex_hulls.vertices]
    #conv_indices = np.vstack((conv_indices,conv_indices[0]))

    # plot the points and bounding box
    # plt.figure()
    # if len(points) == 1:
    #     plt.plot(points[0], points[1], 'o')
    # else:
    #     plt.plot(points[:,0], points[:,1],'o')
    # plt.plot(conv_indices[:,0],conv_indices[:,1],'r--',lw=2)
    # plt.show()

    return corners  #, conv_indices


def ellipse_bbox (points):
    """
    generate a bounding ellipse based on bounding rectangle around a set of points/a object"
    :param  points: np_darray, a set of points belong to one object
    :return  e: the bounding ellipse
    """

    # calculate the rectangle bbox first, generate the ellipse based on the rectangle bbox.
    corners_four, conv_indices = rectangle_bbox(points)

    # center point
    ox = (corners_four[0,0] + corners_four[1,0])/2
    oy = (corners_four[0,1] + corners_four[3,1])/2

    # long axis and short axis &&& angles
    genhaoer = math.sqrt(2)
    side1 = (abs(corners_four[1,0] - corners_four[0,0])) / genhaoer # x-side
    side2 = (abs(corners_four[0,1] - corners_four[3,1])) / genhaoer # y-side

    if (side1 > side2):
        long_axis = side1
        short_axis = side2
        d_angle = 0
    elif (side1 == side2):
        long_axis = side1
        short_axis = side2
        d_angle = 0
    elif (side1 < side2):
        long_axis = side2
        short_axis = side1
        d_angle = 90

    # generate the ellipse bounding box and plot
    e = Ellipse(xy=(ox, oy), width=long_axis * 2, height=short_axis * 2, angle=d_angle)
    fig = plt.figure(0)
    ax = fig.add_subplot(111, aspect='equal')
    ax.add_artist(e)

    plt.scatter(points[:, 0], points[:, 1], c='r')
    plt.plot(conv_indices[:, 0], conv_indices[:, 1], c='b',alpha=0.5)
    e.set_color('black')
    e.set_facecolor('none')
    plt.xlim(20, 40)
    plt.ylim(0, 5)

    ax.grid(True)
    plt.show()
    return e


def bbox_IoU(pointsA, pointsB):
    """
    calculate the Intersection over Union of gt-bbox and predicted-bbox
    :param  cornersA: np_darray, four corners points belong to the bbox of the gt object
            cornersB: np_darray, four corners points belong to the bbox of the corresponding predicted object
    :return  iou: the IoU value of this gt-prediction pair
    """

    cornersA = rectangle_bbox(pointsA)
    cornersB = rectangle_bbox(pointsB)

    # extract bottom-left and top-right points coordinates as boxA and boxB
    blA = cornersA[3]
    trA = cornersA[1]
    boxA = np.concatenate((blA,trA))

    blB = cornersB[3]
    trB = cornersB[1]
    boxB = np.concatenate((blB,trB))

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def bbox_diou(cornersA, cornersB):

    """
    calculate the Intersection over Union of gt-bbox and predicted-bbox
    :param  cornersA: np_darray, four corners points belong to the bbox of the gt object
            cornersB: np_darray, four corners points belong to the bbox of the corresponding predicted object
    :return  iou: the IoU value of this gt-prediction pair
    """

    # extract bottom-left and top-right points coordinates as boxA and boxB
    blA = cornersA[3]
    trA = cornersA[1]
    boxA = np.concatenate((blA,trA))

    blB = cornersB[3]
    trB = cornersB[1]
    boxB = np.concatenate((blB,trB))

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


if __name__ == '__main__':
    for i in range(10):
        a = np.random.randint(1,10,size=(7,2))
        b = np.random.randint(1,10,size=(6,2))
