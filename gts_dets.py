from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import numpy as np
import json
import copy
import os
from collections import defaultdict
from scipy.spatial.qhull import QhullError, ConvexHull
from sequence import Sequence
from radar_scenes.labels import ClassificationLabel
from modified_PDQ import PDQ
from bbox_rectangle_ellipse import rectangle_bbox, ellipse_bbox
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import utils
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist
from bbox_IoU import rectangle_bbox
import os.path as osp

# record experiment logs
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
path = os.path.abspath(os.path.dirname(__file__))
type = sys.getfilesystemencoding()
sys.stdout = Logger('vbgmdpdq.txt')


def ext_objects_and_locations(lst):
    """
        count how many objects and summary their indices
        :param lst: Input list
        :return: k: list of objects, v: corresponding indices
    """

    k = []
    v = []
    dd = defaultdict(list)
    for val, key in enumerate(lst):
        dd[key].append(val)
    for key, value in dd.items():
        k.append(key)
        v.append(value)
    return k, v


def generate_gt_instances(tracks, labels, rd):
    """
           generate ground-truth instances used for PDQ evaluation
           :return: gt_seg_all: (2d-array) g * points,
                                per timeframe, g * points, where points belong to gt instances are true, otherwise false
                    gt_label_all: g
                                per timeframe, g, labels of gt instances
    """
    rd = rd.T
    X = []
    for x in range(len(tracks)):  # initialize X into all-False list
        X.append(False)
    objects, objects_locations = ext_objects_and_locations(tracks)

    # iterate all gt objects in the current timeframe
    rdobject = []
    gt_seg = []
    gt_labels = []
    tem_X = copy.deepcopy(X)
    for j in range(len(objects)):
        if objects[j] != b'':
            # find the label of this object
            loc = objects_locations[j]
            lab = int(labels[loc[0]])
            gt_labels.append(lab)
            # find all points belong to this object
            for l in loc:
                tem_X[l] = True
                rdobject.append(rd[l, :])
            gt_seg.append(tem_X)
            tem_X = copy.deepcopy(X)

    return gt_seg, gt_labels, rdobject


def generate_seg_instances(gt_uuid, seg_vvid, probs):

    ini_seg = np.zeros((1, len(gt_uuid)))
    for p in range(len(probs)):
        vvid = seg_vvid[p]
        prob = probs[p]
        idx = np.where(gt_uuid == vvid)
        ini_seg[:, idx] = prob

    return ini_seg


def evaluate_pdq(seg_vvid, probs):
    gt_seg, gt_labels, rdo = generate_gt_instances(gttracks, gtlabels, gtrcsdir)
    pred_seg = generate_seg_instances(gtvvids, seg_vvid, probs)

    det_label = [[1, 0, 0, 0, 0, 0]]
    cnt = 0
    for g in gt_seg[0]:
        if g:
            cnt += 1
    # PDQ evaluation
    evaluator = PDQ()
    a = np.array(gt_seg)
    b = np.array(gt_labels)
    c = np.array(pred_seg)
    d = np.array(det_label)

    # print(a)
    # print(c)
    print('number of gtpoints:', cnt)

    e = 0
    f = 0
    pdq = evaluator.add_img_eval(a, b, c, d, e, f, bbox_comparison=False)
    TP, FP, FN = evaluator.get_assignment_counts()
    pdq_score = evaluator.get_pdq_score()
    spatial_quality = evaluator.get_spatial_score()
    label_quality = evaluator.get_label_score()
    fg = evaluator.get_fg_quality_score()
    bg = evaluator.get_bg_quality_score()
    #
    #
    print('-'*20)
    print('TPs:',TP,'\n','FPs:',FP,'\n','FNs:',FN,'\n',
          'PDQ score for the current scene:',pdq_score,'\n',
          'final_Qs:',spatial_quality,'\n','final_Ql:',label_quality,'\n',
          'final_fg_quality:',fg,'\n','final_bg_quality:',bg)
    print('-'*20)
    return spatial_quality, pdq_score, rdo


max_Qs = 0
max_pdq = 0
max_frame = 0
cnt = 0
qs_sum = 0
pdq_sum = 0
frame_08 = []
for i in range(0, 1512):
    frame = i
    print('-' * 80)
    print('frame:', frame)
    try:
        _HEATMAP_THRESH = 0.0027
        _2D_MAH_DIST_THRESH = 3.439
        _SMALL_VAL = 1e-14

        # loading ground-truth data
        path1 = 'F:/Desktop/ini_trackids'
        path2 = 'F:/Desktop/ini_labels'
        path3 = 'F:/Desktop/ini_vvids'
        path4 = 'F:/Desktop/ini_features'

        tf = 'track_id %d' % (frame) + '.npy'
        tfilepath = os.path.join(path1, tf)
        gttracks = np.load(tfilepath)

        lf = 'label %d' % (frame) + '.npy'
        lfilepath = os.path.join(path2, lf)
        gtlabels = np.load(lfilepath)

        vf = 'vvid %d' % (frame) + '.npy'
        vfilepath = os.path.join(path3, vf)
        gtvvids = np.load(vfilepath)

        rf = 'rcsdir %d' % (frame) + '.npy'
        rfilepath = os.path.join(path4, rf)
        gtrcsdir = np.load(rfilepath)


        rdpth = 'F:/Desktop/seg_features'
        tpth = 'F:/Desktop/seg_trackids'
        vpth = 'F:/Desktop/seg_vvids'

        rdf = 'rcs_dir %d' % (frame) + '.npy'
        rdfilepath = os.path.join(rdpth, rdf)
        rcs_dir = np.load(rdfilepath)

        tf = 'track_ids %d' % (frame) + '.npy'
        tfilepath = os.path.join(tpth, tf)
        track = np.load(tfilepath)

        vf = 'vvid %d' % (frame) + '.npy'
        vfilepath = os.path.join(vpth, vf)
        vvids = np.load(vfilepath)

        # drop duplicated points and remain unique points
        mdata = DataFrame(rcs_dir).drop_duplicates().values
        track_unique = []
        vvid_unique = []
        for j in range(len(mdata)):
            for k in range(len(rcs_dir)):
                if (rcs_dir[k, 0] == mdata[j, 0]) and (rcs_dir[k, 1] == mdata[j, 1]):
                    track_unique.append(track[k])
                    vvid_unique.append(vvids[k])
                    break

        # normalization (rcs -> [0,1], dir -> [-1,1])
        array0 = mdata[:, 0].reshape(-1, 1)
        array1 = mdata[:, 1].reshape(-1, 1)

        b0 = preprocessing.MinMaxScaler().fit_transform(array0)
        b1 = preprocessing.MaxAbsScaler().fit_transform(array1)
        data = np.concatenate((b0, b1), axis=1)

        # DBSCAN + GMM
        clustering = DBSCAN(eps=0.5, min_samples=3).fit(data)
        core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
        core_samples_mask[clustering.core_sample_indices_] = True
        labels = clustering.labels_
        num_clusters = len(set(labels))

        # gmm_dbscan = GaussianMixture(n_components=num_clusters).fit(data)
        gmm_dbscan = BayesianGaussianMixture(n_components=num_clusters, weight_concentration_prior_type='dirichlet_distribution',
                                     covariance_type='full').fit(data)

        prediction = gmm_dbscan.predict(data)
        print('GMM-prediction:', prediction)

        # find which cluster is the target
        num0 = np.sum(prediction == 0)
        num1 = np.sum(prediction == 1)
        if num0 > num1:
            nummax = 0
            nump = num0
        else:
            nummax = 1
            nump = num1

        covs = gmm_dbscan.covariances_[nummax]
        realmean = gmm_dbscan.means_[nummax]
        print('number of segpoints:', nump)

        cluster0 = []
        vvid0 = []
        for j in range(len(prediction)):
            if prediction[j] == nummax:
                cluster0.append(data[j])
                vvid0.append(vvid_unique[j])

        cluster0 = np.array(cluster0)

        row, col = cluster0.shape
        bbox_nor = rectangle_bbox(cluster0)

        bbox_min = np.zeros((4, 2))
        bbox_min[0, 0] = bbox_nor[0, 0] + covs[0, 0]
        bbox_min[0, 1] = bbox_nor[0, 1] - covs[1, 1]
        bbox_min[1, 0] = bbox_nor[1, 0] - covs[0, 0]
        bbox_min[1, 1] = bbox_nor[1, 1] - covs[1, 1]
        bbox_min[2, 0] = bbox_nor[2, 0] - covs[0, 0]
        bbox_min[2, 1] = bbox_nor[2, 1] + covs[1, 1]
        bbox_min[3, 0] = bbox_nor[3, 0] + covs[0, 0]
        bbox_min[3, 1] = bbox_nor[3, 1] + covs[1, 1]

        bbox_max = np.zeros((4, 2))
        bbox_max[0, 0] = bbox_nor[0, 0] - covs[0, 0]
        bbox_max[0, 1] = bbox_nor[0, 1] + covs[1, 1]
        bbox_max[1, 0] = bbox_nor[1, 0] + covs[0, 0]
        bbox_max[1, 1] = bbox_nor[1, 1] + covs[1, 1]
        bbox_max[2, 0] = bbox_nor[2, 0] + covs[0, 0]
        bbox_max[2, 1] = bbox_nor[2, 1] - covs[1, 1]
        bbox_max[3, 0] = bbox_nor[3, 0] - covs[0, 0]
        bbox_max[3, 1] = bbox_nor[3, 1] - covs[1, 1]

        probs = np.zeros((row, 1))
        for c in range(len(cluster0)):
            if (cluster0[c, 0] >= bbox_min[0, 0]) and (cluster0[c, 0] <= bbox_min[2, 0]) and (cluster0[c, 1] >= bbox_min[2, 1]) \
                and (cluster0[c, 1] <= bbox_min[0, 1]):
                probs[c] = 1
            elif (cluster0[c, 0] >= bbox_nor[0, 0]) and (cluster0[c, 0] <= bbox_nor[2, 0]) and (cluster0[c, 1] >= bbox_nor[2, 1]) \
                and (cluster0[c, 1] <= bbox_nor[0, 1]):
                probs[c] = 0.8
            else:
                probs[c] = 0.5
        print('probs', probs)
        qs1, pdq1, gtopoints = evaluate_pdq(vvid0, probs)

        cnt += 1
        qs_sum += qs1
        pdq_sum += pdq1

        # plot
        # use four corners to generate the rectangle bbox
        ch_min = ConvexHull(bbox_min)
        conv_indices_min = bbox_min[ch_min.vertices]
        conv_indices_min = np.vstack((conv_indices_min, conv_indices_min[0]))

        ch_nor = ConvexHull(bbox_nor)
        conv_indices_nor = bbox_nor[ch_nor.vertices]
        conv_indices_nor = np.vstack((conv_indices_nor, conv_indices_nor[0]))

        ch_max = ConvexHull(bbox_max)
        conv_indices_max = bbox_max[ch_max.vertices]
        conv_indices_max = np.vstack((conv_indices_max, conv_indices_max[0]))

        # plot the points and bounding box
        plt.figure()
        if len(cluster0) == 1:
            plt.plot(cluster0[0], cluster0[1], 'o', color='black')
        else:
            plt.scatter(cluster0[:, 0], cluster0[:, 1], 2, color='indigo', marker='^',zorder=40)

        plt.plot(conv_indices_max[:, 0], conv_indices_max[:, 1], linewidth='2', label="max bbox", color='steelblue', linestyle='-',zorder=10)
        plt.fill(conv_indices_max[:, 0], conv_indices_max[:, 1], color='steelblue')
        plt.plot(conv_indices_nor[:, 0], conv_indices_nor[:, 1], linewidth='2', label="mid bbox", color='yellowgreen',
                 linestyle='-', zorder=20)
        plt.fill(conv_indices_nor[:, 0], conv_indices_nor[:, 1], color='yellowgreen')
        plt.plot(conv_indices_min[:, 0], conv_indices_min[:, 1], linewidth='2', label="min bbox", color='gold',
                 linestyle='-', zorder=30)
        plt.fill(conv_indices_min[:, 0], conv_indices_min[:, 1], color='gold')

        plt.xlabel('rcs')
        plt.ylabel('direction vector')
        #plt.show()

        path = 'F:/Desktop/figs'
        filename = 'gmm_pdq %d' % (i) + '.pdf'
        plt.savefig(os.path.join(path, filename))

        if qs1 > max_Qs:
            max_Qs = qs1
            max_frame = frame
        if pdq1 > max_pdq:
            max_pdq = pdq1

        if qs1 > 0.8:
            frame_08.append(frame)

        print('-' * 80)

    except:
        print('no')

print('frame:', max_frame, 'max_Qs:', max_Qs, 'max_pdq:', max_pdq)
print('average Qs of a sequence:', qs_sum/cnt, 'average pdq of a sequence:', pdq_sum/cnt)
print('frames that qs > 0.8:', frame_08)